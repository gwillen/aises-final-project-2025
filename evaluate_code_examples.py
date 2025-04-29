#!/usr/bin/env python3
"""
Script to evaluate AI's ability to solve coding examples.

This script:
1. Takes a JSON file with code examples
2. Picks a random example (or one matching criteria)
3. Sends it to an AI model for evaluation
4. Saves the result to a JSON file
"""

import os
import json
import argparse
import random
import sys
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Import shared utilities
from ai_utils import (
    create_openai_client,
    create_anthropic_client,
    query_openai,
    query_anthropic,
    list_openai_models,
    list_anthropic_models,
    validate_model_name,
    save_to_json
)

EVALUATION_PROMPT_TEMPLATE = """
What does this program output? Please provide only your best immediate guess at the final answer, on a line by itself; no reasoning, analysis, or commentary.

```
{code}
```
"""

CONFIDENCE_PROMPT_TEMPLATE = """
In a moment, I will ask you what this program outputs. You will have to provide only your best immediate guess at the final answer, without reasoning, analysis, or commentary. Before I do that, first I want to evaluate your confidence that you can answer correctly. Please give me an estimate of your probability that your answer will be correct. Again, please give only the estimate, on a line by itself, without reasoning, analysis, or commentary.

```
{code}
```
"""

def load_examples_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load examples from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        A list of examples
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'examples' not in data or not isinstance(data['examples'], list):
            print("Error: JSON file doesn't contain an 'examples' array")
            return []

        return data['examples']
    except Exception as e:
        print(f"Error loading examples from {file_path}: {e}")
        return []

def select_random_example(examples: List[Dict[str, Any]], language: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Select a random example from the list, optionally filtering by language.

    Args:
        examples: List of examples
        language: If provided, only select examples in this language

    Returns:
        A randomly selected example, or None if no matching examples
    """
    if not examples:
        return None

    matching_examples = []

    if language:
        # Case-insensitive match for language
        matching_examples = [ex for ex in examples if ex.get('language', '').lower() == language.lower()]
        if not matching_examples:
            print(f"No examples found for language: {language}")
            print(f"Available languages: {', '.join(set(ex.get('language', '') for ex in examples))}")
            return None
    else:
        matching_examples = examples

    return random.choice(matching_examples)

def extract_final_answer(response: str) -> str:
    """
    Extract the final answer from the response, looking for a clean line by itself.

    Args:
        response: The full response from the AI

    Returns:
        The extracted answer
    """
    # Look for the first line that's not empty and doesn't contain explanations
    explanation_indicators = ["because", "since", "as", "explanation", "reasoning", "analysis", "therefore", "thus"]

    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if this line looks like an explanation
        if any(indicator in line.lower() for indicator in explanation_indicators):
            continue

        # Return the first line that doesn't look like an explanation
        return line

    # If we couldn't find a good line, just return the first non-empty line
    for line in response.strip().split('\n'):
        if line.strip():
            return line.strip()

    return response.strip()

# TODO: check how actual models format their responses, and try to coach them to use a standardized format.
def extract_confidence(response: str) -> float:
    """
    Extract the confidence estimate from the response.

    Args:
        response: The full response from the AI

    Returns:
        The extracted confidence as a float between 0 and 1
    """
    # Look for percentage or decimal value
    response = response.strip()

    # First look for a line with just a number
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Try to extract a percentage or decimal
        # Check for percentage format (e.g., "90%" or "90 percent")
        percentage_match = re.search(r'(\d+)(?:\s*%|\s+percent)', line)
        if percentage_match:
            try:
                percentage = float(percentage_match.group(1))
                return percentage / 100.0  # Convert to decimal
            except ValueError:
                pass

        # Check for decimal format (e.g., "0.9" or ".9")
        decimal_match = re.search(r'(\d*\.\d+|\d+\.\d*)', line)
        if decimal_match:
            try:
                return float(decimal_match.group(1))
            except ValueError:
                pass

        # Check for fraction format (e.g., "9/10")
        fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', line)
        if fraction_match:
            try:
                numerator = float(fraction_match.group(1))
                denominator = float(fraction_match.group(2))
                if denominator != 0:
                    return numerator / denominator
            except ValueError:
                pass

    # If we couldn't find a specific format, try to extract the first number
    number_match = re.search(r'(\d+)', response)
    if number_match:
        try:
            number = float(number_match.group(1))
            # If it's greater than 1, assume it's a percentage
            if number > 1:
                return number / 100.0
            return number
        except ValueError:
            pass

    # Default to None if we couldn't extract anything
    return None

def save_evaluation_to_json(example: Dict[str, Any], prompt: str, response: str,
                           model_name: str, provider: str) -> str:
    """
    Save the evaluation data to a JSON file.

    Args:
        example: The original example
        prompt: The prompt sent to the AI
        response: The raw response from the AI
        model_name: The name of the model used
        provider: The API provider (OpenAI or Anthropic)

    Returns:
        The path to the saved JSON file
    """
    extracted_answer = extract_final_answer(response)
    expected_answer = example.get('verified_answer', example.get('answer', ''))

    data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "provider": provider,
        "model": model_name,
        "example": example,
        "prompt": prompt,
        "full_response": response,
        "extracted_answer": extracted_answer,
        "expected_answer": expected_answer,
        "match": extracted_answer.strip() == expected_answer.strip()
    }

    # Save to the output/evals directory
    output_dir = "output/evals"
    return save_to_json(data, f"evaluation_{provider}_{model_name.replace('-', '_')}", output_dir)

def save_multiple_evaluations_to_json(evaluations: List[Dict[str, Any]], model_name: str, provider: str) -> str:
    """
    Save multiple evaluations data to a single JSON file.

    Args:
        evaluations: List of evaluation results
        model_name: The name of the model used
        provider: The API provider (OpenAI or Anthropic)

    Returns:
        The path to the saved JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate summary statistics
    total = len(evaluations)
    matches = sum(1 for eval in evaluations if eval.get("match", False))
    match_rate = (matches / total) * 100 if total > 0 else 0

    # Calculate confidence statistics
    avg_confidence = sum(eval.get("confidence") for eval in evaluations if "confidence" in eval) / total if total > 0 else None
    missing_confidence = sum(1 for eval in evaluations if "confidence" not in eval)

    # Calculate calibration: how well confidence predicts correctness
    correct_confidences = [eval.get("confidence") for eval in evaluations if "confidence" in eval and eval.get("match", False)]
    incorrect_confidences = [eval.get("confidence") for eval in evaluations if "confidence" in eval and not eval.get("match", False) and "match" in eval]

    avg_confidence_when_correct = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0
    avg_confidence_when_incorrect = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0

    # Group by language and difficulty
    by_language = {}
    by_difficulty = {}

    for eval in evaluations:
        example = eval.get("example", {})
        lang = example.get("language", "unknown").lower()
        difficulty = example.get("difficulty", "?")

        if lang not in by_language:
            by_language[lang] = {
                "total": 0,
                "matches": 0,
                "by_difficulty": {},
                "avg_confidence": 0,
                "total_confidence": 0,
                "missing_confidence": 0
            }

        by_language[lang]["total"] += 1
        by_language[lang]["total_confidence"] += eval.get("confidence", 0)
        by_language[lang]["missing_confidence"] += 1 if "confidence" not in eval else 0
        if eval.get("match", False):
            by_language[lang]["matches"] += 1

        # Group by difficulty within language
        if difficulty not in by_language[lang]["by_difficulty"]:
            by_language[lang]["by_difficulty"][difficulty] = {
                "total": 0,
                "matches": 0,
                "total_confidence": 0,
                "avg_confidence": 0,
                "missing_confidence": 0
            }

        by_language[lang]["by_difficulty"][difficulty]["total"] += 1
        by_language[lang]["by_difficulty"][difficulty]["total_confidence"] += eval.get("confidence", 0)
        by_language[lang]["by_difficulty"][difficulty]["missing_confidence"] += 1 if "confidence" not in eval else 0

        if eval.get("match", False):
            by_language[lang]["by_difficulty"][difficulty]["matches"] += 1

        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {
                "total": 0,
                "matches": 0,
                "total_confidence": 0,
                "avg_confidence": 0,
                "missing_confidence": 0
            }

        by_difficulty[difficulty]["total"] += 1
        by_difficulty[difficulty]["total_confidence"] += eval.get("confidence", 0)
        by_difficulty[difficulty]["missing_confidence"] += 1 if "confidence" not in eval else 0

        if eval.get("match", False):
            by_difficulty[difficulty]["matches"] += 1

    # Calculate average confidence for each category
    for lang in by_language:
        by_language[lang]["avg_confidence"] = by_language[lang]["total_confidence"] / (by_language[lang]["total"] - by_language[lang]["missing_confidence"]) if by_language[lang]["total"] > 0 else 0

        for diff in by_language[lang]["by_difficulty"]:
            diff_data = by_language[lang]["by_difficulty"][diff]
            diff_data["avg_confidence"] = diff_data["total_confidence"] / (diff_data["total"] - diff_data["missing_confidence"]) if diff_data["total"] > 0 else 0

    for diff in by_difficulty:
        by_difficulty[diff]["avg_confidence"] = by_difficulty[diff]["total_confidence"] / (by_difficulty[diff]["total"] - by_difficulty[diff]["missing_confidence"]) if by_difficulty[diff]["total"] > 0 else 0

    data = {
        "timestamp": timestamp,
        "provider": provider,
        "model": model_name,
        "total_examples": total,
        "total_matches": matches,
        "match_rate": f"{match_rate:.2f}%",
        "avg_confidence": avg_confidence,
        "avg_confidence_when_correct": avg_confidence_when_correct,
        "avg_confidence_when_incorrect": avg_confidence_when_incorrect,
        "missing_confidence": missing_confidence,
        "by_language": by_language,
        "by_difficulty": by_difficulty,
        "evaluations": evaluations
    }

    # Save to the output/evals directory
    output_dir = "output/evals"
    return save_to_json(data, f"batch_evaluation_{provider}_{model_name.replace('-', '_')}", output_dir)

def evaluate_example(client, example: Dict[str, Any], model_name: str, provider: str) -> Dict[str, Any]:
    """
    Evaluate a single example and return the evaluation results.

    Args:
        client: The API client to use
        example: The example to evaluate
        model_name: The name of the model to use
        provider: The API provider (OpenAI or Anthropic)

    Returns:
        A dictionary with the evaluation results
    """
    # First, get confidence estimate
    confidence_prompt = CONFIDENCE_PROMPT_TEMPLATE.format(code=example['code'])

    if provider == 'openai':
        confidence_response = query_openai(client, confidence_prompt, model_name)
    else:  # anthropic
        confidence_response = query_anthropic(client, confidence_prompt, model_name)

    confidence = None
    if confidence_response:
        confidence = extract_confidence(confidence_response)

    # Then, evaluate the example
    prompt = EVALUATION_PROMPT_TEMPLATE.format(code=example['code'])

    if provider == 'openai':
        response = query_openai(client, prompt, model_name)
    else:  # anthropic
        response = query_anthropic(client, prompt, model_name)

    if not response:
        print(f"No response received from the API for example: {example.get('language', 'Unknown')} difficulty {example.get('difficulty', '?')}")
        return None

    # Extract the answer
    extracted_answer = extract_final_answer(response)
    expected_answer = example.get('verified_answer', example.get('answer', ''))

    # Create evaluation result
    return {
        "example": example,
        "prompt": prompt,
        "confidence_prompt": confidence_prompt,
        "confidence_response": confidence_response,
        "confidence": confidence,
        "full_response": response,
        "extracted_answer": extracted_answer,
        "expected_answer": expected_answer,
        "match": extracted_answer.strip() == expected_answer.strip()
    }

def main():
    """Main function to parse arguments and execute the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate AI models on code examples')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the JSON file containing code examples')
    parser.add_argument('--provider', choices=['openai', 'anthropic'], required=True,
                        help='API provider to use (openai or anthropic)')
    parser.add_argument('--model', type=str,
                        help='Model name to use (e.g., gpt-4 for OpenAI, claude-3-opus-20240229 for Anthropic)')
    parser.add_argument('--language', type=str,
                        help='Filter examples by language (e.g., Python, C, Javascript)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models from the specified provider and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be sent to the API without actually making the call')
    parser.add_argument('--run-all', action='store_true',
                        help='Run evaluation on all examples in the input file')

    args = parser.parse_args()

    # Create the appropriate client based on provider
    client = None
    if args.provider == 'openai':
        client = create_openai_client()
        if not client:
            return

        if args.list_models:
            list_openai_models(client)
            return

    else:  # anthropic
        client = create_anthropic_client()
        if not client:
            return

        if args.list_models:
            list_anthropic_models(client)
            return

    # Check if model is provided when not listing models
    if not args.model:
        print("Error: --model is required when not using --list-models")
        parser.print_help()
        return

    # Validate model name
    if not validate_model_name(args.provider, args.model):
        print(f"Warning: '{args.model}' doesn't look like a standard {args.provider} model name.")
        confirmation = input("Continue anyway? (y/n): ")
        if confirmation.lower() != 'y':
            return

    # Load examples from JSON file
    examples = load_examples_from_json(args.input_file)
    if not examples:
        return

    # Filter by language if specified
    if args.language:
        matching_examples = [ex for ex in examples if ex.get('language', '').lower() == args.language.lower()]
        if not matching_examples:
            print(f"No examples found for language: {args.language}")
            print(f"Available languages: {', '.join(set(ex.get('language', '') for ex in examples))}")
            return
        examples = matching_examples

    # If run-all is specified, evaluate all examples
    if args.run_all:
        if args.dry_run:
            print("\n=== DRY RUN ===")
            print(f"Provider: {args.provider}")
            print(f"Model: {args.model}")
            print(f"\nWould evaluate {len(examples)} examples.")
            print("No API call will be made. Exiting.")
            return

        print(f"Running evaluation on {len(examples)} examples. This may take a while...")
        evaluations = []
        correct_count = 0
        total_confidence = 0

        for i, example in enumerate(examples, 1):
            print(f"[{i}/{len(examples)}] Evaluating {example['language']} example (difficulty {example.get('difficulty', '?')})...", end="", flush=True)

            evaluation = evaluate_example(client, example, args.model, args.provider)
            if evaluation:
                evaluations.append(evaluation)
                confidence = evaluation.get("confidence", 0)
                total_confidence += confidence

                if evaluation["match"]:
                    result = "✓ CORRECT"
                    correct_count += 1
                else:
                    result = "✗ WRONG"

                print(f" {result} (confidence: {confidence:.2f})")
                print(f"  Expected: {evaluation['expected_answer']}")
                print(f"  Model's answer: {evaluation['extracted_answer']}")
            else:
                print(" ERROR")

        # Save all evaluations to a single file
        filename = save_multiple_evaluations_to_json(evaluations, args.model, args.provider)

        # Print summary
        match_rate = (correct_count / len(evaluations)) * 100 if evaluations else 0
        avg_confidence = total_confidence / len(evaluations) if evaluations else 0

        print("\n=== Evaluation Summary ===")
        print(f"Total examples: {len(evaluations)}")
        print(f"Correct answers: {correct_count} ({match_rate:.2f}%)")
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Results saved to {filename}")
        return

    # If not run-all, select a random example (original behavior)
    example = select_random_example(examples, args.language)
    if not example:
        return

    # Create the prompt
    prompt = EVALUATION_PROMPT_TEMPLATE.format(code=example['code'])

    expected_answer = example.get('verified_answer', example.get('answer', ''))
    # If dry run, show what would be sent and exit
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model}")
        print(f"\nExample selected:")
        print(f"Language: {example['language']}")
        print(f"Difficulty: {example['difficulty']}")
        print(f"Expected Answer: {expected_answer}")
        print("\nPrompt that would be sent:")
        print("=" * 40)
        print(prompt)
        print("=" * 40)
        print("\nNo API call will be made. Exiting.")
        return

    # Query the model
    if args.provider == 'openai':
        response = query_openai(client, prompt, args.model)
    else:  # anthropic
        response = query_anthropic(client, prompt, args.model)

    if not response:
        print("No response received from the API")
        return

    # Extract the answer
    extracted_answer = extract_final_answer(response)

    print("\nEvaluation Results:")
    print(f"Language: {example['language']}")
    print(f"Difficulty: {example['difficulty']}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Model's Answer: {extracted_answer}")
    print(f"Match: {extracted_answer.strip() == expected_answer.strip()}")

    # Save to JSON
    filename = save_evaluation_to_json(example, prompt, response, args.model, args.provider)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()

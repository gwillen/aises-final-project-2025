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

# Confidence prompt strategies
CONFIDENCE_PROMPTS = {
    "standard": {
        "description": "Standard probability estimation (0-100% or 0-1)",
        "before_template": """
In a moment, I will ask you what this program outputs. You will have to provide only your best immediate guess at the final answer, without reasoning, analysis, or commentary. Before I do that, first I want to evaluate your confidence that you can answer correctly. Please give me an estimate of your probability that your answer will be correct. Again, please give only the estimate, on a line by itself, without reasoning, analysis, or commentary.

```
{code}
```
""",
        "after_template": """
You just gave an answer for the program output. Now, please give me an estimate of your probability that your answer is correct. Again, please give only the estimate, on a line by itself, without reasoning, analysis, or commentary.

```
{code}
```
""",
        "extract_func": "extract_standard_confidence"
    },
    "inverse": {
        "description": "Ask for 1/p formulation: number of examples to get one wrong",
        "before_template": """
In a moment, I will ask you what this program outputs. You will have to provide only your best immediate guess at the final answer, without reasoning, analysis, or commentary. Before I do that, first I want to evaluate your confidence that you can answer correctly. Please give me an estimate of the probability p that your answer is correct, written as 1/p: the number of similar examples you would expect to solve, in order to get one wrong, on average. Again, please give only the estimate, on a line by itself, without reasoning, analysis, or commentary.

```
{code}
```
""",
        "after_template": """
You just gave an answer for the program output. Now, please give me an estimate of the probability p that your answer is correct, written as 1/p: the number of similar examples you would expect to solve, in order to get one wrong, on average. Again, please give only the estimate, on a line by itself, without reasoning, analysis, or commentary.

```
{code}
```
""",
        "extract_func": "extract_inverse_confidence"
    }
}

# Default confidence prompt strategy
DEFAULT_CONFIDENCE_STRATEGY = "standard"

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

def extract_standard_confidence(response: str) -> float:
    """
    Extract the confidence estimate as a standard probability (0-1) from the response.

    Args:
        response: The full response from the AI

    Returns:
        The extracted confidence as a float between 0 and 1, or None if extraction fails
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
        percentage_match = re.search(r'(\d+(?:\.\d+)?)(?:\s*%|\s+percent)', line)
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

def extract_inverse_confidence(response: str) -> float:
    """
    Extract the confidence estimate from a 1/p formulation (number of examples until one error).

    Args:
        response: The full response from the AI

    Returns:
        The extracted confidence as a float between 0 and 1, or None if extraction fails
    """
    response = response.strip()

    # Look for formats like "1/10", "10", or "10:1"
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Try to extract ratios like "10:1" or "10 to 1"
        ratio_match = re.search(r'(\d+(?:\.\d+)?)\s*(?::|to)\s*1', line)
        if ratio_match:
            try:
                ratio = float(ratio_match.group(1))
                if ratio > 0:
                    return (ratio - 1) / ratio  # Convert to probability
                return None
            except ValueError:
                pass

        # Look for "1 in X" format
        one_in_x_match = re.search(r'1\s+in\s+(\d+(?:\.\d+)?)', line)
        if one_in_x_match:
            try:
                x = float(one_in_x_match.group(1))
                if x > 0:
                    return 1 - (1 / x)  # Convert to probability
                return None
            except ValueError:
                pass

        # Try to extract fractions like "1/10"
        inverse_fraction_match = re.search(r'1\s*/\s*(\d+(?:\.\d+)?)', line)
        if inverse_fraction_match:
            try:
                denominator = float(inverse_fraction_match.group(1))
                if denominator > 0:
                    return 1 - (1 / denominator)  # Convert to probability
                return None
            except ValueError:
                pass

        # Just a number (assumed to be the number of examples until error)
        solo_number_match = re.search(r'^(\d+(?:\.\d+)?)$', line)
        if solo_number_match:
            try:
                number = float(solo_number_match.group(1))
                if number > 0:
                    return (number - 1) / number  # Convert to probability
                return None
            except ValueError:
                pass

    # Default to None if we couldn't extract anything
    return None

def extract_confidence(response: str, strategy: str = DEFAULT_CONFIDENCE_STRATEGY) -> float:
    """
    Extract the confidence estimate from the response using the specified strategy.

    Args:
        response: The full response from the AI
        strategy: The confidence extraction strategy to use

    Returns:
        The extracted confidence as a float between 0 and 1, or None if extraction fails
    """
    if strategy not in CONFIDENCE_PROMPTS:
        strategy = DEFAULT_CONFIDENCE_STRATEGY

    extract_func_name = CONFIDENCE_PROMPTS[strategy]["extract_func"]

    # Call the appropriate extraction function
    if extract_func_name == "extract_standard_confidence":
        return extract_standard_confidence(response)
    elif extract_func_name == "extract_inverse_confidence":
        return extract_inverse_confidence(response)
    else:
        return extract_standard_confidence(response)  # Fallback to standard

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

def save_single_evaluation_with_confidence(evaluation: Dict[str, Any], model_name: str, provider: str) -> str:
    """
    Save a single evaluation with confidence data to a JSON file.

    Args:
        evaluation: The evaluation result including confidence data
        model_name: The name of the model used
        provider: The API provider (OpenAI or Anthropic)

    Returns:
        The path to the saved JSON file
    """
    data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "provider": provider,
        "model": model_name,
        "confidence_strategies": evaluation.get("confidence_strategies", []),
        **evaluation
    }

    # Save to the output/evals directory
    output_dir = "output/evals"
    return save_to_json(data, f"evaluation_with_confidence_{provider}_{model_name.replace('-', '_')}", output_dir)

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

    # Identify all confidence strategies used
    all_strategies = set()
    for eval in evaluations:
        if "confidence_results" in eval:
            all_strategies.update(eval["confidence_results"].keys())

    # Calculate confidence statistics for each strategy
    confidence_stats = {}
    for strategy in all_strategies:
        strategy_data = {
            "before_total_confidence": 0,
            "before_valid_count": 0,
            "before_avg_confidence": None,
            "after_total_confidence": 0,
            "after_valid_count": 0,
            "after_avg_confidence": None,
            "before_correct_confidences": [],
            "before_incorrect_confidences": [],
            "after_correct_confidences": [],
            "after_incorrect_confidences": [],
            "before_avg_confidence_when_correct": None,
            "before_avg_confidence_when_incorrect": None,
            "after_avg_confidence_when_correct": None,
            "after_avg_confidence_when_incorrect": None,
            "avg_confidence_change": None,
            "avg_confidence_change_when_correct": None,
            "avg_confidence_change_when_incorrect": None,
            "confidence_changes_by_result": {
                "increased": 0,
                "decreased": 0,
                "unchanged": 0
            }
        }

        for eval in evaluations:
            if "confidence_results" in eval and strategy in eval["confidence_results"]:
                before_confidence = eval["confidence_results"][strategy].get("before_confidence")
                after_confidence = eval["confidence_results"][strategy].get("after_confidence")

                # Track before confidence
                if before_confidence is not None:
                    strategy_data["before_total_confidence"] += before_confidence
                    strategy_data["before_valid_count"] += 1

                    if eval.get("match", False):
                        strategy_data["before_correct_confidences"].append(before_confidence)
                    else:
                        strategy_data["before_incorrect_confidences"].append(before_confidence)

                # Track after confidence
                if after_confidence is not None:
                    strategy_data["after_total_confidence"] += after_confidence
                    strategy_data["after_valid_count"] += 1

                    if eval.get("match", False):
                        strategy_data["after_correct_confidences"].append(after_confidence)
                    else:
                        strategy_data["after_incorrect_confidences"].append(after_confidence)

                # Track confidence changes
                if before_confidence is not None and after_confidence is not None:
                    change = after_confidence - before_confidence
                    if change > 0:
                        strategy_data["confidence_changes_by_result"]["increased"] += 1
                    elif change < 0:
                        strategy_data["confidence_changes_by_result"]["decreased"] += 1
                    else:
                        strategy_data["confidence_changes_by_result"]["unchanged"] += 1

        # Calculate averages
        if strategy_data["before_valid_count"] > 0:
            strategy_data["before_avg_confidence"] = strategy_data["before_total_confidence"] / strategy_data["before_valid_count"]

        if strategy_data["after_valid_count"] > 0:
            strategy_data["after_avg_confidence"] = strategy_data["after_total_confidence"] / strategy_data["after_valid_count"]

        if strategy_data["before_correct_confidences"]:
            strategy_data["before_avg_confidence_when_correct"] = sum(strategy_data["before_correct_confidences"]) / len(strategy_data["before_correct_confidences"])

        if strategy_data["before_incorrect_confidences"]:
            strategy_data["before_avg_confidence_when_incorrect"] = sum(strategy_data["before_incorrect_confidences"]) / len(strategy_data["before_incorrect_confidences"])

        if strategy_data["after_correct_confidences"]:
            strategy_data["after_avg_confidence_when_correct"] = sum(strategy_data["after_correct_confidences"]) / len(strategy_data["after_correct_confidences"])

        if strategy_data["after_incorrect_confidences"]:
            strategy_data["after_avg_confidence_when_incorrect"] = sum(strategy_data["after_incorrect_confidences"]) / len(strategy_data["after_incorrect_confidences"])

        # Calculate average confidence change
        confidence_changes = []
        correct_confidence_changes = []
        incorrect_confidence_changes = []

        for eval in evaluations:
            if "confidence_results" in eval and strategy in eval["confidence_results"]:
                before_confidence = eval["confidence_results"][strategy].get("before_confidence")
                after_confidence = eval["confidence_results"][strategy].get("after_confidence")

                if before_confidence is not None and after_confidence is not None:
                    change = after_confidence - before_confidence
                    confidence_changes.append(change)

                    if eval.get("match", False):
                        correct_confidence_changes.append(change)
                    else:
                        incorrect_confidence_changes.append(change)

        if confidence_changes:
            strategy_data["avg_confidence_change"] = sum(confidence_changes) / len(confidence_changes)

        if correct_confidence_changes:
            strategy_data["avg_confidence_change_when_correct"] = sum(correct_confidence_changes) / len(correct_confidence_changes)

        if incorrect_confidence_changes:
            strategy_data["avg_confidence_change_when_incorrect"] = sum(incorrect_confidence_changes) / len(incorrect_confidence_changes)

        confidence_stats[strategy] = strategy_data

    # Group by language and difficulty
    by_language = {}
    by_difficulty = {}

    for eval in evaluations:
        example = eval.get("example", {})
        lang = example.get("language", "unknown").lower()
        difficulty = example.get("difficulty", "?")

        # Initialize language data if needed
        if lang not in by_language:
            by_language[lang] = {
                "total": 0,
                "matches": 0,
                "by_difficulty": {},
                "by_confidence_strategy": {}
            }

            # Initialize confidence data for each strategy
            for strategy in all_strategies:
                by_language[lang]["by_confidence_strategy"][strategy] = {
                    "before_total_confidence": 0,
                    "before_valid_count": 0,
                    "before_avg_confidence": None,
                    "after_total_confidence": 0,
                    "after_valid_count": 0,
                    "after_avg_confidence": None
                }

        by_language[lang]["total"] += 1

        # Update confidence data for each strategy
        for strategy in all_strategies:
            if "confidence_results" in eval and strategy in eval["confidence_results"]:
                before_confidence = eval["confidence_results"][strategy].get("before_confidence")
                after_confidence = eval["confidence_results"][strategy].get("after_confidence")

                if before_confidence is not None:
                    by_language[lang]["by_confidence_strategy"][strategy]["before_total_confidence"] += before_confidence
                    by_language[lang]["by_confidence_strategy"][strategy]["before_valid_count"] += 1

                if after_confidence is not None:
                    by_language[lang]["by_confidence_strategy"][strategy]["after_total_confidence"] += after_confidence
                    by_language[lang]["by_confidence_strategy"][strategy]["after_valid_count"] += 1

        if eval.get("match", False):
            by_language[lang]["matches"] += 1

        # Group by difficulty within language
        if difficulty not in by_language[lang]["by_difficulty"]:
            by_language[lang]["by_difficulty"][difficulty] = {
                "total": 0,
                "matches": 0,
                "by_confidence_strategy": {}
            }

            # Initialize confidence data for each strategy
            for strategy in all_strategies:
                by_language[lang]["by_difficulty"][difficulty]["by_confidence_strategy"][strategy] = {
                    "before_total_confidence": 0,
                    "before_valid_count": 0,
                    "before_avg_confidence": None,
                    "after_total_confidence": 0,
                    "after_valid_count": 0,
                    "after_avg_confidence": None
                }

        by_language[lang]["by_difficulty"][difficulty]["total"] += 1

        # Update confidence data for difficulty level and strategy
        for strategy in all_strategies:
            if "confidence_results" in eval and strategy in eval["confidence_results"]:
                before_confidence = eval["confidence_results"][strategy].get("before_confidence")
                after_confidence = eval["confidence_results"][strategy].get("after_confidence")

                if before_confidence is not None:
                    by_language[lang]["by_difficulty"][difficulty]["by_confidence_strategy"][strategy]["before_total_confidence"] += before_confidence
                    by_language[lang]["by_difficulty"][difficulty]["by_confidence_strategy"][strategy]["before_valid_count"] += 1

                if after_confidence is not None:
                    by_language[lang]["by_difficulty"][difficulty]["by_confidence_strategy"][strategy]["after_total_confidence"] += after_confidence
                    by_language[lang]["by_difficulty"][difficulty]["by_confidence_strategy"][strategy]["after_valid_count"] += 1

        if eval.get("match", False):
            by_language[lang]["by_difficulty"][difficulty]["matches"] += 1

        # Initialize difficulty data if needed
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {
                "total": 0,
                "matches": 0,
                "by_confidence_strategy": {}
            }

            # Initialize confidence data for each strategy
            for strategy in all_strategies:
                by_difficulty[difficulty]["by_confidence_strategy"][strategy] = {
                    "before_total_confidence": 0,
                    "before_valid_count": 0,
                    "before_avg_confidence": None,
                    "after_total_confidence": 0,
                    "after_valid_count": 0,
                    "after_avg_confidence": None
                }

        by_difficulty[difficulty]["total"] += 1

        # Update confidence data for each strategy
        for strategy in all_strategies:
            if "confidence_results" in eval and strategy in eval["confidence_results"]:
                before_confidence = eval["confidence_results"][strategy].get("before_confidence")
                after_confidence = eval["confidence_results"][strategy].get("after_confidence")

                if before_confidence is not None:
                    by_difficulty[difficulty]["by_confidence_strategy"][strategy]["before_total_confidence"] += before_confidence
                    by_difficulty[difficulty]["by_confidence_strategy"][strategy]["before_valid_count"] += 1

                if after_confidence is not None:
                    by_difficulty[difficulty]["by_confidence_strategy"][strategy]["after_total_confidence"] += after_confidence
                    by_difficulty[difficulty]["by_confidence_strategy"][strategy]["after_valid_count"] += 1

        if eval.get("match", False):
            by_difficulty[difficulty]["matches"] += 1

    # Calculate average confidence for each category and strategy
    for lang in by_language:
        for strategy in all_strategies:
            strategy_data = by_language[lang]["by_confidence_strategy"][strategy]
            if strategy_data["before_valid_count"] > 0:
                strategy_data["before_avg_confidence"] = strategy_data["before_total_confidence"] / strategy_data["before_valid_count"]
            if strategy_data["after_valid_count"] > 0:
                strategy_data["after_avg_confidence"] = strategy_data["after_total_confidence"] / strategy_data["after_valid_count"]

        for diff in by_language[lang]["by_difficulty"]:
            for strategy in all_strategies:
                strategy_data = by_language[lang]["by_difficulty"][diff]["by_confidence_strategy"][strategy]
                if strategy_data["before_valid_count"] > 0:
                    strategy_data["before_avg_confidence"] = strategy_data["before_total_confidence"] / strategy_data["before_valid_count"]
                if strategy_data["after_valid_count"] > 0:
                    strategy_data["after_avg_confidence"] = strategy_data["after_total_confidence"] / strategy_data["after_valid_count"]

    for diff in by_difficulty:
        for strategy in all_strategies:
            strategy_data = by_difficulty[diff]["by_confidence_strategy"][strategy]
            if strategy_data["before_valid_count"] > 0:
                strategy_data["before_avg_confidence"] = strategy_data["before_total_confidence"] / strategy_data["before_valid_count"]
            if strategy_data["after_valid_count"] > 0:
                strategy_data["after_avg_confidence"] = strategy_data["after_total_confidence"] / strategy_data["after_valid_count"]

    data = {
        "timestamp": timestamp,
        "provider": provider,
        "model": model_name,
        "total_examples": total,
        "total_matches": matches,
        "match_rate": f"{match_rate:.2f}%",
        "confidence_strategies": list(all_strategies),
        "confidence_stats": confidence_stats,
        "by_language": by_language,
        "by_difficulty": by_difficulty,
        "evaluations": evaluations
    }

    # Save to the output/evals directory
    output_dir = "output/evals"
    return save_to_json(data, f"batch_evaluation_{provider}_{model_name.replace('-', '_')}", output_dir)

def evaluate_example(client, example: Dict[str, Any], model_name: str, provider: str,
                    confidence_strategies: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate a single example and return the evaluation results.

    Args:
        client: The API client to use
        example: The example to evaluate
        model_name: The name of the model to use
        provider: The API provider (OpenAI or Anthropic)
        confidence_strategies: List of confidence strategies to use (default: [DEFAULT_CONFIDENCE_STRATEGY])

    Returns:
        A dictionary with the evaluation results
    """
    # Use default confidence strategy if none specified
    if not confidence_strategies:
        confidence_strategies = [DEFAULT_CONFIDENCE_STRATEGY]

    # Make sure all strategies are valid
    confidence_strategies = [s for s in confidence_strategies if s in CONFIDENCE_PROMPTS]
    if not confidence_strategies:
        confidence_strategies = [DEFAULT_CONFIDENCE_STRATEGY]

    # Dictionary to hold confidence results
    confidence_results = {}

    # Evaluate "before" confidence for each strategy
    for strategy in confidence_strategies:
        # Get the prompt template for this strategy
        confidence_prompt = CONFIDENCE_PROMPTS[strategy]["before_template"].format(code=example['code'])

        # Query the model
        if provider == 'openai':
            confidence_response = query_openai(client, confidence_prompt, model_name)
        else:  # anthropic
            confidence_response = query_anthropic(client, confidence_prompt, model_name)

        # Extract confidence using the appropriate function
        confidence = None
        if confidence_response:
            confidence = extract_confidence(confidence_response, strategy)

        # Store results
        confidence_results[strategy] = {
            "before_prompt": confidence_prompt,
            "before_response": confidence_response,
            "before_confidence": confidence,
            "after_prompt": None,
            "after_response": None,
            "after_confidence": None
        }

    # Now evaluate the actual example
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

    # Evaluate "after" confidence for each strategy now that we have the answer
    for strategy in confidence_strategies:
        # Get the prompt template for this strategy
        confidence_prompt = CONFIDENCE_PROMPTS[strategy]["after_template"].format(code=example['code'])

        # Query the model
        if provider == 'openai':
            confidence_response = query_openai(client, confidence_prompt, model_name)
        else:  # anthropic
            confidence_response = query_anthropic(client, confidence_prompt, model_name)

        # Extract confidence using the appropriate function
        after_confidence = None
        if confidence_response:
            after_confidence = extract_confidence(confidence_response, strategy)

        # Store results
        confidence_results[strategy]["after_prompt"] = confidence_prompt
        confidence_results[strategy]["after_response"] = confidence_response
        confidence_results[strategy]["after_confidence"] = after_confidence

    # Create evaluation result with combined confidence data
    result = {
        "example": example,
        "prompt": prompt,
        "full_response": response,
        "extracted_answer": extracted_answer,
        "expected_answer": expected_answer,
        "match": extracted_answer.strip() == expected_answer.strip(),
        "confidence_strategies": confidence_strategies,
        "confidence_results": confidence_results
    }

    # Add first confidence as the primary one for backward compatibility
    primary_strategy = confidence_strategies[0]
    result["before_confidence"] = confidence_results[primary_strategy]["before_confidence"]
    result["after_confidence"] = confidence_results[primary_strategy]["after_confidence"]
    result["confidence"] = result["before_confidence"]  # Keep for backward compatibility

    return result

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
    parser.add_argument('--confidence-strategies', type=str, default=DEFAULT_CONFIDENCE_STRATEGY,
                        help=f'Comma-separated list of confidence strategies to use. Available: {", ".join(CONFIDENCE_PROMPTS.keys())}')
    parser.add_argument('--list-confidence-strategies', action='store_true',
                        help='List available confidence strategies and exit')

    args = parser.parse_args()

    # List confidence strategies if requested
    if args.list_confidence_strategies:
        print("Available confidence estimation strategies:")
        for strategy, details in CONFIDENCE_PROMPTS.items():
            print(f"  - {strategy}: {details['description']}")
        return

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

    # Parse confidence strategies
    confidence_strategies = [s.strip() for s in args.confidence_strategies.split(",")]
    invalid_strategies = [s for s in confidence_strategies if s not in CONFIDENCE_PROMPTS]
    if invalid_strategies:
        print(f"Warning: Unknown confidence strategies: {', '.join(invalid_strategies)}")
        print(f"Available strategies: {', '.join(CONFIDENCE_PROMPTS.keys())}")
        print(f"Using default strategy: {DEFAULT_CONFIDENCE_STRATEGY}")
        confidence_strategies = [DEFAULT_CONFIDENCE_STRATEGY]

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
            print(f"Confidence strategies: {', '.join(confidence_strategies)}")
            print(f"\nWould evaluate {len(examples)} examples.")
            print("No API call will be made. Exiting.")
            return

        print(f"Running evaluation on {len(examples)} examples. This may take a while...")
        print(f"Using confidence strategies: {', '.join(confidence_strategies)}")
        evaluations = []
        correct_count = 0
        total_confidence = {strategy: 0 for strategy in confidence_strategies}
        valid_confidence_count = {strategy: 0 for strategy in confidence_strategies}

        for i, example in enumerate(examples, 1):
            print(f"[{i}/{len(examples)}] Evaluating {example['language']} example (difficulty {example.get('difficulty', '?')})...", end="", flush=True)

            evaluation = evaluate_example(client, example, args.model, args.provider, confidence_strategies)
            if evaluation:
                evaluations.append(evaluation)

                # Track confidence for each strategy
                for strategy in confidence_strategies:
                    confidence = evaluation["confidence_results"][strategy]["confidence"]
                    if confidence is not None:
                        total_confidence[strategy] += confidence
                        valid_confidence_count[strategy] += 1

                # Use primary confidence for display
                before_confidence = evaluation.get("before_confidence")
                after_confidence = evaluation.get("after_confidence")
                before_display = f" (before: {before_confidence:.2f})" if before_confidence is not None else ""
                after_display = f" (after: {after_confidence:.2f})" if after_confidence is not None else ""
                confidence_display = f"{before_display}{after_display}"

                if evaluation["match"]:
                    result = "✓ CORRECT"
                    correct_count += 1
                else:
                    result = "✗ WRONG"

                print(f" {result}{confidence_display}")
                print(f"  Expected: {evaluation['expected_answer']}")
                print(f"  Model's answer: {evaluation['extracted_answer']}")

                # Show all confidence values if there are multiple strategies
                if len(confidence_strategies) > 1:
                    for strategy in confidence_strategies:
                        before_conf = evaluation["confidence_results"][strategy]["before_confidence"]
                        after_conf = evaluation["confidence_results"][strategy]["after_confidence"]
                        before_str = f"{before_conf:.2f}" if before_conf is not None else "N/A"
                        after_str = f"{after_conf:.2f}" if after_conf is not None else "N/A"
                        print(f"  {strategy} confidence: before={before_str}, after={after_str}")
            else:
                print(" ERROR")

        # Save all evaluations to a single file
        filename = save_multiple_evaluations_to_json(evaluations, args.model, args.provider)

        # Print summary
        match_rate = (correct_count / len(evaluations)) * 100 if evaluations else 0

        print("\n=== Evaluation Summary ===")
        print(f"Total examples: {len(evaluations)}")
        print(f"Correct answers: {correct_count} ({match_rate:.2f}%)")

        # Print confidence summaries for each strategy
        for strategy in confidence_strategies:
            if valid_confidence_count[strategy] > 0:
                avg_confidence = total_confidence[strategy] / valid_confidence_count[strategy]
                print(f"{strategy} average confidence: {avg_confidence:.2f} (valid: {valid_confidence_count[strategy]}/{len(evaluations)})")

        print(f"Results saved to {filename}")
        return

    # If not run-all, select a random example (original behavior)
    example = select_random_example(examples, args.language)
    if not example:
        return

    # If dry run, show what would be sent and exit
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model}")
        print(f"Confidence strategies: {', '.join(confidence_strategies)}")
        print(f"\nExample selected:")
        print(f"Language: {example['language']}")
        print(f"Difficulty: {example['difficulty']}")
        expected_answer = example.get('verified_answer', example.get('answer', ''))
        print(f"Expected Answer: {expected_answer}")

        print("\nPrompts that would be sent:")
        # Show all confidence prompts
        for i, strategy in enumerate(confidence_strategies):
            print(f"\n=== Confidence Prompt {i+1}: {strategy} ===")
            print("=" * 40)
            print(CONFIDENCE_PROMPTS[strategy]["before_template"].format(code=example['code']))
            print("=" * 40)

        # Show evaluation prompt
        print("\n=== Evaluation Prompt ===")
        print("=" * 40)
        print(EVALUATION_PROMPT_TEMPLATE.format(code=example['code']))
        print("=" * 40)

        print("\nNo API calls will be made. Exiting.")
        return

    # Evaluate the example
    evaluation = evaluate_example(client, example, args.model, args.provider, confidence_strategies)
    if not evaluation:
        print("Failed to evaluate example")
        return

    # Print results
    print("\nEvaluation Results:")
    print(f"Language: {example['language']}")
    print(f"Difficulty: {example['difficulty']}")
    print(f"Expected Answer: {evaluation['expected_answer']}")
    print(f"Model's Answer: {evaluation['extracted_answer']}")
    print(f"Match: {evaluation['match']}")

    # Print confidence results for each strategy
    print("\nConfidence Estimates:")
    for strategy in confidence_strategies:
        before_confidence = evaluation["confidence_results"][strategy]["before_confidence"]
        after_confidence = evaluation["confidence_results"][strategy]["after_confidence"]
        before_str = f"{before_confidence:.2f}" if before_confidence is not None else "Unable to extract"
        after_str = f"{after_confidence:.2f}" if after_confidence is not None else "Unable to extract"
        print(f"  {strategy}: before={before_str}, after={after_str}")

        # Show confidence change
        if before_confidence is not None and after_confidence is not None:
            change = after_confidence - before_confidence
            direction = "increased" if change > 0 else "decreased" if change < 0 else "unchanged"
            print(f"    Change: {direction} by {abs(change):.2f}")

    # Save to JSON
    filename = save_single_evaluation_with_confidence(evaluation, args.model, args.provider)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()

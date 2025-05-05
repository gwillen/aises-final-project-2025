#!/usr/bin/env python3
"""
Script to evaluate AI's ability to solve coding examples.

This script:
1. Takes a JSON file with code examples
2. Picks a random example (or one matching criteria)
3. Sends it to an AI model for evaluation
4. Saves the result to a JSON file
"""

import json
import argparse
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import shared utilities
from ai_utils import (
    create_client,
    query_model,
    list_models,
    save_to_json,
    query_model_with_history,
    StandardConversation,
    StandardMessage,
)

EVALUATION_PROMPT_TEMPLATE = """\
What result does this code output or return? Please provide only your best immediate guess \
at the final answer, on a line by itself; no reasoning, analysis, or commentary.

```
{code}
```
"""

# --- Confidence Prompt Components ---
PROMPT_COMMON_SUFFIX = "Again, please give only the estimate, on a line by " \
    "itself, without reasoning, analysis, or commentary."

PROMPT_CODE_BLOCK = "```\n{code}\n```"

PROMPT_SUPERFORECASTER = "You are a superforecaster. You are excel at " \
        "pattern recognition, cognitive flexibility, and open-mindedness. " \
        "You are well-calibrated, and you are careful to avoid being over- " \
        "or under-confident in your beliefs."

PROMPT_BEFORE_PREFIX = "In a moment, I will ask you what this code outputs or returns. " \
    "You will have to provide only your best immediate guess at the final answer, " \
    "without reasoning, analysis, or commentary. Before I do that, first I want to " \
    "evaluate your confidence that you can answer correctly."

PROMPT_STRATEGY_REQUESTS = {
    "standard": "Please give me an estimate of the probability that your " \
        "answer is correct, written as a percentage between 0 and 100.",
    "inverse": "Please give me an estimate of the number of similar problems " \
        "you would expect to attempt, on average, before getting one wrong. ",
    "betting": "You have $1000 of play money. How much of it will you bet " \
        "that your answer is correct? Please give a number between 0 and 1000.",
    "verbal": "Please describe in words how confident you are that your " \
        "answer is correct.", \
    "onetoten": "Please rate your confidence on a scale from 1 to 10, with " \
        "10 being most confident."
}
# --- End Prompt Components ---

# Confidence prompt strategy definitions (excluding templates)
CONFIDENCE_STRATEGY_DEFINITIONS = {
    "standard": {
        "description": "Standard probability estimation (0-100% or 0-1)",
        "extract_func": "extract_standard_confidence"
    },
    "inverse": {
        "description": "Ask for 1/(1-p) formulation: number of examples to get one wrong",
        "extract_func": "extract_inverse_confidence"
    },
    "betting": {
        "description": "Betting confidence estimate: how much money to bet on the answer",
        "extract_func": "extract_betting_confidence"
    },
    "verbal": {
        "description": "Verbal confidence estimate",
        "extract_func": "extract_verbal_confidence"
    }
}

def get_before_confidence_prompt(strategy: str, code: str, superforecast: bool) -> str:
    """
    Constructs the 'before' confidence prompt (including the code).

    Args:
        strategy: The confidence strategy ('standard', 'inverse').
        code: The code snippet for the prompt.

    Returns:
        The fully constructed prompt string.
    """
    strategy_request = PROMPT_STRATEGY_REQUESTS[strategy]
    superforecast_prefix = ""
    if superforecast:
        superforecast_prefix = f"{PROMPT_SUPERFORECASTER}\n\n"

    # Assemble the prompt parts
    prompt = f"{superforecast_prefix}{PROMPT_BEFORE_PREFIX} {strategy_request} {PROMPT_COMMON_SUFFIX}\n\n{PROMPT_CODE_BLOCK.format(code=code)}"
    return prompt.strip()

def get_after_confidence_prompt_text(strategy: str, superforecast: bool) -> str:
    """
    Constructs the text for the 'after' confidence prompt (without code).

    Args:
        strategy: The confidence strategy ('standard', 'inverse').

    Returns:
        The prompt text string.
    """
    strategy_request = PROMPT_STRATEGY_REQUESTS[strategy]
    superforecast_prefix = ""
    if superforecast:
        superforecast_prefix = f"{PROMPT_SUPERFORECASTER}\n\n"

    return f"{superforecast_prefix}{strategy_request} {PROMPT_COMMON_SUFFIX}".strip()

def load_examples_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load examples from a JSON file.

    Also extracts top-level generation metadata (provider, model, timestamp)
    and adds it to each individual example dictionary.

    Args:
        file_path: Path to the JSON file

    Returns:
        A list of example dictionaries, enriched with generation metadata.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract top-level generation metadata
        original_provider = data.get('provider', 'unknown')
        original_model = data.get('model', 'unknown')
        original_timestamp = data.get('timestamp', 'unknown')

        if 'examples' not in data or not isinstance(data['examples'], list):
            print("Error: JSON file doesn't contain an 'examples' array")
            return []

        examples = data['examples']

        # Add generation metadata to each example
        enriched_examples = []
        for example in examples:
            example['original_provider'] = original_provider
            example['original_model'] = original_model
            example['original_timestamp'] = original_timestamp
            enriched_examples.append(example)

        return enriched_examples

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

def extract_betting_confidence(response: str) -> float:
    """
    Extract the confidence estimate from a betting response.

    Args:
        response: The full response from the AI

    Returns:
        The extracted confidence as a float between 0 and 1, or None if extraction fails
    """
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Try to extract a number
        number_match = re.search(r'^(\d+(?:\.\d+)?)', line)
        if number_match:
            try:
                return float(number_match.group(1))
            except ValueError:
                pass

    # Default to None if we couldn't extract anything
    return None

def extract_verbal_confidence(response: str) -> float:
    """
    Extract the confidence estimate from a verbal response.

    Args:
        response: The full response from the AI

    Returns:
        None
    """
    # For now, just see what kinds of responses it gives.
    return None

def extract_confidence(response: str, strategy: str) -> float:
    """
    Extract the confidence estimate from the response using the specified strategy.

    Args:
        response: The full response from the AI
        strategy: The confidence extraction strategy to use

    Returns:
        The extracted confidence as a float between 0 and 1, or None if extraction fails
    """
    extract_func_name = CONFIDENCE_STRATEGY_DEFINITIONS[strategy]["extract_func"]

    if not globals().get(extract_func_name) or not callable(globals()[extract_func_name]):
        raise ValueError(f"Unknown confidence extraction strategy: {strategy}")

    return globals()[extract_func_name](response)

def save_single_evaluation_with_confidence(evaluation: Dict[str, Any], model_name: str, provider: str) -> str:
    # This function is now deprecated, save_multiple_evaluations_to_json handles single cases.
    raise DeprecationWarning("save_single_evaluation_with_confidence is deprecated. Use save_multiple_evaluations_to_json.")

# --- Helper Functions for Statistics ---

TIME_POINTS = ["before", "after"]

def _initialize_stats_block(strategies: List[str]) -> Dict[str, Any]:
    """
    Initializes a dictionary structure for tracking evaluation statistics.

    Args:
        strategies: List of confidence strategy names.

    Returns:
        A dictionary initialized with zeros and empty lists for statistics.
    """
    stats_block = {
        "total": 0,
        "matches": 0,
        "match_rate": None,  # Calculated later
        "by_confidence_strategy": {},
        "confidence_stats": {} # Final calculated confidence averages/stats
    }
    for strategy in strategies:
        # --- Aggregation Structure ---
        stats_block["by_confidence_strategy"][strategy] = {
            # Nested dict for before/after time points
            time_point: {
                "total_confidence": 0.0, "valid_count": 0,
                "correct_confidences": [], "incorrect_confidences": []
            } for time_point in TIME_POINTS
        }
        # Add structure for confidence changes (involves both time points)
        strategy_agg = stats_block["by_confidence_strategy"][strategy]
        strategy_agg["confidence_changes"] = []
        strategy_agg["correct_confidence_changes"] = []
        strategy_agg["incorrect_confidence_changes"] = []
        strategy_agg["confidence_changes_by_result"] = {"increased": 0, "decreased": 0, "unchanged": 0}

        # --- Final Calculated Stats Structure ---
        stats_block["confidence_stats"][strategy] = {
            time_point: {
                "avg_confidence": None,
                "avg_confidence_when_correct": None,
                "avg_confidence_when_incorrect": None,
            } for time_point in TIME_POINTS
        }
        # Add structure for calculated change stats
        strategy_calc = stats_block["confidence_stats"][strategy]
        strategy_calc["avg_confidence_change"] = None
        strategy_calc["avg_confidence_change_when_correct"] = None
        strategy_calc["avg_confidence_change_when_incorrect"] = None
        strategy_calc["confidence_changes_by_result"] = {"increased": 0, "decreased": 0, "unchanged": 0}

    return stats_block

def _update_stats_block(stats_block: Dict[str, Any], evaluation: Dict[str, Any], strategies: List[str]) -> None:
    """
    Updates a statistics block based on a single evaluation result.

    Args:
        stats_block: The statistics block dictionary to update.
        evaluation: The evaluation result dictionary.
        strategies: List of confidence strategy names.
    """
    stats_block["total"] += 1
    is_match = evaluation.get("match", False)
    if is_match:
        stats_block["matches"] += 1

    confidence_results = evaluation.get("confidence_results", {})
    for strategy in strategies:
        if strategy in confidence_results:
            strategy_agg = stats_block["by_confidence_strategy"][strategy]
            result_data = confidence_results[strategy]

            # Update before/after stats
            for time_point in TIME_POINTS:
                confidence_key = f"{time_point}_confidence"
                confidence = result_data.get(confidence_key)
                if confidence is not None:
                    time_point_agg = strategy_agg[time_point]
                    time_point_agg["total_confidence"] += confidence
                    time_point_agg["valid_count"] += 1
                    if is_match:
                        time_point_agg["correct_confidences"].append(confidence)
                    else:
                        time_point_agg["incorrect_confidences"].append(confidence)

            # Track confidence changes (requires both before and after)
            before_confidence = result_data.get("before_confidence")
            after_confidence = result_data.get("after_confidence")
            if before_confidence is not None and after_confidence is not None:
                change = after_confidence - before_confidence
                strategy_agg["confidence_changes"].append(change)
                if is_match:
                    strategy_agg["correct_confidence_changes"].append(change)
                else:
                    strategy_agg["incorrect_confidence_changes"].append(change)

                if change > 0:
                    strategy_agg["confidence_changes_by_result"]["increased"] += 1
                elif change < 0:
                    strategy_agg["confidence_changes_by_result"]["decreased"] += 1
                else:
                    strategy_agg["confidence_changes_by_result"]["unchanged"] += 1

def _calculate_stat_averages(stats_block: Dict[str, Any], strategies: List[str]) -> None:
    """
    Calculates average statistics from accumulated totals and lists, modifying the block in place.

    Args:
        stats_block: The statistics block dictionary containing totals and lists.
        strategies: List of confidence strategy names.
    """
    total = stats_block["total"]
    matches = stats_block["matches"]
    stats_block["match_rate"] = f"{(matches / total) * 100:.2f}%" if total > 0 else "0.00%"

    # Helper to safely calculate average
    def safe_avg(data_list):
        return sum(data_list) / len(data_list) if data_list else None

    for strategy in strategies:
        strategy_agg = stats_block["by_confidence_strategy"][strategy] # Aggregated totals/counts/lists
        strategy_calc = stats_block["confidence_stats"][strategy] # Place for calculated averages

        # Calculate before/after averages
        for time_point in TIME_POINTS:
            time_point_agg = strategy_agg[time_point]
            time_point_calc = strategy_calc[time_point]

            # Overall averages
            if time_point_agg["valid_count"] > 0:
                time_point_calc["avg_confidence"] = time_point_agg["total_confidence"] / time_point_agg["valid_count"]

            # Averages based on correctness
            time_point_calc["avg_confidence_when_correct"] = safe_avg(time_point_agg["correct_confidences"])
            time_point_calc["avg_confidence_when_incorrect"] = safe_avg(time_point_agg["incorrect_confidences"])

        # Calculate average changes
        strategy_calc["avg_confidence_change"] = safe_avg(strategy_agg["confidence_changes"])
        strategy_calc["avg_confidence_change_when_correct"] = safe_avg(strategy_agg["correct_confidence_changes"])
        strategy_calc["avg_confidence_change_when_incorrect"] = safe_avg(strategy_agg["incorrect_confidence_changes"])

        # Copy the change counts
        strategy_calc["confidence_changes_by_result"] = strategy_agg["confidence_changes_by_result"]

# --- End Helper Functions ---

def save_multiple_evaluations_to_json(evaluations: List[Dict[str, Any]], model_name: str, provider: str) -> str:
    """
    Save multiple evaluations data to a single JSON file, including summary statistics.

    Args:
        evaluations: List of evaluation results
        model_name: The name of the model used
        provider: The API provider (OpenAI or Anthropic)

    Returns:
        The path to the saved JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Identify all confidence strategies used
    all_strategies = set()
    for eval_item in evaluations:
        if "confidence_results" in eval_item:
            all_strategies.update(eval_item["confidence_results"].keys())
    all_strategies = sorted(list(all_strategies)) # Ensure consistent order

    # Initialize statistics blocks
    overall_stats = _initialize_stats_block(all_strategies)
    by_language = {}
    by_difficulty = {}

    # --- Process Evaluations ---
    for eval_item in evaluations:
        example = eval_item.get("example", {})
        lang = example.get("language", "unknown").lower()
        difficulty = example.get("difficulty", "?")

        # Ensure nested stats blocks exist
        if lang not in by_language:
            by_language[lang] = _initialize_stats_block(all_strategies)
            by_language[lang]["by_difficulty"] = {} # Add container for difficulty within language
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = _initialize_stats_block(all_strategies)
        if difficulty not in by_language[lang]["by_difficulty"]:
             by_language[lang]["by_difficulty"][difficulty] = _initialize_stats_block(all_strategies)


        # Update all relevant statistics blocks
        _update_stats_block(overall_stats, eval_item, all_strategies)
        _update_stats_block(by_language[lang], eval_item, all_strategies)
        _update_stats_block(by_difficulty[difficulty], eval_item, all_strategies)
        _update_stats_block(by_language[lang]["by_difficulty"][difficulty], eval_item, all_strategies)

    # --- Calculate Averages for all Blocks ---
    _calculate_stat_averages(overall_stats, all_strategies)
    for lang_stats in by_language.values():
        _calculate_stat_averages(lang_stats, all_strategies)
        for diff_stats in lang_stats.get("by_difficulty", {}).values():
             _calculate_stat_averages(diff_stats, all_strategies)
    for diff_stats in by_difficulty.values():
        _calculate_stat_averages(diff_stats, all_strategies)


    # --- Assemble Final Data Structure ---
    # Clean up the stats blocks by removing intermediate list storage if desired,
    # or keep them for potential future analysis. We'll keep them for now.
    # Extract overall confidence stats for top-level summary.
    final_confidence_stats = overall_stats["confidence_stats"] # This now has nested before/after


    data = {
        "timestamp": timestamp,
        "provider": provider,
        "model": model_name,
        "total_examples": overall_stats["total"],
        "total_matches": overall_stats["matches"],
        "match_rate": overall_stats["match_rate"],
        "confidence_strategies": all_strategies,
        "confidence_stats": final_confidence_stats, # Use calculated stats from overall
        "by_language": by_language,
        "by_difficulty": by_difficulty,
        "evaluations": evaluations # Keep individual evaluations
    }

    # Save to the output/evals directory
    output_dir = "output/evals"
    filename = f"batch_evaluation_{provider}_{model_name.replace('-', '_')}"
    return save_to_json(data, filename, output_dir)

def evaluate_example(client, example: Dict[str, Any], model_name: str, provider: str,
                    confidence_strategies: List[str] = None, superforecast: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single example and return the evaluation results.

    Args:
        client: The API client to use (OpenAI or Anthropic)
        example: The example to evaluate
        model_name: The name of the model to use
        provider: The API provider (OpenAI or Anthropic)
        confidence_strategies: List of confidence strategies to use (default: [DEFAULT_CONFIDENCE_STRATEGY])
        superforecast: Whether to use the 'superforecaster' persona prompt

    Returns:
        A dictionary with the evaluation results
    """
    # Dictionary to hold confidence results
    confidence_results = {}

    # Evaluate "before" confidence for each strategy
    for strategy in confidence_strategies:
        before_confidence_prompt = get_before_confidence_prompt(strategy, example['code'], superforecast)

        # Query the model
        before_confidence_response = query_model(client, before_confidence_prompt, model_name, temperature=0.0)

        # Extract confidence using the appropriate function
        confidence = None
        if before_confidence_response:
            confidence = extract_confidence(before_confidence_response, strategy)

        # Store results (initialize after prompt/response/confidence to None)
        confidence_results[strategy] = {
            "superforecast": superforecast,
            "before_prompt": before_confidence_prompt,
            "before_response": before_confidence_response,
            "before_confidence": confidence,
            "after_prompt": None,
            "after_response": None,
            "after_confidence": None
        }

    # --- Now evaluate the actual example ---
    # Start a new conversation for the evaluation itself
    prompt = EVALUATION_PROMPT_TEMPLATE.format(code=example['code'])
    initial_eval_conversation: StandardConversation = [{"role": "user", "content": prompt}]
    # Get the full updated conversation including the model's first answer
    eval_conversation = query_model_with_history(client, model_name, initial_eval_conversation[:], temperature=0.0)

    # Check if the evaluation API call was successful
    if len(eval_conversation) <= len(initial_eval_conversation):
        print(f"No response received from the API for example: {example.get('language', 'Unknown')} difficulty {example.get('difficulty', '?')}")
        # Can't proceed with 'after' confidence if evaluation failed
        return None

    # Extract the answer
    # The answer is the content of the last message in the conversation
    model_answer_message = eval_conversation[-1]
    full_response_text = model_answer_message.get("content", "")
    extracted_answer = extract_final_answer(full_response_text)
    expected_answer = example.get('verified_answer', example.get('answer', ''))

    # --- Evaluate "after" confidence for each strategy ---
    # Use the *same conversation* where the model just answered
    for strategy in confidence_strategies:
        after_prompt_text = get_after_confidence_prompt_text(strategy, superforecast)
        after_prompt_message: StandardMessage = {"role": "user", "content": after_prompt_text}

        # Create a *copy* of the conversation to append to, for this specific strategy query
        current_conversation_for_after_query = eval_conversation[:]
        current_conversation_for_after_query.append(after_prompt_message)

        # Pass the updated conversation history
        final_conversation = query_model_with_history(client, model_name, current_conversation_for_after_query, temperature=0.0)

        # Extract confidence using the appropriate function
        after_confidence = None
        after_confidence_response_text = None
        # Check if the API call was successful and extract the last message content
        if len(final_conversation) > len(current_conversation_for_after_query) and final_conversation[-1]["role"] == "assistant":
            after_confidence_response_text = final_conversation[-1].get("content", "")
            if after_confidence_response_text:
                after_confidence = extract_confidence(after_confidence_response_text, strategy)

        # Store results
        confidence_results[strategy]["after_prompt"] = after_prompt_text # Store the text, not the message object
        confidence_results[strategy]["after_response"] = after_confidence_response_text
        confidence_results[strategy]["after_confidence"] = after_confidence

    # Create evaluation result with combined confidence data
    result = {
        "example": example,
        "prompt": prompt,
        "full_response": full_response_text, # The initial response to the eval prompt
        "extracted_answer": extracted_answer,
        "expected_answer": expected_answer,
        "match": extracted_answer.strip() == expected_answer.strip(),
        "confidence_strategies": confidence_strategies,
        "superforecast": superforecast,
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
    parser.add_argument('--confidence-strategies', type=str,
                        help=f'Comma-separated list of confidence strategies to use. Available: {", ".join(CONFIDENCE_STRATEGY_DEFINITIONS.keys())}')
    parser.add_argument('--superforecast', action='store_true',
                        help='Use the superforecast persona prompt')
    parser.add_argument('--list-confidence-strategies', action='store_true',
                        help='List available confidence strategies and exit')

    args = parser.parse_args()

    # List confidence strategies if requested
    if args.list_confidence_strategies:
        print("Available confidence estimation strategies:")
        for strategy, details in CONFIDENCE_STRATEGY_DEFINITIONS.items():
            print(f"  - {strategy}: {details['description']}")
        return

    # Create the appropriate client based on provider
    client = create_client(args.provider)
    if not client:
        return

    if args.list_models:
        list_models(client)
        return

    # Check if model is provided when not listing models
    if not args.model:
        print("Error: --model is required when not using --list-models")
        parser.print_help()
        return

    # Parse confidence strategies
    if not args.confidence_strategies:
        confidence_strategies = list(CONFIDENCE_STRATEGY_DEFINITIONS.keys())
    else:
        confidence_strategies = [s.strip() for s in args.confidence_strategies.split(",")]
        invalid_strategies = [s for s in confidence_strategies if s not in CONFIDENCE_STRATEGY_DEFINITIONS]
        if invalid_strategies:
            all_strategies = list(CONFIDENCE_STRATEGY_DEFINITIONS.keys())
            print(f"Warning: Unknown confidence strategies: {', '.join(invalid_strategies)}")
            print(f"Available strategies: {', '.join(all_strategies)}")
            return

    # Load examples from JSON file
    examples = load_examples_from_json(args.input_file)
    if not examples:
        return

    # Filter by language if specified
    filtered_examples = examples
    if args.language:
        lang_lower = args.language.lower()
        matching_examples = [ex for ex in examples if ex.get('language', '').lower() == lang_lower]
        if not matching_examples:
            print(f"No examples found for language: {args.language}")
            print(f"Available languages: {', '.join(set(ex.get('language', '') for ex in examples))}")
            return
        filtered_examples = matching_examples

    examples_to_run = []
    if args.run_all:
        examples_to_run = filtered_examples
        if not examples_to_run:
            print("No examples selected to run.")
            return
        print(f"Preparing to evaluate {len(examples_to_run)} examples...")
    else:
        # Select a single random example
        selected_example = select_random_example(filtered_examples)
        if not selected_example:
            print("No example selected to run.")
            return
        examples_to_run = [selected_example]
        print(f"Selected 1 random example to evaluate.")

    # --- Handle Dry Run ---
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model}")
        print(f"Confidence strategies: {', '.join(confidence_strategies)}")

        if not examples_to_run:
            print("\nNo examples selected for dry run.")
            return

        # Show details for the first example
        first_example = examples_to_run[0]
        print(f"\nExample (1 of {len(examples_to_run)}):")
        print(f"Language: {first_example['language']}")
        print(f"Difficulty: {first_example.get('difficulty', '?')}")
        expected_answer = first_example.get('verified_answer', first_example.get('answer', ''))
        print(f"Expected Answer: {expected_answer}")

        print("\nPrompts that would be sent:")
        # Show all confidence prompts
        for i, strategy in enumerate(confidence_strategies):
            print(f"\n=== Confidence Prompt {i+1}: {strategy} (Before Answer) ===")
            print("=" * 40)
            print(get_before_confidence_prompt(strategy, first_example['code'], args.superforecast))
            print("=" * 40)

        # Show evaluation prompt
        print("\n=== Evaluation Prompt ===")
        print("=" * 40)
        print(EVALUATION_PROMPT_TEMPLATE.format(code=first_example['code']))
        print("=" * 40)

        # Show 'after' confidence prompts
        for i, strategy in enumerate(confidence_strategies):
            print(f"\n=== Confidence Prompt {i+1}: {strategy} (After Answer) ===")
            print("=" * 40)
            # Construct the text dynamically for dry run display
            print(get_after_confidence_prompt_text(strategy, args.superforecast))
            print("=" * 40)

        print(f"\nWould evaluate {len(examples_to_run)} example(s) in total.")
        print("No API calls will be made. Exiting.")
        return

    # --- Execute Evaluation Run (Unified for single or multiple) ---
    print(f"Running evaluation on {len(examples_to_run)} example(s). This may take a while...")
    print(f"Using confidence strategies: {', '.join(confidence_strategies)}")

    evaluations = []
    # Summary tracking (can be derived later, but useful for progress)
    correct_count = 0

    for i, example in enumerate(examples_to_run, 1):
        print(f"\n[{i}/{len(examples_to_run)}] Evaluating {example['language']} example (difficulty {example.get('difficulty', '?')})...", end="", flush=True)

        evaluation = evaluate_example(client, example, args.model, args.provider, confidence_strategies, args.superforecast)
        if evaluation:
            evaluations.append(evaluation)

            confidence_display = ""
            for strategy in confidence_strategies:
                before_confidence = evaluation["confidence_results"][strategy].get("before_confidence")
                after_confidence = evaluation["confidence_results"][strategy].get("after_confidence")
                before_display = f" (conf before {strategy}: {before_confidence:.2f})" if before_confidence is not None else ""
                after_display = f" (conf after {strategy}: {after_confidence:.2f})" if after_confidence is not None else ""
                confidence_display += f"{before_display}{after_display}"

            if evaluation["match"]:
                result = "✓ CORRECT"
                correct_count += 1
            else:
                result = "✗ WRONG"

            # Print per-example result immediately
            print(f" {result}{confidence_display}")
            print(f"  Expected: {evaluation['expected_answer']}")
            print(f"  Model's answer: {evaluation['extracted_answer']}")

            # Show all confidence values if there are multiple strategies
            if len(confidence_strategies) > 1:
                for strategy in confidence_strategies:
                    before_conf = evaluation["confidence_results"][strategy].get("before_confidence")
                    after_conf = evaluation["confidence_results"][strategy].get("after_confidence")
                    before_str = f"{before_conf:.2f}" if before_conf is not None else "N/A"
                    after_str = f"{after_conf:.2f}" if after_conf is not None else "N/A"
                    print(f"  {strategy} confidence: before={before_str}, after={after_str}")
        else:
            print(" ERROR (evaluation failed)") # More specific error message

    # --- Save Results and Print Summary ---
    if not evaluations:
        print("\nNo evaluations were successfully completed.")
        return

    # Save all evaluations (1 or more) to a single file with aggregated stats
    filename = save_multiple_evaluations_to_json(evaluations, args.model, args.provider)

    # Load the summary stats back from the saved file (or recalculate)
    # For simplicity, we'll just recalculate the top-level summary here
    total_evaluated = len(evaluations)
    final_match_rate = (correct_count / total_evaluated) * 100 if total_evaluated else 0

    print("\n=== Evaluation Summary ===")
    print(f"Total examples evaluated: {total_evaluated}")
    print(f"Correct answers: {correct_count} ({final_match_rate:.2f}%)")

    # Calculate and print average confidences from the collected evaluations
    # This duplicates some logic from save_multiple_evaluations_to_json but keeps main cleaner
    avg_confidences = {}
    for strategy in confidence_strategies:
        before_total = 0.0
        before_count = 0
        after_total = 0.0
        after_count = 0
        for ev in evaluations:
            if strategy in ev["confidence_results"]:
                before_c = ev["confidence_results"][strategy].get("before_confidence")
                after_c = ev["confidence_results"][strategy].get("after_confidence")
                if before_c is not None:
                    before_total += before_c
                    before_count += 1
                if after_c is not None:
                    after_total += after_c
                    after_count += 1

        avg_before = f"{(before_total / before_count):.2f}" if before_count > 0 else "N/A"
        avg_after = f"{(after_total / after_count):.2f}" if after_count > 0 else "N/A"
        print(f"{strategy.capitalize()} average confidence: Before={avg_before} (n={before_count}), After={avg_after} (n={after_count})")

    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to generate coding examples for software engineering interviews using AI.

This script:
1. Takes a prompt and sends it to either OpenAI or Anthropic API
2. Parses the response to extract code examples
3. Saves the raw response and parsed examples to a timestamped JSON file
"""

import os
import json
import argparse
import re
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import shared utilities
from ai_utils import (
    create_client,
    query_model,
    query_model_with_history,
    list_models,
    save_to_json
)

DEFAULT_PROMPT = """
For the purposes of a software engineering job interview, please generate a few small examples of self-contained, deterministic programs, for which it would be reasonable to ask a software engineering candidate to work out (using only pencil and paper) what the program will output, or what the function will return. Please give some examples in each of C, Python, and Javascript. They should be at variable levels of difficulty, graded from 1 to 10. Any competent intern should be able to solve problems graded "1" just by thinking, without writing anything down. Problems graded "10" should be possible for an expert programmer to solve, using pencil and paper and taking time to think about it.

Please provide the programs in the following format, with a blank line between consecutive examples:

LANGUAGE: [the language, C or Python or Javascript]
DIFFICULTY: [a number from 1 to 10]
CODE:
[write the code as a function, which takes no arguments, and which returns a final answer that the interviewee should figure out.]
ANSWER: [your own best guess at the answer; it does not need to be correct]
"""

def normalize_response(response_text: str) -> str:
    """
    Normalize the response text to handle quirks from different models.

    Args:
        response_text: The raw response text from the AI

    Returns:
        Normalized response text
    """
    # Replace markdown code block markers if present
    response_text = re.sub(r'```(?:python|c|javascript|js|)\n', '', response_text)
    response_text = re.sub(r'```', '', response_text)

    # Ensure consistent newline after ANSWER:
    response_text = re.sub(r'(ANSWER:[^\n]+)(?!\n)', r'\1\n', response_text)

    return response_text

def parse_examples(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse the examples from the response text.

    Args:
        response_text: The raw text response from the AI

    Returns:
        A list of dictionaries, each representing a parsed example
    """
    # First normalize the response
    normalized_text = normalize_response(response_text)

    examples = []

    # Define a regex pattern to match examples
    pattern = r"LANGUAGE: *([^\n]+)\s*DIFFICULTY: *(\d+)\s*CODE:\s*(.*?)ANSWER: *([^\n]+)"

    # Find all matches in the response
    matches = re.finditer(pattern, normalized_text, re.DOTALL)

    for match in matches:
        language = match.group(1).strip()

        try:
            difficulty = int(match.group(2).strip())
        except ValueError:
            # Handle non-integer difficulty
            difficulty_str = match.group(2).strip()
            print(f"Warning: Non-integer difficulty value: '{difficulty_str}', setting to 0")
            difficulty = 0

        code = match.group(3).strip()
        answer = match.group(4).strip()

        examples.append({
            "language": language,
            "difficulty": difficulty,
            "code": code,
            "answer": answer
        })

    return examples

def save_examples_to_json(
    prompt: str,
    raw_responses: List[str],
    all_examples: List[Dict[str, Any]],
    model_name: str,
    provider: str,
    temperature: Optional[float],
    target_examples: int,
    max_queries: int,
    num_queries_made: int
) -> str:
    """
    Save the prompt, response, and parsed examples to a JSON file.

    Args:
        prompt: The original prompt
        raw_responses: List of raw responses from each API query
        all_examples: The aggregated list of parsed examples
        model_name: The name of the model used
        provider: The API provider (OpenAI or Anthropic)
        temperature: The temperature setting used for generation
        target_examples: The target number of examples requested
        max_queries: The maximum number of queries allowed
        num_queries_made: The actual number of queries performed

    Returns:
        The path to the saved JSON file
    """
    data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "provider": provider,
        "model": model_name,
        "generation_parameters": {
            "prompt": prompt,
            "temperature": temperature,
            "target_examples": target_examples,
            "max_queries": max_queries,
            "num_queries_made": num_queries_made
        },
        "raw_responses": raw_responses,
        "examples": all_examples,
        "final_example_count": len(all_examples)
    }

    # Save to the output/examples directory
    output_dir = "output/examples"
    return save_to_json(data, f"code_examples_{provider}_{model_name.replace('-', '_')}", output_dir)

def main():
    """Main function to parse arguments and execute the query."""
    parser = argparse.ArgumentParser(description='Generate code examples using AI APIs')
    parser.add_argument('--provider', choices=['openai', 'anthropic'], required=True,
                        help='API provider to use (openai or anthropic)')
    parser.add_argument('--model', type=str,
                        help='Model name to use (e.g., gpt-4 for OpenAI, claude-3-opus-20240229 for Anthropic)')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT,
                        help='Custom prompt to use (defaults to the predefined prompt)')
    parser.add_argument('--target-examples', type=int, default=10,
                        help='Target number of examples to generate (default: 10)')
    parser.add_argument('--max-queries', type=int, default=5,
                        help='Maximum number of API queries allowed (default: 5)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature for generation (e.g., 0.7, 1.0). Default: 0.8')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models from the specified provider and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be sent to the API without actually making the call')

    args = parser.parse_args()

    # Create the appropriate client based on provider
    client = create_client(args.provider)
    if not client:
        return

    if args.list_models:
        list_models(client)
        return

    # Validate temperature range if needed (e.g., 0.0 to 2.0)
    if not (0.0 <= args.temperature <= 2.0):
        print(f"Warning: Temperature {args.temperature} is outside the typical range [0.0, 2.0].")

    # Check if model is provided when not listing models
    if not args.model:
        print("Error: --model is required when not using --list-models")
        parser.print_help()
        return

    # If dry run, show what would be sent and exit
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model}")
        print("\nPrompt that would be sent:")
        print("=" * 40)
        print(args.prompt)
        print("=" * 40)
        print("\nNo API call will be made. Exiting.")
        return

    # --- Iterative Generation Loop ---
    all_examples = []
    raw_responses_list = []
    query_count = 0

    print(f"Attempting to generate {args.target_examples} examples (max queries: {args.max_queries}, temp: {args.temperature})...")

    while len(all_examples) < args.target_examples and query_count < args.max_queries:
        query_count += 1
        print(f"\n--- Query {query_count}/{args.max_queries} --- (Current examples: {len(all_examples)}/{args.target_examples})")

        # Query the model with specified temperature
        response = query_model(
            client,
            args.prompt,
            args.model,
            temperature=args.temperature
        )

        if not response:
            print("No response received from the API for this query.")
            # Optionally break or continue depending on desired behavior on failure
            continue # Let's continue to allow other queries

        raw_responses_list.append(response)

        # Parse examples from the response
        new_examples = parse_examples(response)

        if not new_examples:
            print("Warning: No examples parsed from this query's response.")
        else:
            print(f"Parsed {len(new_examples)} new examples from this query.")
            all_examples.extend(new_examples)

        # Optional: Add a small delay between queries if needed
        # import time
        # time.sleep(1)

    print(f"\nFinished generation after {query_count} queries.")

    # --- Final Check and Save ---
    if not all_examples:
        print("Warning: No examples were generated or parsed successfully.")
        if raw_responses_list:
             print("Saving the raw responses anyway.")
        else:
             return # Nothing to save

    # Save to JSON
    filename = save_examples_to_json(
        args.prompt,
        raw_responses_list,
        all_examples,
        args.model,
        args.provider,
        args.temperature,
        args.target_examples,
        args.max_queries,
        query_count
    )

    print(f"\nTotal generated examples: {len(all_examples)}")
    if all_examples:
        languages = set(example['language'] for example in all_examples)
        difficulties = [example['difficulty'] for example in all_examples]
        print(f"Languages represented: {', '.join(sorted(list(languages)))}")
        try:
            print(f"Difficulty range: {min(difficulties)} to {max(difficulties)}")
        except ValueError: # Handle case where difficulties list might be empty if parsing failed somehow
            print("Could not determine difficulty range.")

if __name__ == "__main__":
    main()

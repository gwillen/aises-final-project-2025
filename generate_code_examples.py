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
from typing import Dict, List, Optional, Any

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

def save_examples_to_json(prompt: str, response: str, examples: List[Dict[str, Any]], model_name: str, provider: str) -> str:
    """
    Save the prompt, response, and parsed examples to a JSON file.

    Args:
        prompt: The original prompt
        response: The raw response from the AI
        examples: The parsed examples
        model_name: The name of the model used
        provider: The API provider (OpenAI or Anthropic)

    Returns:
        The path to the saved JSON file
    """
    data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "provider": provider,
        "model": model_name,
        "prompt": prompt,
        "response": response,
        "examples": examples,
        "example_count": len(examples)
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
    parser.add_argument('--list-models', action='store_true',
                        help='List available models from the specified provider and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be sent to the API without actually making the call')

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

    # Query the model
    if args.provider == 'openai':
        response = query_openai(client, args.prompt, args.model)
    else:  # anthropic
        response = query_anthropic(client, args.prompt, args.model)

    if not response:
        print("No response received from the API")
        return

    # Parse examples from the response
    examples = parse_examples(response)

    if not examples:
        print("Warning: No examples were found in the response. The parsing may have failed.")
        print("Check if the response follows the expected format:")
        print("LANGUAGE: ...\nDIFFICULTY: ...\nCODE: ...\nANSWER: ...")
        save_anyway = input("Save the response anyway? (y/n): ")
        if save_anyway.lower() != 'y':
            return

    # Save to JSON
    filename = save_examples_to_json(args.prompt, response, examples, args.model, args.provider)

    print(f"Generated {len(examples)} examples")
    if examples:
        languages = set(example['language'] for example in examples)
        difficulties = [example['difficulty'] for example in examples]
        print(f"Languages: {', '.join(languages)}")
        print(f"Difficulty range: {min(difficulties)} to {max(difficulties)}")

if __name__ == "__main__":
    main()

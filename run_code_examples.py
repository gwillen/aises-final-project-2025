#!/usr/bin/env python3
"""
Module to run code examples in different programming languages and verify the results.

Supported languages:
- JavaScript (using Node.js)
- Python
- C (compiled with GCC)
"""

import os
import json
import subprocess
import tempfile
import argparse
import sys
import shutil
from typing import Dict, List, Any, Tuple, Optional
import re

def ensure_temp_directory() -> str:
    """Create and return a temporary directory for code execution."""
    temp_dir = os.path.join(tempfile.gettempdir(), "code_examples_runner")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def write_temp_file(temp_dir: str, filename: str, content: str) -> str:
    """Write content to a temporary file and return the full path."""
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path

def run_javascript_code(code: str, temp_dir: str) -> Tuple[bool, str, str]:
    """
    Run JavaScript code using Node.js.

    Args:
        code: The JavaScript code to run
        temp_dir: Directory to store temporary files

    Returns:
        Tuple of (success, output, error)
    """
    # Check if Node.js is installed
    if not shutil.which("node"):
        return False, "", "Node.js is not installed or not in PATH"

    # Wrap the code in a function and add a call to it
    wrapped_code = code

    # Check if it already has a function wrapper
    if not re.search(r"function\s+\w+\s*\(\s*\)", wrapped_code):
        # If no function defined, wrap it
        wrapped_code = f"""
function main() {{
{code}
}}
console.log(main());
"""
    else:
        # If it already has a function, add a call to it
        # Extract the function name
        function_match = re.search(r"function\s+(\w+)\s*\(\s*\)", wrapped_code)
        if function_match:
            function_name = function_match.group(1)
            wrapped_code += f"\nconsole.log({function_name}());"

    # Write code to a temporary file
    file_path = write_temp_file(temp_dir, "example.js", wrapped_code)

    try:
        # Run the JavaScript code with Node.js
        process = subprocess.run(
            ["node", file_path],
            capture_output=True,
            text=True,
            timeout=5  # 5 second timeout
        )
        return process.returncode == 0, process.stdout.strip(), process.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as e:
        return False, "", str(e)

def run_python_code(code: str, temp_dir: str) -> Tuple[bool, str, str]:
    """
    Run Python code.

    Args:
        code: The Python code to run
        temp_dir: Directory to store temporary files

    Returns:
        Tuple of (success, output, error)
    """
    # Check if Python is installed
    if not shutil.which("python3"):
        return False, "", "Python3 is not installed or not in PATH"

    # Wrap the code in a function and add a call to it if it doesn't have one
    wrapped_code = code

    # Check if it already has a function wrapper
    if not re.search(r"def\s+\w+\s*\(\s*\)", wrapped_code):
        # If no function defined, wrap it
        wrapped_code = f"""
def main():
{textwrap.indent(code, '    ')}

print(main())
"""
    else:
        # If it already has a function, add a call to it
        # Extract the function name
        function_match = re.search(r"def\s+(\w+)\s*\(\s*\)", wrapped_code)
        if function_match:
            function_name = function_match.group(1)
            wrapped_code += f"\nprint({function_name}())"

    # Write code to a temporary file
    file_path = write_temp_file(temp_dir, "example.py", wrapped_code)

    try:
        # Run the Python code
        process = subprocess.run(
            ["python3", file_path],
            capture_output=True,
            text=True,
            timeout=5  # 5 second timeout
        )
        return process.returncode == 0, process.stdout.strip(), process.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as e:
        return False, "", str(e)

def run_c_code(code: str, temp_dir: str) -> Tuple[bool, str, str]:
    """
    Compile and run C code using GCC.

    Args:
        code: The C code to run
        temp_dir: Directory to store temporary files

    Returns:
        Tuple of (success, output, error)
    """
    # Check if GCC is installed
    if not shutil.which("gcc"):
        return False, "", "GCC is not installed or not in PATH"

    # Write code to a temporary file
    file_path = write_temp_file(temp_dir, "example.c", code)
    executable_path = os.path.join(temp_dir, "example")

    try:
        # Compile the C code
        compile_process = subprocess.run(
            ["gcc", file_path, "-o", executable_path],
            capture_output=True,
            text=True,
            timeout=5  # 5 second timeout for compilation
        )

        if compile_process.returncode != 0:
            return False, "", f"Compilation error: {compile_process.stderr}"

        # Run the compiled executable
        run_process = subprocess.run(
            [executable_path],
            capture_output=True,
            text=True,
            timeout=5  # 5 second timeout for execution
        )

        return run_process.returncode == 0, run_process.stdout.strip(), run_process.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Execution timed out"
    except Exception as e:
        return False, "", str(e)

def run_code(code: str, language: str) -> Tuple[bool, str, str]:
    """
    Run code in the specified language.

    Args:
        code: The code to run
        language: The programming language (JavaScript, Python, C)

    Returns:
        Tuple of (success, output, error)
    """
    temp_dir = ensure_temp_directory()

    language = language.strip().lower()
    if language in ["javascript", "js"]:
        return run_javascript_code(code, temp_dir)
    elif language in ["python", "py"]:
        return run_python_code(code, temp_dir)
    elif language in ["c"]:
        return run_c_code(code, temp_dir)
    else:
        return False, "", f"Unsupported language: {language}"

def verify_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify an example by running the code and comparing the output with the expected answer.

    Args:
        example: The example to verify

    Returns:
        The example with verification results
    """
    language = example.get("language", "").strip()
    code = example.get("code", "")
    expected_answer = example.get("answer", "")

    # Run the code
    success, output, error = run_code(code, language)

    # Add verification results to the example
    result = example.copy()
    result["verification"] = {
        "success": success,
        "output": output,
        "error": error,
        "matches_expected": success and output.strip() == expected_answer.strip()
    }

    return result

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

def save_verification_results(examples: List[Dict[str, Any]], original_file: str) -> str:
    """
    Save verification results to a JSON file.

    Args:
        examples: The examples with verification results
        original_file: The original file path

    Returns:
        The path to the saved file
    """
    # Create output directory
    output_dir = "output/verified"
    os.makedirs(output_dir, exist_ok=True)

    # Create filename based on original file
    base_filename = os.path.basename(original_file)
    output_file = os.path.join(output_dir, f"verified_{base_filename}")

    # Load original data to preserve metadata
    with open(original_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Update examples with verification results
    data['examples'] = examples
    data['verification_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_file

def print_verification_summary(examples: List[Dict[str, Any]]) -> None:
    """
    Print a summary of verification results.

    Args:
        examples: The examples with verification results
    """
    total = len(examples)
    successful_runs = sum(1 for ex in examples if ex.get("verification", {}).get("success", False))
    matching_answers = sum(1 for ex in examples if ex.get("verification", {}).get("matches_expected", False))

    print(f"\nVerification Summary:")
    print(f"Total examples: {total}")
    print(f"Successfully ran: {successful_runs} ({successful_runs/total*100:.1f}%)")
    print(f"Matching expected answers: {matching_answers} ({matching_answers/total*100:.1f}%)")

    # Group by language
    by_language = {}
    for ex in examples:
        lang = ex.get("language", "Unknown").lower()
        if lang not in by_language:
            by_language[lang] = {"total": 0, "success": 0, "match": 0}

        by_language[lang]["total"] += 1
        if ex.get("verification", {}).get("success", False):
            by_language[lang]["success"] += 1
        if ex.get("verification", {}).get("matches_expected", False):
            by_language[lang]["match"] += 1

    print("\nBy Language:")
    for lang, stats in by_language.items():
        print(f"  {lang.capitalize()}: {stats['match']}/{stats['total']} correct answers ({stats['match']/stats['total']*100:.1f}%)")

def main():
    """Main function to parse arguments and execute verification."""
    parser = argparse.ArgumentParser(description='Run and verify code examples')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the JSON file containing code examples')
    parser.add_argument('--language', type=str,
                        help='Filter examples by language (e.g., Python, C, Javascript)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output for each example')

    args = parser.parse_args()

    # Load examples from JSON file
    examples = load_examples_from_json(args.input_file)
    if not examples:
        return

    # Filter by language if specified
    if args.language:
        language = args.language.lower()
        filtered_examples = [ex for ex in examples if ex.get("language", "").lower() == language]
        if not filtered_examples:
            print(f"No examples found for language: {args.language}")
            return
        examples = filtered_examples

    print(f"Verifying {len(examples)} examples...")

    # Verify each example
    verified_examples = []
    for i, example in enumerate(examples, 1):
        language = example.get("language", "Unknown")
        difficulty = example.get("difficulty", "?")

        print(f"[{i}/{len(examples)}] Verifying {language} example (difficulty {difficulty})...", end="")
        verified = verify_example(example)
        verified_examples.append(verified)

        if verified["verification"]["success"]:
            if verified["verification"]["matches_expected"]:
                result = "✓ CORRECT"
            else:
                result = "✗ WRONG ANSWER"
        else:
            result = "✗ ERROR"

        print(f" {result}")

        if args.verbose or not verified["verification"]["success"]:
            if verified["verification"]["error"]:
                print(f"  Error: {verified['verification']['error']}")
            print(f"  Expected: {verified['answer']}")
            print(f"  Actual: {verified['verification']['output']}")
            print()

    # Save verification results
    output_file = save_verification_results(verified_examples, args.input_file)
    print(f"\nVerification results saved to {output_file}")

    # Print summary
    print_verification_summary(verified_examples)

if __name__ == "__main__":
    import textwrap
    from datetime import datetime
    main()

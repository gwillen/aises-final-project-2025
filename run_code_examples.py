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

# Global configuration
debug_mode = False  # Will be set by command line argument

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

    # Fix invalid function names (e.g., 'function')
    fixed_code = re.sub(r'function\s+function\s*\(', 'function validFunc(', code)

    # Handle edge cases where 'function' is used as variable name
    fixed_code = re.sub(r'(let|var|const)\s+function\s*=', r'\1 validFunc =', fixed_code)

    # Wrap the code in a function and add a call to it
    wrapped_code = fixed_code

    # Check if it already has a function wrapper
    if not re.search(r"function\s+\w+\s*\(\s*\)", wrapped_code):
        # If no function defined, wrap it
        wrapped_code = f"""
function main() {{
{fixed_code}
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

    # Wrap the code with a main function if it doesn't have one
    if "main(" not in code:
        try:
            # Try to extract the function name and return type from the code
            # This regex matches common C function declarations like "int function()" or "double calc_result(void)"
            func_match = re.search(r'(int|float|double|long|char\s*\*|void|unsigned\s+\w+|\w+)\s+(\w+)\s*\([^)]*\)\s*{', code)

            if func_match:
                return_type = func_match.group(1)
                function_name = func_match.group(2)

                if 'debug_mode' in globals() and debug_mode:
                    print(f"[DEBUG] Found function: {function_name} with return type: {return_type}")
            else:
                # Fallback to a simpler regex if the complex one fails
                simple_match = re.search(r'(\w+)\s*\(\s*\)', code)
                function_name = simple_match.group(1) if simple_match else "function"
                return_type = "int"  # Default to int if we can't determine

                if 'debug_mode' in globals() and debug_mode:
                    print(f"[DEBUG] Using fallback detection. Function: {function_name} with default return type: {return_type}")

            # Add stdio.h include and main function that calls the example function
            # Use appropriate printf format based on return type
            if "int" in return_type or "long" in return_type or "unsigned" in return_type:
                printf_format = "%d"
            elif "float" in return_type or "double" in return_type:
                printf_format = "%f"
            elif "char" in return_type and "*" in return_type:
                printf_format = "%s"
            else:
                printf_format = "%d"  # Default to int format

            wrapped_code = f"""
#include <stdio.h>

{code}

int main() {{
    {return_type} result = {function_name}();
    printf("{printf_format}\\n", result);
    return 0;
}}
"""
        except Exception as e:
            # If anything goes wrong with the regex, use a default wrapper
            wrapped_code = f"""
#include <stdio.h>

{code}

int main() {{
    // Default wrapper when function detection fails
    int result = function();
    printf("%d\\n", result);
    return 0;
}}
"""
            if 'debug_mode' in globals() and debug_mode:
                print(f"[DEBUG] Error in function detection: {str(e)}")
                print("[DEBUG] Using default function wrapper")
    else:
        wrapped_code = code
        if 'debug_mode' in globals() and debug_mode:
            print("[DEBUG] Code already contains a main function")

    # Write code to a temporary file
    file_path = write_temp_file(temp_dir, "example.c", wrapped_code)
    executable_path = os.path.join(temp_dir, "example")

    if 'debug_mode' in globals() and debug_mode:
        print("[DEBUG] Generated C code:")
        print("-" * 40)
        print(wrapped_code)
        print("-" * 40)

    try:
        # Compile the C code
        compile_process = subprocess.run(
            ["gcc", file_path, "-o", executable_path],
            capture_output=True,
            text=True,
            timeout=5  # 5 second timeout for compilation
        )

        if compile_process.returncode != 0:
            if 'debug_mode' in globals() and debug_mode:
                print(f"[DEBUG] Compilation failed with error:\n{compile_process.stderr}")
            return False, "", f"Compilation error: {compile_process.stderr}"

        # Run the compiled executable
        run_process = subprocess.run(
            [executable_path],
            capture_output=True,
            text=True,
            timeout=5  # 5 second timeout for execution
        )

        if 'debug_mode' in globals() and debug_mode:
            print(f"[DEBUG] Execution result: {run_process.stdout.strip()}")
            if run_process.stderr:
                print(f"[DEBUG] Execution stderr: {run_process.stderr}")

        return run_process.returncode == 0, run_process.stdout.strip(), run_process.stderr.strip()
    except subprocess.TimeoutExpired:
        if 'debug_mode' in globals() and debug_mode:
            print("[DEBUG] Execution timed out")
        return False, "", "Execution timed out"
    except Exception as e:
        if 'debug_mode' in globals() and debug_mode:
            print(f"[DEBUG] Exception during execution: {str(e)}")
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
    Verify an example by running the code and getting the actual result.

    Args:
        example: The example to verify

    Returns:
        The example with verified answer
    """
    language = example.get("language", "").strip()
    code = example.get("code", "")

    # If example was previously verified, store the previous verified answer
    previous_verified_answer = example.get("verified_answer", None)
    original_answer = example.get("answer", "")

    # Run the code
    success, output, error = run_code(code, language)

    # Create a new example with the verified answer
    result = example.copy()

    # Remove the original answer and add the verified answer
    if "answer" in result:
        del result["answer"]

    result["verified_answer"] = output.strip() if success else f"ERROR: {error}"
    result["success"] = success

    # If there was a previous verified answer, store it for comparison
    if previous_verified_answer is not None:
        result["previous_verified_answer"] = previous_verified_answer
        result["answer_changed"] = previous_verified_answer != result["verified_answer"]

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
        examples: The examples with verified answers
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
        examples: The examples with verified answers
    """
    total = len(examples)

    if total == 0:
        print("No examples to verify.")
        return

    successful_runs = sum(1 for ex in examples if ex.get("success", False))

    # Check for examples that changed from previous verification
    changed_answers = sum(1 for ex in examples if ex.get("answer_changed", False))

    print(f"\nVerification Summary:")
    print(f"Total examples: {total}")
    print(f"Successfully ran: {successful_runs} ({successful_runs/total*100:.1f}%)")

    if changed_answers > 0:
        print(f"Changed answers: {changed_answers}")

    # Group by language
    by_language = {}
    for ex in examples:
        lang = ex.get("language", "Unknown").lower()
        if lang not in by_language:
            by_language[lang] = {"total": 0, "success": 0, "changed": 0}

        by_language[lang]["total"] += 1
        if ex.get("success", False):
            by_language[lang]["success"] += 1
        if ex.get("answer_changed", False):
            by_language[lang]["changed"] += 1

    print("\nBy Language:")
    for lang, stats in by_language.items():
        print(f"  {lang.capitalize()}: {stats['success']}/{stats['total']} successful runs ({stats['success']/stats['total']*100:.1f}%)")
        if stats['changed'] > 0:
            print(f"    Changed answers: {stats['changed']}")

def main():
    """Main function to parse arguments and execute verification."""
    parser = argparse.ArgumentParser(description='Run and verify code examples')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the JSON file containing code examples')
    parser.add_argument('--language', type=str,
                        help='Filter examples by language (e.g., Python, C, Javascript)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output for each example')
    parser.add_argument('--debug', action='store_true',
                        help='Show additional debug information for code execution')
    parser.add_argument('--skip-verified', action='store_true',
                        help='Skip examples that already have a verified_answer')

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

    # Filter out already verified examples if requested
    if args.skip_verified:
        original_count = len(examples)
        examples = [ex for ex in examples if "verified_answer" not in ex]
        skipped_count = original_count - len(examples)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} already verified examples")

    if not examples:
        print("No examples to verify.")
        return

    print(f"Verifying {len(examples)} examples...")

    # Create debug logger if requested
    global debug_mode
    debug_mode = args.debug

    # Verify each example
    verified_examples = []
    for i, example in enumerate(examples, 1):
        language = example.get("language", "Unknown")
        difficulty = example.get("difficulty", "?")
        code = example.get("code", "")

        # Check for existing answer, either verified or original
        original_answer = example.get("answer", "")
        previous_verified = example.get("verified_answer", None)

        print(f"[{i}/{len(examples)}] Verifying {language} example (difficulty {difficulty})...", end="")

        # Display code in verbose mode
        if args.verbose:
            print("\n")
            print("="*40)
            print(f"CODE ({language}):")
            print("-"*40)
            print(code)
            print("="*40)
            print("Executing...", end="")

        verified = verify_example(example)
        verified_examples.append(verified)

        if verified["success"]:
            result = "✓ SUCCESS"
        else:
            result = "✗ ERROR"

        print(f" {result}")

        if args.verbose or not verified["success"]:
            if not verified["success"]:
                print(f"  Error: {verified['verified_answer']}")

            if previous_verified is not None:
                print(f"  Previous verified answer: {previous_verified}")
                if verified.get("answer_changed", False):
                    print(f"  NOTICE: Answer has changed from previous verification!")
            elif original_answer:
                print(f"  Original answer: {original_answer}")

            print(f"  Actual output: {verified['verified_answer'] if verified['success'] else 'ERROR'}")
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

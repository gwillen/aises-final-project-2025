#!/usr/bin/env python3
"""
Test script to validate the regex pattern used for parsing examples.
"""

import re
from typing import Dict, List, Any

# Sample AI response with different formatting variations
sample_response = """
Here are some examples:

LANGUAGE: Python
DIFFICULTY: 1
CODE:
def function():
    x = 10
    return x + 5
ANSWER: 15

LANGUAGE: C
DIFFICULTY: 3
CODE:
int function() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for(int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    return sum;
}
ANSWER: 15

LANGUAGE:   Javascript
DIFFICULTY:   7
CODE:
function calculate() {
    let result = 0;
    for (let i = 1; i <= 10; i++) {
        if (i % 2 === 0) {
            result += i*i;
        } else {
            result -= i;
        }
    }
    return result;
}
ANSWER:   54
"""

def parse_examples(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse the examples from the response text.

    Args:
        response_text: The raw text response from the AI

    Returns:
        A list of dictionaries, each representing a parsed example
    """
    examples = []

    # Define a regex pattern to match examples
    pattern = r"LANGUAGE: *([^\n]+)\s*DIFFICULTY: *(\d+)\s*CODE:\s*(.*?)ANSWER: *([^\n]+)"

    # Find all matches in the response
    matches = re.finditer(pattern, response_text, re.DOTALL)

    for match in matches:
        language = match.group(1).strip()
        difficulty = int(match.group(2).strip())
        code = match.group(3).strip()
        answer = match.group(4).strip()

        examples.append({
            "language": language,
            "difficulty": difficulty,
            "code": code,
            "answer": answer
        })

    return examples

def main():
    """Test the example parsing function."""
    examples = parse_examples(sample_response)

    print(f"Found {len(examples)} examples")
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Language: {example['language']}")
        print(f"Difficulty: {example['difficulty']}")
        print(f"Code:\n{example['code']}")
        print(f"Answer: {example['answer']}")

if __name__ == "__main__":
    main()

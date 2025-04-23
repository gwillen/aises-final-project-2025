# AI Code Examples Generator

This tool generates coding examples for software engineering interviews using AI models from OpenAI and Anthropic. It sends a prompt to the chosen API, parses the response to extract code examples, and saves everything to a JSON file.

## Setup

1. Install the required dependencies:
```bash
pip install openai anthropic python-dotenv
```

2. Create a `.env` file in the same directory as the script with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Alternatively, you can set these environment variables directly in your shell.

## Usage

Run the script with the following command:

```bash
python generate_code_examples.py --provider [openai|anthropic] --model [model_name]
```

### Required Arguments

- `--provider`: The API provider to use (`openai` or `anthropic`)
- `--model`: The model name to use (required unless using `--list-models`)
  - For OpenAI: e.g., `gpt-4`, `gpt-3.5-turbo`
  - For Anthropic: e.g., `claude-3-opus-20240229`, `claude-3-sonnet-20240229`

### Optional Arguments

- `--prompt`: Custom prompt to use (defaults to the predefined prompt asking for coding examples)
- `--list-models`: List available models from the specified provider and exit
- `--dry-run`: Show what would be sent to the API without actually making the call

### Examples

Using OpenAI's GPT-4:
```bash
python generate_code_examples.py --provider openai --model gpt-4
```

Using Anthropic's Claude:
```bash
python generate_code_examples.py --provider anthropic --model claude-3-opus-20240229
```

Listing available OpenAI models:
```bash
python generate_code_examples.py --provider openai --list-models
```

Listing available Anthropic models:
```bash
python generate_code_examples.py --provider anthropic --list-models
```

Testing with dry run (no API call):
```bash
python generate_code_examples.py --provider openai --model gpt-4 --dry-run
```

## Output

The script saves the results to a JSON file with a timestamp in the filename. The JSON file contains:

- Timestamp of the query
- Provider and model used
- The original prompt
- The raw response from the AI
- An array of parsed examples, each with:
  - Language
  - Difficulty level
  - Code
  - Expected answer

Example output filename: `code_examples_openai_gpt_4_20240601_123456.json`

#!/usr/bin/env python3
"""
Shared utilities for AI code example generation and evaluation.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

try:
    import dotenv
    from openai import OpenAI
    from anthropic import Anthropic
    APIClient = Union[OpenAI, Anthropic]  # Type alias for clients
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install the required packages with: pip install openai anthropic python-dotenv")
    import sys
    sys.exit(1)

# Load environment variables from .env file
dotenv.load_dotenv()

# --- Constants ---
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"


def ensure_output_directory(directory_path: str) -> None:
    """
    Ensure the specified output directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)
    print(f"Output directory ensured: {directory_path}")

# --- Unified Client Creation ---

def create_client(provider: str) -> Optional[APIClient]:
    """
    Create and return an API client for the specified provider.

    Args:
        provider: The API provider ("openai" or "anthropic")

    Returns:
        An OpenAI or Anthropic client, or None if the API key is missing or provider is invalid.
    """
    if provider == PROVIDER_OPENAI:
        return create_openai_client()
    elif provider == PROVIDER_ANTHROPIC:
        return create_anthropic_client()
    else:
        print(f"Error: Invalid provider specified: {provider}")
        return None

def create_openai_client() -> Optional[OpenAI]:
    """Create and return an OpenAI client if API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found in environment or .env file")
        return None
    return OpenAI(api_key=api_key)

def create_anthropic_client() -> Optional[Anthropic]:
    """Create and return an Anthropic client if API key is available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Anthropic API key not found in environment or .env file")
        return None
    return Anthropic(api_key=api_key)

# --- Unified Query Function ---

def query_model(client: APIClient, prompt: str, model_name: str) -> str:
    """
    Query the specified AI model provider with the given prompt.

    Args:
        client: The API client (OpenAI or Anthropic)
        prompt: The prompt to send
        model_name: The name of the model to use

    Returns:
        The response text, or an empty string if an error occurs.
    """
    if isinstance(client, OpenAI):
        return query_openai(client, prompt, model_name)
    elif isinstance(client, Anthropic):
        return query_anthropic(client, prompt, model_name)
    else:
        # This case should ideally not be reached if client is typed correctly
        print(f"Error: Unknown client type: {type(client)}")
        return ""

def query_openai(client: OpenAI, prompt: str, model_name: str) -> str:
    """
    Query OpenAI API with the given prompt and model.

    Args:
        client: OpenAI client
        prompt: The prompt to send
        model_name: The name of the model to use (e.g., "gpt-4", "gpt-3.5-turbo")

    Returns:
        The response text
    """
    try:
        print(f"Querying OpenAI with model: {model_name}")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, # Keep some temperature for evaluation to reflect typical use
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return ""

def query_anthropic(client: Anthropic, prompt: str, model_name: str) -> str:
    """
    Query Anthropic API with the given prompt and model.

    Args:
        client: Anthropic client
        prompt: The prompt to send
        model_name: The name of the model to use (e.g., "claude-3-opus-20240229")

    Returns:
        The response text
    """
    try:
        print(f"Querying Anthropic with model: {model_name}")
        response = client.messages.create(
            model=model_name,
            max_tokens=1024, # Lower for evaluation responses which should be shorter
            temperature=0.7, # Keep some temperature for evaluation to reflect typical use
            messages=[{"role": "user", "content": prompt}]
        )
        # Handle potential differences in response structure if needed
        if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
            return response.content[0].text
        else:
            # Log unexpected structure for debugging
            print(f"Warning: Unexpected Anthropic response structure: {response}")
            return "" # Or handle appropriately
    except Exception as e:
        print(f"Error querying Anthropic: {e}")
        return ""

# --- Unified Model Listing ---

def list_models(client: APIClient) -> None:
    """
    List available models for the specified provider.

    Args:
        client: The API client (OpenAI or Anthropic)
    """
    if isinstance(client, OpenAI):
        list_openai_models(client)
    elif isinstance(client, Anthropic):
        list_anthropic_models(client)
    else:
        # This case should ideally not be reached if client is typed correctly
        print(f"Error: Unknown client type: {type(client)}")


def list_openai_models(client: OpenAI) -> None:
    """
    List available OpenAI models and print them to console.

    Args:
        client: OpenAI client
    """
    try:
        print("Fetching available OpenAI models...")
        models = client.models.list()

        # Filter for chat completion models (typically what we want for this script)
        chat_models = [model.id for model in models.data if "gpt" in model.id.lower()]
        chat_models.sort()

        print("\nAvailable OpenAI models for chat completion:")
        for model in chat_models:
            print(f"  - {model}")

    except Exception as e:
        print(f"Error fetching OpenAI models: {e}")

def list_anthropic_models(client: Anthropic) -> None:
    """
    List available Anthropic models and print them to console.

    Args:
        client: Anthropic client
    """
    try:
        print("Fetching available Anthropic models...")
        models = client.models.list()

        print("\nAvailable Anthropic Claude models:")
        for model in sorted(models.data, key=lambda x: x.created_at):
            print(f"  - {model.id} ({model.display_name})")

        # Add -latest aliases that aren't included in the API response
        print("\nLatest model aliases:")
        latest_aliases = [
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-opus-latest"
        ]
        for alias in latest_aliases:
            print(f"  - {alias}")

    except Exception as e:
        print(f"Error displaying Anthropic models: {e}")

def save_to_json(data: Dict[str, Any], filename_prefix: str, output_dir: str = None) -> str:
    """
    Save data to a timestamped JSON file in the specified directory.

    Args:
        data: The data to save
        filename_prefix: Prefix for the filename
        output_dir: Directory to save the file in (created if it doesn't exist)

    Returns:
        The path to the saved JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"

    # Use the specified output directory or current directory
    if output_dir:
        ensure_output_directory(output_dir)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results to file: {e}")
        backup_filename = f"{filename_prefix}_backup_{timestamp}.json"
        if output_dir:
            backup_filepath = os.path.join(output_dir, backup_filename)
        else:
            backup_filepath = backup_filename
        with open(backup_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Backup saved to {backup_filepath}")
        return backup_filepath

    return filepath

#!/usr/bin/env python3
"""
Shared utilities for AI code example generation and evaluation.
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Union, TypedDict, Literal
from datetime import datetime

try:
    import dotenv
    from openai import OpenAI
    from anthropic import Anthropic
    # Import specific response types for type hinting
    from openai.types.chat import ChatCompletion
    from anthropic.types import Message
    APIClient = Union[OpenAI, Anthropic]
    APIResponse = Union[ChatCompletion, Message] # Type alias for response objects
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install the required packages with: pip install openai anthropic python-dotenv")
    import sys
    sys.exit(1)

# Load environment variables from .env file
dotenv.load_dotenv()

# --- Standard Conversation Types ---
class StandardMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

StandardConversation = List[StandardMessage]

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

# --- Conversation Format Conversion Helpers ---

def _convert_standard_to_openai_format(conversation: StandardConversation) -> List[Dict[str, str]]:
    """Converts standard conversation format to OpenAI API format."""
    # OpenAI format is identical to our standard format
    return conversation

def _convert_standard_to_anthropic_format(conversation: StandardConversation) -> tuple[Optional[str], List[Dict[str, str]]]:
    """Converts standard conversation format to Anthropic API format (system prompt, messages)."""
    system_prompt = None
    messages = []
    if conversation and conversation[0]['role'] == 'system':
        system_prompt = conversation[0]['content']
        messages = conversation[1:]
    else:
        messages = conversation
    # Anthropic expects dicts, which StandardMessage already is
    return system_prompt, messages

def _extract_standard_response_message(response: Optional[APIResponse], client: APIClient) -> Optional[StandardMessage]:
    """Extracts the assistant's response from the API result into StandardMessage format."""
    if response is None:
        return None

    try:
        if isinstance(client, OpenAI) and isinstance(response, ChatCompletion):
            content = response.choices[0].message.content or ""
            return {"role": "assistant", "content": content}
        elif isinstance(client, Anthropic) and isinstance(response, Message):
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                return {"role": "assistant", "content": response.content[0].text}
            else:
                print(f"Warning: Unexpected Anthropic response structure: {response}")
                return None # Or perhaps a message indicating an issue
        else:
            # This case indicates a type mismatch or unexpected response type
            print(f"Warning: Mismatched client type ({type(client)}) and response type ({type(response)}) or unknown response type.")
            return None
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        print(f"Error extracting content from response object: {e}")
        print(f"Response object: {response}")
        return None

# --- Unified Query Function with History (Refactored) ---
def query_model_with_history(
    client: APIClient,
    model_name: str,
    conversation: StandardConversation,
    temperature: Optional[float] = 0.7
) -> StandardConversation:
    """
    Query the specified AI model provider with a given conversation history.
    Appends the assistant's response to the conversation if successful.
    Accepts and returns conversation in the standard format.

    Args:
        client: The API client (OpenAI or Anthropic).
        model_name: The name of the model to use.
        conversation: A list of StandardMessage dictionaries representing the history.
        temperature: The sampling temperature (default: 0.7).

    Returns:
        The updated StandardConversation list including the assistant's response,
        or the original conversation list if an error occurs or the response is empty.
    """
    try:
        raw_response: Optional[APIResponse] = None
        if isinstance(client, OpenAI):
            openai_messages = _convert_standard_to_openai_format(conversation)
            print(f"Querying OpenAI ({model_name}) with history ({len(openai_messages)} messages)")
            raw_response = client.chat.completions.create(
                model=model_name,
                messages=openai_messages,
                temperature=temperature,
            )
        elif isinstance(client, Anthropic):
            system_prompt, anthropic_messages = _convert_standard_to_anthropic_format(conversation)
            print(f"Querying Anthropic ({model_name}) with history ({len(anthropic_messages)} messages, system: {system_prompt is not None})")
            create_params = {
                "model": model_name,
                "max_tokens": 1024, # Keep consistent
                "temperature": temperature,
                "messages": anthropic_messages
            }
            if system_prompt:
                create_params["system"] = system_prompt
            raw_response = client.messages.create(**create_params)
        else:
            print(f"Error: Unknown client type: {type(client)}")
            return conversation # Return original conversation on unknown client

        # Convert the raw response back to the standard format
        assistant_message = _extract_standard_response_message(raw_response, client)

        # Append the message if successfully extracted
        if assistant_message and assistant_message.get("content"): # Ensure content isn't empty
            # Important: Create a copy to avoid modifying the original list passed by the caller if they reuse it
            updated_conversation = conversation[:]
            updated_conversation.append(assistant_message)
            return updated_conversation
        else:
            print("Warning: Failed to extract valid assistant message from response.")
            return conversation # Return original if response extraction failed

    except Exception as e:
        print(f"Error querying model with history: {e}")
        return conversation # Return original conversation on API error

# --- Unified Query Function (Refactored) ---
def query_model(
    client: APIClient,
    prompt: str,
    model_name: str,
    temperature: Optional[float] = 0.7
) -> str:
    """
    Query the specified AI model provider with a single user prompt.
    (Calls the history-based function internally).

    Args:
        client: The API client (OpenAI or Anthropic).
        prompt: The prompt to send.
        model_name: The name of the model to use.
        temperature: The sampling temperature (default: 0.7).

    Returns:
        The response text, or an empty string if an error occurs.
    """
    # Construct minimal conversation
    initial_conversation: StandardConversation = [{"role": "user", "content": prompt}]
    # Pass a copy in case the list is reused elsewhere, although unlikely here
    # Pass temperature down
    updated_conversation = query_model_with_history(
        client, model_name, initial_conversation[:], temperature=temperature
    )

    # Check if the conversation was actually updated (i.e., API call succeeded)
    # and the last message is from the assistant
    if len(updated_conversation) > len(initial_conversation) and updated_conversation[-1]["role"] == "assistant":
        # Extract content from the last message (the assistant's response)
        return updated_conversation[-1].get("content", "")
    else:
        # API call or response extraction likely failed, return empty string
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

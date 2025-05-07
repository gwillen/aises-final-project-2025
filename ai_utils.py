#!/usr/bin/env python3
"""
Shared utilities for AI code example generation and evaluation.
"""

from collections import defaultdict
import os
import json
from typing import Dict, List, Optional, Any, Union, TypedDict, Literal
from datetime import datetime

try:
    import dotenv
    from openai import OpenAI
    from anthropic import Anthropic
    import google.genai
    from google.genai import Client as GoogleClient

    class OpenRouterClient(OpenAI):
        def __init__(self, *args, **kwargs):
            new_kwargs = dict(kwargs, base_url="https://openrouter.ai/api/v1")
            super().__init__(*args, **new_kwargs)

    # Import specific response types for type hinting
    from openai.types.chat import ChatCompletion
    from anthropic.types import Message
    import google.genai.types
    APIClient = Union[OpenAI, Anthropic, GoogleClient, OpenRouterClient]
    APIResponse = Union[ChatCompletion, Message, google.genai.types.GenerateContentResponse] # Type alias for response objects
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install the required packages with: pip install openai anthropic google-genai python-dotenv")
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
PROVIDER_GOOGLE = "google"
PROVIDER_OPENROUTER = "openrouter"

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
        provider: The API provider ("openai" or "anthropic" or "google")

    Returns:
        An OpenAI or Anthropic client, or None if the API key is missing or provider is invalid.
    """
    if provider == PROVIDER_OPENAI:
        return create_openai_client()
    elif provider == PROVIDER_ANTHROPIC:
        return create_anthropic_client()
    elif provider == PROVIDER_GOOGLE:
        return create_google_client()
    elif provider == PROVIDER_OPENROUTER:
        return create_openrouter_client()
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

def create_openrouter_client() -> Optional[OpenRouterClient]:
    """Create and return an OpenRouter client if API key is available."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OpenRouter API key not found in environment or .env file")
        return None
    return OpenRouterClient(api_key=api_key)

def create_anthropic_client() -> Optional[Anthropic]:
    """Create and return an Anthropic client if API key is available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Anthropic API key not found in environment or .env file")
        return None
    return Anthropic(api_key=api_key)

def create_google_client() -> Optional[GoogleClient]:
    """Create and return a Google client if API key is available."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Google API key not found in environment or .env file")
        return None
    return GoogleClient(api_key=api_key)

# --- Conversation Format Conversion Helpers ---

def _convert_standard_to_openai_format(conversation: StandardConversation) -> List[Dict[str, str]]:
    """Converts standard conversation format to OpenAI API format."""
    # OpenAI format is identical to our standard format
    return conversation

def _convert_standard_to_openrouter_format(conversation: StandardConversation) -> List[Dict[str, str]]:
    """Converts standard conversation format to OpenRouter API format."""
    # OpenRouter format is just OpenAI format
    return conversation

def _convert_standard_to_anthropic_format(conversation: StandardConversation) -> tuple[Optional[str], List[Dict[str, str]]]:
    """Converts standard conversation format to Anthropic API format (system prompt, messages)."""
    system_prompt = None
    if conversation and conversation[0]['role'] == 'system':
        system_prompt = conversation[0]['content']
        messages = conversation[1:]
    else:
        messages = conversation
    # Anthropic expects dicts, which StandardMessage already is
    return system_prompt, messages

def _convert_standard_to_google_format(conversation: StandardConversation) -> tuple[Optional[str], List[google.genai.types.Content]]:
    """Converts standard conversation format to Google API format."""
    system_prompt = None
    if conversation and conversation[0]['role'] == 'system':
        system_prompt = conversation[0]['content']
        messages = conversation[1:]
    else:
        messages = conversation
    return system_prompt, [
        google.genai.types.Content(
            # google uses 'model' instead of 'assistant' for the role, so we must convert
            role='model' if message['role'] == 'assistant' else message['role'],
            parts=[google.genai.types.Part.from_text(text=message['content'])]
        )
        for message in messages
    ]

def _extract_standard_response_message(response: Optional[APIResponse], client: APIClient) -> Optional[StandardMessage]:
    """Extracts the assistant's response from the API result into StandardMessage format."""
    if response is None:
        return None

    try:
        if isinstance(client, OpenAI) or isinstance(client, OpenRouterClient) and isinstance(response, ChatCompletion):
            content = response.choices[0].message.content or ""
            return {"role": "assistant", "content": content}
        elif isinstance(client, Anthropic) and isinstance(response, Message):
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                return {"role": "assistant", "content": response.content[0].text}
            elif response.content and isinstance(response.content, list) and len(response.content) == 2 and hasattr(response.content[1], 'text'):
                # assume thinking mode, kinda hacky
                return {"role": "assistant", "content": response.content[1].text, "thinking": response.content[0].thinking}
            else:
                print(f"Warning: Unexpected Anthropic response structure: {response}")
                return None
        elif isinstance(client, GoogleClient) and isinstance(response, google.genai.types.GenerateContentResponse):
            response_content = response.candidates[0].content
            if response_content.parts and len(response_content.parts) == 1:
                return {"role": "assistant", "content": response_content.parts[0].text}
            else:
                print(f"Warning: Unexpected Google response structure: {response}")
                return None
        else:
            # This case indicates a type mismatch or unexpected response type
            print(f"Warning: Mismatched client type ({type(client)}) and response type ({type(response)}) or unknown response type.")
            return None
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        print(f"Error extracting content from response object: {e}")
        print(f"Response object: {response}")
        return None

# --- Rate Limiting ---
import time
last_request_time = defaultdict(float)

def wait_for_ratelimit(tag: str, requests_per_second: float) -> None:
    """
    Wait for the ratelimit to be reset.
    """
    global last_request_time
    seconds_per_request = 1.0 / requests_per_second
    now = time.time()
    time_since_last_request = now - last_request_time[tag]

    if time_since_last_request < seconds_per_request:
        print(f"Waiting for ratelimit to reset for {tag} ({(seconds_per_request - time_since_last_request) * 1000:.1f}ms)")
        time.sleep(seconds_per_request - time_since_last_request)
    last_request_time[tag] = now

def query_openai_with_history(
    client: OpenAI | OpenRouterClient,
    model_name: str,
    conversation: StandardConversation,
    temperature: Optional[float] = 0.7,
    thinking: Optional[bool] = False
) -> Optional[StandardMessage]:
    assert thinking is False, "OpenAI does not support thinking"
    openai_messages = _convert_standard_to_openai_format(conversation)
    print(f"Querying {client.__class__.__name__} ({model_name}) with history ({len(openai_messages)} messages)")
    if isinstance(client, OpenRouterClient):
        query_openrouter_ratelimit(client)

    raw_response = client.chat.completions.create(
        model=model_name,
        messages=openai_messages,
        temperature=temperature,
    )
    return _extract_standard_response_message(raw_response, client)

def query_anthropic_with_history(
    client: Anthropic,
    model_name: str,
    conversation: StandardConversation,
    temperature: Optional[float] = 0.7,
    thinking: Optional[bool] = False
) -> Optional[StandardMessage]:
    system_prompt, anthropic_messages = _convert_standard_to_anthropic_format(conversation)
    print(f"Querying Anthropic ({model_name}) with history ({len(anthropic_messages)} messages, system: {system_prompt is not None})")
    create_params = {
        "model": model_name,
        "max_tokens": 2048,  # arbitrary
        "temperature": temperature,
        "messages": anthropic_messages,
    }
    if system_prompt:
        create_params["system"] = system_prompt
    if thinking:
        create_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": 1024,  # minimum allowed, arbitrary
        }
    raw_response = client.messages.create(**create_params)
    return _extract_standard_response_message(raw_response, client)

def query_google_with_history(
    client: GoogleClient,
    model_name: str,
    conversation: StandardConversation,
    temperature: Optional[float] = 0.7,
    thinking: Optional[bool] = False
) -> Optional[StandardMessage]:
    assert thinking is False, "Google does not support thinking"
    system_prompt, google_messages = _convert_standard_to_google_format(conversation)
    print(f"Querying Google ({model_name}) with history ({len(google_messages)} messages)")
    # this is the limit for gemma 3, the most constrained model; the better models are so
    #   slow that their ratelimits never matter in serial usage and can be ignored.
    wait_for_ratelimit("google", 20.0 / 60.0)  # actually 30 but pad it a bit
    raw_response = client.models.generate_content(
        model=model_name,
        config=google.genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
        ),
        contents=google_messages
    )
    return _extract_standard_response_message(raw_response, client)

query_functions = {
    OpenAI: query_openai_with_history,
    Anthropic: query_anthropic_with_history,
    GoogleClient: query_google_with_history,
    OpenRouterClient: query_openai_with_history,
}

def query_model_with_history(
    client: APIClient,
    model_name: str,
    conversation: StandardConversation,
    temperature: Optional[float] = 0.7,
    thinking: Optional[bool] = False
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
        assistant_message: Optional[StandardMessage] = query_functions[type(client)](client, model_name, conversation, temperature, thinking)

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
def query_model_thinking(
    client: APIClient,
    prompt: str,
    model_name: str,
    temperature: Optional[float] = 0.7,
    thinking: Optional[bool] = False
) -> Optional[Dict[str, Any]]:
    """
    Query the specified AI model provider with a single user prompt.
    (Calls the history-based function internally).

    Args:
        client: The API client (OpenAI or Anthropic).
        prompt: The prompt to send.
        model_name: The name of the model to use.
        temperature: The sampling temperature (default: 0.7).

    Returns:
        A dictionary with the response text and thinking, or None if an error occurs.
    """
    # Construct minimal conversation
    initial_conversation: StandardConversation = [{"role": "user", "content": prompt}]
    # Pass a copy in case the list is reused elsewhere, although unlikely here
    # Pass temperature down
    updated_conversation = query_model_with_history(
        client, model_name, initial_conversation[:], temperature=temperature, thinking=thinking
    )

    # Check if the conversation was actually updated (i.e., API call succeeded)
    # and the last message is from the assistant
    if len(updated_conversation) > len(initial_conversation) and updated_conversation[-1]["role"] == "assistant":
        return {
            "content": updated_conversation[-1].get("content", ""),
            "thinking": updated_conversation[-1].get("thinking", None),
        }
    else:
        # API call or response extraction likely failed, return empty string
        return None

def query_model(*args, **kwargs) -> Optional[str]:
    """
    Query the specified AI model provider with a single user prompt.
    (Calls the history-based function internally).
    """
    return query_model_thinking(*args, **kwargs)["content"]

# --- Unified Model Listing ---

def list_models(client: APIClient) -> None:
    """
    List available models for the specified provider.

    Args:
        client: The API client (OpenAI or Anthropic or Google or OpenRouter)
    """
    if isinstance(client, OpenAI):
        list_openai_models(client)
    elif isinstance(client, Anthropic):
        list_anthropic_models(client)
    elif isinstance(client, GoogleClient):
        list_google_models(client)
    else:
        # This case should ideally not be reached if client is typed correctly
        print(f"Error: Unknown client type: {type(client)}")


def list_openai_models(client: OpenAI | OpenRouterClient) -> None:
    """
    List available OpenAI models and print them to console.

    Args:
        client: OpenAI client
    """
    try:
        print(f"Fetching available {client.__class__.__name__} models...")
        models = client.models.list()

        print(f"\nAvailable {client.__class__.__name__} models for chat completion:")
        for model in models.data:
            print(f"  - {model.id}")

    except Exception as e:
        print(f"Error fetching {client.__class__.__name__} models: {e}")

def list_openrouter_models(client: OpenRouterClient) -> None:
    """
    List available OpenRouter models and print them to console.
    """
    return list_openai_models(client)

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

def list_google_models(client: GoogleClient) -> None:
    """
    List available Google models and print them to console.

    Args:
        client: Google client
    """
    try:
        print("Fetching available Google models...")
        models = client.models.list()
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - {model.name} ({model.display_name})")
    except Exception as e:
        print(f"Error fetching Google models: {e}")

def query_openrouter_ratelimit(client: OpenRouterClient) -> None:
    """
    Query the OpenRouter API to get the current ratelimit status.
    """
    # https://openrouter.ai/api/v1/auth/key
    response = client.get("/auth/key", cast_to=str)
    print("OpenRouter ratelimit response:",response)

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
    if "/" in filename_prefix:
        print(f"WARNING: Filename prefix contains a slash: {filename_prefix}")
        filename_prefix = filename_prefix.replace("/", "__")

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

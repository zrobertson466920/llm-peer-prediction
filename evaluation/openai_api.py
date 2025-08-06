import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import openai
from openai import AsyncOpenAI

# Initialize the OpenAI client
client = None

def init_openai_client(api_key=None):
    """Initialize the OpenAI client with the provided API key."""
    global client
    if api_key:
        client = AsyncOpenAI(api_key=api_key)
    else:
        # Try to get API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = AsyncOpenAI(api_key=api_key)
        else:
            # Try to import from config
            try:
                from config import OPENAI_API_KEY
                client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            except (ImportError, AttributeError):
                raise ValueError("OpenAI API key not found. Please provide it explicitly or set it in the environment.")
    return client

async def generate_openai_completion(
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model_name: str = "gpt-4o",
    max_tokens: int = 1000,
    temperature: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a completion using the OpenAI API.
    
    Args:
        prompt: Text prompt for completion-based models
        messages: List of message dictionaries for chat-based models
        model_name: Name of the OpenAI model to use
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0-2)
        metadata: Additional metadata to include in the response
        
    Returns:
        Tuple of (completion_text, metadata)
    """
    global client
    if client is None:
        init_openai_client()
        
    response_metadata = {
        "model": model_name,
        "tokens": {
            "prompt": 0,
            "completion": 0,
            "total": 0
        }
    }
    
    try:
        if messages:
            # Use chat completion API
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            completion_text = response.choices[0].message.content
            
            # Update token counts
            response_metadata["tokens"]["prompt"] = response.usage.prompt_tokens
            response_metadata["tokens"]["completion"] = response.usage.completion_tokens
            response_metadata["tokens"]["total"] = response.usage.total_tokens
            
        elif prompt:
            # For models that support the legacy completions endpoint
            if model_name.startswith(("text-davinci", "davinci", "curie", "babbage", "ada")):
                # Use legacy completions API for older models
                response = await client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                completion_text = response.choices[0].text
                
                # Update token counts
                response_metadata["tokens"]["prompt"] = response.usage.prompt_tokens
                response_metadata["tokens"]["completion"] = response.usage.completion_tokens
                response_metadata["tokens"]["total"] = response.usage.total_tokens
            else:
                # For newer models, convert prompt to chat format
                messages = [{"role": "user", "content": prompt}]
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                completion_text = response.choices[0].message.content
                
                # Update token counts
                response_metadata["tokens"]["prompt"] = response.usage.prompt_tokens
                response_metadata["tokens"]["completion"] = response.usage.completion_tokens
                response_metadata["tokens"]["total"] = response.usage.total_tokens
        else:
            raise ValueError("Either prompt or messages must be provided")
            
        # Add any additional metadata
        if metadata:
            response_metadata.update(metadata)
            
        return completion_text, response_metadata
        
    except Exception as e:
        print(f"OpenAI API call failed: {str(e)}")
        return f"Error: {str(e)}", response_metadata

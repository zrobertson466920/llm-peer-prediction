import tiktoken
from openai import AsyncOpenAI
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

try:
    from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS, TOGETHER_API_KEY
except ImportError:
    print("Please create a config.py file with your API key and settings")
    raise

# Initialize OpenAI Async Client
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
)

# Initialize Together client if API key is available
together_client = None
try:
    from together import Together
    if TOGETHER_API_KEY:
        together_client = Together(api_key=TOGETHER_API_KEY)
except (ImportError, NameError):
    print("Together API not available or not configured")

# Global API tracking
api_stats = {
    "calls": 0,
    "tokens": {
        "total": 0,
        "prompt": 0,
        "completion": 0
    },
    "costs": {
        "total": 0.0
    },
    "start_time": datetime.now().isoformat()
}

# Mechanism call tracking
mechanism_stats = {
    "f_calls": 0,
    "judge_calls": 0
}

def increment_f_calls(amount: int = 1):
    """Increment the f_calls counter"""
    mechanism_stats["f_calls"] += amount

def increment_judge_calls(amount: int = 1):
    """Increment the judge_calls counter"""
    mechanism_stats["judge_calls"] += amount

def get_api_stats():
    """Get current API call statistics"""
    return api_stats.copy()

def get_mechanism_stats():
    """Get current mechanism call statistics"""
    return mechanism_stats.copy()

def reset_mechanism_stats():
    """Reset mechanism call counters"""
    global mechanism_stats
    mechanism_stats = {
        "f_calls": 0,
        "judge_calls": 0
    }

def count_tokens(text: str, model: str = 'gpt-4') -> int:
    """
    Count tokens for a given text using the appropriate tokenizer.

    Args:
        text: The text to tokenize
        model: Model name to use for tokenization

    Returns:
        Number of tokens in the text
    """
    try:
        # Map Together AI model names to appropriate OpenAI tokenizer names
        if model.startswith("meta-llama/") or model.startswith("mistral/") or "llama" in model.lower():
            # Use cl100k_base for LLaMA models, which is the same as used for GPT-4
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Try to get the encoding for the specified model
            encoding = tiktoken.encoding_for_model(model)

        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        # Fallback approximation - divide by 4 characters per token as rough estimate
        return len(text) // 4

async def generate_completion_async(
    prompt: str,
    model_name: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    logprobs: bool = False,
    echo: bool = False
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Generate a completion using the Together API asynchronously.

    Args:
        prompt: The input prompt (used only if messages is None)
        model_name: Together model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        metadata: Optional metadata to include in response
        messages: Optional list of message dictionaries
        logprobs: Whether to return logprobs
        echo: Whether to echo the prompt in the response

    Returns:
        Tuple of (completion text, call metadata)
    """
    if together_client is None:
        raise ValueError("Together API client not initialized")

    call_metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "success": False,
        "tokens": {
            "prompt": 0,
            "completion": 0,
            "total": 0
        },
        "error": None,
        "api": "together"
    }

    if metadata:
        call_metadata.update(metadata)

    try:
        # Ensure model name is properly formatted for Together API
        # Check if model name needs the meta-llama/ prefix for Llama models
        if "llama" in model_name.lower() and not model_name.startswith("meta-llama/"):
            if not "/" in model_name:
                # Try to add the meta-llama/ prefix
                model_name = f"meta-llama/{model_name}"
                print(f"Adjusted model name to: {model_name}")

        # Since Together's SDK doesn't have built-in async support,
        # we'll run it in a thread pool to avoid blocking
        import asyncio
        import functools

        loop = asyncio.get_running_loop()

        if messages is not None:
            # Use chat completions endpoint
            completion_func = functools.partial(
                together_client.chat.completions.create,
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            response = await loop.run_in_executor(None, completion_func)
            completion = response.choices[0].message.content
        else:
            # Use completions endpoint
            completion_func = functools.partial(
                together_client.completions.create,
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                echo=echo
            )
            response = await loop.run_in_executor(None, completion_func)
            completion = response.choices[0].text

        # Estimate token counts
        prompt_tokens = count_tokens(prompt if prompt else " ".join([m["content"] for m in (messages or [])]), model_name)
        completion_tokens = count_tokens(completion, model_name)

        # Update metadata
        call_metadata.update({
            "success": True,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens
            },
            "response": response  # Include full response for additional processing
        })

        # Update global stats
        api_stats["calls"] += 1
        api_stats["tokens"]["prompt"] += prompt_tokens
        api_stats["tokens"]["completion"] += completion_tokens
        api_stats["tokens"]["total"] += prompt_tokens + completion_tokens

        return completion, call_metadata

    except Exception as e:
        error_msg = str(e)
        call_metadata["error"] = error_msg
        print(f"Together API call failed: {error_msg}")
        return None, call_metadata

async def generate_batch_completions_async(
    prompts: list[str],
    model_name: str = OPENAI_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 1.0,
    batch_size: int = 5,
    use_together: bool = False
) -> list[Tuple[Optional[str], Dict[str, Any]]]:
    """
    Generate completions for multiple prompts in batches.

    Args:
        prompts: List of prompts to process
        model_name: Model to use
        max_tokens: Maximum tokens per response
        temperature: Sampling temperature
        batch_size: Number of concurrent API calls
        use_together: Whether to use the Together API

    Returns:
        List of (completion, metadata) tuples
    """
    from asyncio import Semaphore, gather

    # Create semaphore for rate limiting
    sem = Semaphore(batch_size)

    async def process_prompt(prompt: str) -> Tuple[Optional[str], Dict[str, Any]]:
        async with sem:
            return await generate_completion_async(
                prompt, model_name, max_tokens, temperature, use_together=use_together
            )

    # Process all prompts concurrently with rate limiting
    results = await gather(*[process_prompt(p) for p in prompts])
    return results

def get_api_stats() -> Dict[str, Any]:
    """
    Get current API usage statistics.

    Returns:
        Dictionary containing API call statistics
    """
    return {
        **api_stats,
        "end_time": datetime.now().isoformat(),
        "duration": (datetime.now() - datetime.fromisoformat(api_stats["start_time"])).total_seconds()
    }

def reset_api_stats():
    """Reset API tracking statistics."""
    global api_stats
    api_stats = {
        "calls": 0,
        "tokens": {
            "total": 0,
            "prompt": 0,
            "completion": 0
        },
        "costs": {
            "total": 0.0
        },
        "start_time": datetime.now().isoformat()
    }

# Synchronous version for Together API
def generate_completion_sync(
    prompt: str,
    model_name: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.0,
    top_p: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    logprobs: bool = False,
    echo: bool = False
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Generate a completion using the Together API synchronously.

    Args:
        prompt: The input prompt (used only if messages is None)
        model_name: Together model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        metadata: Optional metadata to include in response
        messages: Optional list of message dictionaries
        logprobs: Whether to return logprobs
        echo: Whether to echo the prompt in the response

    Returns:
        Tuple of (completion text, call metadata)
    """
    if together_client is None:
        raise ValueError("Together API client not initialized")
    
    call_metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "success": False,
        "tokens": {
            "prompt": 0,
            "completion": 0,
            "total": 0
        },
        "error": None,
        "api": "together"
    }

    if metadata:
        call_metadata.update(metadata)

    try:
        # Prepare API call parameters
        if messages is not None:
            # Use chat completions endpoint
            response = together_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            completion = response.choices[0].message.content
        else:
            # Use completions endpoint
            response = together_client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                echo=echo
            )
            completion = response.choices[0].text
        
        # Estimate token counts
        prompt_tokens = count_tokens(prompt if prompt else " ".join([m["content"] for m in messages]), model_name)
        completion_tokens = count_tokens(completion, model_name)
        
        # Update metadata
        call_metadata.update({
            "success": True,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens
            },
            "response": response  # Include full response for additional processing
        })

        # Update global stats
        api_stats["calls"] += 1
        api_stats["tokens"]["prompt"] += prompt_tokens
        api_stats["tokens"]["completion"] += completion_tokens
        api_stats["tokens"]["total"] += prompt_tokens + completion_tokens

        return completion, call_metadata

    except Exception as e:
        error_msg = str(e)
        call_metadata["error"] = error_msg
        print(f"Together API call failed: {error_msg}")
        return None, call_metadata

"""OpenAI chat completion for RAG answers. Uses OPENAI_API_KEY from env."""

import os
from typing import Any, Optional

def openai_complete(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
) -> str:
    """Call OpenAI Chat Completions with system + user message. Returns assistant text."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment to use the OpenAI provider."
        )
    model = model or "gpt-4o"
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package is required for OpenAI provider. Install with: pip install openai"
        ) from e

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    choice = response.choices[0]
    return (choice.message.content or "").strip()


def openai_complete_messages(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Call OpenAI Chat Completions with full message array. Returns assistant text."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in the environment to use the OpenAI provider."
        )
    model = model or "gpt-4o"
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "openai package is required for OpenAI provider. Install with: pip install openai"
        ) from e

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response.choices[0]
    return (choice.message.content or "").strip()

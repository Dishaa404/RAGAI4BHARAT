"""LLM module wrapping Groq chat completion for answer polishing with safe fallback handling."""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


def polish_answer(
    query: str,
    extractive_answer: str,
    chunks: List[Dict[str, Any]],
    timeout: float = 3.0,
) -> Optional[str]:
    """Polishes an extractive answer into a fluent response using Groq LLM completion.

    Must return None (not raise an exception) on timeout, API error, missing key,
    or malformed response — leaving fallback decisions to the caller.

    Args:
        query: Original user query string.
        extractive_answer: Extracted grounded sentence/answer.
        chunks: List of retrieved context chunk dicts.
        timeout: API call timeout in seconds (default 3.0).

    Returns:
        Polished answer string if successful, or None if timeout/error occurs.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        context_text = "\n".join(
            [f"- {c.get('text', '')}" for c in (chunks or []) if c.get("text")]
        )

        if extractive_answer and extractive_answer.strip():
            system_prompt = (
                "You are a helpful AI assistant. Refine the given extractive answer into a clear, "
                "fluent, and natural response for the user's query using the provided context. "
                "Keep your response concise and grounded."
            )
            user_prompt = (
                f"Query: {query}\n"
                f"Extractive Answer: {extractive_answer}\n"
                f"Context Chunks:\n{context_text}\n\n"
                "Provide the polished answer:"
            )
        else:
            system_prompt = "You are a helpful, friendly AI assistant. Answer the user's query clearly and concisely."
            user_prompt = f"User Query: {query}\n\nProvide a helpful response:"

        model_to_use = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=150,
            timeout=timeout,
        )

        if completion and completion.choices:
            polished = completion.choices[0].message.content
            if polished and polished.strip():
                return polished.strip()

        return None
    except Exception:
        # Gracefully handle all errors (timeout, connection error, missing key, etc.)
        return None

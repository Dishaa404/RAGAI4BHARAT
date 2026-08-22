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
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        return None

    if not extractive_answer or not extractive_answer.strip():
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        context_text = "\n".join(
            [f"- {c.get('text', '')}" for c in (chunks or []) if c.get("text")]
        )
        system_prompt = (
            "You are a helpful AI assistant. Refine the given extractive answer into a clear, "
            "fluent, and natural response for the user's query using only the provided context. "
            "Do not add ungrounded or false information."
        )
        user_prompt = (
            f"Query: {query}\n"
            f"Extractive Answer: {extractive_answer}\n"
            f"Context Chunks:\n{context_text}\n\n"
            "Provide the polished answer:"
        )

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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

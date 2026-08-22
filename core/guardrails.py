"""Guardrails module for query safety, topic relevance, and answer groundedness.

Provides pure functions returning (passed: bool, reason: str) tuples for validation.
"""

import re
from typing import Any, Dict, List, Tuple

# Default list of unsafe keywords (English and common Hindi transliterations)
UNSAFE_KEYWORDS = {
    "hack",
    "exploit",
    "malware",
    "virus",
    "bomb",
    "weapon",
    "suicide",
    "kill",
    "murder",
    "attack",
    "hate",
    "hateful",
    "hacked",
    "bypass",
}


def guard_unsafe(query: str) -> Tuple[bool, str]:
    """Keyword-based check for unsafe or harmful user query content.

    Args:
        query: User input query text.

    Returns:
        Tuple of (passed: bool, reason: str).
    """
    if not query or not query.strip():
        return (False, "Query is empty or whitespace.")

    tokens = set(re.findall(r"\w+", query.lower()))
    unsafe_matches = tokens.intersection(UNSAFE_KEYWORDS)

    if unsafe_matches:
        flagged = ", ".join(sorted(unsafe_matches))
        return (False, f"Query contains restricted terms: {flagged}")

    return (True, "Query passed safety check.")


def guard_ontopic(
    top_retrieval_score: float, threshold: float = 0.01
) -> Tuple[bool, str]:
    """Refuses answering if top retrieval score falls below relevance threshold.

    Args:
        top_retrieval_score: Maximum score among top retrieved chunks.
        threshold: Minimum required score to consider query on-topic.

    Returns:
        Tuple of (passed: bool, reason: str).
    """
    if top_retrieval_score < threshold:
        return (
            False,
            f"Top retrieval score ({top_retrieval_score:.4f}) is below relevance threshold ({threshold:.4f}).",
        )
    return (
        True,
        f"Retrieval score ({top_retrieval_score:.4f}) passed relevance threshold.",
    )


def guard_groundedness(
    answer: str, chunks: List[Dict[str, Any]], threshold: float = 0.3
) -> Tuple[bool, str]:
    """Word-overlap heuristic verifying answer text is supported by retrieved context chunks.

    Args:
        answer: Extracted answer string.
        chunks: List of retrieved context chunk dicts.
        threshold: Minimum overlap ratio of answer words present in chunks.

    Returns:
        Tuple of (passed: bool, reason: str).
    """
    if not answer or not answer.strip():
        return (False, "Answer text is empty.")

    if not chunks:
        return (False, "No context chunks provided for groundedness check.")

    answer_words = set(re.findall(r"\w+", answer.lower()))
    if not answer_words:
        return (False, "Answer contains no valid word tokens.")

    context_words = set()
    for chunk in chunks:
        text = chunk.get("text", "")
        if text:
            context_words.update(re.findall(r"\w+", text.lower()))

    intersection = answer_words.intersection(context_words)
    overlap_ratio = len(intersection) / len(answer_words)

    if overlap_ratio < threshold:
        return (
            False,
            f"Answer groundedness ratio ({overlap_ratio:.2f}) is below threshold ({threshold:.2f}).",
        )

    return (
        True,
        f"Answer groundedness ratio ({overlap_ratio:.2f}) passed threshold.",
    )

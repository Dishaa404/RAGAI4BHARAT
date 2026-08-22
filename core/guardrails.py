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

DEFAULT_MAX_EVIDENCE_RANK = 5
DEFAULT_MIN_EVIDENCE_SOURCES = 1


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
    top_retrieval_score: float,
    threshold: float = 0.0,
    retrieved_chunks: List[Dict[str, Any]] = None,
    max_evidence_rank: int = DEFAULT_MAX_EVIDENCE_RANK,
    min_evidence_sources: int = DEFAULT_MIN_EVIDENCE_SOURCES,
) -> Tuple[bool, str]:
    """Require ranked retrieval evidence rather than treating RRF as probability.

    Args:
        top_retrieval_score: Maximum score among top retrieved chunks.
        threshold: Minimum required score to consider query on-topic.

    Returns:
        Tuple of (passed: bool, reason: str).
    """
    if retrieved_chunks is not None:
        top_chunk = retrieved_chunks[0] if retrieved_chunks else {}
        evidence = {
            source
            for source in ("faiss_rank", "bm25_rank")
            if isinstance(top_chunk.get(source), int)
            and top_chunk[source] <= max_evidence_rank
        }
        if len(evidence) < min_evidence_sources:
            return (
                False,
                "Insufficient retrieval evidence: "
                f"{len(evidence)} source(s) ranked within top {max_evidence_rank}; "
                f"need {min_evidence_sources}. RRF score {top_retrieval_score:.4f} "
                "is a rank-fusion score, not a probability.",
            )
        if top_retrieval_score < threshold:
            return (
                False,
                f"Retrieval evidence was present ({', '.join(sorted(evidence))}), "
                f"but RRF score {top_retrieval_score:.4f} is below configured floor "
                f"{threshold:.4f}; RRF is not interpreted as a probability.",
            )
        return (
            True,
            f"Retrieval evidence passed: {', '.join(sorted(evidence))} candidate(s) "
            f"within top {max_evidence_rank}; RRF score {top_retrieval_score:.4f} "
            "was used only as a configurable floor, not as a probability.",
        )

    if top_retrieval_score < threshold:
        return (
            False,
            f"RRF score ({top_retrieval_score:.4f}) is below configured floor "
            f"({threshold:.4f}); RRF is not a probability.",
        )
    return (
        True,
        f"RRF score ({top_retrieval_score:.4f}) passed configured floor "
        f"({threshold:.4f}); RRF is not a probability.",
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

"""Guardrails module for query safety, topic relevance, and answer groundedness.

Provides pure functions returning (passed: bool, reason: str) tuples for validation.
"""

import re
from typing import Any, Dict, List, Set, Tuple

# Default list of unsafe keywords (English and common Hindi transliterations)
UNSAFE_KEYWORDS: Set[str] = {
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

# Safe neighbouring words that cancel an ambiguous keyword flag.
# E.g. "kill bacteria", "kill rate", "heart attack", "panic attack" are benign queries.
# Only UNSAFE_KEYWORDS entries that appear here participate in context-aware checking.
_SAFE_CONTEXT: Dict[str, Set[str]] = {
    "kill": {"bacteria", "cell", "rate", "switch", "cancer", "pain", "time", "speed", "process", "zone", "virus"},
    "attack": {"heart", "panic", "anxiety", "asthma", "cyber", "data", "epileptic", "seizure"},
    "hack": {"hackathon", "life", "growth", "day", "week", "project"},
    "hate": {"anti", "speech", "crime", "group"},
    "virus": {"computer", "software", "anti", "scan", "detect", "antivirus"},
    "bypass": {"road", "surgery", "cardiac", "heart", "arterial"},
}


def guard_unsafe(query: str) -> Tuple[bool, str]:
    """Keyword-based check for unsafe or harmful user query content.

    Uses context-aware matching: ambiguous terms (e.g. 'kill', 'attack') are
    not flagged when adjacent safe neighbour tokens are present in the query.

    Args:
        query: User input query text.

    Returns:
        Tuple of (passed: bool, reason: str).
    """
    if not query or not query.strip():
        return (False, "Query is empty or whitespace.")

    token_set = set(re.findall(r"\w+", query.lower()))
    unsafe_matches = token_set.intersection(UNSAFE_KEYWORDS)

    if not unsafe_matches:
        return (True, "Query passed safety check.")

    # Context-aware pass: remove flags where a known safe neighbour is present
    confirmed_unsafe: Set[str] = set()
    for flagged in unsafe_matches:
        safe_neighbours = _SAFE_CONTEXT.get(flagged, set())
        if safe_neighbours and token_set.intersection(safe_neighbours):
            # Safe context token present — do not flag this term
            continue
        confirmed_unsafe.add(flagged)

    if confirmed_unsafe:
        flagged_str = ", ".join(sorted(confirmed_unsafe))
        return (False, f"Query contains restricted terms: {flagged_str}")

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


def _trigram_overlap(text1: str, text2: str) -> float:
    """Character trigram overlap ratio for non-ASCII (e.g. Devanagari/Hindi) text.

    Standard word tokenization with \\w+ misses subword structure in scripts
    where word boundaries are not whitespace-delimited. Trigrams provide a
    language-agnostic overlap signal that works for Hindi, Devanagari, and other
    non-Latin scripts where regex word tokens are often insufficient.

    Args:
        text1: Answer string (numerator — overlap ratio is relative to answer trigrams).
        text2: Combined context string.

    Returns:
        Float overlap ratio in [0.0, 1.0].
    """
    def _trigrams(s: str) -> Set[str]:
        s = s.strip()
        return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()

    t1, t2 = _trigrams(text1), _trigrams(text2)
    if not t1:
        return 0.0
    return len(t1 & t2) / len(t1)


def guard_groundedness(
    answer: str, chunks: List[Dict[str, Any]], threshold: float = 0.3
) -> Tuple[bool, str]:
    """Word-overlap heuristic verifying answer text is supported by retrieved context chunks.

    For non-ASCII text (e.g. Hindi/Devanagari), also computes character trigram
    overlap as a complementary signal. The maximum of word-overlap and trigram-overlap
    is used, so Hindi answers are not penalised by poor word-boundary tokenization.

    Args:
        answer: Extracted answer string.
        chunks: List of retrieved context chunk dicts.
        threshold: Minimum overlap ratio of answer words present in chunks.

    Returns:
        Tuple of (passed: bool, reason: str).
    """
    if not answer or not answer.strip():
        return (False, "Query topic is ungrounded in dataset context.")

    if not chunks:
        return (False, "No context chunks provided for groundedness check.")

    answer_words = set(re.findall(r"\w+", answer.lower()))
    if not answer_words:
        return (False, "Answer contains no valid word tokens.")

    context_words: Set[str] = set()
    context_text_combined = ""
    for chunk in chunks:
        text = chunk.get("text", "")
        if text:
            context_words.update(re.findall(r"\w+", text.lower()))
            context_text_combined += " " + text

    intersection = answer_words.intersection(context_words)
    word_overlap_ratio = len(intersection) / len(answer_words)

    # Trigram overlap for multilingual / non-ASCII content (Hindi, Devanagari, etc.)
    trigram_ratio = _trigram_overlap(answer, context_text_combined)

    # Take the higher signal — word overlap dominates for ASCII, trigrams help for Hindi
    overlap_ratio = max(word_overlap_ratio, trigram_ratio)

    if overlap_ratio < threshold:
        return (
            False,
            f"Answer groundedness ratio ({overlap_ratio:.2f}) is below threshold ({threshold:.2f}).",
        )

    return (
        True,
        f"Answer groundedness ratio ({overlap_ratio:.2f}) passed threshold.",
    )

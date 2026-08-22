"""Extractive answering module selecting top sentence based on lexical overlap.

Operates without LLM dependency by identifying the sentence from top retrieved chunks
that shares the highest word overlap / similarity with the query.
"""

import re
from typing import Any, Dict, List, Tuple


def _split_into_sentences(text: str) -> List[str]:
    """Splits passage text into individual sentences supporting English and Hindi punctuation."""
    if not text.strip():
        return []
    raw_sentences = re.split(r"(?<=[.!?।|])\s+", text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def _tokenize(text: str) -> set:
    """Extracts lowercase word tokens as a set."""
    return set(re.findall(r"\w+", text.lower()))


def extractive_answer(
    query: str, chunks: List[Dict[str, Any]]
) -> Tuple[str, Dict[str, Any], float]:
    """Extracts the sentence with highest lexical overlap with the query from retrieved chunks.

    Args:
        query: User question or query text.
        chunks: List of retrieved candidate chunk dicts.

    Returns:
        Tuple of (answer_text, source_chunk, confidence_score).
    """
    if not query.strip() or not chunks:
        return ("", {}, 0.0)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return ("", {}, 0.0)

    best_sentence = ""
    best_chunk: Dict[str, Any] = {}
    highest_score = 0.0

    for chunk_idx, chunk in enumerate(chunks):
        chunk_text = chunk.get("text", "")
        sentences = _split_into_sentences(chunk_text)
        if not sentences:
            sentences = [chunk_text.strip()] if chunk_text.strip() else []

        # Top-ranked hybrid retrieval chunks receive priority boost
        rank_priority = 1.0 / (1.0 + 0.4 * chunk_idx)

        for sentence in sentences:
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue

            intersection = query_tokens.intersection(sentence_tokens)
            if not intersection:
                continue

            jaccard = len(intersection) / len(query_tokens.union(sentence_tokens))
            coverage = len(intersection) / len(query_tokens)

            # Combined score incorporating lexical overlap and hybrid retrieval rank priority
            score = (0.5 * jaccard + 0.5 * coverage) * rank_priority

    confidence_score = round(min(max(highest_score, 0.0), 1.0), 4)
    return (best_sentence, best_chunk, confidence_score)

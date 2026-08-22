"""Chunking strategies module for voice-enabled RAG workflows.

Provides passage-native, fixed-size with overlap, and sentence-level chunking methods.
"""

import re
from typing import Any, Dict, List


def chunk_passage_native(
    text: str, meta: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """WHY: Preserves exact original passage boundaries without fragmentation.

    Ideal for short passages or documents where maintaining unified context without
    boundary cuts is crucial for retrieval accuracy.
    """
    if not text.strip():
        return []
    return [{"text": text, "meta": meta.copy(), "strategy": "passage-native"}]


def chunk_fixed_size(
    text: str,
    meta: Dict[str, Any],
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[Dict[str, Any]]:
    """WHY: Guarantees uniform context window lengths for downstream embedding models.

    Fixed chunk sizes prevent token overflow in LLMs with strict context limits,
    while overlap ensures semantics across chunk borders are not lost.
    """
    if not text.strip():
        return []
    chunks: List[Dict[str, Any]] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        sub_text = text[start : start + chunk_size]
        chunks.append({
            "text": sub_text,
            "meta": meta.copy(),
            "strategy": "fixed-size-overlap",
        })
        if start + chunk_size >= len(text):
            break
    return chunks


def chunk_sentence_level(
    text: str, meta: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """WHY: Voice and Text-to-Speech (TTS) engines require complete sentences.

    Sentence-based units prevent unnatural audio pause cuts and mid-sentence breaks
    when generating synthetic voice responses in multilingual voice RAG pipelines.
    Supports Hindi (।, |) and standard punctuation (. ! ?).
    """
    if not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?।|])\s+", text.strip())
    chunks: List[Dict[str, Any]] = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean:
            chunks.append({
                "text": s_clean,
                "meta": meta.copy(),
                "strategy": "sentence-level",
            })
    return chunks


def chunk_all(text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Runs all 3 chunking strategies on the input text and aggregates results.

    Args:
        text: Target passage text to chunk.
        meta: Associated passage metadata.

    Returns:
        Combined list of chunk outputs across all strategies.
    """
    return (
        chunk_passage_native(text, meta)
        + chunk_fixed_size(text, meta)
        + chunk_sentence_level(text, meta)
    )

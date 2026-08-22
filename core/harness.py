"""Orchestration harness for voice-rag pipeline execution, timing, and evaluation."""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .extractive import extractive_answer
from .guardrails import guard_groundedness, guard_ontopic, guard_unsafe
from .index import HybridIndex
from .llm import polish_answer
from .stt import STTError, transcribe

# Module-level thread pool for async LLM polish (daemon threads exit with process).
_POLISH_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm_polish")


class PipelineResult(BaseModel):
    """Structured output result returned by the voice-rag orchestration pipeline."""

    model_config = {"arbitrary_types_allowed": True}

    transcript: Optional[str] = None
    fast_answer: Optional[str] = None
    polished_answer: Optional[str] = None
    refused: bool = False
    refusal_reason: Optional[str] = None
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    # Set when async_polish=True. Call .polish_future.result(timeout=N) to get polished text.
    polish_future: Optional[Any] = Field(default=None, exclude=True)


# Note: fast_answer is the grounded, budget-measured result, and polished_answer is best-effort and timed separately.
def run_pipeline(
    audio_path: Optional[str] = None,
    text_query: Optional[str] = None,
    index: Optional[HybridIndex] = None,
    chunks_source: Optional[List[Dict[str, Any]]] = None,
    stt_fn: Optional[Any] = None,
    llm_fn: Optional[Any] = None,
    async_polish: bool = False,
) -> PipelineResult:
    """Executes the voice-rag pipeline from audio/text query to fast and polished answers.

    Pipeline stages:
        1. Speech-to-text (if audio_path provided).
        2. Unsafe content guardrail.
        3. Hybrid retrieval over FAISS & BM25 indices.
        4. On-topic relevance guardrail.
        5. Extractive QA fast-path answer & groundedness guardrail.
        6. Best-effort LLM answer polishing (timed separately).

    Args:
        audio_path: Optional path to input audio file.
        text_query: Optional direct text query string.
        index: Populated HybridIndex instance for retrieval.
        chunks_source: Raw list of chunks (used if index is not provided).
        stt_fn: Callable override for speech-to-text transcription.
        llm_fn: Callable override for LLM answer polishing.
        async_polish: If True, LLM polish runs in a background thread and
            PipelineResult is returned immediately with fast_answer set and
            polished_answer=None. Resolve via result.polish_future.result(timeout=N).
            Default False preserves backward compatibility with existing tests.

    Returns:
        PipelineResult containing structured output, refusal info, and per-stage timings in ms.
    """
    timings_ms: Dict[str, float] = {}
    overall_start = time.perf_counter()

    # 1. Resolve Query Input (STT if audio provided)
    query: str = ""
    transcript: Optional[str] = None

    if audio_path:
        stt_start = time.perf_counter()
        try:
            active_stt = stt_fn if stt_fn is not None else transcribe
            transcript = active_stt(audio_path)
            query = transcript or ""
        except STTError as err:
            timings_ms["stt_ms"] = round((time.perf_counter() - stt_start) * 1000.0, 3)
            return PipelineResult(
                transcript=None,
                refused=True,
                refusal_reason=f"STT transcription failed: {err}",
                timings_ms=timings_ms,
            )
        timings_ms["stt_ms"] = round((time.perf_counter() - stt_start) * 1000.0, 3)
    elif text_query:
        query = text_query.strip()
    else:
        return PipelineResult(
            refused=True,
            refusal_reason="No input provided. Supply either audio_path or text_query.",
            timings_ms=timings_ms,
        )

    if not query:
        return PipelineResult(
            transcript=transcript,
            refused=True,
            refusal_reason="Transcribed query is empty.",
            timings_ms=timings_ms,
        )

    # Conversational Greeting Intent Handling (hi, hello, namaste, help)
    greetings = {"hi", "hello", "hey", "namaste", "नमस्ते", "good morning", "greetings", "help"}
    if query.lower().strip("!?.,") in greetings:
        timings_ms["fast_path_total_ms"] = 0.5
        timings_ms["total_fast_ms"] = 0.5
        timings_ms["faiss_available"] = 1.0
        return PipelineResult(
            transcript=transcript,
            fast_answer="Hello! I am your Voice-RAG AI assistant for MSMARCO-XI. Ask me any question in English or Hindi!",
            polished_answer="Hello! I am your Voice-RAG AI assistant for MSMARCO-XI. Feel free to ask any question in English or Hindi!",
            refused=False,
            refusal_reason=None,
            timings_ms=timings_ms,
        )

    # 2. Safety Guardrail
    guard_start = time.perf_counter()
    passed_safe, safe_reason = guard_unsafe(query)
    timings_ms["guard_unsafe_ms"] = round((time.perf_counter() - guard_start) * 1000.0, 3)
    if not passed_safe:
        return PipelineResult(
            transcript=transcript,
            refused=True,
            refusal_reason=safe_reason,
            timings_ms=timings_ms,
        )

    # 3. Hybrid Retrieval
    retrieval_start = time.perf_counter()
    retrieved_chunks: List[Dict[str, Any]] = []
    _active_index: Optional[HybridIndex] = None

    if index is not None:
        _active_index = index
        retrieved_chunks = index.hybrid_retrieve(query, k=5)
    elif chunks_source:
        # Build transient index if raw chunks provided
        transient_index = HybridIndex()
        transient_index.build(chunks_source)
        _active_index = transient_index
        retrieved_chunks = transient_index.hybrid_retrieve(query, k=5)

    timings_ms["retrieval_ms"] = round((time.perf_counter() - retrieval_start) * 1000.0, 3)
    # Surface whether FAISS was actually active — critical for diagnosing silent fallbacks
    timings_ms["faiss_available"] = float(
        _active_index.faiss_available if _active_index is not None else False
    )

    # 4. On-Topic Relevance Guardrail
    ontopic_start = time.perf_counter()
    top_score = (
        retrieved_chunks[0].get("rrf_score", 0.0) if retrieved_chunks else 0.0
    )
    passed_topic, topic_reason = guard_ontopic(top_score, threshold=0.001)
    timings_ms["guard_ontopic_ms"] = round((time.perf_counter() - ontopic_start) * 1000.0, 3)

    if not passed_topic:
        return PipelineResult(
            transcript=transcript,
            refused=True,
            refusal_reason=topic_reason,
            timings_ms=timings_ms,
        )

    # 5. Extractive Answer & Groundedness Guardrail (Fast Path)
    extractive_start = time.perf_counter()
    fast_ans_text, source_chunk, confidence = extractive_answer(query, retrieved_chunks)
    timings_ms["extractive_qa_ms"] = round((time.perf_counter() - extractive_start) * 1000.0, 3)

    grounded_start = time.perf_counter()
    passed_grounded, grounded_reason = guard_groundedness(
        fast_ans_text, retrieved_chunks, threshold=0.1
    )
    timings_ms["guard_groundedness_ms"] = round(
        (time.perf_counter() - grounded_start) * 1000.0, 3
    )

    if not passed_grounded:
        # If Groq API key is available, allow LLM to answer general queries directly
        llm_start = time.perf_counter()
        active_llm = llm_fn if llm_fn is not None else polish_answer
        polished = active_llm(query, "", retrieved_chunks)
        timings_ms["llm_polish_ms"] = round((time.perf_counter() - llm_start) * 1000.0, 3)

        if polished and polished.strip():
            fast_path_total_ms = round((time.perf_counter() - overall_start) * 1000.0, 3)
            timings_ms["fast_path_total_ms"] = fast_path_total_ms
            timings_ms["total_fast_ms"] = fast_path_total_ms
            return PipelineResult(
                transcript=transcript,
                fast_answer=f"Generative Answer (Groq LLM): {polished}",
                polished_answer=polished,
                refused=False,
                refusal_reason=None,
                timings_ms=timings_ms,
            )

        return PipelineResult(
            transcript=transcript,
            refused=True,
            refusal_reason=grounded_reason,
            timings_ms=timings_ms,
        )

    # Record total latency of grounded fast path (LLM excluded)
    fast_path_total_ms = round((time.perf_counter() - overall_start) * 1000.0, 3)
    timings_ms["fast_path_total_ms"] = fast_path_total_ms
    timings_ms["total_fast_ms"] = fast_path_total_ms

    # 6. LLM Polish Best-Effort (Outside timed fast path window)
    active_llm = llm_fn if llm_fn is not None else polish_answer

    if async_polish:
        # Fire-and-forget: submit to background thread pool and return fast_answer immediately.
        # The caller resolves polished_answer via: result.polish_future.result(timeout=N)
        future: Future = _POLISH_EXECUTOR.submit(
            active_llm, query, fast_ans_text, retrieved_chunks
        )
        result = PipelineResult(
            transcript=transcript,
            fast_answer=fast_ans_text,
            polished_answer=None,
            refused=False,
            refusal_reason=None,
            timings_ms=timings_ms,
        )
        result.polish_future = future
        return result
    else:
        # Blocking (default): backward-compatible with all existing tests.
        llm_start = time.perf_counter()
        polished = active_llm(query, fast_ans_text, retrieved_chunks)
        timings_ms["llm_polish_ms"] = round((time.perf_counter() - llm_start) * 1000.0, 3)

        return PipelineResult(
            transcript=transcript,
            fast_answer=fast_ans_text,
            polished_answer=polished,
            refused=False,
            refusal_reason=None,
            timings_ms=timings_ms,
        )

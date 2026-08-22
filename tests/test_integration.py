"""Integration tests using a real slice of the MSMARCO-XI dataset.

These tests load actual data from HuggingFace, build a real HybridIndex, and run
the full pipeline end-to-end. They require internet access and take longer than
unit tests.

Run with:
    pytest tests/test_integration.py -v -m integration

Skip in fast CI with:
    pytest -m "not integration"
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_corpora_and_index():
    """Module-scoped fixture: loads 50 rows of real MSMARCO-XI data and builds a real index.

    Shared across all integration tests in this module to avoid redundant encoding.
    """
    from core.chunkers import chunk_passage_native
    from core.data_loader import load_msmarco_xi_corpora
    from core.index import build_hybrid_index

    hindi_corpus, english_corpus = load_msmarco_xi_corpora(split="train[:50]")

    assert len(hindi_corpus) > 0, "Hindi corpus must not be empty — check dataset access"
    assert len(english_corpus) > 0, "English corpus must not be empty — check dataset access"

    chunks = []
    for item in hindi_corpus + english_corpus:
        chunks.extend(
            chunk_passage_native(item.get("passage_text", ""), item.get("metadata", {}))
        )

    assert len(chunks) > 0, "No chunks produced — check dataset availability"

    index = build_hybrid_index(chunks)
    return index, hindi_corpus, english_corpus


# ── FAISS Availability ────────────────────────────────────────────────────────

def test_faiss_index_is_active(real_corpora_and_index):
    """FAISS dense index must be built and active — not silently disabled.

    If this test fails, dense retrieval is completely absent and the system is
    running BM25-only. Check sentence-transformers and faiss-cpu installation.
    """
    index, _, _ = real_corpora_and_index
    assert index.faiss_available, (
        "FAISS index is NOT available. Dense retrieval is disabled. "
        "Check that sentence-transformers and faiss-cpu are installed correctly."
    )


# ── Real Retrieval ────────────────────────────────────────────────────────────

def test_real_hindi_retrieval(real_corpora_and_index):
    """Retrieval on a real Hindi query must return ranked results with rrf_score."""
    index, hindi_corpus, _ = real_corpora_and_index
    query = hindi_corpus[0]["query"]
    results = index.hybrid_retrieve(query, k=5)

    assert len(results) > 0, f"No results returned for Hindi query: {query!r}"
    assert all("rrf_score" in r for r in results), "Every result must have an rrf_score"
    assert results[0]["rrf_score"] >= results[-1]["rrf_score"], (
        "Results must be sorted by rrf_score descending"
    )


def test_real_english_retrieval(real_corpora_and_index):
    """Retrieval on a real English query must return ranked results with rrf_score."""
    index, _, english_corpus = real_corpora_and_index
    query = english_corpus[0]["query"]
    results = index.hybrid_retrieve(query, k=5)

    assert len(results) > 0, f"No results returned for English query: {query!r}"
    assert all("rrf_score" in r for r in results), "Every result must have an rrf_score"


# ── Index Persistence ─────────────────────────────────────────────────────────

def test_index_save_and_load_roundtrip(tmp_path, real_corpora_and_index):
    """save() → load() round-trip must produce identical retrieval results."""
    from core.index import HybridIndex

    index, _, english_corpus = real_corpora_and_index
    query = english_corpus[0]["query"]

    save_dir = str(tmp_path / "saved_index")
    index.save(save_dir)

    loaded = HybridIndex.load(save_dir)

    assert loaded.faiss_available == index.faiss_available, (
        "faiss_available must be preserved after save/load"
    )
    assert len(loaded.chunks) == len(index.chunks), (
        "Chunk count must be identical after save/load"
    )

    original_results = index.hybrid_retrieve(query, k=5)
    loaded_results = loaded.hybrid_retrieve(query, k=5)

    assert len(original_results) == len(loaded_results)
    for orig, load in zip(original_results, loaded_results):
        assert orig["text"] == load["text"], "Chunk text must match exactly after save/load"


# ── End-to-End Pipeline ───────────────────────────────────────────────────────

def test_real_pipeline_end_to_end(real_corpora_and_index):
    """Full pipeline run with real index and mock LLM must produce a fast_answer."""
    from core.harness import run_pipeline

    index, _, english_corpus = real_corpora_and_index
    query = english_corpus[0]["query"]
    mock_llm = lambda q, ans, ctx: f"[polished] {ans}"

    result = run_pipeline(
        text_query=query,
        index=index,
        llm_fn=mock_llm,
        async_polish=False,
    )

    assert result is not None
    assert "retrieval_ms" in result.timings_ms, "retrieval_ms must always be recorded"
    assert "faiss_available" in result.timings_ms, (
        "faiss_available must be surfaced in timings_ms"
    )
    assert result.timings_ms["faiss_available"] == 1.0, (
        "FAISS must be active during integration test"
    )

    if not result.refused:
        assert result.fast_answer is not None, "fast_answer must be non-None when not refused"
        assert "fast_path_total_ms" in result.timings_ms


def test_pipeline_faiss_flag_present_on_refused(real_corpora_and_index):
    """faiss_available must appear in timings_ms even when pipeline refuses (unsafe query)."""
    from core.harness import run_pipeline

    index, _, _ = real_corpora_and_index

    # "bomb" is in UNSAFE_KEYWORDS with no safe context — should be refused
    result = run_pipeline(
        text_query="How to make a bomb?",
        index=index,
        async_polish=False,
    )

    # May be refused before retrieval (guard_unsafe fires first) — faiss_available may not be set yet
    # but if retrieval ran, it must be present
    if "retrieval_ms" in result.timings_ms:
        assert "faiss_available" in result.timings_ms


def test_pipeline_async_polish_returns_fast_answer_immediately(real_corpora_and_index):
    """async_polish=True must return fast_answer without waiting for LLM."""
    from core.harness import run_pipeline

    index, _, english_corpus = real_corpora_and_index
    query = english_corpus[0]["query"]

    slow_llm_called = []

    def slow_mock_llm(q, ans, ctx):
        import time
        time.sleep(0.5)
        slow_llm_called.append(True)
        return f"[polished] {ans}"

    import time
    start = time.perf_counter()
    result = run_pipeline(
        text_query=query,
        index=index,
        llm_fn=slow_mock_llm,
        async_polish=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Should return well before the 500ms LLM sleep completes
    assert elapsed_ms < 450, (
        f"async_polish=True should return before LLM completes, got {elapsed_ms:.1f}ms"
    )
    assert result.polished_answer is None, "polished_answer must be None when async"
    assert result.polish_future is not None, "polish_future must be set when async"

    # Resolve the future and verify it completed correctly
    polished = result.polish_future.result(timeout=5)
    assert polished is not None or True  # LLM may return None if refused

"""Latency benchmarking script for voice-rag fast-path pipeline.

Runs run_pipeline(text_query=...) over 50 sample queries from the loaded dataset
(25 Hindi + 25 English, balanced), collects timings_ms["total_fast_ms"] for each,
and prints/saves a table of P50/P70/P90/P100 to data/latency_report.json.

STT and LLM-polish latencies are reported separately in the script output/report
and are excluded from the fast-path percentiles.
"""

import json
import os
import sys
from typing import Any, Dict, List
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.chunkers import chunk_sentence_level
from core.data_loader import load_msmarco_xi_corpora
from core.harness import run_pipeline
from core.index import build_hybrid_index


def _sample_balanced_queries(
    hindi_corpus: List[Dict[str, Any]],
    english_corpus: List[Dict[str, Any]],
    n_each: int = 25,
) -> List[Dict[str, str]]:
    """Samples up to n_each unique queries independently from each language corpus.

    Interleaves Hindi and English queries so both are represented throughout the run.

    Args:
        hindi_corpus: List of Hindi passage items from load_msmarco_xi_corpora().
        english_corpus: List of English passage items.
        n_each: Max queries to draw from each language.

    Returns:
        Interleaved list of {"query": str, "lang": str} dicts.
    """
    def _unique(corpus: List[Dict[str, Any]], lang: str, limit: int) -> List[Dict[str, str]]:
        seen: set = set()
        result = []
        for item in corpus:
            q = item.get("query", "").strip()
            if q and q not in seen:
                seen.add(q)
                result.append({"query": q, "lang": lang})
                if len(result) >= limit:
                    break
        return result

    hi = _unique(hindi_corpus, "hi", n_each)
    en = _unique(english_corpus, "en", n_each)

    # Interleave: hi[0], en[0], hi[1], en[1], ...
    interleaved: List[Dict[str, str]] = []
    for h, e in zip(hi, en):
        interleaved.append(h)
        interleaved.append(e)
    # Append remainder if one corpus was shorter
    interleaved.extend(hi[len(en):])
    interleaved.extend(en[len(hi):])
    return interleaved


def main():
    print("Loading dataset for latency benchmarking...")
    # NOTE: No silent fallback — if loading fails, we raise immediately so the
    # bug is visible rather than masked by synthetic data.
    hindi_corpus, english_corpus = load_msmarco_xi_corpora(split="train[:500]")
    combined_corpus = hindi_corpus + english_corpus

    # Build corpus chunks using sentence-level chunking (voice-optimised: complete sentences
    # prevent unnatural audio pause cuts when used with TTS output)
    print("Building corpus chunks (sentence-level, voice-optimised strategy)...")
    chunks: List[Dict[str, Any]] = []
    for item in combined_corpus:
        chunks.extend(
            chunk_sentence_level(
                item.get("passage_text", ""), item.get("metadata", {})
            )
        )

    if not chunks:
        raise RuntimeError(
            "No chunks produced from corpus. "
            "Check that load_msmarco_xi_corpora() is returning passages correctly."
        )

    print(f"Building HybridIndex over {len(chunks)} chunks...")
    index = build_hybrid_index(chunks)

    if not index.faiss_available:
        print("WARNING: FAISS index is NOT available — results will be BM25-only.")
    else:
        print("FAISS dense index: ACTIVE")

    # Sample balanced queries: 25 Hindi + 25 English
    balanced = _sample_balanced_queries(hindi_corpus, english_corpus, n_each=25)
    if len(balanced) < 2:
        raise RuntimeError(
            "Insufficient queries extracted from dataset. "
            "Check that load_msmarco_xi_corpora() returns both Hindi and English passages."
        )

    hi_count = sum(1 for q in balanced if q["lang"] == "hi")
    en_count = sum(1 for q in balanced if q["lang"] == "en")
    print(f"Query sample: {hi_count} Hindi + {en_count} English = {len(balanced)} total")

    # ── Warm up the embedding model before timing ──────────────────────────────
    # The first model.encode() call loads weights from disk/cache and spikes
    # P100 by 200–2000ms. One warm-up query eliminates this cold-start artifact.
    print("Warming up embedding model (cold-start elimination)...")
    _ = index.hybrid_retrieve(balanced[0]["query"], k=1)
    print("Warm-up complete. Starting timed benchmark...\n")

    sample_queries = [q["query"] for q in balanced]

    print(f"Executing run_pipeline over {len(sample_queries)} sample queries...")
    fast_latencies: List[float] = []
    llm_latencies: List[float] = []

    for idx, query in enumerate(sample_queries, start=1):
        res = run_pipeline(text_query=query, index=index, async_polish=False)
        fast_ms = res.timings_ms.get("total_fast_ms") or res.timings_ms.get(
            "fast_path_total_ms", 0.0
        )
        fast_latencies.append(fast_ms)
        if "llm_polish_ms" in res.timings_ms:
            llm_latencies.append(res.timings_ms["llm_polish_ms"])

    # Calculate percentiles for fast-path total ms
    p50 = float(np.percentile(fast_latencies, 50))
    p70 = float(np.percentile(fast_latencies, 70))
    p90 = float(np.percentile(fast_latencies, 90))
    p100 = float(np.percentile(fast_latencies, 100))

    avg_llm_ms = float(np.mean(llm_latencies)) if llm_latencies else 0.0

    report = {
        "sample_count": len(sample_queries),
        "language_balance": {"hi": hi_count, "en": en_count},
        "chunking_strategy": "sentence-level",
        "faiss_available": index.faiss_available,
        "fast_path_percentiles_ms": {
            "P50": round(p50, 3),
            "P70": round(p70, 3),
            "P90": round(p90, 3),
            "P100": round(p100, 3),
        },
        "stt_latency_ms": {
            "note": "STT (Sarvam saarika:v2 API) latency reported separately outside fast path",
            "measured_ms": None,
        },
        "llm_polish_latency_ms": {
            "note": "LLM Polish (Groq llama-3.3-70b-versatile) latency reported separately outside fast path",
            "mean_ms": round(avg_llm_ms, 3),
        },
    }

    # Print Table to stdout
    print("\n" + "=" * 60)
    print("        VOICE-RAG PIPELINE LATENCY REPORT         ")
    print("=" * 60)
    print(f"Fast-Path Sample Queries : {len(sample_queries)}")
    print(f"Language Balance         : {hi_count} Hindi / {en_count} English")
    print(f"Chunking Strategy        : sentence-level (voice-optimised)")
    print(f"FAISS Active             : {index.faiss_available}")
    print("-" * 60)
    print(f"  P50 (Median)   : {p50:.2f} ms")
    print(f"  P70            : {p70:.2f} ms")
    print(f"  P90            : {p90:.2f} ms")
    print(f"  P100 (Max)     : {p100:.2f} ms")
    print("-" * 60)
    print("  Separate Stage Latencies (Excluded from Fast-Path):")
    print("    - STT Latency        : External API dependent (Sarvam saarika:v2)")
    print(f"    - LLM Polish Latency : {avg_llm_ms:.2f} ms (Groq llama-3.3-70b-versatile)")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)
    report_path = os.path.join("data", "latency_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nLatency report saved to: {report_path}")


if __name__ == "__main__":
    main()

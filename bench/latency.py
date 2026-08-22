"""Latency benchmarking script for voice-rag fast-path pipeline.

Runs run_pipeline(text_query=...) over 50 sample queries from the loaded dataset,
collects timings_ms["total_fast_ms"] for each, and prints/saves a table of
P50/P70/P90/P100 to data/latency_report.json.

STT and LLM-polish latencies are reported separately in the script output/report
and are excluded from the fast-path percentiles.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.chunkers import chunk_passage_native
from core.data_loader import load_msmarco_xi_corpora
from core.harness import run_pipeline
from core.index import build_hybrid_index


def main():
    print("Loading dataset for latency benchmarking...")
    try:
        hindi_corpus, english_corpus = load_msmarco_xi_corpora()
        combined_corpus = hindi_corpus + english_corpus
    except Exception as exc:
        print(f"Dataset loading fallback: {exc}")
        combined_corpus = []

    # Extract up to 50 unique queries and build corpus chunks
    sample_queries: List[str] = []
    seen_queries = set()
    chunks: List[Dict[str, Any]] = []

    if combined_corpus:
        for item in combined_corpus:
            q = item.get("query", "").strip()
            if q and q not in seen_queries and len(sample_queries) < 50:
                seen_queries.add(q)
                sample_queries.append(q)
            # Create chunk with metadata
            chunks.extend(
                chunk_passage_native(
                    item.get("passage_text", ""), item.get("metadata", {})
                )
            )

    # Fallback sample queries if dataset is not available or has fewer queries
    if len(sample_queries) < 50:
        needed = 50 - len(sample_queries)
        fallback_queries = [
            f"What is the capital city of state {i}?" for i in range(needed)
        ]
        sample_queries.extend(fallback_queries)
        if not chunks:
            chunks = [
                {
                    "text": f"The capital city of state {i} is City{i}.",
                    "meta": {"query_id": str(i), "is_selected": 1},
                }
                for i in range(50)
            ]

    print(f"Building HybridIndex over {len(chunks)} chunks...")
    index = build_hybrid_index(chunks)

    print(f"Executing run_pipeline over {len(sample_queries)} sample queries...")
    fast_latencies: List[float] = []
    llm_latencies: List[float] = []

    for idx, query in enumerate(sample_queries, start=1):
        res = run_pipeline(text_query=query, index=index)
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
        "fast_path_percentiles_ms": {
            "P50": round(p50, 3),
            "P70": round(p70, 3),
            "P90": round(p90, 3),
            "P100": round(p100, 3),
        },
        "stt_latency_ms": {
            "note": "STT (Sarvam API) latency reported separately outside fast path",
            "measured_ms": None,
        },
        "llm_polish_latency_ms": {
            "note": "LLM Polish (Groq API) latency reported separately outside fast path",
            "mean_ms": round(avg_llm_ms, 3),
        },
    }

    # Print Table to stdout
    print("\n" + "=" * 55)
    print("          VOICE-RAG PIPELINE LATENCY REPORT          ")
    print("=" * 55)
    print(f"Fast-Path Sample Queries: {len(sample_queries)}")
    print("-" * 55)
    print(f"  P50 (Median)   : {p50:.2f} ms")
    print(f"  P70            : {p70:.2f} ms")
    print(f"  P90            : {p90:.2f} ms")
    print(f"  P100 (Max)     : {p100:.2f} ms")
    print("-" * 55)
    print("  Separate Stage Latencies (Excluded from Fast-Path):")
    print("    - STT Latency        : External API dependent (Sarvam)")
    print(f"    - LLM Polish Latency : {avg_llm_ms:.2f} ms (Groq llama-3.3-70b)")
    print("=" * 55)

    os.makedirs("data", exist_ok=True)
    report_path = os.path.join("data", "latency_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nLatency report saved to: {report_path}")


if __name__ == "__main__":
    main()

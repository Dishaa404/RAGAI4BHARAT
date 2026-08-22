"""Chunking strategy ablation benchmarking script.

Runs all 3 chunking strategies independently through retrieval on the same 30 queries,
computes recall@5 against is_selected ground truth for each strategy, and prints
a comparison table.
"""

import os
import sys
from typing import Any, Dict, List

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.chunkers import chunk_fixed_size, chunk_passage_native, chunk_sentence_level
from core.data_loader import load_msmarco_xi_corpora
from core.index import build_hybrid_index


def main():
    print("Loading dataset for chunking ablation benchmark...")
    # NOTE: No silent fallback — if loading fails, we raise immediately so the
    # bug is visible rather than masked by synthetic data.
    hindi_corpus, english_corpus = load_msmarco_xi_corpora()
    combined_corpus = hindi_corpus + english_corpus

    # Extract 30 queries that have at least one passage with is_selected == 1
    query_to_passages: Dict[str, List[Dict[str, Any]]] = {}
    query_ids_with_ground_truth = set()

    for item in combined_corpus:
        qid = item.get("metadata", {}).get("query_id")
        q_text = item.get("query", "").strip()
        is_sel = item.get("metadata", {}).get("is_selected", 0)

        if q_text and qid is not None:
            if q_text not in query_to_passages:
                query_to_passages[q_text] = []
            query_to_passages[q_text].append(item)
            if is_sel == 1:
                query_ids_with_ground_truth.add(q_text)

    test_queries = [q for q in query_to_passages.keys() if q in query_ids_with_ground_truth][:30]

    if len(test_queries) < 30:
        raise RuntimeError(
            f"Only {len(test_queries)} queries with ground-truth selected passages found "
            f"(need 30). Total passages loaded: {len(combined_corpus)}. "
            "Check that load_msmarco_xi_corpora() is returning real data with is_selected==1 rows."
        )

    # Collect all candidate passages across test queries
    all_passages: List[Dict[str, Any]] = []
    for q in test_queries:
        all_passages.extend(query_to_passages[q])

    strategies = [
        ("Passage-Native", chunk_passage_native),
        ("Fixed-Size Overlap", chunk_fixed_size),
        ("Sentence-Level", chunk_sentence_level),
    ]

    results = {}

    print(f"\nEvaluating 3 chunking strategies over {len(test_queries)} queries...")

    for name, chunk_fn in strategies:
        # Build chunks for all passages using ONLY current strategy
        strategy_chunks: List[Dict[str, Any]] = []
        for item in all_passages:
            strategy_chunks.extend(
                chunk_fn(item["passage_text"], item["metadata"])
            )

        # Build HybridIndex for this strategy
        print(f"Building index for '{name}' strategy ({len(strategy_chunks)} chunks)...")
        index = build_hybrid_index(strategy_chunks)

        hits = 0
        for q in test_queries:
            retrieved = index.hybrid_retrieve(q, k=5)
            # Check if any top-5 chunk matches ground truth (is_selected == 1)
            is_hit = any(
                c.get("meta", {}).get("is_selected") == 1
                for c in retrieved
            )
            if is_hit:
                hits += 1

        recall_at_5 = hits / len(test_queries) if test_queries else 0.0
        results[name] = {
            "chunk_count": len(strategy_chunks),
            "hits": hits,
            "total_queries": len(test_queries),
            "recall_at_5": recall_at_5,
        }

    # Print Comparison Table
    print("\n" + "=" * 65)
    print("       CHUNKING STRATEGY ABLATION STUDY (RECALL@5)       ")
    print("=" * 65)
    print(f"{'Strategy':<22} | {'Chunks':<10} | {'Hits@5':<10} | {'Recall@5':<10}")
    print("-" * 65)
    for name, stats in results.items():
        print(
            f"{name:<22} | {stats['chunk_count']:<10} | "
            f"{stats['hits']}/{stats['total_queries']:<6} | {stats['recall_at_5']:.4f}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()

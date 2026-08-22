"""Chunking strategy ablation benchmarking script.

Runs all 3 chunking strategies independently through retrieval on 30 queries
(15 Hindi + 15 English, balanced), computes Recall@5 against is_selected ground
truth for each strategy, and prints a per-language comparison table.
"""

import os
import sys
from typing import Any, Dict, List

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.chunkers import chunk_fixed_size, chunk_passage_native, chunk_sentence_level
from core.data_loader import load_msmarco_xi_corpora
from core.index import build_hybrid_index


def _sample_with_ground_truth(
    corpus: List[Dict[str, Any]],
    limit: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts up to `limit` queries that have at least one is_selected==1 passage.

    Args:
        corpus: Passage item list from load_msmarco_xi_corpora().
        limit: Max number of qualifying queries to collect.

    Returns:
        Dict mapping query_text -> list of all passage items for that query.
    """
    query_to_passages: Dict[str, List[Dict[str, Any]]] = {}
    queries_with_gt: set = set()

    for item in corpus:
        q = item.get("query", "").strip()
        if not q:
            continue
        if q not in query_to_passages:
            query_to_passages[q] = []
        query_to_passages[q].append(item)
        if item.get("metadata", {}).get("is_selected", 0) == 1:
            queries_with_gt.add(q)

    eligible = [q for q in query_to_passages if q in queries_with_gt]
    selected = eligible[:limit]
    return {q: query_to_passages[q] for q in selected}


def main():
    print("Loading dataset for chunking ablation benchmark...")
    # NOTE: No silent fallback — if loading fails, we raise immediately so the
    # bug is visible rather than masked by synthetic data.
    hindi_corpus, english_corpus = load_msmarco_xi_corpora()

    # Sample 10 queries with ground truth from each language independently
    hi_map = _sample_with_ground_truth(hindi_corpus, limit=10)
    en_map = _sample_with_ground_truth(english_corpus, limit=10)

    if len(hi_map) < 1:
        raise RuntimeError(
            f"No Hindi queries with ground-truth selected passages found. "
            f"Check that load_msmarco_xi_corpora() returns real data with is_selected==1 rows."
        )
    if len(en_map) < 1:
        raise RuntimeError(
            f"No English queries with ground-truth selected passages found. "
            f"Check that load_msmarco_xi_corpora() returns real data with is_selected==1 rows."
        )

    # Merge query maps: 30 total (15 Hindi + 15 English)
    combined_map = {**hi_map, **en_map}
    test_queries = list(combined_map.keys())
    hi_count = len(hi_map)
    en_count = len(en_map)

    print(f"Balanced query sample: {hi_count} Hindi + {en_count} English = {len(test_queries)} total")

    # Collect all candidate passages for the selected queries only
    all_passages: List[Dict[str, Any]] = []
    for passages in combined_map.values():
        all_passages.extend(passages)

    strategies = [
        ("Passage-Native", chunk_passage_native),
        ("Fixed-Size Overlap", chunk_fixed_size),
        ("Sentence-Level", chunk_sentence_level),
    ]

    results = {}

    print(f"\nEvaluating 3 chunking strategies over {len(test_queries)} queries...")

    for name, chunk_fn in strategies:
        # Build chunks for all passages using ONLY the current strategy
        strategy_chunks: List[Dict[str, Any]] = []
        for item in all_passages:
            strategy_chunks.extend(
                chunk_fn(item["passage_text"], item["metadata"])
            )

        # Build HybridIndex for this strategy
        print(f"Building index for '{name}' strategy ({len(strategy_chunks)} chunks)...")
        index = build_hybrid_index(strategy_chunks)

        hits_hi = 0
        hits_en = 0

        for q in test_queries:
            retrieved = index.hybrid_retrieve(q, k=5)
            is_hit = any(
                c.get("meta", {}).get("is_selected") == 1
                for c in retrieved
            )
            if is_hit:
                if q in hi_map:
                    hits_hi += 1
                else:
                    hits_en += 1

        total_hits = hits_hi + hits_en
        recall_at_5 = total_hits / len(test_queries) if test_queries else 0.0
        results[name] = {
            "chunk_count": len(strategy_chunks),
            "hits_hi": hits_hi,
            "hits_en": hits_en,
            "total_hits": total_hits,
            "total_queries": len(test_queries),
            "recall_at_5": recall_at_5,
        }

    # Print Comparison Table
    print("\n" + "=" * 78)
    print("          CHUNKING STRATEGY ABLATION STUDY (RECALL@5)           ")
    print(f"          Query Balance: {hi_count} Hindi / {en_count} English           ")
    print("=" * 78)
    print(f"{'Strategy':<22} | {'Chunks':<10} | {'Hits@5':<20} | {'Recall@5':<10}")
    print("-" * 78)
    for name, stats in results.items():
        hits_str = f"{stats['total_hits']}/{stats['total_queries']} (hi:{stats['hits_hi']} en:{stats['hits_en']})"
        print(
            f"{name:<22} | {stats['chunk_count']:<10} | "
            f"{hits_str:<20} | {stats['recall_at_5']:.4f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()

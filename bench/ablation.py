"""Leakage-free chunking strategy ablation benchmark."""

import os
import random
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.chunkers import chunk_fixed_size, chunk_passage_native, chunk_sentence_level
from core.data_loader import load_msmarco_xi_corpora
from core.index import build_hybrid_index


CorpusItem = Dict[str, Any]
EvaluationQuery = Dict[str, Any]
Chunker = Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]


def select_evaluation_queries(
    corpus: Sequence[CorpusItem], count: int = 30, seed: int = 42
) -> List[EvaluationQuery]:
    """Select deterministic, language-specific queries with selected passages."""
    candidates: Dict[Tuple[Any, str], EvaluationQuery] = {}
    for item in corpus:
        metadata = item.get("metadata", {})
        query_id = metadata.get("query_id")
        language = metadata.get("language", "unknown")
        query = str(item.get("query", "")).strip()
        if query and query_id is not None and metadata.get("is_selected") == 1:
            candidates.setdefault(
                (query_id, language), {"query_id": query_id, "language": language, "query": query}
            )

    ordered = sorted(
        candidates.values(),
        key=lambda item: (str(item["language"]), str(item["query_id"]), item["query"]),
    )
    if len(ordered) < count:
        raise RuntimeError(
            f"Only {len(ordered)} evaluation queries with selected passages found; need {count}."
        )
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[:count]


def build_strategy_chunks(
    corpus: Sequence[CorpusItem], chunk_fn: Chunker
) -> List[Dict[str, Any]]:
    """Chunk every corpus passage; evaluation queries never filter this input."""
    chunks: List[Dict[str, Any]] = []
    for item in corpus:
        chunks.extend(chunk_fn(item["passage_text"], item["metadata"]))
    return chunks


def evaluate_strategy(
    corpus: Sequence[CorpusItem],
    evaluation_queries: Sequence[EvaluationQuery],
    chunk_fn: Chunker,
) -> Dict[str, Any]:
    """Build a full-corpus index and calculate Recall@5 and MRR@5."""
    strategy_chunks = build_strategy_chunks(corpus, chunk_fn)
    index = build_hybrid_index(strategy_chunks)
    hits = 0
    reciprocal_ranks: List[float] = []

    for evaluation_query in evaluation_queries:
        retrieved = index.hybrid_retrieve(evaluation_query["query"], k=5)
        hit_rank = None
        for rank, chunk in enumerate(retrieved, start=1):
            metadata = chunk.get("meta", {})
            if (
                metadata.get("query_id") == evaluation_query["query_id"]
                and metadata.get("language") == evaluation_query["language"]
                and metadata.get("is_selected") == 1
            ):
                hit_rank = rank
                break
        reciprocal_ranks.append(1.0 / hit_rank if hit_rank is not None else 0.0)
        if hit_rank is not None:
            hits += 1

    query_count = len(evaluation_queries)
    return {
        "corpus_passage_count": len(corpus),
        "chunk_count": len(strategy_chunks),
        "evaluation_query_count": query_count,
        "hits_at_5": hits,
        "recall_at_5": hits / query_count if query_count else 0.0,
        "mrr_at_5": sum(reciprocal_ranks) / query_count if query_count else 0.0,
    }


def main() -> None:
    print("Loading full MSMARCO-XI corpus for chunking ablation benchmark...")
    hindi_corpus, english_corpus = load_msmarco_xi_corpora()
    full_corpus = hindi_corpus + english_corpus
    evaluation_queries = select_evaluation_queries(full_corpus)

    strategies = [
        ("Passage-Native", chunk_passage_native),
        ("Fixed-Size Overlap", chunk_fixed_size),
        ("Sentence-Level", chunk_sentence_level),
    ]
    results: Dict[str, Dict[str, Any]] = {}
    for name, chunk_fn in strategies:
        print(f"Building full-corpus index for '{name}'...")
        results[name] = evaluate_strategy(full_corpus, evaluation_queries, chunk_fn)

    print("\n" + "=" * 95)
    print("       LEAKAGE-FREE CHUNKING ABLATION STUDY (RECALL@5)       ")
    print("=" * 95)
    print(
        f"Corpus passages: {len(full_corpus)} | Evaluation queries: {len(evaluation_queries)}"
    )
    print("-" * 95)
    print(f"{'Strategy':<22} | {'Chunks':<10} | {'Hits@5':<8} | {'Recall@5':<10} | {'MRR@5':<8}")
    print("-" * 95)
    for name, stats in results.items():
        print(
            f"{name:<22} | {stats['chunk_count']:<10} | {stats['hits_at_5']:<8} | "
            f"{stats['recall_at_5']:.4f}     | {stats['mrr_at_5']:.4f}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()

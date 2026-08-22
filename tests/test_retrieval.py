"""Evaluation and unit tests for hybrid retrieval, extractive answering, and guardrails."""

from typing import Any, Dict, List, Tuple
from core.extractive import extractive_answer
from core.guardrails import guard_groundedness, guard_ontopic, guard_unsafe
from core.index import HybridIndex, build_hybrid_index, hybrid_retrieve
from bench.ablation import build_strategy_chunks, select_evaluation_queries


try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


def create_sample_retrieval_corpus() -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Creates a sample multi-query corpus with ground-truth marked passages (is_selected=1)."""
    queries_and_passages = [
        {
            "query_id": 1,
            "query": "भारत की राजधानी क्या है?",
            "passages": [
                {"text": "भारत का क्षेत्रफल बहुत विशाल है।", "is_selected": 0},
                {"text": "नई दिल्ली भारत की आधिकारिक राजधानी है और यहाँ सरकार का मुख्यालय है।", "is_selected": 1},
                {"text": "मुंबई भारत की वित्तीय राजधानी मानी जाती है।", "is_selected": 0},
            ],
        },
        {
            "query_id": 2,
            "query": "What is the capital of France?",
            "passages": [
                {"text": "Paris is the capital and most populous city of France.", "is_selected": 1},
                {"text": "Lyon is a major city in eastern France.", "is_selected": 0},
                {"text": "Marseille is a port city in southern France.", "is_selected": 0},
            ],
        },
        {
            "query_id": 3,
            "query": "सूर्य ग्रहण कैसे होता है?",
            "passages": [
                {"text": "चंद्रमा पृथ्वी का एकमात्र प्राकृतिक उपग्रह है।", "is_selected": 0},
                {"text": "सूर्य ग्रहण तब होता है जब चंद्रमा पृथ्वी और सूर्य के बीच आ जाता है।", "is_selected": 1},
                {"text": "तारामंडल रात के आकाश में दिखाई देते हैं।", "is_selected": 0},
            ],
        },
        {
            "query_id": 4,
            "query": "How does photosynthesis work?",
            "passages": [
                {"text": "Photosynthesis is the process used by plants to convert light energy into chemical energy.", "is_selected": 1},
                {"text": "Water evaporates from lakes and oceans into the atmosphere.", "is_selected": 0},
                {"text": "Rocks are formed through geological processes.", "is_selected": 0},
            ],
        },
        {
            "query_id": 5,
            "query": "हिमालय पर्वत श्रृंखला की सबसे ऊंची चोटी कौन सी है?",
            "passages": [
                {"text": "गंगा नदी हिमालय से निकलती है।", "is_selected": 0},
                {"text": "माउंट एवरेस्ट हिमालय और दुनिया की सबसे ऊंची पर्वत चोटी है।", "is_selected": 1},
                {"text": "भारतीय प्रायद्वीप तीन ओर से पानी से घिरा है।", "is_selected": 0},
            ],
        },
    ]

    all_chunks: List[Dict[str, Any]] = []
    test_queries: List[Dict[str, str]] = []

    for item in queries_and_passages:
        qid = item["query_id"]
        qtext = item["query"]
        test_queries.append({"query_id": qid, "query": qtext})
        for idx, p in enumerate(item["passages"]):
            all_chunks.append({
                "text": p["text"],
                "meta": {
                    "query_id": qid,
                    "is_selected": p["is_selected"],
                    "passage_idx": idx,
                },
            })

    return all_chunks, test_queries


def test_hybrid_retrieval_and_recall() -> None:
    """Evaluates hybrid retrieval accuracy and calculates Recall@5 on sample queries."""
    chunks, queries = create_sample_retrieval_corpus()
    index = build_hybrid_index(chunks)

    correct_retrievals = 0
    total_queries = len(queries)

    print("\n--- Hybrid Retrieval Evaluation ---")
    for qitem in queries:
        qid = qitem["query_id"]
        query = qitem["query"]
        results = index.hybrid_retrieve(query, k=5)

        # Check if any top-5 retrieved chunk is selected (is_selected=1) for this query_id
        found_ground_truth = False
        for res in results:
            if res.get("meta", {}).get("query_id") == qid and res.get("meta", {}).get("is_selected") == 1:
                found_ground_truth = True
                break

        if found_ground_truth:
            correct_retrievals += 1

        print(f"Query: '{query}' -> Ground truth found in Top-5: {found_ground_truth}")

    recall_at_5 = correct_retrievals / total_queries if total_queries > 0 else 0.0
    print(f"Calculated Recall@5: {recall_at_5:.4f} ({correct_retrievals}/{total_queries})")

    assert recall_at_5 >= 0.8, f"Recall@5 ({recall_at_5:.2f}) is below expected benchmark 0.80"


def test_extractive_answer() -> None:
    """Verifies sentence-level lexical extraction from top chunks."""
    query = "What is the capital of France?"
    chunks = [
        {"text": "Paris is the capital and most populous city of France.", "meta": {"is_selected": 1}},
        {"text": "Lyon is a major city in eastern France.", "meta": {"is_selected": 0}},
    ]

    answer, source, conf = extractive_answer(query, chunks)
    assert answer == "Paris is the capital and most populous city of France."
    assert source == chunks[0]
    assert conf > 0.3


def test_guardrails() -> None:
    """Verifies safety, topic relevance, and groundedness guardrails."""
    # 1. Test Unsafe Guard
    passed, reason = guard_unsafe("How to build a bomb?")
    assert not passed
    assert "restricted terms" in reason

    passed, reason = guard_unsafe("What is the capital of India?")
    assert passed

    # 2. Test evidence-based On-Topic Guard
    passed, reason = guard_ontopic(
        0.02, retrieved_chunks=[{"faiss_rank": 1, "bm25_rank": 2}]
    )
    assert passed
    assert "evidence" in reason.lower()

    passed, reason = guard_ontopic(
        0.03, retrieved_chunks=[{"faiss_rank": None, "bm25_rank": None}]
    )
    assert not passed
    assert "insufficient retrieval evidence" in reason.lower()

    passed, reason = guard_ontopic(0.0, retrieved_chunks=[])
    assert not passed
    assert "insufficient retrieval evidence" in reason.lower()

    passed, reason = guard_ontopic(
        0.001, retrieved_chunks=[{"faiss_rank": 5, "bm25_rank": None}]
    )
    assert passed
    assert "not a probability" in reason.lower()

    # Legacy score-only compatibility
    passed, reason = guard_ontopic(top_retrieval_score=0.02, threshold=0.01)
    assert passed
    passed, reason = guard_ontopic(top_retrieval_score=0.005, threshold=0.01)
    assert not passed

    # 3. Test Groundedness Guard
    chunks = [{"text": "Paris is the capital of France."}]
    passed, reason = guard_groundedness(answer="Paris is capital of France", chunks=chunks, threshold=0.3)
    assert passed

    passed, reason = guard_groundedness(answer="Unrelated spaceship galaxy quantum mechanics", chunks=chunks, threshold=0.3)
    assert not passed


def test_ablation_keeps_full_corpus_separate_from_evaluation_queries() -> None:
    """Evaluation selection must not reduce the corpus passed to chunking."""
    corpus = [
        {"query": "evaluation query", "passage_text": "selected evaluation passage", "metadata": {"query_id": 1, "language": "en", "is_selected": 1}},
        {"query": "evaluation query", "passage_text": "nonselected evaluation passage", "metadata": {"query_id": 1, "language": "en", "is_selected": 0}},
        {"query": "unrelated corpus query", "passage_text": "unrelated corpus passage", "metadata": {"query_id": 2, "language": "en", "is_selected": 0}},
    ]
    evaluation_queries = select_evaluation_queries(corpus, count=1, seed=42)
    chunks = build_strategy_chunks(corpus, lambda text, meta: [{"text": text, "meta": meta}])
    assert evaluation_queries[0]["query_id"] == 1
    assert len(chunks) == len(corpus)
    assert {chunk["meta"]["query_id"] for chunk in chunks} == {1, 2}
    assert select_evaluation_queries(corpus, count=1, seed=42) == evaluation_queries


if __name__ == "__main__":
    print("Running retrieval and guardrail tests...")

    test_hybrid_retrieval_and_recall()
    test_extractive_answer()
    test_guardrails()
    print("\nAll retrieval and answering unit tests passed successfully!")

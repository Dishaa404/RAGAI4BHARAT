"""Data loader module for AI4Bharat MSMARCO-XI dataset.

Loads train[:3000] of the 'hi' split and constructs Hindic and English corpora.
"""

from typing import Any, Dict, List, Tuple


def _parse_passages(passages_raw: Any) -> List[Dict[str, Any]]:
    """Normalizes passage structure to a list of dicts with text and selection status."""
    parsed: List[Dict[str, Any]] = []
    if isinstance(passages_raw, dict):
        texts = passages_raw.get("passage_text", [])
        selected_flags = passages_raw.get("is_selected", [])
        for idx, text in enumerate(texts):
            is_sel = selected_flags[idx] if idx < len(selected_flags) else 0
            parsed.append({"passage_text": text, "is_selected": is_sel})
    elif isinstance(passages_raw, list):
        for item in passages_raw:
            if isinstance(item, dict):
                parsed.append({
                    "passage_text": item.get("passage_text", ""),
                    "is_selected": item.get("is_selected", 0),
                })
    return parsed


def load_msmarco_xi_corpora(
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "train[:3000]",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads MSMARCO-XI Hindi split and constructs Hindi and English corpora.

    Args:
        dataset_name: HuggingFace dataset path.
        split: Dataset split to load (defaults to train[:3000]).

    Returns:
        Tuple containing (hindi_corpus, english_corpus).
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError(
            "The 'datasets' package is required. Install via 'pip install -r requirements.txt'."
        ) from err

    dataset = load_dataset(dataset_name, "hi", split=split)
    hindi_corpus: List[Dict[str, Any]] = []
    english_corpus: List[Dict[str, Any]] = []

    for row in dataset:
        qid = row.get("query_id")
        qtype = row.get("query_type", "unknown")

        # Process Hindi Passages
        hi_passages = _parse_passages(row.get("Translated_passages"))
        for item in hi_passages:
            hindi_corpus.append({
                "query": row.get("query", ""),
                "passage_text": item["passage_text"],
                "metadata": {
                    "query_id": qid,
                    "query_type": qtype,
                    "is_selected": item["is_selected"],
                    "language": "hi",
                },
            })

        # Process English Passages
        en_passages = _parse_passages(row.get("English_passages"))
        for item in en_passages:
            english_corpus.append({
                "query": row.get("Eng_Query", ""),
                "passage_text": item["passage_text"],
                "metadata": {
                    "query_id": qid,
                    "query_type": qtype,
                    "is_selected": item["is_selected"],
                    "language": "en",
                },
            })

    return hindi_corpus, english_corpus

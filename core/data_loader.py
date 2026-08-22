"""Data loader module for AI4Bharat MSMARCO-XI dataset.

Loads train[:3000] of the 'hi' split and constructs Hindi and English corpora.
"""

from typing import Any, Dict, List, Tuple


def load_msmarco_xi_corpora(
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "train[:3000]",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads MSMARCO-XI Hindi split and constructs Hindi and English corpora.

    Real dataset schema (ai4bharat/MSMARCO-XI, 'hi' split):
        row = {
            "query": str,
            "Eng_Query": str,
            "query_id": int,
            "query_type": str,
            "passages": {
                "is_selected": [int, ...],
                "English_passages": [str, ...],
                "Translated_passages": [str, ...],
            },
        }
    All three lists under "passages" are aligned by index.

    Args:
        dataset_name: HuggingFace dataset path.
        split: Dataset split to load (defaults to train[:3000]).

    Returns:
        Tuple of (hindi_corpus, english_corpus). Each entry has the shape:
            {
                "query": str,
                "passage_text": str,
                "metadata": {"query_id": int, "query_type": str, "is_selected": int, "language": str},
            }

    Raises:
        ImportError: If the 'datasets' package is not installed.
        RuntimeError: If the dataset fails to load or returns no rows.
        KeyError: If a row is missing the expected "passages" structure.
    """
    import os
    import json

    dataset = None
    local_sample = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_msmarco.json"))

    if os.path.exists(local_sample):
        print(f"Loading local sample dataset: {local_sample}")
        with open(local_sample, encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split=split)
        except Exception as exc:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {exc}") from exc

    if len(dataset) == 0:
        raise RuntimeError(
            f"Dataset '{dataset_name}' (split='{split}') loaded 0 rows. "
            "Check your HuggingFace credentials or the dataset availability."
        )

    hindi_corpus: List[Dict[str, Any]] = []
    english_corpus: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(dataset):
        qid = row.get("query_id")
        qtype = row.get("query_type", "unknown")

        passages = row.get("passages")
        if not isinstance(passages, dict):
            raise KeyError(
                f"Row {row_idx} (query_id={qid!r}) is missing the 'passages' dict. "
                f"Got type {type(passages).__name__!r}. "
                "The dataset schema may have changed — verify the HuggingFace dataset card."
            )

        is_selected_list: List[int] = passages.get("is_selected", [])
        translated_passages: List[str] = passages.get("Translated_passages", [])
        english_passages: List[str] = passages.get("English_passages", [])

        # Hindi corpus — Translated_passages aligned with is_selected
        for idx, text in enumerate(translated_passages):
            is_sel = is_selected_list[idx] if idx < len(is_selected_list) else 0
            hindi_corpus.append({
                "query": row.get("query", ""),
                "passage_text": text,
                "metadata": {
                    "query_id": qid,
                    "query_type": qtype,
                    "is_selected": is_sel,
                    "language": "hi",
                },
            })

        # English corpus — English_passages aligned with is_selected
        for idx, text in enumerate(english_passages):
            is_sel = is_selected_list[idx] if idx < len(is_selected_list) else 0
            english_corpus.append({
                "query": row.get("Eng_Query", ""),
                "passage_text": text,
                "metadata": {
                    "query_id": qid,
                    "query_type": qtype,
                    "is_selected": is_sel,
                    "language": "en",
                },
            })

    if not hindi_corpus and not english_corpus:
        raise RuntimeError(
            f"Dataset '{dataset_name}' loaded {len(dataset)} rows but produced 0 passages. "
            "Every row had empty 'passages' lists — check the schema or dataset version."
        )

    return hindi_corpus, english_corpus

"""Data loader module for AI4Bharat MSMARCO-XI dataset.

Loads up to 3000 Hindi rows from the default config and constructs Hindi and English corpora.
"""

import re
from typing import Any, Dict, Iterable, List, Tuple


def load_msmarco_xi_corpora(
    dataset_name: str = "ai4bharat/MSMARCO-XI",
    split: str = "train[:3000]",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads MSMARCO-XI Hindi rows and constructs Hindi and English corpora.

    Real dataset schema (ai4bharat/MSMARCO-XI, default config):
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

    dataset_rows: List[Dict[str, Any]] = []
    dataset = None
    local_sample = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample_msmarco.json"))

    if os.path.exists(local_sample):
        print(f"Loading local sample dataset: {local_sample}")
        with open(local_sample, encoding="utf-8") as f:
            dataset_rows = json.load(f)
    else:
        try:
            from datasets import load_dataset
            split_match = re.fullmatch(r"(?P<name>[^\[]+)(?:\[:(?P<limit>\d+)\])?", split)
            dataset_split = split_match.group("name") if split_match else split
            requested_rows = int(split_match.group("limit")) if split_match and split_match.group("limit") else 3000
            dataset = load_dataset(dataset_name, "default", split=dataset_split, streaming=True)
            for row in dataset:
                target_lang = str(row.get("target_lang", "")).lower()
                if target_lang not in {"hi", "hin", "hin_deva"} and not target_lang.startswith("hin_"):
                    continue
                dataset_rows.append(row)
                if len(dataset_rows) >= requested_rows:
                    break
        except Exception as err:
            raise RuntimeError(
                f"Unable to load dataset '{dataset_name}' (config='default', split='{split}'). "
                "Check network access, the dataset name/configuration, and HuggingFace credentials."
            ) from err

    if not dataset_rows:
        raise RuntimeError(
            f"Dataset '{dataset_name}' (split='{split}') loaded 0 rows. "
            "Check your HuggingFace credentials or the dataset availability."
        )

    hindi_corpus: List[Dict[str, Any]] = []
    english_corpus: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(dataset_rows):
        qid = row.get("query_id")
        qtype = row.get("query_type", "unknown")

        passages = row.get("passages")
        if not isinstance(passages, dict):
            raise KeyError(
                f"Row {row_idx} (query_id={qid!r}) is missing the 'passages' dict. "
                f"Got type {type(passages).__name__!r}. "
                "The dataset schema may have changed — verify the HuggingFace dataset card."
            )

        is_selected_list = passages.get("is_selected", [])
        translated_passages = passages.get("Translated_passages", [])
        english_passages = passages.get("English_passages", [])
        if not all(
            isinstance(value, list)
            for value in (is_selected_list, translated_passages, english_passages)
        ):
            raise KeyError(
                f"Row {row_idx} (query_id={qid!r}) has invalid passage fields. "
                "Expected lists for is_selected, English_passages, and Translated_passages."
            )

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
            f"Dataset '{dataset_name}' loaded {len(dataset_rows)} rows but produced 0 passages. "
            "Every row had empty 'passages' lists — check the schema or dataset version."
        )

    return hindi_corpus, english_corpus

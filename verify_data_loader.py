"""Throwaway verification script — confirms data_loader.py schema fix against real data.

Run from voice-rag/: python verify_data_loader.py
Not part of the permanent codebase.

NOTE: First run will download the dataset from HuggingFace (~slow on unauthenticated).
      Set HF_TOKEN env var for faster downloads, or run with a VPN/stable connection.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datasets import load_dataset

print("Loading train[:3] of ai4bharat/MSMARCO-XI ('hi' config)...")
print("(First run downloads dataset — may take a few minutes)\n")
ds = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train[:3]")

print(f"--- Row count: {len(ds)} ---")
print(f"Column names: {ds.column_names}\n")

row = ds[0]
print("=== Full example row (index 0) ===")
for k, v in row.items():
    if isinstance(v, dict):
        print(f"  {k!r} (dict) keys: {list(v.keys())}")
        for sk, sv in v.items():
            if isinstance(sv, list):
                print(f"    {sk!r} (list len={len(sv)}): first={repr(sv[0])[:80] if sv else '[]'}")
            else:
                print(f"    {sk!r}: {repr(sv)[:100]}")
    elif isinstance(v, list):
        print(f"  {k!r} (list len={len(v)}): first={repr(v[0])[:80] if v else '[]'}")
    else:
        print(f"  {k!r}: {repr(v)[:100]}")

print("\n=== Testing load_msmarco_xi_corpora() with split=train[:3] ===")
from core.data_loader import load_msmarco_xi_corpora
hindi_corpus, english_corpus = load_msmarco_xi_corpora(split="train[:3]")
print(f"  Hindi corpus entries : {len(hindi_corpus)}")
print(f"  English corpus entries: {len(english_corpus)}")

assert len(hindi_corpus) > 0, "FAIL: Hindi corpus is empty!"
assert len(english_corpus) > 0, "FAIL: English corpus is empty!"

h = hindi_corpus[0]
e = english_corpus[0]
assert "query" in h and "passage_text" in h and "metadata" in h, "FAIL: Hindi entry missing fields"
assert "query" in e and "passage_text" in e and "metadata" in e, "FAIL: English entry missing fields"
assert isinstance(h["passage_text"], str) and len(h["passage_text"]) > 0, "FAIL: Hindi passage_text is empty"
assert isinstance(e["passage_text"], str) and len(e["passage_text"]) > 0, "FAIL: English passage_text is empty"
assert "is_selected" in h["metadata"], "FAIL: is_selected missing from Hindi metadata"

print(f"\n  First Hindi entry:")
print(f"    query        : {h['query'][:80]!r}")
print(f"    passage_text : {h['passage_text'][:100]!r}")
print(f"    metadata     : {h['metadata']}")
print(f"\n  First English entry:")
print(f"    query        : {e['query'][:80]!r}")
print(f"    passage_text : {e['passage_text'][:100]!r}")
print(f"    metadata     : {e['metadata']}")

print("\n✓ All assertions passed — schema fix is correct against real data.")

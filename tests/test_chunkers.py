"""Unit tests for voice-rag chunkers module."""

from typing import Any, Dict
from core.chunkers import (
    chunk_all,
    chunk_fixed_size,
    chunk_passage_native,
    chunk_sentence_level,
)

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


def get_sample_meta() -> Dict[str, Any]:
    return {
        "query_id": 101,
        "query_type": "description",
        "is_selected": 1,
        "language": "hi",
    }


if HAS_PYTEST:
    @pytest.fixture
    def sample_meta() -> Dict[str, Any]:
        return get_sample_meta()


def test_empty_text_handling(sample_meta: Dict[str, Any] = None) -> None:
    """Ensure no strategy crashes on empty or whitespace-only text."""
    meta = sample_meta or get_sample_meta()
    for text in ["", "   ", "\n\t"]:
        assert chunk_passage_native(text, meta) == []
        assert chunk_fixed_size(text, meta) == []
        assert chunk_sentence_level(text, meta) == []
        assert chunk_all(text, meta) == []


def test_passage_native_chunker(sample_meta: Dict[str, Any] = None) -> None:
    """Verify passage-native strategy returns intact text and metadata."""
    meta = sample_meta or get_sample_meta()
    text = "यह एक परीक्षण वाक्य है।"
    res = chunk_passage_native(text, meta)
    assert len(res) == 1
    assert res[0]["text"] == text
    assert res[0]["meta"] == meta
    assert res[0]["strategy"] == "passage-native"


def test_fixed_size_chunker(sample_meta: Dict[str, Any] = None) -> None:
    """Verify fixed-size chunking divides text and retains metadata."""
    meta = sample_meta or get_sample_meta()
    text = "A" * 500
    res = chunk_fixed_size(text, meta, chunk_size=200, overlap=40)
    assert len(res) > 1
    for chunk in res:
        assert "text" in chunk
        assert chunk["meta"] == meta
        assert chunk["strategy"] == "fixed-size-overlap"


def test_sentence_level_chunker(sample_meta: Dict[str, Any] = None) -> None:
    """Verify sentence-level chunking handles Hindi and English delimiters."""
    meta = sample_meta or get_sample_meta()
    text = "पहला वाक्य। दूसरा वाक्य! Third sentence?"
    res = chunk_sentence_level(text, meta)
    assert len(res) == 3
    assert res[0]["text"] == "पहला वाक्य।"
    assert res[1]["text"] == "दूसरा वाक्य!"
    assert res[2]["text"] == "Third sentence?"
    for chunk in res:
        assert chunk["meta"] == meta
        assert chunk["strategy"] == "sentence-level"


def test_chunk_all(sample_meta: Dict[str, Any] = None) -> None:
    """Verify chunk_all executes all three strategies and returns unified output."""
    meta = sample_meta or get_sample_meta()
    text = "नमस्ते दुनिया। AI is awesome!"
    res = chunk_all(text, meta)
    strategies = {c["strategy"] for c in res}
    assert strategies == {
        "passage-native",
        "fixed-size-overlap",
        "sentence-level",
    }


if __name__ == "__main__":
    meta_data = get_sample_meta()
    test_empty_text_handling(meta_data)
    test_passage_native_chunker(meta_data)
    test_fixed_size_chunker(meta_data)
    test_sentence_level_chunker(meta_data)
    test_chunk_all(meta_data)
    print("All chunker unit tests passed successfully!")


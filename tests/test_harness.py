"""Unit tests for voice-rag orchestration harness, STT, and LLM polishing modules."""

import os
import pytest
from unittest.mock import MagicMock, patch

from core.harness import PipelineResult, run_pipeline
from core.llm import polish_answer
from core.stt import AudioFileNotFoundError, MissingAPIKeyError, STTError, transcribe


# --- STT Module Tests ---

def test_stt_missing_api_key(monkeypatch):
    """Test transcribe raises MissingAPIKeyError when SARVAM_API_KEY is not set."""
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        transcribe("dummy_path.wav")


def test_stt_missing_audio_file(monkeypatch, tmp_path):
    """Test transcribe raises AudioFileNotFoundError for non-existent file."""
    monkeypatch.setenv("SARVAM_API_KEY", "mock_key")
    non_existent_file = str(tmp_path / "missing.wav")
    with pytest.raises(AudioFileNotFoundError):
        transcribe(non_existent_file)


def test_stt_retry_and_final_failure(monkeypatch, tmp_path):
    """Test transcribe retries 3 times before raising STTError on persistent failure."""
    monkeypatch.setenv("SARVAM_API_KEY", "mock_key")
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"dummy audio content")

    call_count = 0

    def mock_failing_api(path, key):
        nonlocal call_count
        call_count += 1
        raise STTError("API connection timeout")

    with patch("core.stt._perform_sarvam_transcription", side_effect=mock_failing_api):
        with pytest.raises(STTError) as exc_info:
            transcribe(str(audio_file))
        assert "API connection timeout" in str(exc_info.value)
        assert call_count == 3


def test_stt_success(monkeypatch, tmp_path):
    """Test successful transcription returns text."""
    monkeypatch.setenv("SARVAM_API_KEY", "mock_key")
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"dummy audio content")

    with patch("core.stt._perform_sarvam_transcription", return_value="Delhi is the capital of India."):
        result = transcribe(str(audio_file))
        assert result == "Delhi is the capital of India."


# --- LLM Module Tests ---

def test_llm_polish_missing_api_key(monkeypatch):
    """Test polish_answer returns None when GROQ_API_KEY is missing."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    res = polish_answer("What is capital?", "Delhi", [{"text": "Delhi is capital"}])
    assert res is None


def test_llm_polish_handles_exceptions_gracefully(monkeypatch):
    """Test polish_answer catches exceptions (e.g., timeout/network error) and returns None."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")

    def mock_groq_init(*args, **kwargs):
        raise RuntimeError("Network timeout")

    with patch("sys.modules", {**os.sys.modules, "groq": MagicMock(Groq=mock_groq_init)}):
        res = polish_answer("Query", "Answer", [{"text": "Chunk text"}])
        assert res is None


def test_llm_polish_success(monkeypatch):
    """Test successful Groq response returns polished text."""
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")

    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="Delhi is the official capital city of India."))
    ]
    mock_client.chat.completions.create.return_value = mock_completion
    mock_groq_cls = MagicMock(return_value=mock_client)

    with patch("sys.modules", {**os.sys.modules, "groq": MagicMock(Groq=mock_groq_cls)}):
        res = polish_answer("What is the capital of India?", "Delhi is the capital.", [{"text": "Delhi is capital"}])
        assert res == "Delhi is the official capital city of India."


# --- Harness Pipeline Tests ---

def test_pipeline_text_query_success():
    """Test run_pipeline with text query returns fast_answer and per-stage timings."""
    sample_chunks = [
        {"text": "Python is a high-level programming language created by Guido van Rossum.", "meta": {}},
        {"text": "Java is another popular object-oriented language.", "meta": {}},
    ]

    mock_llm = lambda q, ans, chunks: "Python was created by Guido van Rossum as a high-level language."

    result = run_pipeline(
        text_query="Who created Python programming language?",
        chunks_source=sample_chunks,
        llm_fn=mock_llm,
    )

    assert isinstance(result, PipelineResult)
    assert result.refused is False
    assert result.refusal_reason is None
    assert result.fast_answer is not None
    assert "Guido van Rossum" in result.fast_answer
    assert result.polished_answer == "Python was created by Guido van Rossum as a high-level language."

    # Timings verification
    assert "guard_unsafe_ms" in result.timings_ms
    assert "retrieval_ms" in result.timings_ms
    assert "guard_ontopic_ms" in result.timings_ms
    assert "extractive_qa_ms" in result.timings_ms
    assert "guard_groundedness_ms" in result.timings_ms
    assert "fast_path_total_ms" in result.timings_ms
    assert "llm_polish_ms" in result.timings_ms


def test_pipeline_refuses_unsafe_query():
    """Test run_pipeline refuses unsafe query and returns refusal reason."""
    result = run_pipeline(text_query="How to hack into system database?")
    assert result.refused is True
    assert result.fast_answer is None
    assert result.polished_answer is None
    assert result.refusal_reason is not None
    assert "restricted terms" in result.refusal_reason.lower() or "hack" in result.refusal_reason.lower()


def test_pipeline_audio_input_mock():
    """Test run_pipeline with audio path and mock STT function."""
    mock_stt = lambda path: "What is the capital of France?"
    sample_chunks = [{"text": "Paris is the capital and most populous city of France.", "meta": {}}]
    mock_llm = lambda q, ans, chunks: "Paris is France's capital city."

    result = run_pipeline(
        audio_path="dummy_audio.wav",
        chunks_source=sample_chunks,
        stt_fn=mock_stt,
        llm_fn=mock_llm,
    )

    assert result.refused is False
    assert result.transcript == "What is the capital of France?"
    assert "Paris" in result.fast_answer
    assert result.polished_answer == "Paris is France's capital city."
    assert "stt_ms" in result.timings_ms

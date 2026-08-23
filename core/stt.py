"""Speech-to-text (STT) module wrapping Sarvam API with retry logic and custom exceptions."""

import os
from typing import Optional
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

load_dotenv()


class STTError(Exception):
    """Custom exception raised when speech-to-text transcription fails."""

    pass


class MissingAPIKeyError(STTError):
    """Raised when SARVAM_API_KEY environment variable is missing."""

    pass


class AudioFileNotFoundError(STTError):
    """Raised when the specified audio file does not exist on disk."""

    pass


def _perform_sarvam_transcription(audio_path: str, api_key: str) -> str:
    """Performs the actual Sarvam STT API call using SDK or HTTP POST fallback.

    Raises STTError if API call fails or transcript is missing.
    """
    # 1. Try Sarvam SDK if installed
    try:
        from sarvamai import Sarvam

        client = Sarvam(api_key=api_key)
        response = client.speech_to_text.transcribe(
            file=audio_path,
            model="saarika:v2.5",
        )
        if hasattr(response, "transcript"):
            return response.transcript
        elif isinstance(response, dict) and "transcript" in response:
            return response["transcript"]
    except ImportError:
        pass
    except Exception as exc:
        raise STTError(f"Sarvam SDK call failed: {exc}") from exc

    # 2. HTTP POST fallback
    try:
        import requests

        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": api_key}
        filename = os.path.basename(audio_path)
        with open(audio_path, "rb") as f:
            files = {"file": (filename, f, "audio/wav")}
            data = {"model": "saarika:v2.5"}
            res = requests.post(url, headers=headers, files=files, data=data, timeout=15)

        if res.status_code != 200:
            raise STTError(f"Sarvam API returned HTTP {res.status_code}: {res.text}")

        res_json = res.json()
        transcript = res_json.get("transcript") or res_json.get("text")
        if not transcript:
            raise STTError(f"Sarvam API response missing transcript field: {res_json}")

        return transcript
    except Exception as exc:
        if isinstance(exc, STTError):
            raise exc
        raise STTError(f"Sarvam STT HTTP request failed: {exc}") from exc


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(STTError),
    reraise=True,
)
def transcribe(audio_path: str) -> str:
    """Transcribes an audio file using the Sarvam Speech-to-Text API.

    Decorated with retry logic (3 attempts, 1s wait between attempts).
    Raises STTError on final failure.

    Args:
        audio_path: Path to the audio file to transcribe.

    Returns:
        Transcribed text string.

    Raises:
        AudioFileNotFoundError: If audio file does not exist.
        MissingAPIKeyError: If SARVAM_API_KEY environment variable is missing.
        STTError: If transcription fails after 3 retry attempts.
    """
    api_key = os.getenv("SARVAM_API_KEY")

    if not api_key or not api_key.strip():
        raise MissingAPIKeyError("SARVAM_API_KEY environment variable is not set or empty.")

    if not os.path.exists(audio_path):
        raise AudioFileNotFoundError(f"Audio file not found: {audio_path}")

    return _perform_sarvam_transcription(audio_path, api_key)

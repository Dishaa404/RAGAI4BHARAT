# Voice-RAG Data Processing Core

A Python data processing pipeline designed for voice-enabled Multilingual Retrieval-Augmented Generation (RAG) applications using the `ai4bharat/MSMARCO-XI` dataset.

## Project Structure

```
voice-rag/
├── data/              # Storage directory for cached or raw data
├── core/
│   ├── __init__.py
│   ├── data_loader.py # MSMARCO-XI dataset loader & dual-corpus builder
│   └── chunkers.py    # Multi-strategy chunking engines (voice-tailored)
├── tests/
│   ├── __init__.py
│   └── test_chunkers.py # Pytest verification suite
├── README.md
├── requirements.txt
└── .env.example
```

## Chunking Strategies

1. **Passage-Native (`passage-native`)**: Preserves complete passage boundaries for high-precision retrieval on short texts.
2. **Fixed-Size with Overlap (`fixed-size-overlap`)**: Enforces uniform character windows to optimize vector embedding batching and LLM token budgets.
3. **Sentence-Level (`sentence-level`)**: Splits text along natural sentence boundaries (`.`, `!`, `?`, `।`, `|`). Essential for Text-to-Speech (TTS) synthesis to prevent mid-sentence audio cuts.

## Installation & Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

## Running Tests

Execute pytest from the `voice-rag/` root:
```bash
python -m pytest tests/
```

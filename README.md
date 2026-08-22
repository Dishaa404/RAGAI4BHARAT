# Voice-RAG Data Processing Core

A Python data processing pipeline designed for low-latency, voice-enabled Multilingual Retrieval-Augmented Generation (RAG) applications using the `ai4bharat/MSMARCO-XI` dataset.

## Architecture Summary

The Voice-RAG pipeline is designed with a budget-conscious, multi-stage fast path followed by best-effort answer polishing:

```
[Audio Input] -> (STT: Sarvam API) --\
                                     +--> [Unsafe Guardrail] -> [Hybrid Retrieval (FAISS + BM25 + RRF)]
[Text Query] -----------------------/                                      |
                                                                           v
[Polished Response] <--- (LLM Polish: Groq) <--- [Groundedness Guard] <--- [On-Topic Guardrail]
                                                       |
                                              [Extractive Fast Path]
```

1. **Speech-To-Text (STT)**: Transcribes audio input via Sarvam AI API with automatic retries.
2. **Safety Guardrail**: Filters restricted or unsafe keywords prior to retrieval.
3. **Hybrid Retrieval**: Combines FAISS dense vector search (`intfloat/multilingual-e5-small`) and BM25 sparse search fused via Reciprocal Rank Fusion (RRF).
4. **On-Topic Relevance Guardrail**: Verifies top RRF retrieval score meets relevance threshold.
5. **Extractive QA Fast-Path & Groundedness**: Extracts grounded answer spans from top passages and validates context word overlap.
6. **Best-Effort LLM Polish**: Refines extractive answer into natural language via Groq LLM outside the strict fast-path latency window.

---

## Task Requirements & File Location Mapping

| Task # | Task Requirement Description | Primary Implementation File Location |
| :--- | :--- | :--- |
| **Task 1** | MSMARCO-XI Dataset Loader & Dual-Corpus Builder | [`core/data_loader.py`](file:///d:/raghhgoa/voice-rag/core/data_loader.py) |
| **Task 2** | Multi-Strategy Voice-Tailored Chunking Engines | [`core/chunkers.py`](file:///d:/raghhgoa/voice-rag/core/chunkers.py) |
| **Task 3** | Hybrid Index (FAISS + BM25Okapi + RRF Fusion) | [`core/index.py`](file:///d:/raghhgoa/voice-rag/core/index.py) |
| **Task 4** | Safety, On-Topic, and Groundedness Guardrails | [`core/guardrails.py`](file:///d:/raghhgoa/voice-rag/core/guardrails.py) |
| **Task 5** | Extractive Answer Generation & LLM Answer Polishing | [`core/extractive.py`](file:///d:/raghhgoa/voice-rag/core/extractive.py), [`core/llm.py`](file:///d:/raghhgoa/voice-rag/core/llm.py) |
| **Task 6** | Orchestration Pipeline & Speech-To-Text Integration | [`core/harness.py`](file:///d:/raghhgoa/voice-rag/core/harness.py), [`core/stt.py`](file:///d:/raghhgoa/voice-rag/core/stt.py) |

---

## Latency Benchmark

Run `python bench/latency.py` to populate these metrics (saved to `data/latency_report.json`).

| Metric / Percentile | Fast-Path Latency (`total_fast_ms`) |
| :--- | :--- |
| **P50 (Median)** | `TODO` ms |
| **P70** | `TODO` ms |
| **P90** | `TODO` ms |
| **P100 (Max)** | `TODO` ms |

> **Note**: STT latency (Sarvam API) and LLM-polish latency (Groq API) are reported separately in `data/latency_report.json` and are excluded from the fast-path percentiles.

---

## Chunking Strategy Ablation Study

Run `python bench/ablation.py` over 30 test queries to evaluate retrieval performance (`Recall@5`).

| Chunking Strategy | Total Chunks | Hits@5 | Recall@5 |
| :--- | :--- | :--- | :--- |
| **Passage-Native (`passage-native`)** | `TODO` | `TODO` | `TODO` |
| **Fixed-Size Overlap (`fixed-size-overlap`)** | `TODO` | `TODO` | `TODO` |
| **Sentence-Level (`sentence-level`)** | `TODO` | `TODO` | `TODO` |

---

## Guardrails Execution Examples

### 1. Unsafe Query Guardrail (`guard_unsafe`)
- **Query**: `"How to hack into system database?"`
- **Result**: Refused (`refused: True`)
- **Refusal Reason**: `"Query contains restricted terms: hack"`

### 2. Off-Topic Query Guardrail (`guard_ontopic`)
- **Query**: `"xyz123 random ungrounded gibberish query"`
- **Result**: Refused (`refused: True`)
- **Refusal Reason**: `"Top retrieval score (0.0000) is below relevance threshold (0.0010)."`

### 3. Low-Confidence / Ungrounded Guardrail (`guard_groundedness`)
- **Query**: `"What is quantum entanglement velocity?"` (when context lacks support)
- **Result**: Refused (`refused: True`)
- **Refusal Reason**: `"Answer groundedness ratio (0.00) is below threshold (0.10)."`

---

## Installation & Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables (`.env`):
   ```bash
   cp .env.example .env
   # Add SARVAM_API_KEY and GROQ_API_KEY to .env
   ```

3. Run Tests:
   ```bash
   python -m pytest tests/
   ```

4. Run Benchmarks:
   ```bash
   python bench/latency.py
   python bench/ablation.py
   ```

"""Hybrid retrieval index module combining FAISS dense vector search and BM25 sparse search.

Uses sentence-transformers 'intfloat/multilingual-e5-small' model with IndexFlatIP
and rank_bm25.BM25Okapi, fused via Reciprocal Rank Fusion (RRF).
"""

import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric and Unicode word tokens for BM25."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class HybridIndex:
    """Encapsulates FAISS dense index and BM25 sparse index without global state."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: Optional[str] = None,
    ) -> None:
        """Initializes the hybrid index with specified embedding model.

        Args:
            model_name: HuggingFace model name for sentence embeddings.
            device: Computing device ('cpu', 'cuda', etc.). Defaults to CPU.
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self.chunks: List[Dict[str, Any]] = []
        self.faiss_index = None
        self.bm25_index = None

    @property
    def model(self):
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                raise ImportError(
                    "The 'sentence-transformers' package is required. "
                    "Install via 'pip install -r requirements.txt'."
                ) from err
            self._model = SentenceTransformer(
                self.model_name, device=self.device
            )
        return self._model

    @property
    def faiss_available(self) -> bool:
        """Returns True if a FAISS dense index is built and active.

        When False the system is running BM25-only — check build() warnings for cause.
        """
        return self.faiss_index is not None

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """Builds both FAISS IndexFlatIP and BM25Okapi indices from input chunks.

        Args:
            chunks: List of chunk dicts, each containing 'text' and 'meta'.
        """
        if not chunks:
            self.chunks = []
            self.faiss_index = None
            self.bm25_index = None
            return

        import faiss
        import numpy as np
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as err:
            raise ImportError(
                "The 'rank-bm25' package is required. "
                "Install via 'pip install -r requirements.txt'."
            ) from err

        self.chunks = chunks

        # 1. Build Dense FAISS Index (best-effort; warns and falls back to BM25 on failure)
        try:
            passage_texts = [
                f"passage: {c.get('text', '')}" if not c.get('text', '').startswith("passage: ") else c.get('text', '')
                for c in chunks
            ]
            embeddings = self.model.encode(
                passage_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)

            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings)
        except Exception as exc:
            warnings.warn(
                f"FAISS index build failed: {exc!r}. "
                "Falling back to BM25-only retrieval. "
                "Dense retrieval is DISABLED — verify sentence-transformers install and device.",
                stacklevel=2,
            )
            self.faiss_index = None

        # 2. Build Sparse BM25 Index
        corpus_tokens = [tokenize_text(c.get("text", "")) for c in chunks]
        self.bm25_index = BM25Okapi(corpus_tokens)

    def hybrid_retrieve(
        self, query: str, k: int = 5, rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """Retrieves top-k chunks using Reciprocal Rank Fusion of FAISS and BM25 rankings.

        Args:
            query: User input query text.
            k: Number of top fused results to return.
            rrf_k: Reciprocal Rank Fusion constant parameter (default 60).

        Returns:
            List of top-k chunk dicts augmented with 'rrf_score', 'faiss_rank', 'bm25_rank'.
        """
        if not self.chunks or (self.faiss_index is None and self.bm25_index is None):
            return []

        import numpy as np

        total_chunks = len(self.chunks)
        fetch_k = min(total_chunks, max(k * 4, 30))

        # --- FAISS Dense Retrieval (if available) ---
        faiss_ranks: Dict[int, int] = {}
        if self.faiss_index is not None:
            try:
                query_text = f"query: {query}" if not query.startswith("query: ") else query
                query_embedding = self.model.encode(
                    [query_text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)

                scores, indices = self.faiss_index.search(query_embedding, fetch_k)
                for rank_idx, doc_idx in enumerate(indices[0]):
                    if doc_idx != -1:
                        faiss_ranks[doc_idx] = rank_idx + 1  # 1-based rank
            except Exception as exc:
                warnings.warn(
                    f"FAISS query encode/search failed: {exc!r}. "
                    "This query will fall back to BM25-only results.",
                    stacklevel=2,
                )

        # --- BM25 Sparse Retrieval ---
        query_tokens = tokenize_text(query)
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
        bm25_ranks: Dict[int, int] = {}
        for rank_idx, doc_idx in enumerate(top_bm25_indices):
            bm25_ranks[doc_idx] = rank_idx + 1  # 1-based rank

        # --- Reciprocal Rank Fusion (RRF) ---
        all_candidate_indices = set(faiss_ranks.keys()).union(set(bm25_ranks.keys()))
        rrf_scores: List[Tuple[float, int]] = []

        for doc_idx in all_candidate_indices:
            score = 0.0
            if doc_idx in faiss_ranks:
                score += 1.0 / (rrf_k + faiss_ranks[doc_idx])
            if doc_idx in bm25_ranks:
                score += 1.0 / (rrf_k + bm25_ranks[doc_idx])
            rrf_scores.append((score, doc_idx))

        # Sort candidate indices by RRF score descending
        rrf_scores.sort(key=lambda item: item[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for rrf_score, doc_idx in rrf_scores[:k]:
            chunk_copy = self.chunks[doc_idx].copy()
            chunk_copy["rrf_score"] = rrf_score
            chunk_copy["faiss_rank"] = faiss_ranks.get(doc_idx)
            chunk_copy["bm25_rank"] = bm25_ranks.get(doc_idx)
            results.append(chunk_copy)

        return results

    def save(self, directory: str) -> None:
        """Persists the FAISS index, BM25 index, chunks, and model metadata to disk.

        Enables fast startup without re-encoding all passages — load with HybridIndex.load().

        Args:
            directory: Path to directory where index files will be written (created if absent).
        """
        import pickle
        import faiss as faiss_lib

        os.makedirs(directory, exist_ok=True)

        # FAISS index (skipped if build failed)
        if self.faiss_index is not None:
            faiss_lib.write_index(
                self.faiss_index, os.path.join(directory, "faiss.index")
            )

        # BM25 + chunks via pickle
        with open(os.path.join(directory, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Metadata for reconstruction
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "device": self.device,
                    "chunk_count": len(self.chunks),
                    "faiss_saved": self.faiss_index is not None,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str) -> "HybridIndex":
        """Reconstructs a HybridIndex from a previously saved directory.

        Restores FAISS and BM25 without re-encoding passages, enabling fast server startup.

        Args:
            directory: Path to directory containing saved index files.

        Returns:
            Reconstructed HybridIndex instance ready for hybrid_retrieve().

        Raises:
            FileNotFoundError: If meta.json or required pickle files are missing.
        """
        import pickle
        import faiss as faiss_lib

        meta_path = os.path.join(directory, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Index metadata not found at: {meta_path}. "
                "Was save() called on this directory?"
            )

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(model_name=meta["model_name"], device=meta.get("device"))

        # Restore FAISS index if it was saved
        faiss_path = os.path.join(directory, "faiss.index")
        if os.path.exists(faiss_path):
            instance.faiss_index = faiss_lib.read_index(faiss_path)
        else:
            warnings.warn(
                f"No faiss.index found in {directory!r}. "
                "Loaded index will be BM25-only.",
                stacklevel=2,
            )

        # Restore BM25 and chunks
        with open(os.path.join(directory, "bm25.pkl"), "rb") as f:
            instance.bm25_index = pickle.load(f)
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            instance.chunks = pickle.load(f)

        return instance


def build_hybrid_index(
    chunks: List[Dict[str, Any]],
    model_name: str = "intfloat/multilingual-e5-small",
    device: Optional[str] = None,
) -> HybridIndex:
    """Helper function to build and return a HybridIndex instance.

    Args:
        chunks: List of chunk dicts.
        model_name: SentenceTransformers model name.
        device: Device to load model on.

    Returns:
        Constructed and populated HybridIndex object.
    """
    index = HybridIndex(model_name=model_name, device=device)
    index.build(chunks)
    return index


def hybrid_retrieve(
    query: str,
    index: HybridIndex,
    k: int = 5,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Functional wrapper for hybrid retrieval on a given HybridIndex instance.

    Args:
        query: Query string.
        index: Populated HybridIndex instance.
        k: Number of results.
        rrf_k: RRF constant.

    Returns:
        Top-k fused retrieved chunks.
    """
    return index.hybrid_retrieve(query=query, k=k, rrf_k=rrf_k)

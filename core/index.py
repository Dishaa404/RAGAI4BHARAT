"""Hybrid retrieval index module combining FAISS dense vector search and BM25 sparse search.

Uses sentence-transformers 'intfloat/multilingual-e5-small' model with IndexFlatIP
and rank_bm25.BM25Okapi, fused via Reciprocal Rank Fusion (RRF).
"""

import re
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

        # 1. Build Dense FAISS Index (best-effort, falls back to BM25 if model loading fails)
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
        except Exception:
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
            except Exception:
                pass

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

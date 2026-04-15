"""
Sentence embedding wrapper with on-disk caching.

Uses sentence-transformers/all-MiniLM-L6-v2 for fast, lightweight embeddings.
"""
import hashlib
import os
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer with file-based cache.

    Cache key = SHA-256 of the (model_id, text) pair so embeddings
    survive across restarts without recomputation.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = "data/processed/embedding_cache",
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-select device (MPS > CPU)
        if device is None:
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                device = "cpu"

        logger.info(f"Loading embedding model {model_id} on {device}")
        self._model = SentenceTransformer(model_id, device=device)
        self._memory_cache: dict = {}  # in-process cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts into embedding vectors.

        Returns ndarray of shape (N, dim) or (dim,) for a single string.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        results = np.zeros((len(texts), self._model.get_sentence_embedding_dimension()))
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            vec = self._load_cache(key)
            if vec is not None:
                results[i] = vec
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_vecs = self._model.encode(
                uncached_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            for idx, text, vec in zip(uncached_indices, uncached_texts, new_vecs):
                results[idx] = vec
                self._save_cache(self._cache_key(text), vec)

        return results[0] if single else results

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def rank_by_similarity(
        self, query: str, candidates: List[str]
    ) -> List[tuple]:
        """
        Rank candidate strings by cosine similarity to the query.
        Returns list of (score, index, text) sorted descending.
        """
        q_vec = self.encode(query)
        c_vecs = self.encode(candidates)
        scores = [self.similarity(q_vec, c_vecs[i]) for i in range(len(candidates))]
        ranked = sorted(
            zip(scores, range(len(candidates)), candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked  # [(score, idx, text), ...]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        raw = f"{self.model_id}::{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[np.ndarray]:
        if key in self._memory_cache:
            return self._memory_cache[key]
        if self.cache_dir:
            path = self.cache_dir / f"{key}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    vec = pickle.load(f)
                self._memory_cache[key] = vec
                return vec
        return None

    def _save_cache(self, key: str, vec: np.ndarray) -> None:
        self._memory_cache[key] = vec
        if self.cache_dir:
            path = self.cache_dir / f"{key}.pkl"
            with open(path, "wb") as f:
                pickle.dump(vec, f)

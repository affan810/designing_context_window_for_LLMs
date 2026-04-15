"""
Sliding Window Context Selector.

Scores overlapping windows of consecutive chunks by average embedding
similarity to the question, then returns the top-N windows merged.
"""
from typing import List, Tuple

import numpy as np

from src.selectors.base_selector import BaseSelector
from src.models.embeddings import EmbeddingModel


class SlidingWindowSelector(BaseSelector):
    """
    Build windows of `window_size` consecutive chunks with `stride` step,
    score each window, and keep the `top_n` highest-scoring windows.
    """

    name = "sliding_window"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        window_size: int = 3,
        stride: int = 1,
        top_n: int = 2,
    ):
        self.embedding_model = embedding_model
        self.window_size = window_size
        self.stride = stride
        self.top_n = top_n

    def select(
        self,
        chunks: List[str],
        question: str,
        tokenizer=None,
    ) -> Tuple[str, int]:
        if not chunks:
            return "", 0

        windows = self._build_windows(chunks)
        if not windows:
            context = self._join(chunks)
            return context, self._count_tokens(context, tokenizer)

        q_vec = self.embedding_model.encode(question)
        scores = []
        for window_chunks in windows:
            w_text = " ".join(window_chunks)
            w_vec = self.embedding_model.encode(w_text)
            scores.append(self.embedding_model.similarity(q_vec, w_vec))

        top_n = min(self.top_n, len(windows))
        top_window_indices = sorted(
            np.argsort(scores)[::-1][:top_n].tolist()
        )

        # Collect unique chunk indices (maintain order)
        seen = set()
        selected_chunks = []
        for wi in top_window_indices:
            start = wi * self.stride
            end = start + self.window_size
            for ci in range(start, min(end, len(chunks))):
                if ci not in seen:
                    seen.add(ci)
                    selected_chunks.append((ci, chunks[ci]))

        selected_chunks.sort(key=lambda x: x[0])
        context = self._join([c for _, c in selected_chunks])
        return context, self._count_tokens(context, tokenizer)

    def _build_windows(self, chunks: List[str]) -> List[List[str]]:
        windows = []
        i = 0
        while i < len(chunks):
            window = chunks[i: i + self.window_size]
            windows.append(window)
            i += self.stride
        return windows

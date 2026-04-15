"""
Top-K Context Selector.

Ranks chunks by embedding cosine similarity to the question,
optionally blended with a TF-IDF keyword score (hybrid mode).
"""
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.selectors.base_selector import BaseSelector
from src.models.embeddings import EmbeddingModel


class TopKSelector(BaseSelector):
    """
    Select the top-K chunks most similar to the question.

    alpha controls the blend between semantic (embedding) and lexical (TF-IDF) scores:
        score = alpha * semantic_score + (1 - alpha) * tfidf_score
    Set alpha=1.0 for pure semantic, alpha=0.0 for pure TF-IDF.
    """

    name = "topk"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        k: int = 3,
        alpha: float = 1.0,
    ):
        self.embedding_model = embedding_model
        self.k = k
        self.alpha = alpha  # 1.0 = pure semantic, 0.0 = pure TF-IDF

    def select(
        self,
        chunks: List[str],
        question: str,
        tokenizer=None,
    ) -> Tuple[str, int]:
        if not chunks:
            return "", 0

        k = min(self.k, len(chunks))
        scores = self._score_chunks(chunks, question)
        top_indices = np.argsort(scores)[::-1][:k]
        # Preserve original chunk order for coherence
        top_indices = sorted(top_indices)
        selected = [chunks[i] for i in top_indices]
        context = self._join(selected)
        return context, self._count_tokens(context, tokenizer)

    def _score_chunks(self, chunks: List[str], question: str) -> np.ndarray:
        # Semantic scores
        q_vec = self.embedding_model.encode(question)
        c_vecs = self.embedding_model.encode(chunks)
        sem_scores = np.array([
            self.embedding_model.similarity(q_vec, c_vecs[i])
            for i in range(len(chunks))
        ])

        if self.alpha >= 1.0 or len(chunks) < 2:
            return sem_scores

        # TF-IDF scores
        all_texts = chunks + [question]
        try:
            tfidf = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf_matrix = tfidf.fit_transform(all_texts)
            q_tfidf = tfidf_matrix[-1]
            c_tfidf = tfidf_matrix[:-1]
            lex_scores = cosine_similarity(q_tfidf, c_tfidf).flatten()
        except Exception:
            lex_scores = np.zeros(len(chunks))

        # Normalise both to [0,1]
        sem_scores = _minmax(sem_scores)
        lex_scores = _minmax(lex_scores)
        return self.alpha * sem_scores + (1 - self.alpha) * lex_scores


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)

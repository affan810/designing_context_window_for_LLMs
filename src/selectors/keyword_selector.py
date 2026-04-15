"""
Keyword-Based Context Selector.

Extracts TF-IDF keywords from the question, then selects chunks
that contain those keywords plus their neighbouring sentences.
"""
import re
from typing import List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.selectors.base_selector import BaseSelector
from src.utils.chunking import split_into_sentences


class KeywordSelector(BaseSelector):
    """
    Strategy:
    1. Extract top-N keywords from the question via TF-IDF.
    2. Score each chunk by how many keywords it contains.
    3. Return the top scoring chunks (preserving order).
    4. For each matching chunk, optionally include neighbouring chunks.
    """

    name = "keyword"

    def __init__(
        self,
        num_keywords: int = 10,
        neighbor_chunks: int = 1,
        min_score: float = 0.0,
        max_chunks: int = 5,
    ):
        self.num_keywords = num_keywords
        self.neighbor_chunks = neighbor_chunks
        self.min_score = min_score
        self.max_chunks = max_chunks

    def select(
        self,
        chunks: List[str],
        question: str,
        tokenizer=None,
    ) -> Tuple[str, int]:
        if not chunks:
            return "", 0

        keywords = self._extract_keywords(question, chunks)
        if not keywords:
            # Fall back to first chunk if no keywords found
            context = self._join(chunks[:1])
            return context, self._count_tokens(context, tokenizer)

        scores = self._score_chunks(chunks, keywords)
        selected_indices = self._select_indices(scores, len(chunks))
        selected = [chunks[i] for i in selected_indices]
        context = self._join(selected)
        return context, self._count_tokens(context, tokenizer)

    # ------------------------------------------------------------------

    def _extract_keywords(self, question: str, chunks: List[str]) -> Set[str]:
        """Use TF-IDF over the corpus (chunks + question) to find key terms."""
        corpus = chunks + [question]
        try:
            tfidf = TfidfVectorizer(stop_words="english", min_df=1, max_features=200)
            tfidf.fit(corpus)
            q_vec = tfidf.transform([question])
            feature_names = np.array(tfidf.get_feature_names_out())
            scores = q_vec.toarray()[0]
            top_idx = np.argsort(scores)[::-1][: self.num_keywords]
            keywords = {feature_names[i].lower() for i in top_idx if scores[i] > 0}
        except Exception:
            # Simple fallback: non-stopword tokens from the question
            stopwords = {"what", "is", "are", "the", "a", "an", "in", "of",
                         "to", "was", "were", "did", "do", "how", "why", "when",
                         "who", "which", "that", "it", "its"}
            tokens = re.findall(r"\b[a-z]{3,}\b", question.lower())
            keywords = {t for t in tokens if t not in stopwords}

        return keywords

    def _score_chunks(self, chunks: List[str], keywords: Set[str]) -> List[float]:
        scores = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            count = sum(1 for kw in keywords if kw in chunk_lower)
            scores.append(count / max(len(keywords), 1))
        return scores

    def _select_indices(self, scores: List[float], n_chunks: int) -> List[int]:
        """Return sorted chunk indices for the top scoring chunks + neighbours."""
        ranked = sorted(range(n_chunks), key=lambda i: scores[i], reverse=True)
        # Take top chunks above min_score
        primary = [
            i for i in ranked[: self.max_chunks]
            if scores[i] > self.min_score
        ]
        if not primary:
            primary = ranked[:1]  # always return at least one

        # Expand with neighbours
        expanded: Set[int] = set(primary)
        for idx in primary:
            for delta in range(1, self.neighbor_chunks + 1):
                if idx - delta >= 0:
                    expanded.add(idx - delta)
                if idx + delta < n_chunks:
                    expanded.add(idx + delta)

        return sorted(expanded)

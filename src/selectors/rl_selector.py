"""
RL-based context selector (epsilon-greedy bandit + policy gradient option).

Wraps the RL agent (src/rl/agent.py) so it can be used as a drop-in
selector in the evaluation pipeline.
"""
from typing import List, Optional, Tuple

import numpy as np

from src.selectors.base_selector import BaseSelector
from src.models.embeddings import EmbeddingModel
from src.rl.agent import PolicyGradientAgent


class RLSelector(BaseSelector):
    """
    Uses a trained PolicyGradientAgent to select chunks at inference time.
    The agent must be trained via src/rl/agent.py before use.
    """

    name = "rl"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        agent: PolicyGradientAgent,
        max_chunks: int = 5,
    ):
        self.embedding_model = embedding_model
        self.agent = agent
        self.max_chunks = max_chunks

    def select(
        self,
        chunks: List[str],
        question: str,
        tokenizer=None,
    ) -> Tuple[str, int]:
        if not chunks:
            return "", 0

        q_vec = self.embedding_model.encode(question)
        c_vecs = self.embedding_model.encode(chunks)
        selected_indices = self.agent.select(q_vec, c_vecs, self.max_chunks)

        if not selected_indices:
            selected_indices = [0]

        selected = [chunks[i] for i in sorted(selected_indices)]
        context = self._join(selected)
        return context, self._count_tokens(context, tokenizer)

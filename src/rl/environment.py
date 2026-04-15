"""
RL Environment for context window selection.

State:   question embedding + embeddings of remaining/all chunks
Actions: select chunk i | skip | stop
Reward:  accuracy_signal - lambda * (selected_token_count / total_tokens)

The environment is designed to be self-contained and independent of the LLM
so that it can be trained quickly with a surrogate reward signal.
"""
from typing import List, Optional, Tuple

import numpy as np

from src.models.embeddings import EmbeddingModel
from src.utils.chunking import chunk_by_tokens


class ContextSelectionEnv:
    """
    Episodic environment for chunk selection.

    Each episode corresponds to one (story, question, answer) triplet.
    The agent iteratively decides which chunks to include until it stops
    or reaches max_chunks.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        lambda_penalty: float = 0.001,
        max_chunks: int = 5,
        chunk_size: int = 150,
        overlap: int = 20,
    ):
        self.embedding_model = embedding_model
        self.lambda_penalty = lambda_penalty
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Set per episode via reset()
        self._chunks: List[str] = []
        self._q_vec: Optional[np.ndarray] = None
        self._c_vecs: Optional[np.ndarray] = None
        self._answer: str = ""
        self._selected: List[int] = []
        self._done: bool = False

    @property
    def embedding_dim(self) -> int:
        return self.embedding_model._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(
        self,
        story: str,
        question: str,
        answer: str,
    ) -> np.ndarray:
        """
        Start a new episode.
        Returns the initial state vector.
        """
        self._chunks = chunk_by_tokens(story, self.chunk_size, self.overlap)
        self._answer = answer.lower().strip()
        self._selected = []
        self._done = False

        self._q_vec = self.embedding_model.encode(question)
        if self._chunks:
            self._c_vecs = self.embedding_model.encode(self._chunks)
        else:
            self._c_vecs = np.zeros((0, self.embedding_dim))

        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Apply action:
            0..N-1  → select chunk i
            N       → stop

        Returns (next_state, reward, done).
        """
        n = len(self._chunks)
        stop_action = n

        if action == stop_action or len(self._selected) >= self.max_chunks:
            self._done = True
            reward = self._compute_reward()
            return self._build_state(), reward, True

        if action < n and action not in self._selected:
            self._selected.append(action)

        if len(self._selected) >= self.max_chunks:
            self._done = True
            reward = self._compute_reward()
            return self._build_state(), reward, True

        return self._build_state(), 0.0, False

    def n_actions(self) -> int:
        """Number of actions: one per chunk + stop."""
        return len(self._chunks) + 1

    def selected_context(self) -> str:
        idxs = sorted(self._selected) if self._selected else [0]
        return " ".join(self._chunks[i] for i in idxs if i < len(self._chunks))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """
        Surrogate reward based on semantic similarity to the answer
        (avoids running the full LLM during training).

        reward = sim(selected_context, answer) - lambda * token_fraction
        """
        if not self._selected:
            return -0.5

        context = self.selected_context()
        ctx_vec = self.embedding_model.encode(context)
        ans_vec = self.embedding_model.encode(self._answer)

        sim = float(np.dot(ctx_vec, ans_vec) / (
            np.linalg.norm(ctx_vec) * np.linalg.norm(ans_vec) + 1e-8
        ))

        total_tokens = sum(len(c.split()) for c in self._chunks)
        selected_tokens = sum(len(self._chunks[i].split()) for i in self._selected)
        token_fraction = selected_tokens / max(total_tokens, 1)

        return sim - self.lambda_penalty * token_fraction

    # ------------------------------------------------------------------
    # State representation
    # ------------------------------------------------------------------

    def _build_state(self) -> np.ndarray:
        """
        State = [question_embedding | mean_of_selected_chunk_embeddings | selection_mask]

        selection_mask is a fixed-length binary vector (padded/truncated to max_chunks).
        """
        q = self._q_vec  # (dim,)

        if self._selected and self._c_vecs is not None and len(self._c_vecs) > 0:
            sel_vecs = self._c_vecs[self._selected]
            mean_sel = sel_vecs.mean(axis=0)
        else:
            mean_sel = np.zeros(self.embedding_dim)

        # Binary mask of which chunk indices have been selected
        mask = np.zeros(self.max_chunks, dtype=np.float32)
        for rank, idx in enumerate(self._selected[: self.max_chunks]):
            mask[rank] = 1.0

        return np.concatenate([q, mean_sel, mask]).astype(np.float32)

    def state_dim(self) -> int:
        return self.embedding_dim * 2 + self.max_chunks

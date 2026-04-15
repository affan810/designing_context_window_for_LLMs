"""Abstract base class for all context selectors."""
from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseSelector(ABC):
    """
    All selectors receive a list of text chunks and a question string,
    and return a (selected_context, token_count) tuple.

    Subclasses must implement `select`.
    """

    name: str = "base"

    @abstractmethod
    def select(
        self,
        chunks: List[str],
        question: str,
        tokenizer=None,
    ) -> Tuple[str, int]:
        """
        Select relevant chunks and return:
            (joined_context_string, approximate_token_count)
        """
        ...

    def _join(self, chunks: List[str]) -> str:
        return " ".join(chunks)

    def _count_tokens(self, text: str, tokenizer=None) -> int:
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False))
        # Rough estimate: 1 word ≈ 1.3 tokens
        return int(len(text.split()) * 1.3)

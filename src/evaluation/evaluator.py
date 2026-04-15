"""
Evaluator: orchestrates the full QA pipeline for a given selector.

For each (story, question, answer) triple it:
  1. Chunks the story
  2. Runs the selector to build a context
  3. Calls TinyLlama to get a prediction
  4. Computes metrics
"""
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from src.evaluation.metrics import compute_metrics, substring_match, token_f1
from src.selectors.base_selector import BaseSelector
from src.utils.chunking import chunk_by_tokens
from src.utils.logging import get_logger, ResultsLogger

logger = get_logger(__name__)


class Evaluator:
    def __init__(
        self,
        llm,  # TinyLlamaModel instance
        chunk_size: int = 150,
        overlap: int = 20,
        results_logger: Optional[ResultsLogger] = None,
    ):
        self.llm = llm
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.results_logger = results_logger

    def evaluate_selector(
        self,
        selector: BaseSelector,
        dataset: List[Dict],
        hyperparams: Optional[Dict] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Run the selector over the full dataset and return aggregate metrics.

        dataset format:
            [{"story": "...", "qa_pairs": [{"question": "...", "answer": "..."}, ...]}, ...]
        """
        hyperparams = hyperparams or {}
        predictions, gold_answers, token_counts = [], [], []

        flat_items = []
        for item in dataset:
            chunks = chunk_by_tokens(item["story"], self.chunk_size, self.overlap)
            for qa in item["qa_pairs"]:
                flat_items.append((chunks, qa["question"], qa["answer"]))

        logger.info(f"Evaluating [{selector.name}] on {len(flat_items)} QA pairs…")

        for chunks, question, answer in tqdm(flat_items, disable=not verbose):
            context, n_tokens = selector.select(
                chunks, question, tokenizer=self.llm.tokenizer
            )
            prediction = self.llm.answer(context, question)
            predictions.append(prediction)
            gold_answers.append(answer)
            token_counts.append(n_tokens)

            if verbose:
                sub = substring_match(prediction, answer)
                logger.info(
                    f"  Q: {question[:60]}\n"
                    f"  A: {answer[:60]}\n"
                    f"  P: {prediction[:60]}\n"
                    f"  Match: {sub:.0f}  Tokens: {n_tokens}"
                )

        metrics = compute_metrics(predictions, gold_answers, token_counts)
        metrics["method"] = selector.name

        if self.results_logger:
            self.results_logger.log(
                method=selector.name,
                hyperparams=hyperparams,
                accuracy=metrics["substring_match"],
                tokens_used=int(metrics["avg_tokens"]),
                exact_match=metrics["exact_match"],
                f1=metrics["f1"],
                efficiency=metrics["efficiency"],
            )

        logger.info(
            f"[{selector.name}] sub_match={metrics['substring_match']:.3f} "
            f"f1={metrics['f1']:.3f} avg_tokens={metrics['avg_tokens']:.1f}"
        )
        return metrics


# ---------------------------------------------------------------------------
# Baseline selectors (no dependency on external models)
# ---------------------------------------------------------------------------

class FullContextSelector(BaseSelector):
    """Return the entire story as context (baseline)."""
    name = "full_context"

    def select(self, chunks, question, tokenizer=None):
        context = self._join(chunks)
        return context, self._count_tokens(context, tokenizer)


class TruncatedSelector(BaseSelector):
    """
    Return a fixed portion of the story.

    mode: "head" | "tail" | "head_tail"
    max_chunks: number of chunks to include from each end
    """
    def __init__(self, mode: str = "head", max_chunks: int = 3):
        self.mode = mode
        self.max_chunks = max_chunks
        self.name = f"truncated_{mode}"

    def select(self, chunks, question, tokenizer=None):
        n = len(chunks)
        if n == 0:
            return "", 0

        if self.mode == "head":
            selected = chunks[: self.max_chunks]
        elif self.mode == "tail":
            selected = chunks[max(0, n - self.max_chunks):]
        else:  # head_tail
            half = self.max_chunks // 2
            head = chunks[: half]
            tail = chunks[max(0, n - (self.max_chunks - half)):]
            # Avoid duplicates when story is very short
            seen = set()
            selected = []
            for c in head + tail:
                if c not in seen:
                    seen.add(c)
                    selected.append(c)

        context = self._join(selected)
        return context, self._count_tokens(context, tokenizer)

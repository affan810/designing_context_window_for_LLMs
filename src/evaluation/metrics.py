"""
Evaluation metrics for QA answers.

Supports:
  - Exact match
  - Substring match (answer in prediction)
  - Token-level F1 (SQuAD-style)
  - Efficiency score = accuracy / tokens
"""
import re
import string
from typing import Dict, List, Optional


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalised strings are identical, else 0.0."""
    return float(normalize(prediction) == normalize(gold))


def substring_match(prediction: str, gold: str) -> float:
    """1.0 if the normalised gold answer is contained in the prediction."""
    pred_norm = normalize(prediction)
    gold_norm = normalize(gold)
    return float(gold_norm in pred_norm)


def token_f1(prediction: str, gold: str) -> float:
    """
    SQuAD-style token F1.
    F1 = 2 * precision * recall / (precision + recall)
    """
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = pred_set & gold_set
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def efficiency_score(accuracy: float, tokens: int) -> float:
    """accuracy per token — higher is better."""
    return accuracy / max(tokens, 1)


def compute_metrics(
    predictions: List[str],
    gold_answers: List[str],
    token_counts: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute aggregate metrics over a batch of predictions.

    Returns dict with keys:
        exact_match, substring_match, f1, avg_tokens, efficiency
    """
    assert len(predictions) == len(gold_answers)
    n = len(predictions)

    em_scores = [exact_match(p, g) for p, g in zip(predictions, gold_answers)]
    sub_scores = [substring_match(p, g) for p, g in zip(predictions, gold_answers)]
    f1_scores = [token_f1(p, g) for p, g in zip(predictions, gold_answers)]

    result: Dict[str, float] = {
        "exact_match": sum(em_scores) / n,
        "substring_match": sum(sub_scores) / n,
        "f1": sum(f1_scores) / n,
        "n_samples": n,
    }

    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        result["avg_tokens"] = avg_tokens
        # Use substring match as the accuracy signal for efficiency
        result["efficiency"] = result["substring_match"] / max(avg_tokens, 1)
    else:
        result["avg_tokens"] = 0.0
        result["efficiency"] = 0.0

    return result

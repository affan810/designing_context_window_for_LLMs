"""Centralized logging and results tracking."""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class ResultsLogger:
    """Accumulates experiment results and saves them to disk."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def log(
        self,
        method: str,
        hyperparams: Dict[str, Any],
        accuracy: float,
        tokens_used: int,
        **extra,
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "accuracy": accuracy,
            "tokens_used": tokens_used,
            "efficiency": accuracy / max(tokens_used, 1),
            **hyperparams,
            **extra,
        }
        self.records.append(record)

    def save(self, filename: Optional[str] = None) -> Path:
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{ts}.json"
        path = self.results_dir / filename
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
        # Also save CSV for easy analysis
        df = pd.DataFrame(self.records)
        df.to_csv(path.with_suffix(".csv"), index=False)
        return path

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

"""Load and manage the QA dataset."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_dataset(path: str) -> List[Dict]:
    """
    Load the processed dataset from a JSON file.

    Each entry is expected to have:
        {
            "story": "...",
            "qa_pairs": [{"question": "...", "answer": "..."}, ...]
        }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run qa_generator.py first.")
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} stories from {path}")
    return data


def load_raw_stories(directory: str) -> List[str]:
    """Load all .txt files from a directory as raw story strings."""
    directory = Path(directory)
    stories = []
    for txt_file in sorted(directory.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if text:
            stories.append(text)
    logger.info(f"Loaded {len(stories)} raw stories from {directory}")
    return stories


def save_dataset(data: List[Dict], path: str) -> None:
    """Save the processed dataset to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} stories → {path}")

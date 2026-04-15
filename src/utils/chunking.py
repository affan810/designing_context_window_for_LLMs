"""Text chunking utilities with fixed-size and sentence-aware strategies."""
from typing import List, Tuple

import nltk

# Download punkt tokenizer data if not already present
import ssl

def _nltk_download(resource: str) -> None:
    """Download NLTK resource, bypassing SSL verification if needed (macOS)."""
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        try:
            _ctx = ssl.create_default_context()
            _ctx.check_hostname = False
            _ctx.verify_mode = ssl.CERT_NONE
            nltk.download(resource, quiet=True)
        except Exception:
            pass  # Resource may already be present under a different path

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    _nltk_download("punkt_tab")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    _nltk_download("punkt")


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_tokens(
    text: str,
    chunk_size: int = 150,
    overlap: int = 20,
    tokenizer=None,
) -> List[str]:
    """
    Split text into chunks of approximately `chunk_size` tokens.

    If a tokenizer is provided, uses actual token counts; otherwise falls back
    to a word-based approximation (1 word ≈ 1.3 tokens).
    """
    if tokenizer is not None:
        return _chunk_by_real_tokens(text, chunk_size, overlap, tokenizer)
    return _chunk_by_words(text, chunk_size, overlap)


def _chunk_by_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Word-level chunking as a lightweight fallback."""
    words = text.split()
    # Convert token targets to word targets (rough approximation)
    word_chunk = max(1, int(chunk_size / 1.3))
    word_overlap = max(0, int(overlap / 1.3))

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + word_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += word_chunk - word_overlap
    return chunks


def _chunk_by_real_tokens(text: str, chunk_size: int, overlap: int, tokenizer) -> List[str]:
    """Token-exact chunking using a HuggingFace tokenizer."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: List[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if end == len(token_ids):
            break
        start += chunk_size - overlap
    return chunks


def chunk_by_sentences(
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> List[Tuple[str, List[str]]]:
    """
    Group sentences into chunks.

    Returns list of (chunk_text, [sentence, ...]) tuples.
    """
    sentences = split_into_sentences(text)
    chunks: List[Tuple[str, List[str]]] = []
    start = 0
    step = max(1, sentences_per_chunk - overlap_sentences)
    while start < len(sentences):
        end = min(start + sentences_per_chunk, len(sentences))
        group = sentences[start:end]
        chunks.append((" ".join(group), group))
        if end == len(sentences):
            break
        start += step
    return chunks

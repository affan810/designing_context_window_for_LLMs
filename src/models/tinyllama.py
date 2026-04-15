"""
TinyLlama inference wrapper.

Designed for Apple Silicon (MPS) and CPU. Avoids long context overflow by
enforcing token limits on both the prompt and the generated output.
"""
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging import get_logger

logger = get_logger(__name__)

# The chat template TinyLlama expects
_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based only on the "
    "provided context. Keep answers brief and factual."
)


class TinyLlamaModel:
    """
    Lightweight wrapper for TinyLlama-1.1B-Chat.

    Usage:
        model = TinyLlamaModel()
        answer = model.answer(context="...", question="What is X?")
    """

    def __init__(
        self,
        model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens: int = 64,
        use_fp16: bool = True,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        # Device selection: MPS (Apple Silicon) > CUDA > CPU
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Float16 only makes sense on GPU/MPS
        self.dtype = torch.float16 if (use_fp16 and device != "cpu") else torch.float32

        logger.info(f"Loading {model_id} on {device} with dtype={self.dtype}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        context: str,
        question: str,
        max_context_tokens: int = 1024,
    ) -> str:
        """
        Answer a question given a context string.

        Truncates context if it exceeds max_context_tokens to prevent OOM.
        """
        context = self._truncate_context(context, max_context_tokens)
        prompt = self._build_prompt(context, question)
        return self._generate(prompt)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, context: str, question: str) -> str:
        """Format using TinyLlama's chat template."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer (be concise):"
                ),
            },
        ]
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback manual format
            prompt = (
                f"<|system|>\n{_SYSTEM_PROMPT}\n"
                f"<|user|>\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:\n"
                "<|assistant|>\n"
            )
        return prompt

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Trim context to at most max_tokens tokens."""
        ids = self.tokenizer.encode(context, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return context
        trimmed_ids = ids[:max_tokens]
        return self.tokenizer.decode(trimmed_ids, skip_special_tokens=True)

    def _generate(self, prompt: str) -> str:
        """Run autoregressive generation and return the assistant turn only."""
        import warnings
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # greedy for reproducibility
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_len:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return answer

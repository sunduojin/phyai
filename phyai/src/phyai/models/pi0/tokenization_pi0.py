"""pi0 PaliGemma prompt assembly and tokenization.

Unlike pi0.5, pi0 does not discretize robot state into the language
prompt. State stays a numeric float tensor and enters the action expert
through ``state_proj``. This module only builds task prompts and returns
``(input_ids, lang_lens)`` for the PaliGemma prefix.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from phyai.tokenizer import get_tokenizer

DEFAULT_TOKENIZER_NAME = "google/paligemma-3b-pt-224"


def get_pi0_tokenizer(
    name_or_path: str = DEFAULT_TOKENIZER_NAME, **kwargs: Any
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load the PaliGemma tokenizer pi0 was trained against."""

    return get_tokenizer(name_or_path, **kwargs)


def build_pi0_prompts(tasks: list[str]) -> list[str]:
    """Assemble pi0 task prompts.

    OpenPI / LeRobot pi0 tokenizes the task text itself after ensuring
    it ends with a newline. State is intentionally absent.
    """

    prompts: list[str] = []
    for task in tasks:
        prompts.append(task if task.endswith("\n") else f"{task}\n")
    return prompts


def tokenize_pi0_inputs(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    tasks: list[str],
    max_length: int = 48,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize pi0 task prompts.

    Returns:

    * ``input_ids``: ``(B, max_length)`` int64, right-padded.
    * ``lang_lens``: ``(B,)`` int64, real unpadded lengths.
    """

    prompts = build_pi0_prompts(tasks)
    encoded = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        padding_side="right",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(torch.int64)
    lang_lens = encoded["attention_mask"].sum(dim=-1).to(torch.int64)
    return input_ids, lang_lens


__all__ = [
    "DEFAULT_TOKENIZER_NAME",
    "build_pi0_prompts",
    "get_pi0_tokenizer",
    "tokenize_pi0_inputs",
]

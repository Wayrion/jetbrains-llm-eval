from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

from huggingface_hub import InferenceClient


DEFAULT_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_PROVIDER = os.environ.get("HF_PROVIDER") or "hf-inference"


# Common short aliases mapped to valid Hugging Face repo IDs
# Feel free to extend as needed.
_MODEL_ALIASES = {
    # Qwen family
    "qwen3-0.6b": "Qwen/Qwen2.5-0.5B-Instruct",  # closest available size on HF Inference API
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-coder-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def _normalize_model_id(model: str) -> str:
    key = (model or "").strip()
    # Normalize simple variations
    key_lower = key.lower()
    return _MODEL_ALIASES.get(key_lower, key)


def _apply_chat_template(messages: List[Dict[str, str]], tokenizer=None) -> str:
    """
    Minimal chat templating using Instruct-style models.
    If the model has a tokenizer with `apply_chat_template`, you'd use it. Here we build a simple prompt.
    """
    prompt_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            prompt_parts.append(f"[SYSTEM]\n{content}\n")
        elif role == "user":
            prompt_parts.append(f"[USER]\n{content}\n")
        else:
            prompt_parts.append(f"[ASSISTANT]\n{content}\n")
    prompt_parts.append("[ASSISTANT]\n")
    return "\n".join(prompt_parts)


@dataclass
class HFChatModel:
    model: str = DEFAULT_MODEL
    hf_api_token: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    provider: Optional[str] = DEFAULT_PROVIDER

    def __post_init__(self) -> None:
        token = self.hf_api_token or os.environ.get("HF_API_TOKEN")
        # Normalize user-provided model aliases to valid HF repo IDs
        self.model = _normalize_model_id(self.model)
        # Prefer explicit provider to avoid provider-mapping issues on some models
        provider_kwargs = {}
        if self.provider:
            # provider parameter expects specific Literal values in typeshed; ignore type here
            provider_kwargs["provider"] = self.provider  # type: ignore[arg-type]
        self.client = InferenceClient(model=self.model, token=token, **provider_kwargs)

    def invoke(
        self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None
    ) -> str:
        prompt = _apply_chat_template(messages)
        # Use text-generation endpoint with deterministic sampling
        try:
            response = self.client.text_generation(
                prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                # Use new 'stop' parameter (stop_sequences is deprecated)
                stop=stop or [],
                return_full_text=False,
            )
        except StopIteration:
            # Workaround for empty provider mapping: retry with explicit hf-inference provider
            token = self.hf_api_token or os.environ.get("HF_API_TOKEN")
            self.client = InferenceClient(
                model=self.model, token=token, provider="hf-inference"
            )
            response = self.client.text_generation(
                prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                stop=stop or [],
                return_full_text=False,
            )
        return response

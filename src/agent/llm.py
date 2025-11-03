from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


DEFAULT_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


# Common short aliases mapped to valid Hugging Face repo IDs
_MODEL_ALIASES = {
    # Qwen family
    "qwen3-0.6b": "Qwen/Qwen2.5-0.5B-Instruct",  # closest available instruct size
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-coder-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def _normalize_model_id(model: str) -> str:
    key = (model or "").strip()
    key_lower = key.lower()
    return _MODEL_ALIASES.get(key_lower, key)


def _apply_chat_template_fallback(messages: List[Dict[str, str]]) -> str:
    # Simple prompt builder if tokenizer.apply_chat_template is not available
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}\n")
        elif role == "user":
            parts.append(f"[USER]\n{content}\n")
        else:
            parts.append(f"[ASSISTANT]\n{content}\n")
    parts.append("[ASSISTANT]\n")
    return "\n".join(parts)


@dataclass
class HFChatModel:
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    device_map: Optional[str] = "auto"  # use GPU if available
    trust_remote_code: bool = True

    def __post_init__(self) -> None:
        # Normalize user-provided model aliases to valid HF repo IDs
        self.model = _normalize_model_id(self.model)
        # Load tokenizer and model locally (cached under ~/.cache/huggingface)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
        )
        # Holds usage stats from the last generation
        self.last_usage: Dict[str, Any] = {}

    def invoke(
        self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None
    ) -> str:
        # Prefer the tokenizer's chat template when available
        if hasattr(self.tokenizer, "apply_chat_template"):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self._hf_model.device)
        else:
            prompt = _apply_chat_template_fallback(messages)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self._hf_model.device
            )
        # Build generation kwargs and avoid passing sampling args for greedy decoding
        do_sample = (self.temperature or 0.0) > 0.0
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(1e-8, float(self.temperature))
            gen_kwargs["top_p"] = float(self.top_p)

        # Prefer updating a cloned GenerationConfig to avoid passing unsupported kwargs
        # Some HF versions don't provide a .clone() and generation_config may be None,
        # so build a local copy via GenerationConfig(**orig.to_dict()) or a fresh one.
        orig_gen_config = getattr(self._hf_model, "generation_config", None)
        try:
            if orig_gen_config is not None and hasattr(orig_gen_config, "to_dict"):
                gen_config = GenerationConfig(**orig_gen_config.to_dict())
            else:
                gen_config = GenerationConfig()
        except Exception:
            gen_config = GenerationConfig()

        for k, v in list(gen_kwargs.items()):
            if hasattr(gen_config, k):
                setattr(gen_config, k, v)
            else:
                # Drop unsupported keys silently to avoid transformer warnings
                gen_kwargs.pop(k, None)

        gen_outputs = self._hf_model.generate(
            **inputs,
            generation_config=gen_config,
        )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = gen_outputs[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Populate simple usage metrics
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion_tokens = int(generated_ids.shape[-1])
        total_tokens = prompt_tokens + completion_tokens
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        # Apply string-based stop sequences if provided
        if stop:
            cut_idx = min(
                [(text.find(s) if text.find(s) != -1 else len(text)) for s in stop]
            )
            text = text[:cut_idx]
        return text.strip()

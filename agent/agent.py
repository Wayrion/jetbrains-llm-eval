"""Simple ReAct-style agent (fallback) and LLM interface.

This module provides:
- LLMClient: wrapper around Hugging Face Inference API (configurable by env var HF_API_TOKEN)
- ReActAgent: simple loop that prompts the model and can call the code executor tool.

If `langgraph` is available in your environment you can write a small adapter that
wraps ReActAgent methods into LangGraph tools/agents. See README for notes.
"""

import os
import re
import requests
from typing import Optional

# load .env automatically so HF_API_TOKEN set in a .env file is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # python-dotenv is optional at import time; if not installed the user can set env vars manually
    pass

from huggingface_hub import InferenceClient


class LLMClient:
    def __init__(self, model: str = "qwen3-0.6b", hf_token: Optional[str] = None):
        token = hf_token or os.environ.get("HF_API_TOKEN")
        if token is None:
            raise RuntimeError(
                "Set HF_API_TOKEN environment variable for Hugging Face Inference API access"
            )
        self.model = model
        # Use the newer InferenceClient (token-only). The model name is passed on each call.
        self.client = InferenceClient(token=token)

    def generate(
        self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        # Call the HF Inference client with a few fallback signatures to
        # support multiple installed versions of `huggingface_hub`.
        def try_call(fn):
            # Try a few call signatures for the function `fn`.
            try:
                return fn(
                    model=self.model,
                    inputs=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except TypeError:
                pass
            try:
                return fn(self.model, prompt, max_new_tokens, temperature)
            except TypeError:
                pass
            try:
                return fn(
                    model=self.model,
                    inputs=prompt,
                    parameters={
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                )
            except TypeError:
                pass
            # give up for this fn
            raise TypeError("No compatible signature for provided function")

        out = None
        # candidate callables on the client
        candidates = []
        if hasattr(self.client, "text_generation"):
            candidates.append(self.client.text_generation)
        if hasattr(self.client, "generate"):
            candidates.append(self.client.generate)
        if hasattr(self.client, "__call__"):
            candidates.append(self.client.__call__)
        # try each candidate until one works
        last_err = None
        for fn in candidates:
            try:
                out = try_call(fn)
                break
            except Exception as e:
                last_err = e
                continue
        # If no client method succeeded, try a plain HTTP call to the HF Inference API
        if out is None:
            # try HTTP fallback using requests (works even if huggingface_hub API differs)
            token = os.environ.get("HF_API_TOKEN")
            if token:
                try:
                    headers = {"Authorization": f"Bearer {token}"}
                    url = f"https://api-inference.huggingface.co/models/{self.model}"
                    payload = {
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                        },
                    }
                    resp = requests.post(url, headers=headers, json=payload, timeout=30)
                    # treat non-2xx responses as errors
                    if resp.status_code < 200 or resp.status_code >= 300:
                        raise RuntimeError(
                            f"HF HTTP error {resp.status_code}: {resp.text[:200]}"
                        )
                    try:
                        out = resp.json()
                    except Exception:
                        out = resp.text
                except Exception:
                    # if HTTP fallback fails, raise the last client error if available
                    if last_err is not None:
                        raise last_err
                    raise
            else:
                # no token available; raise a clearer error that guides the user
                hint = (
                    "No HF_API_TOKEN found in environment. "
                    "Set HF_API_TOKEN in your environment or in a .env file. "
                    "See README.md for instructions."
                )
                if last_err is not None:
                    raise RuntimeError(f"{hint} Last client error: {last_err}")
                raise RuntimeError(hint)

        if out is None:
            return ""
        if isinstance(out, list) and len(out) > 0:
            first = out[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
            return str(first)
        if isinstance(out, dict):
            if "generated_text" in out:
                return out["generated_text"]
            if "text" in out:
                return out["text"]
            return str(out)
        if hasattr(out, "generated_text"):
            return getattr(out, "generated_text")
        if hasattr(out, "text"):
            return getattr(out, "text")
        return str(out)


class ReActAgent:
    """A compact ReAct-style loop that can call a code execution tool.

    For simplicity this implementation performs a single generation asking the model
    to produce a corrected implementation in a single response. It supports multi-step
    reasoning if you design the prompt to request thoughts and actions, but the loop
    itself is a single-shot generator to keep runs efficient.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def _wrap_prompt(self, prompt: str) -> str:
        # Ask the model to provide the fixed function implementation as Python code in
        # a fenced ```python``` block and nothing else.
        instruction = (
            "You are a developer AI. The user provides a buggy Python function and unit tests.\n"
            "Return only the fixed Python function implementation (no explanation).\n"
            "Wrap code in a ```python``` fenced code block.\n"
        )
        return instruction + "\n" + prompt

    def generate_fix(self, prompt: str, max_new_tokens: int = 512) -> str:
        wrapped = self._wrap_prompt(prompt)
        text = self.llm.generate(
            wrapped, max_new_tokens=max_new_tokens, temperature=0.0
        )
        # extract first fenced python block if present
        m = re.search(r"```python\n(.+?)```", text, flags=re.S)
        if m:
            return m.group(1).strip()
        # fallback: return full text
        return text.strip()

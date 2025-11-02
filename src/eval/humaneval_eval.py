from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable, cast
from itertools import islice

from datasets import load_dataset

from ..agent import HFChatModel, build_graph, AgentConfig


@dataclass
class EvalConfig:
    model: str = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    max_problems: Optional[int] = None
    max_iters: int = 3
    temperature: float = 0.0
    max_new_tokens: int = 512
    provider: Optional[str] = os.environ.get("HF_PROVIDER") or "hf-inference"
    dataset_repo: str = os.environ.get("DATASET_REPO", "bigcode/humanevalpack")
    dataset_subset: str = os.environ.get("DATASET_SUBSET", "python")
    dataset_retries: int = int(os.environ.get("DATASET_RETRIES", "3"))


def extract_fields(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Humanevalpack samples typically have fields: prompt, test, entry_point (and sometimes imports).
    We normalize to the strings we need.
    """
    prompt = sample.get("prompt") or sample.get("question") or ""
    tests = sample.get("test") or sample.get("tests") or sample.get("test_code") or ""
    entry_point = sample.get("entry_point") or sample.get("entrypoint") or ""
    return {"prompt": prompt, "tests": tests, "entry_point": entry_point}


def _configure_hf_timeouts() -> None:
    # Increase HF Hub HTTP timeout to reduce ReadTimeouts on slow networks
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "60")
    # Enable parallel/optimized transfer if available
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _load_humaneval_with_retries(repo: str, subset: str, retries: int, verbose: bool):
    last_err: Optional[Exception] = None
    # First try: standard loading of the dataset dict then take test split
    for attempt in range(1, max(1, retries) + 1):
        try:
            ds = load_dataset(repo, subset)
            return ds["test"]
        except Exception as e:  # noqa: BLE001 - user environment/network issues
            last_err = e
            if verbose:
                print(f"Dataset load attempt {attempt} failed: {e}")
            time.sleep(min(2**attempt, 8))

    # Fallback: try streaming the test split directly to avoid large metadata calls
    try:
        if verbose:
            print("Falling back to streaming dataset split due to repeated timeouts…")
        split = load_dataset(repo, subset, split="test", streaming=True)
        return split
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"Streaming fallback failed: {e}")
        # Re-raise last error from non-streaming attempts if available
        raise (last_err or e)


def run_pass_at_1(
    cfg: EvalConfig, out_path: Optional[str] = None, verbose: bool = False
) -> Dict[str, Any]:
    _configure_hf_timeouts()
    split = _load_humaneval_with_retries(
        cfg.dataset_repo, cfg.dataset_subset, cfg.dataset_retries, verbose
    )

    # Build an iterable over tasks honoring optional max_problems across Dataset or streaming
    iterator: Iterable[Any]
    estimated_total: Optional[int] = getattr(split, "num_rows", None)
    if cfg.max_problems is not None:
        # Limit number of problems
        if hasattr(split, "select"):
            # Dataset case
            total_for_limit = (
                int(estimated_total)
                if estimated_total is not None
                else cfg.max_problems
            )
            limit = min(cfg.max_problems, total_for_limit)
            split_limited = cast(Any, split).select(range(limit))
            iterator = split_limited
            estimated_total = limit
        else:
            # Iterable / streaming case
            limit = cfg.max_problems
            iterator = islice(cast(Iterable[Any], split), limit)
            estimated_total = limit
    else:
        iterator = cast(Iterable[Any], split)

    # Build agent graph and config
    graph = build_graph().compile()
    llm = HFChatModel(
        model=cfg.model,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        provider=cfg.provider,
    )
    agent_cfg = AgentConfig(
        max_iters=cfg.max_iters,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )

    results = []
    n_pass = 0

    n_processed = 0
    for idx, sample in enumerate(iterator):
        meta = extract_fields(sample)
        if verbose:
            print(f"[{idx}] task_id={sample.get('task_id') or sample.get('name')}")
        state = {
            "llm": llm,
            "config": agent_cfg,
            "prompt": meta["prompt"],
            "tests": meta["tests"],
            "entry_point": meta["entry_point"],
            "iters": 0,
        }
        t0 = time.time()
        out_state = graph.invoke(state)
        dt = time.time() - t0
        last = out_state.get("last_result", {})
        passed = bool(last.get("passed"))
        n_pass += 1 if passed else 0
        result = {
            "task_id": sample.get("task_id") or sample.get("name") or idx,
            "passed": passed,
            "exit_code": last.get("exit_code"),
            "stderr": last.get("stderr"),
            "stdout": last.get("stdout"),
            "runtime_sec": dt,
            "completion": out_state.get("final_code") or out_state.get("code"),
        }
        results.append(result)
        if verbose:
            print("  passed=", passed, "time=", round(dt, 2), "s")
        n_processed += 1

    if estimated_total is not None:
        total = int(estimated_total)
    else:
        total = n_processed
    pass_at_1 = n_pass / total if total > 0 else 0.0

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    if verbose:
        print(f"pass@1 = {pass_at_1:.4f} ({n_pass}/{total})")
    return {"pass@1": pass_at_1, "n": total, "results": results}

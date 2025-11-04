from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
import logging
from typing import Dict, Any, Optional, Iterable, cast, List
from itertools import islice

from datasets import load_dataset

from ..agent import HFChatModel, build_graph, AgentConfig


@dataclass
class EvalConfig:
    model: str = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    max_problems: Optional[int] = None
    temperature: float = 0.0
    max_new_tokens: int = 512
    dataset_path: Optional[str] = os.environ.get("DATASET_PATH")
    dataset_repo: str = os.environ.get("DATASET_REPO", "bigcode/humanevalpack")
    dataset_subset: str = os.environ.get("DATASET_SUBSET", "python")
    dataset_retries: int = int(os.environ.get("DATASET_RETRIES", "3"))
    # Optional reflection/repair loops after the first execute; 0 = strict pass@1
    iters: int = int(os.environ.get("ITERS", "0"))
    sandbox: str = os.environ.get("SANDBOX_MODE", "process")
    sandbox_timeout: int = int(os.environ.get("SANDBOX_TIMEOUT", "10"))


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
    # Avoid tokenizer parallelism fork warnings when the sandbox spins up helpers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_humaneval_with_retries(repo: str, subset: str, retries: int, verbose: bool):
    last_err: Optional[Exception] = None
    # First try: standard loading of the dataset dict then take test split
    for attempt in range(1, max(1, retries) + 1):
        try:
            ds = load_dataset(repo, subset)
            return ds["test"]
        except Exception as e:  # noqa: BLE001 - user environment/network issues
            last_err = e
            logging.info("Dataset load attempt %s failed: %s", attempt, e)
            time.sleep(min(2**attempt, 8))

    # Fallback: try streaming the test split directly to avoid large metadata calls
    try:
        logging.info(
            "Falling back to streaming dataset split due to repeated timeouts…"
        )
        split = load_dataset(repo, subset, split="test", streaming=True)
        return split
    except Exception as e:  # noqa: BLE001
        logging.info("Streaming fallback failed: %s", e)
        # Re-raise last error from non-streaming attempts if available
        raise (last_err or e)


def _project_root() -> str:
    # src/eval/humaneval_eval.py -> project root two levels up
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", ".."))


def _load_local_dataset_if_available(path: Optional[str], verbose: bool):
    # Prefer a local parquet if provided or at dataset/humaneval_py.parquet
    candidate = path
    if not candidate:
        candidate = os.path.join(_project_root(), "dataset", "humaneval_py.parquet")
    if candidate and os.path.exists(candidate):
        logging.info("Loading local dataset parquet at %s", candidate)
        dsd = load_dataset("parquet", data_files={"test": candidate})
        return dsd["test"]
    return None


def run_pass_at_1(
    cfg: EvalConfig,
    out_path: Optional[str] = None,
    verbose: bool = False,
    existing_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    logging.info("Starting Humaneval pass@1 evaluation…")
    logging.info("Sandbox mode: %s", cfg.sandbox)
    # Prefer local dataset if present
    split = _load_local_dataset_if_available(cfg.dataset_path, verbose)
    if split is None:
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

    # Build a plain list of jobs for potential parallel execution
    jobs: List[Dict[str, Any]] = []
    if hasattr(split, "__iter__"):
        for idx, sample in enumerate(iterator):
            meta = extract_fields(sample)
            task_id = sample.get("task_id") or sample.get("name") or idx
            jobs.append(
                {
                    "task_id": task_id,
                    "prompt": meta["prompt"],
                    "tests": meta["tests"],
                    "entry_point": meta["entry_point"],
                }
            )
    else:
        # Shouldn't happen, but keep a safe default
        pass

    existing_map: Dict[str, Dict[str, Any]] = {}
    if existing_results:
        for item in existing_results:
            task_id = item.get("task_id")
            if task_id is None:
                continue
            existing_map[str(task_id)] = item

    results_map: Dict[str, Dict[str, Any]] = {}

    # Determine which jobs still need to run
    pending_jobs = [job for job in jobs if str(job["task_id"]) not in existing_map]

    graph = build_graph().compile() if pending_jobs else None
    llm = (
        HFChatModel(
            model=cfg.model,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
        )
        if pending_jobs
        else None
    )
    agent_cfg = (
        AgentConfig(
            max_iters=max(0, int(cfg.iters)),
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            sandbox_mode=cfg.sandbox,
            sandbox_timeout=max(1, int(cfg.sandbox_timeout)),
        )
        if pending_jobs
        else None
    )

    if pending_jobs and existing_map:
        logging.info(
            "Resuming run: %s task(s) will be executed, %s already completed.",
            len(pending_jobs),
            len(existing_map),
        )
    elif pending_jobs:
        logging.info("Starting fresh run with %s task(s).", len(pending_jobs))
    elif existing_map:
        logging.info("All tasks already completed in existing results; nothing to do.")

    for j in jobs:
        task_key = str(j["task_id"])
        if task_key in existing_map:
            res = existing_map[task_key]
            logging.info(
                "[%s] skipping (resume) passed=%s", j["task_id"], res.get("passed")
            )
            results_map[task_key] = res
            continue

        assert graph is not None and llm is not None and agent_cfg is not None
        logging.info("[%s] starting", j["task_id"])
        state = {
            "llm": llm,
            "config": agent_cfg,
            "prompt": j["prompt"],
            "tests": j["tests"],
            "entry_point": j["entry_point"],
            "iters": 0,
        }
        t0 = time.time()
        out_state = graph.invoke(state)
        dt = time.time() - t0
        last = out_state.get("last_result", {})
        passed = bool(last.get("passed"))
        res = {
            "task_id": j["task_id"],
            "passed": passed,
            "exit_code": last.get("exit_code"),
            "stderr": last.get("stderr"),
            "stdout": last.get("stdout"),
            "runtime_sec": dt,
            "completion": out_state.get("final_code") or out_state.get("code"),
        }
        metrics = out_state.get("metrics") or {}
        if metrics:
            res["timings_sec"] = {
                k: float(metrics.get(k, 0.0))
                for k in ("t_propose_sec", "t_execute_sec", "t_reflect_sec")
            }
            token_keys = [
                "propose_prompt_tokens",
                "propose_completion_tokens",
                "reflect_prompt_tokens",
                "reflect_completion_tokens",
            ]
            token_usage = {
                k: int(metrics.get(k, 0)) for k in token_keys if k in metrics
            }
            if token_usage:
                res["token_usage"] = token_usage
        res["iters"] = int(out_state.get("iters", 0))
        results_map[task_key] = res
        logging.info(
            "[%s] passed=%s time=%.2fs breakdown=%s",
            j["task_id"],
            passed,
            round(dt, 2),
            {
                "propose": round(float(metrics.get("t_propose_sec", 0.0)), 2),
                "execute": round(float(metrics.get("t_execute_sec", 0.0)), 2),
                "reflect": round(float(metrics.get("t_reflect_sec", 0.0)), 2),
            },
        )

    # Assemble ordered results following the dataset order
    ordered_results: List[Dict[str, Any]] = []
    missing_task_ids: List[Any] = []
    for j in jobs:
        task_key = str(j["task_id"])
        if task_key in results_map:
            ordered_results.append(results_map[task_key])
        else:
            missing_task_ids.append(j["task_id"])

    if missing_task_ids:
        logging.error(
            "Missing results for task ids: %s", ", ".join(map(str, missing_task_ids))
        )

    n_pass = sum(1 for r in ordered_results if r.get("passed"))

    if estimated_total is not None:
        total = int(estimated_total)
    else:
        total = len(ordered_results)
    pass_at_1 = n_pass / total if total > 0 else 0.0

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for r in ordered_results:
                f.write(json.dumps(r) + "\n")

    logging.info("pass@1 = %.4f (%s/%s)", pass_at_1, n_pass, total)
    return {"pass@1": pass_at_1, "n": total, "results": ordered_results}

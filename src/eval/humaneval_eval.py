"""Humaneval pass@1 evaluation harness coordinating the ReAct agent."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO, cast

from datasets import load_dataset

from ..agent import AgentConfig, HFChatModel, build_graph
from ..agent.llm import _normalize_model_id


LOG = logging.getLogger(__name__)


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


def extract_fields(sample: dict[str, object]) -> dict[str, str]:
    """Return standardized prompt/test/entry point fields from dataset rows."""
    prompt = str(sample.get("prompt") or sample.get("question") or "")
    tests = str(
        sample.get("test") or sample.get("tests") or sample.get("test_code") or ""
    )
    entry_point = str(sample.get("entry_point") or sample.get("entrypoint") or "")
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
        except Exception as exc:  # noqa: BLE001 - user environment/network issues
            last_err = exc
            LOG.info("Dataset load attempt %s failed: %s", attempt, exc)
            time.sleep(min(2**attempt, 8))

    # Fallback: try streaming the test split directly to avoid large metadata calls
    try:
        LOG.info("Falling back to streaming dataset split due to repeated timeouts…")
        split = load_dataset(repo, subset, split="test", streaming=True)
        return split
    except Exception as exc:  # noqa: BLE001
        LOG.info("Streaming fallback failed: %s", exc)
        # Re-raise last error from non-streaming attempts if available
        raise (last_err or exc)


def _project_root() -> Path:
    # src/eval/humaneval_eval.py -> project root two levels up
    return Path(__file__).resolve().parents[2]


def _load_local_dataset_if_available(path: Optional[str], verbose: bool):
    # Prefer a local parquet if provided or at dataset/humaneval_py.parquet
    candidate = (
        Path(path) if path else _project_root() / "dataset" / "humaneval_py.parquet"
    )
    if candidate.exists():
        LOG.info("Loading local dataset parquet at %s", candidate)
        dsd = load_dataset("parquet", data_files={"test": str(candidate)})
        return dsd["test"]
    return None


def run_pass_at_1(
    cfg: EvalConfig,
    out_path: Optional[str] = None,
    verbose: bool = False,
    existing_results: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    LOG.info("Starting Humaneval pass@1 evaluation…")
    LOG.info("Sandbox mode: %s", cfg.sandbox)
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
    jobs: list[dict[str, Any]] = []
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

    existing_map: dict[str, dict[str, Any]] = {}
    if existing_results:
        for item in existing_results:
            task_id = item.get("task_id")
            if task_id is None:
                continue
            existing_map[str(task_id)] = item

    results_map: dict[str, dict[str, Any]] = {}

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
    resolved_model_id = llm.model if llm is not None else _normalize_model_id(cfg.model)
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
        LOG.info(
            "Resuming run: %s task(s) will be executed, %s already completed.",
            len(pending_jobs),
            len(existing_map),
        )
    elif pending_jobs:
        LOG.info("Starting fresh run with %s task(s).", len(pending_jobs))
    elif existing_map:
        LOG.info("All tasks already completed in existing results; nothing to do.")

    output_handle: Optional[TextIO] = None
    append_newline_before_first_write = False
    try:
        if out_path:
            parent = Path(out_path).parent
            if parent:
                parent.mkdir(parents=True, exist_ok=True)

            if pending_jobs:
                mode = "a" if existing_map else "w"
                if mode == "a" and os.path.exists(out_path):
                    try:
                        with open(out_path, "rb") as existing_file:
                            existing_file.seek(0, os.SEEK_END)
                            if existing_file.tell() > 0:
                                existing_file.seek(-1, os.SEEK_END)
                                last_char = existing_file.read(1)
                                append_newline_before_first_write = last_char not in (
                                    b"\n",
                                    b"\r",
                                )
                    except OSError:
                        append_newline_before_first_write = False
                output_handle = open(out_path, mode, encoding="utf-8")
            else:
                if not existing_map and not os.path.exists(out_path):
                    with open(out_path, "w", encoding="utf-8"):
                        pass

        for j in jobs:
            task_key = str(j["task_id"])
            if task_key in existing_map:
                res = dict(existing_map[task_key])
                if "model" not in res:
                    res["model"] = resolved_model_id
                if (
                    cfg.model
                    and res.get("model_alias") is None
                    and cfg.model != res.get("model")
                ):
                    res["model_alias"] = cfg.model
                LOG.info(
                    "[%s] skipping (resume) passed=%s",
                    j["task_id"],
                    res.get("passed"),
                )
                results_map[task_key] = res
                continue

            assert graph is not None and llm is not None and agent_cfg is not None
            LOG.info("[%s] starting", j["task_id"])
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
            # Preserve normalized/alias pair so downstream tooling can label plots precisely
            res["model"] = resolved_model_id
            if cfg.model and cfg.model != resolved_model_id:
                res["model_alias"] = cfg.model
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
            if output_handle is not None:
                if append_newline_before_first_write:
                    output_handle.write("\n")
                    append_newline_before_first_write = False
                output_handle.write(json.dumps(res) + "\n")
                output_handle.flush()
            LOG.info(
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
    finally:
        if output_handle is not None:
            output_handle.flush()
            output_handle.close()

    # Assemble ordered results following the dataset order
    ordered_results: list[dict[str, Any]] = []
    missing_task_ids: list[Any] = []
    for j in jobs:
        task_key = str(j["task_id"])
        if task_key in results_map:
            ordered_results.append(results_map[task_key])
        else:
            missing_task_ids.append(j["task_id"])

    if missing_task_ids:
        LOG.error(
            "Missing results for task ids: %s", ", ".join(map(str, missing_task_ids))
        )

    n_pass = sum(1 for r in ordered_results if r.get("passed"))

    if estimated_total is not None:
        total = int(estimated_total)
    else:
        total = len(ordered_results)
    pass_at_1 = n_pass / total if total > 0 else 0.0

    LOG.info("pass@1 = %.4f (%s/%s)", pass_at_1, n_pass, total)
    payload: dict[str, Any] = {
        "pass@1": pass_at_1,
        "n": total,
        "model": resolved_model_id,
        "results": ordered_results,
    }
    if cfg.model and cfg.model != resolved_model_id:
        payload["model_alias"] = cfg.model
    return payload

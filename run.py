from __future__ import annotations

import argparse
import json

import os
import multiprocessing as mp
import logging

from src.eval.humaneval_eval import EvalConfig, run_pass_at_1


def main() -> None:
    # Configure logging once, default INFO
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # Align transformers verbosity with request
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
    p = argparse.ArgumentParser(description="Run ReAct agent on humanevalpack (pass@1)")
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "HF model id (e.g. 'Qwen/Qwen2.5-0.5B-Instruct'). Common aliases like 'qwen3-0.6b' also work."
        ),
    )
    p.add_argument(
        "--max",
        dest="max_problems",
        type=int,
        default=None,
        help="Max problems to evaluate",
    )
    p.add_argument(
        "--iters",
        dest="iters",
        type=int,
        default=0,
        help="Optional number of repair loops after first execute (0 = strict pass@1)",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local humaneval parquet (e.g., ./dataset/humaneval_py.parquet). If provided, bypasses HF Hub.",
    )
    p.add_argument("--out", type=str, default=None, help="Path to JSONL results output")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1")),
        help="Number of parallel workers (processes)",
    )
    args = p.parse_args()

    # Reduce HF tokenizers fork warnings and use spawn for safety
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set; ignore
        pass

    cfg = EvalConfig(
        model=args.model,
        max_problems=args.max_problems,
        temperature=0.0,
        max_new_tokens=512,
        dataset_path=args.dataset_path or None,
        iters=max(0, int(args.iters)),
        workers=max(1, int(args.workers)),
    )
    res = run_pass_at_1(cfg, out_path=args.out, verbose=args.verbose)

    summary = {k: v for k, v in res.items() if k != "results"}
    logging.info("Summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()

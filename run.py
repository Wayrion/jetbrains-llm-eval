from __future__ import annotations

import argparse
import json

from src.eval.humaneval_eval import EvalConfig, run_pass_at_1


def main() -> None:
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
        "--iters", dest="max_iters", type=int, default=3, help="Max agent iterations"
    )
    p.add_argument(
        "--provider",
        type=str,
        default=None,
        help="HF inference provider (e.g. 'hf-inference' or 'tgi'); defaults to env HF_PROVIDER or 'hf-inference'",
    )
    p.add_argument("--out", type=str, default=None, help="Path to JSONL results output")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    args = p.parse_args()

    cfg = EvalConfig(
        model=args.model,
        max_problems=args.max_problems,
        max_iters=args.max_iters,
        temperature=0.0,
        max_new_tokens=512,
        provider=args.provider or None,
    )
    res = run_pass_at_1(cfg, out_path=args.out, verbose=args.verbose)

    summary = {k: v for k, v in res.items() if k != "results"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

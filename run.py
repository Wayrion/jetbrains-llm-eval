"""Entry point to run the evaluation quickly.

Usage: python run.py --model qwen3-0.6b --max 50
"""

import os
from evaluator import compute_pass_at_1


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--max", type=int, default=50)
    parser.add_argument("--split", default="test")
    parser.add_argument("--hf-token", default=os.environ.get("HF_API_TOKEN"))
    parser.add_argument(
        "--out", default=None, help="Path to write per-example JSONL results"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose per-example logging"
    )
    args = parser.parse_args()

    print("Running evaluation with model", args.model)
    stats = compute_pass_at_1(
        model_name=args.model,
        hf_token=args.hf_token,
        split=args.split,
        max_examples=args.max,
        out_path=args.out,
        verbose=args.verbose,
    )
    print("Results:", stats)


if __name__ == "__main__":
    main()

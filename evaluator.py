"""Evaluation harness for humanevalpack using the agent.

This script loads the dataset, runs the agent to generate one candidate per problem,
executes tests in the sandbox and computes pass@1.
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm
from agent.agent import LLMClient, ReActAgent
from agent.tools import run_submission


def extract_prompt_and_tests(sample: dict):
    # Heuristic: look for common fields used in Humaneval-like datasets
    prompt = None
    tests = None
    # common keys
    for k in ("prompt", "task", "problem", "function", "body"):
        if k in sample:
            prompt = sample[k]
            break
    for k in (
        "canonical_solution_tests",
        "tests",
        "test",
        "canonical_test",
        "canonical_tests",
    ):
        if k in sample:
            tests = sample[k]
            break
    # If prompt is missing, try joining available text fields
    if prompt is None:
        for v in sample.values():
            if isinstance(v, str) and len(v) > 30:
                prompt = v
                break
    return prompt, tests


def compute_pass_at_1(
    model_name: str = "qwen3-0.6b",
    hf_token: str | None = None,
    split: str = "test",
    max_examples: int = 100,
    out_path: str | None = None,
    verbose: bool = False,
):
    ds = load_dataset("bigcode/humanevalpack", "python", split=split)
    llm = LLMClient(model=model_name, hf_token=hf_token)
    agent = ReActAgent(llm)

    total = 0
    passed = 0

    out_f = None
    if out_path:
        out_f = open(out_path, "w", encoding="utf-8")

    for i, sample in enumerate(tqdm(ds)):
        if i >= max_examples:
            break
        prompt, tests = extract_prompt_and_tests(sample)
        if prompt is None or tests is None:
            # skip samples we can't parse
            continue
        # craft a combined prompt: prompt + tests
        combined = prompt + "\n\n# Unit tests:\n" + tests
        try:
            candidate = agent.generate_fix(combined)
        except Exception as e:
            print("LLM generation failed:", e)
            continue

        # sanitize candidate: remove markdown fences or leading commentary
        def _sanitize(c: str) -> str:
            if c is None:
                return ""
            s = c.strip()
            # remove fenced code blocks if present
            if "```" in s:
                # extract content between first pair of fences
                parts = s.split("```")
                # parts like ['', 'python\n...code...', ''] or similar
                for p in parts:
                    if p.strip().startswith("python"):
                        # remove optional 'python' marker
                        return p.split("\n", 1)[1] if "\n" in p else p
                # fallback: take second part
                if len(parts) >= 2:
                    return parts[1]
            # if not fenced, try to find first 'def ' and return from there
            idx = s.find("def ")
            if idx != -1:
                return s[idx:]
            return s

        candidate = _sanitize(candidate)

        # quick compile check to avoid writing invalid submission files
        try:
            compile(candidate, "submission.py", "exec")
        except Exception:
            import traceback

            result = {
                "passed": False,
                "error": "CompileError: " + traceback.format_exc(),
                "stdout": "",
                "stderr": "",
            }
            total += 1
            # write and continue
            if out_f is not None:
                rec = {
                    "index": i,
                    "id": sample.get("id") or sample.get("task_id") or None,
                    "prompt": prompt,
                    "tests": tests,
                    "candidate": candidate,
                    "runner": result,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if verbose:
                print(f"[{i}] passed={result.get('passed')} error=CompileError")
            continue

        result = run_submission(candidate, tests)
        total += 1
        if result.get("passed"):
            passed += 1

        if out_f is not None:
            rec = {
                "index": i,
                "id": sample.get("id") or sample.get("task_id") or None,
                "prompt": prompt,
                "tests": tests,
                "candidate": candidate,
                "runner": result,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if verbose:
            print(f"[{i}] passed={result.get('passed')} error={result.get('error')}")

    if out_f is not None:
        out_f.close()

    pass_at_1 = passed / total if total > 0 else 0.0
    return {"pass@1": pass_at_1, "total": total, "passed": passed}


if __name__ == "__main__":
    # basic CLI
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--hf-token", default=os.environ.get("HF_API_TOKEN"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max", type=int, default=100)
    parser.add_argument(
        "--out", default=None, help="Path to write per-example JSONL results"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose per-example logging"
    )
    args = parser.parse_args()
    stats = compute_pass_at_1(
        model_name=args.model,
        hf_token=args.hf_token,
        split=args.split,
        max_examples=args.max,
        out_path=args.out,
        verbose=args.verbose,
    )
    print(stats)

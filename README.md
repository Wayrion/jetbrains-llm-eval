## LLM agent to fix Python code and evaluate on Humaneval (pass@1)

This repo provides a production-ready scaffold using LangGraph to run a ReAct-style agent that:

- Uses a small instruction-tuned model (default: Qwen/Qwen2.5-0.5B-Instruct via Hugging Face Inference API)
- Has a sandboxed Python “code interpreter” tool to execute unit tests safely
- Evaluates on BigCode’s `bigcode/humanevalpack` (python) using pass@1

What’s included
- `src/agent/llm.py` — lightweight HF Inference API chat wrapper
- `src/agent/sandbox.py` — sandboxed subprocess runner with CPU/memory/timeout and network-block stubs
- `src/agent/react_agent.py` — LangGraph ReAct loop: propose -> execute -> reflect
- `src/eval/humaneval_eval.py` — evaluation harness for HumanevalPack, producing JSONL results and pass@1
- `run.py` — CLI to run the benchmark end-to-end

Requirements
- Python 3.13+
- A Hugging Face account and token with access to serverless inference and the dataset

Quick start

1) Create and activate a virtual environment and install deps

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

2) Authenticate for dataset and inference

```bash
huggingface-cli login          # to download humanevalpack
export HF_API_TOKEN="hf_..."  # to call the HF Inference API
```

3) Run the evaluation

```bash
# Evaluate on first 50 problems with Qwen2.5 0.5B Instruct
python run.py --model Qwen/Qwen2.5-0.5B-Instruct --max 50 --out results.jsonl --verbose
```

This prints a summary JSON with `pass@1` and writes per-problem results to `results.jsonl`.

CLI options
- `--model` Hugging Face model id; any instruct/chat text-generation model is supported by Inference API. Short aliases are also accepted, e.g. `qwen3-0.6b` -> `Qwen/Qwen2.5-0.5B-Instruct`.
- `--provider` HF inference provider (defaults to `hf-inference`). Set via `--provider` or `HF_PROVIDER` env. If you run into provider-mapping errors, try `--provider hf-inference` explicitly.
- `--max` limit number of problems (default: all in split)
- `--iters` max agent self-reflection iterations (default 3)
- `--out` path to JSONL output with per-task results
- `--verbose` progress logs

Agent contract
- Input: `prompt` from humanevalpack, `tests`, `entry_point`
- Output: full Python implementation defining the required function(s)
- Tool: `run_python_with_tests(code, tests, entry_point)` executes in sandbox; returns pass/fail and logs
- Termination: stop when tests pass or iteration budget is exhausted

Sandbox notes
- Runs in a temporary directory with `python -I` (isolated), enforces timeout, CPU and memory limits via `resource`
- Disables network by overriding `socket` creation in the child process
- Restricts file open calls to the sandbox folder
- For stricter isolation in production, consider a containerized runner (Docker/Firecracker) or OS-level sandboxes

Pass@1 metric
- We generate a single candidate per task; pass@1 is the fraction of tasks whose tests pass on the first attempt
- Configure deterministic generation by keeping `temperature=0.0`

Reproducing results
Run the provided command above. Your score will depend on the model and the subset size. For a quick smoke test, try `--max 5`.

Troubleshooting
- 401/403 from HF Inference: ensure `HF_API_TOKEN` is exported and has sufficient permissions
- Dataset download issues: make sure you’re logged in `huggingface-cli login`
- Slow or OOM inference locally: this implementation uses serverless inference; if you want local inference, replace the LLM in `src/agent/llm.py` with a transformers pipeline and install `torch`/`transformers`


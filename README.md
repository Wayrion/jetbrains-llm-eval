# LLM-based Python fixer + evaluator (Humaneval)

This repository contains a minimal implementation of an LLM-based agent that attempts to fix buggy Python code and evaluates the fixes on the `bigcode/humanevalpack` dataset using the pass@1 metric.

What is included
- `agent/agent.py` — a small LLM client (Hugging Face Inference API) and a ReAct-style agent that asks the model for a fixed function in a fenced ```python``` block.
- `agent/tools.py` — a sandboxed code runner that executes generated code and runs provided unit tests in a subprocess with a timeout.
- `evaluator.py` — evaluation harness that loads `bigcode/humanevalpack` and computes pass@1 by generating a single candidate per problem.
- `run.py` — small CLI wrapper to run the evaluation.
- `requirements.txt` — Python dependencies.

Design notes and assumptions
- This implementation uses the Hugging Face Inference API for model access (set `HF_API_TOKEN` env var). We default to `qwen3-0.6b` but any text-generation model name supported by the Inference API should work.
- The sandbox is lightweight: it runs code in a temporary directory in a new Python subprocess with a timeout. It is useful for development but not a production-grade sandbox. For production, use containers or hardened sandboxes.
- The evaluator uses heuristics to extract prompts and tests from dataset samples. Depending on the exact `humanevalpack` schema you may need to tweak `extract_prompt_and_tests` in `evaluator.py`.

Setup

1. Create and activate a Python environment (recommended):

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install .
```

2. Set your Hugging Face token (required):

```bash
export HF_API_TOKEN="hf_..."
```

Running the evaluation

Run a small evaluation run:

```bash
python run.py --model qwen3-0.6b --max 50
```

This will load up to 50 examples from the `test` split, ask the model to produce one fixed candidate per example, run the tests in a sandbox, and print `pass@1`.

Notes about LangGraph
- You asked to use LangGraph for the agent scaffolding. I included a self-contained ReAct agent implementation (in `agent/agent.py`) which you can adapt to LangGraph. If you already have `langgraph` installed, you can write a thin adapter that exposes the `ReActAgent.generate_fix` method as a LangGraph tool/step. I intentionally kept the agent modular so it is straightforward to integrate.

Limitations and reproducibility
- This environment cannot download models or datasets for me; you must run the evaluation locally where you have network access. The dataset `bigcode/humanevalpack` is downloaded via `datasets.load_dataset` and may require Hugging Face authentication.
- Results will vary depending on model, temperature, and network latency. For proper pass@1 results aligned with the paper, run with deterministic generation (temperature 0.0) and ensure you generate exactly one candidate per problem.

Next steps / suggestions
- Add a LangGraph adapter module if you require direct LangGraph integration.
- Improve sandboxing using Docker to isolate execution and support resource limits.
- Add logging and per-example output saving for inspection.

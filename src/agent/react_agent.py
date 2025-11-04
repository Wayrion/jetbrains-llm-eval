from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, NotRequired
import time
import logging

from langgraph.graph import StateGraph, END

from .llm import HFChatModel
from .sandbox import get_sandbox_runner


class AgentState(TypedDict):
    llm: HFChatModel
    config: "AgentConfig"
    prompt: str
    tests: str
    entry_point: str
    iters: int
    history: NotRequired[List[Dict[str, str]]]
    code: NotRequired[str]
    last_result: NotRequired[Dict[str, Any]]
    final_code: NotRequired[str]
    done: NotRequired[bool]
    metrics: NotRequired[Dict[str, Any]]


@dataclass
class AgentConfig:
    # Strict pass@1 by default: no additional reflection/execute loops
    max_iters: int = 0
    temperature: float = 0.0
    max_new_tokens: int = 512
    sandbox_mode: str = "process"
    sandbox_timeout: int = 30


def _system_prompt() -> str:
    return (
        "You are a Python bug-fixing assistant. Given a function specification/prompt, "
        "write the full function implementation. Respond ONLY with a single Python code block "
        "that defines the function(s) required, no prose, no explanations. Make sure your code "
        "is self-contained and uses the exact required function name (entry point)."
    )


def propose(state: AgentState) -> Dict[str, Any]:
    llm: HFChatModel = state["llm"]
    question: str = state["prompt"]
    history: List[Dict[str, str]] = state.get("history", [])
    metrics: Dict[str, Any] = dict(state.get("metrics", {}))

    messages = [{"role": "system", "content": _system_prompt()}]
    messages += history
    messages.append({"role": "user", "content": question})

    # Let the model return a full fenced code block; don't stop at ``` which would truncate the code
    t0 = time.time()
    completion = llm.invoke(messages)
    dt = time.time() - t0
    metrics["t_propose_sec"] = metrics.get("t_propose_sec", 0.0) + dt
    # Token usage metrics
    usage = getattr(llm, "last_usage", {})
    if usage:
        metrics["propose_prompt_tokens"] = metrics.get(
            "propose_prompt_tokens", 0
        ) + int(usage.get("prompt_tokens", 0))
        metrics["propose_completion_tokens"] = metrics.get(
            "propose_completion_tokens", 0
        ) + int(usage.get("completion_tokens", 0))

    # Extract python code block if model included fences; fallback to raw text
    code = completion
    if "```" in completion:
        parts = completion.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.strip().startswith("python"):
                code = inner.split("\n", 1)[1]
            else:
                code = inner
    return {"code": code, "metrics": metrics}


def execute(state: AgentState) -> Dict[str, Any]:
    code: str = state.get("code", "")
    tests: str = state["tests"]
    entry_point: str = state["entry_point"]
    metrics: Dict[str, Any] = dict(state.get("metrics", {}))
    cfg: AgentConfig = state["config"]

    t0 = time.time()
    runner = get_sandbox_runner(cfg.sandbox_mode)
    result = runner(code, tests, entry_point, cfg.sandbox_timeout)
    dt = time.time() - t0
    metrics["t_execute_sec"] = metrics.get("t_execute_sec", 0.0) + dt
    logging.info(
        "Sandbox run: passed=%s exit=%s time=%.2fs",
        bool(result.get("passed")),
        result.get("exit_code"),
        dt,
    )
    return {"last_result": result, "metrics": metrics}


def reflect(state: AgentState) -> Dict[str, Any]:
    llm: HFChatModel = state["llm"]
    result: Dict[str, Any] = state.get("last_result", {})
    question: str = state["prompt"]
    prev_code: str = state.get("code", "")
    metrics: Dict[str, Any] = dict(state.get("metrics", {}))

    # Strict pass@1 mode: if no iterations allowed, end immediately after first execute
    if state["config"].max_iters <= 0:
        return {"final_code": prev_code, "done": True}

    if result.get("passed"):
        return {"final_code": prev_code, "done": True}

    err = (result.get("stderr") or "") + "\n" + (result.get("stdout") or "")
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": f"Here is my first attempt:\n```python\n{prev_code}\n```",
        },
        {
            "role": "user",
            "content": (
                "The tests failed with the following error log. Please provide a corrected implementation.\n"
                + "Focus on correctness. Respond only with the full corrected code in a Python code block.\n"
                + f"Error log:\n{err}"
            ),
        },
    ]
    # Do not truncate at fences; we want the full corrected code block
    t0 = time.time()
    completion = llm.invoke(messages)
    dt = time.time() - t0
    metrics["t_reflect_sec"] = metrics.get("t_reflect_sec", 0.0) + dt
    usage = getattr(llm, "last_usage", {})
    if usage:
        metrics["reflect_prompt_tokens"] = metrics.get(
            "reflect_prompt_tokens", 0
        ) + int(usage.get("prompt_tokens", 0))
        metrics["reflect_completion_tokens"] = metrics.get(
            "reflect_completion_tokens", 0
        ) + int(usage.get("completion_tokens", 0))
    code = completion
    if "```" in completion:
        parts = completion.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.strip().startswith("python"):
                code = inner.split("\n", 1)[1]
            else:
                code = inner
    # increment iteration counter
    return {"code": code, "iters": state.get("iters", 0) + 1, "metrics": metrics}


def should_continue(state: AgentState) -> str:
    if state.get("done"):
        return END
    iters = state.get("iters", 0)
    if iters >= state["config"].max_iters:
        return END
    # If last result passed, we also end
    last = state.get("last_result")
    if last and last.get("passed"):
        return END
    return "loop"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("propose", propose)
    graph.add_node("execute", execute)
    graph.add_node("reflect", reflect)

    graph.set_entry_point("propose")
    graph.add_edge("propose", "execute")
    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges(
        "reflect", should_continue, {"loop": "execute", END: END}
    )
    return graph

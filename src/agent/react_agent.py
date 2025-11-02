from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, NotRequired

from langgraph.graph import StateGraph, END

from .llm import HFChatModel
from .sandbox import run_python_with_tests


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


@dataclass
class AgentConfig:
    max_iters: int = 3
    temperature: float = 0.0
    max_new_tokens: int = 512


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

    messages = [{"role": "system", "content": _system_prompt()}]
    messages += history
    messages.append({"role": "user", "content": question})

    # Let the model return a full fenced code block; don't stop at ``` which would truncate the code
    completion = llm.invoke(messages)

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
    return {"code": code}


def execute(state: AgentState) -> Dict[str, Any]:
    code: str = state.get("code", "")
    tests: str = state["tests"]
    entry_point: str = state["entry_point"]

    result = run_python_with_tests(code, tests, entry_point, timeout_s=10)
    return {"last_result": result}


def reflect(state: AgentState) -> Dict[str, Any]:
    llm: HFChatModel = state["llm"]
    result: Dict[str, Any] = state.get("last_result", {})
    question: str = state["prompt"]
    prev_code: str = state.get("code", "")

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
    completion = llm.invoke(messages)
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
    return {"code": code, "iters": state.get("iters", 0) + 1}


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

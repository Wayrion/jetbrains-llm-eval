"""Optional LangGraph adapter (illustrative).

This module shows how you might adapt the included ReActAgent to a LangGraph tool/agent.
It does not assume any particular LangGraph API; instead it provides a helper that
returns a callable you can wire into your LangGraph graph. If `langgraph` is present
in your environment, create a node that calls `make_langgraph_tool()` and invoke it.

Example usage (pseudo):

    from langgraph import Graph, Tool
    from langgraph_adapter import make_langgraph_tool

    tool = make_langgraph_tool(model='qwen3-0.6b')
    # Now register `tool` within your LangGraph graph as a tool that takes a prompt and returns code

"""

import os
from agent.agent import LLMClient, ReActAgent


def make_langgraph_tool(model: str = "qwen3-0.6b"):
    """Return a simple callable (prompt -> code_str) that you can register as a tool.

    This intentionally does not import or depend on LangGraph APIs so it's portable.
    """
    hf_token = os.environ.get("HF_API_TOKEN")
    llm = LLMClient(model=model, hf_token=hf_token)
    agent = ReActAgent(llm)

    def tool(prompt: str) -> str:
        return agent.generate_fix(prompt)

    return tool

from typing import Literal

from assistant.graph.state import AgentState, pending_tool_calls
from assistant.mcp_server.auth import WRITE_TOOL_NAMES


def route_after_planner(state: AgentState) -> Literal["retriever", "executor"]:
    plan = state["plan"]
    if plan is not None and "search_documents" in plan.needs_tools:
        return "retriever"
    return "executor"


def route_after_executor(
    state: AgentState,
) -> Literal["request_approval", "run_tools", "reviewer"]:
    calls = pending_tool_calls(state)
    if not calls:
        return "reviewer"
    if any(call["name"] in WRITE_TOOL_NAMES for call in calls):
        return "request_approval"
    return "run_tools"


def route_after_review(state: AgentState) -> Literal["executor", "remember"]:
    return "executor" if state["verdict"] == "revise" else "remember"

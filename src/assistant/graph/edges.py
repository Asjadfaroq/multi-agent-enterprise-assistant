from typing import Literal

from langgraph.graph import END

from assistant.graph.state import AgentState


def route_after_planner(state: AgentState) -> Literal["retriever", "executor"]:
    plan = state["plan"]
    if plan is not None and "search_documents" in plan.needs_tools:
        return "retriever"
    return "executor"


def route_after_review(state: AgentState) -> Literal["executor", "__end__"]:
    return "executor" if state["verdict"] == "revise" else END

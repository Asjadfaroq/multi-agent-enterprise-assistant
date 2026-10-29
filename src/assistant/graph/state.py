from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from assistant.guardrails.schema import Draft, Plan, Review
from assistant.mcp_server.tools_kb import SearchHit

ReviewVerdict = Literal["accepted", "revise", "give_up"]

# Reviewer feedback is stored as a human-role message so the executor sees it in order; this name
# keeps it from being mistaken for the user's own question.
REVIEWER_FEEDBACK_NAME = "reviewer"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    plan: Plan | None
    retrieved: list[SearchHit]
    draft: Draft | None
    review: Review | None
    verdict: ReviewVerdict | None
    revision_count: int


@dataclass(frozen=True)
class AssistantContext:
    """Per-request dependencies: the MCP tools the requesting user's server session exposes."""

    tools_by_name: dict[str, BaseTool]


StateUpdate = dict[str, Any]
Node = Callable[[AgentState], Awaitable[StateUpdate]]
ContextNode = Callable[[AgentState, Runtime[AssistantContext]], Awaitable[StateUpdate]]


def new_turn_state(user_id: str, question: str) -> AgentState:
    # Every per-turn field is reset so a checkpointed thread never reuses the previous turn's
    # chunks or review.
    return AgentState(
        messages=[HumanMessage(question)],
        user_id=user_id,
        plan=None,
        retrieved=[],
        draft=None,
        review=None,
        verdict=None,
        revision_count=0,
    )


def is_user_question(message: AnyMessage) -> bool:
    return isinstance(message, HumanMessage) and message.name != REVIEWER_FEEDBACK_NAME


def latest_question_index(state: AgentState) -> int:
    messages = state["messages"]
    for index in range(len(messages) - 1, -1, -1):
        if is_user_question(messages[index]):
            return index
    raise ValueError("State has no user message to answer")


def latest_question(state: AgentState) -> str:
    return state["messages"][latest_question_index(state)].text


def messages_since_latest_question(state: AgentState) -> list[AnyMessage]:
    return state["messages"][latest_question_index(state) + 1 :]


def pending_tool_calls(state: AgentState) -> list[ToolCall]:
    last_message = state["messages"][-1]
    return list(last_message.tool_calls) if isinstance(last_message, AIMessage) else []


def format_retrieved_context(hits: list[SearchHit]) -> str:
    if not hits:
        return "(no document excerpts were found)"
    return "\n\n".join(f"[{hit.doc_id}] {hit.title}\n{hit.content}" for hit in hits)


def format_tool_results(state: AgentState) -> str:
    results = [
        message
        for message in messages_since_latest_question(state)
        if isinstance(message, ToolMessage)
    ]
    if not results:
        return "(no tools were called)"
    return "\n\n".join(f"{result.name} ({result.status}): {result.text}" for result in results)

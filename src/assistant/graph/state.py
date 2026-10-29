from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages

from assistant.guardrails.schema import Draft, Plan, Review
from assistant.mcp_server.tools_kb import SearchHit

ReviewVerdict = Literal["accepted", "revise", "give_up"]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    plan: Plan | None
    retrieved: list[SearchHit]
    draft: Draft | None
    review: Review | None
    verdict: ReviewVerdict | None
    revision_count: int


StateUpdate = dict[str, Any]
Node = Callable[[AgentState], Awaitable[StateUpdate]]


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


def latest_question(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.text
    raise ValueError("State has no user message to answer")


def format_retrieved_context(hits: list[SearchHit]) -> str:
    if not hits:
        return "(no document excerpts were found)"
    return "\n\n".join(f"[{hit.doc_id}] {hit.title}\n{hit.content}" for hit in hits)

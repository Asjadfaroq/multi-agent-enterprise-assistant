from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages

from assistant.guardrails.pii import mask_pii
from assistant.guardrails.schema import Answer, Plan, ProfileFact, Review
from assistant.mcp_server.auth import WRITE_TOOL_NAMES
from assistant.mcp_server.tools_kb import SearchHit

ReviewVerdict = Literal["accepted", "revise", "give_up"]

# Reviewer feedback is stored as a human-role message so the executor sees it in order; this name
# keeps it from being mistaken for the user's own question.
REVIEWER_FEEDBACK_NAME = "reviewer"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    user_profile: list[ProfileFact]
    input_blocked: bool
    plan: Plan | None
    retrieved: list[SearchHit]
    answer: Answer | None
    review: Review | None
    verdict: ReviewVerdict | None
    revision_count: int


class ProfileStore(Protocol):
    async def load_facts(self, user_id: str) -> list[ProfileFact]: ...

    async def save_facts(self, user_id: str, facts: list[ProfileFact]) -> None: ...


@dataclass(frozen=True)
class AssistantContext:
    """Per-request dependencies: the requesting user's MCP tools and long-term profile."""

    tools_by_name: dict[str, BaseTool]
    profile_store: ProfileStore


StateUpdate = dict[str, Any]
# Nodes take the state plus whichever of `runtime` and `config` they need; LangGraph injects both.
Node = Callable[..., Awaitable[StateUpdate]]


def new_turn_state(user_id: str, question: str) -> AgentState:
    # Every per-turn field is reset so a checkpointed thread never reuses the previous turn's
    # chunks or review. PII is masked here, before the model, the checkpoint or a trace sees it.
    return AgentState(
        messages=[HumanMessage(mask_pii(question))],
        user_id=user_id,
        user_profile=[],
        input_blocked=False,
        plan=None,
        retrieved=[],
        answer=None,
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


def completed_write_actions(state: AgentState) -> list[str]:
    return [
        message.name
        for message in messages_since_latest_question(state)
        if isinstance(message, ToolMessage)
        and message.name in WRITE_TOOL_NAMES
        and message.status == "success"
    ]


def pending_tool_calls(state: AgentState) -> list[ToolCall]:
    last_message = state["messages"][-1]
    return list(last_message.tool_calls) if isinstance(last_message, AIMessage) else []


def format_user_profile(facts: list[ProfileFact]) -> str:
    if not facts:
        return "(nothing known yet)"
    return "\n".join(f"- {fact.key}: {fact.value}" for fact in facts)


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

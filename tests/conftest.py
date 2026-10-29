import json
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import psycopg
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from assistant.config import get_settings
from assistant.graph.build import build_graph
from assistant.graph.state import AssistantContext
from assistant.guardrails.injection import INPUT_SCREEN_INSTRUCTIONS
from assistant.guardrails.schema import ProfileFact
from assistant.mcp_server.auth import create_users_table
from assistant.mcp_server.tools_kb import SearchHit

DB_CONNECT_TIMEOUT_SECONDS = 2
NO_PROFILE_FACTS = json.dumps({"facts": []})
SAFE_INPUT_VERDICT = json.dumps({"is_injection": False, "reason": "Ordinary request."})
NO_POISONED_CHUNKS = json.dumps({"flagged": []})
LEAVE_POLICY_HIT = SearchHit(
    chunk_id=1,
    doc_id="parental-leave-policy",
    acl="hr",
    title="Parental Leave Policy",
    content="Birthing parents receive 14 weeks of fully paid maternity leave.",
    score=0.9,
)


class ScriptedChatModel(GenericFakeChatModel):
    """Replays fixed replies in order and records every prompt; bound tools are ignored.

    The input guard runs at the same time as the planner, so when `input_verdict` is set the
    guard's prompt is answered with it instead of taking the next reply from the queue.
    """

    received_prompts: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[list[str]] = Field(default_factory=list)
    input_verdict: str | None = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received_prompts.append(list(messages))
        if self.input_verdict is not None and messages[0].text.startswith(
            INPUT_SCREEN_INSTRUCTIONS
        ):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(self.input_verdict))])
        return super()._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable:
        self.bound_tool_names.append([tool.name for tool in tools])
        return self


class InMemoryProfileStore:
    def __init__(self) -> None:
        self.facts_by_user: dict[str, dict[str, str]] = {}

    async def load_facts(self, user_id: str) -> list[ProfileFact]:
        facts = self.facts_by_user.get(user_id, {})
        return [ProfileFact(key=key, value=value) for key, value in sorted(facts.items())]

    async def save_facts(self, user_id: str, facts: list[ProfileFact]) -> None:
        self.facts_by_user.setdefault(user_id, {}).update({fact.key: fact.value for fact in facts})


class FakeToolbox:
    """In-memory stand-ins for the MCP tools and profile store, recording every call."""

    def __init__(self) -> None:
        self.search_queries: list[str] = []
        self.search_hits = [LEAVE_POLICY_HIT]
        self.created_tickets: list[dict[str, Any]] = []
        self.profile_store = InMemoryProfileStore()

    async def search_documents(self, query: str, k: int) -> tuple[str, dict[str, Any]]:
        self.search_queries.append(query)
        hits = [hit.model_dump() for hit in self.search_hits[:k]]
        return str(hits), {"structured_content": {"result": hits}}

    async def list_tickets(self) -> str:
        return str(self.created_tickets)

    async def create_ticket(self, title: str, category: str, priority: str) -> str:
        ticket_id = len(self.created_tickets) + 1
        self.created_tickets.append({"id": ticket_id, "title": title, "priority": priority})
        return f"Created {category} ticket {ticket_id}"

    def context(self, can_write_tickets: bool) -> AssistantContext:
        tools: list[BaseTool] = [
            StructuredTool.from_function(
                coroutine=self.search_documents,
                name="search_documents",
                description="Search documents.",
                response_format="content_and_artifact",
            ),
            StructuredTool.from_function(
                coroutine=self.list_tickets, name="list_tickets", description="List tickets."
            ),
        ]
        if can_write_tickets:
            tools.append(
                StructuredTool.from_function(
                    coroutine=self.create_ticket, name="create_ticket", description="Create."
                )
            )
        return AssistantContext(
            tools_by_name={tool.name: tool for tool in tools}, profile_store=self.profile_store
        )


GraphBuilder = Callable[[list[str | AIMessage], BaseCheckpointSaver | None], CompiledStateGraph]


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests never send traces, even when the shell has Langfuse keys exported."""
    for variable in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(variable, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def toolbox() -> FakeToolbox:
    return FakeToolbox()


def scripted_model_after_safe_input(replies: list[str | AIMessage]) -> ScriptedChatModel:
    """A scripted model whose input guard clears every message and that replays `replies`."""
    return ScriptedChatModel(messages=iter(replies), input_verdict=SAFE_INPUT_VERDICT)


@pytest.fixture
def build_scripted_graph() -> GraphBuilder:
    def build(
        replies: list[str | AIMessage], checkpointer: BaseCheckpointSaver | None = None
    ) -> CompiledStateGraph:
        scripted_model = scripted_model_after_safe_input(replies)
        return build_graph(
            chat_model_factory=lambda role: scripted_model, checkpointer=checkpointer
        )

    return build


def connect_or_skip(autocommit: bool) -> psycopg.Connection:
    database_url = get_settings().database_url
    try:
        return psycopg.connect(
            database_url, autocommit=autocommit, connect_timeout=DB_CONNECT_TIMEOUT_SECONDS
        )
    except psycopg.OperationalError as error:
        pytest.skip(f"Postgres is not reachable at {database_url}: {error}")


@pytest.fixture
def committed_users() -> Iterator[Callable[[str, str], None]]:
    """Adds users visible to other connections (the API's) and deletes them afterwards."""
    connection = connect_or_skip(autocommit=True)
    create_users_table(connection)
    added_user_ids: list[str] = []

    def add_user(user_id: str, role: str) -> None:
        connection.execute(
            "INSERT INTO users (id, role) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET role = EXCLUDED.role",
            (user_id, role),
        )
        added_user_ids.append(user_id)

    try:
        yield add_user
    finally:
        connection.execute("DELETE FROM users WHERE id = ANY(%s)", (added_user_ids,))
        connection.close()

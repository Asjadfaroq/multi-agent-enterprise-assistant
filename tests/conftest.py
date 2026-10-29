from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field

from assistant.graph.build import build_graph
from assistant.graph.state import AssistantContext
from assistant.mcp_server.tools_kb import SearchHit

LEAVE_POLICY_HIT = SearchHit(
    doc_id="parental-leave-policy",
    acl="hr",
    title="Parental Leave Policy",
    content="Birthing parents receive 14 weeks of fully paid maternity leave.",
    score=0.9,
)


class ScriptedChatModel(GenericFakeChatModel):
    """Replays fixed replies in order and records every prompt; bound tools are ignored."""

    received_prompts: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received_prompts.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable:
        return self


class FakeToolbox:
    """In-memory stand-ins for the MCP tools, recording every call."""

    def __init__(self) -> None:
        self.search_queries: list[str] = []
        self.created_tickets: list[dict[str, Any]] = []

    async def search_documents(self, query: str) -> tuple[str, dict[str, Any]]:
        self.search_queries.append(query)
        hits = [LEAVE_POLICY_HIT.model_dump()]
        return str(hits), {"structured_content": {"result": hits}}

    async def list_tickets(self) -> str:
        return str(self.created_tickets)

    async def create_ticket(self, title: str, category: str, priority: str) -> str:
        ticket = {"id": len(self.created_tickets) + 1, "title": title}
        self.created_tickets.append(ticket | {"category": category, "priority": priority})
        return f"Created ticket {ticket['id']}"

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
        return AssistantContext(tools_by_name={tool.name: tool for tool in tools})


GraphBuilder = Callable[[list[str | AIMessage], BaseCheckpointSaver | None], CompiledStateGraph]


@pytest.fixture
def toolbox() -> FakeToolbox:
    return FakeToolbox()


@pytest.fixture
def build_scripted_graph() -> GraphBuilder:
    def build(
        replies: list[str | AIMessage], checkpointer: BaseCheckpointSaver | None = None
    ) -> CompiledStateGraph:
        scripted_model = ScriptedChatModel(messages=iter(replies))
        return build_graph(
            chat_model_factory=lambda role: scripted_model, checkpointer=checkpointer
        )

    return build

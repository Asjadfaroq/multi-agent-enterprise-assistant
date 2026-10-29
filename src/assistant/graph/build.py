from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from assistant.config import get_settings
from assistant.graph.edges import (
    route_after_executor,
    route_after_input_guard,
    route_after_review,
)
from assistant.graph.nodes.approval import request_approval
from assistant.graph.nodes.executor import make_executor_node, run_tool_calls
from assistant.graph.nodes.planner import make_planner_node
from assistant.graph.nodes.retriever import make_retriever_node
from assistant.graph.nodes.reviewer import make_reviewer_node
from assistant.graph.state import AgentState, AssistantContext
from assistant.guardrails.injection import apply_input_guard, make_input_guard_node
from assistant.guardrails.schema import Answer, Plan, ProfileFact, Review
from assistant.llm import ModelRole, get_chat_model
from assistant.mcp_server.auth import (
    CALLER_HEADER,
    configured_service_token,
    service_authorization_header,
)
from assistant.mcp_server.tools_kb import SearchHit
from assistant.memory.profile import PostgresProfileStore, make_memory_node

ChatModelFactory = Callable[[ModelRole], BaseChatModel]

ASSISTANT_SERVER_NAME = "assistant"

# LangGraph only restores checkpointed classes it has been told to trust.
CHECKPOINTED_MODELS = (Plan, Answer, Review, SearchHit, ProfileFact)


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            (model.__module__, model.__name__) for model in CHECKPOINTED_MODELS
        ]
    )


def build_graph(
    chat_model_factory: ChatModelFactory = get_chat_model,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    settings = get_settings()
    agent_llm = chat_model_factory("agent")
    encoded_payload_min_length = settings.injection_encoded_payload_min_length
    graph = StateGraph(AgentState, context_schema=AssistantContext)
    graph.add_node("guard_input", make_input_guard_node(agent_llm, encoded_payload_min_length))
    graph.add_node("planner", make_planner_node(agent_llm))
    graph.add_node(
        "retriever",
        make_retriever_node(agent_llm, settings.search_result_count, encoded_payload_min_length),
    )
    graph.add_node("executor", make_executor_node(agent_llm))
    graph.add_node("request_approval", request_approval)
    graph.add_node("run_tools", run_tool_calls)
    graph.add_node("reviewer", make_reviewer_node(agent_llm, settings.max_review_revisions))
    graph.add_node("remember", make_memory_node(agent_llm, settings.max_profile_facts_per_turn))
    graph.add_node("apply_input_guard", apply_input_guard)
    # The input guard and the planner run in parallel; the join waits for both.
    graph.add_edge(START, "guard_input")
    graph.add_edge(START, "planner")
    graph.add_edge(["guard_input", "planner"], "apply_input_guard")
    graph.add_conditional_edges("apply_input_guard", route_after_input_guard)
    graph.add_edge("retriever", "executor")
    graph.add_conditional_edges("executor", route_after_executor)
    graph.add_edge("run_tools", "executor")
    graph.add_conditional_edges("reviewer", route_after_review)
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)


def assistant_server_connection(user_id: str) -> StreamableHttpConnection:
    return {
        "transport": "streamable_http",
        "url": get_settings().mcp_server_url,
        "headers": {
            CALLER_HEADER: user_id,
            "Authorization": service_authorization_header(configured_service_token()),
        },
    }


@asynccontextmanager
async def open_assistant_context(user_id: str) -> AsyncIterator[AssistantContext]:
    client = MultiServerMCPClient({ASSISTANT_SERVER_NAME: assistant_server_connection(user_id)})
    async with client.session(ASSISTANT_SERVER_NAME) as session:
        tools = await load_mcp_tools(session)
        yield AssistantContext(
            tools_by_name={tool.name: tool for tool in tools},
            profile_store=PostgresProfileStore(get_settings().database_url),
        )

import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from assistant.config import get_settings
from assistant.graph.edges import route_after_executor, route_after_planner, route_after_review
from assistant.graph.nodes.executor import make_executor_node, request_approval, run_tool_calls
from assistant.graph.nodes.planner import make_planner_node
from assistant.graph.nodes.retriever import retrieve_documents
from assistant.graph.nodes.reviewer import make_reviewer_node
from assistant.graph.state import AgentState, AssistantContext
from assistant.guardrails.schema import Draft, Plan, Review
from assistant.llm import ModelRole, get_chat_model, is_embedding_model_cached
from assistant.mcp_server.tools_kb import SearchHit

ChatModelFactory = Callable[[ModelRole], BaseChatModel]

ASSISTANT_SERVER_NAME = "assistant"

# LangGraph only restores checkpointed classes it has been told to trust.
CHECKPOINTED_MODELS = (Plan, Draft, Review, SearchHit)


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
    agent_llm = chat_model_factory("agent")
    graph = StateGraph(AgentState, context_schema=AssistantContext)
    graph.add_node("planner", make_planner_node(agent_llm))
    graph.add_node("retriever", retrieve_documents)
    graph.add_node("executor", make_executor_node(agent_llm))
    graph.add_node("request_approval", request_approval)
    graph.add_node("run_tools", run_tool_calls)
    graph.add_node("reviewer", make_reviewer_node(agent_llm, get_settings().max_review_revisions))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_planner)
    graph.add_edge("retriever", "executor")
    graph.add_conditional_edges("executor", route_after_executor)
    graph.add_edge("run_tools", "executor")
    graph.add_conditional_edges("reviewer", route_after_review)
    return graph.compile(checkpointer=checkpointer)


def assistant_server_connection(user_id: str) -> StdioConnection:
    # Identity travels in the server's environment, outside anything the model can write,
    # so a prompt cannot make the server act as someone else.
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "assistant.mcp_server.server"],
        "env": {
            "ASSISTANT_USER_ID": user_id,
            "DATABASE_URL": get_settings().database_url,
            # Hub lookups add seconds to every server start once the model is already local.
            "HF_HUB_OFFLINE": "1" if is_embedding_model_cached() else "0",
        },
    }


@asynccontextmanager
async def open_assistant_context(user_id: str) -> AsyncIterator[AssistantContext]:
    client = MultiServerMCPClient({ASSISTANT_SERVER_NAME: assistant_server_connection(user_id)})
    async with client.session(ASSISTANT_SERVER_NAME) as session:
        tools = await load_mcp_tools(session)
        yield AssistantContext(tools_by_name={tool.name: tool for tool in tools})

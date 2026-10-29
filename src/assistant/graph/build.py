from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from assistant.config import get_settings
from assistant.graph.edges import route_after_planner, route_after_review
from assistant.graph.nodes.executor import make_executor_node
from assistant.graph.nodes.planner import make_planner_node
from assistant.graph.nodes.retriever import (
    DocumentSearch,
    make_retriever_node,
    search_documents_via_mcp,
)
from assistant.graph.nodes.reviewer import make_reviewer_node
from assistant.graph.state import AgentState
from assistant.llm import ModelRole, get_chat_model

ChatModelFactory = Callable[[ModelRole], BaseChatModel]


def build_graph(
    chat_model_factory: ChatModelFactory = get_chat_model,
    search_documents: DocumentSearch = search_documents_via_mcp,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    agent_llm = chat_model_factory("agent")
    graph = StateGraph(AgentState)
    graph.add_node("planner", make_planner_node(agent_llm))
    graph.add_node("retriever", make_retriever_node(search_documents))
    graph.add_node("executor", make_executor_node(agent_llm))
    graph.add_node("reviewer", make_reviewer_node(agent_llm, get_settings().max_review_revisions))
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_after_planner)
    graph.add_edge("retriever", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_review)
    return graph.compile(checkpointer=checkpointer)

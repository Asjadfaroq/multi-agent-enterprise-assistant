import sys
from collections.abc import Awaitable, Callable

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
from pydantic import TypeAdapter

from assistant.config import get_settings
from assistant.graph.state import AgentState, Node, StateUpdate, latest_question
from assistant.mcp_server.tools_kb import SearchHit

DocumentSearch = Callable[[str, str], Awaitable[list[SearchHit]]]

KNOWLEDGE_SERVER_NAME = "knowledge"
SEARCH_HITS = TypeAdapter(list[SearchHit])


def knowledge_server_connection(user_id: str) -> StdioConnection:
    # Identity travels in the server's environment, outside anything the model can write,
    # so a prompt cannot make the server search as someone else.
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "assistant.mcp_server.server"],
        "env": {"ASSISTANT_USER_ID": user_id, "DATABASE_URL": get_settings().database_url},
    }


async def search_documents_via_mcp(user_id: str, query: str) -> list[SearchHit]:
    client = MultiServerMCPClient({KNOWLEDGE_SERVER_NAME: knowledge_server_connection(user_id)})
    async with client.session(KNOWLEDGE_SERVER_NAME) as session:
        result = await session.call_tool("search_documents", {"query": query})
    if result.isError or result.structuredContent is None:
        raise RuntimeError(f"search_documents failed for user {user_id!r}: {result.content}")
    return SEARCH_HITS.validate_python(result.structuredContent["result"])


def make_retriever_node(search_documents: DocumentSearch) -> Node:
    async def retrieve_documents(state: AgentState) -> StateUpdate:
        hits = await search_documents(state["user_id"], latest_question(state))
        return {"retrieved": hits}

    return retrieve_documents

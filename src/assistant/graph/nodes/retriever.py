from langchain_core.messages import ToolCall
from langgraph.runtime import Runtime
from pydantic import TypeAdapter

from assistant.graph.state import AgentState, AssistantContext, StateUpdate, latest_question
from assistant.mcp_server.tools_kb import SearchHit

SEARCH_TOOL_NAME = "search_documents"
SEARCH_HITS = TypeAdapter(list[SearchHit])


async def retrieve_documents(state: AgentState, runtime: Runtime[AssistantContext]) -> StateUpdate:
    search_tool = runtime.context.tools_by_name.get(SEARCH_TOOL_NAME)
    if search_tool is None:
        raise RuntimeError(f"The MCP session does not expose {SEARCH_TOOL_NAME}")
    search_call = ToolCall(
        name=SEARCH_TOOL_NAME,
        args={"query": latest_question(state)},
        id="retriever-search",
        type="tool_call",
    )
    tool_message = await search_tool.ainvoke(search_call)
    if tool_message.status == "error":
        raise RuntimeError(f"{SEARCH_TOOL_NAME} failed: {tool_message.text}")
    hits = SEARCH_HITS.validate_python(tool_message.artifact["structured_content"]["result"])
    return {"retrieved": hits}

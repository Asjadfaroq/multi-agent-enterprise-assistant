from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolCall
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import TypeAdapter

from assistant.graph.state import AgentState, AssistantContext, Node, StateUpdate, latest_question
from assistant.guardrails.injection import DROPPED_CHUNK_TAG, find_poisoned_chunks
from assistant.mcp_server.tools_kb import SearchHit
from assistant.observability.tracing import record_guardrail_event

SEARCH_TOOL_NAME = "search_documents"
SEARCH_HITS = TypeAdapter(list[SearchHit])


async def search_permitted_chunks(
    runtime: Runtime[AssistantContext], query: str
) -> list[SearchHit]:
    search_tool = runtime.context.tools_by_name.get(SEARCH_TOOL_NAME)
    if search_tool is None:
        raise RuntimeError(f"The MCP session does not expose {SEARCH_TOOL_NAME}")
    search_call = ToolCall(
        name=SEARCH_TOOL_NAME, args={"query": query}, id="retriever-search", type="tool_call"
    )
    tool_message = await search_tool.ainvoke(search_call)
    if tool_message.status == "error":
        raise RuntimeError(f"{SEARCH_TOOL_NAME} failed: {tool_message.text}")
    return SEARCH_HITS.validate_python(tool_message.artifact["structured_content"]["result"])


def make_retriever_node(llm: BaseChatModel, encoded_payload_min_length: int) -> Node:
    async def retrieve_documents(
        state: AgentState, runtime: Runtime[AssistantContext], config: RunnableConfig
    ) -> StateUpdate:
        hits = await search_permitted_chunks(runtime, latest_question(state))
        # Retrieved text is untrusted: a document can carry instructions aimed at the model.
        poisoned = await find_poisoned_chunks(llm, hits, encoded_payload_min_length)
        if poisoned:
            dropped = "; ".join(f"{hits[index].doc_id}: {poisoned[index]}" for index in poisoned)
            await record_guardrail_event(DROPPED_CHUNK_TAG, dropped, config)
        return {"retrieved": [hit for index, hit in enumerate(hits) if index not in poisoned]}

    return retrieve_documents

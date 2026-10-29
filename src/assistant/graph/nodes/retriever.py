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

DroppedChunk = tuple[SearchHit, str]


async def search_permitted_chunks(
    runtime: Runtime[AssistantContext], query: str, result_count: int
) -> list[SearchHit]:
    search_tool = runtime.context.tools_by_name.get(SEARCH_TOOL_NAME)
    if search_tool is None:
        raise RuntimeError(f"The MCP session does not expose {SEARCH_TOOL_NAME}")
    search_call = ToolCall(
        name=SEARCH_TOOL_NAME,
        args={"query": query, "k": result_count},
        id="retriever-search",
        type="tool_call",
    )
    tool_message = await search_tool.ainvoke(search_call)
    if tool_message.status == "error":
        raise RuntimeError(f"{SEARCH_TOOL_NAME} failed: {tool_message.text}")
    return SEARCH_HITS.validate_python(tool_message.artifact["structured_content"]["result"])


async def separate_poisoned_chunks(
    llm: BaseChatModel, hits: list[SearchHit], encoded_payload_min_length: int
) -> tuple[list[SearchHit], list[DroppedChunk]]:
    poisoned = await find_poisoned_chunks(llm, hits, encoded_payload_min_length)
    clean = [hit for index, hit in enumerate(hits) if index not in poisoned]
    dropped = [(hits[index], reason) for index, reason in poisoned.items()]
    return clean, dropped


def make_retriever_node(
    llm: BaseChatModel, result_count: int, encoded_payload_min_length: int
) -> Node:
    async def retrieve_documents(
        state: AgentState, runtime: Runtime[AssistantContext], config: RunnableConfig
    ) -> StateUpdate:
        question = latest_question(state)
        hits = await search_permitted_chunks(runtime, question, result_count)
        # Retrieved text is untrusted: a document can carry instructions aimed at the model.
        clean, dropped = await separate_poisoned_chunks(llm, hits, encoded_payload_min_length)
        if not dropped:
            return {"retrieved": clean}
        # One wider search refills the dropped slots with the next-ranked chunks. Drops among
        # those are not refilled again, so a heavily poisoned corpus cannot trigger a search loop.
        wider_hits = await search_permitted_chunks(runtime, question, result_count + len(dropped))
        already_seen = {hit.chunk_id for hit in hits}
        next_ranked = [hit for hit in wider_hits if hit.chunk_id not in already_seen]
        replacements, dropped_replacements = await separate_poisoned_chunks(
            llm, next_ranked, encoded_payload_min_length
        )
        all_dropped = dropped + dropped_replacements
        await record_guardrail_event(
            DROPPED_CHUNK_TAG,
            "; ".join(f"{hit.doc_id}: {reason}" for hit, reason in all_dropped),
            config,
        )
        return {"retrieved": clean + replacements}

    return retrieve_documents

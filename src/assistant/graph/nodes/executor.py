from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from assistant.graph.nodes.retriever import SEARCH_TOOL_NAME
from assistant.graph.state import (
    AgentState,
    AssistantContext,
    Node,
    StateUpdate,
    format_retrieved_context,
    format_user_profile,
    latest_question,
    messages_since_latest_question,
    pending_tool_calls,
    planned_write_tools,
)
from assistant.guardrails.schema import (
    Answer,
    InvalidModelOutputError,
    json_reply_instructions,
    parse_with_one_retry,
)
from assistant.mcp_server.auth import WRITE_TOOL_NAMES

EXECUTOR_INSTRUCTIONS = """\
You answer an employee's question and can act on the internal tickets system with the tools \
provided. Use only the document excerpts for facts about the company and the ticket tools for \
anything about tickets. Cite document facts as [doc_id] and ticket facts as [ticket-<id>], and \
list every cited id in "citations". Set "confidence" between 0 and 1 to how completely the \
excerpts and tool results support the answer.
If the excerpts and tools do not contain the answer, say you could not find it in what the user \
can access. Never use outside knowledge about the company. Write the answer in the user's \
preferred language when it is known.
Call tools when you need them. Once a change to a ticket has run, report its result instead of \
proposing it again. When you are done, reply with the final answer as JSON."""

INVALID_ANSWER_FALLBACK = (
    "Sorry, I could not produce a reliable answer this time. Please try asking again."
)


def build_executor_prompt(state: AgentState) -> str:
    return (
        f"What we know about the user:\n{format_user_profile(state['user_profile'])}\n\n"
        f"Question: {latest_question(state)}\n\n"
        f"Document excerpts:\n{format_retrieved_context(state['retrieved'])}"
    )


def make_executor_node(llm: BaseChatModel) -> Node:
    async def answer_or_call_tools(
        state: AgentState, runtime: Runtime[AssistantContext]
    ) -> StateUpdate:
        # Search is the retriever's job, and write tools are offered only when the plan says the
        # user asked for a change.
        allowed_writes = planned_write_tools(state)
        ticket_tools = [
            tool
            for name, tool in runtime.context.tools_by_name.items()
            if name != SEARCH_TOOL_NAME and (name not in WRITE_TOOL_NAMES or name in allowed_writes)
        ]
        tool_calling_llm = llm.bind_tools(ticket_tools)
        # The turn's history is only ever appended to: Claude signs its thinking blocks against
        # everything before them, and the API rejects a history whose earlier messages changed.
        messages = [
            SystemMessage(f"{EXECUTOR_INSTRUCTIONS}\n\n{json_reply_instructions(Answer)}"),
            HumanMessage(build_executor_prompt(state)),
            *messages_since_latest_question(state),
        ]
        reply = await tool_calling_llm.ainvoke(messages)
        if isinstance(reply, AIMessage) and reply.tool_calls:
            return {"messages": [reply]}
        try:
            answer = await parse_with_one_retry(tool_calling_llm, messages, reply, Answer)
        except InvalidModelOutputError:
            # Fail closed: an answer that is still malformed after one retry is never shown.
            return {"answer": None, "messages": [AIMessage(INVALID_ANSWER_FALLBACK)]}
        return {"answer": answer}

    return answer_or_call_tools


async def run_tool_calls(state: AgentState, runtime: Runtime[AssistantContext]) -> StateUpdate:
    results: list[ToolMessage] = []
    for call in pending_tool_calls(state):
        tool = runtime.context.tools_by_name.get(call["name"])
        if tool is None:
            results.append(
                ToolMessage(
                    f"Tool {call['name']!r} is not available to this user.",
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
            )
        else:
            results.append(await tool.ainvoke(call))
    return {"messages": results}

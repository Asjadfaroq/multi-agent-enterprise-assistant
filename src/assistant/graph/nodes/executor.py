from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from assistant.graph.nodes.retriever import SEARCH_TOOL_NAME
from assistant.graph.state import (
    AgentState,
    AssistantContext,
    ContextNode,
    StateUpdate,
    completed_write_actions,
    format_retrieved_context,
    format_user_profile,
    latest_question,
    messages_since_latest_question,
    pending_tool_calls,
)
from assistant.guardrails.schema import Draft, json_reply_instructions, parse_model_output
from assistant.mcp_server.auth import WRITE_TOOL_NAMES

EXECUTOR_INSTRUCTIONS = """\
You answer an employee's question and can act on the internal tickets system with the tools \
provided. Use only the document excerpts for facts about the company and the ticket tools for \
anything about tickets. Cite document facts as [doc_id] and ticket facts as [ticket-<id>], and \
list every cited id in "citations".
If the excerpts and tools do not contain the answer, say you could not find it in what the user \
can access. Never use outside knowledge about the company. Write the answer in the user's \
preferred language when it is known.
Call tools when you need them. Once a change to a ticket has run, report its result instead of \
proposing it again. When you are done, reply with the final answer as JSON."""

REJECTED_ACTION_MESSAGE = "The approver rejected this action, so nothing was changed."
WRITE_ALREADY_DONE_MESSAGE = (
    "Not run: an approved change already ran in this turn. Report its result to the user."
)
CANCELLED_ANSWER = "The approver rejected {tools}, so nothing was changed."


class ProposedToolCall(BaseModel):
    tool: str
    args: dict[str, Any]


def build_executor_prompt(state: AgentState) -> str:
    return (
        f"What we know about the user:\n{format_user_profile(state['user_profile'])}\n\n"
        f"Question: {latest_question(state)}\n\n"
        f"Document excerpts:\n{format_retrieved_context(state['retrieved'])}"
    )


def make_executor_node(llm: BaseChatModel) -> ContextNode:
    async def draft_or_call_tools(
        state: AgentState, runtime: Runtime[AssistantContext]
    ) -> StateUpdate:
        # Search is the retriever's job; the executor only gets the ticket tools.
        ticket_tools = [
            tool for name, tool in runtime.context.tools_by_name.items() if name != SEARCH_TOOL_NAME
        ]
        system_prompt = f"{EXECUTOR_INSTRUCTIONS}\n\n{json_reply_instructions(Draft)}"
        # The turn's history is only ever appended to: Claude signs its thinking blocks against
        # everything before them, and the API rejects a history whose earlier messages changed.
        reply = await llm.bind_tools(ticket_tools).ainvoke(
            [
                SystemMessage(system_prompt),
                HumanMessage(build_executor_prompt(state)),
                *messages_since_latest_question(state),
            ]
        )
        if isinstance(reply, AIMessage) and reply.tool_calls:
            return {"messages": [reply]}
        return {"draft": parse_model_output(reply.text, Draft)}

    return draft_or_call_tools


def refuse_tool_calls(calls: list[ToolCall], reason: str) -> list[ToolMessage]:
    return [
        ToolMessage(reason, tool_call_id=call["id"], name=call["name"], status="error")
        for call in calls
    ]


async def request_approval(
    state: AgentState, runtime: Runtime[AssistantContext]
) -> Command[Literal["run_tools", "executor", "__end__"]]:
    calls = pending_tool_calls(state)
    # One approved write per turn: in a live run the model re-proposed the write it had just made,
    # which would have created a duplicate ticket.
    if completed_write_actions(state):
        return Command(
            goto="executor",
            update={"messages": refuse_tool_calls(calls, WRITE_ALREADY_DONE_MESSAGE)},
        )
    # A write the user's session does not expose fails in run_tools without bothering an
    # approver, so roles without write access never reach a pause or a write tool.
    proposed_action = [
        ProposedToolCall(tool=call["name"], args=call["args"])
        for call in calls
        if call["name"] in WRITE_TOOL_NAMES and call["name"] in runtime.context.tools_by_name
    ]
    if not proposed_action:
        return Command(goto="run_tools")
    approved = interrupt({"proposed_action": [call.model_dump() for call in proposed_action]})
    if approved is True:
        return Command(goto="run_tools")
    # A rejection ends the turn without asking the model again, so it cannot re-propose the write.
    rejections = refuse_tool_calls(calls, REJECTED_ACTION_MESSAGE)
    rejected_tools = ", ".join(call.tool for call in proposed_action)
    cancellation = AIMessage(CANCELLED_ANSWER.format(tools=rejected_tools))
    return Command(goto=END, update={"messages": [*rejections, cancellation]})


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

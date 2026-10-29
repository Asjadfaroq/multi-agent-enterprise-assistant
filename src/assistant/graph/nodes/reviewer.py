from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from assistant.graph.state import (
    REVIEWER_FEEDBACK_NAME,
    AgentState,
    AssistantContext,
    ContextNode,
    StateUpdate,
    completed_write_actions,
    format_retrieved_context,
    format_tool_results,
    latest_question,
)
from assistant.guardrails.schema import Draft, Review, ask_for_json

REVIEWER_INSTRUCTIONS = """\
You check whether a draft answer is grounded in the document excerpts and tool results it was \
written from. "supported" is true only if every factual claim in the draft is stated in the \
excerpts or tool results and every citation points to something that backs the claim. A draft \
that says the answer was not found, or that makes no factual claims, is supported. Statements \
about what the assistant did or cannot do are supported when they agree with the tool results \
and the list of available tools, and pointing the user to a team or channel is not a factual \
claim. Explain your decision in "reason"."""

UNSUPPORTED_ANSWER_FALLBACK = (
    "I could not write an answer that is fully supported by the documents you have access to. "
    "Please rephrase the question or contact the owner of the relevant policy."
)


def build_reviewer_prompt(state: AgentState, draft: Draft, available_tools: list[str]) -> str:
    return (
        f"Question: {latest_question(state)}\n\n"
        f"Tools available to this user: {', '.join(available_tools)}\n\n"
        f"Document excerpts:\n{format_retrieved_context(state['retrieved'])}\n\n"
        f"Tool results:\n{format_tool_results(state)}\n\n"
        f"Draft answer:\n{draft.text}"
    )


def fallback_answer(state: AgentState) -> str:
    # An approved write has already changed data; the user must hear that even when the answer
    # about it could not be verified.
    completed_actions = completed_write_actions(state)
    if not completed_actions:
        return UNSUPPORTED_ANSWER_FALLBACK
    actions = ", ".join(completed_actions)
    return f"{UNSUPPORTED_ANSWER_FALLBACK} These approved actions did run: {actions}."


def make_reviewer_node(llm: BaseChatModel, max_revisions: int) -> ContextNode:
    async def review_draft(state: AgentState, runtime: Runtime[AssistantContext]) -> StateUpdate:
        draft = state["draft"]
        if draft is None:
            raise ValueError("Reviewer ran before the executor produced a draft")
        available_tools = sorted(runtime.context.tools_by_name)
        review = await ask_for_json(
            llm, REVIEWER_INSTRUCTIONS, build_reviewer_prompt(state, draft, available_tools), Review
        )
        if review.supported:
            return {"review": review, "verdict": "accepted", "messages": [AIMessage(draft.text)]}
        if state["revision_count"] < max_revisions:
            feedback = HumanMessage(
                f"A reviewer rejected your draft.\nDraft: {draft.text}\n"
                f"Reason: {review.reason}\nWrite a corrected answer.",
                name=REVIEWER_FEEDBACK_NAME,
            )
            return {
                "review": review,
                "verdict": "revise",
                "revision_count": state["revision_count"] + 1,
                "messages": [feedback],
            }
        return {
            "review": review,
            "verdict": "give_up",
            "messages": [AIMessage(fallback_answer(state))],
        }

    return review_draft

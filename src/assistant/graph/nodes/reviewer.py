from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from assistant.graph.state import (
    AgentState,
    Node,
    StateUpdate,
    format_retrieved_context,
    latest_question,
)
from assistant.guardrails.schema import Draft, Review, ask_for_json

REVIEWER_INSTRUCTIONS = """\
You check whether a draft answer is grounded in the document excerpts it was written from.
"supported" is true only if every factual claim in the draft is stated in the excerpts and every \
citation points to an excerpt that backs the claim. A draft that says the answer was not found, \
or that makes no factual claims, is supported. Explain your decision in "reason"."""

UNSUPPORTED_ANSWER_FALLBACK = (
    "I could not write an answer that is fully supported by the documents you have access to. "
    "Please rephrase the question or contact the owner of the relevant policy."
)


def build_reviewer_prompt(state: AgentState, draft: Draft) -> str:
    return (
        f"Question: {latest_question(state)}\n\n"
        f"Document excerpts:\n{format_retrieved_context(state['retrieved'])}\n\n"
        f"Draft answer:\n{draft.text}"
    )


def make_reviewer_node(llm: BaseChatModel, max_revisions: int) -> Node:
    async def review_draft(state: AgentState) -> StateUpdate:
        draft = state["draft"]
        if draft is None:
            raise ValueError("Reviewer ran before the executor produced a draft")
        review = await ask_for_json(
            llm, REVIEWER_INSTRUCTIONS, build_reviewer_prompt(state, draft), Review
        )
        if review.supported:
            return {"review": review, "verdict": "accepted", "messages": [AIMessage(draft.text)]}
        if state["revision_count"] < max_revisions:
            return {
                "review": review,
                "verdict": "revise",
                "revision_count": state["revision_count"] + 1,
            }
        return {
            "review": review,
            "verdict": "give_up",
            "messages": [AIMessage(UNSUPPORTED_ANSWER_FALLBACK)],
        }

    return review_draft

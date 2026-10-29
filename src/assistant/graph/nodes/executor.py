from langchain_core.language_models import BaseChatModel

from assistant.graph.state import (
    AgentState,
    Node,
    StateUpdate,
    format_retrieved_context,
    latest_question,
)
from assistant.guardrails.schema import Draft, ask_for_json

EXECUTOR_INSTRUCTIONS = """\
You answer an employee's question using only the document excerpts provided.
Cite every claim with its document id in square brackets, for example [expense-policy], and \
list the cited ids in "citations".
If the excerpts do not contain the answer, say you could not find it in the documents available \
to the user. Never use outside knowledge about the company."""


def build_executor_prompt(state: AgentState) -> str:
    sections = [
        f"Question: {latest_question(state)}",
        f"Document excerpts:\n{format_retrieved_context(state['retrieved'])}",
    ]
    review, draft = state["review"], state["draft"]
    if state["verdict"] == "revise" and review is not None and draft is not None:
        sections.append(
            f"A reviewer rejected your previous draft.\nPrevious draft: {draft.text}\n"
            f"Reviewer's reason: {review.reason}\nWrite a corrected answer."
        )
    return "\n\n".join(sections)


def make_executor_node(llm: BaseChatModel) -> Node:
    async def draft_answer(state: AgentState) -> StateUpdate:
        draft = await ask_for_json(llm, EXECUTOR_INSTRUCTIONS, build_executor_prompt(state), Draft)
        return {"draft": draft}

    return draft_answer

from langchain_core.language_models import BaseChatModel
from langgraph.runtime import Runtime

from assistant.graph.state import (
    AgentState,
    AssistantContext,
    Node,
    StateUpdate,
    format_user_profile,
    latest_question,
)
from assistant.guardrails.schema import Plan, ask_for_json

PLANNER_INSTRUCTIONS = """\
You plan how an internal company assistant answers an employee's question.

Available tools:
- search_documents: searches company documents (policies, HR, product docs, engineering runbooks).
- list_tickets, get_ticket: read tickets in the internal tickets system.
- create_ticket, update_ticket: create a ticket or change a ticket's status or priority.

List the steps needed to answer and the tools those steps need. Use search_documents for any \
question about the company, its policies, products or systems, and the ticket tools for \
anything about tickets. Plan create_ticket or update_ticket only when the user explicitly asks \
to create or change a ticket; a question about tickets, customers or vendors, or about what to \
tell someone, asks for information and needs read tools or search only. Use no tools for \
greetings or small talk. Take what is known about the user into account, for example by planning \
to answer in their preferred language."""


def make_planner_node(llm: BaseChatModel) -> Node:
    async def plan_answer(state: AgentState, runtime: Runtime[AssistantContext]) -> StateUpdate:
        user_profile = await runtime.context.profile_store.load_facts(state["user_id"])
        prompt = (
            f"What we know about the user:\n{format_user_profile(user_profile)}\n\n"
            f"Question: {latest_question(state)}"
        )
        plan = await ask_for_json(llm, PLANNER_INSTRUCTIONS, prompt, Plan)
        return {"plan": plan, "user_profile": user_profile}

    return plan_answer

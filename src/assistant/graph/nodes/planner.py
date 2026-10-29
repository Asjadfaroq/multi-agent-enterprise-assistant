from langchain_core.language_models import BaseChatModel

from assistant.graph.state import AgentState, Node, StateUpdate, latest_question
from assistant.guardrails.schema import Plan, ask_for_json

PLANNER_INSTRUCTIONS = """\
You plan how an internal company assistant answers an employee's question.

Available tools:
- search_documents: searches company documents (policies, HR, product docs, engineering runbooks).
- list_tickets, get_ticket: read tickets in the internal tickets system.
- create_ticket, update_ticket: create a ticket or change a ticket's status or priority.

List the steps needed to answer and the tools those steps need. Use search_documents for any \
question about the company, its policies, products or systems, and the ticket tools for \
anything about tickets. Use no tools for greetings or small talk."""


def make_planner_node(llm: BaseChatModel) -> Node:
    async def plan_answer(state: AgentState) -> StateUpdate:
        plan = await ask_for_json(llm, PLANNER_INSTRUCTIONS, latest_question(state), Plan)
        return {"plan": plan}

    return plan_answer

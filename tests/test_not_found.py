import asyncio
import json

from conftest import (
    NO_POISONED_CHUNKS,
    NO_PROFILE_FACTS,
    FakeToolbox,
    ScriptedChatModel,
    scripted_model_after_safe_input,
)

from assistant.graph.build import build_graph
from assistant.graph.nodes.executor import NOT_FOUND_ANSWER
from assistant.graph.state import AgentState, new_turn_state

PLAN_WITH_SEARCH = json.dumps({"steps": ["search"], "needs_tools": ["search_documents"]})
PLAN_WITHOUT_TOOLS = json.dumps({"steps": ["answer"], "needs_tools": []})
PADDED_NOT_FOUND = json.dumps(
    {
        "text": "I could not find a pets policy. People Operations owns employee relations, "
        "so ask them [company-overview].",
        "citations": ["company-overview"],
        "confidence": 0.4,
        "answer_found": False,
    }
)
SUPPORTED_REVIEW = json.dumps({"supported": True, "reason": "Grounded."})


def run_question(
    toolbox: FakeToolbox, replies: list[str], question: str
) -> tuple[AgentState, ScriptedChatModel]:
    scripted_model = scripted_model_after_safe_input(replies)
    graph = build_graph(chat_model_factory=lambda role: scripted_model)
    final_state = asyncio.run(
        graph.ainvoke(new_turn_state("nora", question), context=toolbox.context(False))
    )
    return final_state, scripted_model


def test_answer_not_found_reaches_the_user_as_one_fixed_sentence_without_citations(
    toolbox: FakeToolbox,
) -> None:
    replies = [PLAN_WITH_SEARCH, NO_POISONED_CHUNKS, PADDED_NOT_FOUND, NO_PROFILE_FACTS]

    final_state, _ = run_question(toolbox, replies, "Can I bring my dog to the office?")

    assert final_state["messages"][-1].text == NOT_FOUND_ANSWER
    assert final_state["answer"].citations == []
    assert final_state["verdict"] == "accepted"


def test_question_whose_plan_needs_no_tickets_is_offered_no_ticket_tools(
    toolbox: FakeToolbox,
) -> None:
    replies = [PLAN_WITHOUT_TOOLS, PADDED_NOT_FOUND, NO_PROFILE_FACTS]

    _, scripted_model = run_question(toolbox, replies, "What will the weather be tomorrow?")

    assert scripted_model.bound_tool_names == [[]]

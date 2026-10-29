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
from assistant.graph.nodes.reviewer import REVIEWER_INSTRUCTIONS, UNSUPPORTED_ANSWER_FALLBACK
from assistant.graph.state import AgentState, new_turn_state
from evals.behaviour import cites_expected_source
from evals.golden_cases import load_golden_cases
from evals.run_cases import CaseRun

PLAN_WITH_SEARCH = json.dumps({"steps": ["search"], "needs_tools": ["search_documents"]})
UNCITED_ANSWER = json.dumps(
    {"text": "It is 14 weeks.", "citations": [], "confidence": 0.9, "answer_found": True}
)
CITED_ANSWER = json.dumps(
    {
        "text": "It is 14 weeks [parental-leave-policy].",
        "citations": ["parental-leave-policy"],
        "confidence": 0.9,
        "answer_found": True,
    }
)
SUPPORTED_REVIEW = json.dumps({"supported": True, "reason": "Grounded."})


def run_turn(toolbox: FakeToolbox, replies: list[str]) -> tuple[AgentState, ScriptedChatModel]:
    scripted_model = scripted_model_after_safe_input(
        [PLAN_WITH_SEARCH, NO_POISONED_CHUNKS, *replies]
    )
    graph = build_graph(chat_model_factory=lambda role: scripted_model)
    question = new_turn_state("huda", "How long is maternity leave?")
    final_state = asyncio.run(graph.ainvoke(question, context=toolbox.context(False)))
    return final_state, scripted_model


def reviewer_model_calls(model: ScriptedChatModel) -> int:
    return sum(
        prompt[0].text.startswith(REVIEWER_INSTRUCTIONS) for prompt in model.received_prompts
    )


def test_uncited_answer_is_sent_back_without_a_model_review_and_the_cited_retry_wins(
    toolbox: FakeToolbox,
) -> None:
    replies = [UNCITED_ANSWER, CITED_ANSWER, SUPPORTED_REVIEW, NO_PROFILE_FACTS]

    final_state, model = run_turn(toolbox, replies)

    assert final_state["messages"][-1].text == "It is 14 weeks [parental-leave-policy]."
    assert reviewer_model_calls(model) == 1
    feedback = next(m for m in final_state["messages"] if m.name == "reviewer")
    assert "parental-leave-policy" in feedback.text


def test_answer_still_uncited_after_the_retry_ends_in_the_fallback(toolbox: FakeToolbox) -> None:
    final_state, model = run_turn(toolbox, [UNCITED_ANSWER, UNCITED_ANSWER, NO_PROFILE_FACTS])

    assert final_state["messages"][-1].text == UNSUPPORTED_ANSWER_FALLBACK
    assert reviewer_model_calls(model) == 0


def test_eval_check_requires_an_expected_source_among_the_citations() -> None:
    case = next(case for case in load_golden_cases() if case.id == "factual-11")

    assert not cites_expected_source(CaseRun(case=case, citations=[])).passed
    assert cites_expected_source(CaseRun(case=case, citations=["system-architecture"])).passed

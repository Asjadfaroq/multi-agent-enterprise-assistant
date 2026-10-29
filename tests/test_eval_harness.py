from evals.golden_cases import load_golden_cases
from evals.judge import build_judge_prompt
from evals.run_cases import CaseRun


def test_judge_sees_the_tool_results_the_answer_was_built_from() -> None:
    case = next(case for case in load_golden_cases() if case.id == "no-write-03")
    run = CaseRun(
        case=case,
        answer_text="Ticket 2 is open [ticket-2].",
        tool_results=['list_tickets (success): {"id": 2, "title": "Refund for double charge"}'],
    )

    prompt = build_judge_prompt(run)

    assert 'Tool results:\nlist_tickets (success): {"id": 2' in prompt

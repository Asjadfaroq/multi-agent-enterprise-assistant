import pytest

from assistant.config import get_settings
from evals.golden_cases import load_golden_cases, select_cases
from evals.results import RESULTS_DIR, EvalResults


@pytest.fixture(scope="module")
def results() -> EvalResults:
    results_path = RESULTS_DIR / get_settings().eval_results_file
    if not results_path.exists():
        pytest.fail(f"{results_path} is missing; run `uv run python -m evals.run_evals` first")
    return EvalResults.model_validate_json(results_path.read_text(encoding="utf-8"))


def test_results_cover_the_current_golden_set(results: EvalResults) -> None:
    expected_cases = select_cases(load_golden_cases(), smoke=results.case_selection == "smoke")
    assert {case.id for case in results.cases} == {case.id for case in expected_cases}


def test_faithfulness_meets_the_threshold(results: EvalResults) -> None:
    assert results.overall.faithfulness is not None
    assert results.overall.faithfulness >= get_settings().eval_min_faithfulness


def test_context_precision_meets_the_threshold(results: EvalResults) -> None:
    assert results.overall.context_precision is not None
    assert results.overall.context_precision >= get_settings().eval_min_context_precision


def test_judge_mean_meets_the_threshold(results: EvalResults) -> None:
    assert results.overall.judge_mean is not None
    assert results.overall.judge_mean >= get_settings().eval_min_judge_mean


def test_every_case_shows_its_expected_behaviour(results: EvalResults) -> None:
    misbehaving = [case.id for case in results.cases if not case.behaved]
    assert results.overall.behaviour_pass_rate >= get_settings().eval_min_behaviour_pass_rate, (
        misbehaving
    )

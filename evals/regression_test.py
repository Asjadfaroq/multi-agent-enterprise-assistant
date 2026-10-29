import pytest

from assistant.config import get_settings
from evals.golden_cases import load_golden_cases
from evals.results import RESULTS_PATH, EvalResults


@pytest.fixture(scope="module")
def results() -> EvalResults:
    if not RESULTS_PATH.exists():
        pytest.fail(f"{RESULTS_PATH} is missing; run `uv run python -m evals.run_evals` first")
    return EvalResults.model_validate_json(RESULTS_PATH.read_text(encoding="utf-8"))


def test_results_cover_the_current_golden_set(results: EvalResults) -> None:
    assert {case.id for case in results.cases} == {case.id for case in load_golden_cases()}


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

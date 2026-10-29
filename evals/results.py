from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from assistant.config import PROJECT_ROOT, Settings
from evals.behaviour import CheckResult
from evals.judge import JudgeVerdict
from evals.run_ragas import RagasScores

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
FULL_RESULTS_PATH = RESULTS_DIR / "latest.json"
SMOKE_RESULTS_PATH = RESULTS_DIR / "smoke.json"
CaseSelection = Literal["full", "smoke"]


class CaseResult(BaseModel):
    id: str
    category: str
    user: str
    question: str
    answer: str
    passed: bool
    failure_reasons: list[str]
    behaviour_checks: list[CheckResult]
    judge: JudgeVerdict | None
    judge_error: str | None
    ragas: RagasScores | None
    key_fact_recall: float | None
    latency_seconds: float

    @property
    def behaved(self) -> bool:
        return all(check.passed for check in self.behaviour_checks)


class ScoreSummary(BaseModel):
    cases: int
    passed: int
    behaviour_pass_rate: float
    judge_correctness: float | None
    judge_grounding: float | None
    judge_safety: float | None
    judge_mean: float | None
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    key_fact_recall: float | None


class RunTimings(BaseModel):
    graph_seconds: float
    scoring_seconds: float
    total_seconds: float
    cases_run: int
    cases_from_cache: int


class EvalResults(BaseModel):
    run_at: datetime
    case_selection: CaseSelection
    code_version: str
    timings: RunTimings
    agent_model: str
    judge_model: str
    thresholds: dict[str, float]
    overall: ScoreSummary
    per_category: dict[str, ScoreSummary]
    cases: list[CaseResult]


def mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def failure_reasons(
    checks: list[CheckResult],
    judge: JudgeVerdict | None,
    judge_error: str | None,
    ragas: RagasScores | None,
    settings: Settings,
) -> list[str]:
    reasons = [f"{check.name}: {check.detail}" for check in checks if not check.passed]
    if judge_error is not None:
        reasons.append(f"judge failed: {judge_error}")
    elif judge is not None:
        scores = {
            "correctness": judge.correctness,
            "grounding": judge.grounding,
            "safety": judge.safety,
        }
        low = {
            name: score
            for name, score in scores.items()
            if score < settings.eval_case_min_judge_score
        }
        if low:
            reasons.append(f"judge scores {low}: {judge.reason}")
    if ragas is not None:
        for name, value, minimum in (
            ("faithfulness", ragas.faithfulness, settings.eval_min_faithfulness),
            ("context_precision", ragas.context_precision, settings.eval_min_context_precision),
        ):
            if value is not None and value < minimum:
                reasons.append(f"{name} {value:.2f} is below {minimum}")
        reasons.extend(f"ragas error: {error}" for error in ragas.errors)
    return reasons


def summarise(results: list[CaseResult]) -> ScoreSummary:
    judged = [result.judge for result in results if result.judge is not None]
    ragas = [result.ragas for result in results if result.ragas is not None]
    return ScoreSummary(
        cases=len(results),
        passed=sum(result.passed for result in results),
        behaviour_pass_rate=sum(result.behaved for result in results) / len(results),
        judge_correctness=mean(verdict.correctness for verdict in judged),
        judge_grounding=mean(verdict.grounding for verdict in judged),
        judge_safety=mean(verdict.safety for verdict in judged),
        judge_mean=mean(verdict.mean_score for verdict in judged),
        faithfulness=mean(scores.faithfulness for scores in ragas),
        answer_relevancy=mean(scores.answer_relevancy for scores in ragas),
        context_precision=mean(scores.context_precision for scores in ragas),
        key_fact_recall=mean(result.key_fact_recall for result in results),
    )


def build_eval_results(
    case_results: list[CaseResult],
    settings: Settings,
    case_selection: CaseSelection,
    code_version: str,
    timings: RunTimings,
) -> EvalResults:
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for result in case_results:
        by_category[result.category].append(result)
    return EvalResults(
        run_at=datetime.now().astimezone(),
        case_selection=case_selection,
        code_version=code_version,
        timings=timings,
        agent_model=settings.agent_model,
        judge_model=settings.judge_model,
        thresholds={
            "faithfulness": settings.eval_min_faithfulness,
            "context_precision": settings.eval_min_context_precision,
            "judge_mean": settings.eval_min_judge_mean,
            "behaviour_pass_rate": settings.eval_min_behaviour_pass_rate,
            "case_min_judge_score": settings.eval_case_min_judge_score,
        },
        overall=summarise(case_results),
        per_category={category: summarise(results) for category, results in by_category.items()},
        cases=case_results,
    )

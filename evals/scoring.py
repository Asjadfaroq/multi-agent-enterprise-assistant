import asyncio

from assistant.config import Settings, get_settings
from assistant.llm import get_chat_model
from evals.behaviour import CheckResult, behaviour_checks, key_fact_recall
from evals.judge import JudgeVerdict, judge_case
from evals.results import CaseResult
from evals.run_cases import CaseRun
from evals.run_ragas import (
    RAGAS_CATEGORIES,
    RagasMetrics,
    RagasScores,
    build_ragas_metrics,
    score_with_ragas,
)


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


async def score_run(run: CaseRun, ragas_metrics: RagasMetrics) -> CaseResult:
    settings = get_settings()
    judge: JudgeVerdict | None = None
    judge_error: str | None = None
    try:
        judge = await judge_case(get_chat_model("judge"), run)
    except Exception as error:
        judge_error = repr(error)
    ragas = None
    if run.case.category in RAGAS_CATEGORIES and run.error is None:
        ragas = await score_with_ragas(ragas_metrics, run)
    checks = behaviour_checks(
        run, judge.behaviour_met if judge else None, settings.eval_arabic_min_letter_ratio
    )
    reasons = failure_reasons(checks, judge, judge_error, ragas, settings)
    return CaseResult(
        id=run.case.id,
        category=run.case.category,
        user=run.case.user,
        question=run.case.question,
        answer=run.answer_text,
        passed=not reasons,
        failure_reasons=reasons,
        behaviour_checks=checks,
        judge=judge,
        judge_error=judge_error,
        ragas=ragas,
        key_fact_recall=key_fact_recall(run),
        latency_seconds=run.latency_seconds,
    )


async def score_runs(runs: list[CaseRun]) -> list[CaseResult]:
    ragas_metrics = build_ragas_metrics()
    concurrency_limit = asyncio.Semaphore(get_settings().eval_scoring_concurrency)

    async def score_when_free(run: CaseRun) -> CaseResult:
        async with concurrency_limit:
            return await score_run(run, ragas_metrics)

    return list(await asyncio.gather(*(score_when_free(run) for run in runs)))

import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from assistant.config import get_settings
from assistant.graph.build import build_graph, checkpoint_serializer
from assistant.llm import get_chat_model
from assistant.observability.tracing import flush_traces
from evals.behaviour import behaviour_checks, key_fact_recall
from evals.golden_cases import load_golden_cases
from evals.judge import JudgeVerdict, judge_case
from evals.results import (
    RESULTS_PATH,
    CaseResult,
    EvalResults,
    build_eval_results,
    failure_reasons,
)
from evals.run_cases import CaseRun, clear_pending_approval_threads, reset_tickets, run_case
from evals.run_ragas import RAGAS_CATEGORIES, RagasMetrics, build_ragas_metrics, score_with_ragas

TABLE_COLUMNS = (
    "behaviour_pass_rate",
    "judge_correctness",
    "judge_grounding",
    "judge_safety",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
)


async def run_golden_cases() -> list[CaseRun]:
    settings = get_settings()
    cases = load_golden_cases()
    async with AsyncPostgresSaver.from_conn_string(
        settings.database_url, serde=checkpoint_serializer()
    ) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(checkpointer=checkpointer)
        reset_tickets()
        cleared = await clear_pending_approval_threads(graph, checkpointer)
        print(f"Reset sample tickets and cleared {cleared} pending approval threads")
        runs = []
        # Cases run one at a time: they share the tickets table and per-user memory.
        for number, case in enumerate(cases, start=1):
            run = await run_case(graph, case)
            print(f"[{number}/{len(cases)}] {case.id} {run.latency_seconds:.1f}s")
            runs.append(run)
    return runs


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


def format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def print_report(results: EvalResults) -> None:
    header = f"{'category':22} {'cases':>5} {'passed':>6} " + " ".join(
        f"{column[:12]:>12}" for column in TABLE_COLUMNS
    )
    print(header)
    rows = [*results.per_category.items(), ("overall", results.overall)]
    for category, summary in rows:
        scores = " ".join(f"{format_score(getattr(summary, c)):>12}" for c in TABLE_COLUMNS)
        print(f"{category:22} {summary.cases:>5} {summary.passed:>6} {scores}")
    print("\nFailing cases:")
    for case in results.cases:
        if not case.passed:
            print(f"- {case.id} ({case.user}): " + " | ".join(case.failure_reasons))


async def main() -> None:
    runs = await run_golden_cases()
    results = build_eval_results(await score_runs(runs), get_settings())
    RESULTS_PATH.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    flush_traces()
    print_report(results)
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

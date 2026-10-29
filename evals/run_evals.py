import argparse
import asyncio
import time

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from assistant.config import get_settings
from assistant.graph.build import build_graph, checkpoint_serializer
from assistant.llm import get_chat_model
from assistant.observability.tracing import flush_traces
from evals.behaviour import behaviour_checks, key_fact_recall
from evals.case_cache import cache_path, code_fingerprint, load_cached_runs, save_cached_runs
from evals.golden_cases import (
    GoldenCase,
    golden_set_fingerprint,
    load_golden_cases,
    select_cases,
)
from evals.judge import JudgeVerdict, judge_case
from evals.report import print_report
from evals.results import (
    FULL_RESULTS_PATH,
    SMOKE_RESULTS_PATH,
    CaseResult,
    RunTimings,
    build_eval_results,
    failure_reasons,
)
from evals.run_cases import (
    CaseRun,
    clear_pending_approval_threads,
    reset_tickets,
    run_cases_concurrently,
)
from evals.run_ragas import RAGAS_CATEGORIES, RagasMetrics, build_ragas_metrics, score_with_ragas


async def run_golden_cases(cases: list[GoldenCase]) -> list[CaseRun]:
    settings = get_settings()
    async with AsyncPostgresSaver.from_conn_string(
        settings.database_url, serde=checkpoint_serializer()
    ) as checkpointer:
        await checkpointer.setup()
        graph = build_graph(checkpointer=checkpointer)
        reset_tickets()
        cleared = await clear_pending_approval_threads(graph, checkpointer)
        # Flushed so progress shows even when output is redirected to a file.
        print(f"Reset sample tickets and cleared {cleared} pending approval threads", flush=True)
        return await run_cases_concurrently(graph, cases, settings.eval_case_concurrency)


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


async def main(smoke: bool, ignore_cache: bool) -> None:
    started = time.perf_counter()
    cases = select_cases(load_golden_cases(), smoke)
    code_version = code_fingerprint()
    run_cache = cache_path(golden_set_fingerprint(), code_version)
    cached_runs = {} if ignore_cache else load_cached_runs(run_cache)
    cases_to_run = [case for case in cases if case.id not in cached_runs]
    print(f"{len(cases) - len(cases_to_run)} of {len(cases)} cases from {run_cache.name}")
    new_runs = await run_golden_cases(cases_to_run) if cases_to_run else []
    graph_seconds = time.perf_counter() - started
    all_runs = cached_runs | {run.case.id: run for run in new_runs}
    save_cached_runs(run_cache, all_runs)
    case_results = await score_runs([all_runs[case.id] for case in cases])
    timings = RunTimings(
        graph_seconds=graph_seconds,
        scoring_seconds=time.perf_counter() - started - graph_seconds,
        total_seconds=time.perf_counter() - started,
        cases_run=len(cases_to_run),
        cases_from_cache=len(cases) - len(cases_to_run),
    )
    selection = "smoke" if smoke else "full"
    results = build_eval_results(case_results, get_settings(), selection, code_version, timings)
    results_path = SMOKE_RESULTS_PATH if smoke else FULL_RESULTS_PATH
    results_path.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    flush_traces()
    print_report(results)
    print(f"\nSaved {results_path} in {timings.total_seconds / 60:.1f} min")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the golden set and score it.")
    parser.add_argument("--smoke", action="store_true", help="run the 12 smoke cases only")
    parser.add_argument(
        "--fresh", action="store_true", help="re-run every case instead of using cached outputs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    asyncio.run(main(smoke=arguments.smoke, ignore_cache=arguments.fresh))

from evals.results import EvalResults

TABLE_COLUMNS = (
    "behaviour_pass_rate",
    "judge_correctness",
    "judge_grounding",
    "judge_safety",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
)


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
    timings = results.timings
    print(
        f"\n{timings.cases_run} cases run in {timings.graph_seconds:.0f}s, "
        f"{timings.cases_from_cache} from cache; scoring took {timings.scoring_seconds:.0f}s"
    )
    print("\nFailing cases:")
    for case in results.cases:
        if not case.passed:
            print(f"- {case.id} ({case.user}): " + " | ".join(case.failure_reasons))

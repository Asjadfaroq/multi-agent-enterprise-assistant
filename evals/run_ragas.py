from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any

import instructor
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, Faithfulness
from ragas.metrics.result import MetricResult

from assistant.config import get_settings
from assistant.llm import embed_texts
from evals.golden_cases import Category
from evals.run_cases import CaseRun

RAGAS_CATEGORIES: frozenset[Category] = frozenset({"factual", "multi_hop", "pii", "arabic"})


class RagasScores(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    errors: list[str] = []


class BgeRagasEmbeddings(BaseRagasEmbedding):
    """Lets ragas use the same local bge model the knowledge base is embedded with."""

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return embed_texts([text])[0].tolist()

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_text(text)


@dataclass(frozen=True)
class RagasMetrics:
    faithfulness: Faithfulness
    answer_relevancy: AnswerRelevancy
    context_precision: ContextPrecisionWithReference


def build_ragas_metrics() -> RagasMetrics:
    settings = get_settings()
    # ragas' llm_factory forces a tool call through instructor, which the judge model rejects,
    # so the client is built in instructor's JSON mode here instead.
    client = instructor.from_anthropic(
        AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value()),
        mode=instructor.Mode.ANTHROPIC_JSON,
    )
    llm = InstructorLLM(client=client, model=settings.judge_model, provider="anthropic")
    # ragas always sends temperature and top_p, which this Anthropic SDK rejects for the model.
    for unsupported_argument in ("temperature", "top_p"):
        llm.model_args.pop(unsupported_argument)
    return RagasMetrics(
        faithfulness=Faithfulness(llm=llm),
        answer_relevancy=AnswerRelevancy(llm=llm, embeddings=BgeRagasEmbeddings()),
        context_precision=ContextPrecisionWithReference(llm=llm),
    )


async def metric_value(
    name: str, scoring: Awaitable[MetricResult], errors: list[str]
) -> float | None:
    try:
        return (await scoring).value
    except Exception as error:
        # A metric that fails on one case, for example on a malformed judge reply, is reported
        # with that case instead of aborting the run.
        errors.append(f"{name}: {error!r}")
        return None


async def score_with_ragas(metrics: RagasMetrics, run: CaseRun) -> RagasScores:
    scores = RagasScores()
    if not run.answer_text:
        scores.errors.append("no answer to score")
        return scores
    question, answer, contexts = run.case.question, run.answer_text, run.retrieved_contexts
    if contexts:
        scores.faithfulness = await metric_value(
            "faithfulness", metrics.faithfulness.ascore(question, answer, contexts), scores.errors
        )
        scores.context_precision = await metric_value(
            "context_precision",
            metrics.context_precision.ascore(question, run.case.expected_answer, contexts),
            scores.errors,
        )
    else:
        # Nothing retrieved means nothing supports the answer and nothing relevant was found.
        scores.faithfulness, scores.context_precision = 0.0, 0.0
    scores.answer_relevancy = await metric_value(
        "answer_relevancy", metrics.answer_relevancy.ascore(question, answer), scores.errors
    )
    return scores

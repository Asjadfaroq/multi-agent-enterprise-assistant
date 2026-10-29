from functools import lru_cache
from typing import Any

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.predefined_recognizers import EmailRecognizer, PhoneRecognizer
from pydantic import BaseModel

from assistant.config import get_settings

# Saudi national IDs start with 1 and iqama (resident) numbers with 2; both are 10 digits.
# Presidio requires a score per pattern; every match is masked, so it only breaks overlap ties.
SAUDI_ID_PATTERN = Pattern("saudi national id or iqama", r"\b[12]\d{9}\b", 1.0)


@lru_cache
def pii_recognizers() -> tuple[EntityRecognizer, ...]:
    # The recognizers are used directly rather than through AnalyzerEngine, which would need a
    # spaCy language model only for entity types masked here by pattern anyway.
    return (
        EmailRecognizer(),
        PhoneRecognizer(supported_regions=get_settings().pii_phone_regions),
        PatternRecognizer(supported_entity="SAUDI_ID", patterns=[SAUDI_ID_PATTERN]),
    )


def find_pii(text: str) -> list[RecognizerResult]:
    matches = [
        match
        for recognizer in pii_recognizers()
        for match in recognizer.analyze(text, recognizer.supported_entities, None)
    ]
    # Keep the longest, then highest-scoring, match wherever two recognizers overlap.
    matches.sort(key=lambda match: (match.start, -(match.end - match.start), -match.score))
    kept: list[RecognizerResult] = []
    for match in matches:
        if not kept or match.start >= kept[-1].end:
            kept.append(match)
    return kept


def mask_pii(text: str) -> str:
    masked = text
    for match in reversed(find_pii(text)):
        masked = f"{masked[: match.start]}<{match.entity_type}>{masked[match.end :]}"
    return masked


def mask_pii_in_data(*, data: Any, **kwargs: Any) -> Any:
    """Langfuse mask hook: masks every string inside whatever a trace is about to record."""
    if isinstance(data, str):
        return mask_pii(data)
    if isinstance(data, dict):
        return {key: mask_pii_in_data(data=value) for key, value in data.items()}
    if isinstance(data, list | tuple):
        return [mask_pii_in_data(data=item) for item in data]
    return data


def mask_pii_in_tool_result(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return type(result).model_validate(mask_pii_in_data(data=result.model_dump()))
    if isinstance(result, list):
        return [mask_pii_in_tool_result(item) for item in result]
    return mask_pii_in_data(data=result)

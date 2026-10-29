from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, model_validator

from assistant.config import PROJECT_ROOT

GOLDEN_SET_PATH = PROJECT_ROOT / "evals" / "golden" / "golden_set.jsonl"

Category = Literal[
    "factual",
    "multi_hop",
    "permission_restricted",
    "write_action",
    "injection",
    "pii",
    "out_of_scope",
    "no_unrequested_write",
    "arabic",
]
Behaviour = Literal[
    "answer",
    "refuse_restricted",
    "pause_for_approval",
    "refuse_write",
    "block_injection",
    "ignore_poisoned_chunk",
    "answer_without_pii",
    "not_found",
    "answer_without_write",
    "answer_in_arabic",
]
CATEGORY_BEHAVIOURS: dict[Category, frozenset[Behaviour]] = {
    "factual": frozenset({"answer"}),
    "multi_hop": frozenset({"answer"}),
    "permission_restricted": frozenset({"refuse_restricted"}),
    "write_action": frozenset({"pause_for_approval", "refuse_write"}),
    "injection": frozenset({"block_injection", "ignore_poisoned_chunk"}),
    "pii": frozenset({"answer_without_pii"}),
    "out_of_scope": frozenset({"not_found"}),
    "no_unrequested_write": frozenset({"answer_without_write"}),
    "arabic": frozenset({"answer_in_arabic"}),
}


class GoldenCase(BaseModel):
    id: str
    category: Category
    user: str
    question: str
    expected_behavior: Behaviour
    expected_answer: str
    expected_sources: list[str]
    key_facts: list[str]
    forbidden_in_answer: list[str]
    expected_tool: str | None
    expected_tool_args: dict[str, Any] | None

    @model_validator(mode="after")
    def behaviour_fits_category(self) -> Self:
        if self.expected_behavior not in CATEGORY_BEHAVIOURS[self.category]:
            raise ValueError(
                f"{self.id}: behaviour {self.expected_behavior!r} does not fit {self.category!r}"
            )
        return self


def load_golden_cases(path: Path = GOLDEN_SET_PATH) -> list[GoldenCase]:
    with path.open(encoding="utf-8") as golden_file:
        return [GoldenCase.model_validate_json(line) for line in golden_file if line.strip()]

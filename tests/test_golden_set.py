import importlib.util
import json
from typing import Any

import pytest

from assistant.config import PROJECT_ROOT, get_settings
from assistant.guardrails.pii import find_pii
from assistant.knowledge.chunking import Document
from assistant.knowledge.ingest import load_documents
from assistant.mcp_server.auth import (
    WRITE_TOOL_NAMES,
    Role,
    allowed_acls_for_role,
    can_write_tickets,
)

GOLDEN_SET_PATH = PROJECT_ROOT / "evals" / "golden" / "golden_set.jsonl"
GOLDEN_CASE_COUNT = 50
CATEGORY_BEHAVIOURS = {
    "factual": {"answer"},
    "multi_hop": {"answer"},
    "permission_restricted": {"refuse_restricted"},
    "write_action": {"pause_for_approval", "refuse_write"},
    "injection": {"block_injection", "ignore_poisoned_chunk"},
    "pii": {"answer_without_pii"},
    "out_of_scope": {"not_found"},
}


def load_golden_cases() -> list[dict[str, Any]]:
    with GOLDEN_SET_PATH.open() as golden_file:
        return [json.loads(line) for line in golden_file if line.strip()]


def sample_user_roles() -> dict[str, Role]:
    # The golden set targets the users the seed script creates, so it reads them from there.
    spec = importlib.util.spec_from_file_location("seed", PROJECT_ROOT / "scripts" / "seed.py")
    seed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed)
    return seed.SAMPLE_USERS


GOLDEN_CASES = load_golden_cases()
USER_ROLES = sample_user_roles()
DOCUMENTS: dict[str, Document] = {
    document.doc_id: document for document in load_documents(get_settings().sample_docs_dir)
}


def cases_with(field: str, *values: str) -> list[dict[str, Any]]:
    return [case for case in GOLDEN_CASES if case[field] in values]


def readable_doc_ids(user_id: str) -> set[str]:
    allowed_acls = allowed_acls_for_role(USER_ROLES[user_id])
    return {doc_id for doc_id, document in DOCUMENTS.items() if document.acl in allowed_acls}


def test_golden_set_has_fifty_uniquely_named_cases_with_known_users() -> None:
    assert len(GOLDEN_CASES) == GOLDEN_CASE_COUNT
    assert len({case["id"] for case in GOLDEN_CASES}) == GOLDEN_CASE_COUNT
    assert {case["user"] for case in GOLDEN_CASES} <= set(USER_ROLES)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case["id"])
def test_every_case_uses_a_behaviour_allowed_for_its_category(case: dict[str, Any]) -> None:
    assert case["expected_behavior"] in CATEGORY_BEHAVIOURS[case["category"]]


@pytest.mark.parametrize(
    "case",
    cases_with("expected_behavior", "answer", "answer_without_pii"),
    ids=lambda case: case["id"],
)
def test_answerable_cases_cite_docs_the_user_can_read_and_quote_their_facts(
    case: dict[str, Any],
) -> None:
    assert case["expected_sources"] and set(case["expected_sources"]) <= readable_doc_ids(
        case["user"]
    )
    source_text = " ".join(DOCUMENTS[doc_id].body for doc_id in case["expected_sources"]).lower()
    assert all(fact.lower() in source_text for fact in case["key_facts"])


@pytest.mark.parametrize("case", cases_with("category", "multi_hop"), ids=lambda case: case["id"])
def test_multi_hop_cases_need_two_different_docs(case: dict[str, Any]) -> None:
    assert len(set(case["expected_sources"])) == 2


@pytest.mark.parametrize(
    "case", cases_with("category", "permission_restricted"), ids=lambda case: case["id"]
)
def test_restricted_facts_exist_only_in_docs_hidden_from_the_user(case: dict[str, Any]) -> None:
    readable = readable_doc_ids(case["user"])
    for fact in case["forbidden_in_answer"]:
        docs_with_fact = {doc_id for doc_id, doc in DOCUMENTS.items() if fact in doc.body}
        assert docs_with_fact and not docs_with_fact & readable, fact


@pytest.mark.parametrize(
    "case", cases_with("category", "write_action"), ids=lambda case: case["id"]
)
def test_write_cases_match_the_users_ticket_permissions(case: dict[str, Any]) -> None:
    may_write = can_write_tickets(USER_ROLES[case["user"]])
    assert case["expected_tool"] in WRITE_TOOL_NAMES
    assert may_write == (case["expected_behavior"] == "pause_for_approval")


@pytest.mark.parametrize("case", cases_with("category", "pii"), ids=lambda case: case["id"])
def test_pii_cases_contain_pii_the_masker_detects(case: dict[str, Any]) -> None:
    detected = {case["question"][match.start : match.end] for match in find_pii(case["question"])}
    assert set(case["forbidden_in_answer"]) <= detected


def test_poisoned_doc_case_targets_the_planted_claim() -> None:
    [case] = cases_with("expected_behavior", "ignore_poisoned_chunk")
    poisoned_body = DOCUMENTS["vendor-onboarding-faq"].body
    assert all(planted in poisoned_body for planted in case["forbidden_in_answer"])

import pytest

from assistant.config import get_settings
from assistant.guardrails.pii import find_pii
from assistant.knowledge.chunking import Document
from assistant.knowledge.ingest import load_documents
from assistant.mcp_server.auth import WRITE_TOOL_NAMES, allowed_acls_for_role, can_write_tickets
from evals.behaviour import arabic_letter_ratio
from evals.golden_cases import GoldenCase, load_golden_cases
from scripts.seed import SAMPLE_USERS

GOLDEN_CASE_COUNT = 58
GOLDEN_CASES = load_golden_cases()
DOCUMENTS: dict[str, Document] = {
    document.doc_id: document for document in load_documents(get_settings().sample_docs_dir)
}


def cases_where(**field_values: str) -> list[GoldenCase]:
    return [
        case
        for case in GOLDEN_CASES
        if all(getattr(case, field) == value for field, value in field_values.items())
    ]


def readable_doc_ids(user_id: str) -> set[str]:
    allowed_acls = allowed_acls_for_role(SAMPLE_USERS[user_id])
    return {doc_id for doc_id, document in DOCUMENTS.items() if document.acl in allowed_acls}


def case_id(case: GoldenCase) -> str:
    return case.id


def test_golden_set_has_uniquely_named_cases_for_seeded_users() -> None:
    assert len(GOLDEN_CASES) == GOLDEN_CASE_COUNT
    assert len({case.id for case in GOLDEN_CASES}) == GOLDEN_CASE_COUNT
    assert {case.user for case in GOLDEN_CASES} <= set(SAMPLE_USERS)


@pytest.mark.parametrize(
    "case", [case for case in GOLDEN_CASES if case.expected_sources], ids=case_id
)
def test_cases_with_sources_cite_docs_the_user_can_read_and_quote_their_facts(
    case: GoldenCase,
) -> None:
    assert set(case.expected_sources) <= readable_doc_ids(case.user)
    source_text = " ".join(DOCUMENTS[doc_id].body for doc_id in case.expected_sources).lower()
    assert all(fact.lower() in source_text for fact in case.key_facts)


@pytest.mark.parametrize(
    "case",
    [case for case in GOLDEN_CASES if case.expected_behavior in {"answer", "answer_without_pii"}],
    ids=case_id,
)
def test_answerable_cases_name_their_sources(case: GoldenCase) -> None:
    assert case.expected_sources and case.key_facts


@pytest.mark.parametrize("case", cases_where(category="multi_hop"), ids=case_id)
def test_multi_hop_cases_need_two_different_docs(case: GoldenCase) -> None:
    assert len(set(case.expected_sources)) == 2


@pytest.mark.parametrize("case", cases_where(category="permission_restricted"), ids=case_id)
def test_restricted_facts_exist_only_in_docs_hidden_from_the_user(case: GoldenCase) -> None:
    readable = readable_doc_ids(case.user)
    for fact in case.forbidden_in_answer:
        docs_with_fact = {doc_id for doc_id, doc in DOCUMENTS.items() if fact in doc.body}
        assert docs_with_fact and not docs_with_fact & readable, fact


@pytest.mark.parametrize("case", cases_where(category="write_action"), ids=case_id)
def test_write_cases_match_the_users_ticket_permissions(case: GoldenCase) -> None:
    assert case.expected_tool in WRITE_TOOL_NAMES
    assert can_write_tickets(SAMPLE_USERS[case.user]) == (
        case.expected_behavior == "pause_for_approval"
    )


@pytest.mark.parametrize("case", cases_where(category="no_unrequested_write"), ids=case_id)
def test_no_write_cases_come_from_users_who_could_write(case: GoldenCase) -> None:
    assert can_write_tickets(SAMPLE_USERS[case.user])
    assert case.expected_tool is None and case.expected_tool_args is None


@pytest.mark.parametrize("case", cases_where(category="arabic"), ids=case_id)
def test_arabic_cases_are_written_in_arabic_and_name_their_sources(case: GoldenCase) -> None:
    assert arabic_letter_ratio(case.question) >= get_settings().eval_arabic_min_letter_ratio
    assert case.expected_sources and case.key_facts


def test_arabic_cases_cover_public_hr_and_multi_hop_questions() -> None:
    arabic_cases = cases_where(category="arabic")
    assert [case.user for case in arabic_cases].count("nora") == 2
    assert any(case.user == "huda" for case in arabic_cases)
    assert any(case.user == "omar" and len(case.expected_sources) == 2 for case in arabic_cases)


@pytest.mark.parametrize("case", cases_where(category="pii"), ids=case_id)
def test_pii_cases_contain_pii_the_masker_detects(case: GoldenCase) -> None:
    detected = {case.question[match.start : match.end] for match in find_pii(case.question)}
    assert set(case.forbidden_in_answer) <= detected


def test_poisoned_doc_case_targets_the_planted_claim() -> None:
    [case] = cases_where(expected_behavior="ignore_poisoned_chunk")
    poisoned_body = DOCUMENTS["vendor-onboarding-faq"].body
    assert all(planted in poisoned_body for planted in case.forbidden_in_answer)

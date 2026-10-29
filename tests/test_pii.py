from datetime import UTC, datetime

from assistant.config import get_settings
from assistant.graph.state import new_turn_state
from assistant.guardrails.pii import mask_pii, mask_pii_in_data, mask_pii_in_tool_result
from assistant.knowledge.ingest import load_documents
from assistant.mcp_server.tools_tickets import Ticket


def test_email_phone_and_saudi_ids_are_masked() -> None:
    text = (
        "Mail sara.k@larkspur.sa or call +966 50 123 4567. "
        "National ID 1012345678, iqama 2412345678."
    )

    assert mask_pii(text) == (
        "Mail <EMAIL_ADDRESS> or call <PHONE_NUMBER>. National ID <SAUDI_ID>, iqama <SAUDI_ID>."
    )


def test_prices_dates_ticket_numbers_and_limits_are_left_alone() -> None:
    text = "Ticket 19: refund 149 SAR by 2026-10-29; the API allows 600 requests per minute, 24/7."

    assert mask_pii(text) == text


def test_sample_docs_contain_no_pii_except_the_planted_attacker_address() -> None:
    masked_docs = {
        document.doc_id
        for document in load_documents(get_settings().sample_docs_dir)
        if mask_pii(document.body) != document.body
    }

    assert masked_docs == {"vendor-onboarding-faq"}


def test_user_question_is_masked_before_it_enters_the_graph_state() -> None:
    state = new_turn_state("nora", "My iqama is 2412345678, can HR update it?")

    assert state["messages"][0].text == "My iqama is <SAUDI_ID>, can HR update it?"


def test_trace_data_is_masked_at_every_depth() -> None:
    trace_data = {"input": ["Call 0501234567"], "metadata": {"user": "a@b.co", "step": 3}}

    assert mask_pii_in_data(data=trace_data) == {
        "input": ["Call <PHONE_NUMBER>"],
        "metadata": {"user": "<EMAIL_ADDRESS>", "step": 3},
    }


def test_tool_results_keep_their_type_after_masking() -> None:
    now = datetime.now(UTC)
    ticket = Ticket(
        id=1,
        title="Customer ali@example.com cannot log in",
        description="Call back on +966 50 123 4567",
        category="account",
        priority="P2",
        status="open",
        created_by="khalid",
        created_at=now,
        updated_at=now,
    )

    [masked] = mask_pii_in_tool_result([ticket])

    assert isinstance(masked, Ticket)
    assert (masked.title, masked.description) == (
        "Customer <EMAIL_ADDRESS> cannot log in",
        "Call back on <PHONE_NUMBER>",
    )

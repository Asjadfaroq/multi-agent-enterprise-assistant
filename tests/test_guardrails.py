import pytest

from assistant.guardrails.schema import Plan, parse_model_output


def test_json_wrapped_in_a_markdown_fence_is_parsed() -> None:
    reply = '```json\n{"steps": ["search"], "needs_tools": ["search_documents"]}\n```'

    plan = parse_model_output(reply, Plan)

    assert plan.needs_tools == ["search_documents"]


def test_plan_naming_an_unknown_tool_is_rejected_with_the_model_name() -> None:
    reply = '{"steps": ["drop tables"], "needs_tools": ["run_sql"]}'

    with pytest.raises(ValueError, match="LLM reply is not a valid Plan"):
        parse_model_output(reply, Plan)


def test_reply_that_is_not_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="LLM reply is not a valid Plan"):
        parse_model_output("I think we should search the docs.", Plan)

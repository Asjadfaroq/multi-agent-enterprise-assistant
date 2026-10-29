import asyncio
import json

from conftest import FakeToolbox, ScriptedChatModel, connect_or_skip

from assistant.config import get_settings
from assistant.graph.build import build_graph
from assistant.graph.nodes.planner import PLANNER_INSTRUCTIONS
from assistant.graph.state import new_turn_state
from assistant.guardrails.schema import ProfileFact
from assistant.memory.profile import PostgresProfileStore, create_user_memory_table

PLAN_WITHOUT_TOOLS = json.dumps({"steps": ["reply"], "needs_tools": []})
DRAFT = json.dumps({"text": "Noted.", "citations": []})
SUPPORTED_REVIEW = json.dumps({"supported": True, "reason": "No factual claims."})


def run_turn(toolbox: FakeToolbox, question: str, memory_reply: str) -> ScriptedChatModel:
    scripted_model = ScriptedChatModel(
        messages=iter([PLAN_WITHOUT_TOOLS, DRAFT, SUPPORTED_REVIEW, memory_reply])
    )
    graph = build_graph(chat_model_factory=lambda role: scripted_model)
    asyncio.run(graph.ainvoke(new_turn_state("lina", question), context=toolbox.context(False)))
    return scripted_model


def test_facts_the_user_states_are_saved_to_their_profile(toolbox: FakeToolbox) -> None:
    memory_reply = json.dumps(
        {
            "facts": [
                {"key": "team", "value": "Finance"},
                {"key": "preferred_language", "value": "Arabic"},
            ]
        }
    )

    run_turn(toolbox, "I'm on the Finance team, please answer in Arabic.", memory_reply)

    assert toolbox.profile_store.facts_by_user["lina"] == {
        "team": "Finance",
        "preferred_language": "Arabic",
    }


def test_no_more_facts_than_configured_are_saved_per_turn(toolbox: FakeToolbox) -> None:
    facts = [
        {"key": "team", "value": "Finance"},
        {"key": "preferred_language", "value": "Arabic"},
        {"key": "team", "value": "Payments"},
    ]

    run_turn(toolbox, "Finance, Arabic, and I moved to Payments.", json.dumps({"facts": facts}))

    saved_facts = toolbox.profile_store.facts_by_user["lina"]
    assert len(facts) > get_settings().max_profile_facts_per_turn
    assert saved_facts == {"team": "Finance", "preferred_language": "Arabic"}


def test_planner_sees_facts_remembered_in_earlier_turns(toolbox: FakeToolbox) -> None:
    toolbox.profile_store.facts_by_user["lina"] = {"preferred_language": "Arabic"}

    scripted_model = run_turn(toolbox, "Any open P1 tickets?", json.dumps({"facts": []}))

    planner_prompt = next(
        prompt
        for prompt in scripted_model.received_prompts
        if prompt[0].text.startswith(PLANNER_INSTRUCTIONS)
    )
    assert "- preferred_language: Arabic" in planner_prompt[1].text


def test_postgres_profile_keeps_the_latest_value_for_each_fact() -> None:
    connection = connect_or_skip(autocommit=True)
    create_user_memory_table(connection)
    store = PostgresProfileStore(get_settings().database_url)
    try:
        asyncio.run(store.save_facts("test-memory-user", [ProfileFact(key="team", value="HR")]))
        asyncio.run(
            store.save_facts("test-memory-user", [ProfileFact(key="team", value="Finance")])
        )

        assert asyncio.run(store.load_facts("test-memory-user")) == [
            ProfileFact(key="team", value="Finance")
        ]
    finally:
        connection.execute("DELETE FROM user_memory WHERE user_id = 'test-memory-user'")
        connection.close()

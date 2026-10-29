import asyncio
import itertools
import json

from conftest import FakeToolbox, GraphBuilder, ScriptedChatModel
from langchain_core.messages import AIMessage

from assistant.graph.build import build_graph
from assistant.graph.nodes.executor import EXECUTOR_INSTRUCTIONS
from assistant.graph.nodes.reviewer import UNSUPPORTED_ANSWER_FALLBACK
from assistant.graph.state import new_turn_state

QUESTION = "How long is maternity leave?"
PLAN_WITH_SEARCH = json.dumps(
    {"steps": ["search leave policy"], "needs_tools": ["search_documents"]}
)
PLAN_WITHOUT_TOOLS = json.dumps({"steps": ["greet the user"], "needs_tools": []})
GROUNDED_DRAFT = json.dumps(
    {
        "text": "Maternity leave is 14 weeks [parental-leave-policy].",
        "citations": ["parental-leave-policy"],
    }
)
REVISED_DRAFT = json.dumps(
    {
        "text": "It is 14 weeks, fully paid [parental-leave-policy].",
        "citations": ["parental-leave-policy"],
    }
)
SUPPORTED_REVIEW = json.dumps({"supported": True, "reason": "Matches the excerpt."})
UNSUPPORTED_REVIEW = json.dumps({"supported": False, "reason": "Pay is not mentioned."})


def run_graph(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox, llm_replies: list[str]
) -> tuple[list[str], str]:
    graph = build_scripted_graph(llm_replies, None)
    context = toolbox.context(can_write_tickets=False)

    async def collect_visited_nodes_and_answer() -> tuple[list[str], str]:
        visited_nodes: list[str] = []
        final_answer = ""
        async for update in graph.astream(
            new_turn_state("huda", QUESTION), context=context, stream_mode="updates"
        ):
            for node_name, node_update in update.items():
                visited_nodes.append(node_name)
                if node_update and "messages" in node_update:
                    final_answer = node_update["messages"][-1].text
        return visited_nodes, final_answer

    return asyncio.run(collect_visited_nodes_and_answer())


def test_company_question_is_planned_retrieved_drafted_and_reviewed(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [PLAN_WITH_SEARCH, GROUNDED_DRAFT, SUPPORTED_REVIEW]

    visited, answer = run_graph(build_scripted_graph, toolbox, replies)

    assert visited == ["planner", "retriever", "executor", "reviewer"]
    assert answer == "Maternity leave is 14 weeks [parental-leave-policy]."
    assert toolbox.search_queries == [QUESTION]


def test_small_talk_skips_document_search(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [PLAN_WITHOUT_TOOLS, GROUNDED_DRAFT, SUPPORTED_REVIEW]

    visited, _ = run_graph(build_scripted_graph, toolbox, replies)

    assert visited == ["planner", "executor", "reviewer"]
    assert toolbox.search_queries == []


def test_unsupported_draft_is_revised_once_then_accepted(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [
        PLAN_WITH_SEARCH,
        GROUNDED_DRAFT,
        UNSUPPORTED_REVIEW,
        REVISED_DRAFT,
        SUPPORTED_REVIEW,
    ]

    visited, answer = run_graph(build_scripted_graph, toolbox, replies)

    assert visited == ["planner", "retriever", "executor", "reviewer", "executor", "reviewer"]
    assert answer == "It is 14 weeks, fully paid [parental-leave-policy]."


def test_draft_still_unsupported_after_revision_ends_with_fallback_answer(
    build_scripted_graph: GraphBuilder, toolbox: FakeToolbox
) -> None:
    replies = [
        PLAN_WITH_SEARCH,
        GROUNDED_DRAFT,
        UNSUPPORTED_REVIEW,
        REVISED_DRAFT,
        UNSUPPORTED_REVIEW,
    ]

    visited, answer = run_graph(build_scripted_graph, toolbox, replies)

    assert visited == ["planner", "retriever", "executor", "reviewer", "executor", "reviewer"]
    assert answer == UNSUPPORTED_ANSWER_FALLBACK


def test_executor_history_is_only_appended_to_across_tool_calls_and_revisions(
    toolbox: FakeToolbox,
) -> None:
    # Claude signs thinking blocks against the messages before them, so any rewrite of earlier
    # messages makes the next call fail.
    list_tickets_call = AIMessage(
        "", tool_calls=[{"name": "list_tickets", "args": {}, "id": "call-list"}]
    )
    scripted_model = ScriptedChatModel(
        messages=iter(
            [
                PLAN_WITH_SEARCH,
                list_tickets_call,
                GROUNDED_DRAFT,
                UNSUPPORTED_REVIEW,
                REVISED_DRAFT,
                SUPPORTED_REVIEW,
            ]
        )
    )
    graph = build_graph(chat_model_factory=lambda role: scripted_model)

    asyncio.run(graph.ainvoke(new_turn_state("huda", QUESTION), context=toolbox.context(False)))

    executor_prompts = [
        prompt
        for prompt in scripted_model.received_prompts
        if prompt[0].text.startswith(EXECUTOR_INSTRUCTIONS)
    ]
    assert len(executor_prompts) == 3
    for earlier, later in itertools.pairwise(executor_prompts):
        assert later[: len(earlier)] == earlier

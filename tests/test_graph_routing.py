import asyncio
import json

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from assistant.graph.build import build_graph
from assistant.graph.nodes.reviewer import UNSUPPORTED_ANSWER_FALLBACK
from assistant.graph.state import new_turn_state
from assistant.llm import ModelRole
from assistant.mcp_server.tools_kb import SearchHit

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

LEAVE_POLICY_HIT = SearchHit(
    doc_id="parental-leave-policy",
    acl="hr",
    title="Parental Leave Policy",
    content="Birthing parents receive 14 weeks of fully paid maternity leave.",
    score=0.9,
)


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, user_id: str, query: str) -> list[SearchHit]:
        self.calls.append((user_id, query))
        return [LEAVE_POLICY_HIT]


def run_graph(llm_replies: list[str], search: RecordingSearch) -> tuple[list[str], str]:
    fake_llm = FakeListChatModel(responses=llm_replies)

    def fake_chat_model_factory(role: ModelRole) -> FakeListChatModel:
        return fake_llm

    graph = build_graph(chat_model_factory=fake_chat_model_factory, search_documents=search)

    async def collect_visited_nodes_and_answer() -> tuple[list[str], str]:
        visited_nodes: list[str] = []
        final_answer = ""
        async for update in graph.astream(new_turn_state("huda", QUESTION), stream_mode="updates"):
            for node_name, node_update in update.items():
                visited_nodes.append(node_name)
                if node_update and "messages" in node_update:
                    final_answer = node_update["messages"][-1].text
        return visited_nodes, final_answer

    return asyncio.run(collect_visited_nodes_and_answer())


def test_company_question_is_planned_retrieved_drafted_and_reviewed() -> None:
    search = RecordingSearch()

    visited, answer = run_graph([PLAN_WITH_SEARCH, GROUNDED_DRAFT, SUPPORTED_REVIEW], search)

    assert visited == ["planner", "retriever", "executor", "reviewer"]
    assert answer == "Maternity leave is 14 weeks [parental-leave-policy]."
    assert search.calls == [("huda", QUESTION)]


def test_small_talk_skips_document_search() -> None:
    search = RecordingSearch()

    visited, _ = run_graph([PLAN_WITHOUT_TOOLS, GROUNDED_DRAFT, SUPPORTED_REVIEW], search)

    assert visited == ["planner", "executor", "reviewer"]
    assert search.calls == []


def test_unsupported_draft_is_revised_once_then_accepted() -> None:
    replies = [
        PLAN_WITH_SEARCH,
        GROUNDED_DRAFT,
        UNSUPPORTED_REVIEW,
        REVISED_DRAFT,
        SUPPORTED_REVIEW,
    ]

    visited, answer = run_graph(replies, RecordingSearch())

    assert visited == ["planner", "retriever", "executor", "reviewer", "executor", "reviewer"]
    assert answer == "It is 14 weeks, fully paid [parental-leave-policy]."


def test_draft_still_unsupported_after_revision_ends_with_fallback_answer() -> None:
    replies = [
        PLAN_WITH_SEARCH,
        GROUNDED_DRAFT,
        UNSUPPORTED_REVIEW,
        REVISED_DRAFT,
        UNSUPPORTED_REVIEW,
    ]

    visited, answer = run_graph(replies, RecordingSearch())

    assert visited == ["planner", "retriever", "executor", "reviewer", "executor", "reviewer"]
    assert answer == UNSUPPORTED_ANSWER_FALLBACK

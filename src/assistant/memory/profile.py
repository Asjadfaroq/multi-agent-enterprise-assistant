import psycopg
from langchain_core.language_models import BaseChatModel
from langgraph.runtime import Runtime
from psycopg.rows import class_row

from assistant.graph.state import (
    AgentState,
    AssistantContext,
    ContextNode,
    StateUpdate,
    latest_question,
)
from assistant.guardrails.schema import ProfileFact, ProfileFacts, ask_for_json

MEMORY_INSTRUCTIONS = """\
You maintain a short long-term profile of an employee who uses the internal assistant.
From the employee's message, extract only facts they state about themselves: their preferred \
reply language and the team they work in. Do not guess or infer. Return an empty list when the \
message states neither."""


def create_user_memory_table(connection: psycopg.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id text NOT NULL,
            key text NOT NULL,
            value text NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (user_id, key)
        )
        """
    )


class PostgresProfileStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    async def load_facts(self, user_id: str) -> list[ProfileFact]:
        async with (
            await psycopg.AsyncConnection.connect(self.database_url) as connection,
            connection.cursor(row_factory=class_row(ProfileFact)) as cursor,
        ):
            await cursor.execute(
                "SELECT key, value FROM user_memory WHERE user_id = %s ORDER BY key", (user_id,)
            )
            return await cursor.fetchall()

    async def save_facts(self, user_id: str, facts: list[ProfileFact]) -> None:
        async with (
            await psycopg.AsyncConnection.connect(self.database_url) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.executemany(
                "INSERT INTO user_memory (user_id, key, value) VALUES (%s, %s, %s) "
                "ON CONFLICT (user_id, key) "
                "DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                [(user_id, fact.key, fact.value) for fact in facts],
            )


def make_memory_node(llm: BaseChatModel, max_facts_per_turn: int) -> ContextNode:
    async def remember_user_facts(
        state: AgentState, runtime: Runtime[AssistantContext]
    ) -> StateUpdate:
        extracted = await ask_for_json(
            llm, MEMORY_INSTRUCTIONS, latest_question(state), ProfileFacts
        )
        new_facts = extracted.facts[:max_facts_per_turn]
        if new_facts:
            await runtime.context.profile_store.save_facts(state["user_id"], new_facts)
        return {}

    return remember_user_facts

import asyncio
from collections.abc import Iterator

import numpy as np
import psycopg
import pytest
from conftest import connect_or_skip
from mcp.server.fastmcp import FastMCP
from pgvector.psycopg import register_vector

from assistant.config import get_settings
from assistant.knowledge.chunking import Chunk
from assistant.knowledge.ingest import create_chunks_table, replace_document_chunks
from assistant.mcp_server.auth import (
    Role,
    allowed_acls_for_role,
    allowed_acls_for_user,
    create_users_table,
    require_ingest_permission,
    require_ticket_write_permission,
    user_id_from_headers,
)
from assistant.mcp_server.server import build_server, tools_visible_to_role
from assistant.mcp_server.tools_kb import search_chunks
from assistant.mcp_server.tools_tickets import create_tickets_table, insert_ticket

SEARCH_EVERYTHING_K = 1000


@pytest.fixture
def connection() -> Iterator[psycopg.Connection]:
    """Real Postgres connection whose changes are rolled back after each test."""
    connection = connect_or_skip(autocommit=False)
    create_chunks_table(connection, get_settings().embedding_dimensions)
    create_users_table(connection)
    create_tickets_table(connection)
    register_vector(connection)
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def unit_vector() -> np.ndarray:
    vector = np.zeros(get_settings().embedding_dimensions, dtype=np.float32)
    vector[0] = 1.0
    return vector


def insert_test_chunk(connection: psycopg.Connection, doc_id: str, acl: str) -> None:
    chunk = Chunk(doc_id=doc_id, acl=acl, content=f"{acl} text", metadata={"title": doc_id})
    replace_document_chunks(connection, doc_id, [chunk], np.array([unit_vector()]))


def insert_test_user(connection: psycopg.Connection, user_id: str, role: str) -> None:
    connection.execute("INSERT INTO users (id, role) VALUES (%s, %s)", (user_id, role))


def test_viewer_role_may_only_read_public_documents() -> None:
    assert allowed_acls_for_role("viewer") == ["public"]


def test_hr_manager_role_may_read_hr_documents() -> None:
    assert "hr" in allowed_acls_for_role("hr_manager")


def test_viewer_cannot_read_hr_chunks(connection: psycopg.Connection) -> None:
    insert_test_user(connection, "test-viewer", "viewer")
    insert_test_chunk(connection, "test-hr-doc", "hr")
    insert_test_chunk(connection, "test-public-doc", "public")

    allowed_acls = allowed_acls_for_user(connection, "test-viewer")
    hits = search_chunks(connection, unit_vector(), allowed_acls, SEARCH_EVERYTHING_K)

    assert "test-public-doc" in {hit.doc_id for hit in hits}
    assert all(hit.acl == "public" for hit in hits)


def test_hr_manager_can_read_the_hr_chunk_hidden_from_viewers(
    connection: psycopg.Connection,
) -> None:
    insert_test_user(connection, "test-hr-manager", "hr_manager")
    insert_test_chunk(connection, "test-hr-doc", "hr")

    allowed_acls = allowed_acls_for_user(connection, "test-hr-manager")
    hits = search_chunks(connection, unit_vector(), allowed_acls, SEARCH_EVERYTHING_K)

    assert "test-hr-doc" in {hit.doc_id for hit in hits}


def test_unknown_user_is_refused(connection: psycopg.Connection) -> None:
    with pytest.raises(PermissionError, match="Unknown user 'nobody'"):
        allowed_acls_for_user(connection, "nobody")


def test_tool_call_without_caller_header_is_refused() -> None:
    with pytest.raises(PermissionError, match="X-User header is missing"):
        user_id_from_headers({})


def test_viewer_cannot_create_or_update_tickets(connection: psycopg.Connection) -> None:
    insert_test_user(connection, "test-viewer", "viewer")

    with pytest.raises(PermissionError, match="role 'viewer' may not create or update tickets"):
        require_ticket_write_permission(connection, "test-viewer")


def test_agent_ticket_is_recorded_as_created_by_the_agent(connection: psycopg.Connection) -> None:
    insert_test_user(connection, "test-agent", "agent")

    require_ticket_write_permission(connection, "test-agent")
    ticket = insert_ticket(connection, "Printer jam", "Floor 2 printer", "it", "P3", "test-agent")

    assert (ticket.created_by, ticket.status) == ("test-agent", "open")


@pytest.mark.parametrize("role", ["viewer", "agent", "engineer", "hr_manager"])
def test_only_managers_may_ingest_documents(connection: psycopg.Connection, role: str) -> None:
    insert_test_user(connection, f"test-{role}", role)

    with pytest.raises(PermissionError, match="may not ingest documents"):
        require_ingest_permission(connection, f"test-{role}")


def test_manager_may_ingest_documents(connection: psycopg.Connection) -> None:
    insert_test_user(connection, "test-manager", "manager")

    require_ingest_permission(connection, "test-manager")


def test_mcp_server_lists_write_tools_only_for_ticket_writers() -> None:
    all_tools = asyncio.run(FastMCP.list_tools(build_server()))

    def visible_tool_names(role: Role) -> set[str]:
        return {tool.name for tool in tools_visible_to_role(all_tools, role)}

    assert {"create_ticket", "update_ticket"}.isdisjoint(visible_tool_names("viewer"))
    assert {"create_ticket", "update_ticket"} <= visible_tool_names("agent")

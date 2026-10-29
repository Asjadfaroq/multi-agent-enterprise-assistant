from collections.abc import Iterator

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector

from assistant.config import get_settings
from assistant.knowledge.chunking import Chunk
from assistant.knowledge.ingest import create_chunks_table, replace_document_chunks
from assistant.mcp_server.auth import (
    allowed_acls_for_role,
    allowed_acls_for_user,
    create_users_table,
    current_user_id,
    require_ticket_write_permission,
)
from assistant.mcp_server.tools_kb import search_chunks
from assistant.mcp_server.tools_tickets import create_tickets_table, insert_ticket

DB_CONNECT_TIMEOUT_SECONDS = 2
SEARCH_EVERYTHING_K = 1000


@pytest.fixture
def connection() -> Iterator[psycopg.Connection]:
    """Real Postgres connection whose changes are rolled back after each test."""
    settings = get_settings()
    try:
        connection = psycopg.connect(
            settings.database_url, connect_timeout=DB_CONNECT_TIMEOUT_SECONDS
        )
    except psycopg.OperationalError as error:
        pytest.skip(f"Postgres is not reachable at {settings.database_url}: {error}")
    create_chunks_table(connection, settings.embedding_dimensions)
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


def test_search_without_caller_identity_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSISTANT_USER_ID", "")
    get_settings.cache_clear()
    try:
        with pytest.raises(PermissionError, match="ASSISTANT_USER_ID is not set"):
            current_user_id()
    finally:
        get_settings.cache_clear()


def test_viewer_cannot_create_or_update_tickets(connection: psycopg.Connection) -> None:
    insert_test_user(connection, "test-viewer", "viewer")

    with pytest.raises(PermissionError, match="role 'viewer' may not create or update tickets"):
        require_ticket_write_permission(connection, "test-viewer")


def test_agent_ticket_is_recorded_as_created_by_the_agent(connection: psycopg.Connection) -> None:
    insert_test_user(connection, "test-agent", "agent")

    require_ticket_write_permission(connection, "test-agent")
    ticket = insert_ticket(connection, "Printer jam", "Floor 2 printer", "it", "P3", "test-agent")

    assert (ticket.created_by, ticket.status) == ("test-agent", "open")

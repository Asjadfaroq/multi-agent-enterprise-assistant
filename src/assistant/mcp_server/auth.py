from collections.abc import Mapping
from typing import Literal, cast

import psycopg
from mcp.server.fastmcp import Context

from assistant.knowledge.chunking import Acl

Role = Literal["viewer", "agent", "engineer", "hr_manager", "manager"]

ALLOWED_ACLS_BY_ROLE: dict[Role, tuple[Acl, ...]] = {
    "viewer": ("public",),
    "agent": ("public",),
    "engineer": ("public", "engineering"),
    "hr_manager": ("public", "hr"),
    "manager": ("public", "hr", "engineering"),
}

WRITE_TOOL_NAMES = frozenset({"create_ticket", "update_ticket"})
TICKET_WRITER_ROLES: frozenset[Role] = frozenset({"agent", "manager"})
DOCUMENT_INGEST_ROLES: frozenset[Role] = frozenset({"manager"})

# Set by the API from its own authenticated caller; the model never supplies it. The MCP server
# trusts it because it only listens where the API can reach it.
CALLER_HEADER = "X-User"


def create_users_table(connection: psycopg.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS users (id text PRIMARY KEY, role text NOT NULL)")


def user_id_from_headers(headers: Mapping[str, str]) -> str:
    user_id = headers.get(CALLER_HEADER, "")
    if not user_id:
        raise PermissionError(f"The {CALLER_HEADER} header is missing; the caller is unknown")
    return user_id


def caller_user_id(context: Context) -> str:
    request = context.request_context.request
    if request is None:
        raise PermissionError("Tool was called outside an HTTP request; the caller is unknown")
    return user_id_from_headers(request.headers)


def role_for_user(connection: psycopg.Connection, user_id: str) -> Role:
    row = connection.execute("SELECT role FROM users WHERE id = %s", (user_id,)).fetchone()
    if row is None:
        raise PermissionError(f"Unknown user {user_id!r}")
    role = row[0]
    if role not in ALLOWED_ACLS_BY_ROLE:
        raise PermissionError(f"User {user_id!r} has unrecognised role {role!r}")
    return cast(Role, role)


def allowed_acls_for_role(role: Role) -> list[Acl]:
    return list(ALLOWED_ACLS_BY_ROLE[role])


def allowed_acls_for_user(connection: psycopg.Connection, user_id: str) -> list[Acl]:
    return allowed_acls_for_role(role_for_user(connection, user_id))


def can_write_tickets(role: Role) -> bool:
    return role in TICKET_WRITER_ROLES


def require_known_user(connection: psycopg.Connection, user_id: str) -> None:
    role_for_user(connection, user_id)


def require_ticket_write_permission(connection: psycopg.Connection, user_id: str) -> None:
    role = role_for_user(connection, user_id)
    if not can_write_tickets(role):
        raise PermissionError(
            f"User {user_id!r} with role {role!r} may not create or update tickets"
        )


def require_ingest_permission(connection: psycopg.Connection, user_id: str) -> None:
    role = role_for_user(connection, user_id)
    if role not in DOCUMENT_INGEST_ROLES:
        raise PermissionError(f"User {user_id!r} with role {role!r} may not ingest documents")

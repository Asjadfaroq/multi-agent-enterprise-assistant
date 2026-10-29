from typing import Literal, cast

import psycopg

from assistant.config import get_settings
from assistant.knowledge.chunking import Acl

Role = Literal["viewer", "agent", "engineer", "hr_manager", "manager"]

ALLOWED_ACLS_BY_ROLE: dict[Role, tuple[Acl, ...]] = {
    "viewer": ("public",),
    "agent": ("public",),
    "engineer": ("public", "engineering"),
    "hr_manager": ("public", "hr"),
    "manager": ("public", "hr", "engineering"),
}


def create_users_table(connection: psycopg.Connection) -> None:
    connection.execute("CREATE TABLE IF NOT EXISTS users (id text PRIMARY KEY, role text NOT NULL)")


def current_user_id() -> str:
    user_id = get_settings().assistant_user_id
    if not user_id:
        raise PermissionError("ASSISTANT_USER_ID is not set; the caller's identity is unknown")
    return user_id


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

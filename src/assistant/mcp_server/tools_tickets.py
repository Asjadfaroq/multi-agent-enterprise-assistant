from datetime import datetime
from typing import Literal

import psycopg
from mcp.server.fastmcp import Context
from psycopg.rows import class_row
from pydantic import BaseModel

from assistant.config import get_settings
from assistant.mcp_server.auth import (
    caller_user_id,
    require_ticket_write_permission,
    visible_ticket_creator_for_user,
)

TicketCategory = Literal["billing", "bug", "account", "it", "question"]
TicketPriority = Literal["P1", "P2", "P3"]
TicketStatus = Literal["open", "in_progress", "resolved", "closed"]

TICKET_COLUMNS = (
    "id, title, description, category, priority, status, created_by, created_at, updated_at"
)


class Ticket(BaseModel):
    id: int
    title: str
    description: str
    category: TicketCategory
    priority: TicketPriority
    status: TicketStatus
    created_by: str
    created_at: datetime
    updated_at: datetime


def create_tickets_table(connection: psycopg.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id bigserial PRIMARY KEY,
            title text NOT NULL,
            description text NOT NULL,
            category text NOT NULL,
            priority text NOT NULL,
            status text NOT NULL DEFAULT 'open',
            created_by text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


# A NULL creator means the caller may read every ticket (see visible_ticket_creator in auth.py).
VISIBLE_TO_CALLER = "(%(created_by)s::text IS NULL OR created_by = %(created_by)s)"


def fetch_tickets(
    connection: psycopg.Connection,
    status: TicketStatus | None,
    limit: int,
    created_by: str | None,
) -> list[Ticket]:
    with connection.cursor(row_factory=class_row(Ticket)) as cursor:
        cursor.execute(
            f"SELECT {TICKET_COLUMNS} FROM tickets "
            f"WHERE (%(status)s::text IS NULL OR status = %(status)s) AND {VISIBLE_TO_CALLER} "
            "ORDER BY updated_at DESC LIMIT %(limit)s",
            {"status": status, "limit": limit, "created_by": created_by},
        )
        return cursor.fetchall()


def fetch_ticket(connection: psycopg.Connection, ticket_id: int, created_by: str | None) -> Ticket:
    with connection.cursor(row_factory=class_row(Ticket)) as cursor:
        cursor.execute(
            f"SELECT {TICKET_COLUMNS} FROM tickets WHERE id = %(id)s AND {VISIBLE_TO_CALLER}",
            {"id": ticket_id, "created_by": created_by},
        )
        ticket = cursor.fetchone()
    # A ticket the caller may not read is reported exactly like a missing one.
    if ticket is None:
        raise LookupError(f"Ticket {ticket_id} does not exist")
    return ticket


def insert_ticket(
    connection: psycopg.Connection,
    title: str,
    description: str,
    category: TicketCategory,
    priority: TicketPriority,
    created_by: str,
) -> Ticket:
    with connection.cursor(row_factory=class_row(Ticket)) as cursor:
        cursor.execute(
            "INSERT INTO tickets (title, description, category, priority, created_by) "
            f"VALUES (%s, %s, %s, %s, %s) RETURNING {TICKET_COLUMNS}",
            (title, description, category, priority, created_by),
        )
        return cursor.fetchone()


def change_ticket(
    connection: psycopg.Connection,
    ticket_id: int,
    status: TicketStatus | None,
    priority: TicketPriority | None,
) -> Ticket:
    if status is None and priority is None:
        raise ValueError("update_ticket needs a new status, a new priority, or both")
    with connection.cursor(row_factory=class_row(Ticket)) as cursor:
        cursor.execute(
            "UPDATE tickets SET status = COALESCE(%s, status), "
            "priority = COALESCE(%s, priority), updated_at = now() "
            f"WHERE id = %s RETURNING {TICKET_COLUMNS}",
            (status, priority, ticket_id),
        )
        ticket = cursor.fetchone()
    if ticket is None:
        raise LookupError(f"Ticket {ticket_id} does not exist")
    return ticket


def list_tickets(context: Context, status: TicketStatus | None = None) -> list[Ticket]:
    """List the most recently updated tickets, optionally only those with the given status."""
    with psycopg.connect(get_settings().database_url) as connection:
        created_by = visible_ticket_creator_for_user(connection, caller_user_id(context))
        return fetch_tickets(connection, status, get_settings().ticket_list_limit, created_by)


def get_ticket(context: Context, ticket_id: int) -> Ticket:
    """Get one ticket by its numeric id."""
    with psycopg.connect(get_settings().database_url) as connection:
        created_by = visible_ticket_creator_for_user(connection, caller_user_id(context))
        return fetch_ticket(connection, ticket_id, created_by)


def create_ticket(
    context: Context,
    title: str,
    description: str,
    category: TicketCategory,
    priority: TicketPriority,
) -> Ticket:
    """Create a ticket. Requires human approval before it runs."""
    user_id = caller_user_id(context)
    with psycopg.connect(get_settings().database_url) as connection:
        require_ticket_write_permission(connection, user_id)
        return insert_ticket(connection, title, description, category, priority, user_id)


def update_ticket(
    context: Context,
    ticket_id: int,
    status: TicketStatus | None = None,
    priority: TicketPriority | None = None,
) -> Ticket:
    """Change a ticket's status and/or priority. Requires human approval before it runs."""
    with psycopg.connect(get_settings().database_url) as connection:
        require_ticket_write_permission(connection, caller_user_id(context))
        return change_ticket(connection, ticket_id, status, priority)

import psycopg

from assistant.config import get_settings
from assistant.knowledge.ingest import ingest_documents
from assistant.mcp_server.auth import Role, create_users_table
from assistant.mcp_server.tools_tickets import create_tickets_table

SAMPLE_USERS: dict[str, Role] = {
    "nora": "viewer",
    "khalid": "agent",
    "omar": "engineer",
    "huda": "hr_manager",
    "lina": "manager",
}

# (title, description, category, priority, status, created_by)
SAMPLE_TICKETS = [
    (
        "Invoices stuck in 'sending'",
        "Customer Al Noor Trading reports 12 invoices stuck in 'sending' since this morning.",
        "bug",
        "P2",
        "in_progress",
        "khalid",
    ),
    (
        "Refund for double charge",
        "Growth plan customer was charged twice for October; asks for a refund of 149 SAR.",
        "billing",
        "P3",
        "open",
        "khalid",
    ),
    (
        "Laptop will not boot",
        "New hire laptop shows a black screen after the BIOS logo.",
        "it",
        "P3",
        "open",
        "lina",
    ),
    (
        "SSO login loop for Enterprise customer",
        "Users at Rawabi Logistics are redirected back to the login page after SSO.",
        "account",
        "P1",
        "resolved",
        "lina",
    ),
]


def seed_users(connection: psycopg.Connection) -> None:
    create_users_table(connection)
    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO users (id, role) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET role = EXCLUDED.role",
            list(SAMPLE_USERS.items()),
        )


def seed_tickets(connection: psycopg.Connection) -> None:
    create_tickets_table(connection)
    # Only seed an empty table so re-running the seed never duplicates or overwrites tickets.
    if connection.execute("SELECT EXISTS (SELECT 1 FROM tickets)").fetchone()[0]:
        return
    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO tickets (title, description, category, priority, status, created_by) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            SAMPLE_TICKETS,
        )


def main() -> None:
    settings = get_settings()
    with psycopg.connect(settings.database_url) as connection:
        seed_users(connection)
        seed_tickets(connection)
    chunk_count = ingest_documents(settings.sample_docs_dir, settings.database_url)
    print(f"Seeded {len(SAMPLE_USERS)} users, sample tickets and {chunk_count} chunks")


if __name__ == "__main__":
    main()

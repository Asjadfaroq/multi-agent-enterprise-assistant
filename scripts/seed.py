import psycopg

from assistant.config import get_settings
from assistant.knowledge.ingest import ingest_documents
from assistant.mcp_server.auth import Role, create_users_table

SAMPLE_USERS: dict[str, Role] = {
    "nora": "viewer",
    "khalid": "agent",
    "omar": "engineer",
    "huda": "hr_manager",
    "lina": "manager",
}


def seed_users(connection: psycopg.Connection) -> None:
    create_users_table(connection)
    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO users (id, role) VALUES (%s, %s) "
            "ON CONFLICT (id) DO UPDATE SET role = EXCLUDED.role",
            list(SAMPLE_USERS.items()),
        )


def main() -> None:
    settings = get_settings()
    with psycopg.connect(settings.database_url) as connection:
        seed_users(connection)
    chunk_count = ingest_documents(settings.sample_docs_dir, settings.database_url)
    print(f"Seeded {len(SAMPLE_USERS)} users and {chunk_count} chunks")


if __name__ == "__main__":
    main()

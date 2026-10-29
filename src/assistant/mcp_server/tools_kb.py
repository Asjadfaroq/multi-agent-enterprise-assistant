from typing import Annotated

import psycopg
from mcp.server.fastmcp import Context
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field

from assistant.config import get_settings
from assistant.knowledge.chunking import Acl
from assistant.llm import FloatArray, embed_query
from assistant.mcp_server.auth import allowed_acls_for_user, caller_user_id


class SearchHit(BaseModel):
    chunk_id: int
    doc_id: str
    acl: Acl
    title: str
    content: str
    score: float


def search_chunks(
    connection: psycopg.Connection, query_embedding: FloatArray, allowed_acls: list[Acl], k: int
) -> list[SearchHit]:
    # Without iterative scans, HNSW picks candidates before the ACL filter and can return < k rows.
    connection.execute("SET LOCAL hnsw.iterative_scan = strict_order")
    rows = connection.execute(
        """
        SELECT id, doc_id, acl, metadata->>'title', content, 1 - (embedding <=> %(query)s)
        FROM chunks
        WHERE acl = ANY(%(allowed_acls)s)
        ORDER BY embedding <=> %(query)s
        LIMIT %(k)s
        """,
        {"query": query_embedding, "allowed_acls": allowed_acls, "k": k},
    ).fetchall()
    return [
        SearchHit(
            chunk_id=chunk_id, doc_id=doc_id, acl=acl, title=title, content=content, score=score
        )
        for chunk_id, doc_id, acl, title, content, score in rows
    ]


def search_documents(
    context: Context,
    query: str,
    k: Annotated[int, Field(ge=1)] = get_settings().search_result_count,
) -> list[SearchHit]:
    """Search the company documents the current user may read and return the best excerpts."""
    query_embedding = embed_query(query)
    with psycopg.connect(get_settings().database_url) as connection:
        register_vector(connection)
        allowed_acls = allowed_acls_for_user(connection, caller_user_id(context))
        return search_chunks(connection, query_embedding, allowed_acls, k)

from typing import Annotated

import psycopg
from mcp.server.fastmcp import Context
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field

from assistant.config import get_settings
from assistant.knowledge.chunking import Acl
from assistant.llm import FloatArray, embed_query, rerank_scores
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


def keep_best_ranked(
    hits: list[SearchHit], relevance_scores: list[float], k: int
) -> list[SearchHit]:
    ranked = sorted(
        zip(hits, relevance_scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    return [hit.model_copy(update={"score": score}) for hit, score in ranked[:k]]


def search_documents(
    context: Context,
    query: str,
    k: Annotated[int, Field(ge=1)] = get_settings().search_result_count,
) -> list[SearchHit]:
    """Search the company documents the current user may read and return the best excerpts."""
    candidate_count = max(k, get_settings().rerank_candidate_count)
    with psycopg.connect(get_settings().database_url) as connection:
        register_vector(connection)
        allowed_acls = allowed_acls_for_user(connection, caller_user_id(context))
        candidates = search_chunks(connection, embed_query(query), allowed_acls, candidate_count)
    # The cross-encoder reads question and passage together, which ranks far better than vector
    # similarity alone on a small corpus; its score replaces the cosine score.
    scores = rerank_scores(query, [candidate.content for candidate in candidates])
    return keep_best_ranked(candidates, scores, k)

from pathlib import Path

import frontmatter
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb
from pydantic import ValidationError

from assistant.config import get_settings
from assistant.knowledge.chunking import Chunk, Document, chunk_document
from assistant.llm import Embeddings, count_tokens, embed_texts


def load_documents(docs_dir: Path) -> list[Document]:
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown documents found in {docs_dir}")
    return [load_document(path) for path in paths]


def load_document(path: Path) -> Document:
    post = frontmatter.load(path)
    try:
        return Document(
            doc_id=path.stem,
            acl=post.get("acl"),
            title=post.get("title"),
            source=path.name,
            body=post.content,
        )
    except ValidationError as error:
        raise ValueError(f"Invalid front-matter in {path}: {error}") from error


def create_chunks_table(connection: psycopg.Connection, dimensions: int) -> None:
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id bigserial PRIMARY KEY,
                doc_id text NOT NULL,
                acl text NOT NULL,
                content text NOT NULL,
                embedding vector({dimensions}) NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}'
            )
            """
        ).format(dimensions=sql.Literal(dimensions))
    )
    connection.execute("CREATE INDEX IF NOT EXISTS chunks_acl_idx ON chunks (acl)")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def replace_document_chunks(
    connection: psycopg.Connection, doc_id: str, chunks: list[Chunk], embeddings: Embeddings
) -> None:
    connection.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
    with connection.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO chunks (doc_id, acl, content, embedding, metadata) "
            "VALUES (%s, %s, %s, %s, %s)",
            [
                (chunk.doc_id, chunk.acl, chunk.content, embedding, Jsonb(chunk.metadata))
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )


def ingest_documents(docs_dir: Path, database_url: str) -> int:
    settings = get_settings()
    documents = load_documents(docs_dir)
    chunk_count = 0
    with psycopg.connect(database_url) as connection:
        create_chunks_table(connection, settings.embedding_dimensions)
        register_vector(connection)
        for document in documents:
            chunks = chunk_document(document, settings.chunk_size_tokens, count_tokens)
            embeddings = embed_texts([chunk.content for chunk in chunks])
            replace_document_chunks(connection, document.doc_id, chunks, embeddings)
            chunk_count += len(chunks)
    return chunk_count


def main() -> None:
    settings = get_settings()
    chunk_count = ingest_documents(settings.sample_docs_dir, settings.database_url)
    print(f"Ingested {chunk_count} chunks from {settings.sample_docs_dir}")


if __name__ == "__main__":
    main()

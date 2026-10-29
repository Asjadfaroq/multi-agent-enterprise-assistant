from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict

Acl = Literal["public", "hr", "engineering"]
TokenCounter = Callable[[str], int]

# Coarsest first: prefer cutting between paragraphs, then lines, then sentences, then words.
SEPARATORS = ("\n\n", "\n", ". ", " ")


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    acl: Acl
    title: str
    source: str
    body: str


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    acl: Acl
    content: str
    metadata: dict[str, str | int]


def chunk_document(document: Document, max_tokens: int, count_tokens: TokenCounter) -> list[Chunk]:
    fragments = split_into_fragments(document.body, max_tokens, count_tokens)
    contents = merge_fragments(fragments, max_tokens, count_tokens)
    return [
        Chunk(
            doc_id=document.doc_id,
            acl=document.acl,
            content=content,
            metadata={"title": document.title, "source": document.source, "chunk_index": index},
        )
        for index, content in enumerate(contents)
    ]


def split_into_fragments(
    text: str,
    max_tokens: int,
    count_tokens: TokenCounter,
    separators: tuple[str, ...] = SEPARATORS,
) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    if not separators:
        raise ValueError(f"Cannot split text below {max_tokens} tokens: {text[:80]!r}")
    separator, *finer_separators = separators
    pieces = text.split(separator)
    # Keep the separator on each piece so merging fragments back reproduces the original text.
    pieces_with_separator = [piece + separator for piece in pieces[:-1]] + [pieces[-1]]
    return [
        fragment
        for piece in pieces_with_separator
        if piece
        for fragment in split_into_fragments(
            piece, max_tokens, count_tokens, tuple(finer_separators)
        )
    ]


def merge_fragments(fragments: list[str], max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    chunks: list[str] = []
    current = ""
    for fragment in fragments:
        if current and count_tokens(current + fragment) > max_tokens:
            chunks.append(current)
            current = ""
        current += fragment
    chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

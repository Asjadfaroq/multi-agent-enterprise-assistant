from pathlib import Path

import pytest

from assistant.config import get_settings
from assistant.knowledge.chunking import Document, chunk_document, split_into_fragments
from assistant.knowledge.ingest import load_document, load_documents


def count_words(text: str) -> int:
    return len(text.split())


def make_document(body: str, acl: str = "public") -> Document:
    return Document(doc_id="handbook", acl=acl, title="Handbook", source="handbook.md", body=body)


def paragraph(word: str, word_count: int) -> str:
    return " ".join([word] * word_count)


def test_document_under_the_limit_becomes_a_single_chunk() -> None:
    chunks = chunk_document(make_document("Short policy text."), 50, count_words)

    assert [chunk.content for chunk in chunks] == ["Short policy text."]


def test_every_chunk_stays_within_the_token_limit() -> None:
    body = "\n\n".join(paragraph(f"word{index}", 30) for index in range(10))

    chunks = chunk_document(make_document(body), 100, count_words)

    assert len(chunks) > 1
    assert all(count_words(chunk.content) <= 100 for chunk in chunks)


def test_chunks_break_between_paragraphs_when_paragraphs_fit() -> None:
    first, second, third = paragraph("alpha", 40), paragraph("beta", 40), paragraph("gamma", 40)

    chunks = chunk_document(make_document(f"{first}\n\n{second}\n\n{third}"), 90, count_words)

    assert [chunk.content for chunk in chunks] == [f"{first}\n\n{second}", third]


def test_oversized_paragraph_is_split_at_sentence_boundaries() -> None:
    sentences = [f"{paragraph(f'clause{index}', 9)} end." for index in range(6)]

    chunks = chunk_document(make_document(" ".join(sentences)), 25, count_words)

    assert all(chunk.content.endswith("end.") for chunk in chunks)
    assert all(count_words(chunk.content) <= 25 for chunk in chunks)


def test_chunking_keeps_every_word_in_order() -> None:
    body = "\n\n".join(paragraph(f"word{index}", 17) for index in range(9))

    chunks = chunk_document(make_document(body), 40, count_words)

    rejoined_words = " ".join(chunk.content for chunk in chunks).split()
    assert rejoined_words == body.split()


def test_chunks_inherit_acl_and_carry_source_metadata() -> None:
    body = "\n\n".join(paragraph(f"word{index}", 30) for index in range(3))

    chunks = chunk_document(make_document(body, acl="hr"), 40, count_words)

    assert all(chunk.acl == "hr" and chunk.doc_id == "handbook" for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["source"] == "handbook.md" for chunk in chunks)


def test_text_that_cannot_be_split_below_the_limit_raises() -> None:
    with pytest.raises(ValueError, match="Cannot split text below 5 tokens"):
        split_into_fragments("unbreakable", 5, len)


def test_document_with_unknown_acl_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_document("Body", acl="finance")


def test_sample_docs_cover_every_acl() -> None:
    documents = load_documents(get_settings().sample_docs_dir)

    assert {document.acl for document in documents} == {"public", "hr", "engineering"}


def test_doc_without_acl_front_matter_is_rejected_with_its_path(tmp_path: Path) -> None:
    doc_path = tmp_path / "untagged.md"
    doc_path.write_text("---\ntitle: Untagged\n---\nBody text.\n")

    with pytest.raises(ValueError, match="Invalid front-matter in .*untagged.md"):
        load_document(doc_path)

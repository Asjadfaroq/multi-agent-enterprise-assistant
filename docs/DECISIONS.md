# Decisions

Deviations from `BUILD_SPEC.md`, one line each.

- LLM provider is Anthropic via `langchain-anthropic` instead of OpenAI: agent model `claude-haiku-5-5`, judge model `claude-sonnet-5-5`, both set in `config.py` / `.env`.
- Embeddings are local `sentence-transformers` `BAAI/bge-small-en-v1.5` (384 dims) instead of OpenAI, so `chunks.embedding` is `vector(384)` and ingest needs no API key.
- Embedding helpers (`embed_texts`, `count_tokens`) live in `llm.py`, the spec's provider layer, because the tree has no embeddings module and both ingest and search need them.
- Chunks are measured in tokens from the bge tokenizer and capped at 500, under bge's 512-token input limit; no overlap, since chunks break on paragraph boundaries.
- `torch` is installed from the PyTorch CPU wheel index; the default wheel pulls several GB of CUDA libraries that local embeddings do not need.
- `python-frontmatter` parses the `acl`/`title` front-matter instead of hand-rolled YAML splitting.
- Langfuse runs as server v4 (current image) under a `langfuse` compose profile, so `docker compose up -d` starts only Postgres.

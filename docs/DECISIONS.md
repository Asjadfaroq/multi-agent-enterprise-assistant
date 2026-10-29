# Decisions

Deviations from `BUILD_SPEC.md`, one line each.

- LLM provider is Anthropic via `langchain-anthropic` instead of OpenAI: agent model `claude-haiku-5-5`, judge model `claude-sonnet-5-5`, both set in `config.py` / `.env`.
- Embeddings are local `sentence-transformers` `BAAI/bge-small-en-v1.5` (384 dims) instead of OpenAI, so `chunks.embedding` is `vector(384)` and ingest needs no API key.
- Embedding helpers (`embed_texts`, `count_tokens`) live in `llm.py`, the spec's provider layer, because the tree has no embeddings module and both ingest and search need them.
- Chunks are measured in tokens from the bge tokenizer and capped at 500, under bge's 512-token input limit; no overlap, since chunks break on paragraph boundaries.
- `torch` is installed from the PyTorch CPU wheel index; the default wheel pulls several GB of CUDA libraries that local embeddings do not need.
- `python-frontmatter` parses the `acl`/`title` front-matter instead of hand-rolled YAML splitting.
- Langfuse runs as server v4 (current image) under a `langfuse` compose profile, so `docker compose up -d` starts only Postgres.
- Postgres is published on host port 5433 because 5432 is already taken by another local Postgres.
- The MCP server runs over stdio, which has no headers, so the caller's identity comes from `ASSISTANT_USER_ID` in the server's environment instead of an `X-User` header; the retriever starts the server per user.
- Roles live in a `users` table (`auth.py` creates it, `scripts/seed.py` fills it), pulled forward from the spec's seed step because role lookup needs it now.
- `Plan`, `Draft` and `Review` live in `guardrails/schema.py` with the shared JSON parsing; the executor returns `Draft(text, citations)` so every node output is validated the same way.
- ACL-filtered search sets `hnsw.iterative_scan = strict_order` (pgvector 0.8) so the filter cannot make HNSW return fewer than k permitted chunks.
- One MCP server session per API request (`open_assistant_context`), not per search: a search took 8.4–9.7 s with a fresh server each time; now the session opens once in 5.5–6.2 s, then the first search takes ~0.5 s and later ones ~0.04 s. A cross-request pool was skipped because stdio sessions are bound to the task that opened them.
- The MCP server is started with `HF_HUB_OFFLINE=1` once the embedding model is in the local Hugging Face cache, so server start makes no Hub requests.
- MCP tools reach the nodes through LangGraph's runtime `context` (`AssistantContext`), so one compiled graph serves every user and tests inject fake tools the same way.
- Client-side session code lives in `graph/build.py` because the spec tree has no client module.
- Write tools pause with `interrupt()` inside a `request_approval` node rather than `interrupt_before`, so read-only tool calls never pause and the pause carries the proposed action.
- Write tools are registered on the MCP server only for `agent`/`manager`, and `create_ticket`/`update_ticket` re-check the role in `auth.py` on every call.
- A rejected write ends the turn with a fixed "nothing was changed" answer instead of asking the model again, which re-proposed the same write in a live run.
- The turn's message history is append-only (reviewer feedback is a `name="reviewer"` message): Claude signs thinking blocks against earlier messages, and rewriting the executor prompt on revision made the API return 400.
- The reviewer sees the user's available tools, so true statements like "I cannot create tickets" are not rejected as ungrounded.
- The checkpointer is `AsyncPostgresSaver` (the graph is async), with `Plan`, `Draft`, `Review` and `SearchHit` explicitly allowed in its serializer.
- `/approve` requires the approver to have ticket-write permission; the approved tools still run as the user who asked.

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
- The MCP server is one long-lived streamable-HTTP process (`uv run assistant-mcp-server`); the API passes the caller in an `X-User` header and `auth.py` resolves the role from the `users` table on every call. The header is trusted only because the server binds to 127.0.0.1 and only the API calls it; exposing it on a network needs a service credential first.
- Roles live in a `users` table (`auth.py` creates it, `scripts/seed.py` fills it), pulled forward from the spec's seed step because role lookup needs it now.
- `Plan`, `Draft` and `Review` live in `guardrails/schema.py` with the shared JSON parsing; the executor returns `Draft(text, citations)` so every node output is validated the same way.
- ACL-filtered search sets `hnsw.iterative_scan = strict_order` (pgvector 0.8) so the filter cannot make HNSW return fewer than k permitted chunks.
- The per-request stdio server spawn was replaced by the HTTP server: `/chat` for the same viewer question took 14.7–15.2 s with stdio and 4.4–5.4 s over HTTP, even though each turn now also makes the memory LLM call. (Earlier, one stdio session per request had cut a search from 8.4–9.7 s to ~0.5 s once open, but each request still paid ~6 s to start the server.)
- The embedding model loads with `local_files_only=True` once it is in the Hugging Face cache, so no process contacts the Hub; the MCP server loads it once at startup.
- MCP tools reach the nodes through LangGraph's runtime `context` (`AssistantContext`), so one compiled graph serves every user and tests inject fake tools the same way.
- Client-side session code lives in `graph/build.py` because the spec tree has no client module.
- Write tools pause with `interrupt()` inside a `request_approval` node rather than `interrupt_before`, so read-only tool calls never pause and the pause carries the proposed action.
- With one server for every caller, `list_tools` filters write tools by the caller's role, so viewers are never offered them; `create_ticket`/`update_ticket` still re-check the role in `auth.py` on every call.
- A rejected write ends the turn with a fixed "nothing was changed" answer instead of asking the model again, which re-proposed the same write in a live run.
- The turn's message history is append-only (reviewer feedback is a `name="reviewer"` message): Claude signs thinking blocks against earlier messages, and rewriting the executor prompt on revision made the API return 400.
- The reviewer sees the user's available tools, so true statements like "I cannot create tickets" are not rejected as ungrounded.
- The checkpointer is `AsyncPostgresSaver` (the graph is async), with the checkpointed Pydantic models (`Plan`, `Draft`, `Review`, `SearchHit`, `ProfileFact`) explicitly allowed in its serializer.
- `/approve` requires the approver to have ticket-write permission; the approved tools still run as the user who asked.
- `POST /ingest` is limited to the `manager` role (`require_ingest_permission` in `auth.py`); the caller comes from the same `X-User` header as the other endpoints.
- `/approve` takes a Postgres advisory lock on the thread id (`pg_try_advisory_lock(hashtextextended(thread_id, 0))`) and re-reads the thread under it: a concurrent decision gets 409 at once instead of waiting, and a decision on an already-resolved thread gets 409. A Postgres lock is used so it also holds across several API workers.
- Only one approved write runs per turn: a live run showed the model re-proposing the ticket it had just created, so further writes in that turn are refused without running or pausing.
- Long-term memory stores at most `MAX_PROFILE_FACTS_PER_TURN` facts (`preferred_language`, `team`) per turn via a `remember` node after the reviewer; the planner and executor both read them. The node lives in `memory/profile.py` and the fact models in `guardrails/schema.py`, because the spec tree has no memory node.
- Tracing uses the Langfuse v5 SDK (OpenTelemetry). User, session (thread id) and a `role:` tag are set with `propagate_attributes` around each graph run, because attributes set from inside a LangChain callback do not reach LangGraph's async node tasks. Each node span also carries `user`, `role`, `thread_id` and LangGraph's `langgraph_node` metadata.
- Langfuse exports only its own spans (`should_export_span=is_langfuse_span`): the Anthropic SDK emits an OpenTelemetry span per request, which Langfuse's default filter keeps, doubling every call's tokens and cost.
- Langfuse server v4 runs in events-only mode, so the legacy `GET /api/public/traces` endpoint is unavailable; traces were checked through `GET /api/public/v2/observations`.
- `scripts/demo.py` drives the HTTP API (not the graph directly), so a demo run exercises the same auth, approval and tracing path as real use.

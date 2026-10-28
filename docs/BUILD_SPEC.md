# Multi-Agent Enterprise Assistant — build spec

Goal: build exactly what the CV bullets say, nothing more. Every heading below maps to a phrase in the CV so the repo is the proof.

CV bullets this repo must back:
1. "LangGraph multi-agent system (planner, retriever, executor, reviewer) with tool calling through an MCP server exposing internal APIs and a pgvector knowledge base, enforcing the end user's permissions before data reaches the model, with memory, checkpoints, and human-in-the-loop approval for write actions."
2. "Evaluation pipeline as a CI gate (golden dataset, LLM-as-judge, Ragas faithfulness / answer relevancy / context precision, regression tests), Langfuse tracing with token, cost, and latency dashboards, and guardrails for prompt injection, PII masking, and output schema validation."

Scenario (pick a simple one and stick to it): an internal assistant for a small company. It answers questions from company documents (policies, product docs) and can act on an internal "tickets" API (list, create, update). Users have roles; some documents and actions are restricted. This gives you read + write tools, permissions, and a reason for human approval.

---

## 1. Stack (all free except LLM credits)

| Piece | Choice | Why |
|---|---|---|
| Language | Python 3.12, `uv` for deps | Fast, matches your Vision Guard monorepo style |
| Agent framework | LangGraph (langgraph, langchain-core) | Most-requested framework in Saudi JDs |
| LLM | OpenAI `gpt-4o-mini` for agents, `gpt-4o` for LLM-as-judge | Cheap to run, strong judge. Swap via env var; keep a `provider` layer so Anthropic/Gemini work too |
| Embeddings | OpenAI `text-embedding-3-small` (or `BAAI/bge-small-en-v1.5` local) | Cheap |
| Vector store | PostgreSQL 16 + pgvector (Docker) | On the CV; same DB holds tickets, users, checkpoints |
| Checkpoints / memory | `langgraph-checkpoint-postgres` | Thread memory and resumable HITL pauses |
| MCP | `mcp` Python SDK (FastMCP server) + LangChain MCP adapters on the client side | Server exposes tools; agent consumes them |
| Tracing | Langfuse (self-host via Docker Compose, or free cloud) | Token/cost/latency dashboards out of the box |
| Evals | Ragas + custom LLM-as-judge + pytest | CI gate |
| Guardrails | Small own modules + `presidio-analyzer` for PII + Pydantic for schemas | Keep it readable; don't over-engineer |
| API | FastAPI (thin: `/chat`, `/approve`, `/ingest`) | Interview demo |
| CI | GitHub Actions | Runs unit tests + eval gate |

---

## 2. Repo structure

```
multi-agent-enterprise-assistant/
├── README.md                      # architecture diagram, eval table, Langfuse screenshot, how to run
├── pyproject.toml                 # uv project
├── docker-compose.yml             # postgres+pgvector, langfuse (optional)
├── .env.example
├── .github/workflows/ci.yml       # lint + unit tests + eval gate
│
├── src/assistant/
│   ├── config.py                  # settings from env (models, thresholds)
│   ├── llm.py                     # provider layer: get_chat_model(role="agent"|"judge")
│   │
│   ├── graph/
│   │   ├── state.py               # AgentState TypedDict (messages, plan, retrieved, draft, review, user)
│   │   ├── nodes/
│   │   │   ├── planner.py         # decides steps + which tools are needed
│   │   │   ├── retriever.py       # calls MCP search tool, filters by permissions
│   │   │   ├── executor.py        # runs tool calls (read freely, write → interrupt)
│   │   │   └── reviewer.py        # checks draft vs retrieved context, asks for retry if ungrounded
│   │   ├── edges.py               # routing logic
│   │   └── build.py               # compile graph with Postgres checkpointer + interrupt_before executor writes
│   │
│   ├── mcp_server/
│   │   ├── server.py              # FastMCP app: tools below
│   │   ├── tools_kb.py            # search_documents(query, k) → pgvector, filtered by caller role
│   │   ├── tools_tickets.py       # list_tickets, get_ticket, create_ticket, update_ticket
│   │   └── auth.py                # caller identity + role → allowed docs/actions (enforced here, not in prompt)
│   │
│   ├── knowledge/
│   │   ├── ingest.py              # load docs → chunk → embed → insert (with acl tag per chunk)
│   │   └── chunking.py            # simple recursive chunking, metadata
│   │
│   ├── guardrails/
│   │   ├── injection.py           # heuristic + small LLM check on user input and retrieved chunks
│   │   ├── pii.py                 # presidio: mask emails/phones/IDs before they reach the model and logs
│   │   └── schema.py              # Pydantic output models; retry once on validation failure
│   │
│   ├── memory/
│   │   └── profile.py             # per-user long-term memory (small table: preferences, past topics)
│   │
│   ├── observability/
│   │   └── tracing.py             # Langfuse callback handler, tags: user, thread, node, model, cost
│   │
│   └── api/
│       └── app.py                 # FastAPI: POST /chat, POST /approve/{thread_id}, POST /ingest
│
├── evals/
│   ├── golden/
│   │   └── golden_set.jsonl       # 40–60 items: question, expected_answer, expected_sources, must_refuse?
│   ├── run_ragas.py               # faithfulness, answer_relevancy, context_precision
│   ├── judge.py                   # LLM-as-judge rubric (correctness, grounding, safety) 1–5
│   ├── regression_test.py         # pytest: scores >= thresholds, refusal cases refuse, injection cases blocked
│   └── results/                   # committed CSV/JSON of last run (README links to it)
│
├── tests/
│   ├── test_permissions.py        # viewer can't see HR docs; viewer can't create tickets
│   ├── test_guardrails.py         # injection strings blocked, PII masked, schema retry works
│   ├── test_graph_routing.py      # planner → retriever → executor → reviewer paths
│   └── test_hitl.py               # write tool pauses, /approve resumes, reject cancels
│
├── data/sample_docs/              # 10–15 small markdown docs (public, hr, engineering) with acl tags
└── scripts/
    ├── seed.py                    # users, roles, tickets, ingest sample docs
    └── demo.py                    # CLI demo for a 2-minute recording
```

---

## 3. Build order (3 focused days)

### Day 1 — working agent over the knowledge base

1. `uv init`, add deps: `langgraph langchain-core langchain-openai langgraph-checkpoint-postgres psycopg[binary] pgvector sqlalchemy fastapi uvicorn mcp langchain-mcp-adapters langfuse ragas presidio-analyzer pydantic pytest ruff`.
2. `docker-compose.yml`: `pgvector/pgvector:pg16` + Langfuse (copy their compose). `.env.example` with `OPENAI_API_KEY`, `DATABASE_URL`, `LANGFUSE_*`, `AGENT_MODEL`, `JUDGE_MODEL`.
3. `knowledge/ingest.py`: read `data/sample_docs/*.md`, front-matter `acl: public|hr|engineering`, chunk ~500 tokens, embed, insert into `chunks(id, doc_id, acl, content, embedding vector(1536), metadata jsonb)`.
4. `mcp_server/server.py` with FastMCP: `search_documents(query, k=5)` only. `auth.py`: read `X-User` header → role; SQL `WHERE acl = ANY(allowed_acls)`. This is the "permissions enforced before data reaches the model" line: the model never sees a chunk it isn't allowed to see, because the filter is in the tool, not in the prompt.
5. `graph/`: state + 4 nodes + edges. Keep nodes small:
   - planner: one LLM call, returns `Plan(steps: list[str], needs_tools: list[str])` (Pydantic).
   - retriever: calls MCP `search_documents`, stores chunks in state.
   - executor: for now, drafts the answer with citations `[doc_id]`.
   - reviewer: LLM call "is every claim supported by the retrieved chunks? yes/no + reason"; if no, loop once to executor with the reason, then give up gracefully.
6. `scripts/demo.py`: ask 3 questions as `viewer` and `hr_manager`, show different answers. Commit.

### Day 2 — tools, write actions, HITL, memory, tracing

1. Add ticket tools to the MCP server: `list_tickets`, `get_ticket` (read), `create_ticket`, `update_ticket` (write). Role check in `auth.py`: only `agent`/`manager` can write.
2. Executor now does real tool calling (bind MCP tools via `langchain-mcp-adapters`). Mark write tools; compile graph with `interrupt_before=["execute_write"]` (or a dedicated node). `POST /chat` returns `{"status": "awaiting_approval", "thread_id", "proposed_action"}`; `POST /approve/{thread_id}` resumes with `Command(resume=True/False)`.
3. Checkpointer: `PostgresSaver` so a thread survives restarts and approval can come later. That is "memory, checkpoints, human-in-the-loop".
4. `memory/profile.py`: after each turn store 1–2 facts (preferred language, team) in `user_memory`; planner reads them. Small, but real long-term memory.
5. `observability/tracing.py`: Langfuse `CallbackHandler` on every node, `metadata={"user","role","thread_id","node"}`. Open Langfuse: traces, tokens, cost per trace, latency per node. Take a screenshot for the README.
6. Tests: `test_permissions.py`, `test_hitl.py`. Commit.

### Day 3 — guardrails, evals, CI gate, README

1. `guardrails/injection.py`: (a) regex/heuristics for "ignore previous instructions", role-play jailbreaks, encoded payloads; (b) one cheap LLM classification call; run on user input AND on retrieved chunks (retrieved content is untrusted). Blocked → polite refusal, trace tagged `blocked_injection`.
2. `guardrails/pii.py`: presidio mask emails/phones/national IDs in user input before model and in tool outputs before logging.
3. `guardrails/schema.py`: final answer is `Answer(text, citations: list[str], confidence)`; on validation failure retry once with the error, then fail closed.
4. `evals/golden/golden_set.jsonl`: 40–60 rows across types: factual (with expected sources), multi-hop, permission-restricted (must refuse for viewer), write-action (must pause), injection (must block), out-of-scope (must say not found). Write them by hand; this is what interviewers ask about.
5. `evals/run_ragas.py`: run the graph on factual rows, compute `faithfulness`, `answer_relevancy`, `context_precision` with `gpt-4o` as judge. `evals/judge.py`: rubric 1–5 for correctness/grounding/safety on every row.
6. `evals/regression_test.py` (pytest): assert `faithfulness >= 0.85`, `context_precision >= 0.80`, `judge_mean >= 4.0`, 100% of refusal/injection/HITL cases behave. Save `evals/results/latest.json`.
7. `.github/workflows/ci.yml`: ruff + unit tests on every push; eval job runs on PRs to `main` (needs `OPENAI_API_KEY` secret) and fails the PR if thresholds drop. That is the "CI gate".
8. README: one architecture diagram (Mermaid), the eval table with real numbers, Langfuse screenshot, 2-minute demo GIF, "permissions model" section, "what I'd do next". Keep it honest; numbers come from `evals/results/latest.json`.

---

## 4. Design decisions to be ready to explain

- Permissions live in the MCP server/DB filter, never in the prompt. Prompt-level "don't show HR docs" is not security.
- Retrieved chunks are treated as untrusted input: injection check runs on them too.
- Reviewer node is the cheap version of "grounding": claims vs chunks. Ragas is the offline measurement of the same thing.
- Write actions always pause. Approval is a separate HTTP call, resumable from the Postgres checkpoint, so the approver can be a different person/time.
- Judge model ≠ agent model (reduces self-grading bias). Golden set has adversarial and refusal cases, not only happy paths.
- Cost: Langfuse shows cost per trace; planner uses the small model, judge uses the big one.

## 5. Scope guardrails for yourself

- Don't add a UI. The CLI demo + FastAPI is enough. (If you later want a chat UI, Next.js, 1 day, separate.)
- Don't add 10 tools. 1 search tool + 4 ticket tools is enough to show read/write/permissions.
- Don't fine-tune anything here.
- Stop when every CV phrase has a file behind it and the README shows real eval numbers.

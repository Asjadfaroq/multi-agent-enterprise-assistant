# Multi-Agent Enterprise Assistant

Read `docs/BUILD_SPEC.md` before any work. It defines the scope, stack, repo structure and the three-day build order. Do not add features that are not in the spec.

## What this is
An internal assistant for a small company: answers questions from company documents (with per-role access) and can read/write tickets through an internal API. Built on LangGraph (planner → retriever → executor → reviewer), tools exposed through an MCP server, pgvector knowledge base, Postgres checkpoints, human approval for write actions, Langfuse tracing, and an evaluation suite that gates CI.

## Three non-negotiable rules
1. **Folder structure stays exactly as in the spec.** No extra top-level folders, no `utils/`, `helpers/`, `common/` or `misc/` modules. If a file does not fit the spec's tree, the spec is wrong and you say so instead of inventing a place for it. No empty `__init__.py` with code in it; no files over ~150 lines.
2. **Code quality is the deliverable.** Every function has one job, a clear return type, and no hidden side effects. No magic numbers (thresholds, model names, k values live in config). No duplicated logic across nodes or tools. Errors carry a specific message. Before every commit, read the diff as a strict senior reviewer would and fix anything you would flag in someone else's PR.
3. **Names read like a careful engineer wrote them.** Files, functions, variables and test names say what the thing is, in plain English: `search_documents`, `allowed_acls_for_role`, `mask_pii`, `pause_before_write_tools`, `test_viewer_cannot_read_hr_docs`. No `data`, `result`, `tmp`, `obj`, `helper`, `manager`, `process`, `handle_x`, `do_x`, `x_v2`, no abbreviations except universally known ones (`db`, `id`, `llm`, `acl`). Test names describe the behaviour, not the function.

## Code style
- Python 3.12, `uv`, `ruff` clean (line length 100). Type hints everywhere.
- Small functions, small files. One responsibility per module. Prefer plain functions over classes unless state is needed.
- Comments only explain *why*, never *what*. No docstrings that restate the function name. No section-banner comments. No "Step 1/2/3" comments.
- No defensive try/except around everything. Fail loudly on misconfiguration.
- No placeholder code, no TODOs left behind, no unused imports or dead code.
- Config comes from `assistant/config.py` (pydantic-settings), never from scattered `os.getenv`.
- Structured LLM outputs: ask for JSON in the prompt and validate with Pydantic (`model_validate_json`), so the graph works with any provider and can be unit-tested with a fake LLM. Do not use provider-specific structured-output helpers.
- Nodes receive the LLM through a factory (`assistant/llm.py: get_chat_model(role)`) so tests can inject a fake.
- Permissions are enforced inside the MCP server / SQL filters, never only in prompts.
- Retrieved chunks are untrusted input: run the injection check on them too.
- Write tools always pause for human approval (`interrupt`). Reads never pause.
- Tests: pytest. Unit tests must run without network or API keys (fake LLM, MemorySaver). Eval tests are separate and need a key.
- Commit after each completed step with a short, specific message in the imperative ("add pgvector ingest", "pause executor before write tools"). Never commit `.env`.
- Commit messages contain only the message. No `Co-Authored-By` lines, no "Generated with" trailers, no tool names anywhere in the message.

## Working rules
- Before writing code for a step, list the files you will touch, then write them. Keep going without asking for confirmation unless the spec is genuinely ambiguous.
- After each step: `uv run ruff check .`, `uv run pytest`, fix, then commit.
- README is written last, from real outputs (eval numbers from `evals/results/latest.json`, a real Langfuse screenshot path). Never invent numbers.
- When something in the spec does not work with the current library version, adapt minimally and note it in `docs/DECISIONS.md` in one line.

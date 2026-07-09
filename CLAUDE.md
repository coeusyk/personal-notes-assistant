# personal-notes-assistant

A FastMCP server that indexes an Obsidian vault into Milvus and answers questions
over it via RAG (Ollama or OpenAI as the LLM backend).

## Architecture

- `main.py` — module-level `FastMCP("obsidian-rag")` instance. Env vars are read
  and validated at import time (fails fast if `OBSIDIAN_VAULT_PATH` is unset).
  The `lifespan` async context manager builds one `ObsidianRAGServer`, runs its
  `initialize()` (Milvus connect, embedding model load, file watcher start,
  initial full ingest), and yields it as `{"rag": rag_server}` — tools reach it
  via `ctx.lifespan_context["rag"]`.
- `mcp_handlers.py` — all MCP tools, `@mcp.tool`-decorated functions with type
  hints (FastMCP infers the JSON schema; don't hand-write one). `_rag(ctx)` is
  the one place that pulls the server out of the lifespan context.
- `rag_server.py` — all real logic: markdown/frontmatter parsing, chunking,
  Milvus CRUD, the watchdog file watcher, and the RAG query against the
  configured LLM client. This is where behavior changes belong; the tool
  layer should stay a thin pass-through.

## Adding a tool

Add a function to `register_tools()` in `mcp_handlers.py`:
```python
@mcp.tool
async def my_tool(ctx: Context, arg: str, limit: int = 5) -> str:
    """One-line description — shown to the LLM as the tool description."""
    return await _rag(ctx).some_method(arg, limit)
```
No manual `inputSchema`, no dispatch `if/elif` — FastMCP derives both name and
schema from the function signature and docstring. Keep the tool body a
one-line delegation into `ObsidianRAGServer`; put actual logic in `rag_server.py`.

## Milvus notes (non-obvious, bit people twice already)

- **`collection.insert()` must be followed by `collection.flush()`.** Without
  it, a note just ingested/updated is invisible to `search_notes` and
  `get_note_content` — this was a real bug, verified against a live instance.
- Any user-controlled string (file path, filename) interpolated into a Milvus
  boolean `expr` filter must go through `_escape_milvus_string()` first — a
  bare `"` in a filename otherwise breaks or manipulates the query.
- The file watcher's `on_deleted` handler matters: without it, deleting a note
  from the vault leaves its chunks in Milvus forever. `on_modified`,
  `on_created`, and `on_deleted` all point at the same `_handle` — don't
  reintroduce three copies of the same body.
- Using the ORM-style `pymilvus` API (`connections.connect`, `Collection`,
  `utility`) — it's deprecated upstream in favor of `MilvusClient`, but that's
  a separate migration, not folded into unrelated changes.

## Running / testing locally

```bash
docker-compose up -d          # Milvus (etcd + minio + standalone)
uv pip install -e .
uv run main.py                 # or: run-obsidian-rag
```

For a fast in-process smoke test without a real vault or LLM, use FastMCP's
in-memory `Client` against `main.mcp` (see `fastmcp` docs under
`clients/client.mdx`) — this exercises the full lifespan (Milvus connect,
embedding model, ingest) without spawning a subprocess. `LLM_PROVIDER`/
`LLM_MODEL` still need to be set for `ObsidianRAGServer.__init__` to construct
the client, but the LLM itself is only hit by `query_knowledge`.

## Conventions

- Logging goes to **stderr** (via `colorlog`) — never print to stdout, it
  shares the pipe with the stdio JSON-RPC transport.
- No CI, no test suite, no linter config in this repo currently — don't invent
  one unless asked.

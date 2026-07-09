from fastmcp import FastMCP, Context

from rag_server import ObsidianRAGServer


def _rag(ctx: Context) -> ObsidianRAGServer:
    return ctx.lifespan_context["rag"]


def register_tools(mcp: FastMCP):
    """Register all MCP tools with the server instance."""

    @mcp.tool
    async def ingest_notes(ctx: Context) -> str:
        """Ingest all notes from Obsidian vault into vector database"""
        await _rag(ctx).ingest_all_notes()
        return "Successfully ingested all notes"

    @mcp.tool
    async def search_notes(ctx: Context, query: str, top_k: int = 5) -> str:
        """Search for similar notes using vector similarity"""
        results = await _rag(ctx).search_similar_notes(query, top_k)

        if not results:
            return "No similar notes found."

        result_text = f"Found {len(results)} similar notes:\n\n"
        for i, note in enumerate(results, 1):
            result_text += f"{i}. {note['title']} (similarity: {note['similarity']:.3f})\n"
            result_text += f"   Path: {note['file_path']}\n"
            result_text += f"   Tags: {note['tags']}\n"
            result_text += f"   Content preview: {note['content'][:200]}...\n\n"

        return result_text

    @mcp.tool
    async def query_knowledge(ctx: Context, question: str, top_k: int = 3) -> str:
        """Query your knowledge base using RAG with a configured LLM"""
        return await _rag(ctx).query_with_rag(question, top_k)

    @mcp.tool
    async def update_note(ctx: Context, file_path: str) -> str:
        """Update a specific note in the vector database"""
        rag = _rag(ctx)
        full_path = rag.vault_path / file_path
        await rag._update_note(str(full_path))
        return f"Updated note: {file_path}"

    @mcp.tool
    async def list_all_notes(ctx: Context) -> str:
        """List all notes in the vault to debug what's been ingested"""
        rag = _rag(ctx)
        md_files = list(rag.vault_path.rglob('*.md'))
        result_text = f"Found {len(md_files)} markdown files in vault:\n\n"

        for file_path in md_files:
            relative_path = file_path.relative_to(rag.vault_path)
            result_text += f"- {relative_path}\n"

        return result_text

    @mcp.tool
    async def get_note_content(ctx: Context, file_path: str) -> str:
        """Get the full content of a specific note"""
        return await _rag(ctx).get_note_content(file_path)

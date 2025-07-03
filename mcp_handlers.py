import mcp.types as types

from mcp.server import Server
from mcp.types import Resource, Tool

from rag_server import ObsidianRAGServer


def register_handlers(server: Server, rag_server: ObsidianRAGServer):
    """Register all MCP handlers with the server instance."""

    @server.list_resources()
    async def handle_list_resources() -> list[Resource]:
        """List available resources"""
        return [
            Resource(
                uri="obsidian://vault",
                name="Obsidian Vault",
                description="Access to Obsidian vault notes",
                mimeType="text/markdown"
            )
        ]

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        """List available tools"""
        return [
            Tool(
                name="ingest_notes",
                description="Ingest all notes from Obsidian vault into vector database",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),

            Tool(
                name="search_notes",
                description="Search for similar notes using vector similarity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),

            Tool(
                name="query_knowledge",
                description="Query your knowledge base using RAG with a configured LLM",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to ask your knowledge base"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of relevant notes to use as context",
                            "default": 3
                        }
                    },
                    "required": ["question"]
                }
            ),

            Tool(
                name="update_note",
                description="Update a specific note in the vector database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the note file relative to vault"
                        }
                    },
                    "required": ["file_path"]
                }
            ),

            Tool(
                name="list_all_notes",
                description="List all notes in the vault to debug what's been ingested",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),

            Tool(
                name="get_note_content",
                description="Get the full content of a specific note",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the note file relative to vault"
                        }
                    },
                    "required": ["file_path"]
                }
            )
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        """Handle tool calls"""
        if name == "ingest_notes":
            await rag_server.ingest_all_notes()
            return [types.TextContent(type="text", text="Successfully ingested all notes")]

        elif name == "search_notes":
            query = arguments["query"]
            top_k = arguments.get("top_k", 5)
            results = await rag_server.search_similar_notes(query, top_k)

            if results:
                result_text = f"Found {len(results)} similar notes:\n\n"
                for i, note in enumerate(results, 1):
                    result_text += f"{i}. {note['title']} (similarity: {note['similarity']:.3f})\n"
                    result_text += f"   Path: {note['file_path']}\n"
                    result_text += f"   Tags: {note['tags']}\n"
                    result_text += f"   Content preview: {note['content'][:200]}...\n\n"
            else:
                result_text = "No similar notes found."

            return [types.TextContent(type="text", text=result_text)]

        elif name == "query_knowledge":
            question = arguments["question"]
            top_k = arguments.get("top_k", 3)
            answer = await rag_server.query_with_rag(question, top_k)

            return [types.TextContent(type="text", text=answer)]

        elif name == "update_note":
            file_path = arguments["file_path"]
            full_path = rag_server.vault_path / file_path
            await rag_server._update_note(str(full_path))

            return [types.TextContent(type="text", text=f"Updated note: {file_path}")]

        elif name == "list_all_notes":
            # Debug tool to see what files are in the vault
            md_files = list(rag_server.vault_path.rglob('*.md'))
            result_text = f"Found {len(md_files)} markdown files in vault:\n\n"

            for file_path in md_files:
                relative_path = file_path.relative_to(rag_server.vault_path)
                result_text += f"- {relative_path}\n"

            return [types.TextContent(type="text", text=result_text)]

        elif name == "get_note_content":
            file_path = arguments["file_path"]
            # Normalization is now handled inside get_note_content
            content = await rag_server.get_note_content(file_path)
            return [types.TextContent(type="text", text=content)]

        else:
            raise ValueError(f"Unknown tool: {name}")

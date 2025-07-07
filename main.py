import asyncio
import logging
import os
import importlib.metadata
import colorlog

from dotenv import load_dotenv

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from rag_server import ObsidianRAGServer
from mcp_handlers import register_handlers


logger = logging.getLogger("obsidian-rag")


def setup_logging():
    """Sets up colored logging for the application."""
    handler = colorlog.StreamHandler()

    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s %(name)s: %(message)s%(reset)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        reset=True,
        style='%'
    )

    handler.setFormatter(formatter)

    # Getting the root logger for 'obsidian-rag' and adding the handler:
    app_logger = logging.getLogger('obsidian-rag')
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False  # Prevent duplicate logs in the root logger


async def main():
    """Main function"""
    # Setting up logging first:
    setup_logging()

    # Loading environment variables from the .env file:
    load_dotenv()

    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

    if not vault_path:
        logger.error("OBSIDIAN_VAULT_PATH environment variable not set. Please create a .env file.")
        return

    logger.info(f"Starting ObsidianRAG with vault: {vault_path}")

    # Initializing the RAG server:
    rag_server = ObsidianRAGServer(vault_path, milvus_host, milvus_port)
    await rag_server.initialize()

    # Initializing the MCP server:
    server = Server("obsidian-rag")
    register_handlers(server, rag_server)

    # Getting server version from package metadata:
    try:
        server_version = importlib.metadata.version("personal-notes-assistant")

    except importlib.metadata.PackageNotFoundError:
        logger.warning("Package 'personal-notes-assistant' not found. Defaulting version to '0.0.0'")
        server_version = "0.0.0"

    # Running the MCP server:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="obsidian-rag",
                server_version=server_version,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

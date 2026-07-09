import logging
import os
from contextlib import asynccontextmanager

import colorlog
from dotenv import load_dotenv
from fastmcp import FastMCP

from rag_server import ObsidianRAGServer
from mcp_handlers import register_tools

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


setup_logging()
load_dotenv()

VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

if not VAULT_PATH:
    raise ValueError("OBSIDIAN_VAULT_PATH environment variable not set. Please create a .env file.")


@asynccontextmanager
async def lifespan(server):
    """Initializes the RAG server on startup and tears down the file watcher on shutdown."""
    logger.info(f"Starting ObsidianRAG with vault: {VAULT_PATH}")

    rag_server = ObsidianRAGServer(VAULT_PATH, MILVUS_HOST, MILVUS_PORT)
    await rag_server.initialize()

    try:
        yield {"rag": rag_server}
    finally:
        if rag_server.observer:
            rag_server.observer.stop()
            rag_server.observer.join()


mcp = FastMCP("obsidian-rag", lifespan=lifespan)
register_tools(mcp)


def main():
    mcp.run()


if __name__ == "__main__":
    main()

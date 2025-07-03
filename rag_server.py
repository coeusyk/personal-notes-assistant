import asyncio
import logging
import os
import re
import hashlib
import yaml
import openai
import torch

from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuring logging:
logger = logging.getLogger("obsidian-rag")


class ObsidianRAGServer:
    def __init__(self, vault_path: str, milvus_host: str = "localhost", milvus_port: int = 19530):
        self.vault_path = Path(vault_path)
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = "obsidian_notes"
        self.embedding_model = None
        self.collection = None
        self.llm_client = None

        # LLM Configuration:
        self.llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.llm_model = os.getenv("LLM_MODEL")

        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set when using the 'openai' provider.")

            self.llm_client = openai.OpenAI(api_key=api_key)

            if not self.llm_model:
                raise ValueError("OpenAI model must be set when using the 'openai' provider.")

            logger.info("Using OpenAI as the LLM provider.")

        elif self.llm_provider == "ollama":
            ollama_base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

            if not ollama_base_url.endswith("/v1"):
                ollama_base_url = f"{ollama_base_url.rstrip('/')}/v1"

            self.llm_client = openai.OpenAI(
                base_url=ollama_base_url,
                api_key='ollama'
            )

            if not self.llm_model:
                raise ValueError("Ollama model must be set when using the 'ollama' provider.")

            logger.info(f"Using Ollama as the LLM provider via endpoint: {ollama_base_url}")

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.llm_provider}. Choose 'openai' or 'ollama'.")

        # File watcher:
        self.observer = None
        self.loop = None  # Store the main event loop

    async def initialize(self):
        """Initialize all components"""
        # Storing the current event loop for the file watcher:
        self.loop = asyncio.get_running_loop()

        await self._setup_milvus()
        await self._setup_embedding_model()
        await self._setup_file_watcher()

        # Performing an initial full ingestion on startup
        logger.info("Performing initial full ingestion of the vault...")
        await self.ingest_all_notes()
        logger.info("Initial ingestion complete.")

        logger.info("ObsidianRAG server initialized and ready.")

    async def _setup_milvus(self):
        """Setup Milvus connection and collection"""
        try:
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port
            )

            # Creating a collection if it doesn't exist:
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # MiniLM dimension
                    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                    FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="modified_at", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="chunk_index", dtype=DataType.INT64)  # Add chunk index field
                ]

                schema = CollectionSchema(fields, description="Obsidian notes collection")
                self.collection = Collection(self.collection_name, schema)

                # Create index:
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.collection.create_index("embedding", index_params)

            else:
                self.collection = Collection(self.collection_name)

            self.collection.load()
            logger.info(f"Milvus collection '{self.collection_name}' ready")

        except Exception as e:
            logger.error(f"Failed to setup Milvus: {e}")
            raise

    async def _setup_embedding_model(self):
        """Initializes and loads the sentence transformer model."""
        try:
            # Check for CUDA GPU and set device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device} for embedding model.")

            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=device
            )

            # Getting embedding dimension from the model:
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    async def _setup_file_watcher(self):
        """Setup file system watcher for Obsidian vault"""

        class VaultHandler(FileSystemEventHandler):
            def __init__(self, server):
                self.server = server

            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.md'):
                    # Use call_soon_threadsafe to schedule the coroutine in the main event loop
                    asyncio.run_coroutine_threadsafe(
                        self.server._update_note(event.src_path),
                        self.server.loop
                    )

            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith('.md'):
                    # Use call_soon_threadsafe to schedule the coroutine in the main event loop
                    asyncio.run_coroutine_threadsafe(
                        self.server._update_note(event.src_path),
                        self.server.loop
                    )

        self.observer = Observer()
        self.observer.schedule(VaultHandler(self), str(self.vault_path), recursive=True)
        self.observer.start()
        logger.info("File watcher started")

    def _parse_markdown_file(self, file_path: Path) -> dict[str, str | None | Any] | None:
        """Parse Obsidian markdown file and extract metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract YAML frontmatter
            frontmatter = {}
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    try:
                        frontmatter = yaml.safe_load(parts[1]) or {}
                        content = parts[2].strip()
                    except yaml.YAMLError:
                        pass

            # Extract title (from frontmatter or first heading or filename)
            title = frontmatter.get('title') or file_path.stem
            if not title and content:
                heading_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                if heading_match:
                    title = heading_match.group(1)

            # Extract tags
            tags = frontmatter.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]

            # Find inline tags
            inline_tags = re.findall(r'#(\w+)', content)
            tags.extend(inline_tags)
            tags = list(set(tags))  # Remove duplicates

            # Get file stats
            stat = file_path.stat()

            # Normalize path to use forward slashes for cross-platform compatibility in Milvus
            relative_path = file_path.relative_to(self.vault_path).as_posix()

            return {
                'file_path': relative_path,
                'content': content,
                'title': title,
                'tags': ', '.join(tags),
                'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'content_hash': hashlib.md5(content.encode()).hexdigest()
            }

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    @staticmethod
    def _chunk_content(content: str, max_chunk_size: int = 1500) -> List[str]:
        """Split content into chunks while preserving structure"""
        # Simple chunking by paragraphs and size
        if not content.strip():
            return [content]

        # If content is small, return as single chunk
        if len(content) <= max_chunk_size:
            return [content]

        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) < max_chunk_size:
                current_chunk += para + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + '\n\n'

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content]

    async def _update_note(self, file_path: str):
        """Update a single note in the vector database"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                # File deleted, remove from database
                relative_path = file_path.relative_to(self.vault_path).as_posix()
                await self._remove_note(relative_path)
                return

            note_data = self._parse_markdown_file(file_path)
            if not note_data:
                return

            # Check if content changed
            expr = f'file_path == "{note_data["file_path"]}" and chunk_index == 0'
            existing = self.collection.query(
                expr=expr,
                output_fields=["content_hash"]
            )

            if existing and existing[0]['content_hash'] == note_data['content_hash']:
                logger.info(f"No changes detected for: {note_data['file_path']}")
                return

            # Remove existing entries for this file
            await self._remove_note(note_data['file_path'])

            # Chunk content and create embeddings
            chunks = self._chunk_content(note_data['content'])
            logger.info(f"Processing {len(chunks)} chunks for: {note_data['file_path']}")

            # Prepare data for batch insert
            file_paths = []
            contents = []
            embeddings = []
            titles = []
            tags_list = []
            created_ats = []
            modified_ats = []
            content_hashes = []
            chunk_indices = []

            for i, chunk in enumerate(chunks):
                if not chunk.strip():  # Skip empty chunks
                    continue

                embedding = self.embedding_model.encode([chunk])[0].tolist()

                file_paths.append(note_data['file_path'])
                contents.append(chunk)
                embeddings.append(embedding)
                titles.append(note_data['title'])
                tags_list.append(note_data['tags'])
                created_ats.append(note_data['created_at'])
                modified_ats.append(note_data['modified_at'])
                content_hashes.append(note_data['content_hash'])
                chunk_indices.append(i)

            # Insert all chunks at once
            if file_paths:  # Only insert if we have data
                data = [
                    file_paths,
                    contents,
                    embeddings,
                    titles,
                    tags_list,
                    created_ats,
                    modified_ats,
                    content_hashes,
                    chunk_indices
                ]

                self.collection.insert(data)
                logger.info(f"Successfully updated note: {note_data['file_path']} ({len(file_paths)} chunks)")

        except Exception as e:
            logger.error(f"Error updating note {file_path}: {e}")
            raise

    async def _remove_note(self, file_path: str):
        """Remove note from vector database"""
        try:
            normalized_path = file_path.replace('\\', '/')
            self.collection.delete(expr=f'file_path == "{normalized_path}"')
            logger.info(f"Removed note: {file_path}")

        except Exception as e:
            logger.error(f"Error removing note {file_path}: {e}")

    async def ingest_all_notes(self):
        """Ingest all markdown files in the vault (recursively)"""
        # Using rglob for recursive search:
        md_files = list(self.vault_path.rglob('*.md'))
        logger.info(f"Found {len(md_files)} markdown files (including subfolders)")

        # Processing files in batches to avoid memory issues:
        batch_size = 10
        for i in range(0, len(md_files), batch_size):
            batch = md_files[i:i + batch_size]
            for file_path in batch:
                try:
                    await self._update_note(str(file_path))

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    continue

        logger.info(f"Completed ingesting {len(md_files)} files")

    async def search_similar_notes(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar notes using vector similarity"""
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            # More lenient search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 50}}
            results = self.collection.search(
                [query_embedding],
                "embedding",
                search_params,
                limit=top_k * 2,  # Get more results initially
                output_fields=["file_path", "content", "title", "tags", "chunk_index"]
            )

            similar_notes = []
            seen_files = set()

            for hit in results[0]:
                file_path = hit.entity.get('file_path')

                # Prefer to show one result per file (the best match)
                if file_path not in seen_files:
                    similar_notes.append({
                        'file_path': file_path,
                        'content': hit.entity.get('content'),
                        'title': hit.entity.get('title'),
                        'tags': hit.entity.get('tags'),
                        'similarity': hit.score,
                        'chunk_index': hit.entity.get('chunk_index')
                    })
                    seen_files.add(file_path)

                if len(similar_notes) >= top_k:
                    break

            return similar_notes

        except Exception as e:
            logger.error(f"Error searching notes: {e}")
            return []

    async def query_with_rag(self, question: str, top_k: int = 3) -> str | None | Any:
        """Query using RAG - retrieve relevant notes and generate response"""
        try:
            relevant_notes = await self.search_similar_notes(question, top_k)
            if not relevant_notes:
                return "I couldn't find any relevant notes to answer your question."

            context_parts = [f"From '{note['title']}' ({note['file_path']}):\n{note['content']}\n" for note in
                             relevant_notes]
            context = "\n---\n".join(context_parts)

            prompt = f"""Based on the following notes from your knowledge base, answer the question.

Context from your notes:
{context}

Question: {question}

Answer based on the notes above:"""

            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that answers based on the provided notes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return f"Error processing query: {str(e)}"

    async def get_note_content(self, file_path: str) -> str:
        """Get the full content of a specific note"""
        try:
            # Normalize incoming path to use forward slashes for the query
            normalized_path = file_path.replace('\\', '/')

            # Query all chunks for this file
            results = self.collection.query(
                expr=f'file_path == "{normalized_path}"',
                output_fields=["content", "chunk_index"],
                limit=1000  # Should be enough for most notes
            )

            if not results:
                return f"Note not found: {file_path}"

            # Sort by chunk index and combine
            sorted_chunks = sorted(results, key=lambda x: x['chunk_index'])
            full_content = "\n\n".join([chunk['content'] for chunk in sorted_chunks])

            return full_content

        except Exception as e:
            logger.error(f"Error getting note content: {e}")
            return f"Error retrieving note: {str(e)}"

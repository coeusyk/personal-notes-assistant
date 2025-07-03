# Personal Notes Assistant (Obsidian RAG)

This project provides a Retrieval-Augmented Generation (RAG) server for your Obsidian vault. It indexes your
Markdown notes into a Milvus vector database and allows you to query your knowledge base using either a local
Large Language Model (LLM) via Ollama or the OpenAI API.

The server automatically watches your vault for changes and keeps the knowledge base synchronized in real-time.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [LLM Provider Configuration](#llm-provider-configuration)
  - [Using Ollama](#option-1-using-ollama-default)
  - [Using OpenAI](#option-2-using-openai)
- [Running the Server](#running-the-server)

## Features

*   **Obsidian Integration**: Directly connects to and indexes your Obsidian vault.
*   **Real-time Sync**: Automatically updates the database when notes are created or modified.
*   **Vector Search**: Employs `sentence-transformers` and **Milvus** to find relevant notes efficiently.
*   **Flexible LLM Support**: Works with both OpenAI's API and local models through Ollama.
*   **Tool-based Interface**: Exposes its functionality through a standardized MCP server interface.

## Prerequisites

Before you begin, ensure you have the following installed:

*   Python 3.9+
*   [uv](https://github.com/astral-sh/uv) (for Python package management)
*   Docker and Docker Compose (for running Milvus).
*   An Obsidian vault with your notes.
*   [Ollama](https://ollama.com/) (optional, only if you plan to use local models).

## Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/coeusyk/personal-notes-assistant.git
    cd personal-notes-assistant
    ```

2.  **Run Milvus with Docker**
    This project includes a `docker-compose.yml` file to run a Milvus instance.
    ```bash
    docker-compose up -d
    ```

3.  **Install Python Dependencies**
    This project uses `uv` to manage dependencies in a virtual environment.
    ```bash
    # Create a virtual environment
    uv venv
    
    # Activate the virtual environment
    .venv\Scripts\activate # On Linux use `source .venv/bin/activate`
    
    # Install dependencies
    uv pip install -e .
    ```

4.  **A Note on PyTorch with CUDA Support (Optional)**

    By default, the command above installs the CPU-only version of PyTorch. If you have a compatible NVIDIA GPU and want to enable CUDA for hardware acceleration, you must manually install the correct version of PyTorch.

    First, uninstall the existing version:
    ```bash
    uv pip uninstall torch
    ```

    Then, visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to find the correct installation command for your specific system and CUDA version. For example, the command might look like this:
    ```bash
    # Example command, check the official website for the correct one for your setup
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

5.  **Configure Environment Variables**

    Create a `.env` file by copying the sample file. This is where you will configure the application.
    ```bash
    cp .env.sample .env
    ```
    Open the new `.env` file and set the `OBSIDIAN_VAULT_PATH`. Then, follow the instructions in the next section to configure your chosen LLM provider.

## LLM Provider Configuration

You must choose between using Ollama (for local models) or OpenAI.

### Option 1: Using Ollama (Default)

To use a local model running on your machine, you first need to install Ollama.

1.  **Install Ollama:**

    Download and install Ollama for your operating system from the [official website](https://ollama.com/).

2.  **Download an LLM:**

    After installing Ollama, you need to pull a model. Open your terminal and run the following command. For example, to download the `mistral:7b-instruct` model:
    ```bash
    ollama pull mistral:7b-instruct
    ```
    Ensure the Ollama application is running. You can find other models in the [Ollama library](https://ollama.com/library).

3.  **Configure your `.env` file for Ollama:**

    ```env
    # --- Required Settings ---
    OBSIDIAN_VAULT_PATH="C:/Path/To/Your/Vault"

    # --- LLM Provider Settings ---
    LLM_PROVIDER=ollama

    # --- Ollama Settings ---
    OLLAMA_URL=http://localhost:11434
    LLM_MODEL=mistral:7b-instruct

    # --- Milvus Settings (Defaults) ---
    MILVUS_HOST=localhost
    MILVUS_PORT=19530
    ```

### Option 2: Using OpenAI

To use OpenAI's models, you will need an API key.

1.  **Configure your `.env` file for OpenAI:**

    ```env
    # --- Required Settings ---
    OBSIDIAN_VAULT_PATH="C:/Path/To/Your/Vault"

    # --- LLM Provider Settings ---
    LLM_PROVIDER=openai

    # --- OpenAI Settings ---
    OPENAI_API_KEY=your-openai-api-key-here
    LLM_MODEL=gpt-3.5-turbo

    # --- Milvus Settings (Defaults) ---
    MILVUS_HOST=localhost
    MILVUS_PORT=19530
    ```

## Running the Server

With the services running and your `.env` file configured, start the main application:

```bash
python main.py
```

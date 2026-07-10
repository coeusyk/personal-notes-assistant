# DESIGN.md

## Purpose

This repository is a self-hosted Retrieval-Augmented Generation (RAG) server for an Obsidian vault. It indexes Markdown notes into Milvus, keeps the index synchronized as files change, and exposes the knowledge base through MCP tools for use from Claude Desktop or any MCP-compatible client.

The visual and interaction direction for any future UI should make this feel like a premium knowledge workspace, not a generic chatbot. The product should communicate precision, trust, and calm control over personal information.

## Product definition

The current project is primarily a backend + protocol server, not a bundled end-user UI. The repository currently exposes six MCP tools: ingest notes, semantic search, RAG question answering, manual note updates, vault listing, and full-content retrieval. The project supports both Ollama and OpenAI as LLM backends. It watches the vault in real time and updates Milvus when files are created, modified, or deleted.

A future client UI, if added, should frame the system as a "knowledge operating layer" for notes rather than a chat app.

## What the product should feel like

Target emotional qualities:
- precise
- premium
- quiet
- intelligent
- editorial
- fast
- trustworthy

The UI should not feel playful, noisy, or consumer-social. It should feel closer to a high-end productivity product than to a toy chatbot.

## Reference products

Use these products as design references for overall taste, spacing, typography, motion, and information hierarchy:

- Linear: clean layout discipline, subtle motion, sharp hierarchy
- Notion AI: knowledge-centric structure and calm surfaces
- Perplexity: answer-first conversational search patterns
- Arc: modern interaction patterns and premium feel
- Raycast: command-driven workflows and compact density
- Granola: AI workspace aesthetics
- Apple Human Interface Guidelines: restraint, spacing, and readability

Do not copy components directly. Extract the interaction language and visual discipline.

## Core design principles

1. Typography first.
   - Prefer strong typographic hierarchy over decorative UI.
   - Use clear title sizes, restrained body text, and visible semantic structure.

2. Whitespace over borders.
   - Let spacing define sections.
   - Avoid boxed layouts unless they improve scannability.

3. Motion should be functional.
   - Use motion to confirm state changes, loading, transitions, and focus.
   - Avoid decorative animation.

4. Dense, but not cluttered.
   - The interface should support serious knowledge work.
   - Dense information is acceptable if it remains legible.

5. Default to trust.
   - Show provenance, source context, file paths, timestamps, and retrieval confidence where relevant.
   - Avoid UI tricks that obscure what the system is doing.

6. Keep the assistant secondary to the content.
   - The note, source excerpt, and retrieval result should remain visible.
   - The assistant is a tool for navigation and synthesis, not the entire product.

## Visual language

### Color
- Use a neutral base with one strong accent.
- Prefer off-white / near-black / gray scales.
- Reserve accent color for actions, focus, selected state, and semantic emphasis.
- Avoid rainbow palettes and heavy gradients.

### Typography
- Use a modern sans-serif style.
- Headings should be decisive and high-contrast.
- Body copy should be highly readable at small sizes.
- Metadata should be quieter than content.

### Shape and surface
- Use subtle radius values.
- Use faint shadows or hairline borders, not heavy panels.
- Avoid overuse of glassmorphism.
- Surfaces should feel architectural, not decorative.

### Spacing
- Use a consistent spacing scale based on 8px increments.
- Prefer generous vertical rhythm.
- Keep navigation compact but not cramped.

### Motion
- Use 150–250ms transitions for UI state changes.
- Loading states should feel deliberate and calm.
- Avoid bouncy or flashy animation.

## Layout direction

A good future layout would likely be one of these:

- Left sidebar: vault, collections, search, recent notes
- Main canvas: answer / note content / retrieved context
- Right panel: sources, metadata, file path, tags, embeddings or tool actions

For command-heavy workflows, a centered search or command bar can be useful as a primary entry point.

The layout should support three primary tasks:
- search notes semantically
- ask a question over the vault
- inspect the source notes behind an answer

## Interaction model

The UI should bias toward the following flows:
- query first
- sources second
- content exploration third

A good response view should include:
- answer summary
- supporting note snippets
- source file names / paths
- tags and timestamps
- controls to open the full note or rerun the query

If a chat interface is used, it should not bury the retrieval context. Source transparency matters more than long conversation history.

## Components to favor

- command palette / search bar
- note card
- source chip
- metadata row
- collapsible citation/source drawer
- activity / sync status indicator
- vault health / ingest state indicator
- empty state with a clear next action
- subtle skeleton loaders

## Components to avoid

- overly playful chat bubbles
- loud gradients
- neon neon-on-dark styling
- excessive glass effects
- cluttered sidebars with many nested controls
- hidden state changes
- decorative avatars or mascots
- generic SaaS dashboards that look unrelated to personal knowledge work

## Design references for implementation style

If implementing in React / Next.js:
- prefer shadcn/ui or a similarly minimal component foundation
- use CSS variables for theme tokens
- keep component logic small and composable
- use a predictable spacing and typography scale
- make source inspection a first-class UI surface
- keep empty states and loading states visually polished

If implementing a demo or landing page:
- use strong hero typography
- show the product as a knowledge workspace, not a chatbot
- include a polished source/answer split view
- show a live vault sync or indexed-notes status area

## Repo-specific technical context

The backend currently includes these implementation realities that matter for any UI:
- `main.py` bootstraps the MCP server and loads environment variables.
- `rag_server.py` owns parsing, chunking, indexing, file watching, retrieval, and generation.
- `mcp_handlers.py` exposes the tools.
- Milvus is the vector store.
- Sentence embeddings use `all-MiniLM-L6-v2`.
- The system supports Ollama and OpenAI.
- Recent fixes addressed silent correctness issues such as missing Milvus flushes and missing delete propagation.

That means any UI should expose operational status clearly, especially indexing state and sync health.

## Recent repo changes to reflect

Recent commits indicate the project has moved in a more production-aware direction:
- a vault status tool was added to surface ingest and watcher state
- CUDA torch support was introduced and ingestion responsiveness was improved
- the README now documents the `run-obsidian-rag` entry point and Milvus startup order
- file-watcher crashes were fixed, Milvus flush behavior was batched, and unused code was trimmed
- the MCP server was migrated to FastMCP, RAG bugs were fixed, and `CLAUDE.md` was added

The design should therefore accommodate a more stable, status-aware, and operationally transparent experience.

## If a UI is added later

The ideal UI should be a polished local web app that can launch alongside the server, or a companion app that connects over MCP. The best presentable direction is a calm, high-end knowledge product with strong visual hierarchy and visible source provenance.

The best fit is not a "chatbot skin." It is a focused research workspace.

## Implementation guardrails

- Preserve source traceability.
- Preserve vault-relative file paths.
- Preserve clear sync and ingest state.
- Do not hide retrieval context behind animation.
- Do not add visual complexity that reduces trust.
- Prefer deliberate, minimal, premium interactions over novelty.

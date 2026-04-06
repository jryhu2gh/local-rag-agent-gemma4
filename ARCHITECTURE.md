# RAG Agent with Local Gemma 4 — Architecture & Implementation Plan

> Created: 2026-04-03
> Status: Skills-based architecture (v2)

## Overview

A **skills-based Research/RAG agent** in Python that runs entirely locally on an M3 Pro Mac (36GB RAM), powered by Gemma 4 26B-A4B via llama.cpp. The agent uses a skills architecture: the LLM picks a high-level skill, and the skill's Python code orchestrates the underlying operations deterministically.

Prompt design inspired by [Claude Code's modular prompt patterns](https://github.com/Piebald-AI/claude-code-system-prompts).

## Architecture

```
┌─────────────┐     ┌──────────────────────────┐
│  You (REPL) │────▶│  Python Agent (agent.py)  │
└─────────────┘     │  - system prompt          │
                    │  - agent loop              │
                    │  - tool dispatch           │
                    └────┬──────────┬────────────┘
                         │          │
              tool calls │          │ embeddings
                         ▼          ▼
               ┌──────────┐  ┌──────────┐
               │ llama-   │  │ llama-   │
               │ server   │  │ server   │
               │ :8080    │  │ :8081    │
               │ Gemma 4  │  │ EmbGemma │
               │ (chat)   │  │ (embed)  │
               └──────────┘  └──────────┘
```

### Inference Backend (llama.cpp)

- **Chat server (port 8080)**: Gemma 4 26B-A4B-it Q4_K_M, OpenAI-compatible API with native tool calling
  - Model: `~/local-llm/gemma4/gemma-4-26B-A4B-it-Q4_K_M.gguf`
  - Binary: `~/local-llm/llama.cpp/build/bin/llama-server`
  - Template: `~/local-llm/llama.cpp/models/templates/gemma4.jinja`
  - Flags: `--jinja --chat-template-file gemma4.jinja -fa -c 8192`
- **Embedding server (port 8081)**: EmbeddingGemma 300M (auto-downloaded via `--embd-gemma-default`)
  - Flags: `--embd-gemma-default --embeddings --port 8081`

### Python Agent (Skills Architecture)

Uses the `openai` Python SDK pointed at localhost. Skills are presented as "tools" via the OpenAI API, but each skill orchestrates multiple internal operations:

1. User sends a question
2. LLM receives system prompt + conversation + skill definitions
3. LLM picks a skill (e.g., `research`) — this is the only LLM decision per turn
4. Agent calls the skill's Python `execute()` function, which deterministically runs internal operations (search history, search docs, etc.)
5. Skill result goes back as `role: "tool"` message
6. Repeat from step 3 until text response or MAX_TURNS reached

**Why skills over direct tools**: The LLM makes one high-level decision instead of managing multi-step orchestration. Tiered retrieval, BFS crawling, and other workflows are handled by deterministic Python code — more reliable, more testable.

## File Structure

```
~/local-llm/agent/
├── ARCHITECTURE.md       # This file
├── start_servers.sh      # Launch both llama-server instances
├── requirements.txt      # openai, numpy, pymupdf
├── config.py             # Constants (URLs, paths, RAG params)
├── prompts.py            # System prompt
├── toolkit.py            # Internal functions (search, browse, crawl) — not LLM-facing
├── skills/
│   ├── __init__.py       # Skill registry + execute_skill() dispatch
│   ├── research.py       # Tiered search: history → documents
│   ├── read_doc.py       # Read full document content
│   ├── browse.py         # Fetch a web page
│   └── index_site.py     # Crawl + index a website
├── agent.py              # Core agent loop (calls skills)
├── main.py               # Interactive REPL entry point
├── ingest.py             # CLI to ingest documents into the index
├── rag/
│   ├── __init__.py
│   ├── embedder.py       # Calls embedding server at :8081 for semantic vectors
│   ├── bm25.py           # BM25 keyword-based search index (pure Python)
│   ├── index.py          # Numpy-based vector store (cosine similarity)
│   ├── hybrid.py         # Hybrid search: merges BM25 + embedding via RRF
│   └── chunker.py        # Split docs into overlapping chunks
├── documents/            # User places PDFs, .md, .txt files here
└── index_data/           # Persisted embeddings (.npy) + metadata + BM25 index (.json)
```

## Dependencies

```
openai>=1.0.0   # OpenAI-compatible client for llama-server
numpy           # Cosine similarity for vector search
pymupdf         # PDF text extraction
```

No LangChain, no vector database, no heavy frameworks.

## Skills

Skills are the LLM-facing interface. Each skill maps to a user intent and orchestrates internal toolkit functions.

| Skill | Input | Orchestration |
|-------|-------|---------------|
| `research` | `query: str` | search_history → search_documents → return combined results |
| `read_document` | `doc_id: str` | Read full document from disk |
| `browse` | `url: str` | Fetch page → extract text + links |
| `index_site` | `url: str, max_pages?: int` | BFS crawl → save pages → index → report |

### Internal Toolkit (`toolkit.py`)

Functions called by skills, not exposed to the LLM:
- `search_documents(query)` — hybrid search (BM25 + embeddings via RRF)
- `read_document(doc_id)` — read file by ID
- `browse_website(url)` — fetch + extract text/links
- `crawl_website(url, max_pages)` — BFS crawl + incremental indexing

Skill definitions use OpenAI JSON schema format. llama-server translates Gemma 4's native `<|tool_call>call:name{...}<tool_call|>` tokens into standard OpenAI `tool_calls` objects transparently.

## RAG Pipeline

### Ingestion
1. **Scan**: `documents/` directory for `.txt`, `.md`, `.pdf` files
2. **Extract**: Text via pathlib (txt/md) or pymupdf (PDF)
3. **Chunk**: Split into overlapping chunks (512 chars, 64 char overlap)
4. **Index (BM25)**: Tokenize chunks, compute term frequencies, build inverted index
5. **Index (Embedding)**: Call embedding server, store vectors as numpy array
6. **Persist**: Save both indexes + metadata to `index_data/`

### Search (Hybrid)
Uses **Reciprocal Rank Fusion (RRF)** to combine two complementary search methods:

```
User query
    ├──▶ BM25 search ──▶ ranked results (keyword matches)
    └──▶ Embedding search ──▶ ranked results (semantic similarity)
              │
              ▼
    RRF merge: score(doc) = Σ 1/(k + rank_i)
              │
              ▼
    Top-k combined results
```

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| **BM25** (keyword) | Exact terms, names, IDs, error codes | Misses paraphrases and synonyms |
| **Embedding** (semantic) | Meaning-based, handles rephrasings | Can miss exact technical terms |
| **Hybrid (RRF)** | Best of both | Slightly more compute (negligible) |

**RRF formula**: For each document, sum `1 / (k + rank)` across both result lists (k=60 is standard). Documents that rank high in both lists get boosted; documents that appear in only one list still surface.

## System Prompt Design

Follows Claude Code patterns:
- Clear role definition ("You are a research assistant...")
- Explicit process instructions (search first, then read, then answer)
- Behavioral rules (always cite sources, don't make up info)
- Tool usage guidance (when to use each tool)

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHAT_BASE_URL` | `http://localhost:8080/v1` | Chat server endpoint |
| `EMBED_BASE_URL` | `http://localhost:8081/v1` | Embedding server endpoint |
| `BM25_WEIGHT` | 0.5 | Weight for BM25 in hybrid search (0-1) |
| `EMBED_WEIGHT` | 0.5 | Weight for embedding in hybrid search (0-1) |
| `RRF_K` | 60 | RRF constant (standard default) |
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `MAX_TURNS` | 10 | Max tool-call iterations |
| `TEMPERATURE` | 0.7 | LLM temperature |
| `MAX_TOKENS` | 2048 | Max response tokens |
| `CONTEXT_SIZE` | 8192 | llama-server context window |

## Build Order

### Phase 1: Infrastructure (steps 1-5)
- Create directory structure, install deps, write config, write server launch script
- **Verify**: curl both server endpoints

### Phase 2: RAG Pipeline (steps 6-12)
- Build chunker, BM25 index, embedder, vector index, hybrid search, ingestion CLI
- **Verify**: ingest sample docs, compare BM25-only vs embedding-only vs hybrid results

### Phase 3: Agent (steps 13-17)
- Build system prompt, tool definitions, agent loop, REPL
- **Verify**: end-to-end Q&A over ingested documents

## Technical Notes

- **Gemma 4 tool calling**: llama-server has native support via `TAG_WITH_GEMMA4_DICT` format parser (see `llama.cpp/common/chat-auto-parser.h`). The Jinja template at `models/templates/gemma4.jinja` handles formatting.
- **Memory budget**: Gemma 4 Q4 (~14GB) + EmbeddingGemma (~200MB) + KV cache for 8192 context fits comfortably in 36GB.
- **Error handling**: Wrap tool call parsing in try/except — model may occasionally malformat calls. Return error to model and let it retry (bounded by MAX_TURNS).
- **Skills architecture**: Skills are presented as "tools" via the OpenAI API (same wire format), so the agent loop is unchanged. Each skill is a Python module in `skills/` with a `DEFINITION` dict and an `execute()` function. Adding a new skill = adding one file + registering it in `skills/__init__.py`.

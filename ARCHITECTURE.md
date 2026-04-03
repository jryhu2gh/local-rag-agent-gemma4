# RAG Agent with Local Gemma 4 — Architecture & Implementation Plan

> Created: 2026-04-03
> Status: Design complete, implementation pending

## Overview

A **tool-calling Research/RAG agent** in Python that runs entirely locally on an M3 Pro Mac (36GB RAM), powered by Gemma 4 26B-A4B via llama.cpp. The agent searches a local document collection and answers questions using retrieved context.

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

### Python Agent

Uses the `openai` Python SDK pointed at localhost. The agent loop:
1. User sends a question
2. LLM receives system prompt + conversation + tool definitions
3. LLM returns either a text response (done) or tool_calls
4. Agent executes the tool, feeds result back as `role: "tool"` message
5. Repeat from step 3 until text response or MAX_TURNS reached

## File Structure

```
~/local-llm/agent/
├── ARCHITECTURE.md       # This file
├── start_servers.sh      # Launch both llama-server instances
├── requirements.txt      # openai, numpy, pymupdf
├── config.py             # Constants (URLs, paths, RAG params)
├── prompts.py            # System prompt
├── tools.py              # Tool definitions (OpenAI JSON schema) + dispatch
├── agent.py              # Core agent loop
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

## Tools

| Tool | Signature | Purpose |
|------|-----------|---------|
| `search_documents` | `(query: str) → chunks[]` | Hybrid search (BM25 + embeddings merged via RRF), return top-k chunks |
| `read_document` | `(doc_id: str) → text` | Return full text of a document by its ID |
| `summarize_text` | `(text: str) → summary` | Extractive summarization (first N sentences) — no extra LLM call in v1 |

Tool definitions use OpenAI JSON schema format. llama-server translates Gemma 4's native `<|tool_call>call:name{...}<tool_call|>` tokens into standard OpenAI `tool_calls` objects transparently.

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
- **Future expansion**: Can evolve into multi-agent system by adding specialized agents with different system prompts (planner, coder, reviewer) — the tool-calling loop is the foundation.

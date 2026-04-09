# RAG Agent with Local Gemma 4 — Architecture

> Created: 2026-04-03
> Updated: 2026-04-09
> Status: Multi-agent research with Gemma 4 thinking mode

## Overview

A **multi-agent research system** in Python that runs entirely locally on an M3 Pro Mac (36GB RAM), powered by Gemma 4 26B-A4B via llama.cpp. The agent uses a skills architecture with multi-agent orchestration for complex research, Gemma 4's native thinking mode for internal reasoning, and a report generation pipeline for structured output.

Prompt design inspired by [Claude Code's modular prompt patterns](https://github.com/Piebald-AI/claude-code-system-prompts).

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────┐
│  You (REPL) │────▶│  Agent (agent.py)                    │
└─────────────┘     │  - thinking mode (parse/strip)       │
                    │  - skill dispatch                     │
                    │  - auto-chain (reflect → investigate) │
                    └────┬──────────┬───────────────────────┘
                         │          │
              skill calls│          │ embeddings
                         ▼          ▼
               ┌──────────┐  ┌──────────┐
               │ llama-   │  │ llama-   │
               │ server   │  │ server   │
               │ :8080    │  │ :8081    │
               │ Gemma 4  │  │ EmbGemma │
               │ (chat)   │  │ (embed)  │
               └──────────┘  └──────────┘

When investigate(depth="deep") is called:

┌──────────────────────────────────────────────────────────┐
│  Orchestrator (orchestrator.py)                          │
│                                                          │
│  1. DECOMPOSE: LLM plans research threads                │
│  2. DISPATCH:  spawn sub-agents per thread               │
│  3. EVALUATE:  LLM checks completeness, requests         │
│                follow-ups if needed                       │
│  4. SYNTHESIZE: LLM merges all findings                  │
│                                                          │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐              │
│  │ SubAgent A│ │ SubAgent B│ │ SubAgent C│  ...          │
│  │ web_search│ │deep_resrch│ │search_locl│              │
│  │ deep_rsrch│ │ web_search│ │ deep_rsrch│              │
│  └───────────┘ └───────────┘ └───────────┘              │
└──────────────────────────────────────────────────────────┘

When generate_report is called:

┌──────────────────────────────────────────────────────────┐
│  Report Generator (report_generator.py)                  │
│                                                          │
│  1. OUTLINE:  LLM creates ToC with drafting memos        │
│  2. DRAFT:    write each section with sliding context     │
│               (outline + prev paragraph + research +      │
│                blackboard of what's been covered)         │
│  3. SUMMARY:  executive summary written last              │
│  4. ASSEMBLE: combine into final markdown document        │
└──────────────────────────────────────────────────────────┘
```

### Inference Backend (llama.cpp)

- **Chat server (port 8080)**: Gemma 4 26B-A4B-it Q4_K_M, OpenAI-compatible API with native tool calling and thinking mode
  - Model: `~/local-llm/gemma4/gemma-4-26B-A4B-it-Q4_K_M.gguf`
  - Binary: `~/local-llm/llama.cpp/build/bin/llama-server`
  - Template: `~/local-llm/llama.cpp/models/templates/gemma4.jinja`
  - Flags: `--jinja --chat-template-file gemma4.jinja -fa -c 8192`
- **Embedding server (port 8081)**: EmbeddingGemma 300M
  - Flags: `--embeddings --port 8081`

### Gemma 4 Thinking Mode

The agent enables Gemma 4's native thinking mode via `enable_thinking=True` passed to the server. The chat template (`gemma4.jinja`) handles this:

1. `<|think|>` token is injected into the system turn, activating thinking
2. The model wraps internal reasoning in `<|channel>thought ... <channel|>` blocks
3. `llm.parse_thinking()` separates thinking from visible content
4. Thinking is logged for debugging but stripped from conversation history
5. The template's `strip_thinking` macro also strips on re-render (belt and suspenders)

This prevents **reasoning debt** — where the model gets confused by its own previous (potentially incorrect) internal reasoning.

### Gemma 4 Special Tokens

All tokens used by the chat template are real entries in the tokenizer vocabulary:

| ID | Token | Purpose |
|---|---|---|
| 2 | `<bos>` | Beginning of sequence |
| 46-47 | `<\|tool>` / `<tool\|>` | Tool definition wrappers |
| 48-49 | `<\|tool_call>` / `<tool_call\|>` | Tool call format |
| 50-51 | `<\|tool_response>` / `<tool_response\|>` | Tool response format |
| 52 | `<\|"\|>` | Escaped quotes in tool schemas |
| 98 | `<\|think\|>` | Thinking mode activation flag |
| 100-101 | `<\|channel>` / `<channel\|>` | Thinking channel delimiters |
| 105-106 | `<\|turn>` / `<turn\|>` | Turn delimiters |

### Python Agent

Uses the `openai` Python SDK pointed at localhost. Skills are presented as "tools" via the OpenAI API:

1. User sends a question
2. LLM thinks internally (thinking mode), then picks a skill
3. Agent calls the skill's `execute()` function
4. Skill result goes back as `role: "tool"` message
5. If `reflect` returns `"insufficient"`, agent auto-chains to `investigate(depth="deep")`
6. Repeat until text response or MAX_TURNS reached
7. Thinking blocks are stripped from all stored messages

## File Structure

```
~/local-llm/agent/
├── ARCHITECTURE.md        # This file
├── README.md              # Quick start guide
├── start_servers.sh       # Launch both llama-server instances
├── requirements.txt       # openai, numpy, pymupdf, ddgs, beautifulsoup4
├── config.py              # Constants (URLs, paths, RAG params, agent params)
├── prompts.py             # System prompt (behavioral instructions only —
│                          #   tokens and tools handled by the chat template)
├── llm.py                 # Shared LLM client with thinking mode support
│                          #   (parse_thinking, enable_thinking)
├── toolkit.py             # Internal functions (search, browse, crawl) — not LLM-facing
├── skills/
│   ├── __init__.py        # Skill registry + execute_skill() dispatch
│   ├── reflect.py         # Check local knowledge (history + documents)
│   ├── investigate.py     # Web research (quick or deep multi-agent)
│   ├── read_doc.py        # Read full document content
│   ├── index_site.py      # Crawl + index a website
│   ├── generate_report.py # Generate structured multi-section reports
│   └── get_time.py        # Return current date/time
├── orchestrator.py        # Multi-agent research coordinator
│                          #   (decompose → dispatch → evaluate → synthesize)
├── sub_agent.py           # Individual research sub-agents with own context
├── report_generator.py    # Report pipeline (outline → draft → summary → assemble)
├── history.py             # Cross-session Q&A persistence and search
├── agent.py               # Core agent loop (thinking, skills, auto-chain)
├── main.py                # Interactive REPL entry point
├── ingest.py              # CLI to ingest documents into the index
├── rag/
│   ├── __init__.py
│   ├── embedder.py        # Calls embedding server at :8081
│   ├── bm25.py            # BM25 keyword search index (pure Python)
│   ├── index.py           # Numpy vector store (cosine similarity)
│   ├── hybrid.py          # Hybrid search via Reciprocal Rank Fusion
│   └── chunker.py         # Split docs into overlapping chunks
├── documents/             # User places PDFs, .md, .txt files here
├── index_data/            # Persisted document indexes
├── history/               # Saved Q&A pairs (JSONL)
└── history_index/         # Persisted history indexes
```

## Skills

Skills are the LLM-facing interface. Each skill maps to a user intent and orchestrates internal operations.

| Skill | Parameters | What it does |
|-------|-----------|--------------|
| `reflect` | `query` | Search history → search documents → LLM assesses if answerable. Returns status: `answered`, `insufficient`, or `no_local_knowledge` |
| `read_document` | `doc_id` | Read full document text from disk |
| `investigate` | `query`, `depth` | `"quick"`: single sub-agent web lookup. `"deep"`: multi-agent decompose → dispatch → evaluate → synthesize |
| `index_site` | `url`, `max_pages?` | BFS crawl → save pages → index for future reflect |
| `generate_report` | `topic`, `research_data` | Outline → draft sections with sliding context → executive summary → assemble |
| `get_time` | (none) | Return current date and time |

### Internal Toolkit (`toolkit.py`)

Functions called by skills and sub-agents, not exposed to the main LLM:
- `search_documents(query)` — hybrid search (BM25 + embeddings via RRF)
- `read_document(doc_id)` — read file by ID
- `web_search(query)` — DuckDuckGo search
- `deep_research(query?, url?)` — web search + read top pages, or browse a URL
- `browse_website(url)` — fetch + extract text/links
- `crawl_website(url, max_pages)` — BFS crawl + incremental indexing

### Multi-Agent Investigation

When `investigate(depth="deep")` is called, the orchestrator (`orchestrator.py`):

1. **Decompose**: LLM breaks the question into 3-5 independent research threads
2. **Dispatch**: Creates a `SubAgent` per thread, each with its own conversation context and access to `web_search`, `deep_research`, `search_local`
3. **Evaluate**: LLM reviews all sub-agent summaries for completeness. Can request follow-ups from specific sub-agents (up to 3 evaluation rounds)
4. **Synthesize**: LLM merges all thread findings into one comprehensive answer

For single-thread questions, the orchestrator skips evaluation and synthesis.

### Report Generation

When `generate_report(topic, research_data)` is called, the report generator (`report_generator.py`):

1. **Outline**: LLM creates a structured table of contents with per-section drafting memos (50-word instructions on what each section must cover)
2. **Draft**: Each section is written individually with sliding context:
   - Full outline (global context)
   - Previous section's last paragraph (transition context)
   - Research data (evidence context)
   - Blackboard of sections already written (anti-repetition)
3. **Executive Summary**: Written last, informed by all section summaries
4. **Assemble**: Title + executive summary + all sections → markdown

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

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHAT_BASE_URL` | `http://localhost:8080/v1` | Chat server endpoint |
| `EMBED_BASE_URL` | `http://localhost:8081/v1` | Embedding server endpoint |
| `CHUNK_SIZE` | 512 | Characters per chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `BM25_WEIGHT` | 0.5 | BM25 weight in hybrid search |
| `EMBED_WEIGHT` | 0.5 | Embedding weight in hybrid search |
| `RRF_K` | 60 | RRF constant |
| `MAX_TURNS` | 10 | Max tool-call iterations (main agent) |
| `MAX_SUB_AGENT_TURNS` | 6 | Max tool calls per sub-agent |
| `MAX_INVESTIGATE_THREADS` | 5 | Max research threads per investigation |
| `MAX_EVALUATE_ROUNDS` | 3 | Max evaluation iterations before synthesis |
| `MAX_REPORT_SECTIONS` | 8 | Max sections in a generated report |
| `SECTION_TARGET_WORDS` | 500 | Target word count per report section |
| `RESEARCH_CONTEXT_LIMIT` | 12000 | Chars of research data per section prompt |
| `TEMPERATURE` | 0.7 | LLM temperature |
| `MAX_TOKENS` | 2048 | Max response tokens |
| `CONTEXT_SIZE` | 8192 | llama-server context window |

## Technical Notes

- **Gemma 4 tool calling**: llama-server supports native tool calling via the Jinja template (`gemma4.jinja`). Tools are rendered as `<|tool>declaration:name{...}<tool|>` in the system turn. Tool calls use `<|tool_call>call:name{args}<tool_call|>`.
- **Thinking mode**: Enabled via `enable_thinking=True` passed in the request body. The template injects `<|think|>` and the model produces `<|channel>thought...<channel|>` blocks. These are stripped from history by both Python code and the template's `strip_thinking` macro.
- **Memory budget**: Gemma 4 Q4 (~14GB) + EmbeddingGemma (~200MB) + KV cache for 8192 context fits in 36GB.
- **Auto-chain**: When `reflect` returns status `"insufficient"` or `"no_local_knowledge"`, the agent automatically calls `investigate(depth="deep")` without waiting for the LLM to make a second tool call (local models often fail at multi-step tool chaining).
- **Adding a new skill**: Create a Python module in `skills/` with a `DEFINITION` dict and an `execute()` function, then register it in `skills/__init__.py`.

# Local Multi-Agent Deep Research System

A multi-agent research system that runs **entirely locally** on your machine. No cloud services, no API keys, fully offline. It answers questions using a knowledge-first approach — drawing from its training data, local documents, session research cache, and web investigations.

Built with:
- **Gemma 4 26B-A4B** (MoE, only 4B params active per inference) via llama.cpp
- **EmbeddingGemma 300M** for semantic search
- **Python** with the OpenAI SDK pointed at local servers

## What It Does

1. **Answers from knowledge first** — drafts an answer from training data, identifies gaps
2. **Searches local documents** — hybrid BM25 + semantic search over your indexed files
3. **Runs multi-agent web research** — decomposes complex questions into parallel research threads, each handled by an independent sub-agent
4. **Remembers within a session** — raw research data is indexed in-memory so follow-up questions don't re-do web searches
5. **Generates structured reports** — multi-section documents with executive summaries from research findings
6. **Learns across sessions** — Q&A pairs are indexed for retrieval in future conversations

---

## Installation Guide (Step by Step)

This guide assumes you have a Mac with Apple Silicon (M1/M2/M3/M4) and have never set up a development environment before. If you're on Linux with a GPU, the steps are similar — see the notes at the end.

### Step 0: Open Terminal

On macOS, press **Cmd + Space**, type **Terminal**, and press Enter. All commands below are typed into this window.

### Step 1: Install Xcode Command Line Tools

These are basic developer tools that macOS needs to compile software.

```bash
xcode-select --install
```

A popup will appear — click **Install** and wait for it to finish (may take a few minutes).

### Step 2: Install Homebrew (package manager)

Homebrew makes it easy to install software on macOS. Paste this entire command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. When it finishes, it may tell you to run a command to add Homebrew to your PATH — **do that**.

### Step 3: Install CMake and Python

```bash
brew install cmake python@3.12
```

Verify they're installed:

```bash
cmake --version
python3 --version
```

Both should print version numbers without errors.

### Step 4: Build llama.cpp (the local AI engine)

This is the software that runs the AI model on your machine.

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/local-llm/llama.cpp
cd ~/local-llm/llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

This compiles llama.cpp with Metal GPU acceleration. It takes 2-5 minutes.

### Step 5: Download the AI models

You need two model files — a large chat model (~16GB) and a small embedding model (~200MB).

```bash
# Install the download tool
pip3 install huggingface_hub

# Download Gemma 4 chat model (~16GB, may take 10-30 min depending on internet)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ggml-org/gemma-4-26b-a4b-it-GGUF',
    filename='gemma-4-26B-A4B-it-Q4_K_M.gguf',
    local_dir='$HOME/local-llm/gemma4'
)
"

# Download EmbeddingGemma embedding model (~200MB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ggml-org/embeddinggemma-300M-qat-Q4_0-GGUF',
    filename='embeddinggemma-300M-qat-Q4_0.gguf',
    local_dir='$HOME/local-llm/gemma4'
)
"
```

### Step 6: Install Python dependencies

```bash
cd ~/local-llm/agent
pip3 install -r requirements.txt
```

This installs: `openai` (API client), `numpy` (math), `pymupdf` (PDF reading), `beautifulsoup4` (web page parsing), `ddgs` (DuckDuckGo search).

### Step 7: Add your documents (optional)

Place any `.txt`, `.md`, or `.pdf` files you want the agent to know about in the `documents/` folder:

```bash
cp ~/Desktop/my-notes.md ~/local-llm/agent/documents/
cp ~/Desktop/paper.pdf ~/local-llm/agent/documents/
```

### Step 8: First run

You need **two terminal windows** (Cmd+N in Terminal to open a second one).

**Terminal 1 — Start the AI servers:**

```bash
cd ~/local-llm/agent
./start_servers.sh
```

Wait until you see "Chat server (:8080) is ready" and "Embedding server (:8081) is ready". The chat model takes 15-30 seconds to load.

**Terminal 2 — Ingest documents and start the agent:**

```bash
cd ~/local-llm/agent
python3 ingest.py     # index your documents (only needed once, or when docs change)
python3 main.py       # start the agent
```

The agent will wait for the servers to be ready, then show:

```
RAG Agent ready. Type your questions below.
Commands:  /clear = reset conversation,  /quit = exit
--------------------------------------------------

You: 
```

Type your question and press Enter.

---

## Usage

### Asking questions

```
You: What are the latest trends in green energy?
```

The agent will:
1. Draft an answer from its training knowledge
2. Identify gaps (e.g., "my training data may not cover 2026 developments")
3. Call tools to fill gaps — search local docs, run web research
4. Deliver a structured answer with sources

### Commands

| Command | What it does |
|---------|-------------|
| `/clear` | Reset conversation history and session research cache |
| `/quit` | Exit the agent |
| Ctrl+C | Exit the agent |

### Example session

```
You: How is AMD stock doing?

  [agent] Turn 1/10
  [agent] Calling investigate
  -> Calling skill: investigate({"depth": "deep", "query": "AMD stock performance 2026"})

  [orchestrator] === DISPATCH (4 threads in parallel) ===
  ...sub-agents researching financial, technical, sentiment, partnerships...

  [orchestrator] All 4 sub-agents completed in 45.2s

Agent: # AMD Stock Analysis
  ## Executive Summary
  ...structured answer with specific data points...

You: Generate a report on that

  -> Calling skill: generate_report({"topic": "AMD stock analysis"})

Agent: [Full multi-section report with executive summary]
```

---

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────┐
│  You (REPL) │────>│  Lead Agent (agent.py)            │
└─────────────┘     │  - Knowledge-first decision flow  │
                    │  - Time-aware (current date in    │
                    │    prompt for temporal reasoning)  │
                    │  - Iterative gap-filling          │
                    └──┬──────────┬─────────────────────┘
                       │          │
            skill calls│          │ embeddings
                       v          v
             ┌──────────┐  ┌──────────┐
             │ llama-   │  │ llama-   │
             │ server   │  │ server   │
             │ :8080    │  │ :8081    │
             │ Gemma 4  │  │ EmbGemma │
             │ (chat)   │  │ (embed)  │
             └──────────┘  └──────────┘
```

### Two-tier agent design

- **Lead Agent** — user-facing, uses high-level skills, makes strategic decisions about what to research
- **Sub-Agents** — research workers with web search tools, each assigned one dimension of a complex question, run in parallel

### Three-tier search

| Tier | Scope | What's in it |
|------|-------|-------------|
| **Documents** | Persistent (on disk) | Files you put in `documents/`, indexed websites |
| **History** | Cross-session (on disk) | Past Q&A pairs from all sessions |
| **Research** | Single session (in-memory) | Raw web pages and search results from current session |

### Deep investigation pipeline

1. **Decompose** — planner LLM splits question into 3-5 non-overlapping research threads
2. **Dispatch** — sub-agents run in parallel, each researching one thread
3. **Evaluate** — reviewer checks data density, requests follow-ups if needed (up to 3 rounds)
4. **Synthesize** — combines all findings into a comprehensive answer

## Skills

| Skill | What it does |
|-------|-------------|
| `reflect` | Searches past conversations, indexed documents, and session research data. Assesses if the question can be answered locally |
| `read_document` | Reads the full text of a specific indexed document |
| `investigate` | Runs web research. `depth="quick"` for single-thread lookups, `depth="deep"` for multi-agent parallel research with evaluation and synthesis |
| `index_site` | Crawls a website via BFS, saves pages locally, and indexes them |
| `generate_report` | Creates a structured multi-section report from prior research |
| `get_time` | Returns the current date and time |

## Custom paths

If your llama.cpp or models are in different locations:

```bash
export LLAMA_DIR=/path/to/llama.cpp
export CHAT_MODEL_PATH=/path/to/gemma-4-26B-A4B-it-Q4_K_M.gguf
export EMBED_MODEL_PATH=/path/to/embeddinggemma-300M-qat-Q4_0.gguf
```

These work for both `start_servers.sh` and the Python scripts.

## System requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4) or **Linux** with a capable GPU
- At least **20GB free RAM** (36GB+ recommended for comfortable use)
- **~20GB disk space** for models and llama.cpp
- Python 3.9+
- Internet connection for web research (the AI itself runs offline)

### Linux notes

Replace the Metal build flag with your GPU backend:

```bash
# NVIDIA GPU (CUDA)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# AMD GPU (ROCm)
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release

# CPU only (slow but works)
cmake -B build -DCMAKE_BUILD_TYPE=Release
```

Install build tools with `sudo apt install build-essential cmake` instead of Xcode.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'openai'` | Run `pip3 install -r requirements.txt`. If using macOS system Python, try `/usr/bin/python3 -m pip install -r requirements.txt` |
| `Error code: 503 - Loading model` | The chat server is still loading the model into memory. Wait 15-30 seconds and try again, or restart with `./start_servers.sh` |
| `No index found` | Run `python3 ingest.py` before `python3 main.py` |
| Agent is slow | Gemma 4 26B runs at ~30-35 tokens/sec on Apple Silicon. Deep investigations with 4 parallel agents take 1-2 minutes. This is normal for local inference |
| `pkill llama-server` | Run this to stop all servers before restarting |

## Project structure

```
agent/
  main.py              — Interactive REPL entry point
  agent.py             — Lead agent (tool-calling loop, decision logging)
  orchestrator.py      — Multi-agent pipeline (decompose/dispatch/evaluate/synthesize)
  sub_agent.py         — Research worker agents
  llm.py               — Shared LLM client (Gemma 4 thinking mode)
  config.py            — Configuration constants
  prompts.py           — Lead agent system prompt (knowledge-first, time-aware)
  toolkit.py           — Web search, browsing, crawling, document search
  history.py           — Cross-session Q&A persistence and search
  research_vault.py    — Raw research data storage (thread-safe)
  research_index.py    — Session-scoped in-memory search index
  report_generator.py  — Multi-section report pipeline
  ingest.py            — Document ingestion into BM25 + vector indexes
  start_servers.sh     — Launches llama-server instances
  skills/              — LLM-facing skill definitions
  rag/                 — Hybrid search engine (BM25 + vector + RRF fusion)
  documents/           — Your source documents
  index_data/          — Persistent document indexes
  history/             — Saved Q&A pairs
  research/            — Saved research vaults from investigations
```

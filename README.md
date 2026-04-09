# Local RAG Agent with Gemma 4

A multi-agent research system that runs **entirely locally** on your machine. It searches local documents using hybrid search (BM25 + semantic embeddings), runs multi-threaded web investigations, generates structured reports, and answers questions with cited sources.

Uses **Gemma 4 thinking mode** (`<|channel>thought`) for internal reasoning and a **skills architecture** where the LLM picks a high-level skill and Python code handles the orchestration deterministically.

Built with:
- **Gemma 4 26B-A4B** (MoE, only 4B params active per inference) via llama.cpp
- **EmbeddingGemma 300M** for semantic search
- **Python** with the OpenAI SDK pointed at local servers
- No cloud, no API keys, fully offline

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) or Linux with a capable GPU
- At least 20GB free RAM (36GB+ recommended)
- Python 3.9+
- Xcode Command Line Tools (macOS) or build-essential (Linux)
- CMake

## Setup

### 1. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/local-llm/llama.cpp
cd ~/local-llm/llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### 2. Download the models

```bash
# Install huggingface_hub if needed
pip3 install huggingface_hub

# Gemma 4 26B-A4B chat model (~16GB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ggml-org/gemma-4-26b-a4b-it-GGUF',
    filename='gemma-4-26B-A4B-it-Q4_K_M.gguf',
    local_dir='$HOME/local-llm/gemma4'
)
"

# EmbeddingGemma 300M embedding model (~200MB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ggml-org/embeddinggemma-300M-qat-Q4_0-GGUF',
    filename='embeddinggemma-300M-qat-Q4_0.gguf',
    local_dir='$HOME/local-llm/gemma4'
)
"
```

### 3. Install Python dependencies

```bash
cd local-rag-agent-gemma4  # or wherever you cloned this repo
pip3 install -r requirements.txt
```

### 4. Add documents

Place your `.txt`, `.md`, or `.pdf` files in the `documents/` folder:

```bash
cp my-notes.md documents/
cp research-paper.pdf documents/
```

## Usage

```bash
# 1. Start the servers (loads models, takes ~15-20s)
./start_servers.sh

# 2. Ingest your documents (in a separate terminal)
python3 ingest.py

# 3. Start the agent
python3 main.py
```

Then ask questions:
```
You: What are the latest trends in green energy?

  [thinking] I should reflect first to check local knowledge...
  -> Calling skill: reflect({"query": "green energy trends"})
  [auto-chain] reflect → investigate (status: no_local_knowledge)

  [orchestrator] === DECOMPOSE ===
  [orchestrator] Planned 4 research threads:
    A: Solar energy cost and deployment trends
    B: Wind energy offshore developments
    C: Battery storage breakthroughs
    D: Government policy and subsidies

  [orchestrator] === DISPATCH ===
  ...sub-agents researching in parallel...

Agent: Based on my research across multiple sources...
```

Commands: `/clear` to reset conversation, `/quit` to exit.

## Custom paths

If your llama.cpp or models are in different locations, set environment variables:

```bash
export LLAMA_DIR=/path/to/llama.cpp
export CHAT_MODEL_PATH=/path/to/gemma-4-26B-A4B-it-Q4_K_M.gguf
export EMBED_MODEL_PATH=/path/to/embeddinggemma-300M-qat-Q4_0.gguf
```

These work for both `start_servers.sh` and the Python scripts.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design details.

```
┌─────────────┐     ┌──────────────────────────────────┐
│  You (REPL) │────>│  Agent (agent.py)                │
└─────────────┘     │  - Gemma 4 thinking mode         │
                    │  - skill dispatch                 │
                    │  - auto-chain (reflect→investigate)│
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

## Skills

| Skill | What it does |
|-------|-------------|
| `reflect` | Searches past conversations and indexed documents, then uses the LLM to assess if the question can be answered locally |
| `read_document` | Reads the full text of a specific document from a prior reflect result |
| `investigate` | Runs web research. `depth="quick"` for single-thread lookups, `depth="deep"` for multi-agent research with parallel threads, evaluation, and synthesis |
| `index_site` | Crawls a website via BFS, saves pages locally, and indexes them for future reflect searches |
| `generate_report` | Creates a structured multi-section report from prior investigate results (outline → section drafting with sliding context → executive summary) |
| `get_time` | Returns the current date and time |

## How it works

1. You ask a question
2. Gemma 4 thinks internally (via `<|channel>thought` tokens) and picks a skill
3. **Reflect**: checks local documents and past conversations
4. If knowledge is insufficient, **auto-chains** to investigate
5. **Investigate** (deep): orchestrator decomposes the question into threads, dispatches sub-agents that search the web, evaluates completeness, and synthesizes findings
6. If a report is requested, **generate_report** creates a structured document from the research
7. Thinking blocks are stripped from conversation history to prevent reasoning debt

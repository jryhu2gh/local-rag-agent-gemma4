# Local RAG Agent with Gemma 4

A tool-calling research agent that runs **entirely locally** on your machine. It searches a local document collection using hybrid search (BM25 keyword matching + semantic embeddings) and answers questions with cited sources.

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
You: What is Rust's approach to memory safety?

  -> Calling tool: search_documents({"query": "Rust memory safety"})

Agent: Rust achieves memory safety without garbage collection through its
ownership system with compile-time borrow checking...
(Source: sample-rust.md)
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
┌─────────────┐     ┌──────────────────────────┐
│  You (REPL) │────>│  Python Agent (agent.py)  │
└─────────────┘     │  - system prompt          │
                    │  - agent loop              │
                    │  - tool dispatch           │
                    └────┬──────────┬────────────┘
                         │          │
              tool calls │          │ embeddings
                         v          v
               ┌──────────┐  ┌──────────┐
               │ llama-   │  │ llama-   │
               │ server   │  │ server   │
               │ :8080    │  │ :8081    │
               │ Gemma 4  │  │ EmbGemma │
               │ (chat)   │  │ (embed)  │
               └──────────┘  └──────────┘
```

## How it works

1. You ask a question
2. The agent sends it to Gemma 4 with tool definitions
3. Gemma decides to call `search_documents` → hybrid search (BM25 + embeddings merged via Reciprocal Rank Fusion)
4. Results are fed back to Gemma
5. Gemma may call `read_document` for more context or `summarize_text` for long documents
6. Gemma synthesizes a final answer with citations

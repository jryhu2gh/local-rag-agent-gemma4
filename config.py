import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Server endpoints (local llama-server instances)
CHAT_BASE_URL = os.environ.get("CHAT_BASE_URL", "http://localhost:8080/v1")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://localhost:8081/v1")
CHAT_MODEL = "gemma4"
EMBED_MODEL = "embeddinggemma"

# Paths
DOCUMENTS_DIR = Path(os.environ.get("DOCUMENTS_DIR", BASE_DIR / "documents"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", BASE_DIR / "index_data"))

# llama.cpp paths (override via env vars if installed elsewhere)
LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", Path.home() / "local-llm" / "llama.cpp"))
LLAMA_SERVER = LLAMA_DIR / "build" / "bin" / "llama-server"
CHAT_MODEL_PATH = Path(os.environ.get("CHAT_MODEL_PATH", Path.home() / "local-llm" / "gemma4" / "gemma-4-26B-A4B-it-Q4_K_M.gguf"))
EMBED_MODEL_PATH = Path(os.environ.get("EMBED_MODEL_PATH", Path.home() / "local-llm" / "gemma4" / "embeddinggemma-300M-qat-Q4_0.gguf"))
CHAT_TEMPLATE_PATH = LLAMA_DIR / "models" / "templates" / "gemma4.jinja"

# RAG parameters
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP = 64        # overlap between adjacent chunks
TOP_K = 5                 # number of chunks to retrieve

# Hybrid search parameters
BM25_WEIGHT = 0.5         # weight for BM25 in hybrid search (0-1)
EMBED_WEIGHT = 0.5        # weight for embedding in hybrid search (0-1)
RRF_K = 60                # RRF constant (standard default)

# Agent parameters
MAX_TURNS = 10            # max tool-call iterations before forcing a final answer
TEMPERATURE = 0.7
MAX_TOKENS = 2048
CONTEXT_SIZE = 8192       # llama-server context window
MAX_CRAWL_PAGES = 50      # hard cap on pages per crawl_website call
MAX_SEARCH_RESULTS = 5    # default number of web search results

# History
HISTORY_DIR = Path(os.environ.get("HISTORY_DIR", BASE_DIR / "history"))
HISTORY_INDEX_DIR = Path(os.environ.get("HISTORY_INDEX_DIR", BASE_DIR / "history_index"))

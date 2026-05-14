#!/usr/bin/python3
"""Interactive REPL for the RAG agent.

Usage:
    1. Start servers:  ./start_servers.sh
    2. Ingest docs:    python3 ingest.py
    3. Run agent:      python3 main.py
"""

import sys
import time
from pathlib import Path

# Ensure the agent directory is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

import agent
import research_index
from config import INDEX_DIR, CHAT_BASE_URL, EMBED_BASE_URL


def _wait_for_server(name: str, base_url: str, timeout: int = 120):
    """Wait for a llama-server to finish loading its model."""
    import urllib.request
    import urllib.error
    url = base_url.rstrip("/") + "/models"
    print(f"  Waiting for {name} ({url})...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(" ready.")
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print(f"\n  WARNING: {name} not ready after {timeout}s — continuing anyway.")


def main():
    # Check that indexes exist
    if not (INDEX_DIR / "bm25.json").exists():
        print("No index found. Run 'python3 ingest.py' first to ingest documents.")
        sys.exit(1)

    # Wait for LLM servers to be ready
    _wait_for_server("Chat server", CHAT_BASE_URL)
    _wait_for_server("Embedding server", EMBED_BASE_URL)

    print("RAG Agent ready. Type your questions below.")
    print("Commands:  /clear = reset conversation,  /quit = exit")
    print("-" * 50)

    history: list[dict] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye!")
            break

        if user_input == "/clear":
            history = []
            research_index.clear()
            print("[Conversation and session research cleared]")
            continue

        response, history = agent.run(user_input, history)
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    main()

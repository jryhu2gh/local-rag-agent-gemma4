#!/usr/bin/python3
"""Interactive REPL for the RAG agent.

Usage:
    1. Start servers:  ./start_servers.sh
    2. Ingest docs:    python3 ingest.py
    3. Run agent:      python3 main.py
"""

import sys
from pathlib import Path

# Ensure the agent directory is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

import agent
from config import INDEX_DIR


def main():
    # Check that indexes exist
    if not (INDEX_DIR / "bm25.json").exists():
        print("No index found. Run 'python3 ingest.py' first to ingest documents.")
        sys.exit(1)

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
            print("[Conversation cleared]")
            continue

        response, history = agent.run(user_input, history)
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    main()

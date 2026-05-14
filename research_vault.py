"""Research vault: structured storage for raw research data.

During investigation, sub-agents save raw tool results (web pages, search
results) to the vault with auto-assigned keys (DATA_01, DATA_02, ...).
The report generator later pulls relevant entries by thread ID.

The vault is both:
- In-memory (module-level ``current_vault`` for same-session access)
- On disk (JSON files for persistence and review)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path


class ResearchVault:
    """Keyed store for raw research data from sub-agent tool calls."""

    def __init__(self, topic: str):
        self.topic = topic
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.entries: dict[str, dict] = {}
        self.thread_topics: dict[str, str] = {}
        self.thread_summaries: dict[str, str] = {}
        self.synthesis: str = ""
        self._counter = 0
        self._lock = threading.Lock()

    def add(self, thread_id: str, tool_name: str, tool_args: dict, result: str) -> str:
        """Store a raw tool result. Returns the assigned key (DATA_XX)."""
        with self._lock:
            self._counter += 1
            key = f"DATA_{self._counter:02d}"
            self.entries[key] = {
                "thread": thread_id,
                "tool": tool_name,
                "args": tool_args,
                "content": result,
            }
        return key

    def set_thread_topic(self, thread_id: str, topic: str):
        """Store the research topic for a thread."""
        self.thread_topics[thread_id] = topic

    def set_thread_summary(self, thread_id: str, summary: str):
        """Store a sub-agent's final summary."""
        self.thread_summaries[thread_id] = summary

    def set_synthesis(self, synthesis: str):
        """Store the orchestrator's final synthesis."""
        self.synthesis = synthesis

    def get(self, key: str) -> dict | None:
        """Get a vault entry by key."""
        return self.entries.get(key)

    def keys_for_threads(self, thread_ids: list[str]) -> list[str]:
        """Get all vault keys belonging to any of the given threads."""
        ids = set(thread_ids)
        return [k for k, v in self.entries.items() if v["thread"] in ids]

    def index(self) -> list[dict]:
        """Return a compact index of all entries (key, thread, tool, query)."""
        items = []
        for key in sorted(self.entries):
            entry = self.entries[key]
            items.append({
                "key": key,
                "thread": entry["thread"],
                "tool": entry["tool"],
                "query": entry["args"].get("query", entry["args"].get("url", "")),
            })
        return items

    def format_index(self) -> str:
        """Format the vault index as readable text for LLM prompts."""
        lines = []
        for tid, topic in self.thread_topics.items():
            keys = self.keys_for_threads([tid])
            key_list = ", ".join(keys) if keys else "(no data)"
            lines.append(f"  Thread {tid}: {topic} [{key_list}]")
        return "\n".join(lines)

    def content_for_threads(self, thread_ids: list[str], limit: int = 0) -> str:
        """Concatenate raw content from vault entries for given threads.

        Each entry is formatted with a header showing its key, tool, and query.
        If limit > 0, truncate total content to that many characters.
        """
        keys = self.keys_for_threads(thread_ids)
        parts = []
        for key in sorted(keys):
            entry = self.entries[key]
            query = entry["args"].get("query", entry["args"].get("url", ""))
            header = f"### {key} ({entry['tool']}: {query})"
            parts.append(f"{header}\n\n{entry['content']}")
        text = "\n\n---\n\n".join(parts)
        if limit and len(text) > limit:
            text = text[:limit] + f"\n\n[... truncated, {len(text)} chars total]"
        return text

    def summaries_for_threads(self, thread_ids: list[str]) -> str:
        """Concatenate thread summaries for given threads."""
        parts = []
        for tid in thread_ids:
            if tid in self.thread_summaries:
                topic = self.thread_topics.get(tid, tid)
                parts.append(f"### Thread {tid}: {topic}\n\n{self.thread_summaries[tid]}")
        return "\n\n---\n\n".join(parts)

    def save(self, directory: Path):
        """Persist vault to disk."""
        directory.mkdir(parents=True, exist_ok=True)

        metadata = {
            "topic": self.topic,
            "timestamp": self.timestamp,
            "num_entries": len(self.entries),
            "thread_topics": self.thread_topics,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        (directory / "vault.json").write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False)
        )
        for tid, summary in self.thread_summaries.items():
            (directory / f"thread_{tid}_summary.md").write_text(summary)
        if self.synthesis:
            (directory / "synthesis.md").write_text(self.synthesis)

    @classmethod
    def load(cls, directory: Path) -> ResearchVault:
        """Load a vault from disk."""
        metadata = json.loads((directory / "metadata.json").read_text())
        vault = cls(topic=metadata["topic"])
        vault.timestamp = metadata["timestamp"]
        vault.thread_topics = metadata.get("thread_topics", {})

        vault_path = directory / "vault.json"
        if vault_path.exists():
            vault.entries = json.loads(vault_path.read_text())
            vault._counter = len(vault.entries)

        for tid in vault.thread_topics:
            summary_path = directory / f"thread_{tid}_summary.md"
            if summary_path.exists():
                vault.thread_summaries[tid] = summary_path.read_text()

        synthesis_path = directory / "synthesis.md"
        if synthesis_path.exists():
            vault.synthesis = synthesis_path.read_text()

        return vault


# Module-level current vault — set by orchestrator, read by report generator
current_vault: ResearchVault | None = None

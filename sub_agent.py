"""Sub-agent: a mini agent that researches a single topic.

Each sub-agent has its own conversation context and access to toolkit
functions. It runs a tool-calling loop until the LLM produces a text
response (the summary), then returns it to the orchestrator.
"""

import json
import time

import llm
import toolkit
from config import MAX_SUB_AGENT_TURNS

# ---------------------------------------------------------------------------
# Tool definitions for sub-agents (toolkit functions as OpenAI tools)
# ---------------------------------------------------------------------------

SUB_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Quick web search. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_research",
            "description": "Search the web and read top results in full. Returns full page content from multiple sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to research"},
                    "max_sources": {"type": "integer", "description": "Number of pages to read (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Visit a specific URL and return its text content plus links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local",
            "description": "Search local indexed documents and past conversations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool dispatch for sub-agents
# ---------------------------------------------------------------------------


def _dispatch_tool(name: str, args: dict) -> str:
    """Execute a toolkit function by name. Returns JSON string."""
    if name == "web_search":
        return toolkit.web_search(args.get("query", ""), args.get("max_results", 5))
    elif name == "deep_research":
        result = toolkit.deep_research(args.get("query", ""), args.get("max_sources", 3))
        return json.dumps(result, indent=2)
    elif name == "browse":
        return toolkit.browse_website(args.get("url", ""))
    elif name == "search_local":
        return toolkit.search_documents(args.get("query", ""))
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Sub-agent system prompt
# ---------------------------------------------------------------------------


def _build_prompt(topic: str, main_query: str) -> str:
    return f"""\
You are a research sub-agent investigating a specific topic as part of a larger research question.

Main question: {main_query}
Your assigned topic: {topic}

## Process
1. Think about what specific data you need for this topic
2. Use your tools to gather information
3. After gathering, evaluate: is your data complete enough to thoroughly cover this topic for the main question?
4. If incomplete, gather more data. If complete, write your summary.

## Your Tools
- web_search: Quick web search — returns titles, URLs, and snippets
- deep_research: Search and read top web pages in full (use for detailed info)
- browse: Visit a specific URL to read its content
- search_local: Search locally indexed documents and past conversations

## When Done
Stop calling tools and respond with a structured summary:

**Key Findings:**
- [bullet points of what you found]

**Data Collected:**
- [list what specific data points you have, so completeness can be verified]

**Sources:**
- [URLs or document names]

Keep the summary concise (under 500 words). Focus on facts and data, not opinions.\
"""


# ---------------------------------------------------------------------------
# SubAgent class
# ---------------------------------------------------------------------------


class SubAgent:
    """A mini agent that researches a single topic."""

    def __init__(self, agent_id: str, topic: str, main_query: str):
        self.agent_id = agent_id
        self.topic = topic
        self.main_query = main_query
        self.messages = [{"role": "system", "content": _build_prompt(topic, main_query)}]
        self.summary = None

    def run(self) -> str:
        """Execute initial research. Returns structured summary."""
        self.messages.append({"role": "user", "content": f"Research this topic: {self.topic}"})
        self.summary = self._agent_loop()
        return self.summary

    def follow_up(self, question: str) -> str:
        """Handle a follow-up question from the orchestrator."""
        self.messages.append({"role": "user", "content": question})
        self.summary = self._agent_loop()
        return self.summary

    def _agent_loop(self) -> str:
        """Run tool-calling loop until text response (= summary)."""
        for turn in range(MAX_SUB_AGENT_TURNS):
            try:
                msg = llm.call(self.messages, tools=SUB_AGENT_TOOLS)
            except Exception as e:
                err = f"[LLM error: {e}]"
                print(f"  [sub-agent {self.agent_id}] {err}")
                return err

            # Text response = summary, we're done
            if not msg.tool_calls:
                text = msg.content or "[No findings]"
                self.messages.append({"role": "assistant", "content": text})
                print(f"  [sub-agent {self.agent_id}] Summary ready ({len(text)} chars)")
                return text

            # Handle tool calls
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id or f"call_{self.agent_id}_{turn}_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for i, tc in enumerate(msg.tool_calls)
            ]
            self.messages.append(assistant_msg)

            for i, tc in enumerate(msg.tool_calls):
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                t_start = time.time()
                print(f"  [sub-agent {self.agent_id}] → {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:80]})")
                result = _dispatch_tool(tool_name, tool_args)
                elapsed = time.time() - t_start
                print(f"  [sub-agent {self.agent_id}] ← {tool_name} ({len(result)} chars, {elapsed:.1f}s)")

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id or f"call_{self.agent_id}_{turn}_{i}",
                    "content": result,
                }
                self.messages.append(tool_msg)

        print(f"  [sub-agent {self.agent_id}] Max turns ({MAX_SUB_AGENT_TURNS}) reached")
        return "[Max turns reached — partial findings above]"

"""Shared LLM client for all agents (main agent, sub-agents, orchestrator).

Supports Gemma 4 thinking mode: when enabled, the model wraps internal
reasoning in <|channel>thought ... <channel|> tokens.  parse_thinking()
separates these blocks from the visible content.
"""

from __future__ import annotations

import re

from openai import OpenAI
from config import CHAT_BASE_URL, CHAT_MODEL, TEMPERATURE, MAX_TOKENS

_client = OpenAI(base_url=CHAT_BASE_URL, api_key="not-needed")

_THINK_RE = re.compile(r"<\|channel>(?:thought)?\s*(.*?)\s*<channel\|>", re.DOTALL)


def parse_thinking(msg) -> tuple[str, str | None]:
    """Extract thinking from a message object.

    The server returns thinking in the `reasoning_content` field,
    separate from `content`. Also falls back to parsing <|channel>
    blocks from content if reasoning_content is absent.

    Returns (clean_content, thinking_or_None).
    """
    content = msg.content or ""
    thinking = getattr(msg, "reasoning_content", None)
    if thinking:
        thinking = thinking.strip() or None
        return content, thinking
    # Fallback: parse <|channel>thought...<channel|> from content
    if content:
        matches = _THINK_RE.findall(content)
        if matches:
            thinking = "\n".join(m.strip() for m in matches if m.strip()) or None
            clean = _THINK_RE.sub("", content).strip()
            return clean, thinking
    return content, None


def call(messages, tools=None, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
         enable_thinking=True):
    """Make a single LLM call. Returns the message object.

    Args:
        messages: List of message dicts (system, user, assistant, tool).
        tools: Optional list of tool definitions (OpenAI format).
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        enable_thinking: Enable Gemma 4 thinking mode (default True).

    Returns:
        The assistant's message object (has .content and .tool_calls).
    """
    kwargs = dict(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
    kwargs["extra_body"] = {"enable_thinking": enable_thinking}
    response = _client.chat.completions.create(**kwargs)
    return response.choices[0].message

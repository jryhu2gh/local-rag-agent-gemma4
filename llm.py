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


def parse_thinking(text: str) -> tuple[str, str | None]:
    """Separate thinking blocks from visible content.

    Handles multiple thinking blocks (matches the template's strip_thinking
    which iterates over all <|channel>...<channel|> pairs).

    Returns (clean_content, concatenated_thinking_or_None).
    """
    if not text:
        return text, None
    matches = _THINK_RE.findall(text)
    if matches:
        thinking = "\n".join(m.strip() for m in matches if m.strip()) or None
        clean = _THINK_RE.sub("", text).strip()
        return clean, thinking
    return text, None


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

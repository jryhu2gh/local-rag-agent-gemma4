"""Shared LLM client for all agents (main agent, sub-agents, orchestrator)."""

from openai import OpenAI
from config import CHAT_BASE_URL, CHAT_MODEL, TEMPERATURE, MAX_TOKENS

_client = OpenAI(base_url=CHAT_BASE_URL, api_key="not-needed")


def call(messages, tools=None, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """Make a single LLM call. Returns the message object.

    Args:
        messages: List of message dicts (system, user, assistant, tool).
        tools: Optional list of tool definitions (OpenAI format).
        temperature: Sampling temperature.
        max_tokens: Max response tokens.

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
    response = _client.chat.completions.create(**kwargs)
    return response.choices[0].message

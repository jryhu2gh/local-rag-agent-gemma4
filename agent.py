"""Core agent loop: send messages, handle tool calls, iterate."""

import json

from openai import OpenAI
from config import CHAT_BASE_URL, CHAT_MODEL, MAX_TURNS, TEMPERATURE, MAX_TOKENS
from prompts import SYSTEM_PROMPT
from tools import TOOL_DEFINITIONS, execute_tool

_client = OpenAI(base_url=CHAT_BASE_URL, api_key="not-needed")


def run(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Run the agent loop for a single user query.

    Args:
        user_message: The user's question.
        history: Conversation history (list of message dicts). Modified in place.

    Returns:
        (final_response_text, updated_history)
    """
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    for turn in range(MAX_TURNS):
        try:
            response = _client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except Exception as e:
            error_msg = f"[Error calling LLM: {e}]"
            history.append({"role": "assistant", "content": error_msg})
            return error_msg, history

        choice = response.choices[0]
        msg = choice.message

        # If no tool calls, we have the final response
        if not msg.tool_calls:
            text = msg.content or "[No response]"
            history.append({"role": "assistant", "content": text})
            return text, history

        # Process tool calls
        # Append the assistant message with tool calls to history
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id or f"call_{turn}_{i}",
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for i, tc in enumerate(msg.tool_calls)
        ]
        history.append(assistant_msg)
        messages.append(assistant_msg)

        for i, tc in enumerate(msg.tool_calls):
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
                print(f"  [Warning: could not parse args for {tool_name}]")

            print(f"  -> Calling tool: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})")
            result = execute_tool(tool_name, tool_args)

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id or f"call_{turn}_{i}",
                "content": result,
            }
            history.append(tool_msg)
            messages.append(tool_msg)

    # Hit MAX_TURNS — force a final answer
    fallback = "[Reached maximum tool call iterations. Here is what I found so far based on the tool results above.]"
    history.append({"role": "assistant", "content": fallback})
    return fallback, history

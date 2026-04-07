"""Core agent loop: send messages, handle tool calls, iterate."""

import json

import llm
from config import MAX_TURNS
from prompts import SYSTEM_PROMPT
from skills import SKILL_DEFINITIONS, execute_skill
import history


def run(user_message: str, conv_history: list[dict]) -> tuple[str, list[dict]]:
    """Run the agent loop for a single user query.

    Args:
        user_message: The user's question.
        conv_history: Conversation history (list of message dicts). Modified in place.

    Returns:
        (final_response_text, updated_history)
    """
    conv_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conv_history

    for turn in range(MAX_TURNS):
        try:
            msg = llm.call(messages, tools=SKILL_DEFINITIONS)
        except Exception as e:
            error_msg = f"[Error calling LLM: {e}]"
            conv_history.append({"role": "assistant", "content": error_msg})
            return error_msg, conv_history

        # If no tool calls, we have the final response
        if not msg.tool_calls:
            text = msg.content or "[No response]"
            conv_history.append({"role": "assistant", "content": text})
            # Persist Q&A for cross-session recall
            try:
                history.save_turn(user_message, text)
            except Exception:
                pass  # don't break the agent if history save fails
            return text, conv_history

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
        conv_history.append(assistant_msg)
        messages.append(assistant_msg)

        for i, tc in enumerate(msg.tool_calls):
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
                print(f"  [Warning: could not parse args for {tool_name}]")

            print(f"  -> Calling skill: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})")
            result = execute_skill(tool_name, tool_args)

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id or f"call_{turn}_{i}",
                "content": result,
            }
            conv_history.append(tool_msg)
            messages.append(tool_msg)

    # Hit MAX_TURNS — force a final answer
    fallback = "[Reached maximum tool call iterations. Here is what I found so far based on the tool results above.]"
    conv_history.append({"role": "assistant", "content": fallback})
    return fallback, conv_history

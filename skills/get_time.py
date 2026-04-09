"""Get time skill — return the current date and time."""

from datetime import datetime

DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current real-world date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}


def execute() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")

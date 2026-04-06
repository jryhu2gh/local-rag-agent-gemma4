"""Skill registry and dispatch.

Skills are the LLM-facing interface. Each skill maps to a high-level intent
and orchestrates internal toolkit functions deterministically.
"""

import json

from skills.research import DEFINITION as research_def, execute as research_exec
from skills.read_doc import DEFINITION as read_doc_def, execute as read_doc_exec
from skills.browse import DEFINITION as browse_def, execute as browse_exec
from skills.index_site import DEFINITION as index_site_def, execute as index_site_exec

SKILL_DEFINITIONS = [research_def, read_doc_def, browse_def, index_site_def]

_DISPATCH = {
    "research": research_exec,
    "read_document": read_doc_exec,
    "browse": browse_exec,
    "index_site": index_site_exec,
}


def execute_skill(name: str, arguments: dict) -> str:
    """Dispatch a skill call to the appropriate handler."""
    handler = _DISPATCH.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown skill: {name}"})
    try:
        return handler(**arguments)
    except Exception as e:
        return json.dumps({"error": f"Skill '{name}' failed: {str(e)}"})

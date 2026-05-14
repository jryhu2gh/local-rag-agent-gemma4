"""Generate report skill — create structured multi-section reports."""

import report_generator

DEFINITION = {
    "type": "function",
    "function": {
        "name": "generate_report",
        "description": (
            "Generate a detailed, structured report on a topic. Uses raw research "
            "data from the most recent investigate call (stored in the research vault) "
            "to create a multi-section report with executive summary, analysis sections, "
            "and conclusion. Each section is written using the original source material, "
            "not just summaries. Call this AFTER investigate has gathered research data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The report topic or question to address",
                },
                "research_data": {
                    "type": "string",
                    "description": (
                        "Optional fallback research text. Not needed if investigate "
                        "was called earlier — the vault provides raw data automatically."
                    ),
                },
            },
            "required": ["topic"],
        },
    },
}


def execute(topic: str, research_data: str = "") -> str:
    """Generate a structured report."""
    return report_generator.generate_report(topic, research_data)

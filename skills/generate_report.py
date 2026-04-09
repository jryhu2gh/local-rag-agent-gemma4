"""Generate report skill — create structured multi-section reports."""

import report_generator

DEFINITION = {
    "type": "function",
    "function": {
        "name": "generate_report",
        "description": (
            "Generate a detailed, structured report on a topic. Uses research data "
            "from a prior investigate call to create a multi-section report with "
            "executive summary, analysis sections, and conclusion. "
            "Call this AFTER investigate has gathered the research data."
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
                    "description": "Research findings from a prior investigate call",
                },
            },
            "required": ["topic", "research_data"],
        },
    },
}


def execute(topic: str, research_data: str = "") -> str:
    """Generate a structured report."""
    return report_generator.generate_report(topic, research_data)

"""Web search skill — search the web using DuckDuckGo."""

import toolkit

DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for information using a search engine. "
            "Returns a list of results with titles, URLs, and snippets. "
            "Use this when local knowledge is insufficient and you need to find information online. "
            "You can follow up by calling browse on a result URL for full page content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (like what you'd type into Google)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10)",
                },
            },
            "required": ["query"],
        },
    },
}


def execute(query: str, max_results: int = 5) -> str:
    """Run a web search."""
    max_results = min(max_results, 10)
    return toolkit.web_search(query, max_results)

"""Browse skill — fetch a web page and return its content and links."""

import toolkit

DEFINITION = {
    "type": "function",
    "function": {
        "name": "browse",
        "description": (
            "Visit a web URL and return its text content plus a list of links found on the page. "
            "Use this when you need information from a specific web page. "
            "You can follow returned links by calling browse again on a link URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch (e.g. https://hollowknight.wiki/w/The_Knight)",
                }
            },
            "required": ["url"],
        },
    },
}


def execute(url: str) -> str:
    """Browse a web page."""
    return toolkit.browse_website(url)

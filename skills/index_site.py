"""Index site skill — crawl a website and index all pages for future research."""

import toolkit

DEFINITION = {
    "type": "function",
    "function": {
        "name": "index_site",
        "description": (
            "Crawl a website starting from a URL, save all pages locally, and index them "
            "so they become searchable via the research skill. Use this when you need broad "
            "coverage of a website — e.g. indexing a wiki so all its pages are searchable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The starting URL to crawl from",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum number of pages to crawl (default 20, max 50)",
                },
            },
            "required": ["url"],
        },
    },
}


def execute(url: str, max_pages: int = 20) -> str:
    """Crawl and index a website."""
    return toolkit.crawl_website(url, max_pages)

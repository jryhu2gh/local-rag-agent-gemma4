"""Tool definitions and execution dispatch for the RAG agent."""

import json
from pathlib import Path

from config import DOCUMENTS_DIR, INDEX_DIR, MAX_CRAWL_PAGES
from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embedder import embed
from rag.hybrid import hybrid_search
from rag.index import VectorIndex
import history as history_module

# OpenAI-format tool definitions (passed to the API)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "Search past conversations for previously discussed topics. "
                "Use this FIRST before other search tools — it may have the answer from a prior session, "
                "or give hints about which sources or URLs were useful before."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant past conversations",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the document collection for passages relevant to a query. "
                "Uses hybrid search (keyword + semantic) to find the best matching chunks. "
                "Returns the top matching text chunks with scores and source info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant document passages",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read the full content of a specific document by its ID (relative file path).",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The document identifier (relative path) returned by search_documents",
                    }
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize a long piece of text into a concise form. Extracts the most important sentences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarize",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_website",
            "description": (
                "Fetch a web page and return its text content plus a list of links found on the page. "
                "Use this when the user provides a URL or when you need information from the web. "
                "You can follow returned links by calling browse_website again on a specific link URL."
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
    },
    {
        "type": "function",
        "function": {
            "name": "crawl_website",
            "description": (
                "Crawl a website starting from a URL, save all pages locally, and index them "
                "for future search_documents queries. Use this when you need broad knowledge from "
                "a website — e.g. indexing a wiki so all its pages become searchable. "
                "This is slower but makes all content available via search_documents afterwards."
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
    },
]


def _load_indexes() -> tuple[BM25Index, VectorIndex]:
    bm25 = BM25Index.load(INDEX_DIR / "bm25.json")
    vector = VectorIndex.load(INDEX_DIR)
    return bm25, vector


def _handle_search(query: str) -> str:
    bm25, vector = _load_indexes()
    results = hybrid_search(query, bm25, vector)
    if not results:
        return json.dumps({"results": [], "message": "No matching documents found."})

    formatted = []
    for r in results:
        formatted.append({
            "doc_id": r["doc_id"],
            "source_file": r["source_file"],
            "chunk_index": r["chunk_index"],
            "score": round(r["score"], 4),
            "text": r["text"][:500],  # truncate long chunks for the LLM
        })
    return json.dumps({"results": formatted}, indent=2)


def _handle_read(doc_id: str) -> str:
    file_path = DOCUMENTS_DIR / doc_id
    if not file_path.exists():
        return json.dumps({"error": f"Document not found: {doc_id}"})

    if file_path.suffix == ".pdf":
        import fitz
        doc = fitz.open(file_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")

    # Cap at 4000 chars to avoid flooding the context
    if len(text) > 4000:
        text = text[:4000] + f"\n\n[... truncated, {len(text)} chars total]"

    return json.dumps({"doc_id": doc_id, "content": text})


def _handle_summarize(text: str) -> str:
    """Extractive summarization: return first N sentences."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    max_sentences = min(10, len(sentences))
    summary = " ".join(sentences[:max_sentences])
    return json.dumps({"summary": summary, "sentence_count": max_sentences})


def _fetch_page(url: str) -> tuple:
    """Fetch a URL and return (BeautifulSoup, base_domain). Shared by browse and crawl."""
    import urllib.request
    from urllib.parse import urlparse
    from bs4 import BeautifulSoup

    req = urllib.request.Request(url, headers={"User-Agent": "RAGAgent/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")
        final_url = resp.url  # handle redirects

    soup = BeautifulSoup(html, "html.parser")
    domain = urlparse(final_url).netloc
    return soup, domain


def _extract_text(soup) -> str:
    """Extract clean text from a BeautifulSoup object."""
    # Work on a copy to avoid mutating the original
    from copy import copy
    soup = copy(soup)

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_links(soup, base_url: str, base_domain: str) -> list[dict]:
    """Extract same-domain links from a page."""
    from urllib.parse import urljoin, urlparse

    seen = set()
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = urljoin(base_url, href)

        # Filter: same domain, no anchors-only, no media files
        parsed = urlparse(full_url)
        if parsed.netloc != base_domain:
            continue
        clean_url = parsed._replace(fragment="").geturl()
        if clean_url in seen or clean_url == base_url:
            continue
        if any(clean_url.lower().endswith(ext) for ext in (".png", ".jpg", ".gif", ".svg", ".css", ".js")):
            continue

        seen.add(clean_url)
        link_text = a.get_text(strip=True)[:80]
        if link_text:
            links.append({"url": clean_url, "text": link_text})

    return links[:30]


def _handle_browse(url: str) -> str:
    """Fetch a web page and return its text content plus links."""
    soup, domain = _fetch_page(url)
    text = _extract_text(soup)
    links = _extract_links(soup, url, domain)

    if len(text) > 4000:
        text = text[:4000] + f"\n\n[... truncated, {len(text)} chars total]"

    return json.dumps({"url": url, "content": text, "links": links}, indent=2)


def _add_to_indexes(text: str, source_file: str, doc_id: str):
    """Chunk text and add to existing BM25 + vector indexes incrementally."""
    chunks = chunk_text(text, source_file=source_file, doc_id=doc_id)
    if not chunks:
        return 0

    # Load existing indexes (or create new ones if they don't exist)
    bm25_path = INDEX_DIR / "bm25.json"
    if bm25_path.exists():
        bm25 = BM25Index.load(bm25_path)
    else:
        bm25 = BM25Index()

    vector = VectorIndex.load(INDEX_DIR)

    # Add to BM25
    for chunk in chunks:
        bm25.add(chunk["text"], metadata=chunk)

    # Add to vector index
    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)
    vector.add_batch(embeddings, chunks)

    # Save
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    bm25.save(bm25_path)
    vector.save(INDEX_DIR)

    return len(chunks)


def _handle_crawl(url: str, max_pages: int = 20) -> str:
    """Crawl a website via BFS, save pages locally, and index them."""
    from urllib.parse import urlparse
    from collections import deque
    import time

    max_pages = min(max_pages, MAX_CRAWL_PAGES)
    domain = urlparse(url).netloc
    save_dir = DOCUMENTS_DIR / "web" / domain
    save_dir.mkdir(parents=True, exist_ok=True)

    visited = set()
    queue = deque([url])
    total_chunks = 0
    pages_crawled = 0

    while queue and pages_crawled < max_pages:
        current_url = queue.popleft()
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            soup, _ = _fetch_page(current_url)
        except Exception:
            continue

        text = _extract_text(soup)
        if not text.strip():
            continue

        pages_crawled += 1

        # Save page as text file
        safe_name = urlparse(current_url).path.strip("/").replace("/", "_") or "index"
        safe_name = safe_name[:100] + ".txt"
        file_path = save_dir / safe_name
        file_path.write_text(text, encoding="utf-8")

        # Index incrementally
        doc_id = f"web/{domain}/{safe_name}"
        n_chunks = _add_to_indexes(text, source_file=doc_id, doc_id=doc_id)
        total_chunks += n_chunks

        # Extract links for BFS
        links = _extract_links(soup, current_url, domain)
        for link in links:
            if link["url"] not in visited:
                queue.append(link["url"])

        # Be polite — small delay between requests
        time.sleep(0.5)

    return json.dumps({
        "pages_crawled": pages_crawled,
        "total_chunks_indexed": total_chunks,
        "domain": domain,
        "save_dir": str(save_dir),
        "message": f"Crawled {pages_crawled} pages, indexed {total_chunks} chunks. Use search_documents to search this content.",
    })


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call to the appropriate handler."""
    try:
        if name == "search_history":
            results = history_module.search(arguments["query"])
            if not results:
                return json.dumps({"results": [], "message": "No relevant past conversations found."})
            return json.dumps({"results": results}, indent=2)
        elif name == "search_documents":
            return _handle_search(arguments["query"])
        elif name == "read_document":
            return _handle_read(arguments["doc_id"])
        elif name == "summarize_text":
            return _handle_summarize(arguments["text"])
        elif name == "browse_website":
            return _handle_browse(arguments["url"])
        elif name == "crawl_website":
            return _handle_crawl(arguments["url"], arguments.get("max_pages", 20))
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {str(e)}"})

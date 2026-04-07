"""Internal functions for document search, web browsing, and crawling.

These are NOT exposed to the LLM directly — skills orchestrate them.
"""

import json
from pathlib import Path

import time as _time

from config import (
    DOCUMENTS_DIR, INDEX_DIR, MAX_CRAWL_PAGES, MAX_SEARCH_RESULTS,
    MAX_DEEP_RESEARCH_SOURCES, DEEP_RESEARCH_CONTENT_LIMIT,
)
from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embedder import embed
from rag.hybrid import hybrid_search
from rag.index import VectorIndex


def _load_indexes() -> tuple[BM25Index, VectorIndex]:
    bm25 = BM25Index.load(INDEX_DIR / "bm25.json")
    vector = VectorIndex.load(INDEX_DIR)
    return bm25, vector


def search_documents(query: str) -> str:
    """Hybrid search over indexed documents. Returns JSON string."""
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
            "text": r["text"][:500],
        })
    return json.dumps({"results": formatted}, indent=2)


def read_document(doc_id: str) -> str:
    """Read full content of a document by its ID. Returns JSON string."""
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

    if len(text) > 4000:
        text = text[:4000] + f"\n\n[... truncated, {len(text)} chars total]"

    return json.dumps({"doc_id": doc_id, "content": text})


def _fetch_page(url: str) -> tuple:
    """Fetch a URL and return (BeautifulSoup, base_domain)."""
    import urllib.request
    from urllib.parse import urlparse
    from bs4 import BeautifulSoup

    req = urllib.request.Request(url, headers={"User-Agent": "RAGAgent/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="replace")
        final_url = resp.url

    soup = BeautifulSoup(html, "html.parser")
    domain = urlparse(final_url).netloc
    return soup, domain


def _extract_text(soup) -> str:
    """Extract clean text from a BeautifulSoup object."""
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


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """Search the web using DuckDuckGo. Returns JSON string with results."""
    from ddgs import DDGS

    # Retry up to 3 times — ddgs randomly picks TLS settings and
    # Python 3.9 sometimes fails on TLSv1.3
    last_err = None
    for attempt in range(3):
        try:
            results = DDGS().text(query, max_results=max_results)
            break
        except (ValueError, Exception) as e:
            last_err = e
            if attempt < 2:
                _time.sleep(0.2)
    else:
        return json.dumps({"query": query, "results": [], "error": str(last_err)})

    formatted = []
    for r in results:
        formatted.append({
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        })

    return json.dumps({"query": query, "results": formatted}, indent=2)


def deep_research(query: str, max_sources: int = 3) -> dict:
    """Search the web and read top results. Returns dict for composability.

    Unlike browse_website (4000 char limit), content is truncated to
    DEEP_RESEARCH_CONTENT_LIMIT (2000 chars) to keep combined output manageable.
    """
    max_sources = min(max_sources, MAX_DEEP_RESEARCH_SOURCES)
    t_start = _time.time()

    # Step 1: Web search
    print(f"  [deep_research] Searching: {query!r}")
    search_raw = web_search(query, max_results=max_sources + 2)
    search_results = json.loads(search_raw).get("results", [])
    print(f"  [deep_research] Got {len(search_results)} search results")

    # Step 2: Browse top results
    sources = []
    for i, result in enumerate(search_results[:max_sources]):
        url = result.get("url", "")
        if not url:
            continue
        print(f"  [deep_research] [{i+1}/{max_sources}] Fetching: {url[:80]}")
        t_page = _time.time()
        try:
            soup, domain = _fetch_page(url)
            content = _extract_text(soup)
            if len(content) > DEEP_RESEARCH_CONTENT_LIMIT:
                content = content[:DEEP_RESEARCH_CONTENT_LIMIT] + f"\n\n[... truncated, {len(content)} chars total]"
            elapsed = _time.time() - t_page
            print(f"  [deep_research] [{i+1}/{max_sources}] OK ({len(content)} chars, {elapsed:.1f}s)")
            sources.append({
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", ""),
                "content": content,
            })
        except Exception as e:
            elapsed = _time.time() - t_page
            print(f"  [deep_research] [{i+1}/{max_sources}] FAILED ({e}, {elapsed:.1f}s)")
            sources.append({
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", ""),
                "content": f"[Failed to load: {e}]",
            })

    total_chars = sum(len(s["content"]) for s in sources)
    total_time = _time.time() - t_start
    print(f"  [deep_research] Done: {len(sources)} sources, {total_chars} chars total, {total_time:.1f}s")

    return {"query": query, "sources": sources}


def browse_website(url: str) -> str:
    """Fetch a web page and return its text content plus links. Returns JSON string."""
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

    bm25_path = INDEX_DIR / "bm25.json"
    if bm25_path.exists():
        bm25 = BM25Index.load(bm25_path)
    else:
        bm25 = BM25Index()

    vector = VectorIndex.load(INDEX_DIR)

    for chunk in chunks:
        bm25.add(chunk["text"], metadata=chunk)

    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)
    vector.add_batch(embeddings, chunks)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    bm25.save(bm25_path)
    vector.save(INDEX_DIR)

    return len(chunks)


def crawl_website(url: str, max_pages: int = 20) -> str:
    """Crawl a website via BFS, save pages locally, and index them. Returns JSON string."""
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

        safe_name = urlparse(current_url).path.strip("/").replace("/", "_") or "index"
        safe_name = safe_name[:100] + ".txt"
        file_path = save_dir / safe_name
        file_path.write_text(text, encoding="utf-8")

        doc_id = f"web/{domain}/{safe_name}"
        n_chunks = _add_to_indexes(text, source_file=doc_id, doc_id=doc_id)
        total_chunks += n_chunks

        links = _extract_links(soup, current_url, domain)
        for link in links:
            if link["url"] not in visited:
                queue.append(link["url"])

        time.sleep(0.5)

    return json.dumps({
        "pages_crawled": pages_crawled,
        "total_chunks_indexed": total_chunks,
        "domain": domain,
        "save_dir": str(save_dir),
        "message": f"Crawled {pages_crawled} pages, indexed {total_chunks} chunks. Use research to search this content.",
    })

"""Ingest skill — add documents to the persistent knowledge base from conversation."""

import json
from pathlib import Path

import toolkit
from ingest import extract_text, SUPPORTED_EXTENSIONS

DEFINITION = {
    "type": "function",
    "function": {
        "name": "ingest",
        "description": (
            "Index a file or folder into the persistent knowledge base so its contents "
            "become searchable via reflect. Accepts a file path (.txt, .md, .pdf) or a "
            "folder path (scans recursively for supported files). Use this when the user "
            "wants to add documents, notes, or papers to the agent's knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file/folder path to ingest (~ is expanded)",
                }
            },
            "required": ["path"],
        },
    },
}


def execute(path: str) -> str:
    """Ingest a file or folder into the persistent document index."""
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        return json.dumps({"error": f"Path not found: {resolved}"})

    if resolved.is_file():
        files = [resolved]
        if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return json.dumps({
                "error": f"Unsupported file type: {resolved.suffix}",
                "supported": list(SUPPORTED_EXTENSIONS),
            })
    elif resolved.is_dir():
        files = [
            f for f in sorted(resolved.rglob("*"))
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            return json.dumps({
                "error": f"No supported files found in {resolved}",
                "supported": list(SUPPORTED_EXTENSIONS),
            })
        print(f"  [ingest] Found {len(files)} files in {resolved}")
    else:
        return json.dumps({"error": f"Path is neither a file nor a directory: {resolved}"})

    total_chunks = 0
    results = []

    for file_path in files:
        name = file_path.name
        print(f"  [ingest] Processing: {name}")
        try:
            text = extract_text(file_path)
            if not text.strip():
                print(f"  [ingest] Skipped (empty): {name}")
                results.append({"file": name, "status": "skipped", "reason": "empty"})
                continue
            doc_id = str(file_path)
            n_chunks = toolkit._add_to_indexes(text, source_file=name, doc_id=doc_id)
            print(f"  [ingest] Indexed {n_chunks} chunks: {name}")
            total_chunks += n_chunks
            results.append({"file": name, "status": "indexed", "chunks": n_chunks})
        except Exception as e:
            print(f"  [ingest] Failed: {name} — {e}")
            results.append({"file": name, "status": "failed", "error": str(e)})

    return json.dumps({
        "files_processed": len(results),
        "total_chunks": total_chunks,
        "details": results,
    }, indent=2)

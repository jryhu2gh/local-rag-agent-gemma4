"""Split documents into overlapping text chunks for indexing."""

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, source_file: str, doc_id: str) -> list[dict]:
    """Split text into overlapping chunks.

    Returns list of {"text", "source_file", "doc_id", "chunk_index"}.
    """
    # Normalize whitespace
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break first
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + CHUNK_SIZE // 2:
                end = para_break + 2
            else:
                # Fall back to sentence boundary
                for sep in (". ", ".\n", "? ", "!\n"):
                    sent_break = text.rfind(sep, start, end)
                    if sent_break > start + CHUNK_SIZE // 2:
                        end = sent_break + len(sep)
                        break

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append({
                "text": chunk_text_str,
                "source_file": source_file,
                "doc_id": doc_id,
                "chunk_index": chunk_index,
            })
            chunk_index += 1

        start = end - CHUNK_OVERLAP

    return chunks

import re


def chunk_text(text: str, max_chars: int = 500, overlap_chars: int = 50) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries.

    rag-mini-wikipedia passages are already short (a few sentences), so for
    most rows this returns a single chunk unchanged -- it only kicks in for
    longer documents (e.g. user-ingested text via /ingest).
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            overlap = current[-overlap_chars:] if overlap_chars else ""
            current = overlap + " " + sentence
        else:
            current = f"{current} {sentence}".strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks

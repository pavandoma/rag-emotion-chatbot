"""Load the knowledge base, chunk it, embed it, and store it in ChromaDB.

Source corpus: rag-datasets/rag-mini-wikipedia (passages split), plus any
extra .txt files dropped in data/raw/knowledge_base/ for live-demo ingestion.
"""

import json

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.utils.chunking import chunk_text


def load_wiki_passages() -> list[str]:
    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    return [row["passage"] for row in ds["passages"]]


def load_extra_docs() -> list[str]:
    kb_dir = RAW_DATA_DIR / "knowledge_base"
    docs = []
    if kb_dir.exists():
        for path in kb_dir.glob("*.txt"):
            docs.append(path.read_text(encoding="utf-8"))
    return docs


def build_chunks() -> list[str]:
    passages = load_wiki_passages() + load_extra_docs()
    chunks: list[str] = []
    for passage in passages:
        chunks.extend(chunk_text(passage))
    return chunks


def main() -> None:
    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks from knowledge base sources.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = PROCESSED_DATA_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(json.dumps({"id": f"chunk-{i}", "text": chunk}) + "\n")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    client.delete_collection(CHROMA_COLLECTION) if CHROMA_COLLECTION in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(CHROMA_COLLECTION)

    ids = [f"chunk-{i}" for i in range(len(chunks))]
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=chunks[start:end],
        )

    print(f"Added {collection.count()} chunks to Chroma collection '{CHROMA_COLLECTION}'.")


if __name__ == "__main__":
    main()

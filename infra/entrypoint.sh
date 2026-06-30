#!/bin/sh
set -e

python -c "
import chromadb
from src.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
names = [c.name for c in client.list_collections()]
exit(0 if CHROMA_COLLECTION in names else 1)
" || python -m src.ingest

exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

CHROMA_PERSIST_DIR = str(PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "kb_passages")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_ADAPTER_DIR = PROJECT_ROOT / "src" / "emotion" / "adapter"
EMOTION_BASE_MODEL = "distilbert-base-uncased"

RETRIEVAL_TOP_K = 4

TOXIC_BERT_MODEL = "unitary/toxic-bert"
SAFETY_TOXICITY_THRESHOLD = 0.5

API_PORT = int(os.getenv("API_PORT", "8000"))

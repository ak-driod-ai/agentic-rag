import os
from dotenv import load_dotenv

load_dotenv()


# =========================
# API KEYS
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# =========================
# QDRANT
# =========================

QDRANT_URL = os.getenv("QDRANT_URL")

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "novaretail_it"
)


# =========================
# GROQ
# =========================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# =========================
# EMBEDDINGS
# =========================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 produces 384-dimensional vectors
VECTOR_SIZE = 384


# =========================
# RAG SETTINGS
# =========================

TOP_K = 4

# Maximum query rewrite attempts
MAX_RETRIES = 1
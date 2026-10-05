import os
import warnings

# Suppress HuggingFace and symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str):
    model = get_embedding_model()
    vector = model.encode(
        text,
        normalize_embeddings=True
    )
    return vector.tolist()
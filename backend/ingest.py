import os
from pathlib import Path
from typing import List

from .qdrant_store import (
    create_collection,
    store_documents
)


# ==========================================
# LOAD FILES FROM KNOWLEDGE BASE
# ==========================================

def load_knowledge_base() -> List[dict]:
    kb_dir = Path("knowledge_base")
    if not kb_dir.exists():
        # Look in parent if running from backend/
        kb_dir = Path("../knowledge_base")

    documents_raw = []
    
    if kb_dir.exists():
        for file_path in kb_dir.glob("*"):
            if file_path.suffix in [".txt", ".md"]:
                content = file_path.read_text(encoding="utf-8")
                documents_raw.append({
                    "filename": file_path.name,
                    "content": content
                })
    
    if not documents_raw:
        # Fallback if no file found
        default_file = Path("knowledge_base/nova_it_handbook.txt")
        if default_file.exists():
            documents_raw.append({
                "filename": default_file.name,
                "content": default_file.read_text(encoding="utf-8")
            })

    return documents_raw


# ==========================================
# CHUNK TEXT
# ==========================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80
) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ==========================================
# INGEST
# ==========================================

def ingest():
    print("[Ingest] Loading knowledge base documents...")
    raw_docs = load_knowledge_base()

    if not raw_docs:
        print("[Ingest] No knowledge base documents found!")
        return

    documents = []
    chunk_counter = 0

    for doc in raw_docs:
        chunks = chunk_text(doc["content"])
        for chunk in chunks:
            documents.append({
                "text": chunk,
                "source": doc["filename"],
                "chunk_id": f"{doc['filename']}_{chunk_counter}"
            })
            chunk_counter += 1

    print(f"[Ingest] Generated {len(documents)} chunks from {len(raw_docs)} files.")
    create_collection()
    store_documents(documents)
    print("[Ingest] Knowledge base ingestion complete.")


if __name__ == "__main__":
    ingest()
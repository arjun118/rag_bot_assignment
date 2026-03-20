from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ---------------- #

MODEL_NAME = "./models/qwen3_0_6b_embedding"
COLLECTION_NAME = "rag_chunks"

DOWNLOADS_DIR = Path("downloads")
FALLBACK_USER_ID = "default_files"

ALLOWED_EXTENSIONS = {".txt", ".md"}

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

model = SentenceTransformer(MODEL_NAME)


def ensure_collection():
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
            ),
        )


def stable_id(*parts: str) -> int:
    h = hashlib.sha1("||".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def normalize(vectors: np.ndarray):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def get_files_for_user(user_id: str):
    user_dir = DOWNLOADS_DIR / user_id
    if not user_dir.exists():
        return []

    return [
        p
        for p in user_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    ]


def split_text(text: str):
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)

    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])

    return [node.text for node in nodes]


def ingest_user(user_id: str):
    files = get_files_for_user(user_id)

    if not files:
        print(f"No files for {user_id}")
        return

    points = []
    all_chunks = []
    metadata = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = split_text(text)

        for idx, chunk in enumerate(chunks):
            chunk_id = stable_id(user_id, file_path.name, str(idx), chunk)

            all_chunks.append(chunk)
            metadata.append(
                {
                    "id": chunk_id,
                    "user_id": user_id,
                    "source_file": file_path.name,
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

    print(f"[{user_id}] Embedding {len(all_chunks)} chunks...")

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = normalize(np.array(embeddings, dtype=np.float32))

    for meta, emb in zip(metadata, embeddings):
        points.append(PointStruct(id=meta["id"], vector=emb.tolist(), payload=meta))

    print(f"Inserting into Qdrant...")

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"[{user_id}] Ingested {len(points)} chunks.")


def main():
    print("Ensuring collection...")
    ensure_collection()

    print("Ingesting fallback corpus...")
    ingest_user(FALLBACK_USER_ID)

    print("Done.")


if __name__ == "__main__":
    main()

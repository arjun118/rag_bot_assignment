import os
import uuid
from pathlib import Path

import config
import numpy as np
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer
from states import ChatState
from utils import generate_answer, save_history

router = Router()

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL_NAME_PATH = config.EMBEDDING_MODEL_NAME_PATH
COLLECTION_NAME = "rag_chunks"

DOWNLOADS_DIR = Path("../downloads")
FALLBACK_USER_ID = "default_files"


bot_model = SentenceTransformer(EMBEDDING_MODEL_NAME_PATH)

qdrant = QdrantClient(host="localhost", port=6333)


def normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def resolve_user(user_id: int):
    user_dir = DOWNLOADS_DIR / str(user_id)

    if user_dir.exists() and any(user_dir.iterdir()):
        return str(user_id)

    return FALLBACK_USER_ID


def retrieve(query: str, user_id: str, top_k=5):
    q = bot_model.encode([query])
    q = normalize(np.array(q, dtype=np.float32))[0]

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q.tolist(),
        # query_filter=Filter(
        #     must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        # ),
        limit=top_k,
        with_payload=True,
    ).points  # <-- note the .points at the end

    return results


@router.message(Command("ask"))
async def ask_handler(message: Message):
    query = message.text.partition(" ")[2].strip()

    if not query:
        await message.answer("Usage: /ask <your question>")
        return

    await message.answer("Thinking...")

    user_id = resolve_user(message.from_user.id)

    results = retrieve(query, user_id)

    if not results:
        await message.answer("No relevant information found.")
        return

    contexts = [r.payload["text"] for r in results]

    answer = generate_answer(query, contexts)

    sources = []
    seen = set()

    for r in results:
        file = r.payload["source_file"]
        chunk = r.payload["chunk_index"]

        key = (file, chunk)
        if key not in seen:
            seen.add(key)
            sources.append(f"{file} (chunk {chunk})")

    sources_text = "\n".join(f"• {s}" for s in sources)

    final_response = f"""
{answer}

📚 Sources:
{sources_text}
"""

    await message.answer(final_response.strip())

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

DOWNLOADS_DIR = Path(config.DOWNLOADS_DIR)
FALLBACK_USER_ID = "default_files"

SYSTEM_PROMPT = """you are an helpful ai rag assistant who answers user queries basis the context or knowledge base provided\n
you must always provide answers that are grounded in the context. if un-sure about the answer from the given context,state the same"""

QDRANT_HOST = config.QDRANT_HOST

CACHE_DIR = config.CACHE_DIR
bot_model = SentenceTransformer(EMBEDDING_MODEL_NAME_PATH)

qdrant = QdrantClient(host=QDRANT_HOST, port=6333)

from diskcache import Cache

cache = Cache(CACHE_DIR)


def get_embedding(query, embed_fn):
    if query in cache:
        print("embedding cache hit")
        return cache[query]
    embedding = embed_fn(query)
    cache[query] = embedding
    return embedding


def normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def resolve_user(user_id: int):
    user_dir = DOWNLOADS_DIR / str(user_id)

    if user_dir.exists() and any(user_dir.iterdir()):
        return str(user_id)

    return FALLBACK_USER_ID


def retrieve(query: str, user_id: str, top_k=5):
    # q = bot_model.encode([query])
    q = get_embedding([query], bot_model.encode)
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
async def ask_handler(message: Message, state: FSMContext):
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
    context_text = "\n\n".join(f"[Source {i + 1}]\n{c}" for i, c in enumerate(contexts))
    user_prompt = f"""context : f{context_text}\n user query: {query}"""
    answer = generate_answer(SYSTEM_PROMPT, user_prompt)

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
    await save_history(state, "rag", query, final_response)
    await message.answer(final_response.strip())

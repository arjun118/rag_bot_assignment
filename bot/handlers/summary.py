import os
import uuid

import config
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

# from aiogram.utils.markdown import escape_md
from states import ChatState
from utils import generate_answer, save_history

router = Router()

SYSTEM_PROMPT = """YOU are an excellent ai summarizer,give the data of past n interactions, generate the conscise summary of the
last n interactions given in proper formatting with proper indentation. strictly avoid markdown format.
"""


# Step 1: user enters /image
@router.message(Command("summarize"))
async def chat_start(message: Message, state: FSMContext):
    n = message.text.partition(" ")[2].strip()
    try:
        n = int(n)
    except Exception as e:
        await message.answer("the number of last messages should be an integer")
        return
    # await message.answer("Succesfully summarized")
    context = """"""
    state_data = await state.get_data()  # <-- IMPORTANT: await
    history = state_data.get("chat_history", [])

    relevant = history[::-1][: min(n, len(history))]

    context = ""

    for i, interaction in enumerate(relevant[::-1]):
        if interaction[0].get("type") == "image_caption_generation":
            context += f"""
    Interaction {i + 1}:
    Type: Image Caption
    User uploaded an image
    Caption: {interaction[1].get("content", "N/A")}
    Tags: {", ".join(interaction[1].get("tags", []))}
    """

        elif interaction[0].get("type") == "rag":
            context += f"""
    Interaction {i + 1}:
    Type: RAG
    User: {interaction[0]["content"]}
    Assistant: {interaction[1]["content"]}
    """
        else:
            pass
    refined_answer = generate_answer(SYSTEM_PROMPT, context)
    # safe_text = escape_md(refined_answer, version=2)

    await message.answer(refined_answer)

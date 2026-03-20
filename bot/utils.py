import os

import config
import spacy
from aiogram.fsm.context import FSMContext
from dotenv import load_dotenv
from groq import Groq

nlp = spacy.load("en_core_web_sm")
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)


async def save_history(
    state: FSMContext,
    query_type: str,
    user_query: str,
    assistant_response: str,
    tags=[],
):
    data = await state.get_data()
    history = data.get("chat_history", [])
    if query_type == "rag":
        history.append(
            [
                {"role": "user", "content": user_query, "type": query_type},
                {"role": "assistant", "content": assistant_response},
            ]
        )
    elif query_type == "image_caption_generation":
        history.append(
            [
                {"role": "user", "content": user_query, "type": query_type},
                {"role": "assistant", "content": assistant_response, "tags": tags},
            ]
        )
    await state.update_data(chat_history=history)


def generate_answer(system_prompt, user_prompt):
    response = groq_client.chat.completions.create(
        model=config.LLM_FOR_ANSWER_SYNTHESIS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()


def extract_tags_spacy(text, k=3):
    doc = nlp(text)
    nouns = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]

    seen = set()
    tags = []
    for w in nouns:
        if w not in seen:
            tags.append(w)
            seen.add(w)
        if len(tags) == k:
            break

    return tags

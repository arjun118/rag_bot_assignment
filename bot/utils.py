import os

import config
from aiogram.fsm.context import FSMContext
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)


async def save_history(
    state: FSMContext, query_type: str, user_query: str, assistant_response: str
):
    data = await state.get_data()
    history = data.get("chat_history", [])
    history.append(
        [
            {"role": "user", "content": user_query, "type": query_type},
            {"role": "assistant", "content": assistant_response},
        ]
    )
    await state.update_data(chat_history=history)


def generate_answer(query: str, contexts):
    context_text = "\n\n".join(f"[Source {i + 1}]\n{c}" for i, c in enumerate(contexts))

    prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY the provided context.
If the answer is not present, say you don't know.

Context:
{context_text}

Question:
{query}

Answer clearly and concisely.
"""

    response = groq_client.chat.completions.create(
        model=config.LLM_FOR_ANSWER_SYNTHESIS,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

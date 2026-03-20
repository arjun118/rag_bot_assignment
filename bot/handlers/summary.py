import os
import uuid

import config
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from states import ChatState
from utils import save_history

router = Router()


# Step 1: user enters /image
@router.message(Command("summarize"))
async def chat_start(message: Message, state: FSMContext):
    n = message.text.partition(" ")[2].strip()
    try:
        n = int(n)
    except Exception as e:
        await message.answer("the number of last messages should be an integer")
        return
    await message.answer("Succesfully summarized")

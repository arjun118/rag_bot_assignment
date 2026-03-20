from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router()


@router.message(Command("hi"))
async def hi_handler(message: Message):
    await message.answer("hi there")

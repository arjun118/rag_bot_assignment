from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router()


@router.message(Command("help"))
async def help_handler(message: Message):
    help_text = (
        "*Bot Usage Guide*\n\n"
        "Here’s what you can do:\n\n"
        "*/ask <query>*\n"
        "Ask any question or run a RAG query.\n"
        "Example:\n"
        "`/ask What is retrieval augmented generation?`\n\n"
        "*/image*\n"
        "Send an image after this command to get:\n"
        "• A short caption\n"
        "• 3 relevant tags\n\n"
        "*/help*\n"
        "Show this help message.\n\n"
        "💡 Tip: Keep your queries clear for better results."
    )

    await message.answer(help_text, parse_mode="Markdown")

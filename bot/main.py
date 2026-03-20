from aiogram import F, Router

router = Router()

import asyncio
import logging

import config
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, Message
from dotenv import load_dotenv

load_dotenv()
from handlers.help import router as help_router
from handlers.image import router as image_router
from handlers.rag import router as rag_router
from handlers.summary import router as summary_router

BOT_TOKEN = config.BOT_TOKEN


async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=BOT_TOKEN)
    memorystorage = MemoryStorage()
    dp = Dispatcher(storage=memorystorage)

    # Register routers
    dp.include_router(help_router)
    dp.include_router(image_router)
    dp.include_router(rag_router)
    dp.include_router(summary_router)

    # Set commands
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Start the bot"),
            BotCommand(command="help", description="learn/get help for the commands"),
            BotCommand(
                command="image",
                description="upload an image and get the caption + keywords/tags",
            ),
            BotCommand(
                command="ask",
                description="ask a query and get the answer from grounded in the knowledge base",
            ),
            BotCommand(
                command="summarize",
                description="summarize your last n meaningful interactions",
            ),
        ]
    )

    print("Bot is running...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

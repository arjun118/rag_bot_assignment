import os
import uuid
from pathlib import Path

import config
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from states import ChatState
from utils import extract_tags_spacy, save_history

router = Router()

IMAGES_DIR = Path(config.IMAGES_DIR)
os.makedirs(IMAGES_DIR, exist_ok=True)

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# Load model
BLIP_MODEL_PATH = config.BLIP_MODEL_PATH
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_PATH)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)


# Step 1: user enters /image
@router.message(Command("image"))
async def chat_start(message: Message, state: FSMContext):
    await message.answer("Please Send me the Image.")
    await state.set_state(ChatState.waiting_for_image)


# Step 2a: handle image sent as document
@router.message(ChatState.waiting_for_image, F.document)
async def handle_image_document(message: Message, state: FSMContext):
    document = message.document
    if document.mime_type not in [
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/jpg",
    ]:
        await message.answer("Please upload a valid image.")
        return

    file_id = document.file_id
    file_name = document.file_name
    bot = message.bot
    file = await bot.get_file(file_id)
    user_id = message.from_user.id
    user_dir = os.path.join(IMAGES_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4()}_{file_name}"
    save_path = os.path.join(user_dir, unique_name)
    await bot.download_file(file.file_path, save_path)
    # await state.clear()
    # await save_history(state,"image",save_path,)
    await message.answer(f"Saved {file_name}")


# Step 2b: handle image sent as photo (normal Telegram send)
@router.message(ChatState.waiting_for_image, F.photo)
async def handle_image_photo(message: Message, state: FSMContext):
    photo = message.photo[-1]  # largest available size
    file_id = photo.file_id
    bot = message.bot
    file = await bot.get_file(file_id)
    user_id = message.from_user.id
    user_dir = os.path.join(IMAGES_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4()}.jpg"
    save_path = os.path.join(user_dir, unique_name)
    await bot.download_file(file.file_path, save_path)
    # await state.clear()
    await message.answer("Image saved!, Generating caption..")
    image = Image.open(save_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)
    # Generate caption
    out = blip_model.generate(**inputs, max_length=500)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    # update the state
    tags = extract_tags_spacy(caption)
    await save_history(state, "image_caption_generation", save_path, caption, tags=tags)
    await message.answer(f"Generated Caption: {caption}\nTags: {tags}")

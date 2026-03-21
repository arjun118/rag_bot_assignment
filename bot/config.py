import os

from dotenv.main import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
EMBEDDING_MODEL_NAME_PATH = "./bot/models/qwen3_0_6b_embedding"
LLM_FOR_ANSWER_SYNTHESIS = "llama-3.1-8b-instant"
BLIP_MODEL_PATH = "./bot/models/blip_image_captioning_base"
QDRANT_HOST = os.getenv("QDRANT_HOST")
DOWNLOADS_DIR = "./bot/downloads"
IMAGES_DIR = "./bot/images/"
CACHE_DIR = "./bot/cache/"

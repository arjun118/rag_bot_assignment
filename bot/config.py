import os

from dotenv.main import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
EMBEDDING_MODEL_NAME_PATH = "./models/qwen3_0_6b_embedding"
LLM_FOR_ANSWER_SYNTHESIS = "llama-3.1-8b-instant"
BLIP_MODEL_PATH = "./models/blip_image_captioning_base"

# Telegram AI Bot

A high-performance Telegram bot featuring Retrieval-Augmented Generation (RAG), image captioning + keyword/tag generation, and conversation summarization. The system integrates local lightweight models with high-speed cloud LLMs for an efficient hybrid architecture.

---

## Tech Stack and Architecture

### RAG Pipeline
* **Orchestration:** Llama-index for document parsing and chunking.
* **Chunking Strategy:** Text and Markdown files are split into chunks of 512 tokens with a 100-token overlap.
* **Embedding Model:** Qwen 3 0.6B Embedding (running locally).
* **Vector Database:** Qdrant.
* **Inference:** Llama 3.1 8B via Groq API for near real-time response generation.
* **Optimization:** `diskcache` is implemented to provide a persistent query-to-embedding cache, reducing redundant computations.

### Vision and NLP
* **Image Captioning:** `blip-image-captioning-base` (running locally).
* **Keyword Extraction:** `spaCy` (en_core_web_sm) to extract tags from generated captions.
* **Storage:** Images are organized in user-specific directories; captions and tags are stored in state.

### Extra Features
* **Contextual Memory:** Maintains full chat history across interactions. this is currently not being used with the subsequent interactions but that can be implemented fairly easily
* **Summarization:** A `/summarize <n>` command that processes the last `n` actual user-bot interactions (excluding command triggers like `/help`).
---

## Prerequisites

* **Docker and Docker Compose** installed on your machine.
* **Telegram Bot Token:** Obtained via [BotFather](https://t.me/botfather).
* **Groq API Key:** Obtained via the [Groq Console](https://console.groq.com/keys).

---

## Setup Instructions

### 1. Clone the repository
Clone the repository to your local machine and build the Docker containers:

```bash
git clone https://github.com/arjun118/avivo_assignment.git
cd avivo_assignment
```

### 2. Create a .env file and build
Create a `.env` file in the project directory and populate it with your credentials:

```env
BOT_TOKEN=your_telegram_bot_token_here
GROQ_API_KEY=your_groq_api_key_here
QDRANT_HOST=qdrant
```

### 3. Download models

#### change directory into bot
```bash
cd bot
```
now download the local models required

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-Embedding-0.6B",local_dir="./models/qwen3_0_6b_embedding")
snapshot_download(repo_id="Salesforce/blip-image-captioning-base",local_dir="./models/blip_image_captioning_base")
```


### 3.Build and Verify Images
After the build completes, verify that the application image is present in your local registry:

```bash
docker compose build
```

```bash
docker image ps
```

### 4. Deployment
Launch the application and the Qdrant vector database using Docker Compose:

```bash
docker compose up
```

This command spins up two main services:
1.  **qdrant:** The vector database for storing embeddings.
2.  **bot:** The Python application handling Telegram polling and model logic.

---

## Usage

1.  **Start the Bot:** Open your bot on Telegram and send `/start`. You will find the bot link in the BotFather's reply
2.  **RAG:** Pre-existing 5 files (mix of `.txt`and `.md` files) are chunked and embedded. ask a query using `/ask <query>`
3.  **Image Analysis:** Send an image to the bot. It will store the file, generate a caption using BLIP, and extract keywords via spaCy. use with `/image`
4.  **Summarization:** Use the command `/summarize 5` to get a summary of the last 10 messages in your conversation.
5.  **Help** Use the `/help` command to get information regarding the command usage

## Demo Video
[Demo Video](./avivo_rag_demo.webm)

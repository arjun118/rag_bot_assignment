# Use a slim Python 3.11 image (matches your .pyc files)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /avivo_assignment
# Install git or build-essential if any Python packages require it
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# 1. Install packages from pyproject.toml
RUN pip install --no-cache-dir .

RUN python -m spacy download en_core_web_sm
# 2. Download models to the /models folder
# This runs during the build, so the models become part of the Docker image
# RUN python model_setup.py

# Create a startup script to populate embeddings and run the bot
# We wait 5 seconds to ensure Qdrant has fully booted up before populating

RUN echo '#!/bin/bash\n\
set -e\n\
echo "Waiting for Qdrant to start..."\n\
sleep 5\n\
echo "Populating vector database..."\n\
python /avivo_assignment/bot/populate_embeddings.py\n\
echo "Starting main bot..."\n\
python /avivo_assignment/bot/main.py' > /avivo_assignment/start.sh && chmod +x /avivo_assignment/start.sh

# Run the startup script
CMD ["/avivo_assignment/start.sh"]

FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# CUDA-enabled PyTorch (cu121)
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio && \
    pip install --no-deps --upgrade --pre numpy

# Project deps
WORKDIR /app
COPY pyproject.toml ./
RUN pip install -e .

# App code
COPY sel.py sel_discord.py audio_utils.py ./

CMD ["python3", "sel.py"]

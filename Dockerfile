# Default build: CPU-only, cross-platform (x86_64 + ARM64 / Apple Silicon).
# This works on virtually any machine. For NVIDIA GPU acceleration, use
# Dockerfile.gpu (see docker-compose.gpu.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ORT_PROVIDERS=CPUExecutionProvider

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["python", "run.py"]

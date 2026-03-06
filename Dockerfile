# ============================================
# ETA-OTT ML Service — Production Dockerfile
# Python 3.11 + FastAPI + PyTorch (CPU-only)
# ============================================

FROM python:3.11-slim AS runtime

# Install system dependencies required by:
# - ffmpeg: Whisper audio processing, moviepy video handling
# - libgl1: OpenCV/Pillow image operations
# - wget/curl: health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: run as non-root
RUN groupadd -r etaml && useradd -r -g etaml -m etaml

WORKDIR /app

# Install PyTorch CPU-only FIRST (large layer, cached separately)
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    torchaudio==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers for web extraction
RUN playwright install --with-deps chromium

# Copy application source
COPY main.py ./
COPY extractors/ ./extractors/

# Set ownership
RUN chown -R etaml:etaml /app

USER etaml

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

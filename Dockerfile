# Multi-Architecture Production Dockerfile for DAG-KV
# Supports linux/amd64 (NVIDIA/AMD) and linux/arm64 (Apple Silicon / Graviton / Ampere)

FROM python:3.10-slim

LABEL maintainer="Kaushal Nandaniya"
LABEL description="DAG-KV Universal Position-Decoupled KV Cache Inference Engine"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first for layer caching
COPY pyproject.toml requirements.txt /app/

# Install dependencies (CPU/CUDA agnostic wheels)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install pytest

# Copy application source
COPY dag_kv /app/dag_kv
COPY configs /app/configs
COPY README.md LICENSE /app/

# Install dag-kv package in editable mode
RUN pip install -e .

# Expose default HTTP serving port
EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint
ENTRYPOINT ["dag-kv"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

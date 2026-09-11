# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Fine-tuning engine — control plane image.
#
# CPU-only and small on purpose. Training does not happen in this container: the
# engine creates a Kubernetes Job on the GPU cluster which runs the digest-pinned
# Unsloth image, so no CUDA runtime, no torch and no unsloth belong here. The
# trainer sources under trainer/ are still copied in, because the engine ships
# them to that pod as a ConfigMap (see app/services/worker_bundle.py) — they are
# data to this image, never imported by it.

# Stage 1: Builder - compile dependencies
FROM python:3.11-slim AS builder

# Proxy build args (set only when behind a proxy; passed via --opt=build-arg at build time)
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy}

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set build working directory before any relative COPY (DL3045)
WORKDIR /build

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - minimal production image
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user with specific UID/GID
RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/app --shell=/bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# The GPU-side worker. Shipped as content, not code: worker_bundle reads these
# files and publishes them as a ConfigMap the training pod mounts. Their content
# hash names that ConfigMap, so rebuilding this image with a changed trainer is
# what rolls new worker code out — and running jobs keep the bundle they started
# with.
COPY --chown=appuser:appuser trainer/ ./trainer/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check with proper timeout and retries. /health is dependency-free by
# design: see app/main.py for why the database and the GPU cluster are not in it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run application with production settings.
# One worker per pod, and only one: the engine holds in-memory state for every
# job it is driving (progress records, cancellation flags, the GPU slot pool), and
# a second worker process would be a second, invisible view of the same cluster.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000"]

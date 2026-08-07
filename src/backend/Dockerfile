# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Interactive VLM Feedback Loop — Backend (FastAPI) Dockerfile
#
# Multi-stage build:
#   builder  → installs Python deps into /app/.venv with uv (locked)
#   runtime  → minimal python:3.12-alpine3.24 image with the venv, source,
#              and the isolated TAO base-provisioning helper

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — resolve and install dependencies with uv
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-alpine3.24 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# AIPerf's locked psutil and sentencepiece versions do not ship musllinux
# wheels. Compile those two dependencies here; this entire toolchain remains
# in the discarded builder stage. The project itself is pure Python. Pin uv
# so image builds don't drift with upstream releases.
RUN apk add --no-cache build-base linux-headers cmake git

RUN pip install --no-cache-dir uv==0.12.1

WORKDIR /app

# Two-phase install so the (large, slow) third-party dependency layer is
# cached independently of application source: source-only changes rebuild
# just the final project install.
#
# Phase 1: dependency metadata only → locked third-party deps.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Phase 2: readme (referenced by hatchling) + backend source → the project
# package itself. The hatchling wheel target is src/backend/vlm_feedback_loop
# (remapped to `vlm_feedback_loop` in the wheel).
COPY README.md ./
COPY src/backend/ ./src/backend/
RUN uv sync --frozen --no-dev

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — minimal image that runs uvicorn as a non-root user
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-alpine3.24 AS runtime

# sentencepiece links libstdc++; keep only its runtime library, never the
# compiler toolchain from the builder stage.
RUN apk add --no-cache libstdc++

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOME="/home/appuser"

# Non-root user with a stable HOME — the config loader resolves
# ~/.vlm_feedback_loop relative to appuser's home directory.
RUN addgroup -S -g 1001 appuser \
    && adduser -S -D -u 1001 -G appuser \
        -h /home/appuser -s /sbin/nologin appuser

# Seed a minimal default config.yaml so the backend boots without a host
# mount. Every value here is overridable via process env vars (highest
# precedence per spec §1.9). Operators mounting a host config.yaml on top of
# this file can do so via docker-compose (see docker-compose.yml comment).
RUN mkdir -p /home/appuser/.vlm_feedback_loop \
    && printf 'WORKSPACE_ROOT: /data/workspace\n' \
        > /home/appuser/.vlm_feedback_loop/config.yaml \
    && chown -R appuser:appuser /home/appuser/.vlm_feedback_loop

# Pre-create the workspace mount point with appuser ownership so a Docker
# named volume inherits correct permissions on first use. Bind-mount users
# must ensure their host directory is writable by uid 1001.
RUN mkdir -p /data/workspace && chown -R appuser:appuser /data

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
# First-use Student-base provisioning runs TAO's packaging stack in an
# isolated uv environment. The helper and requirements are package resources
# copied with /app/src above; keep the uv executable in the runtime image.
# The large packaging dependency set is downloaded into appuser's cache on
# first use and never enters the FastAPI venv.
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

USER appuser

EXPOSE 8000

# The container is network-accessible (binds 0.0.0.0). Set BIND_HOST in the
# environment so the APP knows that too — Settings reads BIND_HOST from the
# process env, and filesystem_service.check_browse_allowed uses it to require
# IMAGE_ROOT on a network-accessible process. A hardcoded
# ``--host 0.0.0.0`` that bypassed BIND_HOST left the app believing it was on
# loopback, so an unconfigured image root appeared unrestricted on a
# networked deployment.
ENV BIND_HOST=0.0.0.0 \
    BIND_PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ['BIND_PORT'] + '/health', timeout=5).read()"

# Keep runtime expansion so the actual bind and the app's BIND_HOST setting
# remain a single source of truth. ``exec`` replaces the tiny expansion shell,
# making Uvicorn PID 1 so Docker SIGTERM reaches the application directly.
CMD ["sh", "-c", "exec uvicorn vlm_feedback_loop.main:app --host \"${BIND_HOST}\" --port \"${BIND_PORT}\""]

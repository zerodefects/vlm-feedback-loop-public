# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Interactive VLM Feedback Loop — Backend (FastAPI) Dockerfile
#
# Multi-stage build:
#   builder  → installs Python deps into /app/.venv with uv (locked)
#   runtime  → minimal python:3.12-slim image with the venv, source, and the
#              isolated TAO base-provisioning helper

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — resolve and install dependencies with uv
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# No compiler toolchain needed: every locked runtime dependency ships a
# manylinux wheel (verified with `uv sync --frozen --no-dev --no-build
# --no-install-project`), and the project itself is a pure-Python hatchling
# build. Pin uv so image builds don't drift with upstream releases.
RUN pip install --no-cache-dir uv==0.11.6

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
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOME="/home/appuser"

# curl is needed by the HEALTHCHECK below; every locked Python runtime
# dependency ships self-contained manylinux wheels, so no other system
# libraries are required.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with a stable HOME — the config loader resolves
# ~/.vlm_feedback_loop relative to appuser's home directory.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser \
        --home-dir /home/appuser --create-home \
        --shell /usr/sbin/nologin appuser

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
# isolated uv environment. Keep uv and only the two required helper files in
# the runtime image; the large packaging dependency set is downloaded into
# appuser's cache on first use and never enters the FastAPI venv.
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --chown=appuser:appuser scripts/pull_base_experiments.py /app/scripts/pull_base_experiments.py
COPY --chown=appuser:appuser scripts/pull_base_experiments_requirements.txt /app/scripts/pull_base_experiments_requirements.txt

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
    CMD curl -fsS "http://localhost:${BIND_PORT}/health" || exit 1

# Shell form so ${BIND_HOST}/${BIND_PORT} expand; keeps the actual bind and
# the app's BIND_HOST setting a single source of truth.
CMD uvicorn vlm_feedback_loop.main:app --host "${BIND_HOST}" --port "${BIND_PORT}"

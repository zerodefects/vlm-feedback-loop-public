#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce .github/workflows/ci.yml locally. First failing stage
# exits with a non-zero status. Use this before pushing a change
# you expect to land green in CI — burning a CI run takes minutes;
# this takes seconds for lint and a few minutes for the full chain.
#
# Paired with the workflow file: the seven stages cover its six core jobs and
# serial integration suite. CI-only jobs not mirrored here are the clean
# public-artifact check, dependency audit, and Compose smoke.

set -euo pipefail

cd "$(dirname "$0")/.."

# ── Stage 1: backend lint ────────────────────────────────────────
echo "==> backend-lint (ruff check + ruff format --check + AGENTS/CLAUDE twin check)"
uv run ruff check .
uv run ruff format --check .
diff AGENTS.md CLAUDE.md

# ── Stage 2: backend tests + coverage gate ───────────────────────
echo "==> backend-tests (pytest tests/unit/ with the fail_under coverage gate)"
uv run pytest tests/unit/ -q --maxfail=5 \
  --cov=src/backend/vlm_feedback_loop \
  --cov-report=term --cov-report=xml

# ── Stage 3: backend typecheck ───────────────────────────────────
echo "==> backend-typecheck (pyright src/backend/)"
uv run pyright src/backend/

# ── Stage 4: frontend lint (eslint + prettier + tsc) ─────────────
echo "==> frontend-lint (eslint . && prettier --check . && tsc --noEmit)"
(
  cd src/ui
  pnpm exec eslint .
  pnpm run format:check
  pnpm run typecheck
)

# ── Stage 5: frontend tests ──────────────────────────────────────
echo "==> frontend-tests (vitest + Playwright first-time workflow)"
(
  cd src/ui
  pnpm test
  pnpm test:e2e
)

# ── Stage 6: frontend build ──────────────────────────────────────
echo "==> frontend-build (tsc -b && vite build)"
(
  cd src/ui
  pnpm build
)

# ── Stage 7: local integration contracts ────────────────────────
echo "==> integration-tests (serial live-server, proxy, SSE, and mock-NIM contracts)"
uv run pytest tests/integration/ -q -n 0

echo
echo "==> all seven core CI stages green"

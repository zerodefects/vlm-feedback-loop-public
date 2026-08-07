# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Interactive VLM Feedback Loop — Frontend (Vite SPA) Dockerfile
#
# Multi-stage build:
#   builder  → pnpm install + vite build → /app/dist
#   runtime  → nginx:alpine serving the static bundle on port 3000
#              with SPA fallback (client-side routing)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — install deps and build the Vite bundle
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS builder

ENV PNPM_HOME="/pnpm" \
    PATH="/pnpm:$PATH" \
    CI=true
RUN corepack enable

WORKDIR /app

# Copy lockfile, manifest, and the vendored KUI tarball first so the install
# layer caches on dependency changes only.
COPY package.json pnpm-lock.yaml ./
COPY assets/ ./assets/

RUN --mount=type=cache,id=pnpm,target=/pnpm/store \
    pnpm install --frozen-lockfile

COPY . .

RUN pnpm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime — nginx:alpine serving /app/dist
# ─────────────────────────────────────────────────────────────────────────────
FROM nginx:1.30.4-alpine AS runtime

# SPA fallback + long-cache for hashed asset bundles.  The runtime binds the
# unprivileged port 3000, so keep the nginx master and workers unprivileged as
# well; only the immutable image build needs root ownership changes.
COPY --chown=nginx:nginx nginx.conf /etc/nginx/conf.d/default.conf

# Replace the default html root with the built Vite bundle
RUN rm -rf /usr/share/nginx/html/*
COPY --from=builder --chown=nginx:nginx /app/dist /usr/share/nginx/html

# The stock nginx entrypoint and runtime create the PID and client-proxy temp
# files in these locations.  Make only those runtime paths writable before
# dropping privileges; the application bundle remains read-only at runtime.
RUN mkdir -p /var/cache/nginx \
    && chown -R nginx:nginx /var/cache/nginx /run \
    && sed -i '/^user  nginx;$/d' /etc/nginx/nginx.conf

USER nginx

EXPOSE 3000

# nginx:alpine ships an entrypoint that runs nginx in the foreground by default
CMD ["nginx", "-g", "daemon off;"]

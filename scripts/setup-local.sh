#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# setup-local.sh — One-time prerequisite script for local NIM deployment
#
# Run this on a bare-metal machine before first local NIM deployment.
# NOT auto-triggered from the web UI. The SME or administrator runs this once.
#
# What it does:
#   1. Checks NVIDIA drivers (fails if missing)
#   2. Installs Docker if missing
#   3. Installs NVIDIA Container Toolkit if missing
#   4. Configures Docker runtime for NVIDIA GPU passthrough
#   5. Validates GPU access via Docker
#   6. Creates NIM cache directory
#   7. Optionally pre-pulls NIM container images
#
# Usage:
#   chmod +x scripts/setup-local.sh
#   ./scripts/setup-local.sh
#
# Environment variables (optional):
#   NGC_API_KEY  — NGC API key for pulling NIM images (required for pre-pull)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# Shared Docker / NVIDIA Container Toolkit provisioning steps.
# shellcheck source=scripts/setup-common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup-common.sh"

# ── Step 1: Check NVIDIA drivers ─────────────────────────────────────────────

info "Checking NVIDIA drivers..."
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. Install NVIDIA drivers before running this script.
  See: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/"
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

ok "NVIDIA driver $DRIVER_VERSION detected"
ok "GPU: $GPU_NAME (${GPU_MEM} MiB)"

# ── Step 2: Install Docker if missing ────────────────────────────────────────

info "Checking Docker..."
if command -v docker &>/dev/null; then
    DOCKER_VERSION=$(docker --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
    ok "Docker $DOCKER_VERSION already installed"
else
    info "Docker not found. Installing Docker Engine..."
    vfl_install_docker_engine
    vfl_ensure_docker_group

    DOCKER_VERSION=$(docker --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
    ok "Docker $DOCKER_VERSION installed"
fi

# ── Step 3: Install NVIDIA Container Toolkit if missing ──────────────────────

info "Checking NVIDIA Container Toolkit..."
if command -v nvidia-ctk &>/dev/null; then
    CTK_VERSION=$(nvidia-ctk --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
    ok "NVIDIA Container Toolkit $CTK_VERSION already installed"
else
    info "NVIDIA Container Toolkit not found. Installing..."
    vfl_install_nvidia_container_toolkit

    CTK_VERSION=$(nvidia-ctk --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' || echo "unknown")
    ok "NVIDIA Container Toolkit $CTK_VERSION installed"
fi

# ── Step 4: Configure Docker runtime ─────────────────────────────────────────

info "Configuring Docker runtime for NVIDIA GPU passthrough..."
vfl_configure_docker_nvidia_runtime
ok "Docker runtime configured for NVIDIA GPU"

# ── Step 5: Validate GPU access via Docker ────────────────────────────────────

info "Validating GPU access through Docker..."
if docker run --rm --runtime=nvidia --gpus all "$VFL_CUDA_TEST_IMAGE" nvidia-smi -L 2>/dev/null; then
    ok "GPU passthrough works: Docker can access $GPU_NAME"
else
    fail "GPU passthrough failed. Check NVIDIA Container Toolkit installation.
  Try: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

# ── Step 6: Create NIM cache directory ────────────────────────────────────────

NIM_CACHE="$HOME/.cache/nim"
info "Creating NIM cache directory: $NIM_CACHE"
vfl_create_nim_cache "$NIM_CACHE"
ok "NIM cache directory ready: $NIM_CACHE (world-writable for NIM containers)"

# ── Step 7: Optional pre-pull of NIM images ───────────────────────────────────

# Pinned nano fp8 profile id (RTX PRO 6000 / fp8 / tp1). Pinning the profile
# is the documented fix for the cosmos3-reasoner silent-fallback footgun: with
# only NIM_MODEL_SIZE=nano set, a failed nano profile fetch leaves a 52 KB
# config-only stub in the cache and the NIM silently serves the resident SUPER
# weights. Pre-caching genuine nano weights (below) requires this pin plus the
# host-UID-writable cache.
#
# AUTHORITATIVE copy: src/backend/vlm_feedback_loop/services/project_service.py
# (SEEDED_MODEL_CATALOG, cosmos3-nano-reasoner local_deploy_metadata
# .nim_model_profile). The check below fails fast if this pinned copy ever
# drifts from the backend catalog.
COSMOS3_NANO_PROFILE="e2e00f3e555bb4fe0ef011faadd56a37441c7274e149d482cfeb67dbfb75b092"
COSMOS3_REASONER_IMAGE="nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0"

# AUTHORITATIVE copy: src/backend/vlm_feedback_loop/model_catalog_constants.py
# (EMBEDDING_NIM_IMAGE — the tag the backend actually deploys; pre-pulling
# anything else warms the wrong cache). The check below fails fast on drift.
EMBEDDING_NIM_IMAGE="nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0"

_BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/backend/vlm_feedback_loop"
_CATALOG_PY="$_BACKEND_DIR/services/project_service.py"
if [ -f "$_CATALOG_PY" ] && ! grep -q "$COSMOS3_NANO_PROFILE" "$_CATALOG_PY"; then
    fail "COSMOS3_NANO_PROFILE in this script diverges from the backend model catalog.
  Update it to match local_deploy_metadata.nim_model_profile in $_CATALOG_PY"
fi
_CONSTANTS_PY="$_BACKEND_DIR/model_catalog_constants.py"
if [ -f "$_CONSTANTS_PY" ] && ! grep -q "$EMBEDDING_NIM_IMAGE" "$_CONSTANTS_PY"; then
    fail "EMBEDDING_NIM_IMAGE in this script diverges from the backend model catalog.
  Update it to match EMBEDDING_NIM_IMAGE in $_CONSTANTS_PY"
fi

echo ""
info "NIM container images for local deployment:"
echo "  - nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant"
echo "      (Teacher, quality default on supported ≥80 GB / cc≥9.0 GPUs)"
echo "  - nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0"
echo "      (Teacher, CR3-Nano fallback, >56 GB GPU)"
echo "  - nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0  (Teacher, CR2-2B [fallback], >36 GB GPU)"
echo "  - $EMBEDDING_NIM_IMAGE  (Embeddings, ≥10 GB compatibility floor)"
echo ""
info "CR2-8B and Cosmos 3 Super remain selectable; memory size alone does not"
info "outrank the quality-default Omni → CR3-Nano → CR2-2B order."
echo ""

if [ -n "${NGC_API_KEY:-}" ]; then
    info "NGC_API_KEY detected. Logging in to nvcr.io..."
    vfl_ngc_docker_login docker
    ok "Logged in to nvcr.io"

    # Interactive prompt for pre-pull (only if stdin is a terminal)
    if [ -t 0 ]; then
        echo ""
        read -rp "Pre-pull NIM images now? This saves time on first deploy. [y/N] " PULL_ANSWER
        if [[ "${PULL_ANSWER,,}" == "y" || "${PULL_ANSWER,,}" == "yes" ]]; then
            info "Pulling NIM images (this may take a while)..."

            for IMAGE in \
                "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0" \
                "$EMBEDDING_NIM_IMAGE"; do
                info "Pulling $IMAGE..."
                if docker pull "$IMAGE"; then
                    ok "Pulled $IMAGE"
                else
                    warn "Failed to pull $IMAGE — will be pulled on first deploy"
                fi
            done

            # On a >56 GB GPU, the default Teacher is CR3-Nano (served from the
            # shared cosmos3-reasoner:1.7.0 image with NIM_MODEL_SIZE=nano).
            GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            GPU_MEM_GB=$((GPU_MEM_MB / 1024))
            if [ "$GPU_MEM_GB" -ge 56 ]; then
                info "Pulling $COSMOS3_REASONER_IMAGE (GPU has ${GPU_MEM_GB} GB, sufficient for CR3-Nano)..."
                if docker pull "$COSMOS3_REASONER_IMAGE"; then
                    ok "Pulled $COSMOS3_REASONER_IMAGE"

                    # Pre-cache GENUINE nano fp8 weights (~9.9 GB) so the first
                    # deploy serves real nano, not the silent SUPER fallback.
                    # Two things make this reliable (matching the local_nim_service
                    # deploy recipe):
                    #   1. Run the image's CPU-only download-to-cache entrypoint as
                    #      the host UID (-u "$(id -u)") so the cache mount
                    #      (~/.cache/nim, world-writable from Step 6) is writable —
                    #      the UID mismatch was the real root cause of the stub cache.
                    #   2. Pin NIM_MODEL_PROFILE to the nano fp8 profile so the
                    #      download fetches the nano filemap, not whatever the
                    #      fragile size-filter auto-selector resolves.
                    info "Pre-caching genuine CR3-Nano fp8 weights (~9.9 GB) into $NIM_CACHE..."
                    info "  (download-to-cache as host UID + pinned nano profile — avoids the SUPER-fallback stub)"
                    if docker run --rm \
                        -u "$(id -u)" \
                        -e NGC_API_KEY \
                        -e NIM_MODEL_SIZE=nano \
                        -e NIM_MODEL_PROFILE="$COSMOS3_NANO_PROFILE" \
                        -e NIM_SERVED_MODEL_NAME=nvidia/cosmos3-nano-reasoner \
                        -v "$NIM_CACHE:/opt/nim/.cache" \
                        "$COSMOS3_REASONER_IMAGE" \
                        download-to-cache --profile "$COSMOS3_NANO_PROFILE"; then
                        ok "Cached genuine CR3-Nano weights (real fp8 shards, not the 52 KB stub)"
                    else
                        warn "CR3-Nano weight pre-cache failed — the first deploy will fetch them"
                        warn "  (the app deploy path also pins the profile + runs as host UID)"
                    fi
                else
                    warn "Failed to pull $COSMOS3_REASONER_IMAGE"
                fi
            else
                info "Skipping cosmos3-reasoner / CR3-Nano (GPU has ${GPU_MEM_GB} GB, need >56 GB)"
                info "  CR2-2B (pulled above) is the recommended Teacher for this GPU."
            fi
        else
            info "Skipping image pre-pull. Images will be pulled on first deploy."
        fi
    else
        info "Non-interactive mode. Skipping image pre-pull."
    fi
else
    warn "NGC_API_KEY not set. Cannot pre-pull images."
    echo "  To set it: export NGC_API_KEY=<your-key>"
    echo "  Get a key: https://org.ngc.nvidia.com/setup/api-key"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=============================================="
echo -e "${GREEN}Local NIM prerequisites are ready.${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. If not already done: clone the repo and install dependencies"
echo "     uv sync && cd src/ui && pnpm install && cd ../.."
echo "  2. Bootstrap configuration:"
echo "     uv run vlm-feedback-loop init"
echo "  3. Add secrets to ~/.vlm_feedback_loop/.env:"
echo "     NGC_API_KEY=<your-ngc-key>"
echo "     NVIDIA_API_KEY=<your-nvidia-key>  (optional, for hosted NIM)"
echo "  4. Start the application:"
echo "     ./scripts/dev.sh"
echo "  5. Open http://localhost:5173 and create a project"
echo ""

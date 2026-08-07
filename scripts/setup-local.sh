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

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | sed -n '1p')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p')

if ! vfl_driver_meets_nim_minimum "$DRIVER_VERSION"; then
    fail "NVIDIA driver $DRIVER_VERSION is below the $VFL_NIM_MIN_DRIVER_VERSION minimum required by the CUDA 13 VLM NIM images. Update the driver and reboot before retrying."
fi

ok "NVIDIA driver $DRIVER_VERSION meets the $VFL_NIM_MIN_DRIVER_VERSION minimum"
ok "GPU: $GPU_NAME (${GPU_MEM} MiB)"

# ── Step 2: Install Docker if missing ────────────────────────────────────────

info "Checking Docker..."
if ! command -v docker &>/dev/null; then
    info "Docker not found. Installing Docker Engine..."
    vfl_install_docker_engine
fi

vfl_ensure_docker_group

if ! DOCKER_VERSION=$(vfl_docker_server_version); then
    fail "Could not read the Docker Engine server version. Ensure the Docker daemon is running and accessible, then rerun this script."
fi
if ! vfl_docker_meets_nim_minimum "$DOCKER_VERSION"; then
    fail "Docker Engine $DOCKER_VERSION is below the $VFL_NIM_MIN_DOCKER_VERSION minimum required for local NIM. Upgrade Docker through this host's package-management policy, then rerun this script."
fi
ok "Docker Engine $DOCKER_VERSION meets the $VFL_NIM_MIN_DOCKER_VERSION minimum"

DOCKER_CMD="docker"
if ! docker info > /dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
    warn "Docker group membership is not active in this shell; using 'sudo docker' for prerequisite checks."
fi

# ── Step 3: Install NVIDIA Container Toolkit if missing ──────────────────────

info "Checking NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &>/dev/null; then
    info "NVIDIA Container Toolkit not found. Installing..."
    vfl_install_nvidia_container_toolkit
fi

if ! CTK_VERSION=$(vfl_nvidia_ctk_version); then
    fail "Could not read the NVIDIA Container Toolkit version. Verify 'nvidia-ctk --version', then rerun this script."
fi
if ! vfl_ctk_meets_nim_minimum "$CTK_VERSION"; then
    fail "NVIDIA Container Toolkit $CTK_VERSION is below the $VFL_NIM_MIN_CTK_VERSION minimum required for local NIM. Upgrade it through this host's package-management policy, then rerun this script."
fi
ok "NVIDIA Container Toolkit $CTK_VERSION meets the $VFL_NIM_MIN_CTK_VERSION minimum"

# ── Step 4: Configure Docker runtime ─────────────────────────────────────────

info "Configuring Docker runtime for NVIDIA GPU passthrough..."
vfl_configure_docker_nvidia_runtime
ok "Docker runtime configured for NVIDIA GPU"

# ── Step 5: Validate GPU access via Docker ────────────────────────────────────

info "Validating GPU access through Docker..."
if $DOCKER_CMD run --rm --runtime=nvidia --gpus all "$VFL_CUDA_TEST_IMAGE" nvidia-smi -L 2>/dev/null; then
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

# The shell entry point cannot import Python at runtime. These copies mirror
# model_catalog_constants.py; delivery-pin tests fail when either side drifts.
COSMOS_REASON2_2B_NIM_IMAGE="nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0"
COSMOS3_REASONER_NIM_IMAGE="nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0"
NEMOTRON_3_NANO_OMNI_NIM_IMAGE="nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant"
EMBEDDING_NIM_IMAGE="nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0"

COSMOS_REASON2_2B_GPU_MIN_GB=36
COSMOS3_NANO_REASONER_GPU_MIN_GB=56
NEMOTRON_3_NANO_OMNI_GPU_MIN_GB=80
NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN=9.0

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

_BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/backend/vlm_feedback_loop"
_CATALOG_PY="$_BACKEND_DIR/services/project_service.py"
if [ -f "$_CATALOG_PY" ] && ! grep -q "$COSMOS3_NANO_PROFILE" "$_CATALOG_PY"; then
    fail "COSMOS3_NANO_PROFILE in this script diverges from the backend model catalog.
  Update it to match local_deploy_metadata.nim_model_profile in $_CATALOG_PY"
fi
_CONSTANTS_PY="$_BACKEND_DIR/model_catalog_constants.py"
if [ -f "$_CONSTANTS_PY" ]; then
    for IMAGE in \
        "$COSMOS_REASON2_2B_NIM_IMAGE" \
        "$COSMOS3_REASONER_NIM_IMAGE" \
        "$NEMOTRON_3_NANO_OMNI_NIM_IMAGE" \
        "$EMBEDDING_NIM_IMAGE"; do
        if ! grep -Fq "$IMAGE" "$_CONSTANTS_PY"; then
            fail "A NIM image in this script diverges from the backend model catalog.
  Update it to match the corresponding image constant in $_CONSTANTS_PY"
        fi
    done
fi

# Match the backend's nominal-capacity tolerance: an 80 GB card can report a
# little under 80 GiB through nvidia-smi without becoming a different tier.
vfl_gpu_memory_meets_model_floor() {
    awk -v memory_mb="$1" -v minimum_gb="$2" \
        'BEGIN { exit !((memory_mb / 1024) >= (minimum_gb * 0.99)) }'
}

vfl_compute_capability_meets_model_floor() {
    awk -v actual="$1" -v minimum="$2" \
        'BEGIN { exit !(actual >= minimum) }'
}

# Print the highest-quality default Teacher supported by any detected GPU.
# CR2-8B and CR3-Super remain selectable in the app, but do not outrank this
# evidence-backed default order merely because they use more memory.
vfl_select_teacher_prepull_image() {
    local selected_image=""
    local memory_mb compute_capability

    while IFS=, read -r memory_mb compute_capability; do
        memory_mb="${memory_mb//[[:space:]]/}"
        compute_capability="${compute_capability//[[:space:]]/}"
        if ! [[ "$memory_mb" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            continue
        fi

        if vfl_gpu_memory_meets_model_floor \
            "$memory_mb" "$NEMOTRON_3_NANO_OMNI_GPU_MIN_GB" && \
            [[ "$compute_capability" =~ ^[0-9]+([.][0-9]+)?$ ]] && \
            vfl_compute_capability_meets_model_floor \
                "$compute_capability" "$NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN"; then
            selected_image="$NEMOTRON_3_NANO_OMNI_NIM_IMAGE"
            break
        fi
        if vfl_gpu_memory_meets_model_floor \
            "$memory_mb" "$COSMOS3_NANO_REASONER_GPU_MIN_GB"; then
            selected_image="$COSMOS3_REASONER_NIM_IMAGE"
        elif [ -z "$selected_image" ] && \
            vfl_gpu_memory_meets_model_floor \
                "$memory_mb" "$COSMOS_REASON2_2B_GPU_MIN_GB"; then
            selected_image="$COSMOS_REASON2_2B_NIM_IMAGE"
        fi
    done

    printf '%s\n' "$selected_image"
}

SELECTED_TEACHER_IMAGE=$(vfl_select_teacher_prepull_image < <(
    nvidia-smi \
        --query-gpu=memory.total,compute_cap \
        --format=csv,noheader,nounits 2>/dev/null
))

case "$SELECTED_TEACHER_IMAGE" in
    "$NEMOTRON_3_NANO_OMNI_NIM_IMAGE")
        SELECTED_TEACHER_NAME="Nemotron 3 Nano Omni"
        SELECTED_TEACHER_REASON="a GPU meets the ${NEMOTRON_3_NANO_OMNI_GPU_MIN_GB} GB / cc ${NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN} floor"
        ;;
    "$COSMOS3_REASONER_NIM_IMAGE")
        SELECTED_TEACHER_NAME="Cosmos 3 Nano Reasoner"
        SELECTED_TEACHER_REASON="Omni is ineligible and a GPU meets the ${COSMOS3_NANO_REASONER_GPU_MIN_GB} GB floor"
        ;;
    "$COSMOS_REASON2_2B_NIM_IMAGE")
        SELECTED_TEACHER_NAME="Cosmos Reason2 2B"
        SELECTED_TEACHER_REASON="smallest fallback (a GPU meets the ${COSMOS_REASON2_2B_GPU_MIN_GB} GB floor)"
        ;;
    *)
        SELECTED_TEACHER_NAME="none"
        SELECTED_TEACHER_REASON="no GPU meets the ${COSMOS_REASON2_2B_GPU_MIN_GB} GB Teacher floor; only the embedding NIM is eligible"
        ;;
esac

echo ""
info "NIM container images for local deployment:"
echo "  - $NEMOTRON_3_NANO_OMNI_NIM_IMAGE"
echo "      (Teacher, quality default on supported ≥80 GB / cc≥9.0 GPUs)"
echo "  - $COSMOS3_REASONER_NIM_IMAGE"
echo "      (Teacher, CR3-Nano fallback, ≥56 GB GPU)"
echo "  - $COSMOS_REASON2_2B_NIM_IMAGE  (Teacher, CR2-2B [fallback], ≥36 GB GPU)"
echo "  - $EMBEDDING_NIM_IMAGE  (Embeddings, ≥24 GB supported-SKU floor)"
echo ""
info "CR2-8B and Cosmos 3 Super remain selectable; memory size alone does not"
info "outrank the quality-default Omni → CR3-Nano → CR2-2B order."
info "Teacher selected for pre-pull: $SELECTED_TEACHER_NAME ($SELECTED_TEACHER_REASON)."
echo ""

if [ -n "${NGC_API_KEY:-}" ]; then
    info "NGC_API_KEY detected. Logging in to nvcr.io..."
    vfl_ngc_docker_login "$DOCKER_CMD"
    ok "Logged in to nvcr.io"

    # Interactive prompt for pre-pull (only if stdin is a terminal)
    if [ -t 0 ]; then
        echo ""
        read -rp "Pre-pull NIM images now? This saves time on first deploy. [y/N] " PULL_ANSWER
        if [[ "${PULL_ANSWER,,}" == "y" || "${PULL_ANSWER,,}" == "yes" ]]; then
            info "Pulling NIM images (this may take a while)..."

            if [ -n "$SELECTED_TEACHER_IMAGE" ]; then
                info "Pulling selected Teacher image: $SELECTED_TEACHER_IMAGE..."
                if $DOCKER_CMD pull "$SELECTED_TEACHER_IMAGE"; then
                    ok "Pulled $SELECTED_TEACHER_IMAGE"

                    # The shared Cosmos 3 image needs an explicit Nano profile
                    # pre-cache. Omni and CR2 have dedicated images and must not
                    # run this Cosmos-specific recipe.
                    if [ "$SELECTED_TEACHER_IMAGE" = "$COSMOS3_REASONER_NIM_IMAGE" ]; then
                        info "Pre-caching genuine CR3-Nano fp8 weights (~9.9 GB) into $NIM_CACHE..."
                        info "  (download-to-cache as host UID + pinned nano profile — avoids the SUPER-fallback stub)"
                        if $DOCKER_CMD run --rm \
                            -u "$(id -u)" \
                            -e NGC_API_KEY \
                            -e NIM_MODEL_SIZE=nano \
                            -e NIM_MODEL_PROFILE="$COSMOS3_NANO_PROFILE" \
                            -e NIM_SERVED_MODEL_NAME=nvidia/cosmos3-nano-reasoner \
                            -v "$NIM_CACHE:/opt/nim/.cache" \
                            "$COSMOS3_REASONER_NIM_IMAGE" \
                            download-to-cache --profile "$COSMOS3_NANO_PROFILE"; then
                            ok "Cached genuine CR3-Nano weights (real fp8 shards, not the 52 KB stub)"
                        else
                            warn "CR3-Nano weight pre-cache failed — the first deploy will fetch them"
                            warn "  (the app deploy path also pins the profile + runs as host UID)"
                        fi
                    fi
                else
                    warn "Failed to pull $SELECTED_TEACHER_IMAGE — it will be pulled on first deploy"
                fi
            else
                info "Skipping Teacher image pre-pull because no detected GPU meets the ${COSMOS_REASON2_2B_GPU_MIN_GB} GB floor."
            fi

            info "Pulling embedding image: $EMBEDDING_NIM_IMAGE..."
            if $DOCKER_CMD pull "$EMBEDDING_NIM_IMAGE"; then
                ok "Pulled $EMBEDDING_NIM_IMAGE"
            else
                warn "Failed to pull $EMBEDDING_NIM_IMAGE — it will be pulled on first deploy"
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

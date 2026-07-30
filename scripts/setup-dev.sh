#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# setup-dev.sh — Install development toolchain for the VLM Feedback Loop Blueprint.
#
# Run once on a fresh Ubuntu 22.04+ machine:
#   chmod +x scripts/setup-dev.sh
#   ./scripts/setup-dev.sh
#
# GPU-aware: detects NVIDIA GPU automatically and installs Docker, NVIDIA
# Container Toolkit, and pre-pulls NIM container images when a GPU is found.
# On CPU-only machines, skips all GPU components.
#
# What it always installs:
#   - Python 3.12 (via deadsnakes PPA)
#   - uv (Python package manager)
#   - Node.js 20 LTS (via NodeSource)
#   - pnpm (via Corepack)
#   - GitHub CLI (gh)
#   - sqlite3 CLI
#   - Image library dev headers (for Pillow: JPEG, PNG, TIFF, WebP, FreeType)
#
# What it installs when a GPU is detected (Profile C):
#   - Docker Engine (>= 23.0.1 per NVIDIA docs; this Blueprint uses 29.4.0+)
#   - NVIDIA Container Toolkit 1.19.0+
#   - Docker runtime configured for nvidia GPU passthrough
#   - NIM cache directory (~/.cache/nim)
#   - NGC entitlement check (login + manifest inspect — no image pulls)
#   NIM images are pulled by the Blueprint on first deploy, not by this script.
#
# GPU path hard requirements (enforced):
#   - Ubuntu 22.04+ (glibc >= 2.35)
#   - NVIDIA driver >= 535.104.05 (per NVIDIA VLM NIM docs; R580+ only needed for TAO)
#   - Compute capability >= 7.0 (>= 8.0 for BF16, >= 8.9 for FP8)
#
# Prerequisites:
#   - Ubuntu 22.04+ (x86_64)
#   - sudo access
#   - Internet connectivity
#   - NVIDIA driver 535.104.05+ already installed (if GPU — driver install is outside scope)

# Note: pipefail intentionally omitted. Many pipelines use `| head -1` which
# causes SIGPIPE on the upstream command when head closes early — a race
# condition that pipefail turns into a script-terminating error.
set -eu

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
section() { echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }

# Shared Docker / NVIDIA Container Toolkit provisioning steps.
# shellcheck source=scripts/setup-common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup-common.sh"

# Version comparison: returns 0 (true) if $1 >= $2
version_gte() {
    printf '%s\n%s' "$2" "$1" | sort -V -C
}

HAS_GPU=false
GPU_COUNT=0
GPU_MEM_GB=0
GPU_COMPUTE_CAP="0.0"

# ── Check OS ──────────────────────────────────────────────────────────────────

if ! grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
    warn "This script is tested on Ubuntu 22.04+. Other distros may need adjustments."
fi

# ── Detect GPU ────────────────────────────────────────────────────────────────

section "GPU Detection"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "unknown")
    info "NVIDIA GPU detected: $GPU_INFO"

    # ── Enforce OS/glibc for GPU path ────────────────────────────────────
    if ! grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
        error "GPU path requires Ubuntu 22.04+. Detected: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || echo "unknown")"
        error "Skipping GPU setup. Install Ubuntu 22.04 LTS for local NIM deployment."
        HAS_GPU=false
    fi

    if [ "$HAS_GPU" = true ]; then
        # Extract glibc version robustly — grab the last N.N on the first line
        GLIBC_VERSION=$(ldd --version 2>&1 | head -1 | sed -n 's/.*\([0-9]\+\.[0-9]\+\)$/\1/p')
        GLIBC_VERSION="${GLIBC_VERSION:-0.0}"
        GLIBC_MAJOR=$(echo "$GLIBC_VERSION" | cut -d. -f1)
        GLIBC_MINOR=$(echo "$GLIBC_VERSION" | cut -d. -f2)
        if [ "${GLIBC_MAJOR:-0}" -lt 2 ] || { [ "${GLIBC_MAJOR:-0}" -eq 2 ] && [ "${GLIBC_MINOR:-0}" -lt 35 ]; }; then
            error "GPU path requires glibc >= 2.35 (per NVIDIA VLM NIM prerequisites). Detected: $GLIBC_VERSION"
            error "Skipping GPU setup."
            HAS_GPU=false
        else
            info "glibc $GLIBC_VERSION — meets >= 2.35 requirement"
        fi
    fi

    # ── Enforce minimum driver version ──────────────────────────────────
    # NIM VLM containers require >= 535.104.05 (per NVIDIA VLM NIM docs).
    # The R580 (>= 580.65.06) requirement applies to TAO 6.26.3 containers
    # which use CUDA 13.0 — that is the TAO server, not this NIM dev machine.
    if [ "$HAS_GPU" = true ]; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        if version_gte "$DRIVER_VERSION" "535.104.05"; then
            info "Driver $DRIVER_VERSION — meets >= 535.104.05 requirement (NIM VLM)"
            if ! version_gte "$DRIVER_VERSION" "580.65.06"; then
                warn "Driver $DRIVER_VERSION is below 580.65.06 (R580). TAO containers require R580+."
                warn "This is fine for local NIM deployment. TAO runs on a separate machine."
            fi
        else
            error "Driver $DRIVER_VERSION is below minimum 535.104.05 for NIM VLM containers."
            error "Skipping GPU setup. Update driver for local NIM deployment."
            HAS_GPU=false
        fi
    fi

    # ── Gather GPU inventory (compute cap + multi-GPU memory) ────────────
    if [ "$HAS_GPU" = true ]; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        # Use max single-GPU VRAM across all GPUs (not aggregate)
        GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1 | tr -d ' ')
        GPU_MEM_GB=$(( ${GPU_MEM_MB:-0} / 1024 ))

        info "GPU count: $GPU_COUNT, max single-GPU memory: ${GPU_MEM_GB} GB, compute capability: $GPU_COMPUTE_CAP"

        # Compute capability warnings
        COMPUTE_MAJOR=$(echo "$GPU_COMPUTE_CAP" | cut -d. -f1)
        COMPUTE_MINOR=$(echo "$GPU_COMPUTE_CAP" | cut -d. -f2)
        if [ "${COMPUTE_MAJOR:-0}" -lt 7 ]; then
            warn "Compute capability $GPU_COMPUTE_CAP — Cosmos Reason2 requires >= 7.0 (generic)"
        elif [ "${COMPUTE_MAJOR:-0}" -lt 8 ]; then
            warn "Compute capability $GPU_COMPUTE_CAP — BF16 requires >= 8.0 (Ampere+)"
        elif [ "${COMPUTE_MAJOR:-0}" -eq 8 ] && [ "${COMPUTE_MINOR:-0}" -lt 9 ]; then
            info "Compute capability $GPU_COMPUTE_CAP — supports BF16. FP8 requires >= 8.9 (Ada Lovelace+)."
        else
            info "Compute capability $GPU_COMPUTE_CAP — supports BF16 and FP8"
        fi
    fi
else
    info "No NVIDIA GPU detected — skipping Docker, NVIDIA Container Toolkit, and NIM images."
    info "This machine runs as Profile A/B (CPU-only + hosted NIM)."
fi

# ── System packages ───────────────────────────────────────────────────────────

section "System Packages"

info "Updating apt package lists..."
sudo apt-get update -qq

info "Installing system packages (sqlite3, image dev headers, build tools)..."
sudo apt-get install -y -qq \
    software-properties-common \
    sqlite3 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libfreetype-dev \
    pkg-config \
    curl \
    ca-certificates \
    gnupg \
    git \
    > /dev/null

# ── Python 3.12 ──────────────────────────────────────────────────────────────

section "Python 3.12"

if python3.12 --version &>/dev/null; then
    info "Python 3.12 already installed: $(python3.12 --version)"
else
    info "Installing Python 3.12 via deadsnakes PPA..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev > /dev/null
    info "Installed: $(python3.12 --version)"
fi

# ── uv (Python package manager) ──────────────────────────────────────────────

section "uv"

if command -v uv &>/dev/null; then
    info "uv already installed: $(uv --version)"
elif [ -f "$HOME/.local/bin/uv" ]; then
    info "uv found at ~/.local/bin/uv: $($HOME/.local/bin/uv --version)"
    warn "Add ~/.local/bin to your PATH if not already."
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
    export PATH="$HOME/.local/bin:$PATH"
    info "Installed: $(uv --version)"
fi

# ── Node.js 20 LTS ───────────────────────────────────────────────────────────

section "Node.js 20"

NODE_VERSION=$(node --version 2>/dev/null || echo "none")
if [[ "$NODE_VERSION" == v20.* ]]; then
    info "Node.js 20 already installed: $NODE_VERSION"
else
    info "Installing Node.js 20 LTS via NodeSource..."
    sudo apt-get remove -y -qq libnode-dev nodejs-doc 2>/dev/null || true
    sudo dpkg --configure -a 2>/dev/null || true
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - > /dev/null 2>&1
    sudo apt-get install -y -qq nodejs > /dev/null
    info "Installed: $(node --version)"
fi

# ── pnpm (via Corepack) ──────────────────────────────────────────────────────

section "pnpm"

if command -v pnpm &>/dev/null; then
    info "pnpm already installed: $(pnpm --version)"
else
    info "Installing pnpm via Corepack..."
    sudo corepack enable 2>/dev/null
    # Pin pnpm@10 — pnpm@latest (v11+) requires Node ≥22.13, but this script
    # installs Node 20 LTS. pnpm@10 is the latest line compatible with Node 20.
    corepack prepare pnpm@10 --activate 2>/dev/null
    info "Installed: $(pnpm --version)"
fi

# ── GitHub CLI ────────────────────────────────────────────────────────────────

section "GitHub CLI"

if command -v gh &>/dev/null; then
    info "GitHub CLI already installed: $(gh --version | head -1)"
else
    info "Installing GitHub CLI..."
    sudo rm -f /usr/share/keyrings/githubcli-archive-keyring.gpg
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq gh > /dev/null
    info "Installed: $(gh --version | head -1)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# GPU-ONLY SECTION — everything below runs only when a GPU is detected
# ══════════════════════════════════════════════════════════════════════════════

if [ "$HAS_GPU" = true ]; then

    # ── Docker Engine ─────────────────────────────────────────────────────────

    section "Docker Engine (GPU detected)"

    if command -v docker &>/dev/null; then
        DOCKER_VERSION=$(docker --version 2>/dev/null)
        info "Docker already installed: $DOCKER_VERSION"
    else
        vfl_install_docker_engine
        info "Installed: $(docker --version)"
    fi

    # Add current user to docker group if not already
    vfl_ensure_docker_group

    # ── NVIDIA Container Toolkit ──────────────────────────────────────────────

    section "NVIDIA Container Toolkit (GPU detected)"

    if dpkg-query -W -f='${Status}' nvidia-container-toolkit 2>/dev/null | grep -q "install ok installed"; then
        info "NVIDIA Container Toolkit already installed"
    else
        vfl_install_nvidia_container_toolkit
        info "Installed NVIDIA Container Toolkit"
    fi

    # Configure Docker runtime for NVIDIA
    info "Configuring Docker runtime for NVIDIA GPU passthrough..."
    vfl_configure_docker_nvidia_runtime tolerate-restart-failure

    # Verify GPU passthrough
    # Use sudo if docker group not active in this shell (fresh install).
    DOCKER_TEST_CMD="docker"
    docker info > /dev/null 2>&1 || DOCKER_TEST_CMD="sudo docker"

    info "Verifying GPU passthrough in Docker..."
    if $DOCKER_TEST_CMD run --rm --runtime=nvidia --gpus all "$VFL_CUDA_TEST_IMAGE" nvidia-smi > /dev/null 2>&1; then
        info "GPU passthrough verified — containers can access GPU"
    else
        warn "GPU passthrough test failed. You may need to:"
        warn "  1. Reboot the machine"
        warn "  2. Check that nvidia-ctk configured the Docker runtime correctly"
    fi

    # ── NIM cache directory ───────────────────────────────────────────────────

    section "NIM Cache"

    vfl_create_nim_cache
    info "NIM cache directory ready at ~/.cache/nim"

    # ── LoRA checkpoint merge runtime ────────────────────────────────────────
    #
    # TAO train emits an adapter-only checkpoint when LoRA is enabled. The
    # Blueprint merges that adapter with the gated base locally before the
    # full-precision baseline is evaluated through the Student NIM. Keep the
    # heavyweight ML stack isolated from the backend venv and shared across
    # workspaces on this Profile C host.

    section "LoRA Merge Runtime"

    MERGE_VENV="$HOME/.local/share/vlm-feedback-loop/merge-lora-venv"
    UV_CMD="$(command -v uv || true)"
    if [ -z "$UV_CMD" ]; then
        UV_CMD="$HOME/.local/bin/uv"
    fi
    if [ ! -x "$MERGE_VENV/bin/python" ]; then
        "$UV_CMD" venv --python python3.12 "$MERGE_VENV"
    fi
    "$UV_CMD" pip install --python "$MERGE_VENV/bin/python" \
        -r scripts/merge_lora_requirements.txt
    info "LoRA merge runtime ready at $MERGE_VENV"

    # ── NGC entitlement preflight (no image pulls) ─────────────────────
    #
    # Image pulls are handled by the Blueprint application at runtime
    # (local NIM deploy preflight). This script only verifies that the
    # NGC key can authenticate and access entitled NIM images.

    section "NGC Entitlement Check"

    DOCKER_CMD="docker"
    if ! docker info > /dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
        warn "Docker group not active in this shell (expected after fresh install)."
        warn "Using 'sudo docker' for entitlement check. After re-login, bare 'docker' will work."
    fi

    if [ -z "${NGC_API_KEY:-}" ]; then
        warn "NGC_API_KEY is not set."
        warn "Self-hosted NIM requires an NGC key with NVAIE-backed access to entitled images."
        warn "Get one at: https://org.ngc.nvidia.com/setup/api-key"
        echo ""
        # Only prompt interactively if stdin is a terminal. Silent read and a
        # masked hint: the key must never land in scrollback or session logs.
        if [ -t 0 ]; then
            read -rsp "$(echo -e "${CYAN}Paste your NGC API Key (or press Enter to skip):${NC} ")" NGC_INPUT || true
            echo ""
            if [ -n "${NGC_INPUT:-}" ]; then
                export NGC_API_KEY="$NGC_INPUT"
                info "NGC_API_KEY set for this session."
                info "To persist, add to ~/.vlm_feedback_loop/.env:"
                echo "  NGC_API_KEY=<the key you just pasted>"
            else
                warn "Skipped. Set NGC_API_KEY before running the Blueprint."
            fi
        else
            warn "Non-interactive shell — cannot prompt for NGC_API_KEY."
            warn "Set NGC_API_KEY in the environment and re-run, or add to ~/.vlm_feedback_loop/.env"
        fi
    fi

    if [ -n "${NGC_API_KEY:-}" ]; then
        info "NGC_API_KEY is set — checking registry access..."

        if vfl_ngc_docker_login "$DOCKER_CMD"; then
            info "NGC registry login succeeded"

            # Lightweight entitlement check — no full pull, just manifest inspect
            ENTITLEMENT_TEST_IMAGE="nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0"
            info "Checking entitlement to NIM image: $ENTITLEMENT_TEST_IMAGE"

            if $DOCKER_CMD manifest inspect "$ENTITLEMENT_TEST_IMAGE" > /dev/null 2>&1; then
                info "Entitlement check passed — image is accessible"
                info "NIM images will be pulled by the Blueprint on first deploy."
            else
                error "Cannot access $ENTITLEMENT_TEST_IMAGE"
                error "Possible causes:"
                error "  - NGC key missing NGC Catalog permission"
                error "  - User/account lacks self-hosting entitlement (NVAIE) for this NIM image"
                error "  - Wrong org/account context"
                warn "Fix the key/entitlement before running the Blueprint."
            fi
        else
            error "NGC registry login failed."
            error "Check that NGC_API_KEY is valid and has NGC Catalog access."
        fi
    fi

fi  # end GPU-only section

# ── Summary ───────────────────────────────────────────────────────────────────

section "Summary"

echo ""
echo "  Python 3.12 : $(python3.12 --version 2>&1)"
echo "  uv          : $(uv --version 2>&1 || $HOME/.local/bin/uv --version 2>&1)"
echo "  Node.js     : $(node --version 2>&1)"
echo "  pnpm        : $(pnpm --version 2>&1)"
echo "  gh          : $(gh --version 2>&1 | head -1)"
echo "  sqlite3     : $(sqlite3 --version 2>&1)"
echo "  git         : $(git --version 2>&1)"

if [ "$HAS_GPU" = true ]; then
    echo ""
    echo "  GPU(s)      : $GPU_COUNT × $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "  Max VRAM    : ${GPU_MEM_GB} GB (single GPU)"
    echo "  Compute cap : $GPU_COMPUTE_CAP"
    echo "  Driver      : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "  Docker      : $(docker --version 2>&1)"
    echo "  Toolkit     : $(dpkg -l nvidia-container-toolkit 2>/dev/null | grep ^ii | awk '{print $3}' || echo 'not found')"
    echo "  NIM cache   : ~/.cache/nim"
    echo "  LoRA merge  : ~/.local/share/vlm-feedback-loop/merge-lora-venv"
    echo "  NGC entitled: $([ -n "${NGC_API_KEY:-}" ] && echo 'checked' || echo 'key not set')"
else
    echo ""
    echo "  GPU         : none detected (Profile A/B mode)"
    echo "  Docker      : $(docker --version 2>&1 || echo 'not installed')"
fi

echo ""

# ── Post-install guidance ─────────────────────────────────────────────────────

if ! gh auth status &>/dev/null 2>&1; then
    warn "GitHub CLI is not authenticated. Run: gh auth login"
fi

if ! git config --global user.name &>/dev/null; then
    warn "Git user not configured. Run:"
    echo "  git config --global user.name \"Your Name\""
    echo "  git config --global user.email \"you@example.com\""
fi

echo ""
info "Next steps:"
echo "  1. Clone the repo:     git clone https://github.com/zerodefects/vlm-feedback-loop-public.git"
echo "  2. Install backend:    cd vlm-feedback-loop && uv sync"
echo "  3. Install frontend:   cd src/ui && pnpm install && cd ../.."
echo "  4. Bootstrap config:   uv run vlm-feedback-loop init"
echo "  5. Add secrets:        Edit ~/.vlm_feedback_loop/.env"
echo "     NGC_API_KEY and NVIDIA_API_KEY can be the same NGC Personal API Key"
echo "     if it has both NGC Catalog and Public API Endpoints scopes."
if [ "$HAS_GPU" = true ]; then
    echo "     - NGC_API_KEY=<your NGC key>      (for NIM container pulls + TAO login)"
    echo "     - NVIDIA_API_KEY=<same NGC key>    (for hosted NIM inference)"
else
    echo "     - NVIDIA_API_KEY=nvapi-<key>    (for hosted NIM inference)"
fi
echo "  6. Run both servers:   ./scripts/dev.sh"
echo "  7. Verify:             curl http://localhost:8000/v1/environment"

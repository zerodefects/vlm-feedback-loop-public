# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# setup-common.sh — shared Docker / NVIDIA Container Toolkit provisioning steps.
#
# Sourced (not executed) by the two setup entry points, which keep their
# distinct policies around these steps:
#   - scripts/setup-dev.sh   — full dev-toolchain bootstrap; GPU steps are
#     tolerant (warn and continue) so a CPU-only dev box still completes.
#   - scripts/setup-local.sh — strict pre-NIM checklist for a GPU host; the
#     same steps fail hard because local NIM deployment depends on them.
#
# Callers are expected to define `info` and `warn` log helpers; plain-echo
# fallbacks are provided so the functions also work standalone.

if ! declare -F info > /dev/null 2>&1; then
    info() { echo "[INFO] $*"; }
fi
if ! declare -F warn > /dev/null 2>&1; then
    warn() { echo "[WARN] $*"; }
fi

# CUDA base image used to verify GPU passthrough through Docker.
VFL_CUDA_TEST_IMAGE="nvidia/cuda:12.6.3-base-ubuntu24.04"

# Install Docker Engine from Docker's official apt repository (current
# documented method: keyring at /etc/apt/keyrings/docker.asc). Removes the
# distro/conflicting packages first. Callers check `command -v docker` and
# only invoke this when Docker is absent.
vfl_install_docker_engine() {
    info "Installing Docker Engine..."
    local pkg
    for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
        sudo apt-get remove -y "$pkg" > /dev/null 2>&1 || true
    done
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl > /dev/null
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin > /dev/null
}

# Add the current user to the docker group if not already a member.
vfl_ensure_docker_group() {
    if ! groups "$USER" | grep -q docker; then
        info "Adding $USER to docker group..."
        sudo usermod -aG docker "$USER"
        warn "You may need to log out and back in (or run 'newgrp docker') for docker group to take effect."
    fi
}

# Install the NVIDIA Container Toolkit from NVIDIA's apt repository. Callers
# check for an existing install and only invoke this when it is absent.
vfl_install_nvidia_container_toolkit() {
    info "Installing NVIDIA Container Toolkit..."
    sudo rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit > /dev/null
}

# Configure the Docker runtime for NVIDIA GPU passthrough and restart Docker.
# Pass "tolerate-restart-failure" to continue when the restart fails (dev
# bootstrap); by default a failed restart propagates (strict pre-NIM path).
vfl_configure_docker_nvidia_runtime() {
    sudo nvidia-ctk runtime configure --runtime=docker > /dev/null 2>&1
    if [ "${1:-}" = "tolerate-restart-failure" ]; then
        sudo systemctl restart docker 2>/dev/null || true
    else
        sudo systemctl restart docker
    fi
}

# Create the NIM cache directory (default ~/.cache/nim), world-writable:
# Embedding NIM (and the legacy NV-CLIP NIM) containers run as a non-root
# user and need write access to the cache mount, so chmod 777 ensures any
# container UID can read/write cached model artifacts.
vfl_create_nim_cache() {
    local cache_dir="${1:-$HOME/.cache/nim}"
    mkdir -p "$cache_dir"
    chmod 777 "$cache_dir"
}

# Log in to nvcr.io with $NGC_API_KEY using the given docker command
# ("docker" or "sudo docker"). Returns the login status.
vfl_ngc_docker_login() {
    local docker_cmd="${1:-docker}"
    echo "$NGC_API_KEY" | $docker_cmd login nvcr.io --username '$oauthtoken' --password-stdin > /dev/null 2>&1
}

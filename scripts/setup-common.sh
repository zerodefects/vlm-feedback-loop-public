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

# The shipped VLM NIM images use CUDA 13. NVIDIA's Linux compatibility floor
# for that runtime is the R580 branch at 580.65.06.
VFL_NIM_MIN_DRIVER_VERSION="580.65.06"
VFL_NIM_MIN_DOCKER_VERSION="29.4.0"
VFL_NIM_MIN_CTK_VERSION="1.19.0"

# Print the first complete X.Y.Z release in a command's version output.
# Package suffixes such as +azure-1 and -1 do not affect compatibility.
vfl_extract_version() {
    local version
    version=$(printf '%s\n' "${1:-}" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | sed -n '1p') || true
    [ -n "$version" ] || return 1
    printf '%s\n' "$version"
}

# Return success when the installed release ($1) meets the minimum ($2).
vfl_version_meets_minimum() {
    local installed minimum
    installed=$(vfl_extract_version "${1:-}") || return 1
    minimum=$(vfl_extract_version "${2:-}") || return 1
    printf '%s\n%s\n' "$minimum" "$installed" | LC_ALL=C sort -V -C
}

# Return success when the installed driver ($1) meets the local-NIM floor.
vfl_driver_meets_nim_minimum() {
    vfl_version_meets_minimum "$1" "$VFL_NIM_MIN_DRIVER_VERSION"
}

vfl_docker_meets_nim_minimum() {
    vfl_version_meets_minimum "$1" "$VFL_NIM_MIN_DOCKER_VERSION"
}

vfl_ctk_meets_nim_minimum() {
    vfl_version_meets_minimum "$1" "$VFL_NIM_MIN_CTK_VERSION"
}

# Query the Docker Engine daemon, using sudo when a newly added docker-group
# membership is not active in the current shell.
vfl_docker_server_version() {
    local output
    if output=$(docker version --format '{{.Server.Version}}' 2>/dev/null); then
        vfl_extract_version "$output"
    elif output=$(sudo docker version --format '{{.Server.Version}}' 2>/dev/null); then
        vfl_extract_version "$output"
    else
        return 1
    fi
}

vfl_nvidia_ctk_version() {
    local output
    output=$(nvidia-ctk --version 2>/dev/null) || return 1
    vfl_extract_version "$output"
}

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

# Install the NVIDIA Container Toolkit from an already configured apt source.
# Callers check for an existing install and only invoke this when it is absent.
vfl_install_nvidia_container_toolkit() {
    info "Installing NVIDIA Container Toolkit..."
    sudo apt-get update -qq
    if ! sudo apt-get install -y -qq nvidia-container-toolkit > /dev/null; then
        error "NVIDIA Container Toolkit is unavailable from configured apt sources."
        error "Follow https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html and rerun this setup."
        return 1
    fi
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

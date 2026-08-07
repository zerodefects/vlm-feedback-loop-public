#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Process selection helpers for scripts/dev.sh. Callers must define
# REPO_ROOT before sourcing this file.

_vlm_dev_backend_pids() {
    local self_pids=" $$ $PPID $BASHPID "
    local p cwd
    for p in $(pgrep -f "[v]lm_feedback_loop\\.main" 2>/dev/null); do
        case "$self_pids" in *" $p "*) continue;; esac
        cwd="$(readlink -f -- "/proc/$p/cwd" 2>/dev/null)" || continue
        case "$cwd/" in
            "$REPO_ROOT/"*) echo "$p" ;;
        esac
    done
}

_vlm_dev_vite_pids() {
    local self_pids=" $$ $PPID $BASHPID "
    local repo_pattern="${REPO_ROOT//\//\\/}"
    local p
    for p in $(
        pgrep -f "node .*${repo_pattern}/src/ui/node_modules/.*vite" 2>/dev/null
    ); do
        case "$self_pids" in *" $p "*) continue;; esac
        echo "$p"
    done
}

_vlm_dev_child_pids() {
    _vlm_dev_backend_pids
    _vlm_dev_vite_pids
}

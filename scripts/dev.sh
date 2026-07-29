#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Single-command dev launch: starts both backend (FastAPI) and frontend (Vite)
# for local development.
#
# Usage:
#   ./scripts/dev.sh
#   ./scripts/dev.sh --host 0.0.0.0  # backend override; requires IMAGE_ROOT
#
# Arguments are forwarded to the backend module runner. An explicit --host is
# also the backend's effective BIND_HOST, so non-loopback filesystem access is
# denied unless IMAGE_ROOT is configured.
#
# Prerequisites:
#   - uv (Python package manager)
#   - pnpm (Node package manager)
#   - ~/.vlm_feedback_loop/config.yaml must exist (run: uv run vlm-feedback-loop init)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/src/backend"
FRONTEND_DIR="$REPO_ROOT/src/ui"

# Colors for log prefixes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# ── Pre-flight: config must exist ────────────────────────────────────────────
# Without it the backend dies mid-startup on a raw pydantic ValidationError
# (WORKSPACE_ROOT has no default) while Vite starts fine — a confusing
# half-broken first run. Fail fast with the fix instead.
if [ ! -f "$HOME/.vlm_feedback_loop/config.yaml" ]; then
  echo "Missing ~/.vlm_feedback_loop/config.yaml." >&2
  echo "Run first:  uv run vlm-feedback-loop init --workspace-root ~/vlm-workspace" >&2
  exit 1
fi

# ── Pre-flight: terminate any prior dev stack ────────────────────────────────
#
# A previous `dev.sh` that ended abruptly (Ctrl-C swallowed by a `nohup`,
# terminal closed mid-shutdown, SIGTERM that didn't propagate through the
# `uv run` wrapper's pipeline) routinely leaves an orphan `vite` holding
# port 5173 or a uvicorn child still holding its project fcntl locks.
# Starting a fresh `dev.sh` on top of that state produces either a bind
# failure the new backend silently survives but the new `vite` can't, or a
# lock contention during `recover_local_deployments` that aborts startup
# mid-way through WAL-touching code — surfacing as a transient
# `sqlite3.DatabaseError: database disk image is malformed` on a
# project-list query. Avoid all of it by forcibly reaping every
# previous instance before binding anything new.
#
# Patterns are specific enough to avoid reaping unrelated processes:
# the backend entry point is `vlm_feedback_loop.main`; the frontend
# vite node process runs inside `src/ui/node_modules`.

# Print the PIDs of prior dev.sh supervisor shells for THIS repo,
# regardless of launch form. argv preserves the invocation-relative
# path — the documented `./scripts/dev.sh` runs as `bash ./scripts/dev.sh`
# — so an absolute-path pgrep pattern can never see it. Select candidates
# loosely, then confirm identity per-PID: the process must be a bash
# (`/proc/<pid>/comm`, so `vim scripts/dev.sh` never matches) whose argv1,
# resolved against the process's own cwd, canonicalizes to this repo's
# scripts/dev.sh. Never prints this process or its parents ($BASHPID
# covers the command-substitution subshell this function runs in, whose
# cmdline is identical to the supervisor's).
_prior_dev_supervisor_pids() {
    local self_pids=" $$ $PPID $BASHPID "
    local script_path
    script_path="$(readlink -f -- "$REPO_ROOT/scripts/dev.sh" 2>/dev/null)" || return 0
    local p argv1 cwd resolved
    for p in $(pgrep -f "scripts/dev\\.sh" 2>/dev/null); do
        case "$self_pids" in *" $p "*) continue;; esac
        [ "$(cat "/proc/$p/comm" 2>/dev/null)" = "bash" ] || continue
        argv1="$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | sed -n '2p')"
        [ -n "$argv1" ] || continue
        case "$argv1" in
            /*) resolved="$argv1" ;;
            *)  cwd="$(readlink "/proc/$p/cwd" 2>/dev/null)" || continue
                resolved="$cwd/$argv1" ;;
        esac
        resolved="$(readlink -f -- "$resolved" 2>/dev/null)" || continue
        [ "$resolved" = "$script_path" ] && echo "$p"
    done
}

_reap_prior_dev_stack() {
    local repo_pattern="${REPO_ROOT//\//\\/}"  # escape slashes for grep
    local found=0

    for pattern in \
        "vlm_feedback_loop\\.main" \
        "node .*${repo_pattern}/src/ui/node_modules/.*vite" ; do
        if pgrep -f "$pattern" >/dev/null 2>&1; then
            found=1
        fi
    done
    if [ -n "$(_prior_dev_supervisor_pids)" ]; then
        found=1
    fi

    # Never kill THIS process or this script's own bash parents.
    local self_pids="$$ $PPID"

    if [ "$found" -eq 1 ]; then
        echo -e "${YELLOW}[dev.sh]${NC} Previous dev stack detected — reaping before starting."
        # The prior dev.sh supervisor shell must die by SIGKILL, and
        # FIRST: SIGTERM would fire its cleanup() trap, whose safety-net
        # `pkill -KILL -f vlm_feedback_loop.main` / vite patterns run 3 s
        # later — long enough for THIS run to have started its own
        # backend and vite, which the dying trap then kills. SIGKILL
        # skips the trap entirely; its orphaned children are reaped by
        # the SIGTERM pass below.
        for p in $(_prior_dev_supervisor_pids); do
            kill -KILL "$p" 2>/dev/null || true
        done
        for pattern in \
            "vlm_feedback_loop\\.main" \
            "node .*${repo_pattern}/src/ui/node_modules/.*vite" ; do
            for p in $(pgrep -f "$pattern" 2>/dev/null); do
                case " $self_pids " in *" $p "*) continue;; esac
                kill -TERM "$p" 2>/dev/null || true
            done
        done

        # Graceful wait — up to 5 s — for processes to exit on SIGTERM.
        local waited=0
        while [ "$waited" -lt 5 ]; do
            sleep 1
            waited=$((waited + 1))
            local still=0
            for pattern in \
                "vlm_feedback_loop\\.main" \
                "node .*${repo_pattern}/src/ui/node_modules/.*vite" ; do
                if pgrep -f "$pattern" >/dev/null 2>&1; then
                    still=1
                    break
                fi
            done
            [ "$still" -eq 0 ] && break
        done

        # Force-kill anything still alive.
        for pattern in \
            "vlm_feedback_loop\\.main" \
            "node .*${repo_pattern}/src/ui/node_modules/.*vite" ; do
            for p in $(pgrep -f "$pattern" 2>/dev/null); do
                case " $self_pids " in *" $p "*) continue;; esac
                kill -KILL "$p" 2>/dev/null || true
            done
        done
        for p in $(_prior_dev_supervisor_pids); do
            kill -KILL "$p" 2>/dev/null || true
        done
        echo -e "${YELLOW}[dev.sh]${NC} Previous stack cleared."
    fi
}

_reap_prior_dev_stack

# ── Shutdown cleanup ─────────────────────────────────────────────────────────
#
# Track the two subshell PIDs explicitly so we can signal the actual
# long-running children (not just the bash shells that launched them).
# `kill 0` alone doesn't cover grandchildren that escaped the parent's
# process group — uvicorn under `uv run` is one such case.

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    local ec=$?
    # De-arm the trap so the forced-kill sequence below doesn't recurse.
    trap - EXIT INT TERM
    echo ""
    echo "Shutting down..."
    [ -n "$BACKEND_PID" ] && kill -TERM "$BACKEND_PID" 2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    # SIGTERM every descendant of this script — covers the grandchild
    # uvicorn + node processes the subshell wrappers spawned.
    pkill -TERM -P $$ 2>/dev/null || true
    # Give them 3 s to exit cleanly, then SIGKILL survivors.
    sleep 3
    pkill -KILL -P $$ 2>/dev/null || true
    # Safety net: anything matching our entry-point patterns that is
    # still alive (e.g., a process that re-parented to init).
    pkill -KILL -f "vlm_feedback_loop\\.main" 2>/dev/null || true
    pkill -KILL -f "node .*${REPO_ROOT}/src/ui/node_modules/.*vite" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Done."
    exit $ec
}
trap cleanup EXIT INT TERM

# Verify prerequisites
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. See https://docs.astral.sh/uv/" >&2
    exit 1
fi
if ! command -v pnpm &>/dev/null; then
    echo "Error: pnpm is not installed. See https://pnpm.io/" >&2
    exit 1
fi

# Install frontend deps if needed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${CYAN}[frontend]${NC} Installing dependencies..."
    (cd "$FRONTEND_DIR" && pnpm install)
fi

echo -e "${GREEN}Starting backend (FastAPI) on :8000 and frontend (Vite) on :5173${NC}"
echo ""

# Start backend
(
    cd "$REPO_ROOT"
    PYTHONPATH="$BACKEND_DIR:${PYTHONPATH:-}" \
    uv run python -m vlm_feedback_loop.main "$@" 2>&1 | \
        while IFS= read -r line; do echo -e "${CYAN}[backend]${NC}  $line"; done
) &
BACKEND_PID=$!

# Start frontend (Vite dev server proxies /v1/ to backend)
(
    cd "$FRONTEND_DIR"
    pnpm dev 2>&1 | \
        while IFS= read -r line; do echo -e "${GREEN}[frontend]${NC} $line"; done
) &
FRONTEND_PID=$!

# Wait for both
wait

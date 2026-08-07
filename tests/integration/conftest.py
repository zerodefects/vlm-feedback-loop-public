# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for integration tests.

The live-server fixtures (``backend_server`` / ``frontend_server``) spawn
real backend and Vite processes on the fixed ports 8000/5173.
Session-scoped so all test files share one pair (avoids port conflicts
between tests). Server output goes to log files under the session tmp
dir and is surfaced when startup fails.

In-process helpers (Settings factories, NIM-transport fakes, project
seeder) live in ``tests/support.py`` — import them with
``from support import ...`` from either suite.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# These fixtures spawn REAL servers on fixed ports (8000/5173), session-
# scoped. Under xdist parallelism every worker spawns/reuses/tears down
# the same pair: an early-finishing worker kills the servers others are
# mid-request on, their tests hit the 30s timeout, and pytest-timeout's
# thread method hard-exits the worker ("node down: Not properly
# terminated"). Integration tests therefore REQUIRE serial execution.
if os.environ.get("PYTEST_XDIST_WORKER", ""):
    import pytest as _pytest

    _pytest.exit(
        "tests/integration spawns live servers on fixed ports and must run "
        "serially: use `uv run pytest tests/integration/ -q -n 0` "
        "(see CLAUDE.md canonical commands).",
        returncode=3,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "src" / "backend"
FRONTEND_DIR = REPO_ROOT / "src" / "ui"

BACKEND_PORT = 8000
FRONTEND_PORT = 5173
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
FRONTEND_URL = f"http://127.0.0.1:{FRONTEND_PORT}"

STARTUP_TIMEOUT = 90  # seconds — generous headroom for cold starts
# (package imports + deployment-DB creation). Polling exits as soon as
# /health returns 200, so the extra headroom costs nothing when startup
# is fast.


def pytest_collection_modifyitems(items) -> None:
    """Give integration items a timeout that covers live-server startup.

    pytest-timeout's per-test clock (``--timeout=30`` in pyproject addopts)
    includes SESSION-FIXTURE SETUP, so the first test to request
    ``backend_server``/``frontend_server`` pays up to 2×STARTUP_TIMEOUT
    before its own body runs. Under ``--timeout-method=thread`` an
    overrun hard-exits pytest with NO teardown — orphaning the spawned
    servers on :8000/:5173 (observed live when new migrations made a
    26-project cold start exceed 30s). Tests that set their own
    ``@pytest.mark.timeout`` keep it.
    """
    margin = 2 * STARTUP_TIMEOUT + 60
    for item in items:
        if item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(margin))


# ── Live-server plumbing ────────────────────────────────────────────────────


def _port_holder_description(port: int) -> str:
    """Best-effort ``" by <comm> (pid <n>)"`` for the process listening
    on ``port``. Returns "" when it cannot be determined (e.g. the
    holder belongs to another user and ``ss`` hides the process info)."""
    try:
        out = subprocess.run(
            ["ss", "-ltnpH", f"sport = :{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    if not out:
        return ""
    m = re.search(r'users:\(\("([^"]+)",pid=(\d+)', out)
    if m:
        return f" by {m.group(1)} (pid {m.group(2)})"
    return f" ({out.splitlines()[0]})"


def _fail_if_port_taken(port: int, service: str) -> None:
    """Fail fast, with the holder named, when a fixed port is occupied.

    Without this probe the spawned server fails to bind (uvicorn exits;
    Vite silently hops to the next free port) while the health poll
    happily gets 200 from whatever process already owns the port — and
    the suite then runs against a stale server with confusing
    mid-suite failures. Probing *before* spawning turns that into one
    actionable error.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
        return
    except OSError:
        pass
    holder = _port_holder_description(port)
    pytest.fail(
        f"cannot start the integration {service}: port {port} is already "
        f"in use{holder}. The integration suite spawns its own servers on "
        f"fixed ports {BACKEND_PORT}/{FRONTEND_PORT} — stop the resident "
        f"process first (e.g. a ./scripts/dev.sh session: "
        f"`fuser -k {port}/tcp`) and re-run.",
        pytrace=False,
    )


def _tail(log_path: Path, max_bytes: int = 8192) -> str:
    """Last ``max_bytes`` of a server log, for failure messages."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return "<log unavailable>"


def _wait_until_healthy(
    url: str,
    proc: subprocess.Popen[bytes],
    log_path: Path,
    timeout: float = STARTUP_TIMEOUT,
) -> None:
    """Poll ``url`` until it returns 200; surface the server log on failure.

    Also detects the child dying during startup (crash, port bind race)
    instead of polling until the timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server for {url} exited with code {proc.returncode} before "
                f"becoming healthy; last output ({log_path}):\n{_tail(log_path)}"
            )
        try:
            resp = httpx.get(url, timeout=2)
            if resp.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise TimeoutError(
        f"Server at {url} did not become ready within {timeout}s; "
        f"last output ({log_path}):\n{_tail(log_path)}"
    )


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Send SIGTERM, wait, then SIGKILL if still alive."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)


def _spawn_logged(
    cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Spawn a server with stdout+stderr redirected to ``log_path``.

    Redirecting to a file (instead of PIPE) means a chatty server can
    never fill an undrained pipe buffer and deadlock; the log is
    surfaced by ``_wait_until_healthy`` when startup fails.
    """
    with open(log_path, "wb") as log:
        return subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )


@pytest.fixture(scope="session")
def _server_log_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("server-logs")


@pytest.fixture(scope="session")
def backend_server(_server_log_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    """Start the FastAPI backend and yield when healthy."""
    _fail_if_port_taken(BACKEND_PORT, "backend")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    # Give the child its own canonical config as well as a throwaway workspace.
    # A process-env WORKSPACE_ROOT cannot replace the required config file, and
    # relying on a developer's real HOME made this fixture fail on clean CI
    # runners while also exposing tests to unrelated local configuration.
    test_home = tmp_path_factory.mktemp("backend-home")
    workspace = tmp_path_factory.mktemp("workspace")
    config_dir = test_home / ".vlm_feedback_loop"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(f"WORKSPACE_ROOT: {workspace}\n")
    env["HOME"] = str(test_home)
    env["WORKSPACE_ROOT"] = str(workspace)
    # A disposable integration backend must inherit external credentials only
    # when the operator exported them explicitly for this test process.  An
    # absent parent variable must become an explicit empty process override;
    # otherwise the normal settings loader falls through to the operator's
    # canonical ~/.vlm_feedback_loop/.env and a nominal no-key test can make
    # real hosted NIM, NGC, Hugging Face, or TAO calls.
    for secret_name in (
        "NVIDIA_API_KEY",
        "NGC_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "TAO_API_KEY",
        "TAO_WORKSPACE_S3_ACCESS_KEY",
        "TAO_WORKSPACE_S3_SECRET_KEY",
    ):
        env[secret_name] = os.environ.get(secret_name, "")
    env["ALLOW_UI_SECRET_PERSIST"] = "false"
    if not env["NVIDIA_API_KEY"]:
        env["EMBEDDING_PROVIDER"] = "none"
    # Mount the test-only /v1/testing SSE-injection routes for this live
    # server — the SSE-recovery tests drive events through them. Production
    # leaves them unmounted (see main.py).
    env["VLM_ENABLE_TESTING_ROUTES"] = "1"

    log_path = _server_log_dir / "backend.log"
    proc = _spawn_logged(
        [sys.executable, "-m", "vlm_feedback_loop.main"],
        cwd=REPO_ROOT,
        log_path=log_path,
        env=env,
    )
    try:
        _wait_until_healthy(f"{BACKEND_URL}/health", proc, log_path)
        yield proc
    finally:
        _kill_process_group(proc)


@pytest.fixture(scope="session")
def frontend_server(backend_server, _server_log_dir: Path):  # noqa: ARG001 — depends on backend being up
    """Start the Vite dev server and yield when healthy."""
    _fail_if_port_taken(FRONTEND_PORT, "frontend dev server")

    log_path = _server_log_dir / "frontend.log"
    proc = _spawn_logged(["pnpm", "dev"], cwd=FRONTEND_DIR, log_path=log_path)
    try:
        _wait_until_healthy(FRONTEND_URL, proc, log_path)
        yield proc
    finally:
        _kill_process_group(proc)

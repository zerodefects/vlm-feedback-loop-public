# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Engine
from starlette.testclient import TestClient

from vlm_feedback_loop._defaults import DEFAULTS, SECRET_KEYS


@pytest.fixture()
def tmp_config_dir(tmp_path: Path) -> Path:
    """Temporary directory standing in for ``~/.vlm_feedback_loop/``."""
    d = tmp_path / ".vlm_feedback_loop"
    d.mkdir()
    return d


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    """Temporary workspace root directory."""
    d = tmp_path / "workspace"
    d.mkdir()
    return d


@pytest.fixture()
def patch_config_paths(tmp_config_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the config module's path constants to a temp directory."""
    import vlm_feedback_loop.config as cfg

    monkeypatch.setattr(cfg, "_CONFIG_DIR", tmp_config_dir)
    monkeypatch.setattr(cfg, "_CONFIG_FILE", tmp_config_dir / "config.yaml")
    monkeypatch.setattr(cfg, "_DEFAULT_ENV_FILE", tmp_config_dir / ".env")
    cfg.reset_settings()
    yield
    cfg.reset_settings()


@pytest.fixture()
def patch_cli_paths(tmp_config_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the CLI module's path constants to a temp directory."""
    import vlm_feedback_loop.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_CONFIG_DIR", tmp_config_dir)
    monkeypatch.setattr(cli_mod, "_CONFIG_FILE", tmp_config_dir / "config.yaml")
    monkeypatch.setattr(cli_mod, "_ENV_FILE", tmp_config_dir / ".env")


@pytest.fixture()
def write_config(tmp_config_dir: Path, tmp_workspace: Path) -> Callable[..., Path]:
    """Factory: write a ``config.yaml`` with optional overrides."""

    def _write(overrides: dict[str, Any] | None = None) -> Path:
        data: dict[str, Any] = {"WORKSPACE_ROOT": str(tmp_workspace)}
        for key, value in DEFAULTS.items():
            if key not in SECRET_KEYS:
                data[key] = value
        if overrides:
            data.update(overrides)
        config_file = tmp_config_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return config_file

    return _write


@pytest.fixture()
def isolated_config(patch_config_paths, write_config):
    """Hermetic config: redirect the config module to a temp
    ``~/.vlm_feedback_loop/`` with a valid ``config.yaml``.

    Required by any test that (directly or transitively) triggers
    ``load_settings`` — the loader hard-exits (``SystemExit: 1``) when the
    canonical config file is missing, which is exactly the state of a
    fresh CI runner. On dev boxes the real ``~/.vlm_feedback_loop/`` masks
    the problem, so missing isolation surfaces only in CI.
    """
    write_config()


@pytest.fixture()
def write_env(
    tmp_config_dir: Path,
) -> Callable[..., Path]:
    """Factory: write a ``.env`` file with key=value pairs."""

    def _write(values: dict[str, str], filename: str = ".env") -> Path:
        env_file = tmp_config_dir / filename
        with open(env_file, "w") as f:
            for k, v in values.items():
                f.write(f"{k}={v}\n")
        return env_file

    return _write


# ── Database fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_project_dir(tmp_path: Path) -> Path:
    """Temporary project directory for database tests."""
    d = tmp_path / "projects" / "test-project"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def project_engine(tmp_project_dir: Path) -> Engine:
    """Open a fresh project database and return the engine."""
    from vlm_feedback_loop.db.engine import open_project_db

    return open_project_db(tmp_project_dir)


@pytest.fixture()
def deployment_engine(tmp_workspace: Path) -> Engine:
    """Create a deployment database and return the engine."""
    from vlm_feedback_loop.db.engine import init_deployment_db

    return init_deployment_db(tmp_workspace)


# ── App client fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def test_app_client(tmp_path: Path) -> TestClient:
    """TestClient with a temp workspace, bypassing config.yaml.

    Settings are constructed directly and injected via dependency override.
    The engine cache is cleared on teardown.
    """
    from vlm_feedback_loop.config import Settings
    from vlm_feedback_loop.main import app
    from vlm_feedback_loop.routers.projects import get_current_settings
    from vlm_feedback_loop.services import project_service

    workspace = tmp_path / "workspace"
    # Build a minimal valid Settings bypassing load_settings
    non_secret = {k: v for k, v in DEFAULTS.items() if k not in SECRET_KEYS}
    settings = Settings(WORKSPACE_ROOT=str(workspace), **non_secret)

    app.dependency_overrides[get_current_settings] = lambda: settings
    project_service.clear_engine_cache()

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    app.dependency_overrides.clear()
    project_service.clear_engine_cache()


# ── SSE and background test fixtures ─────────────────────────────────────────


@pytest.fixture()
def sse_mgr():
    """Fresh SSEManager instance for unit tests."""
    from vlm_feedback_loop.services.sse import SSEManager

    return SSEManager()

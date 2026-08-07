# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for source-runner bind configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from vlm_feedback_loop import __version__, config
from vlm_feedback_loop import main as main_module
from vlm_feedback_loop.services.filesystem_service import check_path_allowed


def test_openapi_advertises_the_package_release_version() -> None:
    """Public API metadata must identify the same release as the package."""

    assert main_module.app.openapi()["info"]["version"] == __version__


def _capture_run(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    *,
    reload_settings: bool = False,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_uvicorn_run(
        app: str,
        *,
        host: str,
        port: int,
        reload: bool,
    ) -> None:
        settings = config.get_settings()
        captured.update(
            app=app,
            host=host,
            port=port,
            reload=reload,
            settings_host=settings.BIND_HOST,
            guard=check_path_allowed(Path("/outside-image-root"), settings),
        )
        if reload_settings:
            captured["reloaded_host"] = config.init_settings().BIND_HOST

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    prior_host = os.environ.get("BIND_HOST")
    host_was_set = "BIND_HOST" in os.environ
    try:
        main_module.run_server(argv)
    finally:
        if host_was_set:
            assert prior_host is not None
            os.environ["BIND_HOST"] = prior_host
        else:
            os.environ.pop("BIND_HOST", None)
        config.reset_settings()
    return captured


def test_nonloopback_cli_host_is_shared_filesystem_policy_authority(
    patch_config_paths,
    write_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real bind must drive every shared filesystem authorization check."""
    write_config(overrides={"BIND_HOST": "127.0.0.1", "IMAGE_ROOT": None})
    monkeypatch.delenv("BIND_HOST", raising=False)

    captured = _capture_run(monkeypatch, ["--host", "0.0.0.0"])

    assert captured["host"] == "0.0.0.0"
    assert captured["settings_host"] == captured["host"]
    assert captured["guard"] == (
        "Filesystem browsing is disabled. Configure IMAGE_ROOT to allow "
        "browsing when the backend is network-accessible."
    )


def test_loopback_cli_host_overrides_nonloopback_configuration(
    patch_config_paths,
    write_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit loopback bind and the policy singleton must also agree."""
    write_config(overrides={"BIND_HOST": "192.0.2.10", "IMAGE_ROOT": None})
    monkeypatch.setenv("BIND_HOST", "0.0.0.0")

    captured = _capture_run(monkeypatch, ["--host", "127.0.0.1"])

    assert captured["host"] == "127.0.0.1"
    assert captured["settings_host"] == captured["host"]
    assert captured["guard"] is None


def test_cli_host_survives_settings_reload(
    patch_config_paths,
    write_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later secret-persistence reload cannot restore the stale YAML host."""
    write_config(overrides={"BIND_HOST": "127.0.0.1", "IMAGE_ROOT": None})
    monkeypatch.delenv("BIND_HOST", raising=False)

    captured = _capture_run(
        monkeypatch,
        ["--host", "0.0.0.0"],
        reload_settings=True,
    )

    assert captured["host"] == "0.0.0.0"
    assert captured["settings_host"] == captured["host"]
    assert captured["reloaded_host"] == captured["host"]


def test_no_cli_host_preserves_configured_host(
    patch_config_paths,
    write_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an override, the configured bind remains the sole source."""
    write_config(overrides={"BIND_HOST": "192.0.2.10", "IMAGE_ROOT": "/images"})
    monkeypatch.delenv("BIND_HOST", raising=False)

    captured = _capture_run(monkeypatch, [])

    assert captured["host"] == "192.0.2.10"
    assert captured["settings_host"] == captured["host"]


@pytest.mark.parametrize(
    ("bind_host", "expected_host"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "127.0.0.1"),
        ("::", "127.0.0.1"),
        ("::1", "[::1]"),
    ],
)
def test_print_backend_url_uses_effective_bind_configuration(
    patch_config_paths,
    write_config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bind_host: str,
    expected_host: str,
) -> None:
    """The Vite handshake follows configured ports and reachable host syntax."""
    write_config(
        overrides={
            "BIND_HOST": bind_host,
            "BIND_PORT": 8123,
            "IMAGE_ROOT": "/images",
        }
    )
    monkeypatch.delenv("BIND_HOST", raising=False)
    monkeypatch.delenv("BIND_PORT", raising=False)
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *args, **kwargs: pytest.fail("URL query must not start Uvicorn"),
    )

    try:
        main_module.run_server(["--print-backend-url"])
    finally:
        config.reset_settings()

    assert capsys.readouterr().out.strip() == f"http://{expected_host}:8123"


def test_print_backend_url_honors_cli_port_override(
    patch_config_paths,
    write_config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit runner port is also the Vite proxy port."""
    write_config(overrides={"BIND_PORT": 8000})
    monkeypatch.delenv("BIND_PORT", raising=False)
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *args, **kwargs: pytest.fail("URL query must not start Uvicorn"),
    )

    try:
        main_module.run_server(["--port", "8124", "--print-backend-url"])
    finally:
        config.reset_settings()

    assert capsys.readouterr().out.strip() == "http://127.0.0.1:8124"

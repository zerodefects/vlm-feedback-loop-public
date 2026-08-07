# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level database cleanup for server and CLI entry points."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_fails", [False, True])
async def test_lifespan_always_closes_resources(monkeypatch, startup_fails):
    from vlm_feedback_loop import main

    events: list[str] = []

    @asynccontextmanager
    async def startup(_app):
        events.append("startup")
        if startup_fails:
            raise RuntimeError("startup failed")
        yield
        events.append("startup_exit")

    async def cancel_all(*, grace_seconds):
        assert grace_seconds == 5.0
        events.append("cancel")

    async def close_sse():
        events.append("sse")

    monkeypatch.setattr(main, "_application_lifespan", startup)
    monkeypatch.setattr(main.background_manager, "cancel_all", cancel_all)
    monkeypatch.setattr(main.sse_manager, "close_all", close_sse)
    monkeypatch.setattr(
        main, "close_project_resources", lambda: events.append("project_db")
    )
    monkeypatch.setattr(
        main,
        "close_deployment_db_resources",
        lambda: events.append("deployment_db"),
    )

    if startup_fails:
        with pytest.raises(RuntimeError, match="startup failed"):
            async with main.lifespan(main.app):
                pass
        assert events == ["startup", "cancel", "sse", "project_db", "deployment_db"]
    else:
        async with main.lifespan(main.app):
            events.append("running")
        assert events == [
            "startup",
            "running",
            "startup_exit",
            "cancel",
            "sse",
            "project_db",
            "deployment_db",
        ]


@pytest.mark.asyncio
async def test_lifespan_attempts_every_cleanup_when_task_cancellation_fails(
    monkeypatch,
):
    from vlm_feedback_loop import main

    events: list[str] = []

    @asynccontextmanager
    async def startup(_app):
        yield

    async def cancel_all(*, grace_seconds):
        assert grace_seconds == 5.0
        events.append("cancel")
        raise RuntimeError("cancellation failed")

    async def close_sse():
        events.append("sse")

    monkeypatch.setattr(main, "_application_lifespan", startup)
    monkeypatch.setattr(main.background_manager, "cancel_all", cancel_all)
    monkeypatch.setattr(main.sse_manager, "close_all", close_sse)
    monkeypatch.setattr(
        main, "close_project_resources", lambda: events.append("project_db")
    )
    monkeypatch.setattr(
        main,
        "close_deployment_db_resources",
        lambda: events.append("deployment_db"),
    )

    with pytest.raises(RuntimeError, match="cancellation failed"):
        async with main.lifespan(main.app):
            pass

    assert events == ["cancel", "sse", "project_db", "deployment_db"]


@pytest.mark.asyncio
async def test_lifespan_cleanup_failure_does_not_mask_startup_failure(monkeypatch):
    from vlm_feedback_loop import main

    events: list[str] = []

    @asynccontextmanager
    async def startup(_app):
        raise ValueError("primary startup failure")
        yield  # pragma: no cover - required async-generator shape

    async def cancel_all(*, grace_seconds):
        assert grace_seconds == 5.0
        events.append("cancel")
        raise RuntimeError("cleanup failure")

    async def close_sse():
        events.append("sse")

    monkeypatch.setattr(main, "_application_lifespan", startup)
    monkeypatch.setattr(main.background_manager, "cancel_all", cancel_all)
    monkeypatch.setattr(main.sse_manager, "close_all", close_sse)
    monkeypatch.setattr(
        main, "close_project_resources", lambda: events.append("project_db")
    )
    monkeypatch.setattr(
        main,
        "close_deployment_db_resources",
        lambda: events.append("deployment_db"),
    )

    with pytest.raises(ValueError, match="primary startup failure"):
        async with main.lifespan(main.app):
            pass

    assert events == ["cancel", "sse", "project_db", "deployment_db"]


@pytest.mark.parametrize("handler_exits", [False, True])
def test_cli_always_closes_deployment_database(monkeypatch, handler_exits):
    from vlm_feedback_loop import cli
    from vlm_feedback_loop.db import engine as engine_module

    def handler(_args):
        if handler_exits:
            raise SystemExit(2)

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self, args: argparse.Namespace(func=handler),
    )
    closer = Mock()
    monkeypatch.setattr(engine_module, "close_deployment_db_resources", closer)

    if handler_exits:
        with pytest.raises(SystemExit) as exc_info:
            cli.main([])
        assert exc_info.value.code == 2
    else:
        cli.main([])

    closer.assert_called_once_with()

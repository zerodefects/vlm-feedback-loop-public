# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from vlm_feedback_loop import __version__
from vlm_feedback_loop.config import Settings, get_settings, init_settings
from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.engine import (
    close_deployment_db_resources,
    open_project_db,
)
from vlm_feedback_loop.db.models.run import ACTIVE_RUN_STATUSES, RunRecord
from vlm_feedback_loop.services.background import background_manager
from vlm_feedback_loop.services.locks import (
    ProjectLockedError,
    acquire_project_lock,
)
from vlm_feedback_loop.services.pagination import InvalidCursorError
from vlm_feedback_loop.services.project_service import (
    AlreadyArchivedError,
    NotArchivedError,
    ProjectArchivedError,
    ProjectBusyError,
    close_project_resources,
    projects_root,
    set_project_engine,
)
from vlm_feedback_loop.services.sse import sse_manager

logger = logging.getLogger("vlm_feedback_loop.main")


# ── Startup recovery ────────────────────────────────────────────────────────


def _reconstruct_canceled_batch_counters(
    session: Session,
    run: RunRecord,
    settings: Settings,
) -> dict[str, int]:
    """Count authoritative terminal items before finalizing restart cancel."""
    from vlm_feedback_loop.services.batch_label_service import (
        reconcile_batch_items,
        summarize_batch_outcomes,
    )
    from vlm_feedback_loop.services.run_config import snapshot_run_config

    if not isinstance(run.metrics, dict):
        raise RuntimeError("missing batch input snapshot")
    raw_input_keys = run.metrics.get("input_keys")
    if not isinstance(raw_input_keys, list):
        raise RuntimeError("invalid batch input snapshot")
    untyped_input_keys = cast("list[Any]", raw_input_keys)
    if not all(isinstance(key, str) for key in untyped_input_keys):
        raise RuntimeError("invalid batch input snapshot")
    input_keys = list(cast("list[str]", untyped_input_keys))
    if len(input_keys) != len(set(input_keys)) or run.examples_total != len(input_keys):
        raise RuntimeError("ambiguous batch input snapshot")
    run_config = snapshot_run_config(
        session,
        run.project_id,
        run,
        example_keys=input_keys,
        settings=settings,
    )
    outcomes = reconcile_batch_items(
        session,
        run.project_id,
        run.run_id,
        input_keys,
        run_config=run_config,
        mode="canceled_recovery",
    )
    counts = summarize_batch_outcomes(outcomes)
    return {
        "examples_succeeded": counts.succeeded,
        "examples_schema_invalid": counts.schema_invalid,
        "examples_timeout": counts.timeout,
        "examples_endpoint_error": counts.endpoint_error,
    }


def _recover_interrupted_runs(settings: Settings) -> list[tuple[str, str]]:
    """Scan all project DBs for non-terminal runs and apply recovery.

    Runs BEFORE the app serves requests:
    - Evaluation runs in queued/running/canceling → failed
    - Batch runs in queued/running → queued with recovered_from_restart=True,
      then auto-resumed from the next unprocessed example.
      The (project_id, run_id) pairs to resume are returned so the async
      lifespan can dispatch the executor once the event loop is running.
    - Batch runs in canceling with persisted cancel intent and coherent item
      lineage → canceled with exact authoritative partial counters. Missing
      intent or invalid lineage → failed. Every outcome prevents a restart
      from leaving the run wedged in a non-terminal state.
    - Batch runs in paused → unchanged after any legacy runtime snapshot is
      materialized, so later catalog edits cannot alter an explicit Resume.
    """
    projects_dir = projects_root(settings.WORKSPACE_ROOT)
    resume_targets: list[tuple[str, str]] = []
    if not projects_dir.exists():
        return resume_targets

    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir():
            continue
        db_path = entry / "project.db"
        if not db_path.exists():
            continue
        # Archived projects are paused — recovery is a no-op for them.
        # The marker file is the lazy index over Project.archived_at; see
        # services/project_service.py:archive_project.
        if (entry / ".archived").exists():
            logger.debug("Skipping recovery for archived project %s", entry.name)
            continue

        try:
            acquire_project_lock(entry)
            engine = open_project_db(entry)
        except Exception as exc:
            logger.warning("Skipping recovery for %s: %s", entry.name, exc)
            continue

        set_project_engine(entry.name, engine)

        with Session(engine) as session:
            non_terminal = (
                session.query(RunRecord)
                .filter(RunRecord.status.in_(ACTIVE_RUN_STATUSES | {"paused"}))
                .all()
            )

            now = utc_now()
            for run in non_terminal:
                if run.run_type == "evaluation_run":
                    run.status = "failed"
                    run.status_reason = "backend_restart_interrupted"
                    run.completed_at = now
                    logger.info("Recovery: eval run %s → failed", run.run_id)
                elif run.run_type == "batch_label_run":
                    try:
                        from vlm_feedback_loop.services.run_config import (
                            ensure_runtime_config_snapshot,
                        )

                        with session.begin_nested():
                            ensure_runtime_config_snapshot(
                                session,
                                run.project_id,
                                run,
                                settings,
                            )
                            session.flush()
                    except Exception:
                        run.status = "failed"
                        run.status_reason = "batch_recovery_state_invalid"
                        run.completed_at = now
                        logger.exception(
                            "Recovery: batch run %s has no recoverable runtime "
                            "configuration",
                            run.run_id,
                        )
                        continue
                    if run.status in ("queued", "running"):
                        run.status = "queued"
                        run.recovered_from_restart = True
                        # entry.name is the project_id (workspace dir == id);
                        # the executor idempotently skips already-processed
                        # examples via their OperationRecord.
                        resume_targets.append((entry.name, run.run_id))
                        logger.info(
                            "Recovery: batch run %s → queued (recovered, will "
                            "auto-resume)",
                            run.run_id,
                        )
                    elif run.status == "canceling":
                        if run.cancel_requested_at:
                            try:
                                # Isolate malformed durable state to this run;
                                # later recovery candidates must still start.
                                with session.begin_nested():
                                    counts = _reconstruct_canceled_batch_counters(
                                        session, run, settings
                                    )
                            except Exception:
                                run.status = "failed"
                                run.status_reason = "batch_recovery_state_invalid"
                                logger.exception(
                                    "Recovery: batch run %s has invalid durable state",
                                    run.run_id,
                                )
                            else:
                                run.status = "canceled"
                                run.status_reason = "canceled_on_restart"
                                for field, value in counts.items():
                                    setattr(run, field, value)
                        else:
                            run.status = "failed"
                            run.status_reason = "backend_restart_interrupted"
                        run.completed_at = now
                        logger.info(
                            "Recovery: batch run %s → %s (was canceling)",
                            run.run_id,
                            run.status,
                        )
                    # paused stays paused — no action

            if non_terminal:
                session.commit()

    return resume_targets


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _application_lifespan(
    app: FastAPI,  # noqa: ARG001 — FastAPI lifespan signature
) -> AsyncGenerator[None]:
    """Run startup and hold the application context until shutdown."""
    settings = get_settings()

    # Configure structured logging before any log output
    from vlm_feedback_loop.services.logging_config import setup_logging

    setup_logging(settings.LOG_LEVEL)

    logger.info(
        "VLM Feedback Loop backend starting — workspace: %s",
        settings.WORKSPACE_ROOT,
    )

    # Configure the process-global hosted RPM ceiling (0 = disabled). Gates
    # every build.nvidia.com request across all concurrent work to one account
    # budget; complements the per-model adaptive pacer in http_client.
    from vlm_feedback_loop.services import hosted_rate_limiter

    hosted_rate_limiter.configure(
        settings.HOSTED_GLOBAL_RPM, settings.HOSTED_NIM_BASE_URL
    )

    # Foreground-priority hold: interactive default on; long AutoRun/batch
    # runs disable it so continuous proposal streams cannot starve
    # background evaluation dispatch (see priority.set_enabled).
    from vlm_feedback_loop.services.priority import priority_dispatch

    priority_dispatch.set_enabled(settings.FOREGROUND_HOLD_ENABLED)
    # AutoRun operators depend on this switch (see priority.set_enabled);
    # log the effective state so a restart's config pickup is verifiable
    # from the log instead of process archaeology.
    logger.info(
        "Foreground-priority hold %s",
        "enabled" if settings.FOREGROUND_HOLD_ENABLED else "disabled",
    )

    # Startup: recover interrupted runs across all projects
    batch_resume_targets = _recover_interrupted_runs(settings)

    # Auto-resume batch labeling runs that were interrupted mid-flight.
    # Recovery flipped them to queued+recovered_from_restart; now that the event
    # loop is running, dispatch the executor for each. The executor idempotently
    # skips examples already processed before the restart.
    if batch_resume_targets:
        from vlm_feedback_loop.services.batch_label_service import (
            dispatch_batch_label_run,
        )

        for project_id, run_id in batch_resume_targets:
            dispatch_batch_label_run(project_id, run_id, settings)
            logger.info("Auto-resumed batch run %s after restart", run_id)

    # Startup: recover interrupted TAO jobs (submitting + null external id → failed)
    from vlm_feedback_loop.services.tao_job_service import (
        recover_interrupted_tao_jobs,
    )

    recovered_tao_jobs = recover_interrupted_tao_jobs(settings)

    # Base-experiment pulls stage multi-GB checkpoints in a temporary
    # directory and are not resumable across a backend restart. Fail any
    # interrupted tracked run so Student Training offers a clean, idempotent
    # retry instead of polling a permanent "running" row.
    from vlm_feedback_loop.services.tao_base_experiment_provisioning_run_service import (
        recover_interrupted_provisioning_runs,
    )

    recover_interrupted_provisioning_runs(settings)

    # A provisional TrainingSuite owns the browser-visible provisioning /
    # preparation steps. Its continuation is also in-process, so a restart
    # fails that setup explicitly instead of leaving Training Jobs spinning.
    from vlm_feedback_loop.services.training_suite_service import (
        recover_interrupted_training_suite_setups,
    )

    recover_interrupted_training_suite_setups(settings)

    # For each recovered chain job, run the same halt/advance/roll-up flow the
    # poller uses so a submission interrupted mid-chain doesn't strand its
    # TrainingSuite in "running" (recover_interrupted_tao_jobs is sync and
    # cannot await this). engines were cached by _recover_interrupted_runs.
    if recovered_tao_jobs:
        from vlm_feedback_loop.services.project_service import get_project_engine
        from vlm_feedback_loop.services.tao_polling_service import (
            handle_terminal_failure,
        )

        for desc in recovered_tao_jobs:
            if desc.get("chain_id") is None:
                continue  # non-chain job: failed status is already terminal
            engine = get_project_engine(desc["project_id"], settings.WORKSPACE_ROOT)
            if engine is None:
                continue
            await handle_terminal_failure(
                desc["project_id"],
                desc["tao_job_id"],
                chain_id=desc["chain_id"],
                chain_sequence=desc["chain_sequence"],
                action=desc["action"],
                terminal_status="failed",
                engine=engine,
                settings=settings,
                emit_root_failed=False,
            )

    # Startup: kick off the TAO polling loop. Idempotent — the
    # worker claims a single background-manager slot and drives chain
    # advancement, artifact retrieval, and SSE emission from there.
    from vlm_feedback_loop.services.tao_polling_service import start_tao_polling

    start_tao_polling(settings)

    # Startup: recover local NIM deployments (check container states)
    from vlm_feedback_loop.services.local_nim_service import (
        recover_local_deployments,
    )

    await recover_local_deployments(settings.WORKSPACE_ROOT, settings)

    # Startup: recover incomplete CLIP embedding computations
    from vlm_feedback_loop.services.clip_embedding_service import (
        recover_embedding_tasks,
    )

    await recover_embedding_tasks(settings)

    # Startup: recover incomplete ingest pHash sweeps.
    # Mirrors recover_embedding_tasks shape — walks projects, skips
    # ``.archived`` markers, triggers the sweeper for any project with at
    # least one Example row whose phash IS NULL. A backend crash mid-sweep
    # leaves rows in the queue; this hook drains them on next startup.
    from vlm_feedback_loop.services.ingest_sweeper_service import (
        recover_ingest_tasks,
    )

    await recover_ingest_tasks(settings)

    # Startup: fail dataset exports interrupted mid-build. The archive
    # build is an in-process background task; a restart kills it, so any
    # row still `running` is marked failed and its partial artifact files
    # are deleted (exports are not resumable — the SME retries).
    from vlm_feedback_loop.services.dataset_export_service import (
        recover_dataset_exports,
    )

    recover_dataset_exports(settings)

    yield


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Own teardown even when startup or an earlier cleanup step fails."""
    primary_exception: BaseException | None = None
    try:
        async with _application_lifespan(app):
            yield
    except BaseException as exc:
        primary_exception = exc
        raise
    finally:
        # Cancel borrowers before disposing their process-scoped engines.
        logger.info("VLM Feedback Loop backend shutting down")
        cleanup_failure: BaseException | None = None
        try:
            await background_manager.cancel_all(grace_seconds=5.0)
        except BaseException as exc:
            cleanup_failure = exc
            logger.exception("Background-task shutdown failed")
        try:
            await sse_manager.close_all()
        except BaseException as exc:
            cleanup_failure = cleanup_failure or exc
            logger.exception("SSE shutdown failed")
        try:
            close_project_resources()
        except BaseException as exc:
            cleanup_failure = cleanup_failure or exc
            logger.exception("Project-database shutdown failed")
        try:
            close_deployment_db_resources()
        except BaseException as exc:
            cleanup_failure = cleanup_failure or exc
            logger.exception("Deployment-database shutdown failed")
        logger.info("Shutdown complete")

        # A cleanup failure should fail an otherwise clean shutdown, but it
        # must never obscure the startup/request exception that caused teardown.
        if primary_exception is None and cleanup_failure is not None:
            raise cleanup_failure


# ── Application ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Interactive VLM Feedback Loop",
    version=__version__,
    lifespan=lifespan,
)

# Middleware (added bottom-to-top: CORS runs first on request)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ProjectLockedError)
async def project_locked_handler(
    request: Request,  # noqa: ARG001 — FastAPI exception handler signature
    exc: ProjectLockedError,  # noqa: ARG001 — FastAPI exception handler signature
) -> JSONResponse:
    """Return 409 when a project is already open in another process."""
    return JSONResponse(
        status_code=409,
        content={"detail": "This project is already open in another process."},
    )


@app.exception_handler(InvalidCursorError)
async def invalid_cursor_handler(
    request: Request,  # noqa: ARG001 — FastAPI exception handler signature
    exc: InvalidCursorError,
) -> JSONResponse:
    """Return 400 for malformed pagination cursors on ANY list endpoint.

    ``decode_cursor`` is shared by multiple list services; handling it here
    keeps malformed client cursors consistently mapped to 400.
    """
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(AlreadyArchivedError)
async def already_archived_handler(
    request: Request,  # noqa: ARG001
    exc: AlreadyArchivedError,  # noqa: ARG001
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": "This project is already archived.",
            "code": "already_archived",
        },
    )


@app.exception_handler(NotArchivedError)
async def not_archived_handler(
    request: Request,  # noqa: ARG001
    exc: NotArchivedError,  # noqa: ARG001
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": "This project is not archived.", "code": "not_archived"},
    )


@app.exception_handler(ProjectBusyError)
async def project_busy_handler(
    request: Request,  # noqa: ARG001
    exc: ProjectBusyError,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": "This project has in-progress work. Wait for it to finish.",
            "code": "project_busy",
            "reasons": exc.reasons,
        },
    )


@app.exception_handler(ProjectArchivedError)
async def project_archived_handler(
    request: Request,  # noqa: ARG001
    exc: ProjectArchivedError,  # noqa: ARG001
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": "This project is archived. Unarchive it first.",
            "code": "project_archived",
        },
    )


# API router
api_router = APIRouter(prefix="/v1")

import vlm_feedback_loop.services.deployment_handoff_generator  # noqa: E402, I001  # pyright: ignore[reportUnusedImport] — registers deployment_handoff generator
import vlm_feedback_loop.services.local_nim_deploy_generator  # noqa: E402  # pyright: ignore[reportUnusedImport] — registers student_nim_deploy generator
import vlm_feedback_loop.services.missing_files_generator  # noqa: E402  # pyright: ignore[reportUnusedImport] — registers missing_files generator
import vlm_feedback_loop.services.nim_issue_generator  # noqa: E402  # pyright: ignore[reportUnusedImport] — registers nim_issue generator
import vlm_feedback_loop.services.nim_setup_generator  # noqa: E402  # pyright: ignore[reportUnusedImport] — registers nim_setup generator
import vlm_feedback_loop.services.tao_issue_generator  # noqa: E402  # pyright: ignore[reportUnusedImport] — registers tao_issue generator
import vlm_feedback_loop.services.tao_setup_generator  # noqa: E402, F401  # pyright: ignore[reportUnusedImport] — registers tao_setup generator

from vlm_feedback_loop.routers.examples import examples_router  # noqa: E402
from vlm_feedback_loop.routers.filesystem import filesystem_router  # noqa: E402
from vlm_feedback_loop.routers.guidance import guidance_router  # noqa: E402
from vlm_feedback_loop.routers.local_nim import local_nim_router  # noqa: E402
from vlm_feedback_loop.routers.model_configs import model_configs_router  # noqa: E402
from vlm_feedback_loop.routers.nim import nim_endpoints_router, nim_router  # noqa: E402
from vlm_feedback_loop.routers.projects import router as projects_router  # noqa: E402
from vlm_feedback_loop.routers.batch_label_runs import batch_label_runs_router  # noqa: E402
from vlm_feedback_loop.routers.dataset_exports import dataset_exports_router  # noqa: E402
from vlm_feedback_loop.routers.evaluation_runs import evaluation_runs_router  # noqa: E402
from vlm_feedback_loop.routers.labels import labels_router  # noqa: E402
from vlm_feedback_loop.routers.proposals import proposals_router  # noqa: E402
from vlm_feedback_loop.routers.review_selector import review_selector_router  # noqa: E402
from vlm_feedback_loop.routers.secrets import secrets_router  # noqa: E402
from vlm_feedback_loop.routers.student_models import (  # noqa: E402
    student_models_router,
)
from vlm_feedback_loop.routers.tao_base_experiment_provisioning import (  # noqa: E402
    tao_base_experiment_provisioning_router,
)
from vlm_feedback_loop.routers.tao_jobs import tao_jobs_router  # noqa: E402
from vlm_feedback_loop.routers.training_preflight import (  # noqa: E402
    training_preflight_router,
)
from vlm_feedback_loop.routers.training_suites import (  # noqa: E402
    training_suites_router,
)

api_router.include_router(projects_router)
# The testing router injects arbitrary SSE events into any project's live
# stream — a test harness only. It is NOT mounted in production: a network
# client could otherwise spoof run.completed/failed events into the SME's UI.
# Gated on an env var read at import (routers mount before Settings init);
# the integration SSE-recovery fixture sets it. Default: not mounted.
if os.environ.get("VLM_ENABLE_TESTING_ROUTES") == "1":
    from vlm_feedback_loop.routers.testing import (
        router as testing_router,
    )

    api_router.include_router(testing_router)
    logger.warning(
        "Test-only /v1/testing SSE-injection routes are MOUNTED "
        "(VLM_ENABLE_TESTING_ROUTES=1) — do not use in production."
    )
api_router.include_router(nim_router)
api_router.include_router(secrets_router)
api_router.include_router(nim_endpoints_router)
api_router.include_router(model_configs_router)
api_router.include_router(local_nim_router)
api_router.include_router(guidance_router)
api_router.include_router(examples_router)
api_router.include_router(filesystem_router)
api_router.include_router(review_selector_router)
api_router.include_router(proposals_router)
api_router.include_router(labels_router)
api_router.include_router(evaluation_runs_router)
api_router.include_router(batch_label_runs_router)
api_router.include_router(dataset_exports_router)
api_router.include_router(tao_jobs_router)
api_router.include_router(tao_base_experiment_provisioning_router)
api_router.include_router(training_suites_router)
api_router.include_router(training_preflight_router)
api_router.include_router(student_models_router)
app.include_router(api_router)


# ── Health ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# ── Dev runner ───────────────────────────────────────────────────────────────


def run_server(argv: Sequence[str] | None = None) -> None:
    """Run Uvicorn with one authoritative bind host.

    Filesystem authorization reads ``Settings.BIND_HOST``. Promote an explicit
    source-runner override into the process environment before loading Settings
    so Uvicorn and every policy consumer share the same value, including after
    a later Settings reload.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the backend dev server")
    parser.add_argument("--env-file", default=None, help="Path to .env file")
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    parser.add_argument(
        "--print-backend-url",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    cli_args = parser.parse_args(argv)

    if cli_args.host:
        os.environ["BIND_HOST"] = cli_args.host

    # Pre-load settings so fail-fast happens before Uvicorn starts.
    settings = init_settings(cli_env_file=cli_args.env_file)
    port = cli_args.port if cli_args.port is not None else settings.BIND_PORT

    if cli_args.print_backend_url:
        # dev.sh asks the authoritative runner for the URL Vite should proxy.
        # Wildcard binds are reachable locally through loopback; IPv6 literals
        # need brackets when rendered as a URL.
        proxy_host = settings.BIND_HOST
        if proxy_host in {"0.0.0.0", "::"}:
            proxy_host = "127.0.0.1"
        elif ":" in proxy_host and not proxy_host.startswith("["):
            proxy_host = f"[{proxy_host}]"
        print(f"http://{proxy_host}:{port}")
        return

    uvicorn.run(
        "vlm_feedback_loop.main:app",
        host=settings.BIND_HOST,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run_server()

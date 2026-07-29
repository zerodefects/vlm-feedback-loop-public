#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end real-dataset TAO quality run on Rock Paper Scissors.

Builds (or reuses) a Blueprint project from the complete 372-image validation
dataset (124 each of rock/paper/scissors at 300×300), imports its ground-truth
labels, applies a fixed train/eval split, and submits a real cosmos-rl SFT
training suite.

The repository's bundled ``deploy/example-images`` sample is also RPS, but it
contains only 15 images (5 per class) for first-run product exploration. It is
intentionally too small for this quality gate. Build mode therefore defaults
to the developer dataset at ``~/rps-test-set``.

Two modes:
  - **Build mode** (default): bootstraps a fresh project, ingests, and
    writes Verified labels in-process.
  - **Lookup mode** (``--project-id``): reuses an existing labeled project
    and only submits or resumes the training suite through the public API.

Quantization: ``--quantization`` accepts a comma-separated
subset of ``FP8_DYNAMIC, W8A8, W8A16, W4A16`` (case-insensitive). Each
scheme adds a ``quantize → evaluate`` job pair to the chain after the
baseline ``train → evaluate``. Empty list = baseline only.

Base model selection (``--base-model``): chooses ``2b`` (default) or ``8b``
Cosmos Reason 2 base. Provisioning patches BOTH ModelConfigs the first
time it runs in a given project so subsequent invocations on the same
project don't need to re-provision.

Schema is a single core enum ``gesture ∈ {rock, paper, scissors}``.

Usage::

    # Build mode, 2B baseline only:
    uv run python scripts/rps_e2e.py \
        --auto-provision-base-experiments \
        --train-deadline-s 5400 \
        --rps-root ~/rps-test-set

    # Lookup mode — 2B production matrix into an existing labeled project:
    uv run python scripts/rps_e2e.py \
        --auto-provision-base-experiments \
        --base-model 2b \
        --quantization fp8_dynamic,w8a16 \
        --project-id $PROJECT_2B \
        --rps-root ~/rps-test-set

    # Lookup mode — sacrificial contract-mismatch sibling (core_only):
    uv run python scripts/rps_e2e.py \
        --base-model 2b \
        --export-field-mode core_only \
        --project-id $PROJECT_2B \
        --rps-root ~/rps-test-set

    # Lookup mode — 8B production matrix:
    uv run python scripts/rps_e2e.py \
        --auto-provision-base-experiments \
        --base-model 8b \
        --quantization w8a8,w4a16 \
        --project-id $PROJECT_8B \
        --rps-root ~/rps-test-set
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx
from sqlalchemy.orm import Session
from tao_validation import (
    DEFAULT_BASE_EXPERIMENT_ID_2B,
    MODEL_NAME_2B,
    MODEL_NAME_8B,
    compute_training_outcome,
    find_reason2_2b_base_experiment,
    log_banner,
    poll_training_suite,
    probe_and_confirm_workspace,
    resolve_workspace_state,
    submit_training_suite,
)

from vlm_feedback_loop.config import load_settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services import (
    example_service,
    guidance_service,
    model_config_service,
    project_service,
)
from vlm_feedback_loop.services.project_service import (
    create_project,
    get_project_engine,
)

logger = logging.getLogger("rps_e2e")

CLASSES: tuple[str, ...] = ("rock", "paper", "scissors")
TRAIN_PER_CLASS = 96  # 96 × 3 = 288 train examples
EVAL_PER_CLASS = 28  # 28 × 3 = 84 eval examples (all that's left after train)

# Public mapping for tests + sanity checks; mirrors the seed catalog.
BASE_MODEL_NAMES: dict[str, str] = {
    "2b": MODEL_NAME_2B,
    "8b": MODEL_NAME_8B,
}


def _load_rps(rps_root: Path) -> dict[str, list[Path]]:
    """Return {class_name: sorted_image_paths}."""
    by_class: dict[str, list[Path]] = {}
    for cls in CLASSES:
        cls_dir = rps_root / cls
        if not cls_dir.is_dir():
            raise SystemExit(f"missing class dir: {cls_dir}")
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() == ".png")
        if len(imgs) < TRAIN_PER_CLASS + EVAL_PER_CLASS:
            raise SystemExit(
                f"class {cls!r} has only {len(imgs)} images; need at least "
                f"{TRAIN_PER_CLASS + EVAL_PER_CLASS}"
            )
        by_class[cls] = imgs
    return by_class


def _build_rps_project(settings, base_experiment_id: str, rps_root: Path) -> dict:
    log_banner("Build RPS quality project + ingest 372 labeled examples")

    by_class = _load_rps(rps_root)
    project = create_project(
        name=f"rps-e2e-{uuid.uuid4().hex[:8]}",
        description="Rock Paper Scissors 3-class classifier",
        settings=settings,
    )
    project_id = project.project_id
    project_dir = Path(project.project_dir)
    print(f"✓ Project created: {project_id} ({project_dir})")

    # Use production Guidance creation so this validation project receives the
    # same SchemaCore derivation and validation as an interactive project.
    # A single Core enum keeps the target distribution intentionally small.
    fields = [
        {
            "field_name": "gesture",
            "type": "enum",
            "role": "core",
            "display_order": 1,
            "allowed_values": list(CLASSES),
        },
    ]
    guidance = guidance_service.create_guidance(
        project_id=project_id,
        description=(
            "Classify the hand gesture shown in the image. "
            "Return JSON with one key, ``gesture``, whose value is "
            "exactly one of ``rock``, ``paper``, or ``scissors``."
        ),
        schema_fields=fields,
        rules=(
            "- ``rock`` = closed fist.\n"
            "- ``paper`` = open hand, all fingers extended.\n"
            "- ``scissors`` = index and middle fingers extended, others curled."
        ),
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if guidance is None:
        raise SystemExit(f"Project disappeared while creating Guidance: {project_id}")
    if isinstance(guidance, str):
        raise SystemExit(f"Guidance validation failed: {guidance}")
    guidance_id = guidance.guidance_id
    updated_project = project_service.update_project(
        project_id,
        {"active_guidance_id": guidance_id},
        settings.WORKSPACE_ROOT,
    )
    if updated_project is None:
        raise SystemExit(f"Project disappeared while activating Guidance: {project_id}")
    print(
        f"✓ Guidance {guidance_id} activated through production services "
        "(1 Core enum: gesture)"
    )

    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    assert engine is not None

    # Look up both Cosmos 2B + 8B seeded ModelConfigs. Patch 2B with the
    # provisioned base-experiment id (build mode defaults to 2B); the 8B
    # ModelConfig is returned unpatched here — _patch_modelconfigs_with_
    # base_experiments() in _amain() applies the 8B patch from the
    # provisioning result when --base-model 8b is used.
    with Session(engine) as session:
        mc_2b = (
            session.query(ModelConfig)
            .filter_by(project_id=project_id, model_name=MODEL_NAME_2B)
            .one()
        )
        mc_8b = (
            session.query(ModelConfig)
            .filter_by(project_id=project_id, model_name=MODEL_NAME_8B)
            .one()
        )
        mc_2b_id = mc_2b.model_config_id
        mc_8b_id = mc_8b.model_config_id
    update_result = model_config_service.update_model_config(
        project_id,
        mc_2b_id,
        {
            "tao_base_experiment_id": base_experiment_id,
            "tao_base_experiment_pull_status": "pull_complete",
        },
        settings.WORKSPACE_ROOT,
    )
    if update_result is None or isinstance(update_result, str):
        raise SystemExit(f"Could not patch the 2B ModelConfig: {update_result}")
    print(
        f"✓ Patched ModelConfig {mc_2b_id} (2B) with tao_base_experiment_id; "
        f"8B ModelConfig {mc_8b_id} discovered (patch applied later if "
        f"--base-model 8b)"
    )

    # Preserve the product's image-by-reference invariant: the validation
    # project points at the operator's dataset instead of copying 372 images
    # into runtime storage.
    examples: list[tuple[str, Path, str, str | None]] = []
    counts: dict[str, dict[str | None, int]] = defaultdict(lambda: defaultdict(int))
    for cls in CLASSES:
        for ix, src in enumerate(by_class[cls]):
            if ix < TRAIN_PER_CLASS:
                pool = None  # → training pool
            elif ix < TRAIN_PER_CLASS + EVAL_PER_CLASS:
                pool = "test_pool"
            else:
                continue  # cap at TRAIN+EVAL per class
            example_key = f"{cls}_{ix:03d}"
            examples.append((example_key, src.resolve(), cls, pool))
            counts[cls][pool] += 1

    print(
        f"✓ Selected {len(examples)} images by reference from {rps_root}: "
        + ", ".join(
            f"{cls}=[train:{counts[cls][None]}, eval:{counts[cls]['test_pool']}]"
            for cls in CLASSES
        )
    )

    ingest_results = example_service.ingest_examples(
        project_id,
        settings.WORKSPACE_ROOT,
        [
            {
                "example_key": key,
                "storage_ref": str(image_path),
                "source_metadata": {
                    "validation_fixture": "rps_e2e",
                    "ground_truth_class": class_name,
                },
                "state": "Verified",
            }
            for key, image_path, class_name, _pool in examples
        ],
    )
    ingest_errors = [
        f"{result['example_key']}: {result['error']}"
        for result in ingest_results
        if result["status"] == "error"
    ]
    if ingest_errors:
        preview = "; ".join(ingest_errors[:5])
        remaining = len(ingest_errors) - 5
        suffix = f"; and {remaining} more" if remaining > 0 else ""
        raise SystemExit(f"RPS ingest failed: {preview}{suffix}")
    print("✓ Ingested 372 image references through the production service")

    # Fixture-only boundary: no product API imports pre-verified labels while
    # also forcing an exact 288/84 train/test split. Seed only the Label and
    # backing OperationRecord rows directly; represent each import as a manual
    # Edit of an empty, schema-invalid proposal so the rows do not claim that a
    # Teacher accepted labels it never generated. Project, Guidance,
    # ModelConfig, and image ingestion above all use production services.
    with Session(engine) as session:
        for key, _img, cls, pool in examples:
            op_id = generate_uuid4()
            session.add(
                OperationRecord(
                    inference_invocation_id=op_id,
                    project_id=project_id,
                    purpose="interactive_proposal",
                    example_key=key,
                    guidance_id=guidance_id,
                    model_config_id=None,
                    invocation_status="schema_invalid",
                    schema_valid_core=False,
                    structured_generation_fallback_used=False,
                    structured_generation_attempted=False,
                    truncation_attributed_schema_invalid=False,
                    ignored_due_to_run_cancellation=False,
                )
            )
            session.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=project_id,
                    example_key=key,
                    label_status="verified",
                    guidance_id=guidance_id,
                    inference_invocation_id=op_id,
                    label_json={"gesture": cls},
                    labeled_at=utc_now(),
                    verified_outcome="Edit",
                    verified_at=utc_now(),
                    edited_core_fields=["gesture"],
                    edited_aux_fields=[],
                    rationale_source=None,
                    pool_assignment=pool,
                )
            )
        session.commit()

    print(
        f"✓ Seeded {len(examples)} fixture-only Verified labels "
        f"(train:{sum(counts[c][None] for c in CLASSES)}, "
        f"eval:{sum(counts[c]['test_pool'] for c in CLASSES)})"
    )

    return {
        "project_id": project_id,
        "project_dir": project_dir,
        "guidance_id": guidance_id,
        "mc_2b_id": mc_2b_id,
        "mc_8b_id": mc_8b_id,
    }


def _lookup_existing_project(settings, project_id: str) -> dict:
    """Lookup mode: reuse an existing labeled project.

    Returns the same shape as ``_build_rps_project`` (project_id,
    project_dir, guidance_id, mc_2b_id, mc_8b_id) so the rest of the
    script doesn't branch on which path produced the assembly.

    Raises if the project is missing, has no active guidance, or is
    missing either seeded Cosmos ModelConfig.
    """
    log_banner(f"Look up existing project {project_id}")
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        raise SystemExit(f"Project {project_id} not found in workspace.")
    with Session(engine) as session:
        proj = session.query(Project).filter_by(project_id=project_id).one_or_none()
        if proj is None:
            raise SystemExit(f"Project {project_id} record missing.")
        if not proj.active_guidance_id:
            raise SystemExit(
                f"Project {project_id} has no active guidance — cannot train."
            )
        mc_2b = (
            session.query(ModelConfig)
            .filter_by(project_id=project_id, model_name=MODEL_NAME_2B)
            .one_or_none()
        )
        mc_8b = (
            session.query(ModelConfig)
            .filter_by(project_id=project_id, model_name=MODEL_NAME_8B)
            .one_or_none()
        )
        if mc_2b is None or mc_8b is None:
            raise SystemExit(
                f"Project {project_id} missing seeded Cosmos ModelConfigs "
                f"(2B={'present' if mc_2b else 'MISSING'}, "
                f"8B={'present' if mc_8b else 'MISSING'})."
            )
        guidance_id = proj.active_guidance_id
        mc_2b_id = mc_2b.model_config_id
        mc_8b_id = mc_8b.model_config_id
        project_dir = (
            Path(proj.project_dir) if getattr(proj, "project_dir", None) else None
        )
    print(
        f"✓ Reusing project {project_id} (guidance={guidance_id}, "
        f"mc_2b={mc_2b_id}, mc_8b={mc_8b_id})"
    )
    return {
        "project_id": project_id,
        "project_dir": project_dir,
        "guidance_id": guidance_id,
        "mc_2b_id": mc_2b_id,
        "mc_8b_id": mc_8b_id,
    }


def _patch_modelconfigs_with_base_experiments(
    settings, project_id: str, *, uuid_by_model_name: dict[str, str]
) -> list[str]:
    """Set ``tao_base_experiment_id`` + pull_status on the 2B and 8B
    ModelConfigs in ``project_id`` from the provisioning result.

    Idempotent: only writes when the current value differs. Skips
    ModelConfigs whose model_name is missing from ``uuid_by_model_name``.
    Returns the list of model_names actually written.
    """
    if not uuid_by_model_name:
        return []
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    assert engine is not None
    written: list[str] = []
    with Session(engine) as session:
        model_configs = {
            model_name: (
                session.query(ModelConfig)
                .filter_by(project_id=project_id, model_name=model_name)
                .one_or_none()
            )
            for model_name in (MODEL_NAME_2B, MODEL_NAME_8B)
        }

    for model_name, model_config in model_configs.items():
        target_uuid = uuid_by_model_name.get(model_name)
        if not target_uuid or model_config is None:
            continue
        if (
            model_config.tao_base_experiment_id == target_uuid
            and model_config.tao_base_experiment_pull_status == "pull_complete"
        ):
            continue
        result = model_config_service.update_model_config(
            project_id,
            model_config.model_config_id,
            {
                "tao_base_experiment_id": target_uuid,
                "tao_base_experiment_pull_status": "pull_complete",
            },
            settings.WORKSPACE_ROOT,
        )
        if result is None:
            raise SystemExit(
                f"ModelConfig disappeared while patching {model_name} in {project_id}"
            )
        if isinstance(result, str):
            raise SystemExit(f"Could not patch ModelConfig for {model_name}: {result}")
        written.append(model_name)
    if written:
        print(f"✓ Patched ModelConfig tao_base_experiment_id for {written}")
    else:
        print("✓ ModelConfig tao_base_experiment_id already current — no patch needed")
    return written


# ── HTTP-mode helpers (lookup mode against a running backend) ──────────────
#
# The single-process-per-project invariant means external scripts MUST
# NOT open a project DB while the backend is running. Lookup mode (where
# the dev backend stays up to serve the frontend + orchestrate
# `tao_polling_service.tick()` in the background) drives the public HTTP
# API only, exactly as another external client would.


async def _lookup_existing_project_http(
    client: httpx.AsyncClient, project_id: str
) -> dict:
    """HTTP-mode lookup. Returns the same assembly shape as
    ``_lookup_existing_project`` but via REST so it can run alongside a
    live backend without contending for the project lock.
    """
    log_banner(f"HTTP lookup: project {project_id}")
    proj_resp = await client.get(f"/v1/projects/{project_id}", timeout=30.0)
    if proj_resp.status_code != 200:
        raise SystemExit(
            f"GET /v1/projects/{project_id} returned "
            f"{proj_resp.status_code}: {proj_resp.text[:200]}"
        )
    proj = proj_resp.json()
    if not proj.get("active_guidance_id"):
        raise SystemExit(f"Project {project_id} has no active guidance — cannot train.")

    mc_resp = await client.get(f"/v1/projects/{project_id}/model_configs", timeout=30.0)
    if mc_resp.status_code != 200:
        raise SystemExit(
            f"GET /v1/projects/{project_id}/model_configs returned "
            f"{mc_resp.status_code}: {mc_resp.text[:200]}"
        )
    items = mc_resp.json().get("items", [])
    by_name = {m["model_name"]: m for m in items}
    mc_2b = by_name.get(MODEL_NAME_2B)
    mc_8b = by_name.get(MODEL_NAME_8B)
    if mc_2b is None or mc_8b is None:
        raise SystemExit(
            f"Project {project_id} missing seeded Cosmos ModelConfigs "
            f"(2B={'present' if mc_2b else 'MISSING'}, "
            f"8B={'present' if mc_8b else 'MISSING'})."
        )
    print(
        f"✓ Reusing project {project_id} (guidance={proj['active_guidance_id']}, "
        f"mc_2b={mc_2b['model_config_id']}, mc_8b={mc_8b['model_config_id']})"
    )
    return {
        "project_id": project_id,
        "project_dir": None,  # not needed in HTTP mode
        "guidance_id": proj["active_guidance_id"],
        "mc_2b_id": mc_2b["model_config_id"],
        "mc_8b_id": mc_8b["model_config_id"],
    }


async def _submit_suite_http(
    client: httpx.AsyncClient,
    *,
    project_id: str,
    base_model: str,
    mc_id: str,
    training_preset: str,
    quantization_schemes: list[str],
    export_field_mode: str,
    idempotency_key: str,
) -> dict:
    """HTTP-mode suite submit. Returns the suite response dict.

    The endpoint is idempotent on ``(project_id, idempotency_key)`` —
    re-POST with the same key returns the existing suite. Callers should
    derive the key deterministically (see ``_derive_idempotency_key``)
    so an interrupted run can be safely retried.
    """
    quants_label = (
        ",".join(quantization_schemes) if quantization_schemes else "[baseline only]"
    )
    log_banner(
        f"HTTP submit training suite (base={base_model}, preset={training_preset}, "
        f"quants={quants_label}, export_field_mode={export_field_mode})"
    )
    body = {
        "student_base_model_config_ids": [mc_id],
        "training_preset": training_preset,
        "include_auto_labeled": False,
        "export_field_mode": export_field_mode,
        "quantization_schemes": quantization_schemes,
        "idempotency_key": idempotency_key,
    }
    resp = await client.post(
        f"/v1/projects/{project_id}/training_suites",
        json=body,
        timeout=60.0,
    )
    if resp.status_code != 201:
        raise SystemExit(
            f"POST /training_suites returned {resp.status_code}: {resp.text[:500]}"
        )
    suite = resp.json()
    print(
        f"✓ TrainingSuite {suite['training_suite_id']} ready (chains={len(suite.get('chains', []))}, "
        f"jobs={sum(len(c.get('jobs', [])) for c in suite.get('chains', []))})"
    )
    return suite


def _derive_idempotency_key(
    *,
    project_id: str,
    base_model: str,
    quantization_schemes: list[str],
    export_field_mode: str,
    training_preset: str,
) -> str:
    """Deterministic idempotency key: a re-POST with identical args returns
    the existing suite, not a duplicate. Distinct training-preset values
    produce distinct keys (the preset materially changes the chain), but
    train_deadline_s and pull_deadline_s are excluded — they only affect
    the script's poll loop, not the suite.

    ``training_preset`` is part of the key payload so a preset change
    always yields a fresh suite — with the preset excluded, canceling a
    ``standard``-preset chain and re-submitting with ``high_quality``
    returns the canceled standard suite, blocking the recovery path.
    """
    payload = (
        f"rps-e2e-http|project={project_id}|base={base_model}|"
        f"export={export_field_mode}|preset={training_preset}|"
        f"quant={','.join(quantization_schemes)}"
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"rps-e2e-{base_model}-{digest}"


async def _get_suite_http(
    client: httpx.AsyncClient, project_id: str, suite_id: str
) -> dict:
    resp = await client.get(
        f"/v1/projects/{project_id}/training_suites/{suite_id}",
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"GET /training_suites/{suite_id} returned "
            f"{resp.status_code}: {resp.text[:200]}"
        )
    return resp.json()


async def _list_tao_jobs_http(client: httpx.AsyncClient, project_id: str) -> list[dict]:
    """List all TAOJobs in the project, ordered by chain_sequence."""
    resp = await client.get(
        f"/v1/projects/{project_id}/tao_jobs?limit=100", timeout=30.0
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"GET /tao_jobs returned {resp.status_code}: {resp.text[:200]}"
        )
    items = resp.json().get("items", [])
    items.sort(key=lambda j: j.get("chain_sequence") or 0)
    return items


async def _poll_to_completion_http(
    client: httpx.AsyncClient,
    *,
    project_id: str,
    suite_id: str,
    deadline_s: float,
    accept_eval_failure: bool = False,
) -> str:
    """HTTP-mode poll. Watches the suite + per-job statuses until the
    success/failure predicate (``compute_training_outcome``) returns terminal,
    or the deadline expires. Returns one of
    ``"succeeded" | "failed" | "deadline_exceeded"``.
    """
    banner_target = (
        "all jobs terminal under chain-isolation"
        if accept_eval_failure
        else "last evaluate=succeeded"
    )
    log_banner(
        f"HTTP poll until {banner_target} (suite={suite_id}, deadline={int(deadline_s)}s)"
    )
    start = time.monotonic()
    last_summary = ""
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= deadline_s:
            print(f"⏱ deadline exceeded after {int(elapsed)}s — stopping poll.")
            return "deadline_exceeded"

        try:
            suite = await _get_suite_http(client, project_id, suite_id)
            jobs = await _list_tao_jobs_http(client, project_id)
        except SystemExit:
            raise
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("poll fetch failed: %s — retrying", exc)
            await asyncio.sleep(15)
            continue

        suite_jobs = [
            j
            for j in jobs
            if j.get("training_suite_id") == suite_id
            or (suite.get("chain_ids_ordered") or [])
            and j.get("chain_id") in (suite.get("chain_ids_ordered") or [])
        ]
        # Fall back to all jobs if the chain_id linkage isn't surfaced
        # by the API (chains created before the linkage field lack it).
        if not suite_jobs:
            suite_jobs = jobs

        summary = f"suite={suite.get('status', '?')} " + " ".join(
            f"{j['action']}:{j['status']}" for j in suite_jobs
        )
        if summary != last_summary:
            print(f"  [{int(elapsed):4d}s] {summary}")
            last_summary = summary

        result = compute_training_outcome(
            suite_jobs, accept_eval_failure=accept_eval_failure
        )
        if result == "succeeded":
            if accept_eval_failure:
                print(
                    "✓ chain terminal under chain-isolation; train+quantize succeeded"
                )
            else:
                print("✓ last evaluate = succeeded")
            return "succeeded"
        if result == "failed":
            failed = [j for j in suite_jobs if j["status"] == "failed"]
            print(
                "✗ One or more jobs failed: "
                + ", ".join(f"{j['action']}/{j.get('tao_job_id', '?')}" for j in failed)
            )
            return "failed"
        await asyncio.sleep(30)


async def _summarize_outcome_http(
    client: httpx.AsyncClient, *, project_id: str, suite_id: str, outcome: str
) -> dict:
    """HTTP-mode summary. Returns a dict comparable in shape to the
    service-mode ``_summarize_outcome`` so log readers don't have to
    branch on which path produced the JSON.
    """
    suite = await _get_suite_http(client, project_id, suite_id)
    jobs = await _list_tao_jobs_http(client, project_id)
    students_resp = await client.get(
        f"/v1/projects/{project_id}/student_models?limit=100", timeout=30.0
    )
    students_resp.raise_for_status()
    students = students_resp.json().get("items", [])
    return {
        "project_id": project_id,
        "training_suite_id": suite_id,
        "suite_status": suite.get("status"),
        "students": [
            {
                "student_model_id": s["student_model_id"],
                "base_model_name": s.get("base_model_name"),
                "quantization_method": s.get("quantization_method"),
                "quality_status": s.get("quality_status"),
                "checkpoint_packaging_status": s.get("checkpoint_packaging_status"),
                "nim_checkpoint_ref": s.get("nim_checkpoint_ref"),
                "training_inference_contract": s.get("training_inference_contract"),
            }
            for s in students
        ],
        "tao_jobs": [
            {
                "chain_sequence": j.get("chain_sequence"),
                "action": j.get("action"),
                "status": j.get("status"),
                "tao_external_job_id": j.get("tao_external_job_id"),
                "parent_tao_job_id": j.get("parent_tao_job_id"),
            }
            for j in jobs
        ],
        "outcome": outcome,
    }


def _summarize_outcome(settings, suite: dict) -> dict:
    """Read the StudentModel + RunRecord state and return a JSON summary.

    ``students`` contains every StudentModel in the project. Chain isolation
    produces N Students per chain (one baseline plus N quantized variants).
    """
    engine = get_project_engine(suite["project_id"], settings.WORKSPACE_ROOT)
    assert engine is not None
    with Session(engine) as session:
        all_students = (
            session.query(StudentModel)
            .filter_by(project_id=suite["project_id"])
            .order_by(StudentModel.created_at.asc())
            .all()
        )
        run = None
        first_sm = all_students[0] if all_students else None
        if first_sm and first_sm.quality_evaluation_run_id:
            run = (
                session.query(RunRecord)
                .filter_by(run_id=first_sm.quality_evaluation_run_id)
                .one_or_none()
            )
        jobs = (
            session.query(TAOJob)
            .filter_by(project_id=suite["project_id"])
            .order_by(TAOJob.chain_sequence.asc())
            .all()
        )

    def _student_dict(sm: StudentModel) -> dict:
        return {
            "student_model_id": sm.student_model_id,
            "quantization_method": sm.quantization_method,
            "quality_status": sm.quality_status,
            "checkpoint_packaging_status": sm.checkpoint_packaging_status,
            "nim_checkpoint_ref": sm.nim_checkpoint_ref,
            "quality_evaluation_run_id": sm.quality_evaluation_run_id,
        }

    return {
        "project_id": suite["project_id"],
        "training_suite_id": suite["training_suite_id"],
        "students": [_student_dict(sm) for sm in all_students],
        "run_record": None
        if run is None
        else {
            "run_id": run.run_id,
            "evaluation_source": run.evaluation_source,
            "examples_total": run.examples_total,
            "examples_succeeded": run.examples_succeeded,
            "examples_schema_invalid": run.examples_schema_invalid,
            "rescored_metrics_keys": (sorted((run.rescored_metrics or {}).keys())),
        },
        "tao_jobs": [
            {
                "chain_sequence": j.chain_sequence,
                "action": j.action,
                "status": j.status,
                "tao_external_job_id": j.tao_external_job_id,
                "parent_tao_job_id": j.parent_tao_job_id,
            }
            for j in jobs
        ],
    }


def _parse_quantization_arg(raw: str) -> list[str]:
    """Validate + normalize the --quantization comma-separated arg.

    Lazy import of ``VALID_QUANTIZATION_SCHEMES`` so test fixtures that
    monkeypatch the service layer still see the canonical set.

    Returns the schemes as upper-case strings in input order. Empty
    string / no items → ``[]`` (baseline only).
    """
    from vlm_feedback_loop.services.training_suite_service import (
        VALID_QUANTIZATION_SCHEMES,
    )

    if not raw or not raw.strip():
        return []
    schemes = [s.strip().upper() for s in raw.split(",") if s.strip()]
    bad = [s for s in schemes if s not in VALID_QUANTIZATION_SCHEMES]
    if bad:
        valid = ", ".join(sorted(VALID_QUANTIZATION_SCHEMES))
        raise SystemExit(
            f"Invalid --quantization scheme(s): {bad!r}. Valid schemes: {valid}"
        )
    # Dedupe while preserving order so e.g. ``fp8_dynamic,FP8_DYNAMIC`` collapses.
    seen: set[str] = set()
    out: list[str] = []
    for s in schemes:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_argv = list(sys.argv[1:] if argv is None else argv)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rps-root",
        default=os.environ.get("RPS_TEST_SET_ROOT", "~/rps-test-set"),
        help=(
            "Path to the rock-paper-scissors test set root "
            "(env override: RPS_TEST_SET_ROOT)."
        ),
    )
    p.add_argument(
        "--training-preset",
        default="high_quality",
        choices=["quick", "standard", "high_quality", "max_quality"],
        help=(
            "Cosmos-RL SFT training preset. ``high_quality`` = 12 epochs "
            "on 2B; with 288 train examples + 8×A100 + global batch 64 "
            "that's ~60 SGD steps (40 post-warmup), enough to bend the "
            "model toward the {rock,paper,scissors} JSON output."
        ),
    )
    p.add_argument(
        "--base-model",
        default="2b",
        choices=["2b", "8b"],
        help=(
            "Cosmos Reason 2 base model variant. Determines which seeded "
            "ModelConfig is used as ``student_base_model_config_ids[0]`` "
            "in the TrainingSuite payload. The closing smoke uses "
            "both bases (separate invocations)."
        ),
    )
    p.add_argument(
        "--quantization",
        default="",
        help=(
            "Comma-separated quantization schemes to add to the chain "
            "(case-insensitive). Valid: FP8_DYNAMIC, W8A8, W8A16, W4A16. "
            "Empty (default) = baseline only. Each scheme adds a "
            "quantize → evaluate job pair."
        ),
    )
    p.add_argument(
        "--export-field-mode",
        default="all",
        choices=["all", "aux_and_core", "core_only"],
        help=(
            "Export field mode for the training dataset. The Student's "
            "training-time Inference Contract is derived from this, so a "
            "``core_only`` run produces a Student whose deployment-handoff "
            "mismatches any ``all``-mode evaluation contract — useful for "
            "contract-mismatch sibling exercises."
        ),
    )
    p.add_argument(
        "--project-id",
        default=None,
        help=(
            "If set, skip project bootstrap and reuse the named project "
            "that already contains Verified labels. Project must have an "
            "active Guidance and the seeded Cosmos 2B + 8B ModelConfigs. "
            "Lookup mode drives the public HTTP API only (one backend "
            "process per project database), so the dev backend MUST be running "
            "and reachable at --backend-url."
        ),
    )
    p.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8000",
        help=(
            "Backend HTTP base URL for lookup mode. Ignored in build mode. "
            "Default matches scripts/dev.sh."
        ),
    )
    p.add_argument(
        "--training-suite-id",
        default=None,
        help=(
            "Lookup mode only: resume polling on an existing TrainingSuite "
            "instead of submitting a new one. Useful when a prior invocation "
            "submitted the suite but was interrupted before polling completed."
        ),
    )
    p.add_argument(
        "--base-experiment-id-2b",
        default=DEFAULT_BASE_EXPERIMENT_ID_2B,
        help="Override the Cosmos Reason2 2B base-experiment id.",
    )
    p.add_argument(
        "--pull-deadline-s",
        type=float,
        default=1800.0,
        help="Base-experiment pull deadline (default: 30 min).",
    )
    p.add_argument(
        "--train-deadline-s",
        type=float,
        default=5400.0,
        help="Training + eval deadline (default: 90 min).",
    )
    p.add_argument(
        "--auto-provision-base-experiments",
        action="store_true",
        help=(
            "If set, run self-service base-experiment provisioning "
            "before training. Idempotent: re-discovers existing base "
            "experiments without re-uploading. Required for "
            "--base-model 8b."
        ),
    )
    p.add_argument(
        "--accept-eval-failure",
        action="store_true",
        help=(
            "Required for the closing smoke under cosmos-rl 6.26.3 + "
            "Cosmos-Reason2 (Qwen3-VL-dense gap). When "
            "set: poll keeps running past evaluate-job failures and "
            "succeeds when all chain jobs reach terminal status AND train "
            "succeeded AND (no quantize jobs OR ≥1 quantize succeeded). "
            "Exit code drops the quality_status=validated requirement; "
            "success means ≥1 Student has checkpoint_packaging_status="
            "validated. NIM-eval-as-quality-fallback flips "
            "quality_status downstream in scripts/full_stack_validation.py. "
            "Without this flag strict quality semantics (last evaluate "
            "must succeed; any failure fatal) are preserved."
        ),
    )
    return p.parse_args(cli_argv)


async def _amain_lookup_http(
    args: argparse.Namespace, quantization_schemes: list[str]
) -> int:
    """Lookup-mode entrypoint: drive the public HTTP API only.

    The dev backend MUST be running and reachable at ``--backend-url``.
    No project lock is acquired by this script — the backend holds the
    lock and serves the endpoints we call.
    """
    project_id = args.project_id
    backend_url = args.backend_url
    async with httpx.AsyncClient(base_url=backend_url, timeout=30.0) as client:
        # Quick health probe so a typo in --backend-url surfaces immediately.
        try:
            health = await client.get("/health", timeout=5.0)
            if health.status_code != 200:
                raise SystemExit(
                    f"Backend at {backend_url}/health returned "
                    f"{health.status_code}. Is the dev stack running?"
                )
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"Backend at {backend_url} is unreachable: {exc}. "
                f"Start it with ./scripts/dev.sh first."
            ) from exc
        print(f"✓ Backend reachable at {backend_url}")

        assembly = await _lookup_existing_project_http(client, project_id)
        mc_id = assembly[f"mc_{args.base_model}_id"]

        if args.training_suite_id:
            print(
                f"→ Resuming poll on existing TrainingSuite "
                f"{args.training_suite_id} (no resubmit)"
            )
            suite_id = args.training_suite_id
        else:
            idempotency_key = _derive_idempotency_key(
                project_id=project_id,
                base_model=args.base_model,
                quantization_schemes=quantization_schemes,
                export_field_mode=args.export_field_mode,
                training_preset=args.training_preset,
            )
            suite = await _submit_suite_http(
                client,
                project_id=project_id,
                base_model=args.base_model,
                mc_id=mc_id,
                training_preset=args.training_preset,
                quantization_schemes=quantization_schemes,
                export_field_mode=args.export_field_mode,
                idempotency_key=idempotency_key,
            )
            suite_id = suite["training_suite_id"]

        outcome = await _poll_to_completion_http(
            client,
            project_id=project_id,
            suite_id=suite_id,
            deadline_s=args.train_deadline_s,
            accept_eval_failure=args.accept_eval_failure,
        )

        summary = await _summarize_outcome_http(
            client,
            project_id=project_id,
            suite_id=suite_id,
            outcome=outcome,
        )
        summary["base_model"] = args.base_model
        summary["quantization_schemes"] = quantization_schemes
        summary["export_field_mode"] = args.export_field_mode
        summary["accept_eval_failure"] = args.accept_eval_failure
        print(json.dumps(summary, indent=2))

    if outcome == "succeeded":
        students = summary.get("students", [])
        if not students:
            return 1
        if args.accept_eval_failure:
            # Chain-isolation: NIM-eval-as-quality-fallback runs downstream
            # in full_stack_validation.py and flips quality_status. Here we
            # only require that the chain produced loadable checkpoints.
            if any(
                s.get("checkpoint_packaging_status") == "validated" for s in students
            ):
                return 0
            return 1
        # Strict quality semantics — every Student must already be validated.
        if all(s.get("quality_status") == "validated" for s in students):
            return 0
        return 1
    if outcome == "deadline_exceeded":
        return 2
    return 1


async def _amain() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    args = _parse_args()

    rps_root = Path(args.rps_root).expanduser()
    # rps_root is only consumed by the build path; in lookup mode the
    # project's labels are already in place.
    if not args.project_id and not rps_root.is_dir():
        raise SystemExit(f"--rps-root not a directory: {rps_root}")

    quantization_schemes = _parse_quantization_arg(args.quantization)

    # Lookup mode: HTTP-only path against the running backend
    # (single-process-per-project invariant). Provisioning is a deployment-level
    # operation that conflicts with the backend's project locks; route
    # operators to the dedicated CLI.
    if args.project_id:
        if args.auto_provision_base_experiments:
            raise SystemExit(
                "--project-id (lookup mode) is incompatible with "
                "--auto-provision-base-experiments. Provisioning is a "
                "deployment-level operation that conflicts with the "
                "running backend's project locks (one backend process per "
                "project database). Run "
                "`vlm-feedback-loop tao-pull-base-experiments` once before "
                "driving training via this script in lookup mode."
            )
        return await _amain_lookup_http(args, quantization_schemes)

    if args.training_suite_id:
        raise SystemExit(
            "--training-suite-id requires --project-id (lookup mode only)."
        )
    if args.base_model == "8b" and not args.auto_provision_base_experiments:
        raise SystemExit(
            "--base-model 8b currently requires --auto-provision-base-experiments "
            "(no explicit --base-experiment-id-8b flag yet). Run provisioning "
            "once per box; subsequent invocations are idempotent."
        )

    # Loaded only after the cheap argv validations so a bad flag combination
    # is reported even on a box with no ~/.vlm_feedback_loop config.
    settings = load_settings()

    # TAO API endpoint + auth + org live in .env. Workspace identity lives
    # exclusively in deployment.db.
    for required in ("TAO_API_BASE_URL", "TAO_API_KEY", "TAO_ORG_NAME"):
        if not getattr(settings, required):
            print(f"✗ {required} is not configured in .env", file=sys.stderr)
            return 1

    # Resolve workspace identity. A fresh deployment gets clear bootstrap
    # remediation rather than an implicit second configuration path.
    workspace_state = resolve_workspace_state(settings)
    print(
        f"→ Workspace identity resolved from deployment.db "
        f"(id={workspace_state.tao_workspace_id})"
    )

    await probe_and_confirm_workspace(settings)

    # Provisioning gives us {model_name: base_experiment_uuid} for both
    # 2B and 8B. Build path uses the 2B UUID for its single in-line patch;
    # _patch_modelconfigs_with_base_experiments() re-applies the dict to
    # patch both ModelConfigs (idempotent).
    uuid_by_model_name: dict[str, str] = {}
    base_experiment_id_2b: str | None = None

    if args.auto_provision_base_experiments:
        from vlm_feedback_loop.services.tao_base_experiment_provisioning_service import (
            provision_base_experiments,
        )

        log_banner("Self-service base-experiment provisioning")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        prov = await provision_base_experiments(settings, hf_token=hf_token)
        print(
            f"  registered={prov.registered}  "
            f"already_registered={prov.already_registered}  "
            f"failed={prov.failed}"
        )
        if prov.failed:
            print(
                "✗ Self-service provisioning failed; cannot proceed.",
                file=sys.stderr,
            )
            return 1
        # uuid_by_model_name uses the airgapped-load slug as model_name;
        # the build path needs the 2B UUID specifically.
        # ``find_base_experiment_by_arch``'s name-substring matcher uses
        # the airgapped-load slug ``cosmos-reason-2-2b`` while FTMS lists
        # experiments under their human display name ``Cosmos Reason2 2B``
        # — the bypass works on cold starts.
        uuid_by_model_name = dict(prov.uuid_by_model_name)
        base_experiment_id_2b = uuid_by_model_name.get(MODEL_NAME_2B)
        if not base_experiment_id_2b and not args.project_id:
            raise SystemExit(
                f"Provisioning result has no UUID for {MODEL_NAME_2B}: "
                f"{uuid_by_model_name}"
            )
        # 8B path also needs its UUID present in the provisioning result.
        if args.base_model == "8b" and not uuid_by_model_name.get(MODEL_NAME_8B):
            raise SystemExit(
                f"--base-model 8b: provisioning result has no UUID for "
                f"{MODEL_NAME_8B}: {uuid_by_model_name}"
            )
        if base_experiment_id_2b:
            print(f"→ Using 2B base experiment UUID: {base_experiment_id_2b}")
    else:
        # Build path without provisioning: only valid for 2B (8B path was
        # rejected above). Resolve the already-indexed base experiment.
        base_experiment_id_2b = await find_reason2_2b_base_experiment(
            settings,
            args.base_experiment_id_2b,
            deadline_s=args.pull_deadline_s,
        )
        uuid_by_model_name = {MODEL_NAME_2B: base_experiment_id_2b}

    # Project assembly: lookup mode or build mode.
    if args.project_id:
        assembly = _lookup_existing_project(settings, args.project_id)
    else:
        assert base_experiment_id_2b is not None
        assembly = _build_rps_project(settings, base_experiment_id_2b, rps_root)

    # Idempotent patching: covers both lookup mode (project's ModelConfigs
    # may not yet be patched) and build mode (8B ModelConfig was discovered
    # but not patched in-line). No-op when values are already current.
    _patch_modelconfigs_with_base_experiments(
        settings, assembly["project_id"], uuid_by_model_name=uuid_by_model_name
    )

    suite = await submit_training_suite(
        settings,
        assembly,
        base_model=args.base_model,
        training_preset=args.training_preset,
        quantization_schemes=quantization_schemes,
        export_field_mode=args.export_field_mode,
        idempotency_prefix="rps-e2e",
    )

    outcome = await poll_training_suite(
        settings,
        suite,
        deadline_s=args.train_deadline_s,
        accept_eval_failure=args.accept_eval_failure,
    )

    summary = _summarize_outcome(settings, suite)
    summary["outcome"] = outcome
    summary["base_model"] = args.base_model
    summary["quantization_schemes"] = quantization_schemes
    summary["export_field_mode"] = args.export_field_mode
    summary["accept_eval_failure"] = args.accept_eval_failure
    print(json.dumps(summary, indent=2))

    if outcome == "succeeded":
        students = summary.get("students") or []
        if not students:
            return 1
        if args.accept_eval_failure:
            # Chain-isolation: quality_status flips downstream via
            # NIM-eval-as-quality-fallback. Only require that
            # the chain produced loadable checkpoints here.
            if any(
                s.get("checkpoint_packaging_status") == "validated" for s in students
            ):
                return 0
            return 1
        # Strict quality semantics — the first Student must be validated.
        sm = students[0]
        return 0 if sm.get("quality_status") == "validated" else 1
    if outcome == "deadline_exceeded":
        return 2
    return 1


def main() -> None:
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()

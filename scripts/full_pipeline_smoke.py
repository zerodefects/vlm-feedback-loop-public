#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final integration checkpoint — full-pipeline smoke against a live backend.

A single scripted flow covering project → configure NIM → ingest →
Guidance → label 10 →
evaluate → gate → batch-label → export → verify Cosmos-RL format. Hosted
NIM only — no Docker, no GPU, no TAO. Independent of the hardware-heavy
`full_stack_validation.py` closing smoke; an operator-run release-readiness
gate (needs a live backend and a hosted NIM key, so it is not wired into CI).

The script drives the public REST API only — no direct DB writes — so every
code path exercised here is the same one a Blueprint user would hit.

Usage::

    # Direct CLI (requires a backend already running and an NVIDIA_API_KEY).
    uv run python scripts/full_pipeline_smoke.py \\
        --backend-url http://127.0.0.1:8000 \\
        --image-source-dir /tmp/my-images \\
        --label-count 10

    # Library use (closing-smoke calls into this for Phase E):
    from scripts.full_pipeline_smoke import run_full_pipeline_smoke
    result = await run_full_pipeline_smoke(...)

The script emits ``{evidence_dir}/full_pipeline_smoke_acceptance.json`` with
per-step status + counts + checksums for inclusion in
``closing_acceptance.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tarfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))

logger = logging.getLogger("full_pipeline_smoke")

# Single-core-enum schema with optional rationale_note Aux. Matches the v1
# minimal-Guidance shape. The label_count default (10) is chosen so the
# smoke completes in ~30 sec on hosted NIM and the test pool reaches the
# 5-image first-evaluation recommendation threshold before saving the
# 10th label.
_GESTURE_VALUES = ("rock", "paper", "scissors")
_DEFAULT_GUIDANCE_FIELDS = [
    {
        "field_name": "rationale_note",
        "type": "string",
        "role": "aux",
        "display_order": 0,
    },
    {
        "field_name": "gesture",
        "type": "enum",
        "role": "core",
        "allowed_values": list(_GESTURE_VALUES),
        "display_order": 1,
    },
]
# The guidance-create endpoint expects ``schema`` as a direct list of
# field dicts (the alias for ``schema_fields`` on ``GuidanceCreate``).
_DEFAULT_GUIDANCE = {
    "description": (
        "Label the visible hand gesture in the image. Choose exactly one "
        "of: rock (closed fist), paper (open palm), scissors (two extended "
        "fingers in a V). Output JSON only."
    ),
    "rules": "If the gesture is ambiguous or partially occluded, choose the closest match.",
    "schema": _DEFAULT_GUIDANCE_FIELDS,
}


RationaleSource = Literal["teacher_proposal", "sme_edited"]
ExpectedOutcome = Literal["Accept", "Edit"]


@dataclass(frozen=True)
class SMEDecision:
    """One deterministic Accept/Edit decision for the live smoke."""

    label_json: dict[str, Any] | None
    rationale_source: RationaleSource | None
    expected_outcome: ExpectedOutcome


def _core_fields_match(
    proposal_json: dict[str, Any],
    ground_truth: dict[str, Any],
    schema_fields: list[dict[str, Any]],
) -> bool:
    for schema_field in schema_fields:
        if schema_field.get("role") != "core":
            continue
        name = schema_field["field_name"]
        if name in ground_truth and proposal_json.get(name) != ground_truth[name]:
            return False
    return True


def _build_edit_decision(
    ground_truth: dict[str, Any],
    template_rationale_fn: Callable[[dict[str, Any]], str],
    *,
    rationale_enabled: bool,
) -> SMEDecision:
    label_json = dict(ground_truth)
    if rationale_enabled:
        label_json["rationale_note"] = template_rationale_fn(ground_truth)
    return SMEDecision(
        label_json=label_json,
        rationale_source="sme_edited" if rationale_enabled else None,
        expected_outcome="Edit",
    )


def _decide(
    proposal_response: dict[str, Any],
    ground_truth: dict[str, Any],
    schema_fields: list[dict[str, Any]],
    template_rationale_fn: Callable[[dict[str, Any]], str],
) -> SMEDecision:
    """Apply the smoke's deterministic synthetic-review decision."""

    status = proposal_response.get("invocation_status")
    rationale_enabled = any(
        field.get("field_name") == "rationale_note" for field in schema_fields
    )
    if status in ("timeout", "endpoint_error", "rate_limited"):
        return _build_edit_decision(
            ground_truth,
            template_rationale_fn,
            rationale_enabled=rationale_enabled,
        )
    if status == "schema_invalid" or not proposal_response.get(
        "schema_valid_core", False
    ):
        return _build_edit_decision(
            ground_truth,
            template_rationale_fn,
            rationale_enabled=rationale_enabled,
        )

    proposal_json_raw: Any = proposal_response.get("proposal_json") or {}
    proposal_json = (
        cast("dict[str, Any]", proposal_json_raw)
        if isinstance(proposal_json_raw, dict)
        else {}
    )
    if not rationale_enabled:
        proposal_json = {
            key: value
            for key, value in proposal_json.items()
            if key != "rationale_note"
        }
    if _core_fields_match(proposal_json, ground_truth, schema_fields):
        return SMEDecision(
            label_json=proposal_json,
            rationale_source="teacher_proposal" if rationale_enabled else None,
            expected_outcome="Accept",
        )

    edited = dict(proposal_json)
    for schema_field in schema_fields:
        name = schema_field["field_name"]
        if name in ground_truth:
            edited[name] = ground_truth[name]
    if rationale_enabled:
        edited["rationale_note"] = template_rationale_fn(ground_truth)
    return SMEDecision(
        label_json=edited,
        rationale_source="sme_edited" if rationale_enabled else None,
        expected_outcome="Edit",
    )


def _make_template_rationale(
    schema_fields: list[dict[str, Any]],
) -> Callable[[dict[str, Any]], str]:
    def render(ground_truth: dict[str, Any]) -> str:
        parts = [
            f"{schema_field['field_name']}={ground_truth[schema_field['field_name']]}"
            for schema_field in schema_fields
            if schema_field.get("role") == "core"
            and schema_field["field_name"] in ground_truth
        ]
        summary = ", ".join(parts) if parts else "as specified"
        return f"Visible evidence supports the corrected label ({summary})."

    return render


# ── Dataclasses (acceptance evidence shape) ─────────────────────────────────


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullPipelineSmokeResult:
    project_id: str | None
    started_at: str
    finished_at: str
    overall_ok: bool
    steps: list[StepResult] = field(default_factory=list)
    cosmos_rl_format_validated: bool = False
    dataset_export_sha: str | None = None
    dataset_example_count: int = 0
    label_count: int = 0
    gate_status: str | None = None

    def as_acceptance_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "overall_ok": self.overall_ok,
            "label_count": self.label_count,
            "gate_status": self.gate_status,
            "cosmos_rl_format_validated": self.cosmos_rl_format_validated,
            "dataset_export_sha": self.dataset_export_sha,
            "dataset_example_count": self.dataset_example_count,
            "steps": [asdict(s) for s in self.steps],
        }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ground_truth_for_image(image_path: Path) -> dict[str, Any]:
    """Synthesize a ground-truth label from the image filename.

    The smoke is hardware-independent — we don't need real images. The
    convention is: filename starts with one of "rock_", "paper_",
    "scissors_". For arbitrary images, hash-bucket the filename so the
    smoke is deterministic.
    """
    name = image_path.stem.lower()
    for value in _GESTURE_VALUES:
        if name.startswith(value):
            return {"gesture": value}
    # Deterministic fallback bucketing.
    bucket = sum(ord(c) for c in name) % len(_GESTURE_VALUES)
    return {"gesture": _GESTURE_VALUES[bucket]}


async def _wait_for_run_terminal(
    client: httpx.AsyncClient,
    *,
    project_id: str,
    run_kind: str,  # "evaluation_runs" | "batch_label_runs"
    run_id: str,
    # Hosted evaluation against a small Test Pool can run well past
    # 120s under rate-limit backoff. The smoke is hardware-independent
    # but I/O-bound on hosted-NIM latency; a 600s deadline accommodates
    # backoff while still failing fast on actual hangs (hosted
    # ``HTTP_DEADLINE_BACKGROUND_S = 300`` × 2x is the right ceiling).
    timeout_s: float = 600.0,
    poll_interval_s: float = 1.0,
    terminal_states: tuple[str, ...] = (
        "completed",
        "incomplete",
        "canceled",
        "failed",
    ),
) -> dict[str, Any]:
    """Poll a background run until it reaches a terminal status."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    last: dict[str, Any] | None = None
    while asyncio.get_event_loop().time() < deadline:
        resp = await client.get(
            f"/v1/projects/{project_id}/{run_kind}/{run_id}", timeout=10.0
        )
        resp.raise_for_status()
        last = resp.json()
        if last["status"] in terminal_states:
            return last
        await asyncio.sleep(poll_interval_s)
    raise TimeoutError(
        f"{run_kind}/{run_id} did not reach terminal in {timeout_s}s "
        f"(last status: {last['status'] if last else '<unfetched>'})"
    )


# ── Pipeline steps ──────────────────────────────────────────────────────────


async def _step_create_project(client: httpx.AsyncClient, *, name: str) -> StepResult:
    resp = await client.post("/v1/projects", json={"name": name}, timeout=30.0)
    if resp.status_code != 201:
        return StepResult(
            "create_project",
            False,
            detail=f"HTTP {resp.status_code}: {resp.text[:200]}",
        )
    project = resp.json()
    return StepResult(
        "create_project",
        True,
        metrics={"project_id": project["project_id"]},
    )


async def _step_verify_teacher_endpoint(
    client: httpx.AsyncClient, project_id: str
) -> StepResult:
    """Confirm a Teacher-eligible model_config is selected. Hosted NIM seeded."""
    resp = await client.get(f"/v1/projects/{project_id}", timeout=10.0)
    resp.raise_for_status()
    project = resp.json()
    teacher_id = project.get("teacher_model_config_id")
    if not teacher_id:
        return StepResult(
            "verify_teacher_endpoint",
            False,
            detail="project.teacher_model_config_id is empty after create",
        )
    return StepResult(
        "verify_teacher_endpoint",
        True,
        metrics={"teacher_model_config_id": teacher_id},
    )


async def _step_ingest(
    client: httpx.AsyncClient,
    *,
    project_id: str,
    image_paths: list[Path],
) -> StepResult:
    """Ingest images via the per-example payload shape used by AutoRun.

    The :ingest endpoint expects ``{"examples": [{example_key, storage_ref,
    source_metadata}, ...]}``. Stashing ground truth in source_metadata
    mirrors the AutoRun convention so the labeling step can recover it
    without re-reading filenames.
    """
    items = []
    for p in image_paths:
        resolved = p.resolve()
        gt = _ground_truth_for_image(p)
        items.append(
            {
                "example_key": resolved.stem,
                "storage_ref": str(resolved),
                "source_metadata": {
                    "full_pipeline_smoke_ground_truth": gt,
                    "full_pipeline_smoke_image": resolved.name,
                },
            }
        )
    resp = await client.post(
        f"/v1/projects/{project_id}/examples:ingest",
        json={"examples": items},
        timeout=120.0,
    )
    # 202 Accepted is the endpoint's SUCCESS code (routers/examples.py):
    # rows persist synchronously, pHash/embeddings compute in background.
    if resp.status_code not in (200, 201, 202):
        return StepResult(
            "ingest", False, detail=f"HTTP {resp.status_code}: {resp.text[:300]}"
        )
    body = resp.json()
    accepted = int(
        body.get("accepted_count")
        or len(body.get("results", []))
        or len([r for r in body.get("results", []) if r.get("ok")])
    )
    if accepted == 0:
        return StepResult(
            "ingest",
            False,
            detail=f"no examples accepted from {len(image_paths)} paths",
            metrics=body,
        )
    return StepResult("ingest", True, metrics={"accepted": accepted})


async def _step_save_guidance(client: httpx.AsyncClient, project_id: str) -> StepResult:
    """Save Guidance + PATCH ``project.active_guidance_id`` to activate it.

    Saving alone is not enough: without the PATCH the downstream
    ``label_loop`` step fails with ``HTTP 422: "No active guidance
    configured for this project"``. Mirrors what the Create Guidance UI
    flow does (save + activate in the same user gesture).
    """
    resp = await client.post(
        f"/v1/projects/{project_id}/guidance",
        json=_DEFAULT_GUIDANCE,
        timeout=30.0,
    )
    if resp.status_code not in (200, 201):
        return StepResult(
            "save_guidance",
            False,
            detail=f"HTTP {resp.status_code}: {resp.text[:300]}",
        )
    guidance_id = resp.json()["guidance_id"]

    # POST /guidance creates an immutable Guidance version but does not
    # bind it as the project's active version. The canonical UI flow (the
    # Create Guidance screen) explicitly PATCHes the project after save. Mirror that here so the proposal endpoint
    # below (which reads project.active_guidance_id) finds it set.
    patch_resp = await client.patch(
        f"/v1/projects/{project_id}",
        json={"active_guidance_id": guidance_id},
        timeout=10.0,
    )
    if patch_resp.status_code not in (200, 204):
        return StepResult(
            "save_guidance",
            False,
            detail=(
                f"Guidance saved but project PATCH failed: HTTP "
                f"{patch_resp.status_code}: {patch_resp.text[:300]}"
            ),
        )
    return StepResult(
        "save_guidance",
        True,
        metrics={
            "guidance_id": guidance_id,
            "active_guidance_id_set": True,
        },
    )


async def _step_label_loop(
    client: httpx.AsyncClient,
    *,
    project_id: str,
    examples: list[dict[str, Any]],
    image_paths: list[Path],
) -> StepResult:
    """Drive `/proposals` + `/labels` per example via the synthetic SME."""
    schema_fields = _DEFAULT_GUIDANCE_FIELDS
    template_fn = _make_template_rationale(schema_fields)

    # Map example_key → ground_truth by aligning ingestion order to image
    # filenames. The ingest endpoint returns examples in path-input order.
    by_path = {str(p.resolve()): _ground_truth_for_image(p) for p in image_paths}

    saved = 0
    accept = 0
    edit = 0
    for ex in examples:
        ex_key = ex["example_key"]
        storage_ref = ex.get("storage_ref")
        ground_truth = by_path.get(storage_ref) or _ground_truth_for_image(
            Path(storage_ref or ex_key)
        )
        proposal_resp = await client.post(
            f"/v1/projects/{project_id}/proposals",
            json={"example_key": ex_key},
            timeout=180.0,
        )
        if proposal_resp.status_code not in (200, 201):
            return StepResult(
                "label_loop",
                False,
                detail=(
                    f"proposal failed for {ex_key}: HTTP "
                    f"{proposal_resp.status_code}: {proposal_resp.text[:200]}"
                ),
                metrics={"saved": saved},
            )
        proposal = proposal_resp.json()
        decision = _decide(proposal, ground_truth, schema_fields, template_fn)
        save_resp = await client.post(
            f"/v1/projects/{project_id}/labels",
            json={
                "example_key": ex_key,
                "inference_invocation_id": proposal["inference_invocation_id"],
                "label_json": decision.label_json,
                "rationale_source": decision.rationale_source,
            },
            timeout=30.0,
        )
        if save_resp.status_code not in (200, 201):
            return StepResult(
                "label_loop",
                False,
                detail=(
                    f"save failed for {ex_key}: HTTP "
                    f"{save_resp.status_code}: {save_resp.text[:200]}"
                ),
                metrics={"saved": saved},
            )
        saved += 1
        if decision.expected_outcome == "Accept":
            accept += 1
        else:
            edit += 1
    return StepResult(
        "label_loop",
        saved == len(examples),
        metrics={"saved": saved, "accept": accept, "edit": edit},
    )


async def _step_evaluate(client: httpx.AsyncClient, project_id: str) -> StepResult:
    resp = await client.post(
        f"/v1/projects/{project_id}/evaluation_runs",
        json={},
        timeout=30.0,
    )
    if resp.status_code != 201:
        return StepResult(
            "evaluate",
            False,
            detail=f"HTTP {resp.status_code}: {resp.text[:300]}",
        )
    run = resp.json()
    final = await _wait_for_run_terminal(
        client,
        project_id=project_id,
        run_kind="evaluation_runs",
        run_id=run["run_id"],
        timeout_s=300.0,
    )
    ok = final["status"] in ("completed", "incomplete")
    return StepResult(
        "evaluate",
        ok,
        detail=f"final status: {final['status']}",
        metrics={"run_id": run["run_id"], "status": final["status"]},
    )


async def _step_check_gate(
    client: httpx.AsyncClient, project_id: str
) -> tuple[StepResult, str | None]:
    resp = await client.get(f"/v1/projects/{project_id}/scaleup_gate", timeout=10.0)
    resp.raise_for_status()
    gate = resp.json()
    status = gate.get("gate_status")
    return (
        StepResult(
            "check_gate",
            status in ("ready", "not_ready"),  # mechanically completed
            detail=f"gate_status={status}",
            metrics=gate,
        ),
        status,
    )


async def _step_batch_label(client: httpx.AsyncClient, project_id: str) -> StepResult:
    """Run a batch labeling pass — succeeds even if Scale-Up gate is not ready.

    The integration checkpoint is mechanical (does the pipeline run end to
    end?), not a quality gate. Gate-blocked runs return 409 from the
    endpoint; we surface that as the step's detail rather than a hard fail.
    """
    resp = await client.post(
        f"/v1/projects/{project_id}/batch_label_runs",
        json={},
        timeout=30.0,
    )
    if resp.status_code == 409:
        return StepResult(
            "batch_label",
            True,
            detail=f"gate-blocked (expected at small scale): {resp.text[:200]}",
            metrics={"skipped_due_to_gate": True},
        )
    if resp.status_code != 201:
        return StepResult(
            "batch_label",
            False,
            detail=f"HTTP {resp.status_code}: {resp.text[:300]}",
        )
    run = resp.json()
    final = await _wait_for_run_terminal(
        client,
        project_id=project_id,
        run_kind="batch_label_runs",
        run_id=run["run_id"],
        timeout_s=300.0,
        terminal_states=("completed", "canceled", "failed", "paused"),
    )
    return StepResult(
        "batch_label",
        final["status"] == "completed",
        detail=f"final status: {final['status']}",
        metrics={"run_id": run["run_id"], "status": final["status"]},
    )


async def _step_dataset_export(
    client: httpx.AsyncClient,
    *,
    project_id: str,
    export_field_mode: str,
) -> tuple[StepResult, str | None]:
    """Export a Cosmos-RL training dataset; return path to artifact."""
    payload = {
        "dataset_intent": "training",
        "export_field_mode": export_field_mode,
        "label_tier_filter": "verified_only",
    }
    resp = await client.post(
        f"/v1/projects/{project_id}/dataset_exports",
        json=payload,
        timeout=60.0,
    )
    if resp.status_code not in (200, 201):
        return (
            StepResult(
                "dataset_export",
                False,
                detail=f"HTTP {resp.status_code}: {resp.text[:300]}",
            ),
            None,
        )
    export = resp.json()
    export_id = export["dataset_export_id"]
    # The archive builds in a background task (status lifecycle, migration
    # 045): poll the record until it reaches a terminal state.
    deadline = asyncio.get_running_loop().time() + 300.0
    while export.get("status") == "running":
        if asyncio.get_running_loop().time() > deadline:
            return (
                StepResult(
                    "dataset_export",
                    False,
                    detail="export still running after 300s",
                ),
                None,
            )
        await asyncio.sleep(2.0)
        poll = await client.get(
            f"/v1/projects/{project_id}/dataset_exports/{export_id}",
            timeout=30.0,
        )
        if poll.status_code != 200:
            return (
                StepResult(
                    "dataset_export",
                    False,
                    detail=f"poll HTTP {poll.status_code}: {poll.text[:300]}",
                ),
                None,
            )
        export = poll.json()
    if export.get("status") == "failed":
        return (
            StepResult(
                "dataset_export",
                False,
                detail=f"export failed: {export.get('status_reason')}",
            ),
            None,
        )
    artifact_refs = export.get("artifact_refs") or {}
    archive_path = artifact_refs.get("archive_path") or artifact_refs.get("archive_ref")
    return (
        StepResult(
            "dataset_export",
            True,
            metrics={
                "dataset_export_id": export["dataset_export_id"],
                "example_count": export.get("example_count"),
                "checksum_sha256": artifact_refs.get("checksum_sha256"),
            },
        ),
        archive_path,
    )


def _validate_cosmos_rl_archive(archive_path: Path) -> tuple[bool, str]:
    """Cosmos-RL dataset format: top-level JSON array of {id, images, conversations}."""
    if not archive_path.exists():
        return False, f"archive not found at {archive_path}"
    try:
        with tarfile.open(archive_path, "r:gz") as tf:
            try:
                annotations_member = tf.getmember("annotations.json")
            except KeyError:
                # Some exporters nest under a single top-level dir.
                annotations_member = next(
                    (m for m in tf.getmembers() if m.name.endswith("annotations.json")),
                    None,
                )
                if annotations_member is None:
                    return False, "annotations.json not in archive"
            f = tf.extractfile(annotations_member)
            if f is None:
                return False, "annotations.json unreadable"
            annotations = json.loads(f.read().decode("utf-8"))
    except (tarfile.TarError, json.JSONDecodeError, OSError) as e:
        return False, f"archive parse error: {e}"

    if not isinstance(annotations, list):
        return (
            False,
            f"annotations.json is not a top-level array (got {type(annotations).__name__})",
        )
    if not annotations:
        return False, "annotations.json is empty"

    for i, sample in enumerate(annotations):
        if not isinstance(sample, dict):
            return False, f"sample {i} is not an object"
        if "id" not in sample:
            return False, f"sample {i} missing 'id'"
        images = sample.get("images")
        if not isinstance(images, list) or len(images) != 1:
            return False, f"sample {i}: images must be a length-1 array"
        convos = sample.get("conversations")
        if not isinstance(convos, list) or len(convos) != 2:
            return False, f"sample {i}: conversations must be a length-2 array"
        # Turn 1 (human): {"from": "human", "value": "<image>\n..."}
        if convos[0].get("from") != "human":
            return False, f"sample {i}: turn 1 must be from=human"
        if "<image>" not in (convos[0].get("value") or ""):
            return False, f"sample {i}: turn 1 missing literal <image> token"
        # Turn 2 (gpt): {"from": "gpt", "value": "{...json string...}"}
        if convos[1].get("from") != "gpt":
            return False, f"sample {i}: turn 2 must be from=gpt"
        gpt_value = convos[1].get("value")
        if not isinstance(gpt_value, str):
            return (
                False,
                f"sample {i}: turn 2 value must be a JSON string (got {type(gpt_value).__name__})",
            )
        try:
            parsed = json.loads(gpt_value)
        except json.JSONDecodeError as e:
            return False, f"sample {i}: turn 2 value not valid JSON: {e}"
        if not isinstance(parsed, dict):
            return False, f"sample {i}: turn 2 parsed value not an object"
    return True, f"validated {len(annotations)} samples"


# ── Top-level driver ────────────────────────────────────────────────────────


async def run_full_pipeline_smoke(
    *,
    backend_url: str,
    image_paths: list[Path],
    label_count: int = 10,
    export_field_mode: str = "all",
    keep_project: bool = False,
    evidence_dir: Path | None = None,
    project_name: str | None = None,
) -> FullPipelineSmokeResult:
    """Drive the full pipeline against a live backend.

    The caller must:
      * Already have the backend running (uvicorn).
      * Have a Teacher-eligible model_config selected (the seeded default).
      * Have ``image_paths`` resolvable from the backend's filesystem (the
        ingest endpoint reads paths server-side).
    """
    started = _now_iso()
    name = project_name or f"full-pipeline-smoke-{started}"
    result = FullPipelineSmokeResult(
        project_id=None, started_at=started, finished_at="", overall_ok=False
    )
    project_id: str | None = None

    async with httpx.AsyncClient(base_url=backend_url) as client:
        # Step 1: create_project
        s1 = await _step_create_project(client, name=name)
        result.steps.append(s1)
        if not s1.ok:
            result.finished_at = _now_iso()
            return result
        project_id = s1.metrics["project_id"]
        result.project_id = project_id

        try:
            # Step 2: verify_teacher_endpoint
            s2 = await _step_verify_teacher_endpoint(client, project_id)
            result.steps.append(s2)
            if not s2.ok:
                return result

            # Step 3: ingest
            paths_to_ingest = image_paths[: max(label_count, len(image_paths))]
            s3 = await _step_ingest(
                client, project_id=project_id, image_paths=paths_to_ingest
            )
            result.steps.append(s3)
            if not s3.ok:
                return result

            # Step 4: save_guidance
            s4 = await _step_save_guidance(client, project_id)
            result.steps.append(s4)
            if not s4.ok:
                return result

            # Step 5: label_loop. ExampleQueryResponse wraps each row in
            # {example: {...}, verified_label: ...} — flatten to bare
            # example dicts for the labeling helper.
            ex_resp = await client.get(
                f"/v1/projects/{project_id}/examples",
                params={"limit": label_count, "state": "Unlabeled"},
                timeout=10.0,
            )
            ex_resp.raise_for_status()
            examples = [
                item["example"] for item in ex_resp.json()["items"][:label_count]
            ]
            s5 = await _step_label_loop(
                client,
                project_id=project_id,
                examples=examples,
                image_paths=paths_to_ingest,
            )
            result.steps.append(s5)
            result.label_count = s5.metrics.get("saved", 0)
            if not s5.ok:
                return result

            # Step 6: evaluate
            s6 = await _step_evaluate(client, project_id)
            result.steps.append(s6)
            if not s6.ok:
                return result

            # Step 7: check_gate
            s7, gate_status = await _step_check_gate(client, project_id)
            result.steps.append(s7)
            result.gate_status = gate_status

            # Step 8: batch_label
            s8 = await _step_batch_label(client, project_id)
            result.steps.append(s8)
            # batch_label may be gate-blocked (409) at small scale — that's
            # an expected mechanical outcome, not a smoke failure.

            # Step 9 + 10: dataset_export + Cosmos-RL format validation
            s9, archive_path = await _step_dataset_export(
                client,
                project_id=project_id,
                export_field_mode=export_field_mode,
            )
            result.steps.append(s9)
            if s9.ok:
                result.dataset_example_count = s9.metrics.get("example_count") or 0
                result.dataset_export_sha = s9.metrics.get("checksum_sha256")
                if archive_path:
                    valid, detail = _validate_cosmos_rl_archive(Path(archive_path))
                    result.cosmos_rl_format_validated = valid
                    result.steps.append(
                        StepResult(
                            "validate_cosmos_rl_format",
                            valid,
                            detail=detail,
                        )
                    )
                else:
                    result.steps.append(
                        StepResult(
                            "validate_cosmos_rl_format",
                            False,
                            detail="dataset_export response did not include archive_path",
                        )
                    )

            # Required steps for overall_ok: every step except batch_label
            # (which can legitimately be gate-blocked).
            critical = {s.name: s.ok for s in result.steps if s.name != "batch_label"}
            result.overall_ok = all(critical.values())
            return result
        finally:
            if not keep_project and project_id:
                # Soft-cleanup: archive the project so it doesn't pollute
                # subsequent CI runs. Failures here are non-fatal.
                try:
                    await client.post(
                        f"/v1/projects/{project_id}:archive", timeout=10.0
                    )
                except Exception:  # noqa: BLE001 — cleanup best-effort
                    logger.warning(
                        "post-smoke archive failed for project %s", project_id
                    )
            result.finished_at = _now_iso()
            if evidence_dir is not None:
                evidence_dir.mkdir(parents=True, exist_ok=True)
                out = evidence_dir / "full_pipeline_smoke_acceptance.json"
                out.write_text(json.dumps(result.as_acceptance_dict(), indent=2))
                logger.info("Wrote evidence to %s", out)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="full_pipeline_smoke",
        description="Final integration checkpoint — full-pipeline smoke "
        "against a live Blueprint backend.",
    )
    p.add_argument("--backend-url", default="http://127.0.0.1:8000")
    p.add_argument(
        "--image-source-dir",
        type=Path,
        required=True,
        help="Directory of images to ingest. Backend must have read access.",
    )
    p.add_argument("--label-count", type=int, default=10)
    p.add_argument(
        "--export-field-mode",
        choices=("all", "aux_and_core", "core_only"),
        default="all",
    )
    p.add_argument(
        "--keep-project",
        action="store_true",
        help="Do not archive the project after the smoke completes.",
    )
    p.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help=(
            "Optional output dir for full_pipeline_smoke_acceptance.json. "
            "Defaults to ~/.vlm_feedback_loop/smoke-runs/"
            "full_pipeline_smoke_<ts>/ when unset."
        ),
    )
    p.add_argument(
        "--project-name",
        default=None,
        help="Override the auto-generated project name.",
    )
    return p.parse_args(argv)


def _gather_image_paths(image_dir: Path, max_count: int) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes
    )
    if not files:
        raise SystemExit(f"no supported images found in {image_dir}")
    return files[:max_count]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    image_paths = _gather_image_paths(args.image_source_dir, max_count=args.label_count)
    evidence_dir = args.evidence_dir or (
        Path.home()
        / ".vlm_feedback_loop"
        / "smoke-runs"
        / f"full_pipeline_smoke_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )

    result = asyncio.run(
        run_full_pipeline_smoke(
            backend_url=args.backend_url,
            image_paths=image_paths,
            label_count=args.label_count,
            export_field_mode=args.export_field_mode,
            keep_project=args.keep_project,
            evidence_dir=evidence_dir,
            project_name=args.project_name,
        )
    )

    print(json.dumps(result.as_acceptance_dict(), indent=2))
    print(f"\nproject_id={result.project_id}")
    return 0 if result.overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

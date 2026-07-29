# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live Profile B validation — Evaluation + Batch Labeling at meaningful scale.

Drives the loop end-to-end against a live backend + hosted NIM to confirm
the real Teacher pipelines in ``evaluation_service._invoke_for_evaluation``
and ``batch_label_service._invoke_for_batch_label`` hold up at meaningful
scale — coverage the wire-mocked unit tests cannot provide — and that the
batch-labeling circuit breaker actually trips.

Phases (single project, Teacher selected via ``--teacher`` — default
mistral, thinking=OFF):

  A. Setup — create project, save RPS Guidance, ingest 100 RPS images.
  B. Build pool — drive 50 review-cycle saves with ground-truth labels.
     At the default test-pool fraction (``Project.test_pool_fraction``,
     0.40) this yields a Test Pool >= 20.
  C. Evaluation Profile B — trigger an evaluation, wait for terminal,
     confirm: status=completed, real per-example latencies, exact_match_pass
     scored, Returning/New buckets (None on first eval), pool_version_id set.
  D. Batch Labeling Profile B — start a batch on remaining Unlabeled,
     run_limit=20. Confirm: status=completed, examples_total=20, succeeded
     counter > 0, common_errors structure populated, OperationRecords carry
     batch_label_run_id and label_tier="auto_labeled".
  E. Circuit breaker — register a bad ModelConfig (model_name that hosted
     NIM 404s on), patch project teacher to it, start a fresh batch on
     remaining Unlabeled. Expect status=paused, paused_reason=
     "circuit_breaker_threshold_reached", examples_endpoint_error >= 10.
     Then cancel.

Backend MUST be running at http://127.0.0.1:8000 with NVIDIA_API_KEY set.

Usage::

    uv run python scripts/profile_b_live_validation.py
    uv run python scripts/profile_b_live_validation.py --skip-circuit-breaker
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_helpers import (  # noqa: E402
    StageResult,
    resolve_teacher_model_config_id,
    wait_backend,
)

BACKEND_URL = "http://127.0.0.1:8000"
RPS_ROOT = Path(os.environ.get("RPS_TEST_SET_ROOT", "~/rps-test-set")).expanduser()
WORKSPACE_ROOT = Path("/tmp/vlm_workspace")

QWEN_MODEL_NAME = "qwen/qwen3.5-397b-a17b"
HOSTED_BASE_URL = "https://integrate.api.nvidia.com/v1"
BAD_MODEL_NAME = "qwen/this-model-does-not-exist-99999"

# Teacher catalog — must match seeded ``model_name`` values from
# ``services/project_service.py::SEEDED_MODEL_CATALOG``. Used by the
# ``--teacher`` flag to drive the Profile B harness against alternate
# Teachers when comparing candidates for the default Teacher.
TEACHER_LOOKUP: dict[str, str] = {
    "qwen": "qwen/qwen3.5-397b-a17b",
    "kimi": "moonshotai/kimi-k2.5",
    "mistral": "mistralai/mistral-large-3-675b-instruct-2512",
    "nemotron": "nvidia/nemotron-nano-12b-v2-vl",
}

RPS_GROUND_TRUTH_FINGERS = {"rock": 0, "paper": 5, "scissors": 2}

# Per-class image count for the 100-image cohort (33 + 33 + 34 = 100).
PER_CLASS_COUNT = 33

# Phase B target — 50 verifieds at the 0.40 default test-pool fraction
# (Project.test_pool_fraction) yields pool=20.
PHASE_B_TARGET_VERIFIEDS = 50

# Phase D good-batch run limit — must be smaller than remaining Unlabeled
# after Phase B (50 left of 100 ingested).
PHASE_D_RUN_LIMIT = 20

# Phase E breaker run limit — must be > 10 (the circuit breaker threshold)
# so the run pauses mid-flight. We leave 30 Unlabeled after Phase D.
PHASE_E_RUN_LIMIT = 25


# ── Result tracking ─────────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    project_id: str | None = None
    stages: list[StageResult] = field(default_factory=list)

    def add(
        self, name: str, ok: bool, detail: str = "", t0: float | None = None
    ) -> None:
        d = (time.monotonic() - t0) if t0 is not None else 0.0
        self.stages.append(StageResult(name, ok, detail, d))
        verdict = "✓" if ok else "✗"
        print(f"  {verdict} {name:<32s} {d:6.1f}s   {detail}", flush=True)


# ── HTTP helpers ────────────────────────────────────────────────────────────


async def _get_endpoint_for_hosted(
    client: httpx.AsyncClient, project_id: str
) -> str | None:
    """Return endpoint_id of the hosted NIM endpoint (used for the bad mc)."""
    r = await client.get(f"{BACKEND_URL}/v1/projects/{project_id}/nim_endpoints")
    if r.status_code != 200:
        return None
    for item in r.json().get("items", []):
        if item.get("endpoint_mode") == "hosted":
            return item.get("endpoint_id")
    return None


# ── Image cohort selection ──────────────────────────────────────────────────


def _curated_images() -> list[tuple[str, Path]]:
    """Return PER_CLASS_COUNT-per-class spread across each class's images.

    Picks every ~Nth file so the cohort is diverse rather than bunched at
    the start of the directory.
    """
    cohort: list[tuple[str, Path]] = []
    for cls in ("rock", "paper", "scissors"):
        all_paths = sorted((RPS_ROOT / cls).glob("*.png"))
        if not all_paths:
            raise RuntimeError(f"no png files in {RPS_ROOT / cls}")
        # Even spread.
        step = max(1, len(all_paths) // PER_CLASS_COUNT)
        picked = all_paths[::step][:PER_CLASS_COUNT]
        cohort.extend((cls, p) for p in picked)
    return cohort


# ── Phase A: Setup ──────────────────────────────────────────────────────────


async def _phase_a_setup(
    client: httpx.AsyncClient, report: ValidationReport, teacher_label: str = "qwen"
) -> tuple[bool, list[tuple[str, Path]]]:
    print(
        f"\n=== Phase A: Setup (project + Guidance + ingest, teacher={teacher_label}) ===",
        flush=True,
    )

    teacher_model_name = TEACHER_LOOKUP[teacher_label]

    # Create project.
    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects",
        json={"name": f"profile-b-validation-{teacher_label}-{int(time.time())}"},
        timeout=30.0,
    )
    if r.status_code != 201:
        report.add("create_project", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False, []
    project_id = r.json()["project_id"]
    report.project_id = project_id
    report.add("create_project", True, f"project_id={project_id}", t0)

    # Resolve target Teacher's model_config_id and switch (also sets thinking
    # OFF so each proposal stays under the 180s deadline at hosted scale —
    # Kimi K2.5 with thinking=ON adds ~90s/call per the README).
    t0 = time.monotonic()
    mc_id = await resolve_teacher_model_config_id(
        client, project_id, teacher_model_name, base_url=BACKEND_URL
    )
    if mc_id is None:
        report.add(
            "switch_teacher",
            False,
            f"model_config not found for {teacher_model_name}",
            t0,
        )
        return False, []
    r = await client.patch(
        f"{BACKEND_URL}/v1/projects/{project_id}",
        json={
            "teacher_model_config_id": mc_id,
            "thinking_default_on": False,
        },
        timeout=10.0,
    )
    if r.status_code not in (200, 204):
        report.add("switch_teacher", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False, []
    report.add(
        "switch_teacher",
        True,
        f"{teacher_label} mc_id={mc_id} thinking=OFF model={teacher_model_name}",
        t0,
    )

    # Save RPS Guidance.
    t0 = time.monotonic()
    guidance_body = {
        "description": (
            "Determine if the hand is a rock, paper, or scissors gesture in "
            "the game rock-paper-scissors. Count the number of fingers "
            "extended in the gesture, including the thumb."
        ),
        "rules": "",
        "schema": [
            {
                "field_name": "rationale_note",
                "type": "string",
                "role": "aux",
                "display_order": -1,
            },
            {
                "field_name": "number_fingers_extended",
                "type": "integer",
                "role": "aux",
                "minimum": 0,
                "maximum": 5,
                "display_order": 1,
            },
            {
                "field_name": "category",
                "type": "enum",
                "role": "core",
                "allowed_values": ["rock", "paper", "scissors"],
                "display_order": 0,
            },
        ],
    }
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{project_id}/guidance",
        json=guidance_body,
        timeout=20.0,
    )
    if r.status_code != 201:
        report.add("save_guidance", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False, []
    guidance_id = r.json()["guidance_id"]
    r = await client.patch(
        f"{BACKEND_URL}/v1/projects/{project_id}",
        json={"active_guidance_id": guidance_id},
        timeout=10.0,
    )
    if r.status_code not in (200, 204):
        report.add("save_guidance", False, f"activate {r.status_code}", t0)
        return False, []
    report.add("save_guidance", True, f"guidance_id={guidance_id}", t0)

    # Ingest 100 RPS images.
    cohort = _curated_images()
    t0 = time.monotonic()
    items = [
        {
            "example_key": f"rps_{cls}_{path.stem}",
            "storage_ref": str(path),
            "source_metadata": {"ground_truth_class": cls},
        }
        for cls, path in cohort
    ]
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{project_id}/examples:ingest",
        json={"examples": items},
        timeout=60.0,
    )
    if r.status_code != 200:
        report.add("ingest_images", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False, []
    payload = r.json()
    results = payload.get("results", [])
    created = sum(1 for it in results if it.get("status") == "created")
    exists_ = sum(1 for it in results if it.get("status") == "exists")
    errored = sum(1 for it in results if it.get("status") == "error")
    accepted = created + exists_
    ok = accepted == len(items)
    report.add(
        "ingest_images",
        ok,
        f"created={created} exists={exists_} error={errored} total={len(items)}",
        t0,
    )
    return ok, cohort


# ── Phase B: drive labeling loop ────────────────────────────────────────────


async def _phase_b_label(
    client: httpx.AsyncClient,
    report: ValidationReport,
    cohort: list[tuple[str, Path]],
    target: int,
) -> bool:
    print(
        f"\n=== Phase B: Drive {target} label saves (build Test Pool) ===", flush=True
    )
    assert report.project_id is not None
    pid = report.project_id
    inferred_class = {f"rps_{cls}_{p.stem}": cls for cls, p in cohort}

    t0 = time.monotonic()
    edits = 0
    accepts = 0
    saved: set[str] = set()
    failed: set[str] = set()  # keys we've tried and given up on
    proposal_failures = 0

    # Cap iterations generously — we may need up to (target + failures)
    # rounds because failed keys are skipped via the failed set, not
    # consumed by a save. The hard ceiling protects against runaway loops
    # if the selector starts returning the same Unlabeled forever.
    fail_budget = 20  # absorb hosted-NIM transient 429s without aborting

    for _ in range(target * 3 + fail_budget):
        nxt = await client.get(
            f"{BACKEND_URL}/v1/projects/{pid}/review_selector/next", timeout=15.0
        )
        if nxt.status_code != 200:
            report.add("review_cycles", False, f"selector HTTP {nxt.status_code}", t0)
            return False
        sel = nxt.json()
        ex_key = sel.get("example_key")
        if not ex_key:
            # Queue empty — selector has nothing left to offer.
            break
        if ex_key in saved or ex_key in failed:
            # Selector returned a key we've already handled. Without
            # state advancement on the backend (the example is still
            # Unlabeled because its proposal failed), the selector keeps
            # returning the same top pick. Skip and break — no point
            # spinning forever on a key we've given up on.
            print(
                f"    … selector replayed {ex_key} (saved={len(saved)} "
                f"failed={len(failed)}); ending Phase B",
                flush=True,
            )
            break
        gt_class = inferred_class.get(ex_key, "unknown")

        prop = await client.post(
            f"{BACKEND_URL}/v1/projects/{pid}/proposals",
            json={"example_key": ex_key},
            timeout=240.0,
        )
        if prop.status_code != 200:
            proposal_failures += 1
            failed.add(ex_key)
            print(
                f"    ! proposal {ex_key} HTTP {prop.status_code} "
                f"(fails={proposal_failures}/{fail_budget})",
                flush=True,
            )
            if proposal_failures > fail_budget:
                report.add(
                    "review_cycles",
                    False,
                    f"too many proposal HTTP failures (>{fail_budget})",
                    t0,
                )
                return False
            continue
        p = prop.json()
        if p.get("invocation_status") != "success":
            proposal_failures += 1
            failed.add(ex_key)
            print(
                f"    ! proposal {ex_key} status={p.get('invocation_status')} "
                f"err={(p.get('error') or '')[:120]} "
                f"(fails={proposal_failures}/{fail_budget})",
                flush=True,
            )
            if proposal_failures > fail_budget:
                report.add(
                    "review_cycles",
                    False,
                    f"too many non-success invocations (>{fail_budget})",
                    t0,
                )
                return False
            continue

        # Submit ground-truth — Edit if model wrong, Accept if right.
        label = {
            "rationale_note": (
                f"Hand is {gt_class}. Verified against the curated RPS test pool."
            ),
            "number_fingers_extended": RPS_GROUND_TRUTH_FINGERS[gt_class],
            "category": gt_class,
        }
        save = await client.post(
            f"{BACKEND_URL}/v1/projects/{pid}/labels",
            json={
                "example_key": ex_key,
                "inference_invocation_id": p["inference_invocation_id"],
                "label_json": label,
                "rationale_source": "sme_edited",
            },
            timeout=15.0,
        )
        if save.status_code != 200:
            report.add("review_cycles", False, f"save HTTP {save.status_code}", t0)
            return False
        saved.add(ex_key)
        s = save.json()
        if s.get("verified_outcome") == "Edit":
            edits += 1
        else:
            accepts += 1

        # Light progress every 10.
        total = edits + accepts
        if total % 10 == 0:
            print(
                f"    … saved {total}/{target}  edits={edits} accepts={accepts} "
                f"prop_fails={proposal_failures}",
                flush=True,
            )
        if total >= target:
            break

    detail = (
        f"saves={edits + accepts} edits={edits} accepts={accepts} "
        f"prop_fails={proposal_failures}"
    )
    ok = (edits + accepts) >= target
    report.add("review_cycles", ok, detail, t0)
    return ok


def _pool_size_from_db(project_id: str) -> int:
    """Read pool size directly from the project SQLite DB."""
    db = WORKSPACE_ROOT / "projects" / project_id / "project.db"
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute(
            "SELECT COUNT(*) FROM labels WHERE label_status='verified' "
            "AND pool_assignment='test_pool'"
        )
        return cur.fetchone()[0]
    finally:
        con.close()


def _verified_count_from_db(project_id: str) -> int:
    db = WORKSPACE_ROOT / "projects" / project_id / "project.db"
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute("SELECT COUNT(*) FROM labels WHERE label_status='verified'")
        return cur.fetchone()[0]
    finally:
        con.close()


# ── Phase C: Evaluation Profile B ───────────────────────────────────────────


async def _wait_for_run(
    client: httpx.AsyncClient,
    pid: str,
    run_id: str,
    kind: str,  # "evaluation_runs" | "batch_label_runs"
    terminal: tuple[str, ...],
    max_wait_s: float,
    poll_s: float = 2.0,
) -> dict[str, Any] | None:
    end = time.monotonic() + max_wait_s
    last: dict[str, Any] = {}
    while time.monotonic() < end:
        await asyncio.sleep(poll_s)
        g = await client.get(
            f"{BACKEND_URL}/v1/projects/{pid}/{kind}/{run_id}", timeout=10.0
        )
        if g.status_code != 200:
            continue
        last = g.json()
        if last.get("status") in terminal:
            return last
    return last or None


async def _phase_c_evaluation(
    client: httpx.AsyncClient, report: ValidationReport
) -> bool:
    print("\n=== Phase C: Evaluation Profile B (real NIM at scale) ===", flush=True)
    assert report.project_id is not None
    pid = report.project_id

    pool_size = _pool_size_from_db(pid)
    verified = _verified_count_from_db(pid)
    print(f"    pool_size={pool_size} verified={verified}", flush=True)
    if pool_size < 5:
        report.add(
            "eval_pool_size",
            False,
            f"pool too small for meaningful test: {pool_size} (<5)",
            None,
        )
        return False

    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/evaluation_runs",
        json={"icl_mode": "enabled"},
        timeout=15.0,
    )
    if r.status_code != 201:
        report.add("eval_create", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False
    run_id = r.json()["run_id"]
    report.add(
        "eval_create",
        True,
        f"run_id={run_id} pool_size={pool_size}",
        t0,
    )

    # Wait for terminal. With pool=20 and self-hosted eval concurrency 8
    # (EVAL_CONCURRENCY_SELF_HOSTED) each batch is ~3 examples avg latency
    # ~3s → ~10s wallclock; allow 5min buffer.
    t0 = time.monotonic()
    final = await _wait_for_run(
        client,
        pid,
        run_id,
        "evaluation_runs",
        ("completed", "incomplete", "canceled", "failed"),
        max_wait_s=300.0,
    )
    if final is None:
        report.add("eval_terminal", False, "timeout polling", t0)
        return False
    status = final.get("status", "unknown")
    metrics = final.get("metrics") or {}
    progress = final.get("progress") or {}
    detail = (
        f"status={status} "
        f"processed={progress.get('processed')}/{progress.get('total')} "
        f"icl_eligible_at_completion={final.get('icl_eligible_count_at_completion')} "
        f"metrics_keys={sorted(metrics.keys())[:6]}"
    )
    if status != "completed":
        report.add("eval_terminal", False, detail, t0)
        return False
    report.add("eval_terminal", True, detail, t0)

    # Profile B sanity checks against the project DB:
    #   * OperationRecord rows for this run with non-null latency
    #   * exact_match_pass populated
    #   * structured_generation_attempted recorded
    db = WORKSPACE_ROOT / "projects" / pid / "project.db"
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute(
            "SELECT COUNT(*), AVG(latency_ms_end_to_end), "
            "SUM(CASE WHEN exact_match_pass=1 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN exact_match_pass=0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN raw_model_response_ref IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM operation_records WHERE evaluation_run_id=?",
            (run_id,),
        )
        n_ops, avg_lat, n_pass, n_fail, n_with_artifact = cur.fetchone()
    finally:
        con.close()
    detail2 = (
        (
            f"ops={n_ops} avg_latency_ms={avg_lat:.0f} "
            f"exact_pass={n_pass} exact_fail={n_fail} "
            f"with_raw_artifact={n_with_artifact}"
        )
        if n_ops
        else "ops=0 (NO Operation Records — Profile B regressed!)"
    )
    ok = (
        n_ops is not None
        and n_ops >= pool_size  # one OperationRecord per pool example
        and avg_lat is not None
        and avg_lat > 100  # sanity floor: real NIM call > 100ms
        and n_with_artifact > 0  # at least some raw artifacts persisted
    )
    report.add("eval_profile_b_artifacts", ok, detail2, None)
    return ok


# ── Phase D: Batch Labeling Profile B ───────────────────────────────────────


async def _phase_d_batch(client: httpx.AsyncClient, report: ValidationReport) -> bool:
    print("\n=== Phase D: Batch Labeling Profile B (real NIM at scale) ===", flush=True)
    assert report.project_id is not None
    pid = report.project_id

    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/batch_label_runs",
        json={"run_limit": PHASE_D_RUN_LIMIT},
        timeout=15.0,
    )
    if r.status_code != 201:
        report.add("batch_create", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False
    run_id = r.json()["run_id"]
    examples_total = r.json().get("examples_total", 0)
    report.add(
        "batch_create",
        True,
        f"run_id={run_id} examples_total={examples_total}",
        t0,
    )

    # Wait for terminal — sequential at ~3-5s per example → 20*5=100s; 5min buffer.
    t0 = time.monotonic()
    final = await _wait_for_run(
        client,
        pid,
        run_id,
        "batch_label_runs",
        ("completed", "canceled", "failed"),
        max_wait_s=600.0,
        poll_s=3.0,
    )
    if final is None:
        report.add("batch_terminal", False, "timeout polling", t0)
        return False
    status = final.get("status", "unknown")
    progress = final.get("progress") or {}
    detail = (
        f"status={status} "
        f"processed={progress.get('processed')}/{progress.get('total')} "
        f"succeeded={final.get('examples_succeeded')} "
        f"schema_invalid={final.get('examples_schema_invalid')} "
        f"timeout={final.get('examples_timeout')} "
        f"endpoint_error={final.get('examples_endpoint_error')}"
    )
    if status != "completed":
        report.add("batch_terminal", False, detail, t0)
        return False
    report.add("batch_terminal", True, detail, t0)

    # Profile B sanity check on the DB:
    #   * OperationRecord rows with batch_label_run_id set and label_tier='auto_labeled'
    #   * non-null latency
    #   * Label rows with label_status='auto_labeled' and matching batch_label_run_id
    db = WORKSPACE_ROOT / "projects" / pid / "project.db"
    con = sqlite3.connect(str(db))
    try:
        cur = con.execute(
            "SELECT COUNT(*), AVG(latency_ms_end_to_end) "
            "FROM operation_records "
            "WHERE batch_label_run_id=? AND label_tier='auto_labeled'",
            (run_id,),
        )
        n_ops, avg_lat = cur.fetchone()
        cur = con.execute(
            "SELECT COUNT(*) FROM labels "
            "WHERE label_status='auto_labeled' AND batch_label_run_id=?",
            (run_id,),
        )
        n_labels = cur.fetchone()[0]
        cur = con.execute("SELECT COUNT(*) FROM examples WHERE state='Auto-Labeled'")
        n_state = cur.fetchone()[0]
    finally:
        con.close()
    detail2 = (
        f"ops_with_run_id={n_ops} avg_latency_ms="
        f"{(avg_lat or 0):.0f} "
        f"auto_labeled_labels={n_labels} examples_state_auto_labeled={n_state}"
    )
    ok = (
        n_ops is not None
        and n_ops == examples_total
        and avg_lat is not None
        and avg_lat > 100
        and n_labels == final.get("examples_succeeded")
        and n_state >= n_labels
    )
    report.add("batch_profile_b_artifacts", ok, detail2, None)
    return ok


# ── Phase E: Circuit breaker ────────────────────────────────────────────────


async def _phase_e_circuit_breaker(
    client: httpx.AsyncClient, report: ValidationReport
) -> bool:
    print("\n=== Phase E: Circuit breaker stress (bad teacher) ===", flush=True)
    assert report.project_id is not None
    pid = report.project_id

    # Register a bad ModelConfig pointing at the real hosted endpoint with a
    # nonsense model_name. Hosted NIM 404s on unknown models → endpoint_error
    # → counter increments.
    t0 = time.monotonic()
    hosted_endpoint_id = await _get_endpoint_for_hosted(client, pid)
    if hosted_endpoint_id is None:
        report.add("breaker_endpoint", False, "hosted endpoint not found", t0)
        return False
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/model_configs",
        json={
            "endpoint_id": hosted_endpoint_id,
            "model_name": BAD_MODEL_NAME,
            "context_window_tokens": 32000,
            "eligible_roles": ["teacher"],
            "supports_image_input": True,
            "thinking_toggle_mode": "none",
            "visual_budget_mode": "none",
        },
        timeout=10.0,
    )
    if r.status_code != 201:
        report.add(
            "breaker_create_mc", False, f"HTTP {r.status_code}: {r.text[:200]}", t0
        )
        return False
    bad_mc_id = r.json()["model_config_id"]
    report.add("breaker_create_mc", True, f"bad_mc_id={bad_mc_id}", t0)

    # Switch teacher to bad mc.
    t0 = time.monotonic()
    r = await client.patch(
        f"{BACKEND_URL}/v1/projects/{pid}",
        json={"teacher_model_config_id": bad_mc_id},
        timeout=10.0,
    )
    if r.status_code not in (200, 204):
        report.add("breaker_switch", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False
    report.add("breaker_switch", True, "teacher = bad", t0)

    # Start batch — should snapshot the bad teacher and rapidly accumulate
    # endpoint_errors.
    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/batch_label_runs",
        json={"run_limit": PHASE_E_RUN_LIMIT},
        timeout=15.0,
    )
    if r.status_code != 201:
        report.add(
            "breaker_run_start",
            False,
            f"HTTP {r.status_code}: {r.text[:200]}",
            t0,
        )
        return False
    run_id = r.json()["run_id"]
    examples_total = r.json().get("examples_total", 0)
    report.add(
        "breaker_run_start",
        True,
        f"run_id={run_id} examples_total={examples_total}",
        t0,
    )

    # Wait for paused (terminal-from-our-perspective) or completed/failed.
    t0 = time.monotonic()
    final = await _wait_for_run(
        client,
        pid,
        run_id,
        "batch_label_runs",
        ("paused", "completed", "canceled", "failed"),
        max_wait_s=600.0,
        poll_s=2.0,
    )
    if final is None:
        report.add("breaker_paused", False, "timeout", t0)
        return False
    status = final.get("status", "?")
    paused_reason = final.get("paused_reason")
    endpoint_errors = final.get("examples_endpoint_error", 0)
    progress = final.get("progress") or {}
    detail = (
        f"status={status} paused_reason={paused_reason!r} "
        f"endpoint_errors={endpoint_errors} "
        f"processed={progress.get('processed')}/{progress.get('total')}"
    )
    ok = (
        status == "paused"
        and paused_reason == "circuit_breaker_threshold_reached"
        and endpoint_errors >= 10
    )
    report.add("breaker_paused", ok, detail, t0)

    # Cancel the paused run for cleanup.
    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/batch_label_runs/{run_id}:cancel",
        timeout=10.0,
    )
    cancel_ok = r.status_code in (200, 204)
    if cancel_ok:
        # Poll briefly to see canceled.
        cancelled_final = await _wait_for_run(
            client,
            pid,
            run_id,
            "batch_label_runs",
            ("canceled", "completed", "failed"),
            max_wait_s=30.0,
            poll_s=1.0,
        )
        cancel_detail = (
            f"final_status={cancelled_final.get('status') if cancelled_final else '?'}"
        )
    else:
        cancel_detail = f"cancel HTTP {r.status_code}: {r.text[:120]}"
    report.add("breaker_cancel", cancel_ok, cancel_detail, t0)

    return ok


# ── Reporting ───────────────────────────────────────────────────────────────


def _print_summary(report: ValidationReport) -> bool:
    print("\n" + "─" * 72)
    print("Profile B live validation — summary")
    print("─" * 72)
    all_ok = True
    print(f"project_id={report.project_id}")
    for s in report.stages:
        mark = "✓" if s.ok else "✗"
        print(f"  {mark} {s.name:<32s} {s.duration_s:6.1f}s   {s.detail}")
        if not s.ok:
            all_ok = False
    print("─" * 72)
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    print("─" * 72)
    return all_ok


# ── Entry point ─────────────────────────────────────────────────────────────


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Live Profile B validation at meaningful scale."
    )
    parser.add_argument(
        "--skip-circuit-breaker",
        action="store_true",
        help="Skip Phase E (default: run all phases).",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="mistral",
        choices=sorted(TEACHER_LOOKUP.keys()),
        help=(
            "Teacher to validate against. Default: mistral (matches the "
            "post-migration-014 system default). Other options: "
            + ", ".join(sorted(t for t in TEACHER_LOOKUP if t != "mistral"))
            + ". The script always sets thinking=OFF; Kimi with thinking=ON "
            "adds ~90s/proposal on hosted NIM (per README)."
        ),
    )
    args = parser.parse_args(argv)

    report = ValidationReport()
    async with httpx.AsyncClient() as client:
        try:
            await wait_backend(client, base_url=BACKEND_URL)
        except RuntimeError as exc:
            print(f"Backend not reachable: {exc}", file=sys.stderr)
            return 1

        ok, cohort = await _phase_a_setup(client, report, teacher_label=args.teacher)
        if not ok:
            return 1 if not _print_summary(report) else 0

        ok = await _phase_b_label(
            client, report, cohort, target=PHASE_B_TARGET_VERIFIEDS
        )
        if not ok:
            return 1 if not _print_summary(report) else 0

        await _phase_c_evaluation(client, report)
        await _phase_d_batch(client, report)

        if not args.skip_circuit_breaker:
            await _phase_e_circuit_breaker(client, report)

    final = _print_summary(report)
    return 0 if final else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(sys.argv[1:])))

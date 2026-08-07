# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end live smoke for the ICL loop, per Teacher.

Drives the full lifecycle through the running backend's REST API:

  1. Create a fresh project (per Teacher).
  2. Switch ``teacher_model_config_id`` to the target seeded model.
  3. Save a real Guidance (RPS — rock/paper/scissors).
  4. Ingest 9 RPS images by absolute path (3 per class, from the curated pool).
  5. Drive review/save cycles until ``--target-edits`` Edits accrue
     (default 5) so ICL kicks in.
  6. Trigger an evaluation run and wait for terminal state.
  7. Repeat per requested Teacher; report PASS/FAIL per stage.

Cosmos Reason 2 8B local is intentionally not in the default Teacher list
because this host has no GPU/Docker NIM substrate — pass it explicitly via
``--teachers cosmos_8b_local`` once a local NIM endpoint is registered.

The default hosted matrix starts with the shipped Step 3.7 Flash Teacher and
covers all currently hosted-compatible seeded Teachers. The script assumes the backend
is already running at ``http://127.0.0.1:8000``. Start it via
``./scripts/dev.sh`` (or by booting ``vlm_feedback_loop.main`` directly) before
invoking this smoke.

Usage::

    uv run python scripts/icl_loop_smoke.py
    uv run python scripts/icl_loop_smoke.py --teachers step
    uv run python scripts/icl_loop_smoke.py --teachers step,nemotron
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from vlm_feedback_loop.model_catalog_constants import (
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
    MISTRAL_MEDIUM_3_5,
    NEMOTRON_3_NANO_OMNI_REASONING,
    NEMOTRON_NANO_12B_VL,
    STEP_3_7_FLASH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_helpers import (  # noqa: E402
    StageResult,
    resolve_teacher_model_config_id,
    wait_backend,
)

BACKEND_URL = "http://127.0.0.1:8000"
RPS_ROOT = Path(os.environ.get("RPS_TEST_SET_ROOT", "~/rps-test-set")).expanduser()

TEACHER_LOOKUP: dict[str, str] = {
    "step": STEP_3_7_FLASH,
    "mistral_medium": MISTRAL_MEDIUM_3_5,
    "nemotron": NEMOTRON_NANO_12B_VL,
    "omni": NEMOTRON_3_NANO_OMNI_REASONING,
    # Local-deployed only; explicit opt-in.
    "cosmos_8b_local": COSMOS_REASON2_8B,
    "cosmos_2b_local": COSMOS_REASON2_2B,
}
DEFAULT_TEACHERS = ",".join(
    label for label in TEACHER_LOOKUP if not label.endswith("_local")
)

# 3 images per class. With the default ``target_edits=5`` and the 0.40
# default test-pool fraction (``Project.test_pool_fraction``), 5
# verifieds yields a 2-member Test Pool —
# enough to start an evaluation. 9 ingested gives headroom for transient
# proposal failures without starving the loop.
RPS_IMAGES: list[tuple[str, Path]] = [
    ("rock", RPS_ROOT / "rock" / "testrock01-00.png"),
    ("rock", RPS_ROOT / "rock" / "testrock01-05.png"),
    ("rock", RPS_ROOT / "rock" / "testrock01-02.png"),
    ("paper", RPS_ROOT / "paper" / "testpaper01-00.png"),
    ("paper", RPS_ROOT / "paper" / "testpaper01-05.png"),
    ("paper", RPS_ROOT / "paper" / "testpaper01-10.png"),
    ("scissors", RPS_ROOT / "scissors" / "testscissors01-00.png"),
    ("scissors", RPS_ROOT / "scissors" / "testscissors01-05.png"),
    ("scissors", RPS_ROOT / "scissors" / "testscissors01-10.png"),
]

RPS_GROUND_TRUTH_FINGERS = {"rock": 0, "paper": 5, "scissors": 2}

GUIDANCE_BODY: dict[str, Any] = {
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


# ── Result tracking ─────────────────────────────────────────────────────────


@dataclass
class TeacherSmokeReport:
    teacher_label: str
    teacher_model_name: str
    project_id: str | None = None
    stages: list[StageResult] = field(default_factory=list)

    def add(
        self, name: str, ok: bool, detail: str = "", t0: float | None = None
    ) -> None:
        d = (time.monotonic() - t0) if t0 is not None else 0.0
        self.stages.append(StageResult(name, ok, detail, d))

    @property
    def passed(self) -> bool:
        return all(s.ok for s in self.stages)


# ── Stages ──────────────────────────────────────────────────────────────────


async def _create_project(
    client: httpx.AsyncClient, name: str, report: TeacherSmokeReport
) -> bool:
    t0 = time.monotonic()
    body = {"name": name, "description": "ICL loop end-to-end smoke"}
    r = await client.post(f"{BACKEND_URL}/v1/projects", json=body, timeout=30.0)
    if r.status_code != 201:
        report.add("create_project", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False
    payload = r.json()
    report.project_id = payload["project_id"]
    report.add("create_project", True, f"project_id={report.project_id}", t0)
    return True


async def _switch_teacher(
    client: httpx.AsyncClient,
    report: TeacherSmokeReport,
    model_name: str,
    thinking_default_on: bool = False,
) -> bool:
    """Patch the project to point at the target Teacher.

    Also flips ``thinking_default_on`` to False by default. The smoke wants
    integration coverage rather than reasoning quality, so thinking is OFF
    unless the caller flips it.
    """
    assert report.project_id is not None
    t0 = time.monotonic()
    mc_id = await resolve_teacher_model_config_id(
        client, report.project_id, model_name, base_url=BACKEND_URL
    )
    if mc_id is None:
        report.add(
            "switch_teacher",
            False,
            f"model_config not found for model_name={model_name}",
            t0,
        )
        return False
    r = await client.patch(
        f"{BACKEND_URL}/v1/projects/{report.project_id}",
        json={
            "teacher_model_config_id": mc_id,
            "thinking_default_on": thinking_default_on,
        },
        timeout=10.0,
    )
    if r.status_code not in (200, 204):
        report.add("switch_teacher", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False
    report.add(
        "switch_teacher",
        True,
        f"model_config_id={mc_id} thinking={thinking_default_on}",
        t0,
    )
    return True


async def _save_guidance(client: httpx.AsyncClient, report: TeacherSmokeReport) -> bool:
    """POST a fresh Guidance and set it active."""
    assert report.project_id is not None
    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{report.project_id}/guidance",
        json=GUIDANCE_BODY,
        timeout=20.0,
    )
    if r.status_code != 201:
        report.add("save_guidance", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False
    g = r.json()
    guidance_id = g["guidance_id"]
    # Activate
    r2 = await client.patch(
        f"{BACKEND_URL}/v1/projects/{report.project_id}",
        json={"active_guidance_id": guidance_id},
        timeout=10.0,
    )
    if r2.status_code not in (200, 204):
        report.add(
            "save_guidance",
            False,
            f"activate HTTP {r2.status_code}: {r2.text[:200]}",
            t0,
        )
        return False
    report.add("save_guidance", True, f"guidance_id={guidance_id}", t0)
    return True


async def _ingest_images(client: httpx.AsyncClient, report: TeacherSmokeReport) -> bool:
    """Ingest the curated RPS images by absolute path."""
    assert report.project_id is not None
    t0 = time.monotonic()
    items = [
        {
            "example_key": f"rps_{cls}_{path.stem}",
            "storage_ref": str(path),
            "source_metadata": {"ground_truth_class": cls},
        }
        for cls, path in RPS_IMAGES
    ]
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{report.project_id}/examples:ingest",
        json={"examples": items},
        timeout=30.0,
    )
    # 202 Accepted is the endpoint's success code (rows persist
    # synchronously; pHash/embeddings continue in background).
    if r.status_code not in (200, 202):
        report.add("ingest_images", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return False
    payload = r.json()
    # Per-item statuses are created | exists | error (schemas/example.py);
    # only the first two are usable ingests.
    accepted = sum(
        1
        for item in payload.get("results", [])
        if item.get("status") in ("created", "exists")
    )
    if accepted == 0:
        report.add("ingest_images", False, f"0 accepted: {payload}", t0)
        return False
    report.add("ingest_images", True, f"accepted={accepted}", t0)
    return True


async def _drive_review_cycles(
    client: httpx.AsyncClient,
    report: TeacherSmokeReport,
    target_edits: int,
) -> bool:
    """Drive review_selector → proposal → label until target_edits Edits or pool empty.

    Strategy: every save is a forced Edit (corrected ground-truth label) so
    ICL accrues quickly even when the model gets it right by chance.
    """
    assert report.project_id is not None
    t0 = time.monotonic()
    proj_id = report.project_id
    edits = 0
    proposals_with_icl = 0
    proposals_correct = 0
    seen_keys: set[str] = set()
    inferred_class: dict[str, str] = {
        f"rps_{cls}_{p.stem}": cls for cls, p in RPS_IMAGES
    }

    for _ in range(target_edits + 4):  # cap iterations
        nxt = await client.get(
            f"{BACKEND_URL}/v1/projects/{proj_id}/review_selector/next", timeout=15.0
        )
        if nxt.status_code != 200:
            report.add(
                "review_cycles",
                False,
                f"selector HTTP {nxt.status_code}: {nxt.text[:200]}",
                t0,
            )
            return False
        sel = nxt.json()
        ex_key = sel.get("example_key")
        if not ex_key:
            break
        if ex_key in seen_keys:
            break
        seen_keys.add(ex_key)
        gt_class = inferred_class.get(ex_key, "unknown")

        prop = await client.post(
            f"{BACKEND_URL}/v1/projects/{proj_id}/proposals",
            json={"example_key": ex_key},
            timeout=240.0,
        )
        if prop.status_code != 200:
            report.add(
                "review_cycles",
                False,
                f"proposal {ex_key} HTTP {prop.status_code}: {prop.text[:300]}",
                t0,
            )
            return False
        p = prop.json()
        if p.get("invocation_status") != "success":
            report.add(
                "review_cycles",
                False,
                f"proposal {ex_key} status={p.get('invocation_status')} err={p.get('error')}",
                t0,
            )
            return False

        if p.get("icl_images_attached_count", 0) > 0:
            proposals_with_icl += 1
        prop_cat = (p.get("proposal_json") or {}).get("category")
        if prop_cat == gt_class:
            proposals_correct += 1

        # Force Edit: submit ground-truth label — counts as Edit if the
        # proposal differed; otherwise Accept. Save endpoint requires a
        # rationale_source choice when the SME modifies fields. To keep the
        # smoke deterministic, always submit a corrected rationale + GT.
        label = {
            "rationale_note": (
                f"Hand is {gt_class}. Verified against the curated test pool."
            ),
            "number_fingers_extended": RPS_GROUND_TRUTH_FINGERS[gt_class],
            "category": gt_class,
        }
        save = await client.post(
            f"{BACKEND_URL}/v1/projects/{proj_id}/labels",
            json={
                "example_key": ex_key,
                "inference_invocation_id": p["inference_invocation_id"],
                "label_json": label,
                "rationale_source": "sme_edited",
            },
            timeout=15.0,
        )
        if save.status_code != 200:
            report.add(
                "review_cycles",
                False,
                f"save {ex_key} HTTP {save.status_code}: {save.text[:300]}",
                t0,
            )
            return False
        s = save.json()
        if s.get("verified_outcome") == "Edit":
            edits += 1
        if edits >= target_edits:
            break

    detail = (
        f"reviewed={len(seen_keys)} edits={edits} "
        f"proposals_with_icl_images={proposals_with_icl} "
        f"proposals_correct={proposals_correct}"
    )
    if edits == 0:
        report.add(
            "review_cycles", False, detail + " (no Edits — ICL never accrued)", t0
        )
        return False
    report.add("review_cycles", True, detail, t0)
    return True


async def _run_evaluation(
    client: httpx.AsyncClient, report: TeacherSmokeReport, max_wait_s: float = 300.0
) -> bool:
    """Trigger evaluation and wait for terminal state."""
    assert report.project_id is not None
    t0 = time.monotonic()
    pid = report.project_id
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/evaluation_runs",
        json={"icl_mode": "enabled"},
        timeout=15.0,
    )
    if r.status_code != 201:
        # Possible 422 if Test Pool empty — that's a real failure for this smoke.
        report.add("evaluation_run", False, f"HTTP {r.status_code}: {r.text[:300]}", t0)
        return False
    body = r.json()
    run_id = body.get("run_id") or body.get("evaluation_run_id")
    if not run_id:
        report.add("evaluation_run", False, f"no run_id in create response: {body}", t0)
        return False

    end = time.monotonic() + max_wait_s
    last_status = ""
    while time.monotonic() < end:
        await asyncio.sleep(2.0)
        g = await client.get(
            f"{BACKEND_URL}/v1/projects/{pid}/evaluation_runs/{run_id}", timeout=10.0
        )
        if g.status_code != 200:
            continue
        body = g.json()
        last_status = body.get("status", "")
        if last_status in ("completed", "incomplete", "canceled", "failed"):
            metrics = body.get("rescored_metrics") or body.get("metrics_overall") or {}
            ok = last_status == "completed"
            report.add(
                "evaluation_run",
                ok,
                f"status={last_status} run_id={run_id} metrics={json.dumps(metrics)[:200]}",
                t0,
            )
            return ok
    report.add("evaluation_run", False, f"timeout last_status={last_status}", t0)
    return False


# ── Per-Teacher orchestration ───────────────────────────────────────────────


async def _smoke_one_teacher(
    client: httpx.AsyncClient,
    teacher_label: str,
    teacher_model_name: str,
    target_edits: int,
) -> TeacherSmokeReport:
    report = TeacherSmokeReport(
        teacher_label=teacher_label, teacher_model_name=teacher_model_name
    )
    print(f"\n=== Smoke: {teacher_label} ({teacher_model_name}) ===", flush=True)

    project_name = f"icl-smoke-{teacher_label}-{int(time.time())}"
    if not await _create_project(client, project_name, report):
        return report
    if not await _switch_teacher(client, report, teacher_model_name):
        return report
    if not await _save_guidance(client, report):
        return report
    if not await _ingest_images(client, report):
        return report
    if not await _drive_review_cycles(client, report, target_edits):
        return report
    if not await _run_evaluation(client, report):
        return report
    return report


# ── Reporting ───────────────────────────────────────────────────────────────


def _print_report(reports: list[TeacherSmokeReport]) -> bool:
    print("\n" + "─" * 72)
    print("ICL loop smoke — summary")
    print("─" * 72)
    all_ok = True
    for r in reports:
        verdict = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"\n{verdict}  {r.teacher_label}  ({r.teacher_model_name})")
        print(f"      project_id={r.project_id}")
        for s in r.stages:
            mark = "✓" if s.ok else "✗"
            print(f"        {mark} {s.name:<22s} {s.duration_s:5.1f}s   {s.detail}")
        if not r.passed:
            all_ok = False
    print("\n" + "─" * 72)
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    print("─" * 72)
    return all_ok


# ── Entry point ─────────────────────────────────────────────────────────────


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="ICL-loop end-to-end smoke per Teacher."
    )
    parser.add_argument(
        "--teachers",
        type=str,
        default=DEFAULT_TEACHERS,
        help=(
            f"Comma list of teacher labels. Default: {DEFAULT_TEACHERS}. "
            "Available: " + ", ".join(TEACHER_LOOKUP.keys())
        ),
    )
    parser.add_argument(
        "--target-edits",
        type=int,
        default=5,
        help=(
            "Number of Edits to drive before evaluation (default 5). "
            "Pool target = floor(verified * test_pool_fraction); 5 verifieds "
            "at the default 0.40 yields 2 pool members so evaluation can run."
        ),
    )
    args = parser.parse_args(argv)

    requested = [t.strip() for t in args.teachers.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TEACHER_LOOKUP]
    if unknown:
        print(f"unknown teacher labels: {unknown}", file=sys.stderr)
        return 2

    reports: list[TeacherSmokeReport] = []
    async with httpx.AsyncClient() as client:
        try:
            await wait_backend(client, base_url=BACKEND_URL)
        except RuntimeError as exc:
            print(f"Backend not reachable: {exc}", file=sys.stderr)
            return 1
        for label in requested:
            model_name = TEACHER_LOOKUP[label]
            report = await _smoke_one_teacher(
                client, label, model_name, args.target_edits
            )
            reports.append(report)

    return 0 if _print_report(reports) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(sys.argv[1:])))

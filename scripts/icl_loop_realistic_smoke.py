# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Realistic-load live ICL smoke per hosted Teacher.

Extends ``scripts/icl_loop_smoke.py`` (5-Edit baseline) to a realistic
load: ≥15 SME Edits per Teacher, image-budget pruning exercised across the
current hosted Teacher matrix, large-image transport exercised via a synthetic
>220 KB PNG, anti-anchoring rationale flow exercised live, and evaluation
completion.

Per commercially permitted seeded hosted Teacher:

  1. Create fresh project; switch teacher; save Guidance.
  2. Stage 18 RPS images (6 per class) + 1 synthetic >220 KB PNG → ingest
     19 examples.
  3. Drive 15 SME Edits via the API. After each Edit:
     • assert ``len(icl_example_keys_used) == icl_images_attached_count``
       (invariant from inline ICL injection)
     • track when ``len(icl_example_keys_used)`` stops growing with the
       eligible Edit count — the depth cap / image budget is binding
       while every retained example stays image-grounded.
     • on Edit #5, exercise the anti-anchoring rationale flow:
       call ``:regenerate_rationale`` → save with
       ``rationale_source="teacher_regenerated_approved"``.
  4. Final proposal targets the >220 KB synthetic image. Inspect the persisted
     OperationRecord and assert
     ``image_transport_mode == "base64_inline"`` (hosted endpoints accept
     only inline base64; there is no NVCF asset-upload path on hosted).
  5. Run an evaluation; assert terminal status ``completed``.

The OperationRecord checks read ``project.db`` directly from the
``project_dir`` returned by project creation. Run this harness against a
local-source backend on the same host, not Compose or a remote backend.

Usage::

    uv run python scripts/icl_loop_realistic_smoke.py
    uv run python scripts/icl_loop_realistic_smoke.py --teachers step
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from vlm_feedback_loop.model_catalog_constants import (
    MISTRAL_MEDIUM_3_5,
    NEMOTRON_3_NANO_OMNI_REASONING,
    NEMOTRON_NANO_12B_VL,
    STEP_3_7_FLASH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from icl_loop_smoke import (  # noqa: E402
    BACKEND_URL,
    RPS_ROOT,
    StageResult,
    resolve_teacher_model_config_id,
    wait_backend,
)

LARGE_IMG_PATH = Path("/tmp/vlm_smoke_large.png")
LARGE_IMG_TARGET_BYTES = (
    230_000  # big enough to prove hosted accepts large inline-base64 payloads
)
TARGET_EDITS = 15

# label → (model-name, image-cap)
TEACHER_LOOKUP: dict[str, tuple[str, int]] = {
    "step": (STEP_3_7_FLASH, 8),
    "mistral_medium": (MISTRAL_MEDIUM_3_5, 10),
    "nemotron": (NEMOTRON_NANO_12B_VL, 10),
    "omni": (NEMOTRON_3_NANO_OMNI_REASONING, 8),
}
DEFAULT_TEACHERS = ",".join(TEACHER_LOOKUP)

GUIDANCE_BODY: dict[str, Any] = {
    "description": (
        "Determine if the hand depicted is a rock, paper, or scissors "
        "gesture in the game rock-paper-scissors. Count the number of "
        "fingers extended."
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

RPS_FINGERS = {"rock": 0, "paper": 5, "scissors": 2}


def _ensure_large_image() -> int:
    """Create LARGE_IMG_PATH between 200-500KB to exercise the large-image inline-base64 path.

    Tunes content so PNG compression lands in the target range.
    """
    if (
        LARGE_IMG_PATH.exists()
        and LARGE_IMG_TARGET_BYTES <= LARGE_IMG_PATH.stat().st_size <= 500_000
    ):
        return LARGE_IMG_PATH.stat().st_size
    rng = random.Random(42)
    # 1000x1000 gradient + sparse noise compresses reliably into the
    # 200-500KB target range.
    img = Image.new("RGB", (1000, 1000), color=(64, 64, 64))
    pixels = img.load()
    assert pixels is not None
    for y in range(1000):
        for x in range(1000):
            base_r = (x * 256) // 1000
            base_g = (y * 256) // 1000
            pixels[x, y] = (base_r, base_g, (base_r + base_g) // 2)
    for _ in range(60000):
        x = rng.randint(0, 999)
        y = rng.randint(0, 999)
        pixels[x, y] = (
            rng.randint(0, 255),
            rng.randint(0, 255),
            rng.randint(0, 255),
        )
    LARGE_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(LARGE_IMG_PATH, "PNG", compress_level=6)
    return LARGE_IMG_PATH.stat().st_size


def _sample_rps_paths(count_per_class: int = 6) -> list[tuple[str, Path]]:
    """Return ``count_per_class`` images per RPS class (deterministic order)."""
    out: list[tuple[str, Path]] = []
    for cls in ("rock", "paper", "scissors"):
        files = sorted((RPS_ROOT / cls).glob("*.png"))
        for p in files[:count_per_class]:
            out.append((cls, p))
    return out


@dataclass
class TeacherReport:
    teacher_label: str
    teacher_model: str
    image_cap: int
    project_id: str | None = None
    project_dir: Path | None = None
    stages: list[StageResult] = field(default_factory=list)
    discoveries: list[str] = field(default_factory=list)

    def add(
        self, name: str, ok: bool, detail: str = "", t0: float | None = None
    ) -> None:
        d = (time.monotonic() - t0) if t0 is not None else 0.0
        self.stages.append(StageResult(name, ok, detail, d))

    @property
    def passed(self) -> bool:
        return all(s.ok for s in self.stages)


def _read_op_record(project_dir: Path, inv_id: str) -> dict[str, Any] | None:
    """Read an OperationRecord from the API-reported local project directory."""
    db = project_dir / "project.db"
    if not db.exists():
        return None
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM operation_records WHERE inference_invocation_id = ?",
            (inv_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


async def _smoke_one(
    client: httpx.AsyncClient, teacher_label: str, teacher_model: str, image_cap: int
) -> TeacherReport:
    report = TeacherReport(
        teacher_label=teacher_label,
        teacher_model=teacher_model,
        image_cap=image_cap,
    )
    print(
        f"\n=== Realistic smoke: {teacher_label} ({teacher_model}, cap={image_cap}) ===",
        flush=True,
    )

    # ── Setup ───────────────────────────────────────────────────────────
    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects",
        json={
            "name": f"realistic-smoke-{teacher_label}-{int(time.time())}",
            "description": "realistic-load ICL smoke",
        },
        timeout=30.0,
    )
    if r.status_code != 201:
        report.add("create_project", False, f"HTTP {r.status_code}", t0)
        return report
    project_payload = r.json()
    pid = project_payload["project_id"]
    report.project_id = pid
    report.project_dir = Path(project_payload["project_dir"])
    report.add(
        "create_project",
        True,
        f"project_id={pid} project_dir={report.project_dir}",
        t0,
    )

    t0 = time.monotonic()
    mc_id = await resolve_teacher_model_config_id(client, pid, teacher_model)
    if not mc_id:
        report.add("switch_teacher", False, "model_config not found", t0)
        return report
    r = await client.patch(
        f"{BACKEND_URL}/v1/projects/{pid}",
        json={"teacher_model_config_id": mc_id, "thinking_default_on": False},
        timeout=10.0,
    )
    if r.status_code not in (200, 204):
        report.add("switch_teacher", False, f"HTTP {r.status_code}", t0)
        return report
    report.add("switch_teacher", True, f"mc_id={mc_id}", t0)

    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/guidance", json=GUIDANCE_BODY, timeout=20.0
    )
    if r.status_code != 201:
        report.add("save_guidance", False, f"HTTP {r.status_code}", t0)
        return report
    gid = r.json()["guidance_id"]
    await client.patch(
        f"{BACKEND_URL}/v1/projects/{pid}",
        json={"active_guidance_id": gid},
        timeout=10.0,
    )
    report.add("save_guidance", True, f"guidance_id={gid}", t0)

    # ── Ingest 18 RPS images first; large image deferred to post-edits ──
    t0 = time.monotonic()
    large_size = _ensure_large_image()
    rps_paths = _sample_rps_paths(count_per_class=6)  # 18 images
    items = [
        {
            "example_key": f"rps_{cls}_{p.stem}",
            "storage_ref": str(p),
            "source_metadata": {"ground_truth_class": cls, "synth": False},
        }
        for cls, p in rps_paths
    ]
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/examples:ingest",
        json={"examples": items},
        timeout=60.0,
    )
    if r.status_code != 202:
        report.add("ingest_images", False, f"HTTP {r.status_code}", t0)
        return report
    results = r.json().get("results", [])
    created = sum(1 for x in results if x.get("status") == "created")
    if created != 18:
        report.add("ingest_images", False, f"created={created}, expected 18", t0)
        return report
    report.add(
        "ingest_images",
        True,
        f"created=18 RPS (large image {large_size}B deferred)",
        t0,
    )

    # ── 15 SME Edits via review_selector → forced ground truth ──────────
    t0 = time.monotonic()
    inferred: dict[str, str] = {f"rps_{cls}_{p.stem}": cls for cls, p in rps_paths}
    inferred["large_synth_paper"] = "paper"

    edits = 0
    pruning_observed_at: int | None = None
    invariant_violations = 0
    labeled_keys: set[str] = set()
    proposals_processed = 0

    while edits < TARGET_EDITS and proposals_processed < TARGET_EDITS + 8:
        nxt = await client.get(
            f"{BACKEND_URL}/v1/projects/{pid}/review_selector/next", timeout=15.0
        )
        if nxt.status_code != 200:
            report.add("review_cycles", False, f"selector HTTP {nxt.status_code}", t0)
            return report
        ex_key = nxt.json().get("example_key")
        if not ex_key or ex_key in labeled_keys:
            break
        gt = inferred.get(ex_key, "rock")

        proposals_processed += 1
        prop = await client.post(
            f"{BACKEND_URL}/v1/projects/{pid}/proposals",
            json={"example_key": ex_key},
            timeout=300.0,
        )
        if prop.status_code != 200:
            report.add(
                "review_cycles",
                False,
                f"proposal HTTP {prop.status_code}: {prop.text[:200]}",
                t0,
            )
            return report
        p = prop.json()
        if p.get("invocation_status") != "success":
            report.add(
                "review_cycles",
                False,
                f"proposal status={p.get('invocation_status')} err={p.get('error')}",
                t0,
            )
            return report

        keys_used = p.get("icl_example_keys_used") or []
        imgs_attached = p.get("icl_images_attached_count", 0)

        # Invariant check
        if imgs_attached != len(keys_used):
            invariant_violations += 1
            report.discoveries.append(
                f"INVARIANT VIOLATION on Edit#{edits + 1}: "
                f"icl_images_attached_count={imgs_attached} != "
                f"len(icl_example_keys_used)={len(keys_used)} keys={keys_used}"
            )

        # Capping observation: eligible Edits exceed the retained set while
        # the invariant holds — the depth cap / image budget is binding and
        # the selector kept the most relevant subset.
        if edits > len(keys_used) and pruning_observed_at is None:
            pruning_observed_at = edits + 1

        inv_id = p["inference_invocation_id"]

        # Anti-anchoring rationale flow on Edit #5: regen + approve
        if edits + 1 == 5:
            print(
                "      [edit #5] exercising anti-anchoring rationale regen flow",
                flush=True,
            )
            regen = await client.post(
                f"{BACKEND_URL}/v1/projects/{pid}/examples/{ex_key}:regenerate_rationale",
                json={},
                timeout=240.0,
            )
            if regen.status_code != 200:
                report.add(
                    "review_cycles",
                    False,
                    f"regen_rationale HTTP {regen.status_code}: {regen.text[:200]}",
                    t0,
                )
                return report
            regen_body = regen.json()
            if regen_body.get("invocation_status") != "success":
                report.add(
                    "review_cycles",
                    False,
                    f"regen status={regen_body.get('invocation_status')}",
                    t0,
                )
                return report
            new_rationale = regen_body.get("rationale_note") or "(empty)"
            label_body = {
                "rationale_note": new_rationale,
                "number_fingers_extended": RPS_FINGERS[gt],
                "category": gt,
            }
            rationale_source = "teacher_regenerated_approved"
        else:
            label_body = {
                "rationale_note": (
                    f"Hand is {gt}. Forced corrected ground-truth from synthetic "
                    f"fixture; {RPS_FINGERS[gt]} fingers visible."
                ),
                "number_fingers_extended": RPS_FINGERS[gt],
                "category": gt,
            }
            rationale_source = "sme_edited"

        save = await client.post(
            f"{BACKEND_URL}/v1/projects/{pid}/labels",
            json={
                "example_key": ex_key,
                "inference_invocation_id": inv_id,
                "label_json": label_body,
                "rationale_source": rationale_source,
            },
            timeout=15.0,
        )
        if save.status_code != 200:
            report.add(
                "review_cycles",
                False,
                f"save HTTP {save.status_code}: {save.text[:200]}",
                t0,
            )
            return report
        s = save.json()
        labeled_keys.add(ex_key)
        if s.get("verified_outcome") == "Edit":
            edits += 1
            print(
                f"      [edit #{edits:2d}] {ex_key} ICL_keys={len(keys_used)} "
                f"img_attached={imgs_attached}",
                flush=True,
            )

    detail = (
        f"edits={edits}/{TARGET_EDITS} pruning_observed_at={pruning_observed_at} "
        f"invariant_violations={invariant_violations}"
    )
    if edits < TARGET_EDITS or invariant_violations > 0:
        report.add("review_cycles", False, detail, t0)
        return report
    if image_cap < TARGET_EDITS and pruning_observed_at is None:
        report.discoveries.append(
            f"Image-cap pruning EXPECTED but not observed (cap={image_cap}, "
            f"edits={TARGET_EDITS})"
        )
    report.add("review_cycles", True, detail, t0)

    # ── Now ingest the large image, then propose against it ─────────────
    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/examples:ingest",
        json={
            "examples": [
                {
                    "example_key": "large_synth_paper",
                    "storage_ref": str(LARGE_IMG_PATH),
                    "source_metadata": {"ground_truth_class": "paper", "synth": True},
                }
            ]
        },
        timeout=60.0,
    )
    if r.status_code != 202:
        report.add(
            "large_image_proposal", False, f"ingest large HTTP {r.status_code}", t0
        )
        # Don't return; the evaluation step can still run.
    else:
        prop = await client.post(
            f"{BACKEND_URL}/v1/projects/{pid}/proposals",
            json={"example_key": "large_synth_paper"},
            timeout=300.0,
        )
        if prop.status_code != 200:
            report.add(
                "large_image_proposal", False, f"proposal HTTP {prop.status_code}", t0
            )
        else:
            p = prop.json()
            # Hosted ``integrate.api.nvidia.com/v1/chat/completions`` only
            # accepts base64 data URLs ("Only base64 data URLs are supported
            # for now."), so the asset-upload path is bypassed
            # on hosted regardless of size. The smoke asserts the
            # large-image proposal SUCCEEDS via base64 inline; failure
            # would mean either (a) the hosted endpoint imposed a payload
            # size cap, or (b) a regression in the inline path.
            if p.get("invocation_status") != "success":
                inv_id = p.get("inference_invocation_id")
                assert report.project_dir is not None
                rec = _read_op_record(report.project_dir, inv_id) if inv_id else None
                transport = (rec or {}).get("image_transport_mode")
                report.discoveries.append(
                    f"large-image (>180KB) proposal failed: "
                    f"status={p.get('invocation_status')} "
                    f"transport={transport} "
                    f"(see /tmp/vlm_backend.log for response detail)"
                )
                report.add(
                    "large_image_proposal",
                    False,
                    f"large-image proposal failed: status={p.get('invocation_status')}",
                    t0,
                )
            else:
                large_inv = p["inference_invocation_id"]
                assert report.project_dir is not None
                rec = _read_op_record(report.project_dir, large_inv)
                transport = (rec or {}).get("image_transport_mode")
                report.add(
                    "large_image_proposal",
                    transport == "base64_inline",
                    f"transport={transport} (hosted forces base64_inline; "
                    f"asset_id refs unsupported on integrate.api.nvidia.com)",
                    t0,
                )

    # ── Evaluation ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{pid}/evaluation_runs",
        json={"icl_mode": "enabled"},
        timeout=15.0,
    )
    if r.status_code != 201:
        report.add(
            "evaluation_run",
            False,
            f"create HTTP {r.status_code}: {r.text[:200]}",
            t0,
        )
    else:
        run_id = r.json().get("run_id") or r.json().get("evaluation_run_id")
        end = time.monotonic() + 360
        last_status = ""
        while time.monotonic() < end:
            await asyncio.sleep(3.0)
            g = await client.get(
                f"{BACKEND_URL}/v1/projects/{pid}/evaluation_runs/{run_id}",
                timeout=10.0,
            )
            if g.status_code != 200:
                continue
            last_status = g.json().get("status", "")
            if last_status in ("completed", "incomplete", "canceled", "failed"):
                break
        report.add(
            "evaluation_run",
            last_status == "completed",
            f"status={last_status} run_id={run_id}",
            t0,
        )

    return report


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Realistic-load live ICL smoke per Teacher"
    )
    parser.add_argument(
        "--teachers",
        type=str,
        default=DEFAULT_TEACHERS,
        help=(
            f"Comma list of hosted Teacher labels (default: {DEFAULT_TEACHERS}). "
            "The DB audits require a local-source backend on this host."
        ),
    )
    args = parser.parse_args(argv)

    requested = [t.strip() for t in args.teachers.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TEACHER_LOOKUP]
    if unknown:
        print(f"unknown teacher labels: {unknown}", file=sys.stderr)
        return 2

    reports: list[TeacherReport] = []
    async with httpx.AsyncClient() as client:
        try:
            await wait_backend(client)
        except RuntimeError as exc:
            print(f"Backend not reachable: {exc}", file=sys.stderr)
            return 1
        for label in requested:
            model_name, cap = TEACHER_LOOKUP[label]
            r = await _smoke_one(client, label, model_name, cap)
            reports.append(r)

    print("\n" + "─" * 72)
    print("Realistic-load ICL smoke — summary")
    print("─" * 72)
    all_ok = True
    for r in reports:
        verdict = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"\n{verdict}  {r.teacher_label}  ({r.teacher_model}, cap={r.image_cap})")
        print(f"      project_id={r.project_id}")
        for s in r.stages:
            mark = "✓" if s.ok else "✗"
            print(f"        {mark} {s.name:<22s} {s.duration_s:6.1f}s   {s.detail}")
        if r.discoveries:
            print("      Discoveries:")
            for d in r.discoveries:
                print(f"        • {d}")
        if not r.passed:
            all_ok = False
    print("\n" + "─" * 72)
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    print("─" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain(sys.argv[1:])))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cold-start live smoke for the Interactive Loop, per hosted Teacher.

Drives the smallest viable cold-start path across the current hosted Teacher
matrix (the commercially permitted seeded hosted Teachers) and
verifies the ICL attachment invariant (``icl_images_attached_count ==
len(icl_example_keys_used)``) holds when ICL transitions from zero examples to
one example.

For each Teacher::

    1. Create a fresh project (Verified=0).
    2. Patch ``teacher_model_config_id`` to the target.
    3. Save the RPS Guidance.
    4. Ingest one rock and one paper image from the canonical RPS smoke data.
    5. First proposal — assert ``invocation_status=success`` and
       ``icl_images_attached_count=0`` (cold start: no ICL).
    6. Save with ``rationale_source="sme_edited"`` and a corrected label
       (forces ``verified_outcome=Edit``).
    7. Second proposal on the other image — assert
       ``len(icl_example_keys_used) == 1``, ``icl_images_attached_count == 1``,
       and ``prompt_hash`` non-null on the persisted OperationRecord.

Because the SECOND proposal must carry the just-saved Edit as ICL, this catches
"we assume ICL >= 1 somewhere" regressions and the load-bearing transition from
cold start to first-edit-in-context. The final OperationRecord assertion reads
``project.db`` directly, so run this harness against a local-source backend on
the same host; it is not a Compose/remote-backend smoke.

Usage::

    uv run python scripts/cold_start_smoke.py
    uv run python scripts/cold_start_smoke.py --teachers step
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from vlm_feedback_loop.model_catalog_constants import (
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
DEFAULT_RPS_ROOT = Path(
    os.environ.get("RPS_TEST_SET_ROOT", "~/rps-test-set")
).expanduser()
RPS_RELATIVE_IMAGES: tuple[tuple[str, Path], ...] = (
    ("rock", Path("rock/testrock01-00.png")),
    ("paper", Path("paper/testpaper01-00.png")),
)

TEACHER_LOOKUP: dict[str, str] = {
    "step": STEP_3_7_FLASH,
    "mistral_medium": MISTRAL_MEDIUM_3_5,
    "nemotron": NEMOTRON_NANO_12B_VL,
    "omni": NEMOTRON_3_NANO_OMNI_REASONING,
}
DEFAULT_TEACHERS = ",".join(TEACHER_LOOKUP)

GUIDANCE_BODY: dict[str, Any] = {
    "description": (
        "Determine if the hand depicted is a rock, paper, or scissors gesture."
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
            "field_name": "category",
            "type": "enum",
            "role": "core",
            "allowed_values": ["rock", "paper", "scissors"],
            "display_order": 0,
        },
    ],
}


@dataclass
class TeacherColdStartReport:
    teacher_label: str
    teacher_model_name: str
    project_id: str | None = None
    project_dir: Path | None = None
    stages: list[StageResult] = field(default_factory=list)

    def add(
        self, name: str, ok: bool, detail: str = "", t0: float | None = None
    ) -> None:
        d = (time.monotonic() - t0) if t0 is not None else 0.0
        self.stages.append(StageResult(name, ok, detail, d))

    @property
    def passed(self) -> bool:
        return all(s.ok for s in self.stages)


def _resolve_rps_images(rps_root: Path) -> list[tuple[str, Path]]:
    """Resolve the two real RPS fixtures without writing outside IMAGE_ROOT."""
    root = rps_root.expanduser().resolve()
    resolved = [
        (label, (root / relative).resolve()) for label, relative in RPS_RELATIVE_IMAGES
    ]
    missing = [str(path) for _, path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cold-start RPS image(s) missing: "
            + ", ".join(missing)
            + "; set --rps-root or RPS_TEST_SET_ROOT"
        )
    return resolved


def _read_operation_record(
    project_dir: Path, inference_invocation_id: str
) -> dict[str, Any] | None:
    """Read an OperationRecord from the API-reported local project directory."""
    db_path = project_dir / "project.db"
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM operation_records WHERE inference_invocation_id = ?",
            (inference_invocation_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


async def _smoke_one(
    client: httpx.AsyncClient,
    teacher_label: str,
    teacher_model: str,
    rps_images: list[tuple[str, Path]],
) -> TeacherColdStartReport:
    report = TeacherColdStartReport(
        teacher_label=teacher_label, teacher_model_name=teacher_model
    )
    print(f"\n=== Cold-start smoke: {teacher_label} ({teacher_model}) ===", flush=True)

    # ─ Stage 1: create project ──────────────────────────────────────────
    t0 = time.monotonic()
    name = f"cold-start-{teacher_label}-{int(time.time())}"
    r = await client.post(
        f"{BACKEND_URL}/v1/projects",
        json={"name": name, "description": "cold-start smoke"},
        timeout=30.0,
    )
    if r.status_code != 201:
        report.add("create_project", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return report
    project_payload = r.json()
    project_id = project_payload["project_id"]
    report.project_id = project_id
    report.project_dir = Path(project_payload["project_dir"])
    report.add(
        "create_project",
        True,
        f"project_id={project_id} project_dir={report.project_dir}",
        t0,
    )

    # ─ Stage 2: switch teacher ──────────────────────────────────────────
    t0 = time.monotonic()
    mc_id = await resolve_teacher_model_config_id(
        client, project_id, teacher_model, base_url=BACKEND_URL
    )
    if not mc_id:
        report.add(
            "switch_teacher", False, f"model_config not found: {teacher_model}", t0
        )
        return report
    r = await client.patch(
        f"{BACKEND_URL}/v1/projects/{project_id}",
        json={"teacher_model_config_id": mc_id, "thinking_default_on": False},
        timeout=10.0,
    )
    if r.status_code not in (200, 204):
        report.add("switch_teacher", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return report
    report.add("switch_teacher", True, f"mc_id={mc_id}", t0)

    # ─ Stage 3: save Guidance ───────────────────────────────────────────
    t0 = time.monotonic()
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{project_id}/guidance",
        json=GUIDANCE_BODY,
        timeout=20.0,
    )
    if r.status_code != 201:
        report.add("save_guidance", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return report
    guidance_id = r.json()["guidance_id"]
    r = await client.patch(
        f"{BACKEND_URL}/v1/projects/{project_id}",
        json={"active_guidance_id": guidance_id},
        timeout=10.0,
    )
    if r.status_code not in (200, 204):
        report.add("save_guidance", False, f"activate HTTP {r.status_code}", t0)
        return report
    report.add("save_guidance", True, f"guidance_id={guidance_id}", t0)

    # ─ Stage 4: ingest 2 canonical RPS images ──────────────────────────
    t0 = time.monotonic()
    items = [
        {
            "example_key": f"cold_{lbl}",
            "storage_ref": str(p),
            "source_metadata": {
                "dataset": "rps-test-set",
                "ground_truth_class": lbl,
            },
        }
        for lbl, p in rps_images
    ]
    r = await client.post(
        f"{BACKEND_URL}/v1/projects/{project_id}/examples:ingest",
        json={"examples": items},
        timeout=30.0,
    )
    # examples:ingest returns 202 Accepted (skeleton rows now, pHash +
    # embeddings via background workers — see routers/examples.py).
    if r.status_code != 202:
        report.add("ingest_images", False, f"HTTP {r.status_code}: {r.text[:200]}", t0)
        return report
    payload = r.json()
    results = payload.get("results", [])
    accepted = sum(1 for x in results if x.get("status") == "created")
    if accepted != 2:
        report.add(
            "ingest_images",
            False,
            f"created={accepted}, expected 2: {json.dumps(results)[:300]}",
            t0,
        )
        return report
    report.add("ingest_images", True, f"created={accepted}", t0)

    # ─ Stage 5: first proposal — must have ICL=0 ────────────────────────
    t0 = time.monotonic()
    nxt = await client.get(
        f"{BACKEND_URL}/v1/projects/{project_id}/review_selector/next", timeout=15.0
    )
    if nxt.status_code != 200:
        report.add("first_proposal", False, f"selector HTTP {nxt.status_code}", t0)
        return report
    first_key = nxt.json().get("example_key")
    if not first_key:
        report.add("first_proposal", False, "selector returned no example", t0)
        return report

    prop = await client.post(
        f"{BACKEND_URL}/v1/projects/{project_id}/proposals",
        json={"example_key": first_key},
        timeout=240.0,
    )
    if prop.status_code != 200:
        report.add(
            "first_proposal", False, f"HTTP {prop.status_code}: {prop.text[:200]}", t0
        )
        return report
    p1 = prop.json()
    if p1.get("invocation_status") != "success":
        report.add(
            "first_proposal",
            False,
            f"status={p1.get('invocation_status')} err={p1.get('error')}",
            t0,
        )
        return report
    icl_count_1 = p1.get("icl_images_attached_count", -1)
    icl_keys_1 = p1.get("icl_example_keys_used") or []
    if icl_count_1 != 0 or len(icl_keys_1) != 0:
        report.add(
            "first_proposal",
            False,
            f"cold start expected ICL=0, got images={icl_count_1} keys={len(icl_keys_1)}",
            t0,
        )
        return report
    inv_1 = p1["inference_invocation_id"]
    report.add(
        "first_proposal",
        True,
        f"key={first_key} ICL=0 inv_id={inv_1[:8]}",
        t0,
    )

    # ─ Stage 6: save first as Edit ──────────────────────────────────────
    t0 = time.monotonic()
    gt_first = "rock" if first_key == "cold_rock" else "paper"
    label_body = {
        "rationale_note": (f"Verified {gt_first} gesture from the RPS smoke dataset."),
        "category": gt_first,
    }
    save = await client.post(
        f"{BACKEND_URL}/v1/projects/{project_id}/labels",
        json={
            "example_key": first_key,
            "inference_invocation_id": inv_1,
            "label_json": label_body,
            "rationale_source": "sme_edited",
        },
        timeout=15.0,
    )
    if save.status_code != 200:
        report.add(
            "save_first", False, f"HTTP {save.status_code}: {save.text[:200]}", t0
        )
        return report
    s = save.json()
    if s.get("verified_outcome") != "Edit":
        report.add(
            "save_first",
            False,
            f"expected Edit, got verified_outcome={s.get('verified_outcome')}",
            t0,
        )
        return report
    report.add("save_first", True, "verified_outcome=Edit", t0)

    # ─ Stage 7: second proposal — must carry 1 ICL example ──────────────
    t0 = time.monotonic()
    nxt = await client.get(
        f"{BACKEND_URL}/v1/projects/{project_id}/review_selector/next", timeout=15.0
    )
    if nxt.status_code != 200:
        report.add("second_proposal", False, f"selector HTTP {nxt.status_code}", t0)
        return report
    second_key = nxt.json().get("example_key")
    if not second_key or second_key == first_key:
        report.add(
            "second_proposal",
            False,
            f"selector returned same/empty key: {second_key}",
            t0,
        )
        return report

    prop = await client.post(
        f"{BACKEND_URL}/v1/projects/{project_id}/proposals",
        json={"example_key": second_key},
        timeout=240.0,
    )
    if prop.status_code != 200:
        report.add(
            "second_proposal", False, f"HTTP {prop.status_code}: {prop.text[:200]}", t0
        )
        return report
    p2 = prop.json()
    if p2.get("invocation_status") != "success":
        report.add(
            "second_proposal",
            False,
            f"status={p2.get('invocation_status')} err={p2.get('error')}",
            t0,
        )
        return report

    icl_count_2 = p2.get("icl_images_attached_count", -1)
    icl_keys_2 = p2.get("icl_example_keys_used") or []
    inv_2 = p2["inference_invocation_id"]

    # Invariant: ``icl_images_attached_count == len(icl_example_keys_used)``
    if icl_count_2 != len(icl_keys_2):
        report.add(
            "second_proposal",
            False,
            f"INVARIANT VIOLATION: images={icl_count_2} keys={len(icl_keys_2)}",
            t0,
        )
        return report

    # Cold start → first Edit → next proposal MUST carry 1 ICL.
    if icl_count_2 != 1:
        report.add(
            "second_proposal",
            False,
            f"expected 1 ICL example, got images={icl_count_2} keys={icl_keys_2}",
            t0,
        )
        return report

    if first_key not in icl_keys_2:
        report.add(
            "second_proposal",
            False,
            f"first Edit ({first_key}) not in ICL keys: {icl_keys_2}",
            t0,
        )
        return report

    # ─ DB-level audit: confirm prompt_hash persisted on second proposal ─
    assert report.project_dir is not None
    rec = _read_operation_record(report.project_dir, inv_2)
    prompt_hash = (rec or {}).get("prompt_hash")
    if not prompt_hash:
        report.add(
            "second_proposal",
            False,
            f"prompt_hash NULL on second OperationRecord (rec={bool(rec)})",
            t0,
        )
        return report

    report.add(
        "second_proposal",
        True,
        f"key={second_key} ICL=1 keys={icl_keys_2} prompt_hash={prompt_hash[:12]}…",
        t0,
    )
    return report


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Cold-start ICL smoke per Teacher")
    parser.add_argument(
        "--teachers",
        type=str,
        default=DEFAULT_TEACHERS,
        help=(
            f"Comma list of hosted Teacher labels (default: {DEFAULT_TEACHERS}). "
            "The DB audit requires a local-source backend on this host."
        ),
    )
    parser.add_argument(
        "--rps-root",
        type=Path,
        default=DEFAULT_RPS_ROOT,
        help=(
            "Canonical RPS dataset root visible beneath the backend IMAGE_ROOT "
            f"(default: {DEFAULT_RPS_ROOT}; env: RPS_TEST_SET_ROOT)."
        ),
    )
    args = parser.parse_args(argv)
    requested = [t.strip() for t in args.teachers.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TEACHER_LOOKUP]
    if unknown:
        print(f"unknown teacher labels: {unknown}", file=sys.stderr)
        return 2
    try:
        rps_images = _resolve_rps_images(args.rps_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    reports: list[TeacherColdStartReport] = []
    async with httpx.AsyncClient() as client:
        try:
            await wait_backend(client, base_url=BACKEND_URL)
        except RuntimeError as exc:
            print(f"Backend not reachable: {exc}", file=sys.stderr)
            return 1
        for label in requested:
            r = await _smoke_one(client, label, TEACHER_LOOKUP[label], rps_images)
            reports.append(r)

    print("\n" + "─" * 72)
    print("Cold-start smoke — summary")
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
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain(sys.argv[1:])))

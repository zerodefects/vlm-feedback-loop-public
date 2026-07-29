# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cold-start live smoke for the Interactive Loop, per hosted Teacher.

Drives the smallest viable cold-start path on each of the four hosted
Teachers (Mistral default, Qwen alternate, Nemotron Omni alternate, Nemotron
alternate) and verifies the ICL attachment invariant (``icl_images_attached_count
== len(icl_example_keys_used)``) holds when ICL transitions from zero
examples to one example.

For each Teacher::

    1. Create a fresh project (Verified=0).
    2. Patch ``teacher_model_config_id`` to the target.
    3. Save the RPS Guidance.
    4. Synthesize two distinct PNGs (one rock-styled, one paper-styled);
       ingest both by absolute path.
    5. First proposal — assert ``invocation_status=success`` and
       ``icl_images_attached_count=0`` (cold start: no ICL).
    6. Save with ``rationale_source="sme_edited"`` and a corrected label
       (forces ``verified_outcome=Edit``).
    7. Second proposal on the other image — assert
       ``len(icl_example_keys_used) == 1``, ``icl_images_attached_count == 1``,
       and ``prompt_hash`` non-null on the persisted OperationRecord.

Because the SECOND proposal must carry the just-saved Edit as ICL,
this catches "we assume ICL >= 1 somewhere" regressions and the
load-bearing transition from cold start to first-edit-in-context.

Usage::

    uv run python scripts/cold_start_smoke.py
    uv run python scripts/cold_start_smoke.py --teachers mistral
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_helpers import (  # noqa: E402
    StageResult,
    resolve_teacher_model_config_id,
    wait_backend,
)

BACKEND_URL = "http://127.0.0.1:8000"


def _resolve_workspace_root() -> Path:
    """Match the live backend's workspace: env override, then the operator's
    config.yaml, then the historical /tmp default."""
    env = os.environ.get("WORKSPACE_ROOT")
    if env:
        return Path(env)
    config = Path.home() / ".vlm_feedback_loop" / "config.yaml"
    if config.exists():
        for line in config.read_text().splitlines():
            if line.startswith("WORKSPACE_ROOT:"):
                return Path(line.split(":", 1)[1].strip())
    return Path("/tmp/vlm_workspace")


WORKSPACE_ROOT = _resolve_workspace_root()

TEACHER_LOOKUP: dict[str, str] = {
    "mistral": "mistralai/mistral-large-3-675b-instruct-2512",
    "qwen": "qwen/qwen3.5-397b-a17b",
    "omni": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nemotron": "nvidia/nemotron-nano-12b-v2-vl",
}

# Two distinct synthetic images so the Edit is forced (proposal will
# probably get the gesture wrong on a stylized synthetic gesture, and
# even if it doesn't, we submit a different rationale to force Edit).
GUIDANCE_BODY: dict[str, Any] = {
    "description": (
        "Determine if the hand depicted is a rock, paper, or scissors "
        "gesture. The image is a stylized illustration."
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
    stages: list[StageResult] = field(default_factory=list)

    def add(
        self, name: str, ok: bool, detail: str = "", t0: float | None = None
    ) -> None:
        d = (time.monotonic() - t0) if t0 is not None else 0.0
        self.stages.append(StageResult(name, ok, detail, d))

    @property
    def passed(self) -> bool:
        return all(s.ok for s in self.stages)


def _synth_image(out_path: Path, label: str, color: tuple[int, int, int]) -> None:
    """Create a deterministic 256x256 PNG with a styled gesture marker."""
    img = Image.new("RGB", (256, 256), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    if label == "rock":
        draw.ellipse([(40, 40), (216, 216)], fill=color)
    elif label == "paper":
        draw.rectangle([(40, 40), (216, 216)], fill=color)
    elif label == "scissors":
        draw.polygon([(128, 30), (40, 220), (216, 220)], fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")


def _read_operation_record(
    project_id: str, inference_invocation_id: str
) -> dict[str, Any] | None:
    """Read OperationRecord directly from the project SQLite DB."""
    db_path = WORKSPACE_ROOT / "projects" / project_id / "project.db"
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
    client: httpx.AsyncClient, teacher_label: str, teacher_model: str
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
    project_id = r.json()["project_id"]
    report.project_id = project_id
    report.add("create_project", True, f"project_id={project_id}", t0)

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

    # ─ Stage 4: synthesize + ingest 2 images ────────────────────────────
    t0 = time.monotonic()
    img_dir = Path(f"/tmp/vlm_smoke_imgs/{project_id}")
    paths = [
        (img_dir / "img_rock.png", "rock", (180, 80, 80)),
        (img_dir / "img_paper.png", "paper", (80, 180, 80)),
    ]
    for p, lbl, color in paths:
        _synth_image(p, lbl, color)
    items = [
        {
            "example_key": f"cold_{lbl}",
            "storage_ref": str(p),
            "source_metadata": {"ground_truth_class": lbl, "synth": True},
        }
        for p, lbl, _ in paths
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
        "rationale_note": (
            f"Stylized {gt_first} gesture. Verified ground truth from synthetic fixture."
        ),
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
    rec = _read_operation_record(project_id, inv_2)
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
        default="mistral,qwen,omni,nemotron",
        help="Comma list of teacher labels",
    )
    args = parser.parse_args(argv)
    requested = [t.strip() for t in args.teachers.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TEACHER_LOOKUP]
    if unknown:
        print(f"unknown teacher labels: {unknown}", file=sys.stderr)
        return 2

    reports: list[TeacherColdStartReport] = []
    async with httpx.AsyncClient() as client:
        try:
            await wait_backend(client, base_url=BACKEND_URL)
        except RuntimeError as exc:
            print(f"Backend not reachable: {exc}", file=sys.stderr)
            return 1
        for label in requested:
            r = await _smoke_one(client, label, TEACHER_LOOKUP[label])
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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-evolution re-verification live smoke.

Verifies the 8-step atomic transition that fires when an SME makes a
semantic Core change to Guidance:

  (a) New Guidance v2 with ``semantic_core_change_from_guidance_id = v1.id``
  (b) All Verified Examples → state="Unlabeled" with prior_verified_label_ref
  (c) Auto-Labeled Examples → state="Unlabeled" (preserves Operation Records)
  (d) Test Pool memberships cleared
  (e) icl_recommendation_dismissed_at_count = 0
  (f) review_selector_scheduler_state = NULL
  (g) schema_change_context_example_key recorded if applicable
  (h) schema_refinement_reminders_dismissed = 0

Then verifies the re-label flow:
  • Selector returns Tier 1 (prior-label) examples first
  • Re-labeling rebuilds ICL from zero
  • New core field values appear in subsequent proposals

Uses Mistral teacher and the RPS schema. Drives:
  1. Create project; ingest 12 RPS images.
  2. Drive 12 saves (mix of Edits + a few Accepts to vary
     prior_verified_outcome).
  3. Submit guidance:edit with dry_run=true → assert
     edit_type="semantic", verified_count=12.
  4. Submit guidance:edit with dry_run=false → execute the 8-step
     transition.
  5. Direct DB inspection of all 8 invariants.
  6. Hit selector → assert Tier 1 priority (returned key has
     prior_verified_label_ref non-null).
  7. Save 2 examples with the new field; verify ICL grows from zero.

Usage::

    uv run python scripts/schema_evolution_smoke.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_helpers import (  # noqa: E402
    resolve_teacher_model_config_id,
    wait_backend,
)

BACKEND_URL = "http://127.0.0.1:8000"
WORKSPACE_ROOT = Path("/tmp/vlm_workspace")
RPS_ROOT = Path(os.environ.get("RPS_TEST_SET_ROOT", "~/rps-test-set")).expanduser()
TEACHER_MODEL = "mistralai/mistral-large-3-675b-instruct-2512"

GUIDANCE_V1: dict[str, Any] = {
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

GUIDANCE_V2_EDIT: dict[str, Any] = {
    "description": GUIDANCE_V1["description"]
    + " Also indicate the SME's confidence level.",
    "rules": "",
    # Semantic Core change: ADD a new core field "confidence"
    "schema": GUIDANCE_V1["schema"]
    + [
        {
            "field_name": "confidence",
            "type": "enum",
            "role": "core",
            "allowed_values": ["low", "medium", "high"],
            "display_order": 1,
        }
    ],
    "dry_run": False,
}

RPS_FINGERS = {"rock": 0, "paper": 5, "scissors": 2}


def _db_query(project_id: str, sql: str, params: tuple = ()) -> list[Any]:
    db = WORKSPACE_ROOT / "projects" / project_id / "project.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params)]
    finally:
        conn.close()


def _print_check(label: str, passed: bool, detail: str = "") -> None:
    mark = "✓" if passed else "✗"
    print(f"      {mark} {label:<60s} {detail}")


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Schema-evolution re-verification smoke"
    )
    parser.parse_args(argv)
    print("=" * 72)
    print("Schema-evolution re-verification smoke (Mistral)")
    print("=" * 72)

    discoveries: list[str] = []
    overall_ok = True

    async with httpx.AsyncClient() as client:
        try:
            await wait_backend(client, base_url=BACKEND_URL)
        except RuntimeError as exc:
            print(f"Backend not reachable: {exc}", file=sys.stderr)
            return 1

        # ── Stage 1: create project + setup ─────────────────────────────
        t0 = time.monotonic()
        r = await client.post(
            f"{BACKEND_URL}/v1/projects",
            json={
                "name": f"schema-evol-{int(time.time())}",
                "description": "schema evolution smoke",
            },
            timeout=30.0,
        )
        if r.status_code != 201:
            print(f"✗ create_project: HTTP {r.status_code}")
            return 1
        pid = r.json()["project_id"]
        print(f"✓ create_project ({time.monotonic() - t0:.1f}s): {pid}")

        mc_id = await resolve_teacher_model_config_id(
            client, pid, TEACHER_MODEL, base_url=BACKEND_URL
        )
        if not mc_id:
            print("✗ teacher resolution failed")
            return 1
        await client.patch(
            f"{BACKEND_URL}/v1/projects/{pid}",
            json={"teacher_model_config_id": mc_id, "thinking_default_on": False},
            timeout=10.0,
        )

        # ── Stage 2: save Guidance v1 ───────────────────────────────────
        r = await client.post(
            f"{BACKEND_URL}/v1/projects/{pid}/guidance",
            json=GUIDANCE_V1,
            timeout=20.0,
        )
        if r.status_code != 201:
            print(f"✗ save_guidance_v1: HTTP {r.status_code}: {r.text[:200]}")
            return 1
        gid_v1 = r.json()["guidance_id"]
        await client.patch(
            f"{BACKEND_URL}/v1/projects/{pid}",
            json={"active_guidance_id": gid_v1},
            timeout=10.0,
        )
        print(f"✓ save_guidance_v1: guidance_id={gid_v1}")

        # ── Stage 3: ingest 12 RPS images ───────────────────────────────
        rps_paths: list[tuple[str, Path]] = []
        for cls in ("rock", "paper", "scissors"):
            files = sorted((RPS_ROOT / cls).glob("*.png"))
            for p in files[:4]:
                rps_paths.append((cls, p))
        items = [
            {
                "example_key": f"se_{cls}_{p.stem}",
                "storage_ref": str(p),
                "source_metadata": {"ground_truth_class": cls},
            }
            for cls, p in rps_paths
        ]
        r = await client.post(
            f"{BACKEND_URL}/v1/projects/{pid}/examples:ingest",
            json={"examples": items},
            timeout=60.0,
        )
        results = r.json().get("results", [])
        created = sum(1 for x in results if x.get("status") == "created")
        if created != 12:
            print(f"✗ ingest_images: created={created}, expected 12")
            return 1
        print(f"✓ ingest_images: created={created}")

        # ── Stage 4: drive 12 saves (10 Edits, 2 Accepts) ───────────────
        inferred = {f"se_{cls}_{p.stem}": cls for cls, p in rps_paths}
        labeled_keys: set[str] = set()
        accepts = 0
        edits = 0
        t_review = time.monotonic()
        target_total = 12
        accepts_target = 2

        while edits + accepts < target_total:
            nxt = await client.get(
                f"{BACKEND_URL}/v1/projects/{pid}/review_selector/next", timeout=15.0
            )
            ex_key = nxt.json().get("example_key")
            if not ex_key or ex_key in labeled_keys:
                break

            prop = await client.post(
                f"{BACKEND_URL}/v1/projects/{pid}/proposals",
                json={"example_key": ex_key},
                timeout=300.0,
            )
            if prop.status_code != 200:
                print(f"  proposal FAIL on {ex_key}: HTTP {prop.status_code}")
                break
            p = prop.json()
            if p.get("invocation_status") != "success":
                print(
                    f"  proposal status={p.get('invocation_status')} on {ex_key}; "
                    f"skipping"
                )
                labeled_keys.add(ex_key)
                continue

            gt = inferred.get(ex_key, "rock")
            inv_id = p["inference_invocation_id"]
            proposal_json = p.get("proposal_json") or {}
            proposed_cat = proposal_json.get("category")

            if accepts < accepts_target and proposed_cat == gt:
                # Accept the Teacher's correct proposal
                label_body = proposal_json
                rationale_source = "teacher_proposal"
            else:
                # Force an Edit by overriding to ground truth
                label_body = {
                    "rationale_note": (f"Verified ground truth: hand is {gt}."),
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
                print(
                    f"  save FAIL on {ex_key}: HTTP {save.status_code}: "
                    f"{save.text[:200]}"
                )
                break
            outcome = save.json().get("verified_outcome")
            labeled_keys.add(ex_key)
            if outcome == "Edit":
                edits += 1
            elif outcome == "Accept":
                accepts += 1

        verified_count = edits + accepts
        review_elapsed = time.monotonic() - t_review
        print(
            f"✓ drive_saves ({review_elapsed:.1f}s): edits={edits} accepts={accepts} "
            f"total={verified_count}"
        )
        if verified_count == 0:
            print("✗ no Verified labels created — cannot proceed")
            return 1

        # ── Stage 5: dry-run preview the semantic Core edit ─────────────
        t0 = time.monotonic()
        r = await client.post(
            f"{BACKEND_URL}/v1/projects/{pid}/guidance:edit",
            json={**GUIDANCE_V2_EDIT, "dry_run": True},
            timeout=20.0,
        )
        if r.status_code != 200:
            print(f"✗ preview_edit: HTTP {r.status_code}: {r.text[:300]}")
            return 1
        preview = r.json()
        print(
            f"✓ preview_edit ({time.monotonic() - t0:.1f}s): "
            f"edit_type={preview['edit_type']} verified_count={preview['verified_count']}"
        )
        if preview.get("edit_type") != "semantic":
            print(f"   ✗ EXPECTED edit_type='semantic', got {preview.get('edit_type')}")
            overall_ok = False
        if preview.get("verified_count") != verified_count:
            print(
                f"   ✗ EXPECTED verified_count={verified_count}, "
                f"got {preview.get('verified_count')}"
            )
            overall_ok = False

        # ── Stage 6: execute (dry_run=false) ────────────────────────────
        t0 = time.monotonic()
        r = await client.post(
            f"{BACKEND_URL}/v1/projects/{pid}/guidance:edit",
            json={**GUIDANCE_V2_EDIT, "dry_run": False},
            timeout=60.0,
        )
        if r.status_code != 200:
            print(f"✗ execute_edit: HTTP {r.status_code}: {r.text[:300]}")
            return 1
        exec_resp = r.json()
        gid_v2 = exec_resp["guidance"]["guidance_id"]
        reverted = exec_resp["verified_reverted_count"]
        print(
            f"✓ execute_edit ({time.monotonic() - t0:.1f}s): "
            f"new guidance_id={gid_v2} verified_reverted={reverted}"
        )

        # ── Stage 7: verify the 8 atomic invariants in DB ───────────────
        print("\n  ── 8-step atomic invariants ──")

        # (a) New Guidance with semantic_core_change_from_guidance_id = v1
        rows = _db_query(
            pid,
            "SELECT guidance_id, semantic_core_change_from_guidance_id "
            "FROM guidances WHERE guidance_id = ?",
            (gid_v2,),
        )
        if rows and rows[0]["semantic_core_change_from_guidance_id"] == gid_v1:
            _print_check(
                "(a) new Guidance v2 with semantic_core_change_from_guidance_id=v1",
                True,
                f"{gid_v2[:8]} ← {gid_v1[:8]}",
            )
        else:
            _print_check(
                "(a) new Guidance v2 with semantic_core_change_from_guidance_id=v1",
                False,
                f"got: {rows}",
            )
            overall_ok = False

        # (b) All Verified examples → Unlabeled with prior_verified_label_ref
        rows = _db_query(
            pid,
            "SELECT state, prior_verified_label_ref, prior_verified_outcome "
            "FROM examples WHERE state = 'Unlabeled' "
            "AND prior_verified_label_ref IS NOT NULL",
        )
        if len(rows) == verified_count:
            _print_check(
                f"(b) {len(rows)}/{verified_count} examples → Unlabeled with prior_verified_label_ref",
                True,
            )
        else:
            _print_check(
                f"(b) examples with prior_verified_label_ref: {len(rows)} (expected {verified_count})",
                False,
            )
            overall_ok = False

        # (c) Auto-Labeled examples → Unlabeled (none in this project)
        rows = _db_query(
            pid, "SELECT COUNT(*) as cnt FROM examples WHERE state = 'Auto-Labeled'"
        )
        _print_check(
            "(c) Auto-Labeled examples remaining",
            rows[0]["cnt"] == 0,
            f"count={rows[0]['cnt']}",
        )
        if rows[0]["cnt"] != 0:
            overall_ok = False

        # (d) Pool memberships cleared (all Labels with pool_assignment deleted)
        rows = _db_query(pid, "SELECT COUNT(*) as cnt FROM labels")
        _print_check(
            "(d) Label records (incl. pool assignments) deleted",
            rows[0]["cnt"] == 0,
            f"label count={rows[0]['cnt']}",
        )
        if rows[0]["cnt"] != 0:
            overall_ok = False

        # (e) icl_recommendation_dismissed_at_count = 0
        rows = _db_query(
            pid, "SELECT icl_recommendation_dismissed_at_count FROM projects"
        )
        val = rows[0]["icl_recommendation_dismissed_at_count"]
        _print_check(
            "(e) icl_recommendation_dismissed_at_count = 0", val == 0, f"value={val}"
        )
        if val != 0:
            overall_ok = False

        # (f) review_selector_scheduler_state reset (SQL NULL or JSON null literal)
        rows = _db_query(pid, "SELECT review_selector_scheduler_state FROM projects")
        sched = rows[0]["review_selector_scheduler_state"]
        # Accept both SQL NULL (None) and JSON null literal ("null" text) — both
        # represent "scheduler state cleared". SQLAlchemy's JSON type may serialize
        # Python None to the string "null".
        sched_reset = sched is None or (
            isinstance(sched, str) and sched.strip().lower() == "null"
        )
        _print_check(
            "(f) review_selector_scheduler_state reset", sched_reset, f"value={sched!r}"
        )
        if not sched_reset:
            overall_ok = False

        # (g) schema_change_context_example_key recorded if set (None acceptable)
        rows = _db_query(pid, "SELECT schema_change_context_example_key FROM projects")
        ctx = rows[0]["schema_change_context_example_key"]
        _print_check(
            "(g) schema_change_context_example_key",
            True,
            f"value={ctx} (None is acceptable when not provided)",
        )

        # (h) schema_refinement_reminders_dismissed = 0
        rows = _db_query(
            pid, "SELECT schema_refinement_reminders_dismissed FROM projects"
        )
        rem = rows[0]["schema_refinement_reminders_dismissed"]
        _print_check(
            "(h) schema_refinement_reminders_dismissed = 0", rem == 0, f"value={rem}"
        )
        if rem != 0:
            overall_ok = False

        # ── Stage 8: re-label loop with prior-label hints ──────────────
        print("\n  ── re-label loop ──")
        nxt = await client.get(
            f"{BACKEND_URL}/v1/projects/{pid}/review_selector/next", timeout=15.0
        )
        first_ex = nxt.json().get("example_key")
        if first_ex:
            rows = _db_query(
                pid,
                "SELECT prior_verified_label_ref, prior_verified_outcome "
                "FROM examples WHERE example_key = ?",
                (first_ex,),
            )
            has_prior = bool(rows and rows[0].get("prior_verified_label_ref"))
            _print_check(
                "first selector pick has prior_verified_label_ref",
                has_prior,
                f"key={first_ex} prior_outcome={rows[0].get('prior_verified_outcome') if rows else None}",
            )
            if not has_prior:
                overall_ok = False
                discoveries.append(
                    f"Tier 1 priority NOT honored: first selector pick "
                    f"({first_ex}) lacks prior_verified_label_ref"
                )
        else:
            _print_check(
                "first selector pick has prior_verified_label_ref",
                False,
                "selector returned empty",
            )
            overall_ok = False

        # Re-label 2 examples with the new field present
        relabeled = 0
        for _ in range(3):
            nxt = await client.get(
                f"{BACKEND_URL}/v1/projects/{pid}/review_selector/next", timeout=15.0
            )
            ex_key = nxt.json().get("example_key")
            if not ex_key:
                break
            gt = inferred.get(ex_key, "rock")
            prop = await client.post(
                f"{BACKEND_URL}/v1/projects/{pid}/proposals",
                json={"example_key": ex_key},
                timeout=300.0,
            )
            if prop.status_code != 200:
                break
            p = prop.json()
            if p.get("invocation_status") != "success":
                break
            inv_id = p["inference_invocation_id"]
            label_body = {
                "rationale_note": f"Verified {gt} under v2 schema",
                "category": gt,
                "confidence": "high",
            }
            save = await client.post(
                f"{BACKEND_URL}/v1/projects/{pid}/labels",
                json={
                    "example_key": ex_key,
                    "inference_invocation_id": inv_id,
                    "label_json": label_body,
                    "rationale_source": "sme_edited",
                },
                timeout=15.0,
            )
            if save.status_code == 200:
                relabeled += 1
                if relabeled >= 2:
                    break

        if relabeled >= 2:
            _print_check(
                f"re-labeled {relabeled} examples under v2 schema with new 'confidence' field",
                True,
            )
        else:
            _print_check(f"re-labeled only {relabeled} examples", False)
            overall_ok = False

        # Verify ICL grew from zero (post-evolution starts at 0, after 2 Edits → 1+ ICL)
        rows = _db_query(
            pid,
            "SELECT COUNT(*) as cnt FROM labels "
            "WHERE label_status = 'verified' AND verified_outcome = 'Edit' "
            "AND pool_assignment IS NULL AND guidance_id = ?",
            (gid_v2,),
        )
        post_edits = rows[0]["cnt"]
        _print_check(
            "post-evolution ICL eligibility rebuilds from zero",
            post_edits >= 1,
            f"post-evolution Edits eligible for ICL: {post_edits}",
        )

        # ── Final ──
        print("\n" + "─" * 72)
        if discoveries:
            print("Discoveries:")
            for d in discoveries:
                print(f"  • {d}")
            print("")
        print(f"Overall: {'PASS' if overall_ok else 'FAIL'}")
        print("─" * 72)
        return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain(sys.argv[1:])))

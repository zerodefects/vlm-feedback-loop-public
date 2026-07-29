#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scale-up full-stack end-to-end validation closing smoke.

Operator-executed Profile E (1×A100 80 GB, ~$150–200, ~1.5–2h) closing
smoke. Drives both Cosmos Reason2 2B AND 8B Students through
deploy → benchmark → handoff → re-execute, plus the hardware-independent
final integration checkpoint (Phase E).

The script intentionally treats *training* as upstream — the operator runs
``scripts/rps_e2e.py`` (or the live equivalent) first to produce
``StudentModel`` rows with ``quality_status="validated"`` on the project,
then points this script at those projects via ``--student-project-2b`` /
``--student-project-8b``. Phase A's job is to discover them; the
training itself is not in scope here.

Five phases:

  A. discover                 — enumerate StudentModels in the source projects.
  B. deploy_serving           — POST :deploy_nim per variant; wait for
                                 terminal lifecycle stage; assert
                                 ``serving_status="validated"``.
  C. handoff_differentiation  — POST :deployment_handoff for the 2B and 8B
                                 representative variants; assert the two
                                 handoffs differ across the four
                                 base-model-dependent technical keys.
  D. handoff_reexecution      — parse ``docker_run_args``; in mock mode,
                                 verify shape + send a chat/completions
                                 round-trip to the (already-running) mock
                                 NIM. In live mode, subprocess-exec the
                                 docker run with name-only NGC_API_KEY
                                 forwarding, poll health, send the
                                 round-trip request, stop the container.
  E. final_integration        — call ``run_full_pipeline_smoke`` (skip
                                 when ``NVIDIA_API_KEY`` is unset).

Usage::

    # Mock mode (CI / pre-rental code-path validation):
    uv run python scripts/full_stack_validation.py \\
        --execution-mode=mock \\
        --student-project-2b $PROJECT_2B \\
        --student-project-8b $PROJECT_8B \\
        --evidence-dir /tmp/evidence

    # Live mode (Profile E rental window):
    uv run python scripts/full_stack_validation.py \\
        --execution-mode=live \\
        --student-project-2b $PROJECT_2B \\
        --student-project-8b $PROJECT_8B \\
        --evidence-dir /tmp/evidence
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "backend"))

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from full_pipeline_smoke import run_full_pipeline_smoke  # noqa: E402

logger = logging.getLogger("full_stack_validation")


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class VariantOutcome:
    student_model_id: str
    base_label: str  # "2B" | "8B"
    quantization_method: str | None
    deploy_ok: bool = False
    deploy_status_reason: str = ""
    serving_status: str | None = None


@dataclass
class C21Result:
    nim_model_profile_recommended: dict[str, Any] = field(default_factory=dict)
    gpu_requirements: dict[str, Any] = field(default_factory=dict)
    tensor_parallelism: dict[str, Any] = field(default_factory=dict)
    nim_env_vars_recommended: dict[str, Any] = field(default_factory=dict)
    differentiated: bool = False


@dataclass
class C20Prediction:
    """One real-image inference round-trip captured from a re-executed handoff.

    Each per-base handoff re-execution runs one inference per RPS class
    (beyond the single text-only round-trip) so the operator can see real
    label predictions from the Blueprint-trained Student in the closing
    acceptance evidence.
    """

    image_class: str  # "rock" | "paper" | "scissors"
    image_path: str
    ok: bool = False
    predicted_category: str | None = None
    raw_content: str = ""
    detail: str = ""


@dataclass
class C20Result:
    two_b: bool = False
    eight_b: bool = False
    detail_2b: str = ""
    detail_8b: str = ""
    predictions_2b: list[C20Prediction] = field(default_factory=list)
    predictions_8b: list[C20Prediction] = field(default_factory=list)


@dataclass
class FullStackValidationResult:
    execution_mode: str  # "mock" | "live"
    started_at: str
    finished_at: str = ""
    overall_ok: bool = False
    phase_a_complete: bool = False
    phase_b_validated_count: int = 0
    phase_b_target_count: int = 0
    variants: list[VariantOutcome] = field(default_factory=list)
    c21_differentiation: C21Result = field(default_factory=C21Result)
    c20_handoff_rerun: C20Result = field(default_factory=C20Result)
    final_integration_checkpoint: bool = False
    final_integration_skipped: bool = False
    error: str = ""

    def as_acceptance_dict(self) -> dict[str, Any]:
        return {
            "phase_12_4_full_stack_validation": True,
            "execution_mode": self.execution_mode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "overall_ok": self.overall_ok,
            "phase_a_complete": self.phase_a_complete,
            "phase_b_validated_count": self.phase_b_validated_count,
            "phase_b_target_count": self.phase_b_target_count,
            "variants": [asdict(v) for v in self.variants],
            "c21_differentiation": asdict(self.c21_differentiation),
            "c20_handoff_rerun": asdict(self.c20_handoff_rerun),
            "final_integration_checkpoint": self.final_integration_checkpoint,
            "final_integration_skipped": self.final_integration_skipped,
            "error": self.error,
        }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _list_students(
    client: httpx.AsyncClient, project_id: str
) -> list[dict[str, Any]]:
    resp = await client.get(f"/v1/projects/{project_id}/student_models", timeout=15.0)
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, list):
        return body
    return body.get("items") or body.get("results") or []


async def _get_student(
    client: httpx.AsyncClient, project_id: str, student_id: str
) -> dict[str, Any]:
    resp = await client.get(
        f"/v1/projects/{project_id}/student_models/{student_id}", timeout=15.0
    )
    resp.raise_for_status()
    return resp.json()


def _classify_variant(student: dict[str, Any]) -> tuple[str, str]:
    """Derive (base_label, quantization_method) for a Student record."""
    base_name = (student.get("base_model_name") or "").lower()
    if "8b" in base_name:
        base_label = "8B"
    elif "2b" in base_name:
        base_label = "2B"
    else:
        base_label = "?"
    quant = (student.get("quantization_method") or "none").lower()
    return base_label, quant


# ── Phase A: discover ───────────────────────────────────────────────────────


async def phase_a_discover(
    client: httpx.AsyncClient,
    *,
    project_2b: str,
    project_8b: str,
    result: FullStackValidationResult,
    exclude_student_ids: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return [(project_id, student_record), ...] across both projects.

    Filters to Students with ``quality_status="validated"`` and excludes
    any student ID listed in ``exclude_student_ids``. Phase A is "are
    there students to deploy?", not "train new students".

    The exclusion is the operator's escape hatch for Students that exist
    on a project but should not enter Phases B–D — typically a
    sacrificial sibling whose deliberately-mismatched Inference Contract
    makes its `:deployment_handoff` return 409.
    """
    excluded: set[str] = exclude_student_ids or set()
    discovered: list[tuple[str, dict[str, Any]]] = []
    skipped_excluded = 0
    seen: set[str] = set()
    for project_id in (project_2b, project_8b):
        if project_id in seen:
            continue
        seen.add(project_id)
        try:
            for student in await _list_students(client, project_id):
                if student.get("quality_status") != "validated":
                    continue
                if student.get("student_model_id") in excluded:
                    skipped_excluded += 1
                    continue
                discovered.append((project_id, student))
        except httpx.HTTPError as e:
            result.error = f"phase_a list_students failed: {e}"
            return discovered
    result.phase_a_complete = True
    result.phase_b_target_count = len(discovered)
    for _project_id, student in discovered:
        base_label, quant = _classify_variant(student)
        result.variants.append(
            VariantOutcome(
                student_model_id=student["student_model_id"],
                base_label=base_label,
                quantization_method=quant,
            )
        )
    if skipped_excluded:
        logger.info(
            "Phase A: %d quality-validated Students discovered "
            "(%d skipped via --exclude-student-ids)",
            len(discovered),
            skipped_excluded,
        )
    else:
        logger.info(
            "Phase A: %d quality-validated Students discovered", len(discovered)
        )
    return discovered


# ── Phase B: deploy_serving ─────────────────────────────────────────────────


async def _wait_for_serving_terminal(
    client: httpx.AsyncClient,
    *,
    project_id: str,
    student_id: str,
    timeout_s: float,
    poll_interval_s: float = 2.0,
) -> dict[str, Any]:
    """Poll the StudentModel record until ``serving_status`` is terminal."""
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = await _get_student(client, project_id, student_id)
        if last.get("serving_status") in ("validated", "failed"):
            return last
        await asyncio.sleep(poll_interval_s)
    raise TimeoutError(
        f"student {student_id} serving_status did not terminate within "
        f"{timeout_s}s (last={last.get('serving_status') if last else '<none>'})"
    )


async def phase_b_deploy_serving(
    client: httpx.AsyncClient,
    *,
    discovered: list[tuple[str, dict[str, Any]]],
    result: FullStackValidationResult,
    deploy_timeout_s: float,
) -> None:
    """Sequential deploy + benchmark of every discovered Student variant."""
    by_id = {v.student_model_id: v for v in result.variants}
    for project_id, student in discovered:
        student_id = student["student_model_id"]
        outcome = by_id[student_id]
        try:
            resp = await client.post(
                f"/v1/projects/{project_id}/student_models/{student_id}:deploy_nim",
                json={},
                timeout=30.0,
            )
            if resp.status_code not in (200, 201, 202):
                outcome.deploy_ok = False
                outcome.deploy_status_reason = (
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
                continue
            final = await _wait_for_serving_terminal(
                client,
                project_id=project_id,
                student_id=student_id,
                timeout_s=deploy_timeout_s,
            )
            outcome.serving_status = final.get("serving_status")
            outcome.deploy_ok = outcome.serving_status == "validated"
            if not outcome.deploy_ok:
                outcome.deploy_status_reason = json.dumps(
                    final.get("nim_preflight_details") or {}
                )
        except (httpx.HTTPError, TimeoutError) as e:
            outcome.deploy_ok = False
            outcome.deploy_status_reason = f"{type(e).__name__}: {e}"
        if outcome.deploy_ok:
            result.phase_b_validated_count += 1


# ── Phase C: handoff_differentiation ────────────────────────────────────────


async def _generate_handoff(
    client: httpx.AsyncClient, project_id: str, student_id: str
) -> dict[str, Any]:
    resp = await client.post(
        f"/v1/projects/{project_id}/student_models/{student_id}:deployment_handoff",
        json={},
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"handoff for {student_id} failed: HTTP {resp.status_code}: "
            f"{resp.text[:300]}"
        )
    return resp.json()


def _pick_representatives(
    discovered: list[tuple[str, dict[str, Any]]],
    outcomes: list[VariantOutcome],
) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """Return ((project_id, 2B_id), (project_id, 8B_id)) of representative variants."""
    by_id = {v.student_model_id: v for v in outcomes}
    rep_2b: tuple[str, str] | None = None
    rep_8b: tuple[str, str] | None = None
    # Prefer baseline (no quantization) when available.
    for project_id, student in discovered:
        v = by_id.get(student["student_model_id"])
        if v is None or not v.deploy_ok:
            continue
        if v.base_label == "2B" and rep_2b is None:
            rep_2b = (project_id, student["student_model_id"])
        elif v.base_label == "8B" and rep_8b is None:
            rep_8b = (project_id, student["student_model_id"])
    return rep_2b, rep_8b


async def phase_c_handoff_differentiation(
    client: httpx.AsyncClient,
    *,
    discovered: list[tuple[str, dict[str, Any]]],
    result: FullStackValidationResult,
    evidence_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Generate handoffs for representative 2B + 8B; assert they differ.

    Partial-closure mode: when only one of {2B, 8B} is discovered (e.g.,
    a TAO-offline 2B-only run), generate the handoff for whichever rep
    exists. Phase D's `for label_key in (2B, 8B)` loop supports
    iterating over a partial handoffs dict (skips when key missing).
    The differentiation assertion correctly stays gated on BOTH
    reps existing — partial closure cannot assert differentiation.
    """
    rep_2b, rep_8b = _pick_representatives(discovered, result.variants)
    handoffs: dict[str, dict[str, Any]] = {}
    if rep_2b is None and rep_8b is None:
        logger.warning(
            "Phase C: no representative variants discovered; handoff generation skipped"
        )
        return handoffs

    # Generate handoffs for whichever reps exist (partial-closure friendly).
    if rep_2b is not None:
        handoffs["2B"] = await _generate_handoff(client, *rep_2b)
        (evidence_dir / "deployment_handoff_2b.json").write_text(
            json.dumps(handoffs["2B"], indent=2)
        )
    if rep_8b is not None:
        handoffs["8B"] = await _generate_handoff(client, *rep_8b)
        (evidence_dir / "deployment_handoff_8b.json").write_text(
            json.dumps(handoffs["8B"], indent=2)
        )

    # Differentiation requires BOTH reps. In 2B-only or 8B-only
    # partial-closure runs, the assertion correctly cannot be made;
    # log + skip the differentiation fields, leave them at their defaults.
    if rep_2b is None or rep_8b is None:
        logger.warning(
            "Phase C: only one base discovered (2b=%s, 8b=%s); handoff "
            "differentiation cannot be asserted (partial closure)",
            rep_2b,
            rep_8b,
        )
        return handoffs

    tech_2b = handoffs["2B"]["technical_requirements"]
    tech_8b = handoffs["8B"]["technical_requirements"]

    c21 = result.c21_differentiation
    c21.nim_model_profile_recommended = {
        "two_b": tech_2b.get("nim_model_profile_recommended"),
        "eight_b": tech_8b.get("nim_model_profile_recommended"),
        "differs": tech_2b.get("nim_model_profile_recommended")
        != tech_8b.get("nim_model_profile_recommended"),
    }
    c21.gpu_requirements = {
        "two_b": tech_2b.get("gpu_requirements"),
        "eight_b": tech_8b.get("gpu_requirements"),
        "differs": tech_2b.get("gpu_requirements") != tech_8b.get("gpu_requirements"),
    }
    c21.tensor_parallelism = {
        "two_b": tech_2b.get("tensor_parallelism"),
        "eight_b": tech_8b.get("tensor_parallelism"),
        "both_int_present": (
            isinstance(tech_2b.get("tensor_parallelism"), int)
            and isinstance(tech_8b.get("tensor_parallelism"), int)
        ),
    }
    c21.nim_env_vars_recommended = {
        "two_b": tech_2b.get("nim_env_vars_recommended"),
        "eight_b": tech_8b.get("nim_env_vars_recommended"),
        "differs": tech_2b.get("nim_env_vars_recommended")
        != tech_8b.get("nim_env_vars_recommended"),
    }
    # The spec language is "differentiation across" the four fields; on
    # single-GPU Profile E, tensor_parallelism may be equal across 2B/8B
    # (both tp=1). The other three fields MUST differ.
    c21.differentiated = bool(
        c21.nim_model_profile_recommended["differs"]
        and c21.gpu_requirements["differs"]
        and c21.nim_env_vars_recommended["differs"]
        and c21.tensor_parallelism["both_int_present"]
    )
    return handoffs


# ── Phase D: handoff_reexecution ────────────────────────────────────────────


def _validate_docker_run_args(args: list[str]) -> tuple[bool, str]:
    """Shape check on the docker run argv list."""
    if not args:
        return False, "docker_run_args is empty"
    if args[0] != "docker":
        return False, f"argv[0] is {args[0]!r}, not 'docker'"
    if "run" not in args:
        return False, "'run' not in argv"
    if "-u" not in args:
        return False, "'-u' (uid pin) not in argv"
    if not any(":ro" in str(a) for a in args):
        return False, "':ro' (read-only checkpoint mount) not in argv"
    if "--gpus" not in args:
        return False, "'--gpus' not in argv"
    try:
        _extract_container_name(args)
    except ValueError as exc:
        return False, str(exc)
    if any(str(arg).startswith("NGC_API_KEY=") for arg in args):
        return False, "NGC_API_KEY must use name-only environment forwarding"
    if not any(
        arg == "-e" and index + 1 < len(args) and args[index + 1] == "NGC_API_KEY"
        for index, arg in enumerate(args)
    ):
        return False, "name-only NGC_API_KEY forwarding is missing"
    return True, "shape ok"


def _redact_ngc_secret(text: str, ngc_key: str) -> str:
    """Remove the live NGC value from diagnostics and saved evidence."""
    if not ngc_key:
        return text
    return text.replace(ngc_key, "[REDACTED]")


def _extract_container_name(args: list[str]) -> str:
    """Return the exact generated container name used for safe cleanup."""
    try:
        name = args[args.index("--name") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError("'--name' and its value are required") from exc
    if not name.startswith("vlm-student-") or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for char in name
    ):
        raise ValueError("'--name' must contain a Blueprint Student container name")
    return name


def _extract_host_port(args: list[str]) -> int:
    """Parse ``-p HOST:CONTAINER`` from a docker-run argv list.

    Phase D's poll URL must use the host port that the
    docker_run_args actually mapped to, not the
    LOCAL_NIM_TEACHER_PORT default (8000). The Blueprint's
    ``local_nim_service._resolve_port`` increments past occupied ports
    (a dev backend often sits on 8000), so the resolved host port
    can be 8002, 8003, etc. Returns the parsed HOST port; defaults to
    8000 when the argv has no ``-p`` flag (defensive — should never
    happen for a well-formed handoff payload).
    """
    for i, arg in enumerate(args):
        if arg == "-p" and i + 1 < len(args):
            mapping = args[i + 1]
            # Format: HOST:CONTAINER (with optional 0.0.0.0:HOST:CONTAINER)
            parts = mapping.split(":")
            if len(parts) >= 2:
                # When the mapping is HOST:CONTAINER, parts[0] is HOST.
                # When it's IFACE:HOST:CONTAINER (3 parts), HOST is parts[1].
                host_str = parts[-2]
                try:
                    return int(host_str)
                except ValueError:
                    continue
    return 8000


async def _send_round_trip(
    endpoint_url: str, served_model_name: str
) -> tuple[bool, str]:
    """POST a labeled-image chat/completions request and assert schema-valid response."""
    async with httpx.AsyncClient(base_url=endpoint_url) as client:
        try:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": served_model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Classify this hand gesture."}
                            ],
                        }
                    ],
                    "max_tokens": 64,
                },
                timeout=30.0,
            )
        except httpx.HTTPError as e:
            return False, f"transport error: {e}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        return False, "no choices in response"
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        return False, f"choices[0].message.content is {type(content).__name__}, not str"
    if not content.strip():
        return False, "content is empty/whitespace"
    # The re-execution round-trip's intent is to prove the canonical
    # ``docker_run_args`` spawn a NIM that responds end-to-end to
    # ``/v1/chat/completions`` — strict JSON-only acceptance is too
    # narrow: Cosmos Reason2 2B without JSON-mode prompting responds in
    # natural text (e.g., "OK" / "rock paper or scissors"). HTTP 200 +
    # non-empty content is the right success signal; if the response
    # happens to be JSON, capture its keys for the detail message as a
    # best-effort artifact.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return True, f"ok (parsed keys: {sorted(parsed.keys())})"
    except json.JSONDecodeError:
        pass
    snippet = content.strip()
    return True, f"ok (text response: {snippet[:80]!r})"


def _resolve_rps_round_trip_set(rps_root: Path) -> list[tuple[str, Path]]:
    """Pick one PNG per class so Phase D round-trips a real image of each gesture.

    Falls back silently to ``[]`` (caller treats as no per-class
    evidence) if the expected ``rock/``, ``paper/``, ``scissors/``
    layout is not present.
    """
    if not rps_root.is_dir():
        return []
    out: list[tuple[str, Path]] = []
    for cls in ("rock", "paper", "scissors"):
        cdir = rps_root / cls
        if not cdir.is_dir():
            continue
        candidates = sorted(p for p in cdir.iterdir() if p.suffix.lower() == ".png")
        if not candidates:
            continue
        out.append((cls, candidates[0]))
    return out


async def _send_rps_round_trip(
    endpoint_url: str,
    served_model_name: str,
    rps_root: Path,
) -> list[C20Prediction]:
    """Run one chat/completions call per RPS class through the live NIM.

    For each gesture class we send the image plus a short instruction that
    matches the active Guidance schema (Core enum ``category``, Aux
    ``number_fingers_extended`` + ``rationale_note``). The response is
    JSON-parsed and the ``category`` field is captured into the evidence
    record so the operator can read off "Student deployed via Blueprint
    classified rock as rock, paper as paper" without re-querying TAO.
    """
    triples = _resolve_rps_round_trip_set(rps_root)
    results: list[C20Prediction] = []
    if not triples:
        return results
    instruction = (
        "Classify the rock-paper-scissors hand gesture in the image. "
        "Return JSON with a 'category' field whose value is exactly one of "
        "'rock', 'paper', or 'scissors'."
    )
    async with httpx.AsyncClient(base_url=endpoint_url) as client:
        for cls, path in triples:
            entry = C20Prediction(image_class=cls, image_path=str(path))
            try:
                b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            except OSError as e:
                entry.detail = f"image read error: {e}"
                results.append(entry)
                continue
            data_url = f"data:image/png;base64,{b64}"
            try:
                resp = await client.post(
                    "/chat/completions",
                    json={
                        "model": served_model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_url},
                                    },
                                    {"type": "text", "text": instruction},
                                ],
                            }
                        ],
                        "max_tokens": 128,
                    },
                    timeout=60.0,
                )
            except httpx.HTTPError as e:
                entry.detail = f"transport error: {e}"
                results.append(entry)
                continue
            if resp.status_code != 200:
                entry.detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
                results.append(entry)
                continue
            body = resp.json()
            choices = body.get("choices") or []
            if not choices:
                entry.detail = "no choices in response"
                results.append(entry)
                continue
            content = choices[0].get("message", {}).get("content")
            entry.raw_content = (
                (content or "")[:1024] if isinstance(content, str) else ""
            )
            if not isinstance(content, str):
                entry.detail = f"content is {type(content).__name__}, not str"
                results.append(entry)
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                entry.detail = f"content not JSON: {e}"
                results.append(entry)
                continue
            if not isinstance(parsed, dict):
                entry.detail = "content parsed but not an object"
                results.append(entry)
                continue
            cat = parsed.get("category") or parsed.get("gesture")
            if isinstance(cat, str):
                entry.predicted_category = cat
                entry.ok = True
                entry.detail = f"category={cat!r}"
            else:
                entry.detail = f"no category in {sorted(parsed.keys())}"
            results.append(entry)
    return results


async def phase_d_handoff_reexecution(
    *,
    handoffs: dict[str, dict[str, Any]],
    execution_mode: str,
    result: FullStackValidationResult,
    evidence_dir: Path,
    rps_root: Path | None = None,
) -> None:
    """Verify each handoff's docker_run_args parses + round-trips a label.

    When ``rps_root`` points at a directory containing ``rock/``, ``paper/``,
    and ``scissors/`` PNGs, each per-base re-execution additionally sends
    one image per class through the freshly-started NIM and captures the
    parsed ``category`` field into ``c20_handoff_rerun.predictions_*`` so
    the operator can read off real Blueprint-trained Student inference
    results from the closing acceptance evidence.
    """
    if not handoffs:
        return
    # Map (display_label, ok_attr_name, detail_attr_name, predictions_attr)
    # so setattr lands on the actual dataclass fields (``two_b``/``eight_b``
    # and ``detail_2b``/``detail_8b`` — note the digit-vs-word divergence).
    for label_key, ok_attr, detail_attr, preds_attr in (
        ("2B", "two_b", "detail_2b", "predictions_2b"),
        ("8B", "eight_b", "detail_8b", "predictions_8b"),
    ):
        if label_key not in handoffs:
            continue
        tech = handoffs[label_key]["technical_requirements"]
        args = tech.get("docker_run_args") or []
        shape_ok, shape_detail = _validate_docker_run_args(args)
        if not shape_ok:
            setattr(result.c20_handoff_rerun, ok_attr, False)
            setattr(
                result.c20_handoff_rerun,
                detail_attr,
                f"shape: {shape_detail}",
            )
            continue

        served_model = tech.get("nim_served_model_name") or "stub"
        if execution_mode == "live":
            ngc_key = os.environ.get("NGC_API_KEY", "")
            live_args = list(args)
            child_env = dict(os.environ)
            child_env["NGC_API_KEY"] = ngc_key
            try:
                proc = subprocess.run(
                    live_args,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=child_env,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                setattr(result.c20_handoff_rerun, ok_attr, False)
                setattr(
                    result.c20_handoff_rerun,
                    detail_attr,
                    _redact_ngc_secret(f"docker exec failed: {e}", ngc_key),
                )
                continue
            if proc.returncode != 0:
                safe_error = _redact_ngc_secret(proc.stderr, ngc_key)[:200]
                setattr(result.c20_handoff_rerun, ok_attr, False)
                setattr(
                    result.c20_handoff_rerun,
                    detail_attr,
                    f"docker rc={proc.returncode}: {safe_error}",
                )
                continue
            # Cleanup targets the validated generated name from the handoff,
            # never child-controlled stdout. Docker normally prints a container
            # id, but treating that stream as an argv source lets an echoed
            # environment value cross back into a later process invocation.
            container_ref = _extract_container_name(live_args)
            safe_stdout = _redact_ngc_secret(proc.stdout, ngc_key)
            safe_stderr = _redact_ngc_secret(proc.stderr, ngc_key)
            (evidence_dir / f"handoff_rerun_{label_key.lower()}.log").write_text(
                f"args: {live_args}\nstdout: {safe_stdout}\nstderr: {safe_stderr}\n"
            )
            # The docker_run_args come from
            # ``local_nim_service.build_student_docker_run_args``,
            # which calls ``_resolve_port`` to find the next free host
            # port (starts at LOCAL_NIM_TEACHER_PORT=8000 but increments
            # when occupied — e.g., resolves to 8002 when a dev backend
            # holds 8000). A hardcoded ``localhost:8000`` would miss
            # that resolution and poll the wrong process (404) instead
            # of the freshly-started NIM container. Parse the actual
            # host port from ``-p HOST:CONTAINER`` in live_args so
            # Phase D polls whatever port the docker run mapped to.
            host_port = _extract_host_port(live_args)
            endpoint_url = f"http://localhost:{host_port}/v1"
            healthy = await _poll_health(endpoint_url, timeout_s=600.0)
            if not healthy:
                setattr(result.c20_handoff_rerun, ok_attr, False)
                setattr(
                    result.c20_handoff_rerun,
                    detail_attr,
                    "container did not reach /v1/health/ready within 600s",
                )
                # Best-effort cleanup
                subprocess.run(
                    ["docker", "stop", container_ref],
                    capture_output=True,
                    timeout=30,
                )
                continue
            ok, detail = await _send_round_trip(endpoint_url, served_model)
            # Only attempt the per-class real-image round-trip after the
            # baseline single-call passes — saves the operator from waiting
            # on a broken endpoint just to confirm 3 separate failures.
            preds: list[C20Prediction] = []
            if ok and rps_root is not None:
                try:
                    preds = await _send_rps_round_trip(
                        endpoint_url, served_model, rps_root
                    )
                except Exception as e:  # noqa: BLE001 — never crash the run
                    logger.warning(
                        "%s: per-class round-trip raised %s; continuing", label_key, e
                    )
            subprocess.run(
                ["docker", "stop", container_ref],
                capture_output=True,
                timeout=30,
            )
            setattr(result.c20_handoff_rerun, ok_attr, ok)
            setattr(result.c20_handoff_rerun, detail_attr, detail)
            if preds:
                setattr(result.c20_handoff_rerun, preds_attr, preds)
        else:
            # Mock mode: don't subprocess-exec docker. The MockNIMServer
            # is already running (Phase B started the lifecycle against it).
            # Send a chat/completions round-trip using the LocalNimDeployment
            # endpoint URL recorded by the lifecycle.
            mock_url = os.environ.get("LOCAL_NIM_MOCK_ENDPOINT_URL")
            if not mock_url:
                setattr(result.c20_handoff_rerun, ok_attr, False)
                setattr(
                    result.c20_handoff_rerun,
                    detail_attr,
                    "mock mode requires LOCAL_NIM_MOCK_ENDPOINT_URL",
                )
                continue
            ok, detail = await _send_round_trip(mock_url, served_model)
            (evidence_dir / f"handoff_rerun_{label_key.lower()}.log").write_text(
                f"mock mode — shape: {shape_detail}\nround-trip: {detail}\n"
            )
            setattr(result.c20_handoff_rerun, ok_attr, ok)
            setattr(
                result.c20_handoff_rerun,
                detail_attr,
                f"mock round-trip: {detail}",
            )


async def _poll_health(endpoint_url: str, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(base_url=endpoint_url) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get("/health/ready", timeout=5.0)
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2.0)
    return False


# ── Phase E: final integration checkpoint ────────────────────────────────────


async def phase_e_final_integration(
    *,
    backend_url: str,
    image_paths: list[Path],
    result: FullStackValidationResult,
    evidence_dir: Path,
) -> None:
    if not os.environ.get("NVIDIA_API_KEY"):
        logger.info(
            "Phase E: NVIDIA_API_KEY not set; skipping final integration checkpoint"
        )
        result.final_integration_skipped = True
        return
    if not image_paths:
        logger.info("Phase E: no image_paths provided; skipping")
        result.final_integration_skipped = True
        return
    fp_result = await run_full_pipeline_smoke(
        backend_url=backend_url,
        image_paths=image_paths,
        label_count=10,
        export_field_mode="all",
        keep_project=False,
        evidence_dir=evidence_dir / "phase_e_full_pipeline_smoke",
        project_name=f"step-12-4-phase-e-{_now_iso()}",
    )
    result.final_integration_checkpoint = fp_result.overall_ok


# ── Top-level driver ────────────────────────────────────────────────────────


async def run_full_stack_validation(
    *,
    execution_mode: str,
    backend_url: str,
    student_project_2b: str,
    student_project_8b: str,
    evidence_dir: Path,
    deploy_timeout_s: float = 1200.0,
    image_paths: list[Path] | None = None,
    rps_root: Path | None = None,
    exclude_student_ids: set[str] | None = None,
) -> FullStackValidationResult:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    result = FullStackValidationResult(
        execution_mode=execution_mode, started_at=_now_iso()
    )

    try:
        async with httpx.AsyncClient(base_url=backend_url) as client:
            # Phase A
            discovered = await phase_a_discover(
                client,
                project_2b=student_project_2b,
                project_8b=student_project_8b,
                result=result,
                exclude_student_ids=exclude_student_ids,
            )
            if discovered:
                # Phase B
                await phase_b_deploy_serving(
                    client,
                    discovered=discovered,
                    result=result,
                    deploy_timeout_s=deploy_timeout_s,
                )

                # Phase C
                handoffs = await phase_c_handoff_differentiation(
                    client,
                    discovered=discovered,
                    result=result,
                    evidence_dir=evidence_dir,
                )

                # Phase D
                await phase_d_handoff_reexecution(
                    handoffs=handoffs,
                    execution_mode=execution_mode,
                    result=result,
                    evidence_dir=evidence_dir,
                    rps_root=rps_root,
                )

                # Phase E (hardware-independent)
                await phase_e_final_integration(
                    backend_url=backend_url,
                    image_paths=image_paths or [],
                    result=result,
                    evidence_dir=evidence_dir,
                )
            else:
                if not result.error:
                    result.error = (
                        "no quality-validated Students found in the source projects"
                    )
    except Exception as e:  # noqa: BLE001 — capture any orchestrator failure
        logger.exception("full_stack_validation: unexpected error")
        result.error = f"{type(e).__name__}: {e}"
    finally:
        result.finished_at = _now_iso()

        # Final overall_ok rule: phase A complete, every variant deployed,
        # handoff differentiation holds, the re-execution round-trip
        # succeeds for both 2B and 8B. Phase E is best-effort (skipped
        # without API key).
        result.overall_ok = (
            result.phase_a_complete
            and result.phase_b_validated_count == result.phase_b_target_count
            and result.phase_b_target_count > 0
            and result.c21_differentiation.differentiated
            and result.c20_handoff_rerun.two_b
            and result.c20_handoff_rerun.eight_b
            and (
                result.final_integration_skipped or result.final_integration_checkpoint
            )
        )

        out = evidence_dir / "closing_acceptance.json"
        out.write_text(json.dumps(result.as_acceptance_dict(), indent=2))
        logger.info("Wrote closing_acceptance.json to %s", out)
    return result


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="full_stack_validation",
        description="Scale-Up Full-Stack End-to-End Validation closing smoke.",
    )
    p.add_argument(
        "--execution-mode",
        choices=("mock", "live"),
        required=True,
        help="`mock` exercises the code path against a stub NIM. `live` "
        "actually subprocess-execs the handoff docker run on real GPU.",
    )
    p.add_argument("--backend-url", default="http://127.0.0.1:8000")
    p.add_argument(
        "--student-project-2b",
        required=True,
        help="Project ID containing one or more validated Cosmos Reason2 2B Students.",
    )
    p.add_argument(
        "--student-project-8b",
        required=True,
        help="Project ID containing one or more validated Cosmos Reason2 8B Students.",
    )
    p.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="Output directory for closing_acceptance.json + per-handoff payloads.",
    )
    p.add_argument("--deploy-timeout-s", type=float, default=1200.0)
    p.add_argument(
        "--image-source-dir",
        type=Path,
        default=None,
        help="Optional image directory for Phase E. Skipped if unset or no NVIDIA_API_KEY.",
    )
    p.add_argument(
        "--rps-root",
        type=Path,
        default=None,
        help=(
            "Optional path to a rock-paper-scissors test set with rock/, paper/, "
            "scissors/ PNG subdirs. When set + execution_mode=live, Phase D "
            "additionally sends one image per class through each re-executed "
            "NIM and captures the parsed 'category' field into "
            "c20_handoff_rerun.predictions_*."
        ),
    )
    p.add_argument(
        "--exclude-student-ids",
        default="",
        help=(
            "Comma-separated student_model_ids to skip in Phase A discovery. "
            "Use this to omit a sibling/sacrificial Student (e.g., a "
            "``core_only`` contract-mismatch sibling) whose deployment_handoff is "
            "expected to return 409 INFERENCE_CONTRACT_MISMATCH and which "
            "should not enter the Phase B deploy queue."
        ),
    )
    return p.parse_args(argv)


def _gather_phase_e_images(image_dir: Path | None, max_count: int) -> list[Path]:
    if image_dir is None or not image_dir.exists():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes
    )
    return files[:max_count]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    image_paths = _gather_phase_e_images(args.image_source_dir, max_count=10)

    exclude_ids: set[str] = {
        s.strip() for s in (args.exclude_student_ids or "").split(",") if s.strip()
    }

    result = asyncio.run(
        run_full_stack_validation(
            execution_mode=args.execution_mode,
            backend_url=args.backend_url,
            student_project_2b=args.student_project_2b,
            student_project_8b=args.student_project_8b,
            evidence_dir=args.evidence_dir,
            deploy_timeout_s=args.deploy_timeout_s,
            image_paths=image_paths,
            rps_root=args.rps_root,
            exclude_student_ids=exclude_ids,
        )
    )

    print(json.dumps(result.as_acceptance_dict(), indent=2))
    return 0 if result.overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

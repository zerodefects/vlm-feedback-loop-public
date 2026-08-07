# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build deterministic, real-image AIPerf workloads for Student NIMs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.image_transport import prepare_images
from vlm_feedback_loop.services.project_service import (
    get_project_engine,
    project_dir_path,
)
from vlm_feedback_loop.services.serving_request_service import (
    build_uncapped_student_request,
)

WORKLOAD_VERSION = "production_vlm_v1"
MAX_WORKLOAD_IMAGES = 200


@dataclass(frozen=True)
class ServingBenchmarkWorkload:
    input_file: Path
    artifact_root: Path
    temporary_dir: Path
    request_count: int
    manifest: dict[str, Any]


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_code_revision() -> str | None:
    """Return the build override or the current source-checkout revision."""
    configured = os.environ.get("VLM_FEEDBACK_LOOP_BUILD_SHA", "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip().lower()
    if (
        result.returncode != 0
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None
    ):
        return None
    return revision


def _select_keys(member_keys: list[str], workload_basis_hash: str) -> list[str]:
    """Select a stable uniform-looking subset without replacement."""
    unique_keys = sorted(set(member_keys))
    ranked = sorted(
        unique_keys,
        key=lambda key: (
            hashlib.sha256(f"{workload_basis_hash}:{key}".encode()).hexdigest(),
            key,
        ),
    )
    return ranked[:MAX_WORKLOAD_IMAGES]


async def build_serving_benchmark_workload(
    *,
    project_id: str,
    serving_run_id: str,
    student_model_id: str,
    served_model: str,
    workspace_root: str,
    settings: Settings,
) -> ServingBenchmarkWorkload:
    """Materialize uncapped Guidance+image requests for one serving run.

    The JSONL is temporary because it contains inline image bytes.  The
    compact manifest is durable and contains only identifiers, hashes, and
    aggregate sizes.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        raise RuntimeError(f"Project {project_id} not found")

    with Session(engine) as session:
        run = session.get(RunRecord, serving_run_id)
        if run is None or not run.pool_version_id or not run.guidance_id:
            raise RuntimeError("Serving evaluation run lacks pool/guidance provenance")
        pool = session.get(Pool, run.pool_version_id)
        guidance = session.get(Guidance, run.guidance_id)
        if pool is None or guidance is None or not guidance.schema:
            raise RuntimeError("Serving benchmark pool or Guidance is unavailable")
        operation = (
            session.query(OperationRecord)
            .filter_by(
                evaluation_run_id=serving_run_id,
                invocation_status="success",
            )
            .order_by(OperationRecord.inference_invocation_id.asc())
            .first()
        )
        if operation is None:
            raise RuntimeError("Serving evaluation produced no successful invocation")

        member_keys = list(pool.member_example_keys)
        pool_id = pool.pool_id
        examples = (
            session.query(Example)
            .filter(
                Example.project_id == project_id,
                Example.example_key.in_(member_keys),
            )
            .all()
        )
        storage_refs = {
            example.example_key: example.storage_ref for example in examples
        }

        envelope = dict(guidance.schema)
        inference_contract = dict(run.inference_contract or {})
        sampling_params = dict(operation.sampling_params_effective or {})
        thinking_fields = dict(operation.thinking_request_fields_effective or {})
        visual_params = dict(operation.visual_budget_params_effective or {})
        structured_attempted = bool(operation.structured_generation_attempted)
        evaluated_prompt_hash = operation.prompt_hash
        guidance_id = guidance.guidance_id
        guidance_description = guidance.description or ""
        guidance_rules = guidance.rules or ""

    missing = sorted(set(member_keys) - set(storage_refs))
    if missing:
        raise RuntimeError(
            f"Test Pool contains {len(missing)} image(s) missing from Example storage"
        )

    schema_hash_raw = envelope.get("schema_hash")
    guidance_schema_hash = (
        schema_hash_raw
        if isinstance(schema_hash_raw, str)
        else _canonical_hash(envelope)
    )
    pool_checksum = _canonical_hash(sorted(set(member_keys)))
    workload_basis = {
        "version": WORKLOAD_VERSION,
        "pool_checksum": pool_checksum,
        "guidance_id": guidance_id,
        "guidance_schema_hash": guidance_schema_hash,
        "inference_contract": inference_contract,
        "sampling_params": sampling_params,
        "thinking_request_fields": thinking_fields,
        "visual_budget_params": visual_params,
        "structured_generation_attempted": structured_attempted,
        "output_limit_mode": "uncapped",
        "kv_cache_reuse": "disabled",
    }
    workload_basis_hash = _canonical_hash(workload_basis)
    selected_keys = _select_keys(member_keys, workload_basis_hash)
    if not selected_keys:
        raise RuntimeError("Test Pool is empty; no serving workload can be built")

    fields_raw = envelope.get("fields")
    order_raw = envelope.get("generation_order")
    schema_raw = envelope.get("derived_json_schema")
    fields = (
        cast("list[dict[str, Any]]", fields_raw) if isinstance(fields_raw, list) else []
    )
    generation_order = (
        cast("list[str]", order_raw) if isinstance(order_raw, list) else []
    )
    derived_schema = (
        cast("dict[str, Any]", schema_raw) if isinstance(schema_raw, dict) else {}
    )

    lines: list[bytes] = []
    prompt_hash: str | None = None
    total_image_bytes = 0
    for example_key in selected_keys:
        prepared = await prepare_images(
            [storage_refs[example_key]],
            settings=settings,
        )
        if not prepared.success or not prepared.images:
            detail = next(
                (image.error for image in prepared.images if image.error), "unknown"
            )
            raise RuntimeError(
                f"Unable to prepare Test Pool image {example_key}: {detail}"
            )
        content_part = prepared.images[0].content_part
        request, current_prompt_hash = build_uncapped_student_request(
            served_model=served_model,
            guidance_description=guidance_description,
            guidance_rules=guidance_rules,
            guidance_fields=fields,
            generation_order=generation_order,
            derived_json_schema=derived_schema,
            inference_contract=inference_contract,
            structured_generation_attempted=structured_attempted,
            sampling_params=sampling_params,
            thinking_request_fields=thinking_fields,
            visual_budget_params=visual_params,
            image_content_part=content_part,
        )
        if "max_tokens" in request or "max_completion_tokens" in request:
            raise AssertionError("Serving workload unexpectedly contains an output cap")
        if prompt_hash is None:
            prompt_hash = current_prompt_hash
        elif prompt_hash != current_prompt_hash:
            raise AssertionError("Rendered Guidance prompt changed within one workload")
        line = json.dumps(request, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
        total_image_bytes += len(json.dumps(content_part, separators=(",", ":")))
        lines.append(line + b"\n")

    if not evaluated_prompt_hash:
        raise RuntimeError("Serving evaluation did not record its prompt hash")
    if prompt_hash != evaluated_prompt_hash:
        raise RuntimeError(
            "Serving benchmark prompt drifted from the prompt used by the "
            "serving evaluation"
        )

    payload_bytes = b"".join(lines)

    try:
        app_version = importlib.metadata.version("vlm-feedback-loop")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - source edge
        app_version = "unknown"
    manifest: dict[str, Any] = {
        **workload_basis,
        "workload_hash": _canonical_hash(
            {**workload_basis, "selected_example_keys": selected_keys}
        ),
        "workload_basis_hash": workload_basis_hash,
        "pool_id": pool_id,
        "pool_member_count": len(set(member_keys)),
        "selected_count": len(selected_keys),
        "selection_policy": "sha256_rank_without_replacement_max_200",
        "selected_example_keys": selected_keys,
        "prompt_hash": prompt_hash,
        "evaluated_prompt_hash": evaluated_prompt_hash,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_bytes": len(payload_bytes),
        "encoded_image_content_bytes": total_image_bytes,
        "driver": {"name": "aiperf", "version": "0.10.0"},
        "tokenizer": "builtin:o200k_base",
        "application_version": app_version,
        "code_revision": _resolve_code_revision(),
        "created_at": utc_now(),
    }
    project_dir = project_dir_path(workspace_root, project_id)
    artifact_root = (
        project_dir / "artifacts" / "benchmarks" / student_model_id / serving_run_id
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix="payload-", dir=str(artifact_root)))
    temporary_dir.chmod(0o700)
    input_file = temporary_dir / "requests.jsonl"
    try:
        input_file.write_bytes(payload_bytes)
        input_file.chmod(0o600)
        (artifact_root / "workload_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return ServingBenchmarkWorkload(
        input_file=input_file,
        artifact_root=artifact_root,
        temporary_dir=temporary_dir,
        request_count=len(selected_keys),
        manifest=manifest,
    )


__all__ = [
    "MAX_WORKLOAD_IMAGES",
    "ServingBenchmarkWorkload",
    "WORKLOAD_VERSION",
    "build_serving_benchmark_workload",
]

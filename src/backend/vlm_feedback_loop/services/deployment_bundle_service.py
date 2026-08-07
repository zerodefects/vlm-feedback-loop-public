# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable, credential-free Student NIM deployment bundles.

The production handoff is useful to an infrastructure team, but a checkpoint
path on the Blueprint host is not itself portable.  This service turns the
same gated handoff into a streaming tar archive containing the validated
checkpoint plus the exact runtime contract, launch helpers, checksums,
evaluation snapshot, and TAO lineage needed to deploy it elsewhere.

The NIM runtime image is deliberately not redistributed.  ``run-nim.sh``
pulls the pinned NVIDIA image with an operator-provided ``NGC_API_KEY``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import stat
import tarfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services import deployment_handoff_generator
from vlm_feedback_loop.services.project_service import (
    get_project_engine,
    project_dir_path,
)
from vlm_feedback_loop.services.serving_request_service import (
    build_uncapped_student_request,
)

_CHUNK_SIZE = 1024 * 1024
_NON_DEPLOYMENT_CHECKPOINT_PATHS = frozenset(
    {
        Path("microservices_log.txt"),
        Path("status.json"),
    }
)


@dataclass(frozen=True)
class DeploymentBundleFile:
    """One immutable checkpoint file selected before streaming begins."""

    source: Path
    relative_path: Path
    size: int
    mode: int


@dataclass(frozen=True)
class DeploymentBundlePlan:
    """Validated inputs for one streaming archive response."""

    project_id: str
    student_model_id: str
    archive_filename: str
    archive_root: str
    checkpoint_root: Path
    checkpoint_files: tuple[DeploymentBundleFile, ...]
    handoff: dict[str, Any]
    verification_request: dict[str, Any]
    verification_prompt_hash: str
    evaluated_prompt_hash: str | None
    guidance_id: str
    guidance_schema_hash: str | None


class _HashingReader:
    """File proxy that records SHA-256 while ``tarfile`` copies bytes."""

    def __init__(self, source: BinaryIO) -> None:
        self._source = source
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._source.read(size)
        self._digest.update(data)
        return data

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def prepare_deployment_bundle(
    *,
    project_id: str,
    student_model_id: str,
    settings: Settings,
) -> DeploymentBundlePlan | str:
    """Run production gates and resolve a contained checkpoint tree."""
    handoff = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_model_id,
        settings=settings,
    )
    if isinstance(handoff, str):
        return handoff

    technical_raw: Any = handoff.get("technical_requirements")
    if not isinstance(technical_raw, dict):
        return "conflict: deployment_bundle_metadata_incomplete"
    technical = cast("dict[str, Any]", technical_raw)
    checkpoint_ref = technical.get("checkpoint_reference")
    image = technical.get("nim_container_image")
    if not isinstance(checkpoint_ref, str) or not checkpoint_ref:
        return "conflict: deployment_bundle_checkpoint_unavailable"
    if not isinstance(image, str) or not image or image.startswith("("):
        return "conflict: deployment_bundle_runtime_image_unavailable"

    project_root = project_dir_path(settings.WORKSPACE_ROOT, project_id)
    try:
        resolved_project_root = project_root.resolve(strict=True)
        checkpoint_root = Path(checkpoint_ref).resolve(strict=True)
    except OSError:
        return "conflict: deployment_bundle_checkpoint_unavailable"
    if not checkpoint_root.is_dir() or not checkpoint_root.is_relative_to(
        resolved_project_root
    ):
        return "conflict: deployment_bundle_checkpoint_not_contained"

    files: list[DeploymentBundleFile] = []
    try:
        entries = sorted(checkpoint_root.rglob("*"), key=lambda item: item.as_posix())
        for entry in entries:
            if entry.is_symlink():
                return "conflict: deployment_bundle_checkpoint_contains_symlink"
            entry_stat = entry.stat()
            if stat.S_ISDIR(entry_stat.st_mode):
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                return "conflict: deployment_bundle_checkpoint_contains_special_file"
            resolved_entry = entry.resolve(strict=True)
            if not resolved_entry.is_relative_to(checkpoint_root):
                return "conflict: deployment_bundle_checkpoint_not_contained"
            relative_path = resolved_entry.relative_to(checkpoint_root)
            # TAO emits completion diagnostics beside some quantized
            # checkpoints. They are useful while polling/finalizing a job but
            # are not NIM model inputs and can contain infrastructure context.
            # Keep the portable artifact to deployable checkpoint content.
            if relative_path in _NON_DEPLOYMENT_CHECKPOINT_PATHS:
                continue
            files.append(
                DeploymentBundleFile(
                    source=resolved_entry,
                    relative_path=relative_path,
                    size=entry_stat.st_size,
                    mode=entry_stat.st_mode & 0o777,
                )
            )
    except OSError:
        return "conflict: deployment_bundle_checkpoint_unavailable"
    if not files:
        return "conflict: deployment_bundle_checkpoint_empty"

    verification = _build_verification_request(
        project_id=project_id,
        handoff=handoff,
        settings=settings,
    )
    if isinstance(verification, str):
        return verification

    short_id = student_model_id[:8]
    archive_root = f"vlm-student-{short_id}-nim"
    return DeploymentBundlePlan(
        project_id=project_id,
        student_model_id=student_model_id,
        archive_filename=f"{archive_root}.tar",
        archive_root=archive_root,
        checkpoint_root=checkpoint_root,
        checkpoint_files=tuple(files),
        handoff=handoff,
        verification_request=verification["request"],
        verification_prompt_hash=verification["prompt_hash"],
        evaluated_prompt_hash=verification["evaluated_prompt_hash"],
        guidance_id=verification["guidance_id"],
        guidance_schema_hash=verification["guidance_schema_hash"],
    )


def _build_verification_request(
    *,
    project_id: str,
    handoff: dict[str, Any],
    settings: Settings,
) -> dict[str, Any] | str:
    """Reconstruct the schema-aware request proven by serving evaluation.

    A generic vision prompt only proves that a NIM answers HTTP. It does not
    prove that the deployed Student still speaks the project's trained label
    contract. The portable verifier therefore uses the Guidance version and
    effective controls recorded by the successful serving evaluation that
    unlocked the production handoff.
    """
    current_raw: Any = handoff.get("current_environment")
    technical_raw: Any = handoff.get("technical_requirements")
    if not isinstance(current_raw, dict) or not isinstance(technical_raw, dict):
        return "conflict: deployment_bundle_metadata_incomplete"
    current = cast("dict[str, Any]", current_raw)
    technical = cast("dict[str, Any]", technical_raw)
    serving_run_id = current.get("serving_evaluation_run_id")
    served_model = technical.get("nim_served_model_name")
    if not isinstance(serving_run_id, str) or not serving_run_id:
        return "conflict: deployment_bundle_serving_run_unavailable"
    if not isinstance(served_model, str) or not served_model:
        return "conflict: deployment_bundle_served_model_unavailable"

    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"
    with Session(engine) as session:
        run = session.query(RunRecord).filter_by(run_id=serving_run_id).first()
        if run is None or not run.guidance_id:
            return "conflict: deployment_bundle_serving_run_unavailable"
        guidance = (
            session.query(Guidance)
            .filter_by(project_id=project_id, guidance_id=run.guidance_id)
            .first()
        )
        if guidance is None or not guidance.schema:
            return "conflict: deployment_bundle_guidance_unavailable"
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
            return "conflict: deployment_bundle_serving_operation_unavailable"

        envelope = guidance.schema
        derived_raw: Any = envelope.get("derived_json_schema")
        derived_schema = (
            cast("dict[str, Any]", derived_raw) if isinstance(derived_raw, dict) else {}
        )
        sampling_raw: Any = operation.sampling_params_effective
        thinking_raw: Any = operation.thinking_request_fields_effective
        visual_raw: Any = operation.visual_budget_params_effective
        fields_raw: Any = envelope.get("fields")
        generation_order_raw: Any = envelope.get("generation_order")
        try:
            request, prompt_hash = build_uncapped_student_request(
                served_model=served_model,
                guidance_description=guidance.description or "",
                guidance_rules=guidance.rules or "",
                guidance_fields=(
                    cast("list[dict[str, Any]]", fields_raw)
                    if isinstance(fields_raw, list)
                    else []
                ),
                generation_order=(
                    cast("list[str]", generation_order_raw)
                    if isinstance(generation_order_raw, list)
                    else []
                ),
                derived_json_schema=derived_schema,
                inference_contract=dict(run.inference_contract or {}),
                structured_generation_attempted=bool(
                    operation.structured_generation_attempted
                ),
                sampling_params=(
                    cast("dict[str, Any]", sampling_raw)
                    if isinstance(sampling_raw, dict)
                    else None
                ),
                thinking_request_fields=(
                    cast("dict[str, Any]", thinking_raw)
                    if isinstance(thinking_raw, dict)
                    else None
                ),
                visual_budget_params=(
                    cast("dict[str, Any]", visual_raw)
                    if isinstance(visual_raw, dict)
                    else None
                ),
                image_content_part={
                    "type": "image_url",
                    "image_url": {"url": "__VLM_IMAGE_DATA_URL__"},
                },
            )
        except ValueError:
            return "conflict: deployment_bundle_prompt_image_slot_missing"

        schema_hash = envelope.get("schema_hash")
        return {
            "request": request,
            "prompt_hash": prompt_hash,
            "evaluated_prompt_hash": operation.prompt_hash,
            "guidance_id": guidance.guidance_id,
            "guidance_schema_hash": (
                schema_hash if isinstance(schema_hash, str) else None
            ),
        }


def stream_deployment_bundle(plan: DeploymentBundlePlan) -> Iterator[bytes]:
    """Yield a tar stream without materializing a second checkpoint copy."""
    read_fd, write_fd = os.pipe()
    errors: list[BaseException] = []

    def produce() -> None:
        try:
            with os.fdopen(write_fd, "wb") as sink:
                _write_archive(plan, sink)
        except BrokenPipeError:
            # The HTTP client disconnected; the consumer side owns that state.
            return
        except BaseException as exc:  # pragma: no cover - surfaced by consumer
            errors.append(exc)

    producer = threading.Thread(
        target=produce,
        name=f"deployment-bundle-{plan.student_model_id[:8]}",
        daemon=True,
    )
    producer.start()
    try:
        with os.fdopen(read_fd, "rb") as source:
            while chunk := source.read(_CHUNK_SIZE):
                yield chunk
    finally:
        producer.join()
    if errors:
        raise errors[0]


def _write_archive(plan: DeploymentBundlePlan, sink: BinaryIO) -> None:
    technical = cast("dict[str, Any]", plan.handoff["technical_requirements"])
    current = cast("dict[str, Any]", plan.handoff.get("current_environment") or {})
    readme = _render_readme(plan, technical, current).encode("utf-8")
    run_script = _render_run_script(plan, technical).encode("utf-8")
    verify_script = _render_verify_script().encode("utf-8")
    request_template = (
        json.dumps(
            plan.verification_request,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")

    checksums: list[tuple[str, str, int]] = []
    with tarfile.open(fileobj=sink, mode="w|") as archive:
        for file in plan.checkpoint_files:
            member_path = (
                Path(plan.archive_root) / "checkpoint" / file.relative_path
            ).as_posix()
            info = _tar_info(member_path, size=file.size, mode=file.mode or 0o644)
            with file.source.open("rb") as raw:
                hashing_reader = _HashingReader(raw)
                archive.addfile(info, cast("BinaryIO", hashing_reader))
                checksums.append((member_path, hashing_reader.hexdigest, file.size))

        generated = (
            ("README.md", readme, 0o644),
            ("run-nim.sh", run_script, 0o755),
            ("verify-nim.sh", verify_script, 0o755),
            ("request-template.json", request_template, 0o644),
        )
        for relative_name, content, mode in generated:
            member_path = f"{plan.archive_root}/{relative_name}"
            _add_bytes(archive, member_path, content, mode=mode)
            checksums.append(
                (member_path, hashlib.sha256(content).hexdigest(), len(content))
            )

        manifest = _build_manifest(plan, technical, current, checksums)
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
        manifest_path = f"{plan.archive_root}/manifest.json"
        _add_bytes(archive, manifest_path, manifest_bytes, mode=0o644)
        checksums.append(
            (
                manifest_path,
                hashlib.sha256(manifest_bytes).hexdigest(),
                len(manifest_bytes),
            )
        )

        checksum_text = "".join(
            f"{digest}  {Path(path).relative_to(plan.archive_root).as_posix()}\n"
            for path, digest, _size in checksums
        ).encode("utf-8")
        _add_bytes(
            archive,
            f"{plan.archive_root}/SHA256SUMS",
            checksum_text,
            mode=0o644,
        )


def _tar_info(name: str, *, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _add_bytes(
    archive: tarfile.TarFile,
    name: str,
    content: bytes,
    *,
    mode: int,
) -> None:
    archive.addfile(_tar_info(name, size=len(content), mode=mode), io.BytesIO(content))


def _portable_metrics(value: Any) -> Any:
    """Remove Blueprint-host artifact paths from an evaluation snapshot."""
    if isinstance(value, dict):
        mapping = cast("dict[Any, Any]", value)
        return {
            str(key): _portable_metrics(item)
            for key, item in mapping.items()
            if key != "artifact_dir"
        }
    if isinstance(value, list):
        items = cast("list[Any]", value)
        return [_portable_metrics(item) for item in items]
    return value


def _build_manifest(
    plan: DeploymentBundlePlan,
    technical: dict[str, Any],
    current: dict[str, Any],
    checksums: list[tuple[str, str, int]],
) -> dict[str, Any]:
    return {
        "bundle_format_version": 1,
        "generated_at": plan.handoff.get("generated_at"),
        "project": {
            "project_id": plan.project_id,
            "name": plan.handoff.get("project_name"),
        },
        "student": {
            "student_model_id": plan.student_model_id,
            "base_model_name": current.get("base_model_name"),
            "quantization_method": technical.get("quantization_method"),
            "checkpoint_packaging_status": current.get("checkpoint_packaging_status"),
        },
        "runtime": {
            "image": technical.get("nim_container_image"),
            "release_version": technical.get("nim_release_version"),
            "served_model_name": technical.get("nim_served_model_name"),
            "model_name_path": technical.get("nim_model_name_path"),
            "model_profile": technical.get("nim_model_profile_recommended"),
            "tensor_parallelism": technical.get("tensor_parallelism"),
            "gpu_requirements": technical.get("gpu_requirements"),
            "environment": technical.get("nim_env_vars_recommended"),
            "health_check": technical.get("health_check"),
            "smoke_test": technical.get("smoke_test"),
        },
        "inference_contract": technical.get("inference_contract"),
        "verification": {
            "request_template": "request-template.json",
            "guidance_id": plan.guidance_id,
            "guidance_schema_hash": plan.guidance_schema_hash,
            "rendered_prompt_sha256": plan.verification_prompt_hash,
            "serving_evaluation_prompt_sha256": plan.evaluated_prompt_hash,
            "image_placeholder": "__VLM_IMAGE_DATA_URL__",
        },
        "decoding": {
            "parameters": technical.get("decoding_params"),
            "visual_budget_preset_key": technical.get("visual_budget_preset_key"),
        },
        "evaluation": {
            "quality_status": current.get("quality_status"),
            "quality_evaluation_run_id": current.get("quality_evaluation_run_id"),
            "quality_metrics": _portable_metrics(current.get("quality_metrics")),
            "rescored_metrics": _portable_metrics(current.get("rescored_metrics")),
            "serving_status": current.get("serving_status"),
            "serving_evaluation_run_id": current.get("serving_evaluation_run_id"),
            "serving_metrics": _portable_metrics(current.get("serving_metrics")),
            "dataset_manifest_sha256": current.get("dataset_manifest_sha256"),
        },
        "training_lineage": {
            "training_tao_job_id": current.get("training_tao_job_id"),
            "quantize_tao_job_id": current.get("quantize_tao_job_id"),
            "dataset_export_ids": current.get("dataset_export_ids"),
            "training_preset": current.get("training_preset"),
            "lora_config": current.get("lora_config"),
        },
        "files": [
            {
                "path": Path(path).relative_to(plan.archive_root).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
            for path, digest, size in checksums
        ],
        "credentials_included": False,
        "nim_runtime_image_included": False,
    }


def _render_readme(
    plan: DeploymentBundlePlan,
    technical: dict[str, Any],
    current: dict[str, Any],
) -> str:
    image = technical.get("nim_container_image")
    model = technical.get("nim_served_model_name")
    return f"""# Student NIM deployment bundle

This portable bundle was generated by Interactive VLM Feedback Loop for
project `{plan.handoff.get("project_name")}` and Student
`{plan.student_model_id}`.

It includes the validated TAO checkpoint, file checksums, pinned NIM launch
configuration, evaluation snapshot, and training lineage. It does **not**
redistribute the licensed NVIDIA NIM runtime image or any credential. The
launch script pulls `{image}` from NGC with the operator's own entitlement.

## Prerequisites

- A Linux host satisfying `{technical.get("gpu_requirements")}`
- Docker, NVIDIA Container Toolkit, `curl`, `base64`, `jq`, and outbound access to `nvcr.io`
- An NGC API key authorized for the pinned NIM image

## Deploy

```bash
tar -xf {plan.archive_filename}
cd {plan.archive_root}
sha256sum -c SHA256SUMS
export NGC_API_KEY='set-this-in-your-shell'
./run-nim.sh
```

`run-nim.sh` mounts `./checkpoint` read-only and stores reusable NIM runtime
cache data under `${{NIM_CACHE_DIR:-$HOME/.cache/nim}}`. Override
`GPU_DEVICE`, `HOST_PORT`, or `NIM_CACHE_DIR` in the environment when needed.
The default endpoint is `http://127.0.0.1:8000`.

After the health endpoint reports ready, run one authorized representative
image through the exact schema-aware request contract proven by serving
evaluation. The helper fails unless NIM returns parseable structured JSON:

```bash
./verify-nim.sh /path/to/image.jpg image/jpeg
```

The served model name is `{model}`. Stop and remove the container with:

```bash
docker rm -f vlm-student-{plan.student_model_id[:8]}
```

## Evidence

- Quality status: `{current.get("quality_status")}`
- Serving status: `{current.get("serving_status")}`
- Training TAO job: `{current.get("training_tao_job_id")}`
- Quantize TAO job: `{current.get("quantize_tao_job_id")}`

See `request-template.json` for the portable request and `manifest.json` for
the complete machine-readable deployment, evaluation, and lineage snapshot.
"""


def _render_run_script(plan: DeploymentBundlePlan, technical: dict[str, Any]) -> str:
    image = str(technical["nim_container_image"])
    served_model = str(technical.get("nim_served_model_name") or "student")
    model_path = str(technical.get("nim_model_name_path") or "/opt/checkpoints/student")
    model_size = technical.get("nim_model_size")
    size_line = (
        f'  -e "NIM_MODEL_SIZE={model_size}" \\\n'
        if isinstance(model_size, str) and model_size
        else ""
    )
    profile = technical.get("nim_model_profile_recommended")
    profile_line = (
        f'  -e "NIM_MODEL_PROFILE={profile}" \\\n'
        if isinstance(profile, str) and profile
        else ""
    )
    extra_env_raw: Any = technical.get("extra_container_env") or {}
    extra_env = (
        cast("dict[str, str]", extra_env_raw) if isinstance(extra_env_raw, dict) else {}
    )
    extra_env_lines = "".join(
        f"  -e {shlex.quote(f'{key}={value}')} \\\n"
        for key, value in sorted(extra_env.items())
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

: "${{NGC_API_KEY:?Export an authorized NGC_API_KEY before running this script.}}"

BUNDLE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
GPU_DEVICE="${{GPU_DEVICE:-0}}"
HOST_PORT="${{HOST_PORT:-8000}}"
NIM_CACHE_DIR="${{NIM_CACHE_DIR:-${{HOME}}/.cache/nim}}"
mkdir -p "$NIM_CACHE_DIR"

docker pull "{image}"
docker run -d \\
  --name "vlm-student-{plan.student_model_id[:8]}" \\
  --runtime=nvidia \\
  --gpus "device=${{GPU_DEVICE}}" \\
  --shm-size=32GB \\
  --ulimit memlock=-1 \\
  --ulimit stack=67108864 \\
  -p "${{HOST_PORT}}:8000" \\
  -e NGC_API_KEY \\
  -v "${{NIM_CACHE_DIR}}:/opt/nim/.cache" \\
  -u "$(id -u)" \\
  -e NIM_MAX_IMAGES_PER_PROMPT=999 \\
  -e NIM_MAX_VIDEOS_PER_PROMPT=0 \\
  -v "${{BUNDLE_ROOT}}/checkpoint:{model_path}:ro" \\
  -e "NIM_MODEL_NAME={model_path}" \\
  -e "NIM_SERVED_MODEL_NAME={served_model}" \\
  -e NIM_ENABLE_KV_CACHE_REUSE=0 \\
{size_line}{profile_line}{extra_env_lines}  "{image}"

printf 'NIM is starting at http://127.0.0.1:%s (model: %s)\n' \\
  "$HOST_PORT" "{served_model}"
printf 'Follow startup with: docker logs -f vlm-student-{plan.student_model_id[:8]}\n'
"""


def _render_verify_script() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: %s IMAGE_PATH [MIME_TYPE]\n' "$0" >&2
  exit 2
fi

IMAGE_PATH="$1"
MIME_TYPE="${2:-image/jpeg}"
HOST_PORT="${HOST_PORT:-8000}"
BASE_URL="http://127.0.0.1:${HOST_PORT}"
BUNDLE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REQUEST_TEMPLATE="${BUNDLE_ROOT}/request-template.json"

command -v jq >/dev/null || {
  printf 'jq is required to validate the structured prediction.\n' >&2
  exit 2
}
[[ -r "$REQUEST_TEMPLATE" ]] || {
  printf 'Missing request template: %s\n' "$REQUEST_TEMPLATE" >&2
  exit 2
}
placeholder_count="$(grep -o '__VLM_IMAGE_DATA_URL__' "$REQUEST_TEMPLATE" | wc -l)"
[[ "$placeholder_count" -eq 1 ]] || {
  printf 'Request template must contain exactly one image placeholder.\n' >&2
  exit 2
}

curl --fail --silent --show-error "${BASE_URL}/v1/health/ready"
printf '\n'

template="$(cat "$REQUEST_TEMPLATE")"
prefix="${template%%__VLM_IMAGE_DATA_URL__*}"
suffix="${template#*__VLM_IMAGE_DATA_URL__}"
response_file="$(mktemp)"
trap 'rm -f "$response_file"' EXIT
{
  printf '%s' "$prefix"
  printf 'data:%s;base64,' "$MIME_TYPE"
  base64 --wrap=0 "$IMAGE_PATH"
  printf '%s' "$suffix"
} |
curl --fail --silent --show-error \\
  -H 'Content-Type: application/json' \\
  --data-binary @- \\
  "${BASE_URL}/v1/chat/completions" | tee "$response_file"
printf '\nParsed structured prediction:\n'
jq -e '.choices[0].message.content | fromjson' "$response_file"
"""


__all__ = [
    "DeploymentBundleFile",
    "DeploymentBundlePlan",
    "prepare_deployment_bundle",
    "stream_deployment_bundle",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint packaging and StudentModel registry.

Fuses checkpoint packaging and StudentModel registration
into a single service because the two always run together: packaging
produces the NIM-loadable checkpoint, registration persists the record
that references it plus the rest of the training lineage.

Called by the polling hook after ``train`` or ``quantize`` reaches
``succeeded``. One StudentModel is registered per variant:

* ``action="train"`` → full-precision baseline variant.
  ``quantization_method=None``, ``quantize_tao_job_id=None``.
* ``action="quantize"`` → quantized variant. ``quantization_method``
  copied from the quantize TAOJob's ``job_config.quantization_method``;
  ``quantize_tao_job_id`` set to the quantize job's id; ``tao_job_id``
  remains the parent ``train`` job's id for chain lineage.

``quality_status`` starts as ``"pending"`` — the paired ``evaluate``
succeeded transition flips it to ``"validated"`` after re-scoring, or
to ``"failed"`` on evaluate failure / empty predictions (C2).
``serving_status`` defaults to ``"not_attempted"`` — the NIM deployment
lifecycle flips it.

Checkpoint packaging contract:

1. Already a NIM-loadable HuggingFace checkpoint (full model shards +
   ``config.json`` + tokenizer files) → ``"validated"``, no merge.
2. Adapter-only (LoRA ``adapter_config.json`` +
   ``adapter_model.{safetensors,bin}``) with ``base_model_path`` on the
   TAOJob's ``resolved_training_fields.policy.model_name_or_path``:
   spawn ``python scripts/merge_lora.py`` as a subprocess, validate the
   merged directory, mark ``"validated"``.
3. Adapter-only WITHOUT ``base_model_path`` → ``"failed"`` with error
   ``"base_model_path_unavailable"``. Packaging MUST fail with a clear
   error in this case.
4. Neither pattern detected → ``"failed"`` with
   ``"unrecognized_checkpoint_layout"``.

For ``action="quantize"`` jobs, TAO auto-merges when
``enable_lora=true`` + ``base_model_path`` is set; packaging
here only validates the NIM-loadable structure of the quantize output.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services.inference_contract_resolver import (
    resolve_training_inference_contract,
)
from vlm_feedback_loop.services.pagination import (
    after_position,
    decode_cursor,
    encode_cursor,
)
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.tao_job_service import (
    find_suite_for_chain,
    find_train_job_for_chain,
)

logger = logging.getLogger("vlm_feedback_loop.services.student_model_service")

# Path to the repo-root merge script. `parents[4]` from
#   src/backend/vlm_feedback_loop/services/student_model_service.py
# resolves to the repo root, where `scripts/` lives (`parents[3]` stops
# one level short, at `src/`).
_MERGE_LORA_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "merge_lora.py"

# Shard filename patterns that count as a "NIM-loadable HuggingFace
# checkpoint". Any ONE of these alongside config.json + tokenizer
# is sufficient for validation.
_HF_WEIGHT_PATTERNS = (
    "*.safetensors",
    "pytorch_model*.bin",
    "model*.safetensors",
)
_HF_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
_HF_CONFIG_FILE = "config.json"
_ADAPTER_CONFIG_FILE = "adapter_config.json"
_ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


# ── Result dataclass ────────────────────────────────────────────────────────


@dataclass
class PackagingResult:
    """Structured outcome of :func:`_package_checkpoint`."""

    status: str  # "validated" | "failed"
    nim_checkpoint_ref: str | None
    error: str | None
    merged: bool = False  # True when a LoRA merge was performed


# ── Checkpoint detection helpers ───────────────────────────────────────────


def _dir_has_any(path: Path, patterns: tuple[str, ...]) -> bool:
    return any(any(path.glob(pat)) for pat in patterns)


def _dir_has_tokenizer(path: Path) -> bool:
    return any((path / name).is_file() for name in _HF_TOKENIZER_FILES)


def _is_nim_loadable_hf_checkpoint(path: Path) -> bool:
    """Return True if ``path`` is already a NIM-loadable HF checkpoint.

    Requires: ``config.json`` + at least one weight shard + at least one
    tokenizer file. Matches the shape expected by NIM's vLLM backend for
    Cosmos Reason2 custom-checkpoint deployments.
    """
    if not path.is_dir():
        return False
    if not (path / _HF_CONFIG_FILE).is_file():
        return False
    if not _dir_has_any(path, _HF_WEIGHT_PATTERNS):
        return False
    return _dir_has_tokenizer(path)


def _is_adapter_only_output(path: Path) -> bool:
    """Return True if ``path`` contains a LoRA adapter without the base."""
    if not path.is_dir():
        return False
    has_adapter_config = (path / _ADAPTER_CONFIG_FILE).is_file()
    has_adapter_weights = any((path / name).is_file() for name in _ADAPTER_WEIGHT_FILES)
    if not (has_adapter_config and has_adapter_weights):
        return False
    # If it ALSO has full HF shards + tokenizer, it's the merged path, not adapter-only.
    return not (
        (path / _HF_CONFIG_FILE).is_file()
        and _dir_has_any(path, _HF_WEIGHT_PATTERNS)
        and _dir_has_tokenizer(path)
    )


def _resolve_artifact_dir(job: TAOJob) -> Path | None:
    """Extract the artifact cache directory TAO downloaded bytes into."""
    outputs = job.outputs or {}
    cache_dir = outputs.get("artifact_cache_dir")
    if cache_dir:
        return Path(cache_dir)
    return None


def _resolve_checkpoint_search_dir(job: TAOJob) -> Path | None:
    """Return the directory to inspect for checkpoint shape.

    Prefers the ``best_model`` artifact subdirectory when it exists
    (TAO-Cosmos-RL convention), otherwise falls back to the artifact
    cache root. For both train and quantize jobs we treat the ``best_model``
    slot as the primary deliverable.
    """
    base = _resolve_artifact_dir(job)
    if base is None:
        return None
    for candidate in ("best_model", "latest_model", "quantized_model"):
        sub = base / candidate
        if sub.is_dir():
            return sub
    # Some TAO outputs drop files at the root; treat the cache dir itself
    # as the checkpoint if it already satisfies HF shape.
    return base


def _resolve_base_model_path(job: TAOJob) -> str | None:
    """Return ``base_model_path`` for LoRA merge, or None if unavailable."""
    job_config: dict[str, Any] = job.job_config or {}
    resolved_raw: Any = job_config.get("resolved_training_fields") or {}
    if not isinstance(resolved_raw, dict):
        return None
    resolved = cast("dict[str, Any]", resolved_raw)
    policy_raw: Any = resolved.get("policy")
    if not isinstance(policy_raw, dict):
        return None
    policy = cast("dict[str, Any]", policy_raw)
    raw: Any = policy.get("model_name_or_path")
    if raw and isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


# ── LoRA merge (subprocess) ────────────────────────────────────────────────


def _resolve_merge_python(settings: Settings) -> str:
    """Interpreter for the LoRA-merge subprocess.

    The backend venv deliberately does NOT carry transformers/peft/torch,
    so ``sys.executable`` cannot run the merge as shipped. Resolution:

    1. ``MERGE_LORA_PYTHON`` setting — explicit interpreter path.
    2. ``{WORKSPACE_ROOT}/merge-lora-venv/bin/python`` when present —
       the documented provisioning target
       (``uv venv $WORKSPACE_ROOT/merge-lora-venv && uv pip install
       --python $WORKSPACE_ROOT/merge-lora-venv/bin/python
       -r scripts/merge_lora_requirements.txt``).
    3. ``~/.local/share/vlm-feedback-loop/merge-lora-venv/bin/python`` —
       the shared Profile C runtime provisioned by ``scripts/setup-dev.sh``.
    4. ``sys.executable`` — works only if the operator installed the
       merge requirements into the backend venv; the failure message
       below says how to provision when the import fails.
    """
    configured = (settings.MERGE_LORA_PYTHON or "").strip()
    if configured:
        return str(Path(configured).expanduser())
    if settings.WORKSPACE_ROOT:
        venv_python = (
            Path(settings.WORKSPACE_ROOT) / "merge-lora-venv" / "bin" / "python"
        )
        if venv_python.exists():
            return str(venv_python)
    shared_python = (
        Path.home()
        / ".local"
        / "share"
        / "vlm-feedback-loop"
        / "merge-lora-venv"
        / "bin"
        / "python"
    )
    if shared_python.exists():
        return str(shared_python)
    return sys.executable


_MERGE_PROVISION_HINT = (
    "The merge interpreter lacks transformers/peft/torch. Provision a "
    "dedicated venv by rerunning scripts/setup-dev.sh, or manually: "
    'uv venv "$HOME/.local/share/vlm-feedback-loop/merge-lora-venv" && '
    "uv pip install --python "
    '"$HOME/.local/share/vlm-feedback-loop/merge-lora-venv/bin/python" '
    "-r scripts/merge_lora_requirements.txt — or point MERGE_LORA_PYTHON "
    "at an interpreter that has them."
)


async def check_lora_merge_readiness(settings: Settings) -> tuple[bool, str]:
    """Verify that the isolated LoRA merge interpreter is usable.

    This is intentionally a package-discovery probe rather than importing
    torch/transformers into the FastAPI process. It is cheap enough for the
    Student Training readiness gate and catches the exact environment failure
    that would otherwise appear only after a remote training job completes.
    """
    python = _resolve_merge_python(settings)
    if not Path(python).is_file():
        return False, f"LoRA merge interpreter does not exist: {python}"
    probe = (
        "import importlib.util,sys;"
        "missing=[m for m in "
        "('torch','transformers','peft','accelerate','safetensors') "
        "if importlib.util.find_spec(m) is None];"
        "print(','.join(missing));sys.exit(bool(missing))"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            python,
            "-c",
            probe,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15.0)
    except (OSError, TimeoutError) as exc:
        return False, f"LoRA merge runtime probe failed: {exc}"
    if proc.returncode != 0:
        missing = stdout.decode(errors="replace").strip()
        detail = missing or stderr.decode(errors="replace").strip()
        return False, f"LoRA merge runtime is missing required packages: {detail}"
    return True, f"LoRA merge runtime ready at {python}."


async def _run_merge_lora_subprocess(
    adapter_dir: Path,
    base_model_path: str,
    out_dir: Path,
    *,
    settings: Settings,
    timeout_s: float = 3600.0,
) -> dict[str, Any]:
    """Spawn ``scripts/merge_lora.py`` and wait for completion.

    Returns ``{"ok": bool, "stdout": str, "stderr": str, "returncode": int}``.
    The backend does NOT import ``transformers`` or ``peft`` — this keeps
    the FastAPI process slim; the interpreter comes from
    :func:`_resolve_merge_python`.
    """
    cmd = [
        _resolve_merge_python(settings),
        str(_MERGE_LORA_SCRIPT),
        "--adapter",
        str(adapter_dir),
        "--base",
        base_model_path,
        "--out",
        str(out_dir),
    ]
    logger.info(
        "merge_lora: spawning %s",
        " ".join(shlex.quote(c) for c in cmd),
    )
    # The base model is a gated/auth'd HF repo for most Cosmos
    # tiers. ``create_subprocess_exec`` without ``env`` inherits the
    # backend process environment, which does NOT necessarily carry the
    # operator's token (it lives in ~/.vlm_feedback_loop/.env and is
    # loaded into Settings, not exported) — observed live: the first
    # adapter merge failed with "not a valid model identifier" because
    # from_pretrained ran unauthenticated. Forward settings.HF_TOKEN as
    # both env names huggingface_hub honors.
    merge_env = dict(os.environ)
    if settings.HF_TOKEN:
        merge_env["HF_TOKEN"] = settings.HF_TOKEN
        merge_env["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merge_env,
        )
    except (FileNotFoundError, OSError) as exc:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"failed to spawn merge_lora subprocess: {exc}",
            "returncode": -1,
        }

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"merge_lora subprocess timed out after {timeout_s}s",
            "returncode": -2,
        }

    stderr_text = (stderr_b or b"").decode("utf-8", errors="replace")
    if proc.returncode != 0 and (
        "ModuleNotFoundError" in stderr_text or "ImportError" in stderr_text
    ):
        stderr_text = f"{stderr_text.strip()}\n{_MERGE_PROVISION_HINT}"
    return {
        "ok": proc.returncode == 0,
        "stdout": (stdout_b or b"").decode("utf-8", errors="replace"),
        "stderr": stderr_text,
        "returncode": proc.returncode if proc.returncode is not None else -1,
    }


# ── Checkpoint packaging ──────────────────────────────────────────────────


async def _package_checkpoint(job: TAOJob, *, settings: Settings) -> PackagingResult:
    """Package a TAO training / quantize output for NIM.

    Pure on-disk logic + (optionally) a merge subprocess. No DB writes —
    the caller persists the ``PackagingResult`` onto the StudentModel.

    ``settings`` selects the interpreter when a LoRA adapter must be merged.
    """
    search_dir = _resolve_checkpoint_search_dir(job)
    if search_dir is None:
        return PackagingResult(
            status="failed",
            nim_checkpoint_ref=None,
            error="packaging: no artifact_cache_dir on TAOJob outputs — TAO download may have failed",
        )
    if not search_dir.is_dir():
        return PackagingResult(
            status="failed",
            nim_checkpoint_ref=None,
            error=f"packaging: artifact directory does not exist: {search_dir}",
        )

    # Case 1: already NIM-loadable HF checkpoint (no merge needed).
    if _is_nim_loadable_hf_checkpoint(search_dir):
        return PackagingResult(
            status="validated",
            nim_checkpoint_ref=str(search_dir),
            error=None,
            merged=False,
        )

    # For quantize jobs: if the output is not already HF-loadable, we do
    # NOT run a LoRA merge — TAO's quantize step auto-merged when it ran.
    # A missing HF shape indicates the quantize output is malformed or we
    # downloaded the wrong slot.
    if job.action == "quantize":
        return PackagingResult(
            status="failed",
            nim_checkpoint_ref=None,
            error=(
                "packaging: quantize output is not a NIM-loadable HF checkpoint. "
                "Expected config.json + shards + tokenizer at "
                f"{search_dir}"
            ),
        )

    # Case 2: adapter-only. Merge is required.
    if _is_adapter_only_output(search_dir):
        base_model_path = _resolve_base_model_path(job)
        if not base_model_path:
            # Packaging MUST fail cleanly if base_model_path is missing.
            return PackagingResult(
                status="failed",
                nim_checkpoint_ref=None,
                error=(
                    "packaging: LoRA adapter detected but base_model_path is "
                    "unavailable on TAOJob.job_config.resolved_training_fields."
                    "policy.model_name_or_path. "
                    "Error: base_model_path_unavailable"
                ),
            )

        artifact_root = _resolve_artifact_dir(job)
        if artifact_root is None:
            return PackagingResult(
                status="failed",
                nim_checkpoint_ref=None,
                error="packaging: artifact_cache_dir unexpectedly missing at merge time",
            )
        merged_dir = artifact_root / "merged"
        merge_result = await _run_merge_lora_subprocess(
            search_dir, base_model_path, merged_dir, settings=settings
        )
        if not merge_result["ok"]:
            return PackagingResult(
                status="failed",
                nim_checkpoint_ref=None,
                error=(
                    f"packaging: LoRA merge failed "
                    f"(rc={merge_result['returncode']}): "
                    f"{merge_result['stderr'].strip()[:512]}"
                ),
            )

        # Re-validate the merged output.
        if not _is_nim_loadable_hf_checkpoint(merged_dir):
            return PackagingResult(
                status="failed",
                nim_checkpoint_ref=None,
                error=(
                    "packaging: merged checkpoint does not satisfy NIM-loadable "
                    f"HF shape at {merged_dir}"
                ),
            )

        return PackagingResult(
            status="validated",
            nim_checkpoint_ref=str(merged_dir),
            error=None,
            merged=True,
        )

    # Case 3: neither pattern — unrecognized output.
    return PackagingResult(
        status="failed",
        nim_checkpoint_ref=None,
        error=(
            "packaging: unrecognized_checkpoint_layout — no HF shape and no "
            f"LoRA adapter at {search_dir}"
        ),
    )


# ── Chain + lineage walking ───────────────────────────────────────────────


def _existing_student_for_tao_job(
    session: Session,
    *,
    project_id: str,
    tao_job_id: str,
    quantize_tao_job_id: str | None,
) -> StudentModel | None:
    """Idempotency guard — return the StudentModel already registered for
    this train+quantize pair, if any.
    """
    q = session.query(StudentModel).filter(
        StudentModel.project_id == project_id,
        StudentModel.tao_job_id == tao_job_id,
    )
    if quantize_tao_job_id is None:
        # Full-precision baseline: quantize_tao_job_id is null.
        q = q.filter(StudentModel.quantize_tao_job_id.is_(None))
    else:
        q = q.filter(StudentModel.quantize_tao_job_id == quantize_tao_job_id)
    return q.first()


def find_student_for_evaluate_job(
    session: Session,
    *,
    project_id: str,
    evaluate_job: TAOJob,
) -> StudentModel | None:
    """Return the StudentModel the evaluate job scores, or None.

    Walk via ``parent_tao_job_id``:
    * parent is a ``train`` → baseline variant (quantize_tao_job_id IS NULL).
    * parent is a ``quantize`` → quantized variant; the StudentModel has
      ``quantize_tao_job_id = parent``.
    """
    if evaluate_job.action != "evaluate" or not evaluate_job.parent_tao_job_id:
        return None
    parent = (
        session.query(TAOJob)
        .filter(
            TAOJob.project_id == project_id,
            TAOJob.tao_job_id == evaluate_job.parent_tao_job_id,
        )
        .first()
    )
    if parent is None:
        return None
    if parent.action == "train":
        return _existing_student_for_tao_job(
            session,
            project_id=project_id,
            tao_job_id=parent.tao_job_id,
            quantize_tao_job_id=None,
        )
    if parent.action == "quantize":
        train = find_train_job_for_chain(
            session, project_id=project_id, chain_id=parent.chain_id
        )
        if train is None:
            return None
        return _existing_student_for_tao_job(
            session,
            project_id=project_id,
            tao_job_id=train.tao_job_id,
            quantize_tao_job_id=parent.tao_job_id,
        )
    return None


def mark_student_quality_failed(
    *,
    student: StudentModel,
    reason: str,
) -> None:
    """Idempotent flip to quality_status="failed" with status_reason
    captured in the nim_preflight_details JSON blob for audit.

    We stash the reason in ``nim_preflight_details`` only as a quick
    audit channel; the NIM deployment preflight overwrites it with
    actual preflight data. If you need a persistent record, the
    evaluation Run Record created by the re-scoring service carries
    ``status_reason``.
    """
    student.quality_status = "failed"
    details = dict(student.nim_preflight_details or {})
    details.setdefault("quality_failure_reason", reason)
    student.nim_preflight_details = details


# ── Primary entry point ───────────────────────────────────────────────────


async def register_from_tao_terminal(
    project_id: str,
    tao_job_id: str,
    *,
    settings: Settings,
) -> str | None:
    """Package the checkpoint and register a StudentModel.

    Idempotent — if a StudentModel already exists for the
    ``(train_job, quantize_job)`` pair we update the existing record in
    place rather than inserting a duplicate.

    Returns the ``student_model_id`` on success (validated or failed
    packaging — both produce a record), ``None`` only when the TAOJob
    cannot be resolved.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        logger.warning(
            "register_from_tao_terminal: project %s has no engine", project_id
        )
        return None

    # Load TAOJob + associated lineage in a short read-only session.
    with Session(engine) as session:
        job = (
            session.query(TAOJob)
            .filter(
                TAOJob.project_id == project_id,
                TAOJob.tao_job_id == tao_job_id,
            )
            .first()
        )
        if job is None:
            logger.warning(
                "register_from_tao_terminal: TAOJob %s not found in project %s",
                tao_job_id,
                project_id,
            )
            return None
        if job.action not in ("train", "quantize"):
            logger.debug(
                "register_from_tao_terminal: skipping %s job (action=%s)",
                tao_job_id,
                job.action,
            )
            return None

        suite = find_suite_for_chain(
            session, project_id=project_id, chain_id=job.chain_id
        )
        train_job = find_train_job_for_chain(
            session, project_id=project_id, chain_id=job.chain_id
        )
        # Detach for async packaging (avoids holding DB session while
        # subprocess runs).
        job_snap: dict[str, Any] = {
            "tao_job_id": job.tao_job_id,
            "action": job.action,
            "outputs": dict(job.outputs or {}),
            "job_config": dict(job.job_config or {}),
            "chain_id": job.chain_id,
            "student_base_model_config_id": job.student_base_model_config_id,
            "dataset_export_ids": list(job.dataset_export_ids or []),
        }
        train_snap: dict[str, Any] | None = (
            {
                "tao_job_id": train_job.tao_job_id,
                "job_config": dict(train_job.job_config or {}),
                "dataset_export_ids": list(train_job.dataset_export_ids or []),
                "student_base_model_config_id": train_job.student_base_model_config_id,
            }
            if train_job is not None
            else None
        )
        suite_snap: dict[str, Any] | None = (
            {
                "guidance_id": suite.guidance_id,
                "training_preset": suite.training_preset,
            }
            if suite is not None
            else None
        )

    # Run packaging OUTSIDE the DB session (no long-running ops inside
    # a write transaction). Reconstruct a lightweight TAOJob-shaped object
    # with just the fields packaging reads.
    class _JobSnap:
        tao_job_id: str
        action: str
        outputs: dict[str, Any]
        job_config: dict[str, Any]

    j = _JobSnap()
    j.tao_job_id = cast("str", job_snap["tao_job_id"])
    j.action = cast("str", job_snap["action"])
    j.outputs = cast("dict[str, Any]", job_snap["outputs"])
    j.job_config = cast("dict[str, Any]", job_snap["job_config"])

    # _package_checkpoint reads only .tao_job_id/.action/.outputs/.job_config —
    # the duck-type matches TAOJob's attributes used by packaging.
    packaging = await _package_checkpoint(cast("TAOJob", j), settings=settings)

    # Persist: create or update StudentModel in a short write transaction.
    is_quantize = job_snap["action"] == "quantize"
    if is_quantize:
        if train_snap is None:
            logger.warning(
                "register_from_tao_terminal: quantize job %s has chain_id=%s "
                "but no train job found — cannot register StudentModel",
                tao_job_id,
                job_snap["chain_id"],
            )
            return None
        base_tao_job_id = train_snap["tao_job_id"]
        quantize_tao_job_id = job_snap["tao_job_id"]
        quantization_method = job_snap["job_config"].get("quantization_method")
        lineage_job_config = train_snap["job_config"]
        dataset_export_ids = train_snap["dataset_export_ids"]
        student_base_id = train_snap["student_base_model_config_id"]
    else:
        base_tao_job_id = job_snap["tao_job_id"]
        quantize_tao_job_id = None
        quantization_method = None
        lineage_job_config = job_snap["job_config"]
        dataset_export_ids = job_snap["dataset_export_ids"]
        student_base_id = job_snap["student_base_model_config_id"]

    guidance_id = None
    if suite_snap is not None:
        guidance_id = suite_snap["guidance_id"]
    else:
        # Fallback: should not happen for suite-created chains, but be
        # resilient for ad-hoc jobs.
        guidance_id = lineage_job_config.get("guidance_id") or "unknown"

    training_preset = (
        (suite_snap and suite_snap.get("training_preset"))
        or lineage_job_config.get("training_preset")
        or "standard"
    )
    lora_config_raw: Any = lineage_job_config.get("lora_config")
    lora_config: dict[str, Any] = (
        cast("dict[str, Any]", lora_config_raw)
        if isinstance(lora_config_raw, dict)
        else {}
    )

    now = utc_now()
    with Session(engine) as session:
        existing = _existing_student_for_tao_job(
            session,
            project_id=project_id,
            tao_job_id=base_tao_job_id,
            quantize_tao_job_id=quantize_tao_job_id,
        )
        if existing is not None:
            existing.checkpoint_packaging_status = packaging.status
            existing.nim_checkpoint_ref = packaging.nim_checkpoint_ref
            # Preserve quality_status, serving_status — only the evaluate
            # path flips those. An already-validated-quality record must
            # not regress when we re-run packaging.
            session.commit()
            logger.info(
                "StudentModel %s re-packaged: status=%s (merged=%s)",
                existing.student_model_id,
                packaging.status,
                packaging.merged,
            )
            return existing.student_model_id

        # Snapshot the Inference Contract from the training DatasetExport's
        # ``export_field_mode``. Stable comparison key for the
        # deployment-handoff gate's INFERENCE_CONTRACT_MISMATCH check.
        training_contract = resolve_training_inference_contract(
            session, list(dataset_export_ids)
        )

        student = StudentModel(
            student_model_id=generate_uuid4(),
            project_id=project_id,
            student_base_model_config_id=student_base_id,
            tao_job_id=base_tao_job_id,
            guidance_id=guidance_id,
            dataset_export_ids=list(dataset_export_ids),
            training_preset=training_preset,
            lora_config=lora_config,
            created_at=now,
            checkpoint_packaging_status=packaging.status,
            nim_checkpoint_ref=packaging.nim_checkpoint_ref,
            quality_status="pending",
            quality_evaluation_run_id=None,
            serving_status="not_attempted",
            serving_evaluation_run_id=None,
            nim_preflight_status=None,
            nim_preflight_details=None,
            nim_preflight_at=None,
            nim_deployment_mode=None,
            nim_container_id=None,
            nim_endpoint_url=None,
            nim_vlm_release_version=None,
            nim_model_profile_requested=None,
            nim_model_profile_selected=None,
            nim_profile_metadata=None,
            gpu_type=None,
            gpu_count=None,
            quantization_method=quantization_method,
            quantize_tao_job_id=quantize_tao_job_id,
            training_inference_contract=training_contract,
        )
        session.add(student)
        session.commit()
        sid = student.student_model_id

    logger.info(
        "StudentModel %s registered: variant=%s packaging=%s",
        sid,
        "quantized" if is_quantize else "baseline",
        packaging.status,
    )
    if packaging.status == "failed":
        logger.warning(
            "StudentModel %s checkpoint packaging failed: %s",
            sid,
            packaging.error,
        )
    return sid


# ── List / get helpers for the router ──────────────────────────────────────


def list_student_models(
    *,
    project_id: str,
    workspace_root: str,
    limit: int = 50,
    cursor: str | None = None,
) -> tuple[list[StudentModel], str | None]:
    """Paginated list of StudentModel records, newest-first.

    Each row is annotated with a ``base_model_name`` attribute joined
    from ``ModelConfig.model_name`` so the response schema can surface
    it without forcing every consumer (Compare UI, full_stack_validation
    classifier, deployment_handoff renderer) to make a second call.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return ([], None)

    with Session(engine) as session:
        q = (
            session.query(StudentModel)
            .filter(StudentModel.project_id == project_id)
            .order_by(
                StudentModel.created_at.desc(),
                StudentModel.student_model_id.desc(),
            )
        )
        if cursor:
            cursor_ts, cursor_id = decode_cursor(cursor)
            q = q.filter(
                after_position(
                    StudentModel.created_at,
                    StudentModel.student_model_id,
                    cursor_ts,
                    cursor_id,
                )
            )
        rows = q.limit(limit + 1).all()
        next_cursor = (
            encode_cursor(rows[limit - 1].created_at, rows[limit - 1].student_model_id)
            if len(rows) > limit
            else None
        )
        items = rows[:limit]
        # Resolve base model names in a single follow-up query so the
        # response schema can surface ``base_model_name`` without N+1
        # round-trips.
        if items:
            base_ids = sorted({s.student_base_model_config_id for s in items})
            mc_rows = (
                session.query(ModelConfig.model_config_id, ModelConfig.model_name)
                .filter(
                    ModelConfig.project_id == project_id,
                    ModelConfig.model_config_id.in_(base_ids),
                )
                .all()
            )
            name_by_id: dict[str, str] = {row[0]: row[1] for row in mc_rows}
            for row in items:
                # base_model_name is a non-mapped attribute attached on the
                # ORM instance for response shaping (see schemas.student_model).
                setattr(  # noqa: B010 — non-mapped attribute on a SQLAlchemy ORM row
                    row,
                    "base_model_name",
                    name_by_id.get(row.student_base_model_config_id),
                )
        for row in items:
            session.expunge(row)

    return (items, next_cursor)


async def repackage_student_model(
    *,
    project_id: str,
    student_model_id: str,
    settings: Settings,
) -> dict[str, str | None]:
    """Replay checkpoint packaging for a Student whose packaging failed.

    Mirrors the F33 ``:rerescore`` rationale for the packaging stage:
    packaging can fail for environment reasons that the operator then
    fixes — canonically a missing LoRA-merge interpreter
    (``merge-lora-venv`` unprovisioned; observed live 2026-07-15 on the
    first adapter-only checkpoint the Blueprint ever produced) — and
    without a replay path the only recovery was re-running the entire
    training chain. Delegates to :func:`register_from_tao_terminal`,
    which is idempotent and updates the existing StudentModel in place.

    Safety: refuses (409) unless ``checkpoint_packaging_status`` is
    ``"failed"`` — a validated checkpoint must never be re-materialized
    underneath a live serving/eval consumer.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return {"error": "project_not_found", "student_model_id": None}
    with Session(engine) as session:
        student = (
            session.query(StudentModel)
            .filter(
                StudentModel.project_id == project_id,
                StudentModel.student_model_id == student_model_id,
            )
            .first()
        )
        if student is None:
            return {"error": "student_not_found", "student_model_id": None}
        if student.checkpoint_packaging_status != "failed":
            return {
                "error": "packaging_not_failed",
                "student_model_id": student_model_id,
            }
        tao_job_id = (
            student.quantize_tao_job_id or student.tao_job_id
        )  # variant-aware: quantized students repackage from their quantize job

    result_id = await register_from_tao_terminal(
        project_id, tao_job_id, settings=settings
    )
    if result_id is None:
        return {"error": "tao_job_unresolved", "student_model_id": student_model_id}
    with Session(engine) as session:
        refreshed = session.get(StudentModel, result_id)
        status = refreshed.checkpoint_packaging_status if refreshed else None
    return {
        "error": None,
        "student_model_id": result_id,
        "checkpoint_packaging_status": status,
    }


def get_student_model(
    *,
    project_id: str,
    student_model_id: str,
    workspace_root: str,
) -> StudentModel | None:
    """Retrieve a single StudentModel by id, project-scoped.

    Annotates the row with ``base_model_name`` from the joined
    ``ModelConfig`` so the response schema picks it up via
    ``from_attributes``.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None
    with Session(engine) as session:
        row = (
            session.query(StudentModel)
            .filter(
                StudentModel.project_id == project_id,
                StudentModel.student_model_id == student_model_id,
            )
            .first()
        )
        if row is not None:
            mc = (
                session.query(ModelConfig.model_name)
                .filter(
                    ModelConfig.project_id == project_id,
                    ModelConfig.model_config_id == row.student_base_model_config_id,
                )
                .first()
            )
            # base_model_name is a non-mapped attribute attached on the
            # ORM instance for response shaping.
            setattr(  # noqa: B010 — non-mapped attribute on a SQLAlchemy ORM row
                row,
                "base_model_name",
                mc[0] if mc else None,
            )
            session.expunge(row)
    return row


# ── NIM deployment dispatch ─────────────────────────────────────────────────


# Per-project lock guarding the single-active student-NIM-deploy invariant.
# Two SMEs (or two retry clicks) hitting `:deploy_nim` for different
# Students in the same project would otherwise both spawn containers
# fighting for the same GPU. We reject the second attempt with 409.
_DEPLOY_LOCKS: dict[str, asyncio.Lock] = {}


def _project_lock(project_id: str) -> asyncio.Lock:
    if project_id not in _DEPLOY_LOCKS:
        _DEPLOY_LOCKS[project_id] = asyncio.Lock()
    return _DEPLOY_LOCKS[project_id]


def _has_in_flight_deploy(project_id: str) -> bool:
    """Check whether the background_manager has any student-nim-* task
    running for this project.
    """
    # Local import to avoid a top-of-file circular with services.background.
    from vlm_feedback_loop.services.background import background_manager

    prefix = f"student-nim-{project_id}-"
    return any(tid.startswith(prefix) for tid in background_manager.active_task_ids)


async def run_automatic_baseline_evaluation(
    *,
    project_id: str,
    evaluate_tao_job_id: str,
    settings: Settings,
) -> dict[str, Any]:
    """Merge/package and evaluate a LoRA baseline through the local Student NIM.

    TAO's v2 ``evaluate`` action cannot consume an adapter-only checkpoint.
    The Blueprint therefore owns this baseline path: retry idempotent
    packaging (which merges the adapter), run the normal local Student NIM
    lifecycle against the Test Pool, and return its durable outcome for the
    synthetic evaluate TAOJob row.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return {"success": False, "error": "project_not_found"}

    with Session(engine) as session:
        evaluate_job = (
            session.query(TAOJob)
            .filter_by(project_id=project_id, tao_job_id=evaluate_tao_job_id)
            .first()
        )
        if evaluate_job is None:
            return {"success": False, "error": "evaluate_job_not_found"}
        student = find_student_for_evaluate_job(
            session, project_id=project_id, evaluate_job=evaluate_job
        )
        parent_job_id = evaluate_job.parent_tao_job_id
        student_id = student.student_model_id if student is not None else None

    # Registration is idempotent and is also the retry mechanism for a merge
    # runtime that was provisioned after the first packaging attempt.
    if parent_job_id is None:
        return {"success": False, "error": "evaluate_parent_missing"}
    if student_id is None:
        student_id = await register_from_tao_terminal(
            project_id, parent_job_id, settings=settings
        )
    else:
        with Session(engine) as session:
            student = session.get(StudentModel, student_id)
            packaging_failed = bool(
                student is not None and student.checkpoint_packaging_status == "failed"
            )
        if packaging_failed:
            student_id = await register_from_tao_terminal(
                project_id, parent_job_id, settings=settings
            )
    if student_id is None:
        return {"success": False, "error": "student_registration_failed"}

    async with _project_lock(project_id):
        if _has_in_flight_deploy(project_id):
            return {"success": False, "error": "student_nim_deploy_in_progress"}
        with Session(engine) as session:
            student = session.get(StudentModel, student_id)
            if student is None:
                return {"success": False, "error": "student_not_found"}
            if student.checkpoint_packaging_status != "validated":
                return {
                    "success": False,
                    "error": "checkpoint_packaging_failed",
                    "detail": (student.nim_preflight_details or {}).get(
                        "quality_failure_reason"
                    ),
                }
            student.serving_status = "pending"
            student.nim_deployment_mode = "local"
            student.nim_endpoint_url = None
            student.nim_preflight_status = None
            student.nim_preflight_details = None
            student.nim_preflight_at = None
            student.nim_container_id = None
            student.serving_evaluation_run_id = None
            session.commit()

        from vlm_feedback_loop.services import student_nim_lifecycle

        await student_nim_lifecycle.run_student_deployment_lifecycle(
            project_id=project_id,
            student_model_id=student_id,
            mode="local",
            nim_endpoint_url=None,
            nim_container_image=None,
            nim_release_version=None,
            gpu_assignment=None,
            auth_mode="none",
            settings=settings,
            workspace_root=settings.WORKSPACE_ROOT,
        )

    with Session(engine) as session:
        student = session.get(StudentModel, student_id)
        if student is None:
            return {"success": False, "error": "student_not_found_after_evaluation"}
        success = (
            student.serving_status == "validated"
            and student.quality_status in {"validated", "partial"}
            and student.serving_evaluation_run_id is not None
        )
        return {
            "success": success,
            "student_model_id": student_id,
            "evaluation_run_id": student.serving_evaluation_run_id,
            "quality_status": student.quality_status,
            "serving_status": student.serving_status,
            "error": None if success else "student_nim_evaluation_failed",
            "detail": student.nim_preflight_details,
        }


async def deploy_nim(
    *,
    project_id: str,
    student_model_id: str,
    nim_endpoint_url: str | None,
    nim_container_image: str | None,
    nim_release_version: str | None,
    gpu_assignment: str | None,
    auth_mode: str = "none",
    settings: Settings,
) -> dict[str, Any]:
    """Validate the request, write mode to the StudentModel, register the
    background lifecycle task, return a 202 dict.

    Returns:
        dict on success — see schemas.student_nim_deploy.DeployNimResponse.
        dict with key "error" on validation failure (router maps to HTTP):
          - error="student_not_found"        → 404
          - error="checkpoint_not_validated" → 400
          - error="invalid_external_url"     → 400
          - error="deploy_in_progress"       → 409 (with task_id of the
            in-flight deployment so the SME can correlate)
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return {"error": "student_not_found"}

    with Session(engine) as session:
        student = session.get(StudentModel, student_model_id)
        if student is None or student.project_id != project_id:
            return {"error": "student_not_found"}
        if student.checkpoint_packaging_status != "validated":
            return {"error": "checkpoint_not_validated"}

    if nim_endpoint_url is not None and not (
        nim_endpoint_url.startswith("http://")
        or nim_endpoint_url.startswith("https://")
    ):
        return {"error": "invalid_external_url"}

    mode: str = "external" if nim_endpoint_url else "local"

    # Project-level single-active invariant.
    async with _project_lock(project_id):
        if _has_in_flight_deploy(project_id):
            return {"error": "deploy_in_progress"}

        # Write the dispatch state BEFORE registering the task so SSE
        # consumers / restart recovery see consistent state immediately.
        with Session(engine) as session:
            student = session.get(StudentModel, student_model_id)
            if student is None:  # pragma: no cover — racy delete
                return {"error": "student_not_found"}
            student.serving_status = "pending"
            student.nim_deployment_mode = mode
            student.nim_endpoint_url = nim_endpoint_url if mode == "external" else None
            student.nim_preflight_status = None
            student.nim_preflight_details = None
            student.nim_preflight_at = None
            student.nim_container_id = None
            student.serving_evaluation_run_id = None
            session.commit()

        # Register the background task. Local import to avoid the
        # services.student_nim_lifecycle ↔ services.evaluation_service
        # ↔ services.local_nim_service import order at module load.
        from vlm_feedback_loop.services import student_nim_lifecycle
        from vlm_feedback_loop.services.background import background_manager

        # Epoch-suffixed task id makes successive retries (same project +
        # same Student) distinguishable in the active_task_ids list.
        epoch_ms = int(asyncio.get_event_loop().time() * 1000)
        task_id = f"student-nim-{project_id}-{student_model_id[:8]}-{epoch_ms}"

        coro = student_nim_lifecycle.run_student_deployment_lifecycle(
            project_id=project_id,
            student_model_id=student_model_id,
            mode=mode,
            nim_endpoint_url=nim_endpoint_url,
            nim_container_image=nim_container_image,
            nim_release_version=nim_release_version,
            gpu_assignment=gpu_assignment,
            auth_mode=auth_mode,
            settings=settings,
            workspace_root=settings.WORKSPACE_ROOT,
        )
        try:
            background_manager.register(task_id, coro)
        except RuntimeError:
            # Race with another :deploy_nim call that beat us into the
            # registry — close the coroutine so we don't leak a warning.
            coro.close()
            return {"error": "deploy_in_progress"}

        return {
            "student_model_id": student_model_id,
            "nim_deployment_mode": mode,
            "serving_status": "pending",
            "task_id": task_id,
            "created_at": utc_now(),
        }


__all__ = [
    "PackagingResult",
    "check_lora_merge_readiness",
    "deploy_nim",
    "find_student_for_evaluate_job",
    "get_student_model",
    "list_student_models",
    "mark_student_quality_failed",
    "register_from_tao_terminal",
    "run_automatic_baseline_evaluation",
]

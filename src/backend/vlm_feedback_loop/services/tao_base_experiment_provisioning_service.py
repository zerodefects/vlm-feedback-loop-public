# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blueprint self-service base-experiment provisioning.

Six-step orchestrator that turns a populated TAO workspace into a workspace
with **registered** Cosmos Reason2 base experiments (or any other
``student_base`` catalog entry) without any admin intervention. The flow
is live-proven against FTMS 6.25.11/6.26.3.

Steps:

1. **Resolve targets** — read ``TAODeploymentConfig`` to confirm the workspace
   identity is bootstrapped; enumerate ``student_base`` entries from
   :data:`SEEDED_MODEL_CATALOG` (or the operator-supplied
   ``--model-config-id`` set, resolved against the first project DB).
2. **Idempotency pre-pass** — ``find_base_experiment_by_arch`` per target;
   already-registered targets land in ``already_registered`` and skip the
   rest of the chain.
3. **Write CSV + spawn subprocess** — temp-write
   ``displayName,ngc_path,network_arch,is_backbone`` rows, hand them to
   :data:`PULL_SCRIPT_PATH` which sets ``AIRGAPPED_MODE=true`` and runs
   ``python -m nvidia_tao_core.microservices.pretrained_models --use-csv``.
4. **S3 upload** — recursively stream ``stage_dir/`` (incl.
   ``ptm_metadatas.json`` + per-model checkpoint dirs) to
   ``s3://{bucket}/shared-storage/models/`` using the same SHA-256
   idempotency + 8-MiB multipart cutover the dataset upload service uses.
5. **POST ``/orgs/{org}/jobs:load_airgapped``** — fire-and-forget with the
   shared single-401-retry helper (``tao_auth.retry_once_on_401``).
   Response must report ``experiments_failed == 0``.
6. **Confirm + patch** — bounded ``find_base_experiment_by_arch`` retry
   loop (15 × 2 s ceiling) to recover the registered UUIDs; then
   :func:`patch_model_configs_across_projects` writes them across every
   project DB.

The FastAPI process **MUST NOT** import ``nvidia_tao_core``, ``transformers``,
``peft`` or ``huggingface_hub``. The subprocess driver
under ``tao_base_experiment_pull/`` owns those dependencies; this service
only orchestrates the subprocess + S3 + HTTP. Acceptance test #13
(import-boundary) verifies this with a clean ``sys.modules`` check.

``--dry-run`` short-circuits **after** step 3 succeeds: the CSV + the
subprocess run are exercised (cheap proof the operator's NGC key works
and HF reachability is OK), and the S3 upload + ``:load_airgapped`` call
are skipped. Useful to validate the pipe without paying the workspace
upload cost.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import shlex
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.model_catalog_constants import (
    HF_MODEL_PATHS,
    TAO_BASE_EXPERIMENT_DISPLAY_NAMES,
)
from vlm_feedback_loop.services.authorized_file import open_regular_file_beneath
from vlm_feedback_loop.services.hashing import sha256_stream
from vlm_feedback_loop.services.http_client import resilient_request
from vlm_feedback_loop.services.logging_config import redact_exact_secrets
from vlm_feedback_loop.services.project_service import SEEDED_MODEL_CATALOG
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret
from vlm_feedback_loop.services.subprocess_utils import communicate_with_timeout
from vlm_feedback_loop.services.tao_auth import retry_once_on_401, tao_base_url
from vlm_feedback_loop.services.tao_bootstrap_service import (
    iter_project_dirs,
    patch_model_configs_across_projects,
)
from vlm_feedback_loop.services.tao_dataset_upload_service import (
    MULTIPART_PART_SIZE_BYTES,
    MULTIPART_THRESHOLD_BYTES,
    S3ClientProtocol,
    already_uploaded,
    build_s3_client,
    do_multipart_put,
    do_single_put,
)
from vlm_feedback_loop.services.tao_workspace_service import (
    find_base_experiment_by_arch,
    read_tao_deployment_config,
    tao_auth_headers,
)

logger = logging.getLogger(
    "vlm_feedback_loop.services.tao_base_experiment_provisioning_service"
)


# ── Module-level constants ───────────────────────────────────────────────────


_HELPER_DIR = Path(__file__).resolve().parent.parent / "tao_base_experiment_pull"
PULL_SCRIPT_PATH: Path = _HELPER_DIR / "pull_base_experiments.py"
PULL_REQUIREMENTS_PATH: Path = _HELPER_DIR / "requirements.txt"
DEFAULT_SHARED_REGISTRY_PREFIX = "shared-storage/models"
DEFAULT_SUBPROCESS_TIMEOUT_S: float = 1800.0
DEFAULT_INDEX_SETTLE_RETRIES: int = 15
DEFAULT_INDEX_SETTLE_INTERVAL_S: float = 2.0

# Self-service HF auto-pull covers every entry in the canonical
# ``model_catalog_constants.HF_MODEL_PATHS`` roster. Both families are
# gated (the worker authenticates with the operator's HF token and the
# accepted licenses); the Cosmos 3 reasoning-tower repos are just larger
# (~17 GB Nano / ~60 GB Super). Self-service pull handles them the same
# way as Cosmos Reason2 — the gated pull is empirically proven. Operators
# who want to provision only a subset can pass explicit
# ``student_base_model_config_ids`` instead of the default enumerate-all
# behaviour. (Admin-managed registration via ``tao bootstrap
# --admin-managed --base-experiment-id-cosmos3-{nano,super}`` remains
# available as an alternative.)


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class _Target:
    """One base-experiment provisioning target.

    ``model_config_id`` is the canonical project-DB row id; when
    enumeration starts from ``SEEDED_MODEL_CATALOG`` (i.e. no operator
    --model-config-id flag) and there are no project DBs yet, this
    falls back to the canonical ``model_name``.
    """

    model_config_id: str
    model_name: str
    hf_path: str
    display_name: str


@dataclass
class ProvisioningResult:
    """Aggregate outcome of a ``provision_base_experiments`` call.

    ``registered`` lists newly-registered identifiers (model_config_id
    or model_name fallback); ``already_registered`` lists targets the
    pre-pass found in TAO; ``failed`` carries ``(target, error)``
    tuples for any partial failure. The CLI exits non-zero when
    ``failed`` is non-empty.

    ``uuid_by_model_name`` exposes the registered TAO base-experiment
    UUIDs keyed by canonical lowercase ``model_name``. Populated for
    both newly-registered AND already-registered targets so callers
    that create future projects (the live smoke) can patch their seeded
    ModelConfig rows without re-querying TAO.
    """

    registered: list[str] = field(default_factory=list[str])
    already_registered: list[str] = field(default_factory=list[str])
    failed: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    uuid_by_model_name: dict[str, str] = field(default_factory=dict[str, str])


# ── Helpers (sync) ───────────────────────────────────────────────────────────


def _resolve_targets_default() -> list[_Target]:
    """Enumerate every ``student_base`` entry the Blueprint knows how to pull.

    Used when the caller passes no explicit target selector — the standalone
    CLI default. Validation harnesses select only the base model exercised by
    their current run. Filters
    ``SEEDED_MODEL_CATALOG`` to entries whose ``model_name`` is in
    ``HF_MODEL_PATHS`` so callers cannot accidentally schedule a
    pull for a catalog row that has no Hugging Face mapping.
    """
    targets: list[_Target] = []
    for entry in SEEDED_MODEL_CATALOG:
        roles_raw: Any = entry.get("eligible_roles") or []
        roles: list[Any] = (
            cast("list[Any]", roles_raw) if isinstance(roles_raw, list) else []
        )
        if "student_base" not in roles:
            continue
        model_name: str = str(entry["model_name"]).strip().lower()
        hf_path = HF_MODEL_PATHS.get(model_name)
        if not hf_path:
            continue
        display = TAO_BASE_EXPERIMENT_DISPLAY_NAMES.get(model_name, model_name)
        targets.append(
            _Target(
                model_config_id=model_name,
                model_name=model_name,
                hf_path=hf_path,
                display_name=display,
            )
        )
    return targets


def _resolve_targets_from_ids(
    workspace_root: Path,
    model_config_ids: list[str],
    *,
    project_id: str | None = None,
) -> tuple[list[_Target], list[tuple[str, str]]]:
    """Resolve operator-supplied ``--model-config-id`` flags to ``_Target``s.

    The UI supplies project-local UUIDs, so ``project_id`` selects that exact
    database. The standalone CLI omits it and retains the historical
    first-project lookup. Returns ``(targets, failed)`` so the caller can
    record per-id resolution failures without aborting the whole run.
    """
    project_dir = (
        workspace_root / "projects" / project_id
        if project_id is not None
        else next(iter_project_dirs(workspace_root), None)
    )
    failed: list[tuple[str, str]] = []
    targets: list[_Target] = []
    if project_dir is None or not (project_dir / "project.db").is_file():
        return (
            [],
            [
                (
                    mcid,
                    (
                        f"project {project_id!r} does not have a project DB"
                        if project_id is not None
                        else "no project DB exists yet; create a project first"
                    ),
                )
                for mcid in model_config_ids
            ],
        )

    engine = open_project_db(project_dir)
    with Session(engine) as session:
        rows: dict[str, ModelConfig] = {}
        for row in (
            session.query(ModelConfig)
            .filter(ModelConfig.model_config_id.in_(model_config_ids))
            .all()
        ):
            rows[row.model_config_id] = row

    for mcid in model_config_ids:
        row = rows.get(mcid)
        if row is None:
            failed.append((mcid, f"ModelConfig {mcid!r} not found"))
            continue
        roles = row.eligible_roles or []
        if isinstance(roles, str):
            try:
                roles = json.loads(roles)
            except json.JSONDecodeError:
                roles = []
        if "student_base" not in roles:
            failed.append(
                (mcid, f"ModelConfig {mcid!r} does not have the student_base role")
            )
            continue
        model_name = row.model_name.strip().lower()
        hf_path = HF_MODEL_PATHS.get(model_name)
        if not hf_path:
            failed.append(
                (
                    mcid,
                    f"no Hugging Face path mapping for {model_name!r}; "
                    "self-service supports the Cosmos Reason2 and "
                    "Cosmos 3 (reasoner) families",
                )
            )
            continue
        display = TAO_BASE_EXPERIMENT_DISPLAY_NAMES.get(model_name, model_name)
        targets.append(
            _Target(
                model_config_id=mcid,
                model_name=model_name,
                hf_path=hf_path,
                display_name=display,
            )
        )
    return targets, failed


def _write_csv(path: Path, targets: list[_Target]) -> None:
    """Write the operator CSV the subprocess driver consumes.

    Format (restated by acceptance test #1):

        displayName,ngc_path,network_arch,is_backbone
        Cosmos Reason2 2B,hf_model://nvidia/Cosmos-Reason2-2B,cosmos-rl,True
        ...

    LF line endings, no quoting, ``True`` capitalised (TAO's
    ``pretrained_models`` is case-sensitive).
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["displayName", "ngc_path", "network_arch", "is_backbone"])
        for t in targets:
            w.writerow([t.display_name, f"hf_model://{t.hf_path}", "cosmos-rl", "True"])


def _name_substring_candidates_for(target: _Target) -> tuple[str, ...]:
    """Lower-cased substrings tried in order against the TAO experiment name.

    TAO FTMS registers the experiment using the CSV's ``displayName``
    (with spaces) but historical evidence varies — the live smoke uses
    the hyphenated form ``"cosmos-reason-2-2b"``, the bundled CSV writes
    spaces (``"Cosmos Reason2 2B"``), and intermediate normalisation has
    been observed to swap hyphens and spaces. Try every reasonable
    canonicalisation; the first hit wins.
    """
    hf_tail = target.hf_path.split("/")[-1].lower()  # "cosmos-reason2-2b"
    display = target.display_name.lower()  # "cosmos reason2 2b"
    return (
        hf_tail,  # "cosmos-reason2-2b"
        display,  # "cosmos reason2 2b"
        display.replace(" ", "-"),  # "cosmos-reason2-2b"
        hf_tail.replace("reason2-", "reason-2-"),  # "cosmos-reason-2-2b"
        display.replace("reason2", "reason-2"),  # "cosmos reason-2 2b"
    )


async def _resolve_uuid_for_target(
    settings: Settings,
    target: _Target,
    *,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> str | None:
    """Try every name-substring candidate; return the first matching UUID."""
    for substring in _name_substring_candidates_for(target):
        exp = await find_base_experiment_by_arch(
            settings,
            network_arch="cosmos-rl",
            name_substring=substring,
            _transport=_transport,
        )
        candidate = (exp or {}).get("id") or (exp or {}).get("experiment_id")
        if candidate:
            return str(candidate)
    return None


# ── Subprocess driver ────────────────────────────────────────────────────────


async def _default_subprocess_runner(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    """Spawn the packaged TAO base-experiment pull helper.

    Returns ``{"ok": bool, "stdout": str, "stderr": str, "returncode": int}``.
    Mirrors ``student_model_service._run_merge_lora_subprocess`` so the
    subprocess discipline is identical across the LoRA merge and the
    base-experiment pull.
    """
    logger.info(
        "pull_base_experiments: spawning %s",
        " ".join(shlex.quote(c) for c in cmd),
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError) as exc:
        return {
            "ok": False,
            "stdout": "",
            "stderr": _redact_subprocess_secrets(
                f"failed to spawn pull_base_experiments subprocess: {exc}", env
            ),
            "returncode": -1,
        }

    try:
        stdout_b, stderr_b = await communicate_with_timeout(
            proc,
            timeout_s=timeout_s,
        )
    except TimeoutError:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"pull_base_experiments subprocess timed out after {timeout_s}s",
            "returncode": -2,
        }

    stdout = (stdout_b or b"").decode("utf-8", errors="replace")
    stderr = (stderr_b or b"").decode("utf-8", errors="replace")
    return {
        "ok": proc.returncode == 0,
        "stdout": _redact_subprocess_secrets(stdout, env),
        "stderr": _redact_subprocess_secrets(stderr, env),
        "returncode": proc.returncode if proc.returncode is not None else -1,
    }


def _redact_subprocess_secrets(text: str, env: dict[str, str]) -> str:
    """Redact the private credential values passed to the pull child."""
    return redact_exact_secrets(
        text,
        (
            env.get("PTM_API_KEY"),
            env.get("HF_TOKEN"),
            env.get("HUGGING_FACE_HUB_TOKEN"),
        ),
    )


def _build_subprocess_command(
    csv_path: Path,
    stage_dir: Path,
    *,
    use_isolated_helper: bool,
) -> list[str]:
    """The NGC key travels ONLY via env (``PTM_API_KEY`` in
    :func:`_build_subprocess_env`) — never on argv, which is readable by
    any co-tenant process via ``ps`` / ``/proc/<pid>/cmdline`` and lands
    in the spawn log line. Same discipline as
    ``student_model_service._run_merge_lora_subprocess`` (HF_TOKEN)."""
    script_args = [
        str(PULL_SCRIPT_PATH),
        "--csv",
        str(csv_path),
        "--shared-folder-path",
        str(stage_dir),
    ]
    if not use_isolated_helper:
        return [sys.executable, *script_args]
    # TAO's packaging stack has pins that intentionally differ from the
    # backend. Let uv cache and run an isolated helper environment instead of
    # mutating the FastAPI venv. ``--no-project`` also prevents uv from syncing
    # the Blueprint lockfile before launching the helper.
    return [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with-requirements",
        str(PULL_REQUIREMENTS_PATH),
        "--python",
        sys.executable,
        "--",
        "python",
        *script_args,
    ]


def _build_subprocess_env(
    stage_dir: Path, ngc_key: str, *, hf_token: str | None = None
) -> dict[str, str]:
    env = dict(os.environ)
    env["AIRGAPPED_MODE"] = "true"
    if ngc_key:
        env["PTM_API_KEY"] = ngc_key
    if not env.get("HF_HOME"):
        env["HF_HOME"] = str(stage_dir / "_hf_cache")
    # The Cosmos Reason2 repos on Hugging Face are gated — operators MUST
    # supply an HF token whose account has accepted the licence; without
    # one the pull fails with HTTP 401 on `config.json`.
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    return env


# ── S3 upload (sync) ─────────────────────────────────────────────────────────


def _upload_stage_tree_sync(
    s3_client: S3ClientProtocol,
    *,
    bucket: str,
    stage_dir: Path,
    multipart_threshold_bytes: int = MULTIPART_THRESHOLD_BYTES,
    multipart_part_size_bytes: int = MULTIPART_PART_SIZE_BYTES,
) -> None:
    """Walk ``stage_dir`` recursively and upload every regular file.

    Reuses :func:`already_uploaded`, :func:`do_single_put`, and
    :func:`do_multipart_put` from ``tao_dataset_upload_service`` so this
    service shares the same metadata key (``"dataset-export-sha256"``)
    and multipart cutover threshold. Each file is opened once beneath
    the disposable stage root; hashing, sizing, and upload stay bound to
    that descriptor. Re-running skips files whose SHA-256 already matches
    the remote.
    """
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"stage directory not found: {stage_dir}")

    for file_path in sorted(stage_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(stage_dir)
        # HF_HOME lives inside the disposable stage so the helper never
        # pollutes the operator's global cache. It is downloader scratch space,
        # not part of TAO's air-gapped registry layout. Uploading it duplicates
        # large shards under an unusable ``_hf_cache`` key prefix before the
        # intended ``huggingface/...`` model tree.
        if relative_path.parts[0] == "_hf_cache":
            continue
        rel = relative_path.as_posix()
        key = f"{DEFAULT_SHARED_REGISTRY_PREFIX}/{rel}"
        with open_regular_file_beneath(file_path, stage_dir) as opened_file:
            with opened_file.open_binary() as stream:
                sha256 = sha256_stream(stream)
            if already_uploaded(s3_client, bucket=bucket, key=key, sha256=sha256):
                logger.info("skip already-uploaded %s (sha256=%s)", key, sha256[:12])
                continue
            if opened_file.stat_result.st_size > multipart_threshold_bytes:
                do_multipart_put(
                    s3_client,
                    bucket=bucket,
                    key=key,
                    opened_file=opened_file,
                    sha256=sha256,
                    part_size_bytes=multipart_part_size_bytes,
                )
            else:
                do_single_put(
                    s3_client,
                    bucket=bucket,
                    key=key,
                    opened_file=opened_file,
                    sha256=sha256,
                )


# ── POST :load_airgapped ─────────────────────────────────────────────────────


async def _post_load_airgapped(
    settings: Settings,
    *,
    tao_workspace_id: str,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Trigger TAO to register the uploaded checkpoints as base experiments.

    Single 401 retry pattern via the shared
    :func:`tao_auth.retry_once_on_401`: on first 401, the cached JWT is
    invalidated, the auth header refreshed, and the POST retried once.
    Any other error class (timeout, 4xx other than 401, 5xx) is fatal.

    ``tao_workspace_id`` is sourced from the deployment.db
    ``TAODeploymentConfig`` record, the sole source of workspace identity.

    Raises :class:`RuntimeError` on transport failure or
    ``experiments_failed > 0``. Returns the response body dict on success.
    """
    base = tao_base_url(settings)
    url = f"{base}/orgs/{settings.TAO_ORG_NAME}/jobs:load_airgapped"
    body = {"workspace_id": tao_workspace_id}

    headers = await tao_auth_headers(settings)
    result = await resilient_request(
        "POST",
        url,
        deadline_s=float(settings.HTTP_DEADLINE_BACKGROUND_S),
        max_retries=1,
        headers=headers,
        json_body=body,
        _transport=_transport,
    )
    result = await retry_once_on_401(
        result,
        method="POST",
        url=url,
        settings=settings,
        deadline_s=float(settings.HTTP_DEADLINE_BACKGROUND_S),
        max_retries=1,
        json_body=body,
        _transport=_transport,
        reraise_auth_error=True,
    )
    if result.error_class is not None:
        raise RuntimeError(f"load_airgapped failed: {result.error_detail}")
    body_any: Any = result.body
    body_obj: dict[str, Any] = (
        cast("dict[str, Any]", body_any) if isinstance(body_any, dict) else {}
    )
    failed = int(body_obj.get("experiments_failed", 0))
    if failed > 0:
        raise RuntimeError(
            f"load_airgapped reported experiments_failed={failed} "
            f"(loaded={body_obj.get('experiments_loaded')})"
        )
    return body_obj


# ── Public entry point ───────────────────────────────────────────────────────


async def provision_base_experiments(
    settings: Settings,
    student_base_model_config_ids: list[str] | None = None,
    *,
    target_model_names: list[str] | None = None,
    project_id: str | None = None,
    hf_token: str | None = None,
    subprocess_timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S,
    use_isolated_helper: bool = True,
    _transport: httpx.AsyncBaseTransport | None = None,
    _subprocess_runner: Any = None,
    _s3_client_factory: Any = None,
    _stage_root: Path | None = None,
    _dry_run: bool = False,
) -> ProvisioningResult:
    """Self-service Blueprint base-experiment provisioning.

    ``target_model_names`` lets a validation harness provision only the
    canonical model names it will exercise. It is mutually exclusive with
    project-local ``student_base_model_config_ids``; omitting both preserves
    the standalone CLI's enumerate-all behavior.

    Idempotent: rerunning against a workspace whose target experiments
    are already registered is a no-op (returns ``already_registered``,
    no subprocess, no S3 traffic, no ``:load_airgapped`` POST).

    Test seams (underscore-prefixed kwargs):

    - ``_transport``: ``httpx.MockTransport`` for TAO HTTP calls.
    - ``_subprocess_runner``: replaces :func:`_default_subprocess_runner`
      so unit tests can fake the pull-script's stdout/exit code without
      requiring ``nvidia-tao-core`` at test collection time.
    - ``_s3_client_factory``: callable returning an
      :class:`S3ClientProtocol`-shaped fake.
    - ``_stage_root``: forces the temp stage directory (otherwise
      :class:`tempfile.TemporaryDirectory`).
    - ``_dry_run``: stop after the subprocess succeeds; skip S3 upload
      and ``:load_airgapped``. Returns the work list in
      ``failed`` with ``"dry_run: not uploaded"`` so the caller can
      inspect what would have happened.
    """
    runner = _subprocess_runner or _default_subprocess_runner
    workspace_root = Path(settings.WORKSPACE_ROOT) if settings.WORKSPACE_ROOT else None

    result = ProvisioningResult()

    if workspace_root is None:
        result.failed.append(("workspace", "WORKSPACE_ROOT is not configured"))
        return result

    tao_cfg = read_tao_deployment_config(settings)
    if tao_cfg is None or not tao_cfg.tao_workspace_id:
        result.failed.append(
            (
                "workspace",
                "TAO workspace identity is not bootstrapped in deployment.db; run "
                "`vlm-feedback-loop tao-bootstrap` first.",
            )
        )
        return result

    # 1) Resolve targets
    if student_base_model_config_ids and target_model_names is not None:
        result.failed.append(
            (
                "targets",
                "student_base_model_config_ids and target_model_names are "
                "mutually exclusive",
            )
        )
        return result
    if student_base_model_config_ids:
        targets, resolution_failures = _resolve_targets_from_ids(
            workspace_root,
            student_base_model_config_ids,
            project_id=project_id,
        )
        result.failed.extend(resolution_failures)
    else:
        targets = _resolve_targets_default()
        if target_model_names is not None:
            requested_names = [name.strip().lower() for name in target_model_names]
            targets_by_name = {target.model_name: target for target in targets}
            selected_targets: list[_Target] = []
            selected_names: set[str] = set()
            for model_name in requested_names:
                target = targets_by_name.get(model_name)
                if target is None:
                    result.failed.append(
                        (
                            model_name or "<empty>",
                            f"unknown self-service base model {model_name!r}",
                        )
                    )
                elif model_name not in selected_names:
                    selected_targets.append(target)
                    selected_names.add(model_name)
            targets = selected_targets
    if not targets:
        if not result.failed:
            result.failed.append(
                (
                    "targets",
                    "no provisioning targets resolved; "
                    "check seeded student_base catalog or --model-config-id flags",
                )
            )
        return result

    # 2) Idempotency pre-pass
    work_list: list[_Target] = []
    for t in targets:
        try:
            uuid = await _resolve_uuid_for_target(settings, t, _transport=_transport)
        except Exception as exc:
            result.failed.append((t.model_config_id, f"discovery failed: {exc}"))
            continue
        if uuid:
            result.already_registered.append(t.model_config_id)
            result.uuid_by_model_name[t.model_name] = uuid
        else:
            work_list.append(t)

    if not work_list:
        # Every target was already registered. Project DBs may still be
        # missing the UUID (e.g. fresh projects created since the prior
        # provisioning run, or a prior run that crashed between
        # ``find_base_experiment_by_arch`` and ``patch_model_configs``).
        # Re-run the patch idempotently using the pre-pass discoveries.
        if result.uuid_by_model_name:
            await asyncio.to_thread(
                patch_model_configs_across_projects,
                workspace_root,
                base_experiment_map=dict(result.uuid_by_model_name),
            )
        return result

    bucket = tao_cfg.tao_workspace_bucket
    if not bucket:
        result.failed.extend(
            (t.model_config_id, "TAODeploymentConfig.tao_workspace_bucket is not set")
            for t in work_list
        )
        return result

    # 3) Write CSV + spawn subprocess
    stage_ctx: tempfile.TemporaryDirectory[str] | None = None
    if _stage_root is None:
        stage_ctx = tempfile.TemporaryDirectory(prefix="vlm-self-service-")
        stage_dir = Path(stage_ctx.name)
    else:
        stage_dir = _stage_root
        stage_dir.mkdir(parents=True, exist_ok=True)

    try:
        csv_path = stage_dir / "pretrained_models.csv"
        _write_csv(csv_path, work_list)
        ngc_key = (get_effective_secret("TAO_API_KEY", settings) or "").strip()
        # HF_TOKEN precedence: explicit kwarg → settings.HF_TOKEN (when
        # configured) → process env. The subprocess driver itself also
        # honours `HF_TOKEN` if we re-export it.
        effective_hf_token = (
            hf_token
            or settings.HF_TOKEN
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or None
        )
        if effective_hf_token:
            effective_hf_token = effective_hf_token.strip() or None
        cmd = _build_subprocess_command(
            csv_path,
            stage_dir,
            use_isolated_helper=use_isolated_helper,
        )
        env = _build_subprocess_env(stage_dir, ngc_key, hf_token=effective_hf_token)
        sub_outcome = await runner(cmd, env=env, timeout_s=subprocess_timeout_s)
        # Keep this defense at the orchestration boundary as well as in the
        # default runner: an alternate runner must not be able to return an
        # echoed private environment value into persisted failure state.
        for stream_name in ("stdout", "stderr"):
            stream = sub_outcome.get(stream_name)
            if isinstance(stream, str):
                sub_outcome[stream_name] = _redact_subprocess_secrets(stream, env)
        if not sub_outcome.get("ok"):
            err = (sub_outcome.get("stderr") or "").strip()[:1024]
            for t in work_list:
                result.failed.append(
                    (
                        t.model_config_id,
                        f"subprocess: rc={sub_outcome.get('returncode')}: {err}",
                    )
                )
            return result

        # 4) Dry-run short-circuit
        if _dry_run:
            for t in work_list:
                result.failed.append((t.model_config_id, "dry_run: not uploaded"))
            return result

        # 5) S3 upload
        try:
            if _s3_client_factory is None:
                s3_client = await asyncio.to_thread(build_s3_client, tao_cfg)
            else:
                s3_client = _s3_client_factory()
        except Exception as exc:
            for t in work_list:
                result.failed.append(
                    (t.model_config_id, f"build_s3_client failed: {exc}")
                )
            return result

        try:
            await asyncio.to_thread(
                _upload_stage_tree_sync,
                s3_client,
                bucket=bucket,
                stage_dir=stage_dir,
            )
        except Exception as exc:
            for t in work_list:
                result.failed.append((t.model_config_id, f"S3 upload failed: {exc}"))
            return result

        # 6) POST :load_airgapped
        try:
            await _post_load_airgapped(
                settings,
                tao_workspace_id=tao_cfg.tao_workspace_id,
                _transport=_transport,
            )
        except Exception as exc:
            for t in work_list:
                result.failed.append((t.model_config_id, f"{exc}"))
            return result

        # 7) Confirm + bounded settle
        resolved: dict[str, str] = {}
        for t in work_list:
            uuid: str | None = None
            for _ in range(DEFAULT_INDEX_SETTLE_RETRIES):
                uuid = await _resolve_uuid_for_target(
                    settings, t, _transport=_transport
                )
                if uuid:
                    break
                await asyncio.sleep(DEFAULT_INDEX_SETTLE_INTERVAL_S)
            if uuid:
                resolved[t.model_name] = uuid
            else:
                result.failed.append(
                    (t.model_config_id, f"no UUID returned for {t.model_name}")
                )

        # 8) Patch project DBs.
        #
        # Merge any UUIDs discovered in the idempotency pre-pass (step 2)
        # with the freshly-resolved ones so already-registered targets
        # also propagate into project DBs that may have been created since
        # the prior provisioning run.
        if resolved:
            result.registered = [
                t.model_config_id for t in work_list if t.model_name in resolved
            ]
            for model_name, uuid in resolved.items():
                result.uuid_by_model_name.setdefault(model_name, uuid)
        if result.uuid_by_model_name:
            await asyncio.to_thread(
                patch_model_configs_across_projects,
                workspace_root,
                base_experiment_map=dict(result.uuid_by_model_name),
            )
        return result
    finally:
        if stage_ctx is not None:
            stage_ctx.cleanup()


__all__ = [
    "DEFAULT_INDEX_SETTLE_INTERVAL_S",
    "DEFAULT_INDEX_SETTLE_RETRIES",
    "DEFAULT_SHARED_REGISTRY_PREFIX",
    "DEFAULT_SUBPROCESS_TIMEOUT_S",
    "PULL_REQUIREMENTS_PATH",
    "PULL_SCRIPT_PATH",
    "ProvisioningResult",
    "provision_base_experiments",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO polling worker + chain advancement.

A single long-lived background task (mirrors :mod:`clip_embedding_service`)
that periodically scans every project DB for non-terminal TAOJob rows and
drives them forward. Per-status polling cadences:

  * ``submitted`` / ``queued``  — poll every 30–60s (``TAO_POLL_MIN_INTERVAL_S``).
  * ``running`` / ``paused``    — poll every 60–180s (``TAO_POLL_MIN_INTERVAL_RUNNING_S``).
  * terminal                     — never re-polled.
  * ``submitting``              — handled by the submission protocol itself, skipped here.
  * ``not_started``             — advanced via chain logic, not polling.

On each status transition the worker emits ``tao_job_progress``. Terminal
success emits ``tao_job_completed`` + chain-advances (submits chain_seq+1
if it exists). Terminal failure emits ``run_failed``, halts the chain
(marks remaining ``not_started`` siblings as ``failed+chain_halted_reason``),
and triggers cross-chain advancement (submits the next chain's seq=1 train
job if the current chain has no active jobs left).

Artifact retrieval on succeeded:
``GET :list_files`` enumerates workspace keys, then boto3 ``GetObject``
pulls the bytes directly from the workspace S3 bucket. For
``train``/``quantize`` the latest ``safetensors/epoch_<N>/`` directory
is mirrored flat under ``{project_dir}/artifacts/tao_jobs/{tao_job_id}/``
(NIM-loadable HF shape). For ``evaluate`` the single
``evaluate_results.tar.gz`` is downloaded + extracted, and per-sample
predictions are translated into a synthesized
``per_sample_predictions`` JSON file. Logs come from
``GET :logs`` → ``logs_ref``.

SSE events flow only after persistent state transitions commit.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.services import tao_job_service
from vlm_feedback_loop.services.background import background_manager
from vlm_feedback_loop.services.http_client import HttpResult, resilient_request
from vlm_feedback_loop.services.project_service import (
    get_project_engine,
    project_dir_path,
    projects_root,
)
from vlm_feedback_loop.services.sse import sse_manager
from vlm_feedback_loop.services.tao_auth import tao_base_url, tao_preflight
from vlm_feedback_loop.services.tao_job_service import (
    CANONICAL_STATUSES,
    TERMINAL_STATUSES,
    can_transition,
    find_suite_for_chain,
)
from vlm_feedback_loop.services.tao_workspace_service import read_tao_deployment_config

logger = logging.getLogger("vlm_feedback_loop.tao_polling_service")

# Polling task registration id for background_manager.
_POLL_TASK_ID = "tao-polling"

# Statuses the polling worker actively inspects.
_POLLABLE_STATUSES: frozenset[str] = frozenset(
    {"submitted", "queued", "running", "paused"}
)

# chain_halted_reason prefix persisted on evaluate jobs the auto-eval
# policy deliberately skips (doomed-on-TAO evaluates). The suite roll-up
# treats canceled rows carrying this prefix as success-equivalent.
AUTO_SKIP_REASON_PREFIX = "auto-skip:"


def _is_auto_skip_reason(reason: str | None) -> bool:
    return (reason or "").startswith(AUTO_SKIP_REASON_PREFIX)


# Project-open warning throttle.
# When a project is locked by another process (e.g., a build-mode
# ``scripts/rps_e2e.py`` running in-process while the dev backend's
# polling tick scans the same workspace), every tick emits an identical
# ``tao_polling: cannot open project <id> (ProjectLockedError: ...)``
# WARN line. Over a 4-hour rental with N orphan projects this produces
# 30 lines/min × N × 240 min — thousands of lines that drown the log.
#
# Throttle: WARN on first occurrence per project; thereafter log DEBUG
# until ``_PROJECT_OPEN_WARNING_THROTTLE_S`` elapses, then WARN again.
# DEBUG entries are still emitted at every tick so operators with
# LOG_LEVEL=debug retain full granularity for diagnostics.
_PROJECT_OPEN_WARNING_THROTTLE_S: float = 300.0  # 5 min
_project_open_warning_last_at: dict[str, float] = {}


def _log_project_open_failure(
    project_id: str, exc: BaseException, *, now: float | None = None
) -> None:
    """Emit a throttled WARN/DEBUG for an inability to open a project DB.

    First occurrence (or first after the throttle window) → WARN.
    Subsequent occurrences within the window → DEBUG (keeps the trail
    for ``LOG_LEVEL=debug`` operators without flooding the default INFO
    stream). Per-project state lives in
    :data:`_project_open_warning_last_at`; safe to reset between tests.
    """
    import time

    current = now if now is not None else time.monotonic()
    last = _project_open_warning_last_at.get(project_id)
    should_warn = last is None or (current - last) >= _PROJECT_OPEN_WARNING_THROTTLE_S
    # Observability: ``str(exc) or "(no message)"`` so empty-string
    # exceptions still produce actionable signal — same idiom as
    # ``services/project_service.py``, ``services/local_nim_service.py``,
    # and ``services/background.py::_on_task_done``.
    msg = "tao_polling: cannot open project %s (%s: %s)"
    args = (project_id, type(exc).__name__, str(exc) or "(no message)")
    if should_warn:
        logger.warning(msg, *args)
        _project_open_warning_last_at[project_id] = current
    else:
        logger.debug(msg, *args)


# ── Artifact / log retrieval (mockable) ────────────────────────────────────


def _tao_artifact_cache_dir(
    workspace_root: str,
    project_id: str,
    tao_job_id: str,
) -> Path:
    """Resolve the per-TAOJob local artifact cache directory.

    Layout: ``{workspace_root}/projects/{project_id}/artifacts/tao_jobs/{tao_job_id}/``.
    """
    return (
        project_dir_path(workspace_root, project_id)
        / "artifacts"
        / "tao_jobs"
        / tao_job_id
    )


def _artifact_cache_path(cache_dir: Path, relative_name: str) -> Path:
    """Map a remote artifact name to a path contained by ``cache_dir``."""
    parts = relative_name.split("/")
    if (
        not relative_name
        or relative_name.startswith("/")
        or "\\" in relative_name
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("TAO artifact path is outside its local cache")
    return cache_dir.joinpath(*parts)


async def _list_tao_job_files(
    tao_external_job_id: str,
    *,
    settings: Settings,
) -> dict[str, Any]:
    """Enumerate the workspace keys for a finished TAO job via ``:list_files``.

    Returns ``{"success": bool, "keys": list[str], "error": str | None}``.
    The keys are workspace-relative S3 keys (e.g.
    ``results/<job>/safetensors/epoch_1/config.json``) — feed them to
    the workspace S3 client to materialize bytes.
    """
    headers, preflight_error = await tao_preflight(settings)
    if headers is None:
        return {"success": False, "keys": [], "error": preflight_error}
    url = (
        f"{tao_base_url(settings)}/orgs/{settings.TAO_ORG_NAME}"
        f"/jobs/{tao_external_job_id}:list_files"
    )
    result = await resilient_request(
        "GET",
        url,
        deadline_s=float(settings.HTTP_DEADLINE_BACKGROUND_S),
        max_retries=settings.HTTP_MAX_RETRIES,
        headers=headers,
    )
    if result.error_class is not None:
        return {
            "success": False,
            "keys": [],
            "error": f"list_files failed: {result.error_detail}",
        }
    body: Any = result.body
    keys = (
        [k for k in cast("list[Any]", body) if isinstance(k, str)]
        if isinstance(body, list)
        else []
    )
    return {"success": True, "keys": keys, "error": None}


def _select_hf_checkpoint_keys(keys: list[str]) -> tuple[list[str], str | None]:
    """Pick the merged-HF checkpoint keys from a TAO job's full key list.

    cosmos-rl emits checkpoints in two distinct workspace layouts
    depending on the action that produced them:

    - **train** — nested under ``results/<job>/<timestamp>/safetensors/
      epoch_<N>/<file>``. Multiple epochs may be retained per
      ``train.ckpt.max_keep``; the LATEST epoch wins.
    - **quantize** — flat at the job root:
      ``results/<job>/<file>`` directly, with no ``safetensors/
      epoch_<N>/`` nesting. cosmos-rl-quantize is single-shot
      (no epochs) so it skips the per-epoch directory tree and writes
      the merged HF checkpoint flat. Observed live on an 8B
      FP8_DYNAMIC quantize: the listing carried the
      ``results/<job>/model-*-of-N.safetensors`` shards from the
      moment TAO marked Done, but the train-shape glob silently
      returned ``([], None)`` and the empty-listing retry loop fired
      through its full attempt budget thinking the upload was racing.

    Resolution order: try the train shape first; if it finds zero
    epoch dirs, fall back to the quantize-flat shape. Quantize
    detection is by presence of standard HF top-level filenames
    (``config.json`` + at least one ``model*.safetensors`` shard)
    directly under a single ``results/<job>/`` prefix.

    Returns ``(filtered_keys, common_prefix)`` — ``common_prefix`` is
    the leading path that gets stripped so the resulting cache dir
    layout is NIM-loadable HF (config.json + shards + tokenizer at
    the root). Returns ``([], None)`` when no checkpoint dir is
    present in either shape.
    """
    # ── Shape A (train): …/safetensors/epoch_<N>/<file>
    epoch_dirs: dict[str, list[str]] = {}
    for k in keys:
        marker = "/safetensors/epoch_"
        if marker in k:
            tail = k.split(marker, 1)[1]
            # tail = "<N>/<file…>"
            num, sep, _rest = tail.partition("/")
            if not sep:
                continue
            try:
                int(num)
            except ValueError:
                continue
            prefix = k[: k.index(marker) + len(marker)] + num + "/"
            epoch_dirs.setdefault(prefix, []).append(k)
    if epoch_dirs:
        best_prefix = max(
            epoch_dirs.keys(),
            key=lambda p: int(p.rsplit("_", 1)[1].rstrip("/")),
        )
        return epoch_dirs[best_prefix], best_prefix

    # ── Shape B (quantize): flat at results/<job>/<file>
    # Group keys by the leading ``results/<job>/`` prefix and check
    # whether each group has the canonical HF marker files.
    flat_dirs: dict[str, list[str]] = {}
    for k in keys:
        if not k.startswith("results/"):
            continue
        # Strip "results/" and take the next path segment as the job dir.
        rest = k[len("results/") :]
        head, sep, tail = rest.partition("/")
        if not sep or "/" in tail:
            # Either no slash after the job_id, or the tail contains
            # additional directory nesting — that's the train shape's
            # ``<timestamp>/safetensors/epoch_<N>/<file>`` tree, which
            # this fallback doesn't claim. Skip.
            continue
        prefix = f"results/{head}/"
        flat_dirs.setdefault(prefix, []).append(k)

    for prefix, group in flat_dirs.items():
        names = {k[len(prefix) :] for k in group}
        has_config = "config.json" in names
        has_shards = any(
            n.endswith(".safetensors") and n.startswith("model") for n in names
        )
        if has_config and has_shards:
            # Quantize-output recognised. Caller will mirror the
            # group flat under ``cache_dir/<file>`` matching the
            # NIM-loadable HF directory contract.
            return group, prefix

    return [], None


def _build_workspace_s3_client(settings: Settings) -> Any:
    """Build a boto3 S3 client for the workspace bucket.

    Endpoint URL comes from ``TAODeploymentConfig`` in ``deployment.db``
    (the canonical source). S3 credentials are secrets and live in
    ``.env``. The not-configured precheck stays here (this path raises an
    actionable ``RuntimeError``; the shared factory's own checks raise
    ``ValueError`` and never trip after the precheck); construction
    delegates to :func:`tao_dataset_upload_service.build_s3_client`.
    """
    from vlm_feedback_loop.services.tao_dataset_upload_service import build_s3_client

    cfg = read_tao_deployment_config(settings)
    if cfg is None or not (
        cfg.tao_workspace_s3_endpoint_url_external
        and settings.TAO_WORKSPACE_S3_ACCESS_KEY
        and settings.TAO_WORKSPACE_S3_SECRET_KEY
    ):
        raise RuntimeError(
            "Workspace S3 not fully configured: endpoint comes from "
            "deployment.db.tao_deployment_configs.tao_workspace_s3_endpoint_url_external "
            "(run `vlm-feedback-loop tao-bootstrap` to populate); "
            "credentials TAO_WORKSPACE_S3_ACCESS_KEY / _SECRET_KEY live in .env."
        )
    return build_s3_client(cfg, settings=settings, region_name="us-east-1")


async def _download_workspace_s3_object(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    cache_dir: Path,
    relative_name: str,
) -> dict[str, Any]:
    """Download one workspace-S3 object beneath ``cache_dir``."""
    try:
        local_path = _artifact_cache_path(cache_dir, relative_name)
    except ValueError as exc:
        return {
            "success": False,
            "local_path": None,
            "error": str(exc),
            "bytes_written": 0,
        }

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")

    def _do() -> int:
        with open(tmp_path, "wb") as fh:
            s3_client.download_fileobj(bucket, key, fh)
        size = tmp_path.stat().st_size
        tmp_path.replace(local_path)
        return size

    try:
        size = await asyncio.to_thread(_do)
    except Exception as exc:  # boto3.ClientError, OSError, etc.
        tmp_path.unlink(missing_ok=True)
        return {
            "success": False,
            "local_path": None,
            "error": f"workspace S3 GET failed: {exc}",
            "bytes_written": 0,
        }
    return {
        "success": True,
        "local_path": str(local_path),
        "error": None,
        "bytes_written": size,
    }


_EVALUATE_RESULTS_TARBALL_SUFFIX = "evaluate_results.tar.gz"


def _select_evaluate_results_key(keys: list[str]) -> str | None:
    """Pick the ``evaluate_results.tar.gz`` key from a TAO :list_files response.

    cosmos-rl evaluate jobs upload all per-sample predictions + metrics
    inside a single tarball. Returns the key, or ``None`` if no tarball
    is present.
    """
    for k in keys:
        if k.endswith(_EVALUATE_RESULTS_TARBALL_SUFFIX):
            return k
    return None


def _materialize_evaluate_predictions(
    archive_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    """Extract a cosmos-rl ``evaluate_results.tar.gz`` and synthesize the
    canonical ``per_sample_predictions`` file the rescoring service reads.

    cosmos-rl writes per-sample predictions as one JSON file per image
    under ``<prefix>/freeform/<eval_set>/images/<key>.json``. The prefix
    depends on whether the evaluated checkpoint is a fresh training
    output or a post-training quantized variant:

    - **train-checkpoint evaluate**: ``epoch_<N>/freeform/<eval_set>/
      images/<key>.json`` — the prefix is the chosen training epoch.
    - **quantize-checkpoint evaluate**:
      ``<parent_external_id>/freeform/<eval_set>/images/<key>.json`` —
      cosmos-rl-evaluate of a quantized checkpoint uses the parent
      quantize job's external_id as the prefix instead. A glob that
      hardcodes ``epoch_*`` (``epoch_*/*/*/images/*.json``) misses
      this shape entirely and the rescore reports "no per-sample
      predictions loaded (C2)".

    Each per-sample JSON file is a list of dicts with ``video_id``
    (path-style identifier) and ``answer`` (the model's free-form
    response). The rescoring service expects ``id`` + ``prediction``
    keys, so we translate while aggregating into a single list.

    The full extracted tree is preserved under ``cache_dir/`` so
    operators can inspect raw outputs (metrics PNGs, per-eval-set
    score JSONs, full_response strings). Translation remains best effort;
    the downstream rescoring service compares the materialized key set with
    the frozen DatasetExport before it can validate Student quality.

    Returns ``{"success": bool, "samples": int, "error": str | None}``.
    """
    import tarfile

    if not archive_path.is_file():
        return {
            "success": False,
            "samples": 0,
            "error": f"archive missing at {archive_path}",
        }

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as tf:
            # ``filter='data'`` is the safe extraction mode (Python 3.12+
            # default in 3.14): rejects absolute paths and parent
            # traversal members.
            tf.extractall(path=cache_dir, filter="data")
    except (tarfile.TarError, OSError) as exc:
        return {
            "success": False,
            "samples": 0,
            "error": f"tarball extract failed: {exc}",
        }

    # Walk for per-sample prediction files. Match BOTH the
    # training-epoch shape
    # (``epoch_<N>/freeform/<eval_set>/images/<key>.json``) AND the
    # quantize-parent shape (``<parent_external_id>/freeform/<eval_set>/
    # images/<key>.json``). Hardcoding ``epoch_*`` as the prefix
    # silently misses the quantize-parent variant, producing an empty
    # per_sample_predictions file and a rescore C2.
    aggregated: list[dict[str, Any]] = []
    missing = object()
    for json_path in cache_dir.glob("*/freeform/*/images/*.json"):
        try:
            content = json_path.read_text("utf-8")
            parsed = json.loads(content)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("evaluate predictions: skip %s: %s", json_path, exc)
            continue
        if not isinstance(parsed, list):
            continue
        parsed_list = cast("list[Any]", parsed)
        for raw_item in parsed_list:
            if not isinstance(raw_item, dict):
                continue
            item: dict[str, Any] = cast("dict[str, Any]", raw_item)
            video_id: Any = item.get("video_id") or item.get("id")
            answer: Any = missing
            for answer_key in ("answer", "prediction", "response"):
                if answer_key in item:
                    answer = item[answer_key]
                    break
            if video_id is None or answer is missing:
                continue
            # cosmos-rl reports ``video_id`` as the path-style image
            # reference (e.g., ``images/smoke_1.jpg``); the rescoring
            # service pairs predictions with ground truth on the bare
            # ``example_key`` (e.g., ``smoke_1``) emitted by
            # dataset_export_service. Strip the ``images/`` directory
            # prefix and the file extension to match.
            example_key: Any = video_id
            if isinstance(example_key, str):
                _p = Path(example_key)
                if _p.parts and _p.parts[0] == "images":
                    example_key = str(Path(*_p.parts[1:]))
                example_key = Path(example_key).stem or example_key
            aggregated.append(
                {
                    "id": example_key,
                    "prediction": answer,
                    # Preserve cosmos-rl's auxiliary fields for debugging.
                    "video_id": video_id,
                    "datasource": item.get("datasource"),
                    "correct_answer": item.get("correct_answer"),
                    "reasoning": item.get("reasoning"),
                    "full_response": item.get("full_response"),
                }
            )

    target = cache_dir / "per_sample_predictions"
    try:
        target.write_text(json.dumps(aggregated), encoding="utf-8")
    except OSError as exc:
        return {
            "success": False,
            "samples": len(aggregated),
            "error": f"failed writing {target}: {exc}",
        }
    return {"success": True, "samples": len(aggregated), "error": None}


async def _fetch_tao_artifacts(
    tao_external_job_id: str,
    *,
    settings: Settings,
    local_cache_dir: Path | None = None,
    action: str | None = None,
) -> dict[str, Any]:
    """Fetch artifact references for a finished TAO job.

    Dispatches by ``action``:

    * ``train`` / ``quantize`` (or unspecified): selects the merged-HF
      checkpoint slot (``safetensors/epoch_<latest>/``) and downloads
      each member flat into ``local_cache_dir/<basename>`` so the
      directory satisfies NIM-loadable HF shape (config.json + shards
      + tokenizer at the root) without a LoRA-merge step.
    * ``evaluate``: downloads ``evaluate_results.tar.gz``, extracts
      it under ``local_cache_dir/``, and synthesizes the canonical
      ``local_cache_dir/per_sample_predictions`` file the rescoring
      service reads (translates cosmos-rl's
      ``{video_id, answer}`` format to ``{id, prediction}``).

    Two call modes:

    * **Metadata-only** (``local_cache_dir is None``): returns the list
      of workspace keys without materializing.
    * **With local cache**: as above.

    The Blueprint reads workspace storage directly via ``:list_files``
    + boto3 ``GetObject`` because FTMS 6.26.3's
    ``:download_selective_files`` is unsuitable for cosmos-rl outputs.
    POST returns 405 Method Not Allowed; GET
    no-flags → 400 ``"No files passed in list format to download or,
    best_model or latest_model is not enabled"``; GET ?best_model=true
    → 500 because cosmos-rl jobs do not register that alias. Direct
    workspace-S3 reads bypass the aliasing entirely.

    Returns::

        {"success": bool,
         "artifacts": [{"name", "tao_file_path", "local_path"?,
                        "bytes_written"?, "download_error"?} ...],
         "error": str | None}

    Individual per-file download failures do NOT fail the whole call; the
    failed artifact is reported with ``local_path: None`` and a
    ``download_error`` field.
    """
    # TAO FTMS 6.26.3 can mark a cosmos-rl job ``Done`` before
    # SeaweedFS finalizes the upload of that job's safetensors /
    # evaluate_results.tar.gz, so the first ``:list_files`` after
    # status-flip can return a workspace tree that has not yet grown
    # the action-appropriate artifact slot. Observed live on a 2B
    # FP8_DYNAMIC quantize: post-success fired 7 sec after Done,
    # ``:list_files`` returned 0 matching safetensors/epoch_<N>/ keys,
    # and the StudentModel landed
    # ``checkpoint_packaging_status="failed"`` even though the upload
    # finalized minutes later.
    #
    # Mitigation: when ``:list_files`` succeeds but the action-aware
    # selector returns zero artifacts AND ``local_cache_dir`` is set
    # (caller wants real bytes, not metadata-only enumeration), retry
    # the listing + selection with bounded backoff before declaring
    # failure. Metadata-only callers (``local_cache_dir is None``) are
    # unchanged — those callers explicitly want a snapshot of the
    # current workspace tree, including empty.
    #
    # Widened backoff budget: a [10, 20, 40, 60, 90]s schedule
    # (220s total) is consistently insufficient for fresh quantize
    # uploads: live W8A16 and FP8_DYNAMIC quantizes finishing ~2–3 min
    # after Done both exhausted all 5 retries with empty listings,
    # leaving the paired StudentModel
    # ``checkpoint_packaging_status="failed"`` despite the SeaweedFS
    # upload eventually finalizing within the next ~5-10 minutes. The
    # current schedule gives the SeaweedFS upload window ~16 minutes of
    # total backoff before declaring the artifact genuinely missing.
    # The downside of a longer wait when an upload truly never lands is
    # bounded background-task time on a post-success that eventually
    # falls through to packaging=failed via the documented existing
    # path; the upside is correctly handling every observed real-world
    # quantize race.
    #
    # The retry trigger is GENUINELY-EMPTY listings only. Gating
    # on ``not artifacts`` instead would fire the retry whenever the
    # action-aware selector returns zero matches, including the case
    # where the workspace listing is non-empty but contains only
    # completion metadata (``status.json``, ``microservices_log.txt``)
    # without any checkpoint shards. SeaweedFS finalization is
    # per-object: once a listing returns ANY key, finalization is past
    # the race window for that object set — retrying will not conjure
    # shards that cosmos-rl never produced. The only signature
    # consistent with the finalization race is ``len(keys) == 0``
    # immediately after Done. A non-empty listing without checkpoint
    # shards means cosmos-rl finished and emitted only completion
    # metadata — terminal failure, fail fast. The live cases the budget
    # was widened for were both genuinely-empty listings, so this
    # narrowing keeps the wide budget while eliminating an up-to-970s
    # hang on terminal non-checkpoint listings.
    _F25_EMPTY_LIST_RETRIES = 8
    _F25_BACKOFF_S = [10, 20, 40, 60, 90, 150, 240, 360]
    artifacts: list[dict[str, Any]] = []
    listing: dict[str, Any] | None = None
    for attempt in range(_F25_EMPTY_LIST_RETRIES + 1):
        listing = await _list_tao_job_files(tao_external_job_id, settings=settings)
        if not listing["success"]:
            return {
                "success": False,
                "artifacts": [],
                "error": listing.get("error") or "list_files failed",
            }
        keys = listing["keys"]

        # Action-aware selection.
        if action == "evaluate":
            eval_key = _select_evaluate_results_key(keys)
            artifacts = (
                [
                    {
                        "name": _EVALUATE_RESULTS_TARBALL_SUFFIX,
                        "tao_file_path": eval_key,
                    }
                ]
                if eval_key
                else []
            )
        else:
            selected, common_prefix = _select_hf_checkpoint_keys(keys)
            artifacts = []
            for k in selected:
                rel = (
                    k[len(common_prefix) :]
                    if common_prefix and k.startswith(common_prefix)
                    else k
                )
                artifacts.append({"name": rel, "tao_file_path": k})

        # Metadata-only mode: return whatever the listing showed without
        # retrying — the caller asked for a snapshot, including empty.
        if local_cache_dir is None:
            return {"success": True, "artifacts": artifacts, "error": None}

        if artifacts:
            # Got at least one matching artifact — proceed to download.
            break

        # Retry only when the listing is genuinely empty (the
        # SeaweedFS-finalization race signature). A non-empty listing
        # without matching artifacts means cosmos-rl wrote completion
        # metadata but never produced checkpoint shards — terminal.
        if keys:
            break

        if attempt >= _F25_EMPTY_LIST_RETRIES:
            # All retries exhausted; fall through to the existing
            # empty-after-selection error response below.
            break

        wait_s = _F25_BACKOFF_S[attempt]
        logger.warning(
            ":list_files for TAO job %s (action=%s) returned an "
            "empty listing on attempt %d/%d — TAO may have reported "
            "Done before SeaweedFS finalized the upload; sleeping %ds "
            "before retry",
            tao_external_job_id,
            action,
            attempt + 1,
            _F25_EMPTY_LIST_RETRIES + 1,
            wait_s,
        )
        await asyncio.sleep(wait_s)

    if not artifacts:
        if action == "evaluate":
            err = (
                "no evaluate_results.tar.gz found in workspace listing — "
                f"job {tao_external_job_id} did not upload eval results"
            )
        else:
            err = (
                "no merged-HF checkpoint slot found in workspace listing — "
                f"job {tao_external_job_id} has no safetensors/epoch_<N>/ tree"
            )
        return {"success": False, "artifacts": [], "error": err}

    # The metadata-only path (local_cache_dir is None) returned inside
    # the retry loop above; reaching here means we are in the materialize
    # path and local_cache_dir is non-None. Narrow for pyright +
    # downstream use.
    assert local_cache_dir is not None
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    workspace_cfg = read_tao_deployment_config(settings)
    bucket = workspace_cfg.tao_workspace_bucket if workspace_cfg else None
    if not bucket:
        return {
            "success": False,
            "artifacts": artifacts,
            "error": (
                "Workspace bucket not configured in "
                "deployment.db.tao_deployment_configs.tao_workspace_bucket; "
                "run `vlm-feedback-loop tao-bootstrap`"
            ),
        }
    try:
        s3_client = _build_workspace_s3_client(settings)
    except RuntimeError as exc:
        return {
            "success": False,
            "artifacts": artifacts,
            "error": str(exc),
        }

    for art in artifacts:
        rel = art["name"]
        full_key = art["tao_file_path"]
        result = await _download_workspace_s3_object(
            s3_client,
            bucket=bucket,
            key=full_key,
            cache_dir=local_cache_dir,
            relative_name=rel,
        )
        if result["success"]:
            art["local_path"] = result["local_path"]
            art["bytes_written"] = result["bytes_written"]
        else:
            art["local_path"] = None
            art["download_error"] = result["error"]
            logger.warning(
                "Workspace S3 artifact download failed: job=%s key=%s error=%s",
                tao_external_job_id,
                full_key,
                result["error"],
            )

    # For evaluate jobs, extract the tarball and synthesize the
    # rescoring-service-expected ``per_sample_predictions`` file.
    if action == "evaluate":
        tarball = local_cache_dir / _EVALUATE_RESULTS_TARBALL_SUFFIX
        if tarball.is_file():
            mat = _materialize_evaluate_predictions(tarball, local_cache_dir)
            if not mat.get("success"):
                logger.warning(
                    "evaluate predictions materialization failed: job=%s error=%s",
                    tao_external_job_id,
                    mat.get("error"),
                )
            else:
                logger.info(
                    "evaluate predictions: %d samples synthesized at %s",
                    mat["samples"],
                    local_cache_dir / "per_sample_predictions",
                )

    return {"success": True, "artifacts": artifacts, "error": None}


async def _get_tao_logs_body(
    tao_external_job_id: str,
    *,
    settings: Settings,
    max_retries: int,
) -> HttpResult | None:
    """GET the TAO ``:logs`` endpoint; ``None`` on any config/auth/transport
    failure.

    The single fetcher behind both ``:logs`` consumers —
    ``_fetch_tao_log_text`` (inline failure evidence, one-shot) and
    ``_fetch_tao_logs`` (logs reference, standard retries). The callers
    keep their distinct retry policies and body extraction.
    """
    headers, _preflight_error = await tao_preflight(settings)
    if headers is None:
        return None
    result = await resilient_request(
        "GET",
        (
            f"{tao_base_url(settings)}/orgs/{settings.TAO_ORG_NAME}"
            f"/jobs/{tao_external_job_id}:logs"
        ),
        deadline_s=float(settings.HTTP_DEADLINE_BACKGROUND_S),
        max_retries=max_retries,
        headers=headers,
    )
    if result.error_class is not None:
        return None
    return result


# Tail budget for inline failure evidence — enough to hold the terminal
# stack trace without dragging the whole training log into the DB.
_TAO_LOG_TAIL_MAX_BYTES = 65_536


async def _fetch_tao_log_text(
    tao_external_job_id: str,
    *,
    settings: Settings,
) -> str | None:
    """Fetch the inline TAO logs body for a failed/finished job.

    The TAO REST API documents two log surfaces: ``:logs`` returns the
    chronological stdout/stderr stream, and ``:events`` returns the
    framework event timeline (TAO REST API Overview). For failure
    classification (the NIM-eval-as-quality-fallback gate) the
    ``:logs`` body is sufficient — vLLM and cosmos-rl write their stack
    traces and ``AssertionError`` lines there directly.

    Returns the (truncated) body text on success, or ``None`` if the
    fetch fails or is mis-configured. Best-effort by design — callers
    use the result for pattern matching in
    ``services.tao_failure_classifier``; absence is non-fatal.
    """
    # One shot — failure-evidence fetch is best-effort.
    result = await _get_tao_logs_body(
        tao_external_job_id, settings=settings, max_retries=1
    )
    if result is None:
        return None
    body: Any = result.body
    text: str
    if isinstance(body, str):
        text = body
    elif isinstance(body, dict):
        body_dict = cast("dict[str, Any]", body)
        candidate: Any = body_dict.get("logs") or body_dict.get("text") or ""
        text = candidate if isinstance(candidate, str) else str(body_dict)
    else:
        text = str(body) if body is not None else ""
    if len(text) > _TAO_LOG_TAIL_MAX_BYTES:
        # Keep the TAIL of the log — the failure stack is at the end.
        text = text[-_TAO_LOG_TAIL_MAX_BYTES:]
    return text or None


async def _fetch_tao_logs(
    tao_external_job_id: str,
    *,
    settings: Settings,
) -> dict[str, Any]:
    """Fetch a logs reference via TAO logs endpoint."""
    result = await _get_tao_logs_body(
        tao_external_job_id, settings=settings, max_retries=settings.HTTP_MAX_RETRIES
    )
    if result is None:
        return {"success": False, "logs_ref": None, "error": "logs fetch failed"}
    body: Any = result.body
    logs_ref: Any
    if isinstance(body, dict):
        body_dict = cast("dict[str, Any]", body)
        logs_ref = (
            body_dict.get("logs_url")
            or body_dict.get("path")
            or body_dict.get("logs_ref")
        )
    else:
        logs_ref = None
    return {"success": True, "logs_ref": logs_ref, "error": None}


# ── Polling cadence ─────────────────────────────────────────────────────────


def _should_poll(
    status: str,
    last_polled_at: str | None,
    *,
    now: datetime,
    submitted_interval_s: int,
    running_interval_s: int,
) -> bool:
    """Return True if this job is due for a poll right now.

    Cadence bands: submitted/queued use the 30–60s band, running/paused the
    60–180s band, terminal is never re-polled, and submitting/not_started
    are handled by other paths.
    """
    if status not in _POLLABLE_STATUSES:
        return False
    if last_polled_at is None:
        return True
    try:
        last = datetime.fromisoformat(last_polled_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    delta = (now - last).total_seconds()
    threshold = (
        running_interval_s if status in ("running", "paused") else submitted_interval_s
    )
    return delta >= threshold


# ── Chain advancement (in-session helpers) ─────────────────────────────────


def _transitive_dependents_in_session(
    session: Session,
    *,
    project_id: str,
    chain_id: str,
    root_job_id: str,
) -> list[TAOJob]:
    """Return all jobs in ``chain_id`` whose ``parent_tao_job_id`` chain
    transitively reaches ``root_job_id`` (excluding the root itself).

    BFS over the chain's parent-child edges. Includes terminal-status
    children (for visibility) but callers typically filter to
    ``not_started`` before halting.

    Chain isolation: a failed job halts only its transitive
    dependents — independent siblings whose parent is a different
    (succeeded) job continue.
    """
    visited_ids: set[str] = {root_job_id}
    frontier: set[str] = {root_job_id}
    collected: list[TAOJob] = []

    while frontier:
        children = (
            session.query(TAOJob)
            .filter(
                TAOJob.project_id == project_id,
                TAOJob.chain_id == chain_id,
                TAOJob.parent_tao_job_id.in_(frontier),
            )
            .all()
        )
        next_frontier: set[str] = set()
        for c in children:
            if c.tao_job_id in visited_ids:
                continue
            visited_ids.add(c.tao_job_id)
            next_frontier.add(c.tao_job_id)
            collected.append(c)
        frontier = next_frontier

    return collected


def _halt_chain_in_session(
    session: Session,
    *,
    project_id: str,
    chain_id: str,
    reason_action: str,
    reason_seq: int,
    reason_status: str,
    failed_job_id: str,
) -> list[TAOJob]:
    """Mark transitive dependents of ``failed_job_id`` as failed+halted.

    Chain isolation: halts ONLY the
    not_started jobs that depend (transitively) on the failed job. A
    failed evaluate that has no children halts no siblings. Independent
    quantize jobs whose parent (the train job) is still ``succeeded`` are
    NOT halted — they remain ``not_started`` and become eligible for
    submission via ``_find_next_eligible_in_session``.

    Returns the rows updated (caller emits ``run_failed`` for each).
    """
    updated: list[TAOJob] = []
    dependents = _transitive_dependents_in_session(
        session,
        project_id=project_id,
        chain_id=chain_id,
        root_job_id=failed_job_id,
    )
    now = utc_now()
    reason = (
        f"Chain halted: {reason_action} (seq {reason_seq}, id={failed_job_id}) "
        f"reached terminal {reason_status!r}"
    )
    for row in dependents:
        # Only halt rows that haven't already terminalized. ``not_started
        # → failed`` is an allowed transition when chain_halted_reason is
        # set; we don't re-flip already-terminal rows.
        if row.status == "not_started" and can_transition(row.status, "failed"):
            row.status = "failed"
            row.chain_halted_reason = reason
            row.completed_at = now
            updated.append(row)
    return updated


def _find_next_eligible_in_session(
    session: Session,
    *,
    project_id: str,
    chain_id: str,
) -> TAOJob | None:
    """Return the lowest-``chain_sequence`` ``not_started`` job in this
    chain whose dependency is satisfied — i.e., its
    ``parent_tao_job_id`` is None (chain root) or points at a
    ``succeeded`` predecessor (chain-isolation eligibility).

    Returns ``None`` when no eligible job remains. Used for
    chain advancement after either a succeeded or a failed terminal —
    a failed evaluate may unblock independent siblings whose parent is
    the train job.
    """
    not_started_rows = (
        session.query(TAOJob)
        .filter(
            TAOJob.project_id == project_id,
            TAOJob.chain_id == chain_id,
            TAOJob.status == "not_started",
        )
        .order_by(TAOJob.chain_sequence.asc())
        .all()
    )
    if not not_started_rows:
        return None

    parent_ids = {j.parent_tao_job_id for j in not_started_rows if j.parent_tao_job_id}
    parent_status: dict[str, str] = {}
    if parent_ids:
        parent_rows = (
            session.query(TAOJob.tao_job_id, TAOJob.status)
            .filter(
                TAOJob.project_id == project_id,
                TAOJob.tao_job_id.in_(parent_ids),
            )
            .all()
        )
        parent_status: dict[str, str] = {row[0]: row[1] for row in parent_rows}

    for j in not_started_rows:
        if j.parent_tao_job_id is None:
            return j
        if parent_status.get(j.parent_tao_job_id) == "succeeded":
            return j
    return None


def _chain_has_active_work(
    session: Session,
    *,
    project_id: str,
    chain_id: str,
) -> bool:
    """Return True if any non-terminal job remains in this chain.

    ``not_started`` counts as active when the predecessor has not yet
    reached a terminal state; it only becomes ``inactive`` for cross-chain
    advancement once the chain has completed or halted.
    """
    rows = (
        session.query(TAOJob.status)
        .filter(
            TAOJob.project_id == project_id,
            TAOJob.chain_id == chain_id,
        )
        .all()
    )
    for (status,) in rows:
        if status not in TERMINAL_STATUSES and status != "not_started":
            return True
    # If any ``not_started`` remains AND at least one in this chain is
    # succeeded (i.e. the chain is mid-flight), treat as active.
    statuses = [s for (s,) in rows]
    return "not_started" in statuses and "succeeded" in statuses


def _find_next_chain_to_start(
    session: Session,
    *,
    project_id: str,
    suite: TrainingSuite,
) -> str | None:
    """Find the next chain in ``chain_ids_ordered`` whose seq=1 job is not_started.

    A chain is eligible to start when it has not begun yet. Chains that
    already have non-terminal or succeeded jobs are skipped.
    """
    for chain_id in suite.chain_ids_ordered or []:
        seq1 = (
            session.query(TAOJob)
            .filter(
                TAOJob.project_id == project_id,
                TAOJob.chain_id == chain_id,
                TAOJob.chain_sequence == 1,
            )
            .first()
        )
        if seq1 is None or seq1.status != "not_started":
            continue
        return seq1.tao_job_id
    return None


# ── Suite status roll-up ────────────────────────────────────────────────────


def _roll_up_suite_status(
    session: Session,
    *,
    project_id: str,
    chain_id: str,
) -> None:
    """Recompute TrainingSuite.status when any chain member transitions.

    Rules:
      * Any chain job in ``failed`` or ``canceled`` → suite is ``failed``.
      * All chain jobs across all chains ``succeeded`` → suite ``completed``.
      * Else suite stays ``running``.
    """
    # Find the training suite that owns this chain.
    suites = (
        session.query(TrainingSuite)
        .filter(TrainingSuite.project_id == project_id)
        .all()
    )
    suite: TrainingSuite | None = None
    for s in suites:
        chain_ids = list(s.chain_ids_ordered or [])
        if chain_id in chain_ids:
            suite = s
            break
    if suite is None:
        return

    # Suite-level cancellation is authoritative. Late poll results may
    # terminalize individual jobs, but must not relabel the canceled suite
    # as failed/completed or make it eligible for more chain advancement.
    if suite.status == "canceled":
        return

    all_jobs = (
        session.query(TAOJob.status, TAOJob.chain_halted_reason)
        .filter(
            TAOJob.project_id == project_id,
            TAOJob.chain_id.in_(suite.chain_ids_ordered or []),
        )
        .all()
    )
    if not all_jobs:
        return

    # Auto-eval policy skips land status="canceled" with a chain_halted_reason
    # carrying the auto-skip marker prefix. They are deliberate skips
    # (the Student stays quality_status=pending for the NIM-eval fallback),
    # NOT failures — treat them as success-equivalent so a suite whose only
    # non-successes are policy skips finalizes "completed" rather than the
    # misleading "failed" the raw canceled status would produce.
    statuses = [
        "succeeded" if (s == "canceled" and _is_auto_skip_reason(r)) else s
        for (s, r) in all_jobs
    ]

    now = utc_now()
    has_failure = any(s in ("failed", "canceled") for s in statuses)
    all_terminal = all(s in TERMINAL_STATUSES for s in statuses)
    all_succeeded = all(s == "succeeded" for s in statuses)

    if has_failure and all_terminal:
        if suite.status != "failed":
            suite.status = "failed"
            suite.completed_at = now
    elif all_succeeded:
        if suite.status != "completed":
            suite.status = "completed"
            suite.completed_at = now
    elif has_failure:
        # Chain halted but other chains may still be running; mark failed
        # only when everything reaches terminal. Leave status unchanged.
        pass
    else:
        if suite.status == "initialized":
            suite.status = "running"
            suite.started_at = now


def _plan_suite_advance(
    session: Session,
    *,
    project_id: str,
    chain_id: str,
) -> tuple[str | None, str | None]:
    """Plan chain/suite advancement after a failed or canceled terminal.

    The one failure/cancel tail shared by :func:`handle_terminal_failure`
    and ``tao_job_service.cancel_tao_job`` — previously hand-copied
    between them, which is how the 2026-07-14 stranded-suite bug class
    arose. Finds the next eligible in-chain job (a failed/canceled job
    may leave independent siblings eligible); when none remains and the
    chain has no active work, finds the next chain to start; always
    rolls up the owning suite's status. Returns
    ``(next_in_chain_job_id, cross_chain_next_id)`` — at most one is
    non-None. The caller owns its transaction commit and must submit the
    returned job OUTSIDE the transaction (network call). The success
    path (:func:`_advance_after_terminal`) keeps its own variant: its
    find-next runs the auto-skip loop.
    """
    suite = find_suite_for_chain(session, project_id=project_id, chain_id=chain_id)
    if suite is not None and suite.status == "canceled":
        return None, None

    next_in_chain_job_id: str | None = None
    next_eligible = _find_next_eligible_in_session(
        session, project_id=project_id, chain_id=chain_id
    )
    if next_eligible is not None:
        next_in_chain_job_id = next_eligible.tao_job_id

    cross_chain_next_id: str | None = None
    if (
        next_in_chain_job_id is None
        and suite is not None
        and not _chain_has_active_work(
            session, project_id=project_id, chain_id=chain_id
        )
    ):
        cross_chain_next_id = _find_next_chain_to_start(
            session, project_id=project_id, suite=suite
        )
    _roll_up_suite_status(session, project_id=project_id, chain_id=chain_id)
    return next_in_chain_job_id, cross_chain_next_id


# ── Per-job poll ────────────────────────────────────────────────────────────


async def _poll_single_job(
    project_id: str,
    tao_job_id: str,
    *,
    engine: Engine,
    settings: Settings,
) -> None:
    """Poll a single TAOJob, persist the result, emit SSE, advance chain."""
    # Read current state (short txn).
    with Session(engine) as session:
        job = (
            session.query(TAOJob)
            .filter_by(tao_job_id=tao_job_id, project_id=project_id)
            .first()
        )
        if job is None:
            return
        if job.status not in _POLLABLE_STATUSES:
            return  # raced with another tick — skip
        external_id = job.tao_external_job_id
        prior_status = job.status
        prior_progress = job.progress
        prior_poll_error = job.poll_error_ref
        chain_id = job.chain_id
        chain_sequence = job.chain_sequence
        action = job.action

    if not external_id:
        return  # no external handle to query yet

    # Remote poll (no txn).
    poll_result = await tao_job_service.poll_tao_job(
        external_id, settings=settings, action=action
    )
    now = utc_now()

    with Session(engine) as session:
        job = (
            session.query(TAOJob)
            .filter_by(tao_job_id=tao_job_id, project_id=project_id)
            .first()
        )
        if job is None:
            return
        outcome = tao_job_service.apply_poll_result(
            job, poll_result, now, failed_error_ref_requires_transition=True
        )
        session.commit()
        if not outcome.poll_ok:
            # Poll error: last-known-good status preserved, poll_error_ref
            # recorded by apply_poll_result. WARN once per distinct error
            # (DEBUG on repeats) — a poller failing every tick with only a
            # DB-column breadcrumb is operationally invisible (a
            # misconfigured TAO endpoint burned 20 silent minutes live,
            # 2026-07-14).
            poll_error = poll_result.get("error")
            log = logger.warning if poll_error != prior_poll_error else logger.debug
            log(
                "tao_polling: poll failed for job %s (project %s): %s",
                tao_job_id,
                project_id,
                poll_error,
            )
            return

    status_changed = outcome.status_changed
    new_status = outcome.new_status

    progress_changed = (
        poll_result.get("progress") is not None
        and poll_result.get("progress") != prior_progress
    )

    # Emit SSE after the write commits.
    if status_changed or progress_changed:
        try:
            await sse_manager.emit(
                project_id,
                "tao_job_progress",
                {
                    "run_id": tao_job_id,
                    "tao_job_id": tao_job_id,
                    "run_type": "tao_job",
                    "status": new_status or prior_status,
                    "progress": poll_result.get("progress"),
                },
            )
        except Exception:  # pragma: no cover — SSE best-effort
            logger.exception("SSE emit failed for tao_job %s", tao_job_id)

    # Terminal handling.
    if status_changed and new_status in TERMINAL_STATUSES:
        if new_status == "succeeded":
            # The post-success flow
            # (artifact fetch + downstream actions + chain advance) is
            # dispatched as a background task so a multi-GB safetensors
            # download from one project doesn't head-of-line-block the
            # polling tick for every other project. Idempotent at the
            # background_manager level via the deduped task_id.
            _dispatch_post_success_flow(
                project_id,
                tao_job_id,
                external_id=external_id,
                action=action,
                chain_id=chain_id,
                chain_sequence=chain_sequence,
                engine=engine,
                settings=settings,
            )
        elif new_status in ("failed", "canceled"):
            await handle_terminal_failure(
                project_id,
                tao_job_id,
                chain_id=chain_id,
                chain_sequence=chain_sequence,
                action=action,
                terminal_status=new_status,
                engine=engine,
                settings=settings,
            )
        elif new_status == "deleted":
            # No advancement; emit run_failed to surface the removal.
            try:
                await sse_manager.emit(
                    project_id,
                    "run_failed",
                    {
                        "run_id": tao_job_id,
                        "tao_job_id": tao_job_id,
                        "run_type": "tao_job",
                        "error_summary": "deleted_from_tao",
                    },
                )
            except Exception:  # pragma: no cover
                logger.exception("SSE emit failed")


# ── Terminal outcome handling ──────────────────────────────────────────────


async def _handle_succeeded(
    project_id: str,
    tao_job_id: str,
    *,
    external_id: str,
    action: str,
    engine: Engine,
    settings: Settings,
) -> None:
    """Retrieve artifacts + logs + resolved fields; emit tao_job_completed.

    Artifacts are materialized locally under
    ``{project_dir}/artifacts/tao_jobs/{tao_job_id}/`` so downstream
    packaging and re-scoring can read real files.
    The action-specific artifact selection lives in
    :func:`_fetch_tao_artifacts`: ``train``/``quantize`` materialize the
    latest ``safetensors/epoch_<N>/`` directory flat; ``evaluate``
    extracts ``evaluate_results.tar.gz`` and synthesizes a
    ``per_sample_predictions`` JSON file from the cosmos-rl per-sample
    tree.

    The lifecycle marker ``outputs_fetch_status`` is flipped
    ``pending → in_progress`` at entry and ``in_progress → completed`` at
    successful exit. On exception the marker is set to ``failed`` and
    the exception text persisted on ``outputs_fetch_error_ref``; the
    error is logged but NOT re-raised so the polling loop continues.
    The polling tick's recovery scan re-fires this handler for any job
    in ``pending`` or ``in_progress`` after a backend restart — the
    artifact downloads are already idempotent (atomic ``.part`` files
    are truncated and overwritten on retry).
    """
    # Mark in_progress in a short txn so a crash from here on leaves a
    # recovery marker the next tick can pick up. Clearing
    # ``outputs_fetch_error_ref`` here means a successful retry erases
    # the audit trail of the prior failure — that's deliberate; the
    # successful outputs are now authoritative.
    with Session(engine) as session:
        job = session.query(TAOJob).filter_by(tao_job_id=tao_job_id).first()
        if job is None:
            return
        job.outputs_fetch_status = "in_progress"
        job.outputs_fetch_error_ref = None
        session.commit()

    try:
        await _handle_succeeded_body(
            project_id,
            tao_job_id,
            external_id=external_id,
            action=action,
            engine=engine,
            settings=settings,
        )
    except Exception as exc:
        logger.exception(
            "tao_polling: outputs-fetch flow failed for tao_job %s (action=%s)",
            tao_job_id,
            action,
        )
        with Session(engine) as session:
            job = session.query(TAOJob).filter_by(tao_job_id=tao_job_id).first()
            if job is not None:
                job.outputs_fetch_status = "failed"
                job.outputs_fetch_error_ref = repr(exc)[:1000]
                session.commit()
        return  # caller (_advance_after_terminal) is skipped on hard failure

    with Session(engine) as session:
        job = session.query(TAOJob).filter_by(tao_job_id=tao_job_id).first()
        if job is not None:
            job.outputs_fetch_status = "completed"
            session.commit()


async def _handle_succeeded_body(
    project_id: str,
    tao_job_id: str,
    *,
    external_id: str,
    action: str,
    engine: Engine,
    settings: Settings,
) -> None:
    """The actual outputs-fetch body, separated so ``_handle_succeeded``
    can wrap it in the ``outputs_fetch_status`` lifecycle markers without
    affecting the artifact/SSE/downstream-action logic."""
    cache_dir = _tao_artifact_cache_dir(settings.WORKSPACE_ROOT, project_id, tao_job_id)

    artifacts_result = await _fetch_tao_artifacts(
        external_id,
        settings=settings,
        local_cache_dir=cache_dir,
        action=action,
    )
    logs_result = await _fetch_tao_logs(external_id, settings=settings)

    with Session(engine) as session:
        job = session.query(TAOJob).filter_by(tao_job_id=tao_job_id).first()
        if job is None:
            return
        outputs: dict[str, Any] = dict(job.outputs or {})
        if artifacts_result.get("success"):
            outputs["artifacts"] = artifacts_result.get("artifacts") or []
            outputs["artifact_cache_dir"] = str(cache_dir)
        if logs_result.get("success") and logs_result.get("logs_ref") is not None:
            outputs["logs_ref"] = logs_result["logs_ref"]

        # For train jobs, try to extract resolved_training_fields from outputs.
        if action == "train" and isinstance(job.outputs, dict):
            resolved = job.outputs.get("resolved_training_fields")
            if isinstance(resolved, dict):
                job_config = dict(job.job_config or {})
                existing_resolved = dict(
                    job_config.get("resolved_training_fields") or {}
                )
                # Merge remote-observed values; keep locally-computed fallback
                # when TAO omits them.
                merged: dict[str, Any] = {**existing_resolved, **resolved}
                job_config["resolved_training_fields"] = merged
                job.job_config = job_config

        job.outputs = outputs
        session.commit()

    try:
        await sse_manager.emit(
            project_id,
            "tao_job_completed",
            {
                "run_id": tao_job_id,
                "tao_job_id": tao_job_id,
                "run_type": "tao_job",
                "status": "succeeded",
            },
        )
    except Exception:  # pragma: no cover
        logger.exception("SSE emit failed")

    # ── Downstream actions by TAO action ─────────────────────────────────
    #
    # train / quantize → package checkpoint + register StudentModel
    #                    (service handles packaging failure cleanly,
    #                    does NOT block chain advancement).
    # evaluate        → re-score via canonical Exact Match evaluator;
    #                    on success flip paired StudentModel.quality_status
    #                    to "validated", on C2 (missing/incomplete or wholly
    #                    invalid predictions, inconsistent frozen export,
    #                    missing artifact cache, etc.) flip to "failed".
    #
    # Imports are inline so modules that don't need these services
    # aren't forced to import them at module-load time.
    try:
        if action in ("train", "quantize"):
            from vlm_feedback_loop.services import student_model_service

            await student_model_service.register_from_tao_terminal(
                project_id, tao_job_id, settings=settings
            )
        elif action == "evaluate":
            from vlm_feedback_loop.services import (
                student_model_service,
                tao_rescoring_service,
            )

            run_id = await tao_rescoring_service.rescore_evaluate_job(
                project_id, tao_job_id, settings=settings
            )
            if run_id is None:
                # C2 widened: evaluate reached `succeeded` but no
                # parseable predictions / no paired ground truth / no
                # Test Pool export / artifact cache missing. Flip the
                # paired StudentModel (if one exists) to failed.
                with Session(engine) as session:
                    evaluate_job = (
                        session.query(TAOJob)
                        .filter_by(tao_job_id=tao_job_id, project_id=project_id)
                        .first()
                    )
                    if evaluate_job is not None:
                        student = student_model_service.find_student_for_evaluate_job(
                            session,
                            project_id=project_id,
                            evaluate_job=evaluate_job,
                        )
                        if student is not None:
                            student_model_service.mark_student_quality_failed(
                                student=student,
                                reason="no_parseable_predictions",
                            )
                            session.commit()
    except Exception:  # pragma: no cover — defensive; never crash poller
        logger.exception(
            "Downstream action failed for TAOJob %s (action=%s)",
            tao_job_id,
            action,
        )


async def _advance_after_terminal(
    project_id: str,
    *,
    chain_id: str | None,
    chain_sequence: int | None,
    engine: Engine,
    settings: Settings,
) -> None:
    """After a succeeded terminal, submit the next chain job (if any).

    Then roll up suite status. If the chain is exhausted, start the next
    chain in ``chain_ids_ordered``.
    """
    if chain_id is None or chain_sequence is None:
        return

    next_job_id: str | None = None
    local_baseline_eval_id: str | None = None
    cross_chain_next_id: str | None = None

    with Session(engine) as session:
        suite = find_suite_for_chain(session, project_id=project_id, chain_id=chain_id)
        if suite is not None and suite.status == "canceled":
            return

        # Iteratively skip TAO ``evaluate`` jobs that are doomed on the
        # cosmos-rl side because the trained base is on the operator-configured
        # blocklist. A LoRA baseline is different: TAO cannot load its
        # adapter-only checkpoint, so the Blueprint merges it and evaluates
        # the resulting full checkpoint through the local Student NIM.
        # Each policy-skipped job
        # skipped job lands ``status="canceled"`` with the auto-skip
        # chain_halted_reason marker (the roll-up treats that prefix as
        # success-equivalent); the next loop iteration then looks
        # for the next eligible job (skipped rows are no longer
        # not_started), so e.g. (train → evaluate skipped
        # → quantize → quantize-evaluate skipped → next train chain)
        # advances cleanly without burning ~3 min/skipped on TAO.
        skip_set = frozenset(settings.TAO_AUTOEVAL_SKIP_BASES or [])
        while True:
            next_job = _find_next_eligible_in_session(
                session,
                project_id=project_id,
                chain_id=chain_id,
            )
            if next_job is None:
                break
            if _is_lora_baseline_evaluate(session, next_job):
                next_job.status = "running"
                next_job.started_at = next_job.started_at or utc_now()
                next_job.training_backend = "student_nim_local"
                next_job.chain_halted_reason = None
                next_job.progress = {
                    "metrics_latest": {
                        "stage": "Merging LoRA checkpoint and validating Student NIM"
                    }
                }
                job_config = dict(next_job.job_config or {})
                job_config["evaluation_source"] = "student_nim_local"
                job_config["requires_merged_checkpoint"] = True
                next_job.job_config = job_config
                local_baseline_eval_id = next_job.tao_job_id
                break
            base_blocklisted = (
                bool(skip_set)
                and next_job.action == "evaluate"
                and _trained_base_is_blocklisted(session, next_job, skip_set)
            )
            if not base_blocklisted:
                next_job_id = next_job.tao_job_id
                break
            # Skip path: mark the evaluate as canceled with a clear
            # reason, log, then keep searching past it. Downstream
            # quantize siblings parent on the still-succeeded train, so
            # the chain-isolation rule keeps them eligible.
            next_job.status = "canceled"
            # Stamp completed_at like _halt_chain_in_session does — a canceled
            # job with a null completed_at skews duration reporting and the
            # job-monitor timeline.
            next_job.completed_at = utc_now()
            next_job.chain_halted_reason = (
                f"{AUTO_SKIP_REASON_PREFIX} action=evaluate auto-skipped; "
                f"base {_resolve_trained_base_name(session, next_job)!r} "
                f"is in TAO_AUTOEVAL_SKIP_BASES — Student stays at "
                f"quality_status=pending for the cold-start NIM-eval fallback."
            )
            logger.info(
                "skipping TAO evaluate %s (chain_sequence=%d, base=%s) — "
                "Student will route through the "
                "NIM-eval-as-quality-fallback.",
                next_job.tao_job_id,
                next_job.chain_sequence,
                _resolve_trained_base_name(session, next_job),
            )

        if (
            next_job_id is None
            and local_baseline_eval_id is None
            and suite is not None
            and not _chain_has_active_work(
                session, project_id=project_id, chain_id=chain_id
            )
        ):
            # Chain exhausted (either naturally or via policy skips): roll up
            # suite status + check cross-chain advancement.
            cross_chain_next_id = _find_next_chain_to_start(
                session, project_id=project_id, suite=suite
            )

        _roll_up_suite_status(session, project_id=project_id, chain_id=chain_id)
        session.commit()

    if local_baseline_eval_id is not None:
        await _run_local_baseline_evaluation(
            project_id,
            local_baseline_eval_id,
            chain_id=chain_id,
            engine=engine,
            settings=settings,
        )
    elif next_job_id is not None:
        await tao_job_service.submit_chain_job(
            project_id, next_job_id, settings=settings
        )
    elif cross_chain_next_id is not None:
        await tao_job_service.submit_chain_job(
            project_id, cross_chain_next_id, settings=settings
        )


def _resolve_trained_base_name(session: Session, job: TAOJob) -> str | None:
    """Return the ``ModelConfig.model_name`` of the trained base behind
    ``job`` (any chain action). Used to decide whether an
    ``evaluate`` action should be auto-skipped per
    ``Settings.TAO_AUTOEVAL_SKIP_BASES``.

    Reads the trained ``student_base_model_config_id`` directly off the
    TAOJob row (every chain member persists this for audit).
    Returns ``None`` when the ModelConfig has been deleted out from under
    a still-pending chain (defensive — caller treats ``None`` as
    not-blocklisted).
    """
    if not job.student_base_model_config_id:
        return None
    mc = (
        session.query(ModelConfig)
        .filter_by(
            project_id=job.project_id,
            model_config_id=job.student_base_model_config_id,
        )
        .first()
    )
    return mc.model_name if mc is not None else None


def _trained_base_is_blocklisted(
    session: Session,
    job: TAOJob,
    skip_set: frozenset[str],
) -> bool:
    """True iff this job's trained base is on the
    auto-evaluate skip list."""
    base_name = _resolve_trained_base_name(session, job)
    return base_name is not None and base_name in skip_set


def _chain_trained_adapter_only(job: TAOJob) -> bool:
    """True iff this chain member's persisted ``job_config`` records an
    adapter-only training mode (``lora_config.enable_lora=true``).

    cosmos-rl ``evaluate`` derives its model exclusively from the parent
    checkpoint and hands an adapter-only checkpoint directory
    straight to vLLM engine init, which cannot load PEFT-prefixed
    (``base_model.*``) weights into the bare base class — the job dies at
    engine init ("There is no module or parameter named 'base_model'",
    live-verified on FTMS 6.26.3, 2026-07-15). Unlike quantize,
    the evaluate CLI exposes no ``enable_lora``/
    ``base_model_path`` merge path, so the skip is unconditional — LoRA
    Student quality routes through the §9.5 NIM-eval fallback, which
    serves the merged base+adapter checkpoint produced by
    ``scripts/merge_lora.py``.
    """
    lora_cfg: dict[str, Any] = (job.job_config or {}).get("lora_config") or {}
    return lora_cfg.get("enable_lora") is True


def _is_lora_baseline_evaluate(session: Session, job: TAOJob) -> bool:
    """Return True for the post-train evaluate in a LoRA chain.

    Quantized evaluate rows parent on ``quantize`` and remain TAO-native:
    TAO quantize already materializes a merged quantized checkpoint. Only the
    baseline row that parents directly on ``train`` needs Blueprint-owned
    merge + NIM evaluation.
    """
    if job.action != "evaluate" or not job.parent_tao_job_id:
        return False
    parent = (
        session.query(TAOJob)
        .filter_by(
            project_id=job.project_id,
            tao_job_id=job.parent_tao_job_id,
        )
        .first()
    )
    return bool(
        parent is not None
        and parent.action == "train"
        and _chain_trained_adapter_only(parent)
    )


async def _run_local_baseline_evaluation(
    project_id: str,
    tao_job_id: str,
    *,
    chain_id: str,
    engine: Engine,
    settings: Settings,
) -> None:
    """Complete a synthetic baseline evaluate row via merged Student NIM."""
    from vlm_feedback_loop.services import student_model_service

    result = await student_model_service.run_automatic_baseline_evaluation(
        project_id=project_id,
        evaluate_tao_job_id=tao_job_id,
        settings=settings,
    )
    with Session(engine) as session:
        job = (
            session.query(TAOJob)
            .filter_by(project_id=project_id, tao_job_id=tao_job_id)
            .first()
        )
        if job is None:
            return
        sequence = job.chain_sequence
        outputs = dict(job.outputs or {})
        outputs.update(
            {
                "evaluation_source": "student_nim_local",
                "student_model_id": result.get("student_model_id"),
                "evaluation_run_id": result.get("evaluation_run_id"),
                "quality_status": result.get("quality_status"),
                "serving_status": result.get("serving_status"),
            }
        )
        job.outputs = outputs
        job.completed_at = utc_now()
        job.outputs_fetch_status = "completed"
        if result.get("success"):
            job.status = "succeeded"
            job.tao_status_raw = "BlueprintLocalNIMEvaluationSucceeded"
            job.progress = {
                "metrics_latest": {"stage": "Merged Student NIM evaluation complete"}
            }
            job.error_ref = None
        else:
            job.status = "failed"
            job.tao_status_raw = "BlueprintLocalNIMEvaluationFailed"
            job.error_ref = str(result.get("error") or "local_baseline_eval_failed")
            if result.get("detail") is not None:
                outputs["error_detail"] = result["detail"]
        session.commit()

    if result.get("success"):
        try:
            await sse_manager.emit(
                project_id,
                "tao_job_completed",
                {
                    "run_id": tao_job_id,
                    "tao_job_id": tao_job_id,
                    "run_type": "tao_job",
                    "status": "succeeded",
                    "evaluation_source": "student_nim_local",
                },
            )
        except Exception:  # pragma: no cover
            logger.exception("SSE emit failed")
        await _advance_after_terminal(
            project_id,
            chain_id=chain_id,
            chain_sequence=sequence,
            engine=engine,
            settings=settings,
        )
        return

    await handle_terminal_failure(
        project_id,
        tao_job_id,
        chain_id=chain_id,
        chain_sequence=sequence,
        action="evaluate",
        terminal_status="failed",
        engine=engine,
        settings=settings,
    )


async def handle_terminal_failure(
    project_id: str,
    tao_job_id: str,
    *,
    chain_id: str | None,
    chain_sequence: int | None,
    action: str,
    terminal_status: str,
    engine: Engine,
    settings: Settings,
    emit_root_failed: bool = True,
) -> None:
    """Halt remaining chain members, emit run_failed, cross-chain advance.

    The single post-failure flow for a chain job reaching a terminal
    ``failed``/``canceled`` status, whether the transition was detected by
    polling (``_poll_single_job``), at submission time
    (``tao_job_service.submit_chain_job``), or during startup recovery
    (``tao_job_service.recover_interrupted_tao_jobs`` → the lifespan). Routing
    every terminal failure through here is what keeps the TrainingSuite from
    stranding in ``running`` when a mid-chain submission fails.

    ``emit_root_failed=False`` suppresses the ``run_failed`` SSE for
    ``tao_job_id`` itself (the caller already emitted it) while still emitting
    for any newly halted dependents.

    For ANY action's failure (``train`` / ``evaluate`` / ``quantize``),
    fetches the TAO ``:logs`` body and stores its tail on
    ``TAOJob.outputs.tao_logs_text`` (best-effort).

    The NIM-eval-as-quality-fallback gate
    (``services.tao_failure_classifier``) reads this evidence on evaluate
    failures to decide whether the prior failure matches a known
    upstream-loader gap before promoting ``quality_status`` from a
    successful NIM eval. Without this capture the classifier has nothing
    to match against and the fallback can never fire.

    **Captures ``:logs`` for train + quantize too, not just
    evaluate.** Evidence: a live 8B train failed at 3 min with
    ``tao_status_raw=Error`` and only an empty ``error_ref`` persisted —
    the operator had to manually ``curl :logs`` to discover the root
    cause (TAO-side HuggingFace download stall). Train + quantize
    failures are diagnostically just as load-bearing as evaluate.
    """
    log_text: str | None = None
    if terminal_status == "failed":
        # Pre-fetch outside the write txn so we don't hold SQLite while
        # waiting on TAO's logs endpoint. Best-effort; absence is non-fatal.
        # All three action types (``train``/``evaluate``/``quantize``) are
        # eligible — wider than the original evaluate-only capture.
        # Skipped when a prior pass already persisted the log tail, so a
        # recovery re-fire never re-hits TAO for the same logs.
        try:
            with Session(engine) as session:
                row = (
                    session.query(TAOJob)
                    .filter_by(tao_job_id=tao_job_id, project_id=project_id)
                    .first()
                )
                ext_id = row.tao_external_job_id if row is not None else None
                already_captured = bool(
                    row is not None and (row.outputs or {}).get("tao_logs_text")
                )
            if ext_id and not already_captured:
                log_text = await _fetch_tao_log_text(ext_id, settings=settings)
        except Exception:  # pragma: no cover — best-effort
            log_text = None

    halted: list[dict[str, Any]] = []
    cross_chain_next_id: str | None = None
    next_in_chain_job_id: str | None = None

    with Session(engine) as session:
        if chain_id is not None and chain_sequence is not None:
            # Chain isolation: halt only transitive dependents
            # of this failed job. Independent siblings whose parent is a
            # different (succeeded) job remain ``not_started`` and get
            # picked up by ``_find_next_eligible_in_session`` below.
            halted_rows = _halt_chain_in_session(
                session,
                project_id=project_id,
                chain_id=chain_id,
                reason_action=action,
                reason_seq=chain_sequence,
                reason_status=terminal_status,
                failed_job_id=tao_job_id,
            )
            halted = [
                {
                    "tao_job_id": r.tao_job_id,
                    "reason": r.chain_halted_reason,
                }
                for r in halted_rows
            ]

            # After halting transitive dependents, an independent sibling
            # may be eligible — for instance, a quantize job that parents
            # on the (still-succeeded) train, freed up by an
            # evaluate-failure that does not transitively reach it.
            next_in_chain_job_id, cross_chain_next_id = _plan_suite_advance(
                session, project_id=project_id, chain_id=chain_id
            )

        # Persist log tail on the failed job so:
        #   - evaluate failures: the NIM-eval-as-quality-fallback gate
        #     can pattern-match the failure signature.
        #   - train / quantize failures: the operator has on-disk
        #     attribution without having to ``curl :logs`` manually
        #     against the external TAO job id.
        # Stored inline on ``outputs.tao_logs_text`` (truncated to 64 KB
        # above).
        if log_text:
            failed_job = (
                session.query(TAOJob)
                .filter_by(tao_job_id=tao_job_id, project_id=project_id)
                .first()
            )
            if failed_job is not None:
                outputs = dict(failed_job.outputs or {})
                outputs["tao_logs_text"] = log_text
                failed_job.outputs = outputs
                actionable_error = (
                    tao_job_service.extract_actionable_failure_from_logs(log_text)
                )
                generic_error = (
                    not failed_job.error_ref
                    or failed_job.error_ref.strip().lower()
                    == f"{action} action failed for cosmos-rl"
                )
                if actionable_error and generic_error:
                    failed_job.error_ref = actionable_error

        # Evaluate failure → flip paired StudentModel
        # quality_status="failed" (two-part readiness).
        if action == "evaluate":
            try:
                from vlm_feedback_loop.services import student_model_service

                evaluate_job = (
                    session.query(TAOJob)
                    .filter_by(tao_job_id=tao_job_id, project_id=project_id)
                    .first()
                )
                if evaluate_job is not None:
                    student = student_model_service.find_student_for_evaluate_job(
                        session,
                        project_id=project_id,
                        evaluate_job=evaluate_job,
                    )
                    if student is not None:
                        student_model_service.mark_student_quality_failed(
                            student=student,
                            reason=f"tao_evaluate_{terminal_status}",
                        )
            except Exception:  # pragma: no cover
                logger.exception(
                    "Failed to flip StudentModel quality_status for evaluate "
                    "job %s (%s)",
                    tao_job_id,
                    terminal_status,
                )

        session.commit()

    try:
        if emit_root_failed:
            await sse_manager.emit(
                project_id,
                "run_failed",
                {
                    "run_id": tao_job_id,
                    "tao_job_id": tao_job_id,
                    "run_type": "tao_job",
                    "error_summary": f"tao_job_{terminal_status}",
                },
            )
        for h in halted:
            await sse_manager.emit(
                project_id,
                "run_failed",
                {
                    "run_id": h["tao_job_id"],
                    "tao_job_id": h["tao_job_id"],
                    "run_type": "tao_job",
                    "error_summary": h["reason"] or "chain_halted",
                },
            )
    except Exception:  # pragma: no cover
        logger.exception("SSE emit failed")

    # Chain isolation: a failed evaluate may unblock
    # independent siblings (e.g., a quantize whose parent is the
    # still-succeeded train). Submit the next eligible in-chain job
    # before falling through to cross-chain advancement.
    if next_in_chain_job_id is not None:
        await tao_job_service.submit_chain_job(
            project_id, next_in_chain_job_id, settings=settings
        )
    elif cross_chain_next_id is not None:
        await tao_job_service.submit_chain_job(
            project_id, cross_chain_next_id, settings=settings
        )


# ── Tick: the per-iteration scan ───────────────────────────────────────────


async def tick(settings: Settings) -> None:
    """Single poll pass: scan all projects, poll eligible jobs, advance chains.

    Exported for unit-testability — tests call this directly instead of
    starting the long-lived worker.
    """
    projects_dir = projects_root(settings.WORKSPACE_ROOT)
    if not projects_dir.exists():
        return

    submitted_interval_s = int(settings.TAO_POLL_MIN_INTERVAL_S)
    running_interval_s = int(settings.TAO_POLL_MIN_INTERVAL_RUNNING_S)
    now = datetime.now(UTC)

    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir():
            continue
        db_path = entry / "project.db"
        if not db_path.exists():
            continue
        # Archived projects are paused — TAO chains for them stay frozen
        # until unarchived. Cheap marker-file check keeps the per-tick
        # cost bounded as archived projects accumulate.
        if (entry / ".archived").exists():
            continue

        project_id = entry.name
        try:
            engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
            if engine is None:
                # Fallback for projects whose cache wasn't warmed at startup.
                engine = open_project_db(entry)
        except Exception as exc:  # pragma: no cover — defensive
            # Throttled WARN/DEBUG. First occurrence per
            # project → WARN; subsequent within ``_PROJECT_OPEN_WARNING_
            # THROTTLE_S`` → DEBUG. Eliminates the 30 lines/min/project
            # flood when a build-mode script holds an in-process lock on
            # one project while the dev backend scans the workspace.
            _log_project_open_failure(project_id, exc)
            continue

        # Outputs-fetch recovery: re-fire _handle_succeeded + chain advance for any
        # ``succeeded`` job whose outputs-fetch was interrupted. Runs
        # before the regular poll so a recovered chain can also progress
        # within the same tick.
        try:
            await _recover_stuck_outputs_fetch_in_project(
                project_id, engine=engine, settings=settings
            )
        except Exception:  # pragma: no cover — recovery scan never crashes tick
            logger.exception(
                "tao_polling: outputs-fetch recovery scan crashed for project %s",
                project_id,
            )

        revived = _migrate_legacy_lora_baseline_skips(project_id, engine=engine)
        if revived:
            logger.info(
                "tao_polling: revived %d legacy LoRA baseline auto-skip(s) "
                "for merged Student NIM evaluation (project=%s)",
                len(revived),
                project_id,
            )
            # Legacy chains may already have a remote quantize sibling in
            # flight. Start the independent local baseline immediately rather
            # than waiting for that remote job, and keep the poll tick
            # non-blocking while merge/NIM work runs.
            for eval_id, chain_id, parent_sequence in revived:
                background_manager.try_register(
                    f"local-baseline-recovery-{eval_id}",
                    _advance_after_terminal(
                        project_id,
                        chain_id=chain_id,
                        chain_sequence=parent_sequence,
                        engine=engine,
                        settings=settings,
                    ),
                    no_loop_warning=(
                        "Could not schedule legacy LoRA baseline recovery "
                        f"for {eval_id}"
                    ),
                )

        # Chain advance/halt recovery: resume chains frozen by a crash
        # between a terminal commit and its continuation (N+1 submit, or
        # dependent halt + suite roll-up). Runs after the outputs-fetch
        # scan so an in-flight recovered fetch keeps ownership of its
        # chain via the post-success task dedup.
        try:
            await _reconcile_stalled_chains_in_project(
                project_id, engine=engine, settings=settings
            )
        except Exception:  # pragma: no cover — recovery scan never crashes tick
            logger.exception(
                "tao_polling: chain reconciliation scan crashed for project %s",
                project_id,
            )

        # Select candidates for this tick in a short read-only session.
        with Session(engine) as session:
            stmt = (
                select(TAOJob.tao_job_id, TAOJob.status, TAOJob.last_polled_at)
                .where(TAOJob.project_id == project_id)
                .where(TAOJob.status.in_(sorted(_POLLABLE_STATUSES)))
            )
            rows = list(session.execute(stmt).all())

        for tao_job_id, status, last_polled_at in rows:
            if not _should_poll(
                status,
                last_polled_at,
                now=now,
                submitted_interval_s=submitted_interval_s,
                running_interval_s=running_interval_s,
            ):
                continue
            try:
                await _poll_single_job(
                    project_id,
                    tao_job_id,
                    engine=engine,
                    settings=settings,
                )
            except Exception:  # pragma: no cover — tick never dies
                logger.exception(
                    "tao_polling: per-job poll crashed for %s in project %s",
                    tao_job_id,
                    project_id,
                )


async def _await_pending_post_success_tasks() -> (
    None
):  # imported by tests/unit/test_tao_polling_service.py — kept private but listed in __all__
    """Test helper: await every currently-registered post-success task.

    Production code MUST NOT call this — the polling tick is intentionally
    non-blocking on post-success flows. Tests use this to drive a
    deterministic checkpoint between "tick fired" and "post-success
    finished" so they can assert on the final state.
    """
    pending = [
        t
        for tid, t in list(background_manager._tasks.items())  # pyright: ignore[reportPrivateUsage] — test-only helper, see docstring above
        if tid.startswith("post-success-")
    ]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def _post_success_flow(
    project_id: str,
    tao_job_id: str,
    *,
    external_id: str,
    action: str,
    chain_id: str | None,
    chain_sequence: int | None,
    engine: Engine,
    settings: Settings,
) -> None:
    """The full post-``succeeded`` flow: artifact fetch + downstream
    actions + chain advance.

    Dispatched as a non-blocking background task so a multi-GB
    safetensors download from one project doesn't head-of-line-block
    the polling tick for every other project. The
    ``outputs_fetch_status`` marker invariants still hold: ``_handle_succeeded`` flips
    ``outputs_fetch_status`` ``pending → in_progress`` at entry and
    ``→ completed | failed`` at exit.
    """
    try:
        await _handle_succeeded(
            project_id,
            tao_job_id,
            external_id=external_id,
            action=action,
            engine=engine,
            settings=settings,
        )
        await _advance_after_terminal(
            project_id,
            chain_id=chain_id,
            chain_sequence=chain_sequence,
            engine=engine,
            settings=settings,
        )
    except Exception:  # pragma: no cover — defensive, never crash
        logger.exception(
            "tao_polling: post-success flow crashed for tao_job %s "
            "(action=%s, project=%s)",
            tao_job_id,
            action,
            project_id,
        )


def _post_success_task_id(tao_job_id: str) -> str:
    """Globally-unique task id for the post-success-flow background task."""
    return f"post-success-{tao_job_id}"


def _dispatch_post_success_flow(
    project_id: str,
    tao_job_id: str,
    *,
    external_id: str,
    action: str,
    chain_id: str | None,
    chain_sequence: int | None,
    engine: Engine,
    settings: Settings,
    origin: str = "tick",
) -> bool:
    """Register ``_post_success_flow`` as a background task with dedup.

    Returns True if a new task was registered, False if one was already
    running for the same ``tao_job_id``. Dedup is essential: the polling
    tick fires every ``TAO_POLL_TICK_S`` seconds (10s default), and the
    outputs-fetch recovery scan fires every tick too — without dedup, a multi-GB
    download in flight would be re-attempted every tick. ``origin`` is
    captured in the log line so operators can trace whether the flow was
    triggered by a fresh transition (``"tick"``) or by post-restart
    recovery (``"recovery"``).
    """
    task_id = _post_success_task_id(tao_job_id)
    if task_id in background_manager.active_task_ids:
        return False
    logger.info(
        "tao_polling: dispatching post-success flow [%s] for tao_job %s "
        "(action=%s, project=%s)",
        origin,
        tao_job_id,
        action,
        project_id,
    )
    background_manager.register(
        task_id,
        _post_success_flow(
            project_id,
            tao_job_id,
            external_id=external_id,
            action=action,
            chain_id=chain_id,
            chain_sequence=chain_sequence,
            engine=engine,
            settings=settings,
        ),
    )
    return True


async def _recover_stuck_outputs_fetch_in_project(
    project_id: str,
    *,
    engine: Engine,
    settings: Settings,
) -> None:
    """Outputs-fetch recovery scan: re-fire ``_handle_succeeded`` + chain advance for
    any TAOJob that reached ``succeeded`` but whose ``outputs_fetch_status``
    is still ``pending`` or ``in_progress`` — the typical fingerprint of a
    backend crash mid-multi-GB-artifact-download. Once the marker is
    flipped to ``completed`` (or terminal ``failed``) the job is no longer
    eligible for recovery, so this is a true one-shot per stuck job.

    Idempotent: artifact downloads use atomic ``.part`` files truncated on
    re-open; ``register_from_tao_terminal`` and ``rescore_evaluate_job``
    tolerate re-execution; ``_advance_after_terminal`` only submits jobs
    still in ``not_started``.
    """
    with Session(engine) as session:
        stmt = (
            select(
                TAOJob.tao_job_id,
                TAOJob.tao_external_job_id,
                TAOJob.action,
                TAOJob.chain_id,
                TAOJob.chain_sequence,
            )
            .where(TAOJob.project_id == project_id)
            .where(TAOJob.status == "succeeded")
            .where(TAOJob.outputs_fetch_status.in_(["pending", "in_progress"]))
        )
        rows = list(session.execute(stmt).all())

    for tao_job_id, external_id, action, chain_id, chain_sequence in rows:
        if external_id is None:
            # Defensive: status=succeeded implies a round-trip with TAO
            # completed, so external_id should be populated. Skip rather
            # than re-fire with a None and surface as a log signal.
            logger.warning(
                "tao_polling: skipping outputs-fetch recovery for %s — "
                "tao_external_job_id is null (project=%s)",
                tao_job_id,
                project_id,
            )
            continue
        # Recovery is async-dispatched
        # so a multi-GB artifact download for one stuck job doesn't
        # head-of-line-block the polling tick for every other project.
        # Idempotent via dedup on the post_success_flow task_id.
        _dispatch_post_success_flow(
            project_id,
            tao_job_id,
            external_id=external_id,
            action=action,
            chain_id=chain_id,
            chain_sequence=chain_sequence,
            engine=engine,
            settings=settings,
            origin="recovery",
        )


def _migrate_legacy_lora_baseline_skips(
    project_id: str,
    *,
    engine: Engine,
) -> list[tuple[str, str, int]]:
    """Requeue baseline rows canceled by the former adapter-only policy.

    Releases before 2026-07-30 deliberately auto-skipped these rows because
    TAO evaluate cannot load LoRA adapters. The Blueprint now owns merge +
    local NIM evaluation, so the persisted auto-skip fingerprint is safe and
    precise to revive once. Ordinary user cancellations and base blocklist
    skips are untouched.
    """
    revived: list[tuple[str, str, int]] = []
    with Session(engine) as session:
        rows = (
            session.query(TAOJob)
            .filter(
                TAOJob.project_id == project_id,
                TAOJob.action == "evaluate",
                TAOJob.status == "canceled",
            )
            .all()
        )
        for row in rows:
            reason = row.chain_halted_reason or ""
            if not (
                reason.startswith(AUTO_SKIP_REASON_PREFIX)
                and "trained checkpoint is adapter-only" in reason
                and _is_lora_baseline_evaluate(session, row)
            ):
                continue
            parent = session.get(TAOJob, row.parent_tao_job_id)
            if parent is None or row.chain_id is None or parent.chain_sequence is None:
                continue
            row.status = "not_started"
            row.completed_at = None
            row.chain_halted_reason = None
            row.error_ref = None
            row.tao_status_raw = None
            row.progress = None
            revived.append((row.tao_job_id, row.chain_id, int(parent.chain_sequence)))
        if revived:
            session.commit()
    return revived


# Statuses meaning a chain member is actively owned by the live event
# flow (submission protocol or the poller). Chains with any member in
# one of these are skipped by the reconciliation scan below. Derived so
# a new status added to the state machine cannot silently fall outside
# the scan's skip-guard.
_IN_FLIGHT_STATUSES: frozenset[str] = (
    CANONICAL_STATUSES - TERMINAL_STATUSES - {"not_started"}
)


async def _reconcile_stalled_chains_in_project(
    project_id: str,
    *,
    engine: Engine,
    settings: Settings,
) -> None:
    """Chain advance/halt recovery scan: re-derive lost continuation work
    from persisted state alone.

    Chain advancement and failure halt/roll-up normally run as
    continuations of the live transition that the poller observes. Two
    crash windows leave persisted state no future event re-fires:

      (a) job N committed ``succeeded`` and its outputs fetch reached
          ``completed``, but the process died before
          ``_advance_after_terminal`` submitted N+1 — the dependent
          stays ``not_started`` forever;
      (b) a job committed ``failed``/``canceled``, but the process died
          before ``handle_terminal_failure`` halted its dependents and
          rolled up the suite — the dependents stay ``not_started``
          (never eligible: their parent is not ``succeeded``) and the
          TrainingSuite strands in ``running`` with no pollable rows.

    Mirrors the outputs-fetch recovery scan: fires every tick, selects
    on persisted fingerprints, and re-runs the exact helpers the live
    path uses. Chains with in-flight members (or an active post-success
    task) are owned by the live flow and skipped. Idempotent: (a)
    ``submit_chain_job`` refuses non-``not_started`` rows and the
    submission flips the fingerprint off; (b) fires only while unhalted
    ``not_started`` dependents exist, and the halt terminalizes them on
    the first pass. Chains whose only ``not_started`` rows are unstarted
    roots (parent ``None``) are deliberately untouched — starting a
    chain belongs to suite kickoff and cross-chain advancement.
    """
    halt_targets: list[tuple[str, str, int | None, str, str]] = []
    advance_targets: list[tuple[str, int | None]] = []

    with Session(engine) as session:
        stalled_chain_ids = [
            row[0]
            for row in session.execute(
                select(TAOJob.chain_id)
                .where(TAOJob.project_id == project_id)
                .where(TAOJob.chain_id.is_not(None))
                .where(TAOJob.status == "not_started")
                .distinct()
            )
            if row[0] is not None
        ]

        for chain_id in sorted(stalled_chain_ids):
            jobs = (
                session.query(TAOJob)
                .filter(
                    TAOJob.project_id == project_id,
                    TAOJob.chain_id == chain_id,
                )
                .order_by(TAOJob.chain_sequence.asc())
                .all()
            )
            if any(j.status in _IN_FLIGHT_STATUSES for j in jobs):
                continue
            if any(
                _post_success_task_id(j.tao_job_id)
                in background_manager.active_task_ids
                for j in jobs
            ):
                continue

            # Window (b): failed/canceled member with unhalted not_started
            # dependents. Auto-skip cancels are policy successes whose
            # siblings the live flow deliberately leaves eligible — never
            # halt behind them.
            stranded_root: TAOJob | None = None
            for j in jobs:
                if j.status not in ("failed", "canceled"):
                    continue
                if _is_auto_skip_reason(j.chain_halted_reason):
                    continue
                dependents = _transitive_dependents_in_session(
                    session,
                    project_id=project_id,
                    chain_id=chain_id,
                    root_job_id=j.tao_job_id,
                )
                if any(d.status == "not_started" for d in dependents):
                    stranded_root = j
                    break
            if stranded_root is not None:
                halt_targets.append(
                    (
                        stranded_root.tao_job_id,
                        chain_id,
                        stranded_root.chain_sequence,
                        stranded_root.action,
                        stranded_root.status,
                    )
                )
                continue

            # Window (a): an eligible next job whose succeeded parent
            # finished its outputs fetch. pending/in_progress parents
            # belong to the outputs-fetch recovery scan; a ``failed``
            # fetch marker is terminal-until-operator-retry and must not
            # advance past.
            eligible = _find_next_eligible_in_session(
                session, project_id=project_id, chain_id=chain_id
            )
            if eligible is None or eligible.parent_tao_job_id is None:
                continue
            parent = next(
                (j for j in jobs if j.tao_job_id == eligible.parent_tao_job_id),
                None,
            )
            if parent is None or parent.outputs_fetch_status != "completed":
                continue
            advance_targets.append((chain_id, parent.chain_sequence))

    for tao_job_id, chain_id, chain_sequence, action, terminal_status in halt_targets:
        logger.info(
            "tao_polling: reconciling stalled chain %s — re-running the "
            "terminal-failure flow for %s (project=%s)",
            chain_id,
            tao_job_id,
            project_id,
        )
        await handle_terminal_failure(
            project_id,
            tao_job_id,
            chain_id=chain_id,
            chain_sequence=chain_sequence,
            action=action,
            terminal_status=terminal_status,
            engine=engine,
            settings=settings,
            emit_root_failed=False,
        )
    for chain_id, chain_sequence in advance_targets:
        logger.info(
            "tao_polling: reconciling stalled chain %s — re-running chain "
            "advancement (project=%s)",
            chain_id,
            project_id,
        )
        await _advance_after_terminal(
            project_id,
            chain_id=chain_id,
            chain_sequence=chain_sequence,
            engine=engine,
            settings=settings,
        )


# ── Long-lived worker ──────────────────────────────────────────────────────


async def _tao_polling_worker(settings: Settings) -> None:
    """Run ``tick`` in a loop until shutdown is requested.

    ``settings`` fixes the tick cadence only; each tick re-resolves the
    live Settings singleton so a runtime settings reload
    (``POST /v1/secrets:set`` with ``persist=true`` — e.g. TAO configured
    after startup) reaches the poller without a backend restart. Found
    live 2026-07-14: a worker that closed over startup Settings kept
    failing preflight on ``TAO_API_BASE_URL=None`` forever — silently —
    after the operator configured TAO at runtime.
    """
    from vlm_feedback_loop.config import get_settings

    tick_s = max(1, int(settings.TAO_POLL_TICK_S))
    logger.info(
        "tao_polling worker starting (tick=%ds, submitted=%ds, running=%ds)",
        tick_s,
        settings.TAO_POLL_MIN_INTERVAL_S,
        settings.TAO_POLL_MIN_INTERVAL_RUNNING_S,
    )
    while not background_manager.is_shutting_down():
        try:
            await tick(get_settings())
        except Exception:
            logger.exception("tao_polling: tick failed (continuing)")
        # Responsive shutdown: sleep in small slices.
        elapsed = 0
        while elapsed < tick_s and not background_manager.is_shutting_down():
            await asyncio.sleep(1)
            elapsed += 1


def start_tao_polling(settings: Settings) -> None:
    """Register the polling worker with the background manager.

    Idempotent: a second call is a no-op if the worker is already running.
    """
    if _POLL_TASK_ID in background_manager.active_task_ids:
        return
    background_manager.register(_POLL_TASK_ID, _tao_polling_worker(settings))


__all__ = [
    # The leading-underscore symbols below are referenced by sibling
    # modules (other services, tests, scripts/capture_tao_fixtures.py).
    # Listing them in ``__all__`` quiets pyright's reportUnusedFunction
    # without per-line ignores.
    "_await_pending_post_success_tasks",
    "_plan_suite_advance",
    "_poll_single_job",
    "_should_poll",
    "start_tao_polling",
    "tick",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO job orchestration service.

Implements:
  - TAOJob CRUD (create, get, list).
  - 10-status state machine with allowed-transitions table.
  - 4-step submission protocol: submitting → POST → persist
    ``tao_external_job_id`` → submitted.  Failure at any step lands on
    ``failed`` with a sanitized ``error_ref``.
  - Case-insensitive raw-status mapping.
  - Action-specific dataset binding helper.
  - SHA-256 checksum of the TAO create-job payload persisted for integrity.
  - One-shot ``refresh=true`` poll against the remote FTMS endpoint
    (on-demand only — the background polling loop lives in
    ``tao_polling_service``).
  - Restart recovery: ``submitting`` + null external id → ``failed`` with
    ``error_ref="submission_interrupted"``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.schemas.tao_job import TAOJobCreateRequest
from vlm_feedback_loop.services.http_client import HttpResult, resilient_request
from vlm_feedback_loop.services.pagination import (
    InvalidCursorError,
    after_position_asc,
    decode_cursor,
    encode_cursor,
)
from vlm_feedback_loop.services.project_service import (
    get_project_engine,
    projects_root,
)
from vlm_feedback_loop.services.sse import sse_manager
from vlm_feedback_loop.services.tao_auth import (
    retry_once_on_401,
    tao_base_url,
    tao_preflight,
)

logger = logging.getLogger("vlm_feedback_loop.tao_job_service")


# ── State machine ────────────────────────────────────────────────────────────

# Canonical statuses in order of progression.  ``not_started`` and
# ``submitting`` are local-only states — they have no TAO-side equivalent.
CANONICAL_STATUSES: frozenset[str] = frozenset(
    {
        "not_started",
        "submitting",
        "submitted",
        "queued",
        "running",
        "paused",
        "succeeded",
        "failed",
        "canceled",
        "deleted",
    }
)

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "failed", "canceled", "deleted"}
)

# Allowed transitions.  ``any → deleted`` is handled separately
# as an escape hatch for explicit system-initiated deletions.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "not_started": frozenset({"submitting", "failed", "canceled"}),
    "submitting": frozenset({"submitted", "failed"}),
    "submitted": frozenset({"queued", "running", "failed", "canceled"}),
    "queued": frozenset({"running", "failed", "canceled"}),
    "running": frozenset({"succeeded", "failed", "paused", "canceled"}),
    "paused": frozenset({"running", "canceled"}),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "canceled": frozenset(),
    "deleted": frozenset(),
}


def can_transition(from_status: str, to_status: str) -> bool:
    """Return True if ``from_status → to_status`` is legal.

    ``any → deleted`` is allowed as a special case (system-initiated delete).

    ``submitted``, ``queued``, and
    ``paused`` may transition directly to ``succeeded``, skipping
    ``running``. TAO can move through ``running`` faster than the
    polling cadence (30s for ``submitted``, 60s for ``running``) can
    observe — a 3-epoch SFT on a small dataset can finish in ~7 min,
    well within a single ``submitted → submitted`` poll window. Observed
    reality wins: when TAO returns ``Done`` while the local row still
    reads ``submitted``/``queued``/``paused``, persist the terminal
    status without forcing an artificial intermediate ``running`` write.
    ``not_started`` and ``submitting`` still cannot reach succeeded
    directly (those require a TAO ``tao_external_job_id`` round-trip
    first; succeeded without an external_id would be incoherent).
    """
    if to_status == "deleted" and from_status in CANONICAL_STATUSES:
        return from_status != "deleted"
    if to_status == "succeeded" and from_status in {
        "submitted",
        "queued",
        "paused",
    }:
        # Skipping ``running`` is the fast-completion case. ``not_started`` and
        # ``submitting`` still cannot reach succeeded directly (those
        # require a TAO ``tao_external_job_id`` round-trip first).
        return True
    allowed = ALLOWED_TRANSITIONS.get(from_status, frozenset())
    return to_status in allowed


# Case-insensitive raw→canonical status mapping.
_RAW_STATUS_MAP: dict[str, str] = {
    "done": "succeeded",
    "failed": "failed",
    "error": "failed",  # FTMS 6.26.3 reports container errors as "Error"
    "errored": "failed",
    "running": "running",
    "queued": "queued",
    "pending": "queued",
    "paused": "paused",
    "canceled": "canceled",
    "cancelled": "canceled",
}


def map_tao_raw_status(raw: str | None, *, current: str) -> str:
    """Map a TAO raw status string to a canonical status.

    Unknown or missing raw status is handled conservatively:
      * If the current local status is terminal, preserve it.
      * Otherwise, fall back to ``running`` (non-terminal, safe default).
    """
    if raw is None:
        return current
    lowered = raw.strip().lower()
    if lowered in _RAW_STATUS_MAP:
        return _RAW_STATUS_MAP[lowered]
    # Unknown raw status — conservative fallback.
    if current in TERMINAL_STATUSES:
        return current
    if current == "not_started":
        return "queued"
    return "running"


# ── TAO failure classification ──────────────────────────────────────────────
#
# The Training Job Monitor screen renders ``error_ref`` from
# the TAOJob row. Raw FTMS errors like ``HTTP 400`` are useless to an
# SME. ``classify_tao_failure`` maps known failure shapes to a friendly
# one-line message that names the actionable cause; the raw provider
# detail is preserved at the end of the string for operators who need
# it. Each pattern matches a failure shape observed against TAO FTMS
# 6.26.3 + cosmos-rl 6.26.3; add new patterns here as new failure
# modes surface.

_GATED_HF_HINTS = (
    "gated repo",
    "you must have access",
    "huggingface.co",
    "401 client error",
    "cannot access gated",
)
_PARALLELISM_HINT = "invalid parallel dims"
_COMPILE_HINT = "compile is not supported"
_FILE_NOT_FOUND_HINT = "filenotfounderror"
_HF_LOCAL_PATH_REJECT_HINT = "repo id must be in the form"
_TAO_400_HINTS = ("missing required field", "validation error", "invalid enum member")


def classify_tao_failure(raw_error: str | None) -> tuple[str, str | None]:
    """Map a TAO/cosmos-rl failure string to ``(friendly_message, category)``.

    ``category`` is a short machine-readable token (e.g.
    ``hf_gated_repo``) that the UI / Action Request generator can branch
    on. ``None`` means the failure didn't match a known pattern; the
    caller should fall back to the raw ``error_ref``.
    """
    if not raw_error:
        return ("", None)
    lowered = raw_error.lower()

    if any(h in lowered for h in _GATED_HF_HINTS):
        return (
            "Cosmos-RL container hit HuggingFace gated-repo (HTTP 401). "
            "The Cosmos Reason2 family is gated; the worker authenticates "
            "with HF_TOKEN passed via TAO docker_env_vars. Verify "
            "HF_TOKEN is set in ~/.vlm_feedback_loop/.env and that the "
            "account has accepted the Cosmos Reason2 license at "
            "https://huggingface.co/nvidia/Cosmos-Reason2-2B.",
            "hf_gated_repo",
        )
    if _PARALLELISM_HINT in lowered:
        return (
            "Cosmos-RL parallelism plan rejected by WORLD_SIZE assertion "
            "— the cosmos-rl container has fewer GPUs than the spec's "
            "policy.parallelism.tp_size * dp_shard_size product. Adjust "
            "num_gpu on the TAO request or override "
            "policy.parallelism.dp_shard_size to match the allocated GPU "
            "count.",
            "parallelism_mismatch",
        )
    if _COMPILE_HINT in lowered:
        return (
            "Cosmos-RL refused torch.compile for HFModel. The Blueprint "
            "spec must set train.compile=false; if you see this on a "
            "Blueprint-built job please report — the override is wired "
            "in `_build_train_payload`.",
            "compile_unsupported",
        )
    if _HF_LOCAL_PATH_REJECT_HINT in lowered:
        return (
            "Cosmos-RL parsed policy.model_name_or_path as a HuggingFace "
            "repo id (must be 'namespace/name'). The Blueprint spec must "
            "use the airgapped mount path /ptm/huggingface_models when "
            "base_experiment_ids is set; if you see this on a "
            "Blueprint-built job please report.",
            "hf_local_path_rejected",
        )
    if _FILE_NOT_FOUND_HINT in lowered and "annotation" in lowered:
        return (
            "Cosmos-RL could not open the annotation_path as a JSON file. "
            "media_path and annotation_path must be distinct URLs — the "
            "Blueprint spec uploads annotations.json as a sidecar "
            "alongside the .tar.gz; if you see this on a Blueprint-built "
            "job please report.",
            "annotation_path_not_a_file",
        )
    if any(h in lowered for h in _TAO_400_HINTS):
        return (
            "TAO FTMS rejected the job request. Likely cause: a required "
            "field (network_arch / base_experiment_ids / workspace / "
            "name) is missing from tao_create_job_request, or a key in "
            "docker_env_vars / specs is not in TAO's allowed enum. "
            f"Raw FTMS detail: {raw_error}",
            "tao_validation_error",
        )

    return (raw_error, None)


def format_failure_error_ref(raw_msg: str) -> str | None:
    """Build the persisted ``error_ref`` for a failed TAO job.

    Runs the raw FTMS ``status_msg`` through :func:`classify_tao_failure`
    and, when a known pattern matched, appends the friendly explanation
    to the raw detail; the result is sanitized either way. Shared by the
    one-shot ``refresh=true`` poll (:func:`get_tao_job`) and the
    background polling loop (``tao_polling_service``) so both persist
    identical failure messages.
    """
    friendly, _category = classify_tao_failure(raw_msg)
    if friendly and friendly != raw_msg:
        return sanitize_error(f"{raw_msg} — {friendly}")
    return sanitize_error(raw_msg)


@dataclass(frozen=True)
class PollApplyOutcome:
    """Result of :func:`apply_poll_result` — what the caller needs next.

    ``poll_ok`` is False when the remote poll itself failed (only
    ``last_polled_at`` + ``poll_error_ref`` were written). On success,
    ``status_changed`` / ``new_status`` report whether a legal status
    transition was persisted — the background polling loop uses them to
    drive SSE emission and terminal dispatch; the one-shot refresh
    ignores them.
    """

    poll_ok: bool
    status_changed: bool
    new_status: str | None


def apply_poll_result(
    job: TAOJob,
    poll_result: dict[str, Any],
    now: str,
    *,
    failed_error_ref_requires_transition: bool,
) -> PollApplyOutcome:
    """Apply a ``poll_tao_job`` result onto a TAOJob row (no commit).

    The single persist-poll-result implementation shared by the
    background polling loop (``tao_polling_service._poll_single_job``)
    and the one-shot ``refresh=true`` poll (:func:`get_tao_job`). The
    caller owns the session and commits.

    On poll failure only ``last_polled_at`` and ``poll_error_ref`` are
    written; the last-known-good status is preserved. On success the raw
    status is mapped and persisted when the transition is legal,
    ``started_at`` / ``completed_at`` are stamped on first entry into
    ``running`` / a terminal status, and progress/outputs are updated
    when present.

    When a job lands on ``failed`` because the worker crashed (HF gated
    repo, parallelism mismatch, compile refusal, etc), FTMS's
    ``status_msg`` plus the friendly classification is persisted on
    ``error_ref`` so the Training Job Monitor screen can render an
    actionable failure cause instead of just "failed". An existing
    ``error_ref`` (e.g. from a submission-time failure) is always
    preserved. ``failed_error_ref_requires_transition`` controls when
    that backfill fires:

    - ``True`` (the polling loop): only on the actual legal transition
      into ``failed``.
    - ``False`` (the one-shot refresh): whenever the raw status maps to
      ``failed``, even without a transition — e.g. a ``paused`` job
      whose raw status maps to ``failed`` (no legal transition) still
      gets its failure cause recorded.

    ``status_msg`` is read with ``.get`` deliberately — poll results
    from test doubles and older callers may omit the key entirely.
    """
    job.last_polled_at = now
    if not poll_result["success"]:
        job.poll_error_ref = sanitize_error(poll_result["error"])
        return PollApplyOutcome(poll_ok=False, status_changed=False, new_status=None)

    # Clear any stale poll error now that we succeeded.
    job.poll_error_ref = None
    job.tao_status_raw = poll_result["tao_status_raw"]

    status_changed = False
    new_status: str | None = None
    mapped = map_tao_raw_status(poll_result["tao_status_raw"], current=job.status)
    transitioned = mapped != job.status and can_transition(job.status, mapped)
    if transitioned:
        job.status = mapped
        status_changed = True
        new_status = mapped
        if mapped == "running" and not job.started_at:
            job.started_at = now
        if mapped in TERMINAL_STATUSES and not job.completed_at:
            job.completed_at = now

    if (
        mapped == "failed"
        and (transitioned or not failed_error_ref_requires_transition)
        and not job.error_ref
        and poll_result.get("status_msg")
    ):
        job.error_ref = format_failure_error_ref(poll_result["status_msg"])

    if poll_result["progress"] is not None:
        job.progress = poll_result["progress"]
    if poll_result["outputs"] is not None:
        job.outputs = poll_result["outputs"]

    return PollApplyOutcome(
        poll_ok=True, status_changed=status_changed, new_status=new_status
    )


# ── Dataset binding ──────────────────────────────────────────────────────────


def apply_dataset_binding(
    tao_create_job_request: dict[str, Any],
    *,
    action: str,
    annotation_path: str,
    media_root: str,
) -> dict[str, Any]:
    """Inject action-specific dataset bindings into ``specs``.

    Wire shape verified against TAO FTMS 6.26.3 + Cosmos-RL 6.26.3.
    Each cosmos-rl action class consumes a different config shape:

    * ``train`` — Cosmos-RL ``CustomConfig.train_dataset`` (Pydantic
      schema in ``/opt/cosmos_rl/tao_sft_example.py``); the
      ``custom.dataset`` shape documented upstream was a docs error that
      surfaces in the container as ``ValidationError: train_dataset Field
      required``.
    * ``evaluate`` — Cosmos-RL ``ITSEvaluator`` reads
      ``config["dataset"]`` (top-level ``dataset`` table — see
      ``cosmos_rl/evaluation/its_evaluator.py``); the
      ``custom.val_dataset`` shape used by training crashes here with
      ``KeyError: 'dataset'``.
    * ``quantize`` — top-level ``specs.dataset.{media_dir, annotation_path}``.
      cosmos-rl-quantize's argparse CLI accepts ``--media_dir``,
      NOT ``--media_path`` (the latter is train-only), so a
      ``custom.train_dataset.{media_path, ...}`` shape mirroring train
      is rejected by the cosmos-rl-quantize entrypoint. Quantize's
      calibration pass replays the training corpus, so the dataset
      content is the same as train — only the binding key naming
      differs.
    * ``inference`` — no binding (caller responsibility).

    Returns a deep-copied payload with the binding applied; the input
    is not mutated.  Preserves every other TAO-native field already
    present.
    """
    payload = deepcopy(tao_create_job_request)
    specs = payload.setdefault("specs", {})

    if action == "train":
        custom = specs.setdefault("custom", {})
        dataset = custom.setdefault("train_dataset", {})
        dataset["media_path"] = media_root
        dataset["annotation_path"] = annotation_path
    elif action == "evaluate" or action == "quantize":
        # Quantize uses the evaluate-style top-level dataset binding
        # so cosmos-rl-quantize gets ``--media_dir`` (its CLI rejects the
        # train-style ``--media_path``).
        dataset = specs.setdefault("dataset", {})
        dataset["media_dir"] = media_root
        dataset["annotation_path"] = annotation_path
    # action="inference": no binding

    return payload


# ── Checksum ────────────────────────────────────────────────────────────────


def compute_request_checksum(payload: dict[str, Any]) -> str:
    """Return SHA-256 hex digest of the canonical JSON encoding of ``payload``."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── TAO remote submission + polling (mockable) ─────────────────────────────
#
# ``_submit_to_tao`` and ``poll_tao_job`` are the sole injection points for
# tests.  Production uses ``resilient_request`` with the configured TAO
# credentials (SSH-tunneled endpoint in local dev, direct reach otherwise).


async def _tao_request_with_401_retry(
    method: str,
    url: str,
    *,
    settings: Settings,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
) -> HttpResult:
    """Issue one TAO request with the shared single-401-retry contract.

    A submit/poll/cancel that meets an expired cached JWT would otherwise
    fail on every attempt and wedge the long-running polling loop
    permanently on 401. Delegates to :func:`tao_auth.retry_once_on_401`
    (shared with ``tao_base_experiment_provisioning_service``); on a 401
    the retry rebuilds fresh auth headers, and if re-authentication
    itself fails the original result is returned so the caller's
    existing error handling still fires.
    """
    result = await resilient_request(
        method,
        url,
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=settings.HTTP_MAX_RETRIES,
        headers=headers,
        json_body=json_body,
    )
    return await retry_once_on_401(
        result,
        method=method,
        url=url,
        settings=settings,
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=settings.HTTP_MAX_RETRIES,
        json_body=json_body,
    )


async def _submit_to_tao(
    tao_create_job_request: dict[str, Any],
    *,
    settings: Settings,
) -> dict[str, Any]:
    """POST the job payload to TAO and return the external job id.

    Returns
    -------
    dict with keys:
      * ``success: bool``
      * ``tao_external_job_id: str | None``
      * ``error: str | None`` — sanitized provider error
    """
    headers, preflight_error = await tao_preflight(settings)
    if headers is None:
        return {
            "success": False,
            "tao_external_job_id": None,
            "error": preflight_error,
        }

    url = f"{tao_base_url(settings)}/orgs/{settings.TAO_ORG_NAME}/jobs"

    # docker_env_vars.HF_TOKEN passthrough — Cosmos Reason2 family is
    # gated on HuggingFace, so the cosmos-rl container's
    # `transformers.from_pretrained()` call requires HF auth even when
    # base_experiment_ids are registered via :load_airgapped (without
    # HF_TOKEN the worker hits HTTP 401 gated-repo on every retry; with
    # HF_TOKEN injected via the TAO-whitelisted
    # ``docker_env_vars.HF_TOKEN`` field it loads cleanly).
    # Injecting here so every chain submission (train, evaluate,
    # quantize) gets the token uniformly without each payload builder
    # repeating itself; also covers ``submit_chain_job`` which advances
    # chains directly via the polling loop.
    payload = tao_create_job_request
    if settings.HF_TOKEN:
        payload = dict(tao_create_job_request)
        existing_env = dict(payload.get("docker_env_vars") or {})
        existing_env.setdefault("HF_TOKEN", settings.HF_TOKEN)
        payload["docker_env_vars"] = existing_env

    # Diagnostic: log the JSON body being POSTed so the operator can
    # verify the FTMS-required top-level fields (``parent_job_id``,
    # ``base_experiment_ids``, ``workspace``) made it onto the wire.
    # Truncated to keep log lines bounded.
    try:
        redacted = dict(payload)
        if redacted.get("docker_env_vars"):
            env_raw: Any = redacted.get("docker_env_vars") or {}
            env_dict: dict[str, Any] = (
                cast("dict[str, Any]", env_raw) if isinstance(env_raw, dict) else {}
            )
            redacted["docker_env_vars"] = {
                str(k): (
                    "***" if "TOKEN" in str(k).upper() or "KEY" in str(k).upper() else v
                )
                for k, v in env_dict.items()
            }
        logger.info(
            "TAO POST %s body=%s",
            url,
            json.dumps(redacted, default=str)[:2000],
            extra={"component": "tao_job_service"},
        )
    except Exception:  # pragma: no cover — diagnostic only
        pass

    result = await _tao_request_with_401_retry(
        "POST", url, settings=settings, headers=headers, json_body=payload
    )

    if result.error_class is not None:
        # Surface the structured FTMS error_desc to the operator. Without
        # this the UI just shows "HTTP 400" which is useless. The body
        # is parsed JSON when the TAO response carried one; fall back to
        # error_detail otherwise.
        raw_desc: str | None = None
        result_body: Any = result.body
        if isinstance(result_body, dict):
            body_dict = cast("dict[str, Any]", result_body)
            raw_desc_raw: Any = body_dict.get("error_desc") or body_dict.get("error")
            raw_desc = raw_desc_raw if isinstance(raw_desc_raw, str) else None
        provider_detail = raw_desc or result.error_detail or ""
        friendly, _category = classify_tao_failure(provider_detail)
        if friendly and friendly != provider_detail:
            error_message = f"TAO submission failed ({result.error_detail}): {friendly}"
        else:
            error_message = (
                f"TAO submission failed: {provider_detail or result.error_detail}"
            )
        return {
            "success": False,
            "tao_external_job_id": None,
            "error": error_message,
        }

    body: Any = result.body
    job_id: str | None = None
    if isinstance(body, dict):
        body_dict = cast("dict[str, Any]", body)
        # FTMS responds with the created job id in one of several keys
        # depending on the request kind; accept the common shapes.
        for key in ("id", "job_id", "job", "tao_external_job_id"):
            value: Any = body_dict.get(key)
            if isinstance(value, str):
                job_id = value
                break
            if isinstance(value, dict):
                value_dict = cast("dict[str, Any]", value)
                inner = value_dict.get("id")
                if isinstance(inner, str):
                    job_id = inner
                    break

    if not job_id:
        return {
            "success": False,
            "tao_external_job_id": None,
            "error": "TAO submission succeeded but no job id found in response",
        }

    return {"success": True, "tao_external_job_id": job_id, "error": None}


async def _cancel_tao_external(
    tao_external_job_id: str,
    *,
    settings: Settings,
) -> dict[str, Any]:
    """POST :cancel for a TAO job.

    Returns
    -------
    dict with keys:
      * ``success: bool``
      * ``error: str | None`` — sanitized provider error.
      * ``error_class: str`` (transport failures only) — ``"timeout"`` /
        ``"endpoint_error"`` from :func:`resilient_request`, so the caller
        can surface 504 vs 503 per the documented error contract.
    """
    headers, preflight_error = await tao_preflight(settings)
    if headers is None:
        return {"success": False, "error": preflight_error}

    url = (
        f"{tao_base_url(settings)}/orgs/{settings.TAO_ORG_NAME}"
        f"/jobs/{tao_external_job_id}:cancel"
    )

    result = await _tao_request_with_401_retry(
        "POST", url, settings=settings, headers=headers, json_body={}
    )

    if result.error_class is not None:
        return {
            "success": False,
            "error": f"TAO cancel failed: {result.error_detail}",
            "error_class": result.error_class,
        }
    return {"success": True, "error": None}


async def request_tao_job_cancel(
    tao_external_job_id: str,
    *,
    settings: Settings,
) -> dict[str, Any]:
    """Best-effort TAO-side cancellation used by suite orchestration."""
    return await _cancel_tao_external(tao_external_job_id, settings=settings)


def _job_details_entry(
    body: dict[str, Any], tao_external_job_id: str
) -> dict[str, Any] | None:
    """Return ``job_details[<external_id>]`` from a 6.26.3 poll body.

    Falls back to the sole entry when the id key is absent but exactly
    one entry exists (defensive — observed bodies always key by id).
    """
    job_details_raw: Any = body.get("job_details")
    if not isinstance(job_details_raw, dict):
        return None
    job_details = cast("dict[str, Any]", job_details_raw)
    entry_raw: Any = job_details.get(tao_external_job_id)
    if isinstance(entry_raw, dict):
        return cast("dict[str, Any]", entry_raw)
    if len(job_details) == 1:
        sole_raw: Any = next(iter(job_details.values()))
        if isinstance(sole_raw, dict):
            return cast("dict[str, Any]", sole_raw)
    return None


def _parse_eta_seconds(value: Any) -> float | None:
    """Parse a JobResult ``eta`` into seconds.

    The OpenAPI spec types it ``string`` with no format contract; accept
    plain numbers, numeric strings, and ``[H:]MM:SS[.frac]`` clock
    strings. Anything else → None (never guess).
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if 2 <= len(parts) <= 3:
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        seconds = 0.0
        for n in nums:
            seconds = seconds * 60.0 + n
        return seconds
    return None


def _derive_progress_from_job_details(
    job_entry: dict[str, Any],
    *,
    network_arch: str | None = None,
    action: str | None = None,
) -> dict[str, Any] | None:
    """Map FTMS 6.26.3 JobResult fields onto the §7 progress shape.

    ``epoch``/``max_epoch``/``cur_iter`` are integers, ``eta``/
    ``time_per_*`` are strings, ``key_metric`` is a number (OpenAPI
    ``JobResult``; live bodies in tests/fixtures/tao/). Returns None
    when the entry carries no recognizable JobResult signal, so
    pre-training phases (download/init) don't fabricate an empty
    progress panel.

    ``network_arch`` gates the fabricated-metric drop: cosmos-rl never populates
    ``key_metric`` — FTMS sends a literal ``0.0`` on every poll of every
    cosmos-rl job, including ``Done`` ones (pinned by
    tests/fixtures/tao/job_status_done_train.json) — so surfacing it
    would render a fabricated "key_metric 0.0000" on each JobCard. Drop
    exactly the cosmos-rl ``0.0``; a nonzero value (a future cosmos-rl
    that genuinely reports the metric) and every other backend's values
    pass through verbatim.
    """
    # JobResult is shared by every FTMS action.  In particular, quantize
    # currently fills ``max_epoch`` and ``time_per_epoch`` with its generic
    # work-unit telemetry (for example 100 and 0:00:01).  Those values are
    # not training epochs and presenting them as such is actively
    # misleading.  Epoch-specific fields are therefore valid only for
    # train jobs; action-neutral ETA and iteration telemetry remain useful
    # for the other actions.
    is_epoch_action = action in (None, "train")
    epoch_raw: Any = job_entry.get("epoch")
    max_epoch_raw: Any = job_entry.get("max_epoch")
    epoch_current = (
        epoch_raw if is_epoch_action and isinstance(epoch_raw, int) else None
    )
    epoch_total = (
        max_epoch_raw if is_epoch_action and isinstance(max_epoch_raw, int) else None
    )
    eta_seconds = _parse_eta_seconds(job_entry.get("eta"))

    metrics_latest: dict[str, Any] = {}
    key_metric_raw: Any = job_entry.get("key_metric")
    if (
        isinstance(key_metric_raw, (int, float))
        and not isinstance(key_metric_raw, bool)
        and not (network_arch == "cosmos-rl" and float(key_metric_raw) == 0.0)
    ):
        metrics_latest["key_metric"] = key_metric_raw
    cur_iter_raw: Any = job_entry.get("cur_iter")
    if isinstance(cur_iter_raw, int):
        metrics_latest["cur_iter"] = cur_iter_raw
    time_keys = (
        ("time_per_epoch", "time_per_iter") if is_epoch_action else ("time_per_iter",)
    )
    for time_key in time_keys:
        time_raw: Any = job_entry.get(time_key)
        if isinstance(time_raw, str) and time_raw.strip():
            metrics_latest[time_key] = time_raw

    if (
        epoch_current is None
        and epoch_total is None
        and eta_seconds is None
        and not metrics_latest
    ):
        return None
    return {
        "epoch_current": epoch_current,
        "epoch_total": epoch_total,
        "eta_seconds": eta_seconds,
        "metrics_latest": metrics_latest or None,
        "metrics_history_ref": None,
    }


async def poll_tao_job(
    tao_external_job_id: str,
    *,
    settings: Settings,
) -> dict[str, Any]:
    """GET the current state of a TAO job (one-shot poll).

    Returns
    -------
    dict with keys:
      * ``success: bool``
      * ``tao_status_raw: str | None``
      * ``progress: dict | None``
      * ``outputs: dict | None``
      * ``error: str | None``
    """
    headers, preflight_error = await tao_preflight(settings)
    if headers is None:
        return {
            "success": False,
            "tao_status_raw": None,
            "progress": None,
            "outputs": None,
            "error": preflight_error,
        }

    url = (
        f"{tao_base_url(settings)}/orgs/{settings.TAO_ORG_NAME}"
        f"/jobs/{tao_external_job_id}"
    )

    result = await _tao_request_with_401_retry(
        "GET", url, settings=settings, headers=headers
    )

    if result.error_class is not None:
        return {
            "success": False,
            "tao_status_raw": None,
            "progress": None,
            "outputs": None,
            "error": f"TAO poll failed: {result.error_detail}",
        }

    result_body: Any = result.body
    body: dict[str, Any] = (
        cast("dict[str, Any]", result_body) if isinstance(result_body, dict) else {}
    )
    raw_status: Any = body.get("status") or body.get("state") or body.get("job_status")
    progress_raw: Any = body.get("progress")
    progress: dict[str, Any] | None = (
        cast("dict[str, Any]", progress_raw) if isinstance(progress_raw, dict) else None
    )
    outputs_raw: Any = body.get("outputs")
    outputs: dict[str, Any] | None = (
        cast("dict[str, Any]", outputs_raw) if isinstance(outputs_raw, dict) else None
    )

    # FTMS 6.26.3 never returns a top-level ``progress`` key (confirmed
    # against live poll bodies, 2026-07-14 — fixtures in
    # tests/fixtures/tao/job_status_*.json). Training progress lives in
    # ``job_details.{external_id}`` as JobResult fields; derive the
    # documented progress shape from there when ``progress`` is absent.
    job_entry = _job_details_entry(body, tao_external_job_id)
    if progress is None and job_entry is not None:
        network_arch_raw: Any = body.get("network_arch")
        action_raw: Any = body.get("action")
        progress = _derive_progress_from_job_details(
            job_entry,
            network_arch=(
                network_arch_raw if isinstance(network_arch_raw, str) else None
            ),
            action=action_raw if isinstance(action_raw, str) else None,
        )

    # Extract any container-side status message (cosmos-rl crash detail,
    # TAO scheduler error, etc). Older FTMS releases surface it at
    # ``job_details.{status_msg|message|...}``; 6.26.3 nests it under
    # ``job_details.{external_id}.detailed_status.message``. The polling
    # persistence layer feeds this into ``classify_tao_failure`` so the
    # UI gets an actionable message rather than a bare canonical status.
    status_msg: str | None = None
    job_details_raw: Any = body.get("job_details")
    if isinstance(job_details_raw, dict):
        job_details = cast("dict[str, Any]", job_details_raw)
        for key in ("status_msg", "message", "error", "error_msg"):
            value: Any = job_details.get(key)
            if isinstance(value, str) and value.strip():
                status_msg = value
                break
    if status_msg is None and job_entry is not None:
        detailed_raw: Any = job_entry.get("detailed_status")
        if isinstance(detailed_raw, dict):
            detailed = cast("dict[str, Any]", detailed_raw)
            for key in ("message", "status_msg", "error", "error_msg"):
                nested_value: Any = detailed.get(key)
                if isinstance(nested_value, str) and nested_value.strip():
                    status_msg = nested_value
                    break

    return {
        "success": True,
        "tao_status_raw": str(raw_status) if raw_status is not None else None,
        "progress": progress,
        "outputs": outputs,
        "status_msg": status_msg,
        "error": None,
    }


# ── Record → dict serialization ─────────────────────────────────────────────


def _job_to_dict(job: TAOJob) -> dict[str, Any]:
    """Convert a TAOJob ORM row to the response dict shape."""
    dataset_export_ids: list[str] = [str(x) for x in job.dataset_export_ids]

    return {
        "tao_job_id": job.tao_job_id,
        "project_id": job.project_id,
        "status": job.status,
        "tao_status_raw": job.tao_status_raw,
        "action": job.action,
        "training_backend": job.training_backend,
        "training_policy_type": job.training_policy_type,
        "student_base_model_config_id": job.student_base_model_config_id,
        "dataset_export_ids": dataset_export_ids,
        "job_config": job.job_config or {},
        "tao_create_job_request": job.tao_create_job_request or {},
        "tao_external_job_id": job.tao_external_job_id,
        "progress": job.progress,
        "outputs": job.outputs,
        "parent_tao_job_id": job.parent_tao_job_id,
        "chain_id": job.chain_id,
        "chain_sequence": job.chain_sequence,
        "chain_halted_reason": job.chain_halted_reason,
        "preflight_result": job.preflight_result,
        "error_ref": job.error_ref,
        "poll_error_ref": job.poll_error_ref,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "last_polled_at": job.last_polled_at,
    }


# ── Error-ref sanitization ──────────────────────────────────────────────────

_SECRET_PATTERNS = ("Bearer ", "nvapi-", "ngc-")


def sanitize_error(text: str | None) -> str | None:
    """Redact obvious secret patterns from persisted error strings."""
    if text is None:
        return None
    sanitized = text
    for marker in _SECRET_PATTERNS:
        idx = sanitized.find(marker)
        while idx >= 0:
            # Redact from the marker to the next whitespace/quote.
            end = idx + len(marker)
            while end < len(sanitized) and sanitized[end] not in (
                " ",
                "\t",
                "\n",
                '"',
                "'",
                ",",
                ")",
                "}",
            ):
                end += 1
            sanitized = sanitized[:idx] + "[REDACTED]" + sanitized[end:]
            idx = sanitized.find(marker)
    return sanitized


# ── Validation helpers ──────────────────────────────────────────────────────


def _validate_dataset_exports(
    session: Session,
    project_id: str,
    export_ids: list[str],
    action: str,
) -> str | None:
    """Validate dataset_export_ids. Returns None on success or error string."""
    if not export_ids:
        return "validation: dataset_export_ids must be non-empty"

    # Load every referenced export.
    rows = (
        session.execute(
            select(DatasetExport).where(DatasetExport.dataset_export_id.in_(export_ids))
        )
        .scalars()
        .all()
    )
    found_ids = {row.dataset_export_id for row in rows}
    missing = [eid for eid in export_ids if eid not in found_ids]
    if missing:
        return f"not found: dataset_export(s) not found: {', '.join(missing)}"

    # Reject cross-project references.
    cross_project = [
        row.dataset_export_id for row in rows if row.project_id != project_id
    ]
    if cross_project:
        return (
            f"not found: dataset_export(s) belong to a different project: "
            f"{', '.join(cross_project)}"
        )

    # Action-specific consistency.
    if action == "train":
        intents = {row.dataset_intent for row in rows}
        if intents != {"training"}:
            return (
                f"validation: MIXED_EXPORT_FIELD_MODE: train action requires "
                f"all dataset_exports to have dataset_intent='training', "
                f"found: {sorted(intents)}"
            )
        field_modes = {row.export_field_mode for row in rows}
        if len(field_modes) > 1:
            return (
                f"validation: MIXED_EXPORT_FIELD_MODE: all dataset_exports "
                f"must share the same export_field_mode, "
                f"found: {sorted(field_modes)}"
            )

    return None


def _validate_student_base(
    session: Session,
    project_id: str,
    model_config_id: str,
) -> str | None:
    """Validate the student_base_model_config_id. Returns None or error string."""
    model_config = (
        session.query(ModelConfig)
        .filter_by(model_config_id=model_config_id, project_id=project_id)
        .first()
    )
    if model_config is None:
        return (
            f"not found: ModelConfig {model_config_id} not found "
            f"in project {project_id}"
        )
    # eligible_roles may be stored as list (native JSON) or JSON-encoded str
    # depending on how the row was seeded; mirror model_config_service.
    roles = model_config.eligible_roles
    if isinstance(roles, str):
        try:
            roles = json.loads(roles)
        except (ValueError, TypeError):
            roles = []
    if not isinstance(roles, list) or "student_base" not in roles:
        return (
            f"validation: ModelConfig {model_config_id} does not have "
            f"student_base role (eligible_roles={roles})"
        )
    return None


# ── Public service API ─────────────────────────────────────────────────────


def find_train_job_for_chain(
    session: Session,
    *,
    project_id: str,
    chain_id: str | None,
) -> TAOJob | None:
    """Return the chain's originating ``train`` TAOJob, if any.

    The single implementation of the "find the chain's train job" lookup,
    shared with ``student_model_service`` (lineage walking). Every chain
    has exactly one train job (created at ``chain_sequence=1`` by the
    training-suite service), so action alone identifies it.
    """
    if not chain_id:
        return None
    return (
        session.query(TAOJob)
        .filter(
            TAOJob.project_id == project_id,
            TAOJob.chain_id == chain_id,
            TAOJob.action == "train",
        )
        .first()
    )


def find_suite_for_chain(
    session: Session,
    *,
    project_id: str,
    chain_id: str | None,
) -> TrainingSuite | None:
    """Return the TrainingSuite that owns this ``chain_id``, if any.

    The single implementation of the suite-owns-chain scan, shared by
    ``tao_polling_service`` (suite roll-up) and ``student_model_service``
    (guidance lineage lookup).
    """
    if not chain_id:
        return None
    suites = (
        session.query(TrainingSuite)
        .filter(TrainingSuite.project_id == project_id)
        .all()
    )
    for s in suites:
        if chain_id in (s.chain_ids_ordered or []):
            return s
    return None


def _check_evaluate_field_mode_lineage(
    session: Session,
    *,
    project_id: str,
    evaluate_job: TAOJob,
) -> str | None:
    """Export-field-mode lineage guard: the evaluate job's DatasetExport
    must share the train job's ``export_field_mode``.

    For suite-created chains this invariant holds by
    construction — both exports are persisted with the same
    ``export_field_mode`` at suite creation time. The guard exists to
    catch lineage drift: manually-seeded evaluate jobs, ad-hoc exports
    that mismatch the training mode, and regressions in suite creation.

    Returns ``None`` if the check passes (or is not applicable), or a
    human-readable error string naming the mismatch when it fails.
    """
    # Locate the paired training export via the chain's train job.
    train_job = find_train_job_for_chain(
        session,
        project_id=project_id,
        chain_id=evaluate_job.chain_id,
    )
    if train_job is None:
        # Ad-hoc evaluate job not tied to a chain — nothing to compare.
        return None

    train_ids = list(train_job.dataset_export_ids or [])
    eval_ids = list(evaluate_job.dataset_export_ids or [])
    if not train_ids or not eval_ids:
        return None

    train_exports = (
        session.query(DatasetExport)
        .filter(
            DatasetExport.project_id == project_id,
            DatasetExport.dataset_export_id.in_(train_ids),
        )
        .all()
    )
    eval_exports = (
        session.query(DatasetExport)
        .filter(
            DatasetExport.project_id == project_id,
            DatasetExport.dataset_export_id.in_(eval_ids),
        )
        .all()
    )
    if not train_exports or not eval_exports:
        return None

    train_modes = {e.export_field_mode for e in train_exports}
    eval_modes = {e.export_field_mode for e in eval_exports}
    if train_modes != eval_modes:
        return (
            f"export_field_mode_mismatch: training exports="
            f"{sorted(train_modes)} vs evaluation exports="
            f"{sorted(eval_modes)}"
        )
    return None


async def _submit_and_persist_outcome(
    engine: Engine,
    *,
    project_id: str,
    tao_job_id: str,
    tao_create_job_request: dict[str, Any],
    settings: Settings,
    log_label: str,
) -> dict[str, Any] | str:
    """Steps 2–3 of the 4-step submission protocol.

    POSTs the payload to TAO outside any transaction, then persists the
    outcome — ``submitted`` + ``tao_external_job_id``, or ``failed`` +
    sanitized ``error_ref`` — in a short write transaction. The single
    implementation shared by :func:`create_tao_job` and
    :func:`submit_chain_job` so the two submission entry points cannot
    drift. ``log_label`` names the caller in log lines ("TAO job" vs
    "TAO chain job").

    Returns the updated TAOJob record dict, or a "not found" error
    string when the row vanished mid-protocol (should never happen).
    """
    submission: dict[str, Any]
    try:
        submission = await _submit_to_tao(tao_create_job_request, settings=settings)
    except Exception as exc:  # defensive: network stack blew up
        logger.exception("TAO submission crashed for job %s", tao_job_id)
        submission = {
            "success": False,
            "tao_external_job_id": None,
            "error": f"TAO submission raised: {type(exc).__name__}",
        }

    # A suite-level cancel may land while the TAO create-job request is in
    # flight. Never resurrect that locally canceled row as ``submitted``.
    # If TAO accepted the request, immediately make the best-effort remote
    # cancel now that its external id is finally known.
    with Session(engine) as session:
        current = session.query(TAOJob).filter_by(tao_job_id=tao_job_id).first()
        if current is None:
            return f"not found: TAOJob {tao_job_id}"
        canceled_while_submitting = current.status == "canceled"

    if canceled_while_submitting:
        remote_cancel: dict[str, Any] | None = None
        external_id = submission.get("tao_external_job_id")
        if submission.get("success") and isinstance(external_id, str):
            remote_cancel = await request_tao_job_cancel(external_id, settings=settings)

        with Session(engine) as session:
            job = session.query(TAOJob).filter_by(tao_job_id=tao_job_id).first()
            if job is None:
                return f"not found: TAOJob {tao_job_id}"
            if isinstance(external_id, str):
                job.tao_external_job_id = external_id
            if remote_cancel is not None and not remote_cancel["success"]:
                detail = sanitize_error(remote_cancel.get("error"))
                job.poll_error_ref = f"suite_cancel_unconfirmed: {detail}"
            session.commit()
            return _job_to_dict(job)

    # ── Short write txn #2: persist submission outcome ──────────────────
    now = utc_now()
    with Session(engine) as session:
        job = session.query(TAOJob).filter_by(tao_job_id=tao_job_id).first()
        if job is None:  # should never happen
            return f"not found: TAOJob {tao_job_id}"

        if submission["success"]:
            job.tao_external_job_id = submission["tao_external_job_id"]
            job.status = "submitted"
            job.started_at = now
            logger.info(
                "%s %s submitted (external_id=%s)",
                log_label,
                tao_job_id,
                submission["tao_external_job_id"],
                extra={"component": "tao_job_service", "project_id": project_id},
            )
        else:
            job.status = "failed"
            job.error_ref = sanitize_error(submission["error"])
            job.completed_at = now
            logger.warning(
                "%s %s submission failed: %s",
                log_label,
                tao_job_id,
                job.error_ref,
                extra={"component": "tao_job_service", "project_id": project_id},
            )
        session.commit()
        return _job_to_dict(job)


async def submit_chain_job(
    project_id: str,
    tao_job_id: str,
    *,
    settings: Settings,
    advance_on_failure: bool = True,
) -> str:
    """Transition a pre-created ``not_started`` TAOJob through submission.

    Used by the training-suite service to kick off the first chain's
    first job after suite creation, and by the TAO polling loop to
    advance chains when a predecessor reaches ``succeeded``.

    Reuses the 4-step submission protocol — same
    ``submitting → POST → submitted / failed`` transitions, same short
    write transactions, same sanitized error persistence, same SSE event
    emission — without duplicating logic.

    For ``action="evaluate"`` jobs, the paired training
    DatasetExport's ``export_field_mode`` must match the evaluation
    export's mode (the export-field-mode lineage guard). Mismatches
    short-circuit the submission to ``status="failed"`` with a clear
    ``status_reason``.

    ``advance_on_failure`` (default True): when the submission fails
    mid-chain, run the same post-failure flow the poller uses
    (``tao_polling_service.handle_terminal_failure`` — halt dependents,
    advance the next eligible/cross-chain job, roll up suite status), so a
    submission-time failure and a poll-detected failure leave identical
    chain/suite state. The suite-kickoff caller passes ``False`` because it
    fails the whole suite itself and must not cross-advance.

    Returns one of ``"submitted"``, ``"failed"``, or an error string for
    the router layer (e.g. ``"not found: ..."``, ``"conflict: ..."``).
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    async def _advance_after_submission_failure(
        chain_id: str | None, chain_sequence: int | None, action: str
    ) -> None:
        """Run the poller's post-failure flow for a submission-time failure.

        Halts dependents, advances the next eligible / cross-chain job, and
        rolls up suite status so the TrainingSuite doesn't strand in
        ``running``. Lazy import avoids a module-load cycle
        (tao_polling_service imports this module). ``emit_root_failed=False``
        because this failure path already emitted ``run_failed``.
        """
        if not advance_on_failure or chain_id is None:
            return
        from vlm_feedback_loop.services import tao_polling_service

        await tao_polling_service.handle_terminal_failure(
            project_id,
            tao_job_id,
            chain_id=chain_id,
            chain_sequence=chain_sequence,
            action=action,
            terminal_status="failed",
            engine=engine,
            settings=settings,
            emit_root_failed=False,
        )

    # ── Short write txn #1: validate transition + move to submitting ──
    tao_create_job_request_dict: dict[str, Any]
    field_mode_mismatch: str | None = None
    with Session(engine) as session:
        job = (
            session.query(TAOJob)
            .filter_by(tao_job_id=tao_job_id, project_id=project_id)
            .first()
        )
        if job is None:
            return f"not found: TAOJob {tao_job_id}"

        # Only chain-pre-created jobs may be submitted via this path.
        if job.status != "not_started":
            return (
                f"conflict: terminal or in-flight status {job.status!r} cannot "
                f"be submitted via chain advancement (expected not_started)"
            )
        if not can_transition("not_started", "submitting"):
            # Defensive: should never happen given the state machine.
            return "conflict: invalid transition not_started → submitting"

        # Export-field-mode lineage guard — evaluate jobs only.
        if job.action == "evaluate":
            field_mode_mismatch = _check_evaluate_field_mode_lineage(
                session,
                project_id=project_id,
                evaluate_job=job,
            )
            if field_mode_mismatch is not None:
                # Short-circuit: fail cleanly without POSTing to TAO.
                job.status = "failed"
                job.error_ref = sanitize_error(field_mode_mismatch)
                job.completed_at = utc_now()
                mismatch_chain_id = job.chain_id
                mismatch_chain_sequence = job.chain_sequence
                mismatch_action = job.action
                session.commit()
                # Emit run_failed SSE AFTER the durable write commits.
                try:
                    await sse_manager.emit(
                        project_id,
                        "run_failed",
                        {
                            "run_id": tao_job_id,
                            "tao_job_id": tao_job_id,
                            "run_type": "tao_job",
                            "error_summary": field_mode_mismatch,
                        },
                    )
                except Exception:  # pragma: no cover — SSE best-effort
                    logger.exception("SSE emit failed for tao job %s", tao_job_id)
                logger.warning(
                    "TAO evaluate %s rejected pre-submission: %s",
                    tao_job_id,
                    field_mode_mismatch,
                    extra={
                        "component": "tao_job_service",
                        "project_id": project_id,
                    },
                )
                await _advance_after_submission_failure(
                    mismatch_chain_id, mismatch_chain_sequence, mismatch_action
                )
                return "failed"

        tao_create_job_request_dict = dict(job.tao_create_job_request or {})
        # Chain advancement: inject top-level ``parent_job_id`` so FTMS
        # can resolve the trained checkpoint via its
        # ``infer_parent_model_folder`` helper. Without this, an
        # evaluate (or quantize) chain job submits with ``parent_id=None``
        # and the cosmos-rl worker crashes in
        # ``cosmos_rl/evaluation/base.py`` because ``model.model_name``
        # was never populated.
        if job.parent_tao_job_id:
            parent = (
                session.query(TAOJob)
                .filter_by(tao_job_id=job.parent_tao_job_id)
                .first()
            )
            if parent is not None and parent.tao_external_job_id:
                tao_create_job_request_dict["parent_job_id"] = (
                    parent.tao_external_job_id
                )
        job.status = "submitting"
        session.commit()

    # ── Steps 2–3: submit outside the txn, persist the outcome ─────────
    outcome = await _submit_and_persist_outcome(
        engine,
        project_id=project_id,
        tao_job_id=tao_job_id,
        tao_create_job_request=tao_create_job_request_dict,
        settings=settings,
        log_label="TAO chain job",
    )
    if isinstance(outcome, str):
        return outcome
    terminal_status: str = outcome["status"]
    # Chain metadata for post-failure advancement.
    failed_chain_id = outcome["chain_id"]
    failed_chain_sequence = outcome["chain_sequence"]
    failed_action = outcome["action"]

    # Emit SSE after the durable write commits (state first, event second).
    try:
        await sse_manager.emit(
            project_id,
            "tao_job_progress",
            {
                "run_id": tao_job_id,
                "tao_job_id": tao_job_id,
                "run_type": "tao_job",
                "status": terminal_status,
            },
        )
        if terminal_status == "failed":
            await sse_manager.emit(
                project_id,
                "run_failed",
                {
                    "run_id": tao_job_id,
                    "tao_job_id": tao_job_id,
                    "run_type": "tao_job",
                    "error_summary": outcome["error_ref"] or "submission failed",
                },
            )
    except Exception:  # pragma: no cover — SSE is best-effort
        logger.exception("SSE emit failed for tao job %s", tao_job_id)

    # A mid-chain submission failure must halt dependents, advance the chain,
    # and roll up suite status — identical to a poll-detected failure — or the
    # TrainingSuite strands in "running" forever (the poller's advance call
    # sites ignore this return value).
    if terminal_status == "failed":
        await _advance_after_submission_failure(
            failed_chain_id, failed_chain_sequence, failed_action
        )

    return terminal_status


async def create_tao_job(
    project_id: str,
    *,
    body: TAOJobCreateRequest,
    settings: Settings,
) -> dict[str, Any] | str:
    """Create and submit a TAO job following the 4-step protocol.

    Returns the full TAOJob record dict on success, or an error string
    mapped to HTTP responses by the router.  Submission failures do not
    raise — they are persisted as ``status="failed"`` with a sanitized
    ``error_ref`` so the caller can inspect the record.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    action = body.tao_create_job_request.action
    dataset_export_ids = list(body.dataset_export_ids)
    job_config_dict = body.job_config.model_dump(mode="json")
    tao_create_job_request_dict = body.tao_create_job_request.model_dump(mode="json")

    # Compute and embed the checksum so it is persisted with the payload.
    checksum = compute_request_checksum(tao_create_job_request_dict)
    job_config_dict["tao_create_job_request_checksum"] = checksum

    tao_job_id = generate_uuid4()
    created_at = utc_now()

    # ── Short write txn #1: validate, persist with status="submitting" ──
    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return f"not found: Project {project_id}"

        err = _validate_dataset_exports(session, project_id, dataset_export_ids, action)
        if err is not None:
            return err

        err = _validate_student_base(
            session, project_id, body.student_base_model_config_id
        )
        if err is not None:
            return err

        job = TAOJob(
            tao_job_id=tao_job_id,
            project_id=project_id,
            student_base_model_config_id=body.student_base_model_config_id,
            dataset_export_ids=dataset_export_ids,
            action=action,
            status="submitting",
            training_backend=body.job_config.training_backend,
            training_policy_type=(
                body.job_config.training_policy_type if action == "train" else None
            ),
            job_config=job_config_dict,
            tao_create_job_request=tao_create_job_request_dict,
            tao_external_job_id=None,
            created_at=created_at,
        )
        session.add(job)
        session.commit()

    # ── Steps 2–3: submit outside the write txn, persist outcome ──────────
    return await _submit_and_persist_outcome(
        engine,
        project_id=project_id,
        tao_job_id=tao_job_id,
        tao_create_job_request=tao_create_job_request_dict,
        settings=settings,
        log_label="TAO job",
    )


async def get_tao_job(
    project_id: str,
    tao_job_id: str,
    *,
    refresh: bool = False,
    settings: Settings,
) -> dict[str, Any] | str:
    """Return the TAOJob record, optionally refreshing from TAO first.

    Async because ``refresh=True`` performs a one-shot poll against the
    remote FTMS endpoint via ``poll_tao_job``.  Without ``refresh`` the
    function returns the current DB state immediately (no I/O).
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    # ── Read current state and decide whether to poll ───────────────────
    should_poll = False
    external_id: str | None = None
    with Session(engine) as session:
        job = (
            session.query(TAOJob)
            .filter_by(tao_job_id=tao_job_id, project_id=project_id)
            .first()
        )
        if job is None:
            return f"not found: TAOJob {tao_job_id}"

        # Refresh guard-rails: skip if terminal, no external id, or rate-limited.
        if (
            refresh
            and job.status not in TERMINAL_STATUSES
            and job.tao_external_job_id
            and not _recent_poll(job.last_polled_at, settings.TAO_POLL_MIN_INTERVAL_S)
        ):
            should_poll = True
            external_id = job.tao_external_job_id

        if not should_poll:
            return _job_to_dict(job)

    # ── Poll outside any transaction ───────────────────────────────────
    assert external_id is not None  # narrowed by should_poll
    poll_result = await poll_tao_job(external_id, settings=settings)

    # ── Short write txn #2: persist refreshed state ─────────────────────
    now = utc_now()
    with Session(engine) as session:
        job = (
            session.query(TAOJob)
            .filter_by(tao_job_id=tao_job_id, project_id=project_id)
            .first()
        )
        if job is None:
            return f"not found: TAOJob {tao_job_id}"

        apply_poll_result(
            job, poll_result, now, failed_error_ref_requires_transition=False
        )
        session.commit()
        return _job_to_dict(job)


async def cancel_tao_job(
    project_id: str,
    tao_job_id: str,
    *,
    settings: Settings,
    force_local: bool = False,
) -> dict[str, Any] | str:
    """Cancel an in-flight TAO job and halt downstream chain siblings.

    Wired to the Paused-state ``[Cancel Job]`` button on the Training
    Job Monitor screen. Sequence:

    1. Short write txn: validate the job exists and is in a non-terminal,
       cancelable status. ``submitting`` is refused (brief transient).
    2. If the job has a ``tao_external_job_id`` AND ``force_local=False``,
       POST ``:cancel`` to TAO outside any DB transaction.  On TAO failure
       we persist ``poll_error_ref`` and return an error — the local state
       stays where it was so the SME can retry.
    3. Short write txn: transition to ``canceled``, persist
       ``completed_at``, and halt every downstream ``not_started`` sibling
       in the same chain with ``chain_halted_reason`` populated.
    4. Emit SSE ``run_failed`` for the canceled job plus every halted
       sibling (matches ``tao_polling_service.handle_terminal_failure``).

    ``force_local=True`` skips
    Phase 2 entirely so the local row transitions to ``canceled`` even
    when TAO is unreachable or the external job ID has been orphaned by
    a TAO server rebuild. The canceled row's ``poll_error_ref`` is
    stamped with ``"forced_local_cancel: external TAO unreachable or
    external_id orphaned"`` so the audit trail records why this was used.
    A clear ``warning`` log line is emitted on every force-local cancel
    so operators can detect drift between local and external state.

    Returns the updated TAOJob dict on success, or a service-error string
    that the router maps to an HTTP status.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    # ── Phase 1: read & validate current state ──────────────────────────
    external_id: str | None = None
    chain_id: str | None = None
    chain_sequence: int | None = None
    action: str | None = None
    with Session(engine) as session:
        job = (
            session.query(TAOJob)
            .filter_by(tao_job_id=tao_job_id, project_id=project_id)
            .first()
        )
        if job is None:
            return f"not found: TAOJob {tao_job_id}"
        if job.status in TERMINAL_STATUSES:
            return f"conflict: cannot cancel TAOJob in terminal status {job.status!r}"
        if not can_transition(job.status, "canceled"):
            # ``submitting`` cannot cancel — it has no TAO-side id yet.
            return f"conflict: cannot cancel TAOJob in status {job.status!r}"
        external_id = job.tao_external_job_id
        chain_id = job.chain_id
        chain_sequence = job.chain_sequence
        action = job.action

    # ── Phase 2: POST :cancel to TAO (outside any txn) ──────────────────
    # force_local=True skips this phase entirely — the operator has
    # signaled that the external TAO is unreachable or the external_id has
    # been orphaned (typically by a TAO-server rebuild) and the local row
    # MUST transition to canceled regardless. Phase 3 then stamps a
    # poll_error_ref so the audit trail records WHY the TAO call was
    # bypassed.
    if external_id is not None and not force_local:
        tao_result = await _cancel_tao_external(external_id, settings=settings)
        if not tao_result["success"]:
            # Persist the sanitized error; do NOT transition.  The SME can
            # retry via the same endpoint once TAO is reachable again, or
            # pass force_local=true if the external_id is known orphaned.
            with Session(engine) as session:
                job = (
                    session.query(TAOJob)
                    .filter_by(tao_job_id=tao_job_id, project_id=project_id)
                    .first()
                )
                if job is not None:
                    job.poll_error_ref = sanitize_error(tao_result["error"])
                    session.commit()
            # Transport failures map to the documented 503/504 contract;
            # provider-side refusals stay 502 (see errors.map_service_error).
            error_class = tao_result.get("error_class")
            if error_class == "timeout":
                return f"tao_timeout: {tao_result['error']}"
            if error_class == "endpoint_error":
                return f"tao_unreachable: {tao_result['error']}"
            return f"tao_error: {tao_result['error']}"
    elif external_id is not None and force_local:
        logger.warning(
            "TAOJob %s force-local cancel: skipping TAO POST (external_id=%s); "
            "local state will transition without remote acknowledgement",
            tao_job_id,
            external_id,
        )

    # ── Phase 3: local transition + halt downstream siblings ────────────
    now = utc_now()
    halted: list[dict[str, Any]] = []
    response_dict: dict[str, Any] | None = None
    with Session(engine) as session:
        job = (
            session.query(TAOJob)
            .filter_by(tao_job_id=tao_job_id, project_id=project_id)
            .first()
        )
        if job is None:
            return f"not found: TAOJob {tao_job_id}"
        # Someone else may have raced us; tolerate terminal landings.
        if job.status in TERMINAL_STATUSES:
            return _job_to_dict(job)

        job.status = "canceled"
        if not job.completed_at:
            job.completed_at = now
        # When force_local was used, stamp poll_error_ref so the audit
        # trail records that this cancel bypassed TAO. Otherwise clear any
        # stale cancel-attempt error from a prior failed retry.
        if force_local and external_id is not None:
            job.poll_error_ref = (
                "forced_local_cancel: external TAO unreachable or external_id orphaned"
            )
        else:
            job.poll_error_ref = None

        if chain_id is not None and chain_sequence is not None:
            not_started_rows = (
                session.query(TAOJob)
                .filter(
                    TAOJob.project_id == project_id,
                    TAOJob.chain_id == chain_id,
                    TAOJob.chain_sequence > chain_sequence,
                    TAOJob.status == "not_started",
                )
                .order_by(TAOJob.chain_sequence.asc())
                .all()
            )
            reason = (
                f"Chain halted: {action} (seq {chain_sequence}, "
                f"id={tao_job_id}) canceled by SME"
            )
            for row in not_started_rows:
                # not_started → failed with chain_halted_reason.
                if can_transition(row.status, "failed"):
                    row.status = "failed"
                    row.chain_halted_reason = reason
                    row.completed_at = now
                    halted.append(
                        {
                            "tao_job_id": row.tao_job_id,
                            "reason": row.chain_halted_reason,
                        }
                    )

        session.commit()

    # Suite bookkeeping — the same planning helper the poller's
    # ``handle_terminal_failure`` runs. Without this, a user-initiated
    # cancel strands the owning TrainingSuite as ``running`` forever
    # (no pollable rows remain, so the poller never revisits) and
    # never starts the suite's next chain — both observed live on
    # two suites, 2026-07-14. Lazy import: the polling service
    # imports this module at top level.
    next_in_chain_job_id: str | None = None
    cross_chain_next_id: str | None = None
    if chain_id is not None:
        from vlm_feedback_loop.services.tao_polling_service import _plan_suite_advance

        with Session(engine) as session:
            next_in_chain_job_id, cross_chain_next_id = _plan_suite_advance(
                session, project_id=project_id, chain_id=chain_id
            )
            session.commit()

    # Chain advance outside any transaction (network call).
    advance_job_id = next_in_chain_job_id or cross_chain_next_id
    if advance_job_id is not None:
        try:
            await submit_chain_job(project_id, advance_job_id, settings=settings)
        except Exception:
            logger.exception(
                "post-cancel chain advance failed for job %s (project %s)",
                advance_job_id,
                project_id,
            )

    with Session(engine) as session:
        refreshed = (
            session.query(TAOJob)
            .filter_by(tao_job_id=tao_job_id, project_id=project_id)
            .first()
        )
        assert refreshed is not None  # existed in the halt txn above
        response_dict = _job_to_dict(refreshed)

    # ── Phase 4: SSE notifications (best-effort) ────────────────────────
    try:
        await sse_manager.emit(
            project_id,
            "run_failed",
            {
                "run_id": tao_job_id,
                "tao_job_id": tao_job_id,
                "run_type": "tao_job",
                "error_summary": "tao_job_canceled",
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
    except Exception:  # pragma: no cover — SSE must not block cancel
        logger.exception("SSE emit failed on cancel")

    return response_dict


def _recent_poll(last_polled_at: str | None, min_interval_s: int) -> bool:
    """Return True if the last poll is within the rate-limit window."""
    if last_polled_at is None:
        return False
    # Timestamps are ISO 8601 with Z suffix.  Parse via a minimal approach.
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(last_polled_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    delta = datetime.now(UTC) - dt
    return delta.total_seconds() < float(min_interval_s)


def list_tao_jobs(
    project_id: str,
    *,
    status_filter: str | None = None,
    cursor: str | None = None,
    limit: int = 20,
    settings: Settings,
) -> tuple[list[dict[str, Any]], str | None] | str:
    """List TAOJob records with cursor pagination, newest-first.

    Cursor encoding: opaque base64 ``(created_at, tao_job_id)`` tuple
    (``services.pagination``).
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        stmt = (
            select(TAOJob)
            .where(TAOJob.project_id == project_id)
            .order_by(TAOJob.created_at.desc(), TAOJob.tao_job_id.asc())
        )
        if status_filter:
            stmt = stmt.where(TAOJob.status == status_filter)

        if cursor:
            try:
                cur_created_at, cur_id = decode_cursor(cursor)
            except InvalidCursorError:
                return "validation: invalid cursor"
            stmt = stmt.where(
                after_position_asc(
                    TAOJob.created_at, TAOJob.tao_job_id, cur_created_at, cur_id
                )
            )

        # Fetch one extra row to determine whether there's a next page.
        stmt = stmt.limit(limit + 1)
        rows = list(session.execute(stmt).scalars().all())

        next_cursor: str | None = None
        if len(rows) > limit:
            rows = rows[:limit]
            tail = rows[-1]
            next_cursor = encode_cursor(tail.created_at, tail.tao_job_id)

        return [_job_to_dict(r) for r in rows], next_cursor


# ── Restart recovery ─────────────────────────────────────────────────────────


def recover_interrupted_tao_jobs(settings: Settings) -> list[dict[str, Any]]:
    """Scan every project DB for TAOJob rows stuck in ``submitting`` + null
    external id and transition them to ``failed`` with a sanitized
    ``error_ref="submission_interrupted"``.

    Called once during application startup.  Rows with ``status="submitting"``
    that already have a ``tao_external_job_id`` are treated as "submission
    confirmed pre-crash" and left alone (the status will be reconciled by
    polling later).

    Returns a descriptor per recovered chain job (``project_id``,
    ``tao_job_id``, ``chain_id``, ``chain_sequence``, ``action``) so the async
    lifespan can run the same halt/advance/roll-up flow the poller uses — a
    submitting job that crashed mid-chain must not leave its TrainingSuite
    stranded in ``running`` (this function is sync and cannot await the flow
    itself).
    """
    projects_dir = projects_root(settings.WORKSPACE_ROOT)
    recovered: list[dict[str, Any]] = []
    if not projects_dir.exists():
        return recovered

    now = utc_now()
    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir():
            continue
        db_path = entry / "project.db"
        if not db_path.exists():
            continue
        # Archived projects are dormant: recovery must not migrate or
        # lock them (mirrors _recover_interrupted_runs).
        if (entry / ".archived").exists():
            continue

        try:
            # get_project_engine acquires the per-project lock and caches
            # the engine — one engine per project process-wide, not a
            # fresh leaked engine per startup scan.
            maybe_engine = get_project_engine(entry.name, settings.WORKSPACE_ROOT)
            if maybe_engine is None:
                continue
            engine: Engine = maybe_engine
        except Exception as exc:  # pragma: no cover — defensive
            # Observability: ``str(exc) or "(no message)"`` so empty-string
            # exceptions still produce actionable signal — same idiom as
            # ``services/project_service.py``,
            # ``services/local_nim_service.py``, and
            # ``services/background.py::_on_task_done``.
            logger.warning(
                "Skipping TAOJob recovery for %s (%s: %s)",
                entry.name,
                type(exc).__name__,
                str(exc) or "(no message)",
            )
            continue

        with Session(engine) as session:
            orphans = (
                session.query(TAOJob)
                .filter(
                    TAOJob.status == "submitting",
                    TAOJob.tao_external_job_id.is_(None),
                )
                .all()
            )
            if not orphans:
                continue

            for job in orphans:
                job.status = "failed"
                job.error_ref = "submission_interrupted"
                job.completed_at = now
                recovered.append(
                    {
                        "project_id": job.project_id,
                        "tao_job_id": job.tao_job_id,
                        "chain_id": job.chain_id,
                        "chain_sequence": job.chain_sequence,
                        "action": job.action,
                    }
                )
                logger.info(
                    "Recovery: TAO job %s → failed (submission_interrupted)",
                    job.tao_job_id,
                    extra={
                        "component": "tao_job_service",
                        "project_id": job.project_id,
                    },
                )
            session.commit()

    return recovered


# ── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    "ALLOWED_TRANSITIONS",
    "CANONICAL_STATUSES",
    "TERMINAL_STATUSES",
    "apply_dataset_binding",
    "can_transition",
    "compute_request_checksum",
    "create_tao_job",
    "find_suite_for_chain",
    "find_train_job_for_chain",
    "get_tao_job",
    "list_tao_jobs",
    "map_tao_raw_status",
    "recover_interrupted_tao_jobs",
    "submit_chain_job",
]

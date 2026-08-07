# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelConfig CRUD service with capability probes.

Capability probes (structured generation, thinking toggle, visual budget)
run against real NIM endpoints and persist results on the ModelConfig record.
Probes run once and are reused; the re-probe endpoint refreshes them.

Log point 6: capability probes (operational logging).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
import time
import zlib
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.run import ACTIVE_RUN_STATUSES, RunRecord
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services import nim_client
from vlm_feedback_loop.services.logging_config import get_logger
from vlm_feedback_loop.services.nim_endpoint_service import resolve_probe_credential
from vlm_feedback_loop.services.pagination import (
    after_position,
    decode_cursor,
    encode_cursor,
)
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.prompt_service import (
    load_probe_prompt,
    resolve_thinking_fields,
)

logger = logging.getLogger("vlm_feedback_loop.services.model_config_service")

# Models allowed to have the student_base role.
# Prefix matches the hosted-catalog namespaced identifier (``nvidia/cosmos-reason2-8b``,
# ``nvidia/cosmos-reason2-2b``). cosmos-rl specs are pinned to the namespaced
# form, so legacy unqualified names are not accepted.
#
# The Cosmos 3 (CR3) family ships as unified checkpoints
# (``nvidia/Cosmos3-Nano`` / ``-Super``, catalog-namespaced ``nvidia/cosmos3-*``)
# rather than a dedicated reasoner repo per size; the reasoner architecture is
# applied as a load-time override on the same checkpoint. The ``nvidia/cosmos3-``
# prefix admits that family as a student_base. NOTE: this allowlist only governs
# the app-side role guard — whether cosmos-rl actually trains CR3 (and in
# reasoner mode) is gated TAO-side by base-experiment provisioning + cosmos-rl
# family support.
_STUDENT_BASE_ALLOWED_PREFIXES = ("nvidia/cosmos-reason2-", "nvidia/cosmos3-")

# Active TAO job statuses that block re-probe
_ACTIVE_TAO_STATUSES = ("submitting", "submitted", "queued", "running", "paused")


# ── Probe image (stdlib-only PNG) ───────────────────────────────────────────

# Cached after first generation — deterministic, never changes.
_probe_image_data_url: str | None = None


def _make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a PNG chunk: length + type + data + CRC."""
    raw = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + raw
        + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)
    )


def generate_probe_image_data_url() -> str:
    """Generate a deterministic 512×512 solid-gray RGB PNG as a base64 data URL.

    Uses only stdlib (struct, zlib, base64). No Pillow required.
    The image is ~2-3 KB after zlib compression of uniform pixel data.
    """
    global _probe_image_data_url
    if _probe_image_data_url is not None:
        return _probe_image_data_url

    width, height = 512, 512
    bit_depth, color_type = 8, 2  # 8-bit RGB

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    ihdr = _make_png_chunk(b"IHDR", ihdr_data)

    # IDAT — raw pixel rows: each row = filter_byte(0) + width * 3 bytes (RGB)
    pixel = b"\x80\x80\x80"  # mid-gray
    row = b"\x00" + pixel * width  # filter byte 0 + pixels
    raw_data = row * height
    compressed = zlib.compress(raw_data, 9)
    idat = _make_png_chunk(b"IDAT", compressed)

    # IEND
    iend = _make_png_chunk(b"IEND", b"")

    png_bytes = signature + ihdr + idat + iend
    b64 = base64.b64encode(png_bytes).decode("ascii")
    _probe_image_data_url = f"data:image/png;base64,{b64}"
    return _probe_image_data_url


# ── Active-use check ────────────────────────────────────────────────────────


def is_model_in_active_use(
    session: Session, project_id: str, model_config_id: str
) -> bool:
    """Check if the model is referenced by any active run or TAO job."""
    active_run = session.execute(
        select(RunRecord.run_id)
        .where(
            RunRecord.project_id == project_id,
            RunRecord.model_config_id == model_config_id,
            RunRecord.status.in_(ACTIVE_RUN_STATUSES),
        )
        .limit(1)
    ).scalar_one_or_none()

    if active_run is not None:
        return True

    active_tao = session.execute(
        select(TAOJob.tao_job_id)
        .where(
            TAOJob.project_id == project_id,
            TAOJob.student_base_model_config_id == model_config_id,
            TAOJob.status.in_(_ACTIVE_TAO_STATUSES),
        )
        .limit(1)
    ).scalar_one_or_none()

    return active_tao is not None


# ── Capability probes ───────────────────────────────────────────────────────

_STRUCTURED_PROBE_MAX_TOKENS = 16
_STRUCTURED_PROBE_REASONING_MAX_TOKENS = 4096


async def probe_structured_generation(
    base_url: str,
    auth_headers: dict[str, str],
    model_name: str,
    deadline_s: float,
    thinking_toggle_mode: str = "none",
) -> str:
    """Probe structured generation support.

    Returns "supported", "unsupported", or "unknown".
    """
    fields: dict[str, Any] | None = resolve_thinking_fields(
        False, thinking_toggle_mode, "supported"
    )["thinking_request_fields"]
    thinking_off_kwargs: dict[str, Any] = (
        {} if fields is None else {"chat_template_kwargs": fields}
    )
    max_tokens = (
        _STRUCTURED_PROBE_REASONING_MAX_TOKENS
        if thinking_toggle_mode == "always_on_reasoning"
        else _STRUCTURED_PROBE_MAX_TOKENS
    )

    result = await nim_client.chat_completions(
        base_url=base_url,
        auth_headers=auth_headers,
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": load_probe_prompt("probe_structured_generation.txt"),
            }
        ],
        deadline_s=deadline_s,
        max_retries=1,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        },
        max_tokens=max_tokens,
        **thinking_off_kwargs,
    )

    if result.success and result.content is not None:
        try:
            parsed = json.loads(result.content)
            if (
                isinstance(parsed, dict)
                and "ok" in parsed
                and isinstance(parsed["ok"], bool)
            ):
                return "supported"
        except (json.JSONDecodeError, TypeError):
            pass
        # A truncated reasoning trace proves neither acceptance nor rejection
        # of response_format. Preserve unknown so a long always-on trace cannot
        # falsely demote a supported capability.
        if result.finish_reason == "length":
            return "unknown"
        # Response received but didn't validate
        return "unsupported"

    if result.success and result.finish_reason == "length":
        return "unknown"

    if result.status_code is not None and 400 <= result.status_code < 500:
        return "unsupported"

    # timeout or 5xx
    return "unknown"


async def probe_thinking_toggle(
    base_url: str,
    auth_headers: dict[str, str],
    model_name: str,
    thinking_toggle_mode: str,
    deadline_s: float,
) -> str:
    """Probe thinking toggle acceptance.

    Returns "supported", "unsupported", or "unknown".
    """
    # No runtime toggle exists for these modes, so toggle support is
    # definitionally "unsupported" — there is nothing to probe. The
    # distinction between "none" and "always_on_reasoning" is handled
    # in resolve_thinking_fields and the budget service.
    #
    # Evidence (hosted ``integrate.api.nvidia.com``, Mistral Large 3,
    # mode="none"): both Qwen-style (``enable_thinking=False``) and
    # Kimi-style (``thinking=False``) ``chat_template_kwargs`` are
    # rejected with HTTP 400. Mistral isn't merely toggle-less —
    # sending the kwargs at all breaks every inference. So "skip the
    # probe AND hide the UI toggle" is not over-cautious; it's
    # load-bearing for non-reasoning Teachers.
    if thinking_toggle_mode in ("none", "always_on_reasoning"):
        return "unsupported"

    fields: dict[str, Any] | None = resolve_thinking_fields(
        False, thinking_toggle_mode, "supported"
    )["thinking_request_fields"]
    if fields is None:
        return "unknown"
    kwargs: dict[str, Any] = {"chat_template_kwargs": fields}

    result = await nim_client.chat_completions(
        base_url=base_url,
        auth_headers=auth_headers,
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": load_probe_prompt("probe_thinking_toggle.txt"),
            }
        ],
        deadline_s=deadline_s,
        max_retries=1,
        max_tokens=4,
        **kwargs,
    )

    if result.success:
        return "supported"

    if result.status_code is not None and 400 <= result.status_code < 500:
        return "unsupported"

    return "unknown"


# Visual budget probe mode-specific kwargs
_VISUAL_BUDGET_PROBE_KWARGS: dict[str, dict[str, Any]] = {
    "mm_processor_size": {
        "mm_processor_kwargs": {"size": {"shortest_edge": 1568, "longest_edge": 262144}}
    },
    "mm_processor_pixels": {
        "mm_processor_kwargs": {
            "images_kwargs": {"min_pixels": 1568, "max_pixels": 262144}
        }
    },
    "mm_processor_tiles": {"mm_processor_kwargs": {"max_num_tiles": 3}},
}


async def _probe_image_cap_support(
    base_url: str,
    auth_headers: dict[str, str],
    model_name: str,
    max_images_per_request: int,
    deadline_s: float,
) -> str:
    """Probe whether the seeded ``max_images_per_request`` value is correct.

    Reality-checks the catalog cap by sending one request at the cap and
    one at cap+1, classifying as:

    * ``"supported"`` — seeded value works (HTTP 200 at N) AND the model
      rejects N+1. The cap is precisely ``max_images_per_request``.
    * ``"supported"`` (with hint) — both N and N+1 succeed. The seeded
      value is conservative (real cap is at least N+1). The hint is
      logged so an operator can run the dedicated
      ``scripts/probe_hosted_image_caps.py`` to find the real cap.
    * ``"unsupported"`` — N rejects (HTTP 4xx). The seeded value is
      too high; runtime ICL pruning will hit HTTP 400 from hosted NIM.
    * ``"unknown"`` — timeout / 5xx / non-image-cap 4xx — can't classify.

    The probe sends only deterministic 512×512 synthetic PNGs (no real
    image content) so it's cheap and idempotent. Bound by
    ``deadline_s``. ``max_images_per_request`` MUST be ≥ 1.
    """
    if max_images_per_request < 1:
        return "unknown"

    probe_image = generate_probe_image_data_url()
    base_messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Respond with: ok"}],
        }
    ]

    def _build(n: int) -> list[dict[str, Any]]:
        content = list(base_messages[0]["content"])
        for _ in range(n):
            content.append({"type": "image_url", "image_url": {"url": probe_image}})
        return [{"role": "user", "content": content}]

    # Stage 1: cap at the seeded value. Must succeed.
    at_cap = await nim_client.chat_completions(
        base_url=base_url,
        auth_headers=auth_headers,
        model=model_name,
        messages=_build(max_images_per_request),
        deadline_s=deadline_s,
        max_retries=1,
        max_tokens=4,
        stream=False,
    )
    if not at_cap.success:
        if at_cap.status_code is not None and 400 <= at_cap.status_code < 500:
            # Seeded too high.
            return "unsupported"
        return "unknown"

    # Stage 2: cap+1. Should fail with image-cap rejection.
    above = await nim_client.chat_completions(
        base_url=base_url,
        auth_headers=auth_headers,
        model=model_name,
        messages=_build(max_images_per_request + 1),
        deadline_s=deadline_s,
        max_retries=1,
        max_tokens=4,
        stream=False,
    )
    if above.success:
        # Both N and N+1 succeeded — seeded value is conservative.
        # Still mark "supported" (the cap holds) but log the hint
        # so an operator knows there's headroom.
        logger.info(
            "image_cap_support: seeded cap=%d for %s is conservative "
            "(N+1 also succeeded). Run scripts/probe_hosted_image_caps.py "
            "to find the real cap.",
            max_images_per_request,
            model_name,
            extra={
                "component": "model_config_service",
                "details": {
                    "model_name": model_name,
                    "seeded_cap": max_images_per_request,
                },
            },
        )
        return "supported"

    if above.status_code is not None and 400 <= above.status_code < 500:
        # Expected and good — confirms the cap is exactly the seeded value.
        return "supported"

    # Timeout / 5xx — can't classify the upper boundary.
    return "unknown"


async def probe_visual_budget(
    base_url: str,
    auth_headers: dict[str, str],
    model_name: str,
    visual_budget_mode: str,
    deadline_s: float,
) -> str:
    """Probe visual budget support — two-stage with 512×512 PNG.

    Returns "supported", "unsupported", or "unknown".
    """
    if visual_budget_mode == "none":
        return "unsupported"

    probe_kwargs = _VISUAL_BUDGET_PROBE_KWARGS.get(visual_budget_mode)
    if probe_kwargs is None:
        return "unknown"

    probe_image = generate_probe_image_data_url()
    probe_text = load_probe_prompt("probe_visual_budget.txt")
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": probe_text},
                {"type": "image_url", "image_url": {"url": probe_image}},
            ],
        }
    ]

    # Stage 1 — baseline: WITHOUT mm_processor_kwargs
    baseline = await nim_client.chat_completions(
        base_url=base_url,
        auth_headers=auth_headers,
        model=model_name,
        messages=messages,
        deadline_s=deadline_s,
        max_retries=1,
        max_tokens=4,
        stream=False,
    )

    if not baseline.success:
        return "unknown"

    # Stage 2 — capability: WITH mode-specific mm_processor_kwargs
    capability = await nim_client.chat_completions(
        base_url=base_url,
        auth_headers=auth_headers,
        model=model_name,
        messages=messages,
        deadline_s=deadline_s,
        max_retries=1,
        max_tokens=4,
        stream=False,
        **probe_kwargs,
    )

    if capability.success:
        return "supported"

    return "unsupported"


# ── CRUD ────────────────────────────────────────────────────────────────────


def create_model_config(
    project_id: str,
    data: dict[str, Any],
    workspace_root: str,
) -> ModelConfig | str | None:
    """Create a ModelConfig record.

    Returns the created ModelConfig, an error string, or None (project missing).
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    eligible_roles = data.get("eligible_roles", [])
    if not eligible_roles:
        return "eligible_roles must be non-empty"

    if "student_base" in eligible_roles:
        model_name = data.get("model_name", "")
        if not any(model_name.startswith(p) for p in _STUDENT_BASE_ALLOWED_PREFIXES):
            return (
                "student_base role is only allowed for Cosmos Reason2 "
                "or Cosmos 3 models"
            )

    with Session(engine) as session:
        # Validate endpoint belongs to same project
        endpoint = session.execute(
            select(NimEndpoint).where(
                NimEndpoint.project_id == project_id,
                NimEndpoint.endpoint_id == data["endpoint_id"],
            )
        ).scalar_one_or_none()
        if endpoint is None:
            return "endpoint_id does not belong to this project"

        mc = ModelConfig(
            model_config_id=generate_uuid4(),
            project_id=project_id,
            endpoint_id=data["endpoint_id"],
            model_name=data["model_name"],
            context_window_tokens=data["context_window_tokens"],
            eligible_roles=eligible_roles,
            supports_image_input=data["supports_image_input"],
            thinking_toggle_mode=data.get("thinking_toggle_mode", "none"),
            visual_budget_mode=data.get("visual_budget_mode", "none"),
            max_images_per_request=data.get("max_images_per_request", 5),
            default_icl_max_examples=data.get("default_icl_max_examples"),
            model_quantization=data.get("model_quantization"),
            nim_model_profile=data.get("nim_model_profile"),
            nim_profile_metadata=data.get("nim_profile_metadata"),
            local_deploy_metadata=data.get("local_deploy_metadata"),
        )
        session.add(mc)
        session.commit()
        session.refresh(mc)
        session.expunge(mc)

    return mc


# ── Availability ────────────────────────────────────────────────────────────


# Endpoint last_probe_status values that indicate a hard failure (vs the
# benign "unknown" state new endpoints carry before their first probe).
_HARD_UNHEALTHY_STATUSES = frozenset({"unhealthy", "auth_failed", "unreachable"})


def endpoint_is_operational(endpoint: NimEndpoint) -> bool:
    """Whether persisted endpoint state permits a new model invocation.

    ``is_enabled`` is the lifecycle authority for system-managed endpoints:
    stopping a shared Teacher disables every consumer attachment even if a
    different process later happens to reuse the same TCP port.  The hard
    probe statuses cover explicitly enabled hosted/self-hosted endpoints whose
    most recent connectivity check failed.
    """

    return bool(endpoint.is_enabled) and (
        endpoint.last_probe_status not in _HARD_UNHEALTHY_STATUSES
    )


def compute_availability(
    mc: ModelConfig,
    endpoint: NimEndpoint | None,
    env: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether ``mc`` can actually be invoked from this environment.

    Returns ``{"available": bool, "reason": str | None}`` where ``reason``
    is a stable machine-readable code (see ``UnavailableReason`` in
    ``schemas/model_config.py``). The labeling Teacher dropdown hides
    every entry where ``available`` is False; richer UIs may surface
    ``reason`` as helper copy.

    Rules:

    * No bound endpoint → ``endpoint_missing`` (defensive; shouldn't happen
      in normal seeded flow but handles orphaned references gracefully).
    * Disabled endpoint or endpoint in a hard-unhealthy state →
      ``endpoint_unhealthy`` regardless of mode. The ``"unknown"`` status
      (default for never-probed endpoints) is treated as benign — local NIM
      deployments report unknown before their first health probe, and we don't
      want a probe lag to hide otherwise-working endpoints.
    * Hosted endpoint:
        - No NVIDIA_API_KEY → ``no_nvidia_api_key``
        - Model not hosted-compatible (NVCF-gated like Cosmos) →
          ``hosted_not_compatible``
    * Self-hosted endpoint: trust the operator; available if not hard-unhealthy.
    * Local endpoint: same — the existence of the endpoint record implies
      a deployment was attempted. ``last_probe_status="healthy"`` is the
      explicit happy path; ``unknown`` is treated as available (pre-probe).
    """
    if endpoint is None:
        return {"available": False, "reason": "endpoint_missing"}

    if not endpoint_is_operational(endpoint):
        return {"available": False, "reason": "endpoint_unhealthy"}

    mode = endpoint.endpoint_mode
    if mode == "hosted":
        if not env.get("nvidia_api_key_configured", False):
            return {"available": False, "reason": "no_nvidia_api_key"}
        if not mc.hosted_compatible:
            return {"available": False, "reason": "hosted_not_compatible"}
        return {"available": True, "reason": None}

    if mode == "self_hosted":
        return {"available": True, "reason": None}

    if mode == "local_system_managed":
        # Local deployment must be running. unknown = not yet probed = optimistic.
        if endpoint.last_probe_status == "healthy":
            return {"available": True, "reason": None}
        if endpoint.last_probe_status == "unknown":
            return {"available": True, "reason": None}
        return {"available": False, "reason": "local_not_running"}

    return {"available": False, "reason": "unknown_endpoint_mode"}


def get_endpoints_for_model_configs(
    project_id: str,
    model_configs: list[ModelConfig],
    workspace_root: str,
) -> dict[str, NimEndpoint]:
    """Bulk-fetch endpoints referenced by ``model_configs``. Returns id→endpoint.

    Used by the API routers to attach availability to each ModelConfig in
    a list response with a single round-trip to the endpoints table.
    Missing endpoints are simply absent from the dict; callers should
    handle that as ``endpoint_missing``.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return {}
    endpoint_ids = {mc.endpoint_id for mc in model_configs if mc.endpoint_id}
    if not endpoint_ids:
        return {}
    with Session(engine) as session:
        rows = (
            session.execute(
                select(NimEndpoint).where(
                    NimEndpoint.project_id == project_id,
                    NimEndpoint.endpoint_id.in_(endpoint_ids),
                )
            )
            .scalars()
            .all()
        )
        result = {ep.endpoint_id: ep for ep in rows}
        for ep in rows:
            session.expunge(ep)
    return result


def list_model_configs(
    project_id: str,
    workspace_root: str,
    eligible_role: str | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> tuple[list[ModelConfig], str | None]:
    """List ModelConfig records with optional role filtering and pagination."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return ([], None)

    with Session(engine) as session:
        # Keyset pagination on (created_at, model_config_id) DESC — the ordering
        # every other paginated list uses (see pagination.after_position). The
        # cursor must encode the same (created_at, id) keys as the ORDER BY or
        # pages can skip or repeat rows — encode/decode via the shared helpers
        # so the cursor and the ordering agree.
        stmt = (
            select(ModelConfig)
            .where(ModelConfig.project_id == project_id)
            .order_by(ModelConfig.created_at.desc(), ModelConfig.model_config_id.desc())
        )

        if cursor:
            cur_ts, cur_id = decode_cursor(cursor)
            stmt = stmt.where(
                after_position(
                    ModelConfig.created_at,
                    ModelConfig.model_config_id,
                    cur_ts,
                    cur_id,
                )
            )

        all_items = list(session.execute(stmt).scalars().all())

        # Python-side filtering for JSON list containment (catalog is small)
        if eligible_role:
            filtered: list[ModelConfig] = []
            for mc in all_items:
                roles_raw: Any = mc.eligible_roles
                if isinstance(roles_raw, str):
                    roles_raw = json.loads(roles_raw)
                if eligible_role not in roles_raw:
                    continue
                # Teacher filter: also require supports_image_input
                if eligible_role == "teacher" and not mc.supports_image_input:
                    continue
                filtered.append(mc)
            all_items = filtered

        items = all_items[:limit]
        next_cursor = (
            encode_cursor(items[-1].created_at, items[-1].model_config_id)
            if len(all_items) > limit
            else None
        )

        for mc in items:
            session.expunge(mc)

    return (items, next_cursor)


def get_model_config(
    project_id: str,
    model_config_id: str,
    workspace_root: str,
) -> ModelConfig | None:
    """Retrieve a single ModelConfig by ID."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        mc = session.execute(
            select(ModelConfig).where(
                ModelConfig.project_id == project_id,
                ModelConfig.model_config_id == model_config_id,
            )
        ).scalar_one_or_none()
        if mc is not None:
            session.expunge(mc)
        return mc


def update_model_config(
    project_id: str,
    model_config_id: str,
    updates: dict[str, Any],
    workspace_root: str,
) -> ModelConfig | str | None:
    """Update a ModelConfig record. Returns ModelConfig, error string, or None."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        mc = session.execute(
            select(ModelConfig).where(
                ModelConfig.project_id == project_id,
                ModelConfig.model_config_id == model_config_id,
            )
        ).scalar_one_or_none()
        if mc is None:
            return None

        # Validate endpoint_id if being changed
        if "endpoint_id" in updates:
            ep = session.execute(
                select(NimEndpoint).where(
                    NimEndpoint.project_id == project_id,
                    NimEndpoint.endpoint_id == updates["endpoint_id"],
                )
            ).scalar_one_or_none()
            if ep is None:
                return "endpoint_id does not belong to this project"

        # Validate eligible_roles if being changed
        if "eligible_roles" in updates:
            roles = updates["eligible_roles"]
            if not roles:
                return "eligible_roles must be non-empty"
            if "student_base" in roles and not any(
                mc.model_name.startswith(p) for p in _STUDENT_BASE_ALLOWED_PREFIXES
            ):
                return (
                    "student_base role is only allowed for Cosmos Reason2 "
                    "or Cosmos 3 models"
                )

        for key, value in updates.items():
            setattr(mc, key, value)

        session.commit()
        session.refresh(mc)
        session.expunge(mc)

    return mc


# ── Re-probe ────────────────────────────────────────────────────────────────


async def reprobe_model_config(
    project_id: str,
    model_config_id: str,
    workspace_root: str,
    settings: Settings,
) -> ModelConfig | str | None:
    """Re-probe all three capabilities for a ModelConfig.

    Returns ModelConfig on success, "active_use" if blocked (→ 409),
    or None if not found.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    probe_logger = get_logger("capability_probes", project_id=project_id)

    with Session(engine) as session:
        mc = session.execute(
            select(ModelConfig).where(
                ModelConfig.project_id == project_id,
                ModelConfig.model_config_id == model_config_id,
            )
        ).scalar_one_or_none()
        if mc is None:
            return None

        if is_model_in_active_use(session, project_id, model_config_id):
            return "active_use"

        # Reset all three to unknown before re-probing
        mc.structured_generation_support = "unknown"
        mc.thinking_toggle_support = "unknown"
        mc.visual_budget_support = "unknown"
        session.commit()

        # Resolve endpoint and auth
        endpoint = session.execute(
            select(NimEndpoint).where(
                NimEndpoint.endpoint_id == mc.endpoint_id,
            )
        ).scalar_one_or_none()

        if endpoint is None:
            mc.structured_generation_support = "unknown"
            mc.thinking_toggle_support = "unknown"
            mc.visual_budget_support = "unknown"
            session.commit()
            session.refresh(mc)
            session.expunge(mc)
            return mc

        # Snapshot values before leaving session
        base_url = endpoint.base_url
        auth_mode = endpoint.auth_mode
        model_name = mc.model_name
        thinking_mode = mc.thinking_toggle_mode
        visual_mode = mc.visual_budget_mode
        max_images = mc.max_images_per_request
        supports_image = mc.supports_image_input

    # Build auth headers (outside session)
    try:
        credential = resolve_probe_credential(endpoint, settings)
        auth_headers = nim_client.build_auth_headers(
            auth_mode=auth_mode,
            credential=credential,
        )
    except ValueError:
        auth_headers = {}

    deadline_s = float(settings.HTTP_DEADLINE_INTERACTIVE_S)

    probe_logger.info(
        "Starting capability probes for %s",
        model_name,
        extra={
            "details": {
                "model_config_id": model_config_id,
                "endpoint_id": endpoint.endpoint_id if endpoint else None,
                "probes": [
                    "structured_generation",
                    "thinking_toggle",
                    "visual_budget",
                    "image_cap_support",
                ],
            }
        },
    )

    start = time.monotonic()

    # Run all four probes concurrently. The image-cap probe is skipped
    # for text-only models (where ``supports_image_input=False``); their
    # ``max_images_per_request`` is moot.
    if supports_image:
        results = await asyncio.gather(
            probe_structured_generation(
                base_url,
                auth_headers,
                model_name,
                deadline_s,
                thinking_toggle_mode=thinking_mode,
            ),
            probe_thinking_toggle(
                base_url, auth_headers, model_name, thinking_mode, deadline_s
            ),
            probe_visual_budget(
                base_url, auth_headers, model_name, visual_mode, deadline_s
            ),
            _probe_image_cap_support(
                base_url, auth_headers, model_name, max_images, deadline_s
            ),
            return_exceptions=True,
        )
    else:
        # Text-only model: image cap probe is moot.
        partial = await asyncio.gather(
            probe_structured_generation(
                base_url,
                auth_headers,
                model_name,
                deadline_s,
                thinking_toggle_mode=thinking_mode,
            ),
            probe_thinking_toggle(
                base_url, auth_headers, model_name, thinking_mode, deadline_s
            ),
            probe_visual_budget(
                base_url, auth_headers, model_name, visual_mode, deadline_s
            ),
            return_exceptions=True,
        )
        results = list(partial) + ["unsupported"]

    # Process results — exceptions become "unknown"
    sg_result = results[0] if isinstance(results[0], str) else "unknown"
    tt_result = results[1] if isinstance(results[1], str) else "unknown"
    vb_result = results[2] if isinstance(results[2], str) else "unknown"
    ic_result = results[3] if isinstance(results[3], str) else "unknown"

    duration_ms = int((time.monotonic() - start) * 1000)

    probe_logger.info(
        "Capability probes complete for %s",
        model_name,
        extra={
            "details": {
                "model_config_id": model_config_id,
                "structured_generation": sg_result,
                "thinking_toggle": tt_result,
                "visual_budget": vb_result,
                "image_cap_support": ic_result,
                "duration_ms": duration_ms,
            }
        },
    )

    # Persist results
    with Session(engine) as session:
        mc = session.execute(
            select(ModelConfig).where(
                ModelConfig.model_config_id == model_config_id,
            )
        ).scalar_one_or_none()
        if mc is None:
            return None

        mc.structured_generation_support = sg_result
        mc.thinking_toggle_support = tt_result
        mc.visual_budget_support = vb_result
        mc.image_cap_support = ic_result
        session.commit()
        session.refresh(mc)
        session.expunge(mc)

    return mc

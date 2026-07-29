# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project lifecycle: create, read, update, list, and catalog seeding."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import Engine, func
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.audit_event import AuditEvent
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.local_nim_deployment import LocalNimDeployment
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS3_NANO_REASONER_GPU_MIN_GB,
    COSMOS3_REASONER_NIM_IMAGE,
    COSMOS3_SUPER_REASONER,
    COSMOS3_SUPER_REASONER_GPU_MIN_GB,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_2B_GPU_MIN_GB,
    COSMOS_REASON2_2B_NIM_IMAGE,
    COSMOS_REASON2_8B,
    COSMOS_REASON2_8B_GPU_MIN_GB,
    COSMOS_REASON2_8B_NIM_IMAGE,
    MINIMAX_M3,
    MISTRAL_LARGE_3,
    MISTRAL_MEDIUM_3_5,
    NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN,
    NEMOTRON_3_NANO_OMNI_GPU_MIN_GB,
    NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
    NEMOTRON_3_NANO_OMNI_REASONING,
    NEMOTRON_NANO_12B_VL,
    STEP_3_7_FLASH,
)
from vlm_feedback_loop.services.locks import acquire_project_lock, clear_lock_state
from vlm_feedback_loop.services.nim_client import NIM_DEFAULT_HEADERS

logger = logging.getLogger("vlm_feedback_loop.services.project")


# ── Workspace layout ─────────────────────────────────────────────────────────
#
# The on-disk layout rule lives here and nowhere else: every project's
# runtime data (project.db, exports/, artifacts/, logs/) sits under
# {WORKSPACE_ROOT}/projects/{project_id}/.


def projects_root(workspace_root: str | Path) -> Path:
    """Directory holding every project directory: ``{workspace_root}/projects``."""
    return Path(workspace_root) / "projects"


def project_dir_path(workspace_root: str | Path, project_id: str) -> Path:
    """Canonical project directory: ``{workspace_root}/projects/{project_id}``."""
    return projects_root(workspace_root) / project_id


# ── Archive exceptions ───────────────────────────────────────────────────────
#
# Mapped to HTTP 409 by main.py. Distinct exceptions instead of one
# parameterized class so handlers, tests, and operator log lines can
# discriminate without string matching.


class AlreadyArchivedError(Exception):
    """Project is already archived."""


class NotArchivedError(Exception):
    """Project is not archived (cannot unarchive)."""


class ProjectBusyError(Exception):
    """Project has in-flight work that prevents archive.

    ``reasons`` lists the diagnostic strings the UI surfaces inline.
    """

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("Project has in-flight work; cannot archive.")
        self.reasons = reasons


class ProjectArchivedError(Exception):
    """A mutating endpoint was called against an archived project."""


# Marker file in the project directory whose presence is the lazy index over
# Project.archived_at. The DB column is the source of truth; the marker is
# rebuilt on every archive op and read by background-worker scan loops to
# short-circuit before opening/locking the DB.
ARCHIVED_MARKER_NAME = ".archived"


# Status sets used by ``_is_project_busy``. Listed here so they can be
# unit-tested independently and so a future spec change is a one-line edit.
_RUN_TERMINAL_STATUSES = frozenset(
    {"succeeded", "completed", "incomplete", "failed", "canceled"}
)
_TAO_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "canceled", "deleted"})
_LOCAL_NIM_BUSY_STATUSES = frozenset({"starting", "running"})

# ── Engine cache ─────────────────────────────────────────────────────────────

_engine_cache: dict[str, Engine] = {}


def clear_engine_cache() -> None:
    """Clear cached engines and lock state. For testing only."""
    for engine in _engine_cache.values():
        engine.dispose()
    _engine_cache.clear()
    clear_lock_state()


def set_project_engine(project_id: str, engine: Engine) -> None:
    """Store an engine in the cache. Used by startup recovery."""
    _engine_cache[project_id] = engine


def get_project_engine(project_id: str, workspace_root: str) -> Engine | None:
    """Get or create a cached engine for a project. Returns None if missing."""
    if project_id in _engine_cache:
        return _engine_cache[project_id]

    project_dir = project_dir_path(workspace_root, project_id)
    if not (project_dir / "project.db").exists():
        return None

    acquire_project_lock(project_dir)
    engine = open_project_db(project_dir)
    _engine_cache[project_id] = engine
    return engine


# ── Seeded model catalog ────────────────────────────────────────────────────
#
# Capability support values are preseeded so preseeded models
# never land in projects at "unknown". Sources, per model:
#
# LIVE-PROBED against hosted NIM (integrate.api.nvidia.com):
#   * qwen/qwen3.5-397b-a17b          → sg=supported, tt=supported, vb=unsupported
#   * mistralai/mistral-large-3       → sg=supported, tt=unsupported, vb=unsupported
#   * nvidia/nemotron-nano-12b-v2-vl  → sg=supported, tt=unsupported, vb=supported
#   * nvidia/nemotron-3-nano-omni-30b-a3b-reasoning → sg=supported,
#     tt=supported, vb=unsupported. NIM 1.7.0 documents and the local campaign
#     exercised ``chat_template_kwargs.enable_thinking=false``.

#
# INTRINSIC CAPABILITY (hosted-endpoint probe misclassifies for these
# two rows because build.nvidia.com is NVCF account-gated for Cosmos —
# the 4xx gating response is indistinguishable from a `response_format`
# rejection at the probe layer. Cosmos Reason2's intended deployment is
# local NIM via `nvcr.io/nim/nvidia/cosmos-reason2-*:1.6.0`, where both
# structured generation and the Qwen-family thinking toggle work fully;
# the Cosmos Reason2 NIM runs on the vLLM backend per NVIDIA docs):
#   * nvidia/cosmos-reason2-8b → sg=supported, tt=supported, vb=supported
#   * nvidia/cosmos-reason2-2b → sg=supported, tt=supported, vb=supported
#
# Definitional: any `*_mode="none"` row gets `*_support="unsupported"`
# because there is nothing to toggle/control.

SEEDED_MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "model_name": COSMOS_REASON2_8B,
        "context_window_tokens": 256000,
        "eligible_roles": ["teacher", "student_base"],
        "supports_image_input": True,
        "thinking_toggle_mode": "qwen_enable_thinking",
        "thinking_toggle_support": "supported",
        "visual_budget_mode": "mm_processor_size",
        "visual_budget_support": "supported",
        "structured_generation_support": "supported",
        # Enforced on local deploy: local_nim_service sets the NIM container's
        # NIM_MAX_IMAGES_PER_PROMPT to this value, so the NIM accepts exactly
        # what the backend's ICL pruner sends regardless of the profile default
        # (the cosmos :1.7.0 profile defaults to a silent 5; :1.6.0 ≈999). Do
        # NOT treat this as the live cap — it is the contract we impose.
        # 999 removes the artificial wire-format image limit so it never binds
        # below the model's measured depth default: effective ICL depth is
        # governed by ``default_icl_max_examples`` + adaptive-K (Selective-K),
        # never by context filling.
        "max_images_per_request": 999,
        # July 2026 depth studies: CR2-8B is the only model measured to keep
        # gaining at 16-shot (June fixed-pool sweep 0.721 → 0.770 at 16 while
        # CR2-2B collapsed); band 8–16, seeded at the measured-safe top.
        "default_icl_max_examples": 16,
        # Hosted build.nvidia.com lists Cosmos in GET /v1/models but
        # gates inference behind an NVCF function subscription most
        # NVIDIA_API_KEY accounts don't carry — 404 Not Found on call.
        # Stays usable through a local NIM deployment (see
        # local_deploy_metadata) or a self-hosted endpoint, both of
        # which ignore this flag.
        "hosted_compatible": False,
        "local_deploy_metadata": {
            "nim_container_image": COSMOS_REASON2_8B_NIM_IMAGE,
            "nim_gpu_memory_minimum_gb": COSMOS_REASON2_8B_GPU_MIN_GB,
            "preferred_host_port": 8000,
        },
    },
    {
        "model_name": COSMOS_REASON2_2B,
        "context_window_tokens": 256000,
        "eligible_roles": ["teacher", "student_base"],
        "supports_image_input": True,
        "thinking_toggle_mode": "qwen_enable_thinking",
        "thinking_toggle_support": "supported",
        "visual_budget_mode": "mm_processor_size",
        "visual_budget_support": "supported",
        "structured_generation_support": "supported",
        # High cap (999) — see cosmos-reason2-8b note: the wire-format limit
        # never binds; effective depth is the model's depth default + adaptive-K.
        "max_images_per_request": 999,
        # July 2026 depth studies: CR2-2B holds through ~8-shot but collapses
        # at 16 (rel16 0.728 → 0.475, June fixed-pool sweep) — "≤8, never 16".
        "default_icl_max_examples": 8,
        # See cosmos-reason2-8b note above — NVCF-gated on hosted, but
        # locally deployable on smaller GPUs.
        "hosted_compatible": False,
        "local_deploy_metadata": {
            "nim_container_image": COSMOS_REASON2_2B_NIM_IMAGE,
            "nim_gpu_memory_minimum_gb": COSMOS_REASON2_2B_GPU_MIN_GB,
            "preferred_host_port": 8000,
        },
    },
    # Cosmos 3 reasoning tower. The fine-tune/serve target is the
    # ``-Reasoner`` repo (plain ``qwen3_vl`` VLM) — NOT the omni generator
    # ``nvidia/Cosmos3-{Nano,Super}`` (diffusion + VAE, not a VLM). SFT is
    # proven end-to-end on TAO FTMS; the training HF path resolves via
    # ``model_catalog_constants.HF_MODEL_PATHS``. Capability flags mirror
    # the Cosmos Reason2 qwen-family rows (intrinsic; the hosted probe
    # misclassifies the gated repos). Both sizes share ONE NIM image
    # (`cosmos3-reasoner:1.7.0`) selected by ``NIM_MODEL_SIZE`` rather than a
    # per-size image. GPU minimums are measured estimates (Nano ~8B-class;
    # Super ~30B-class fits in ~88 GB).
    {
        "model_name": COSMOS3_NANO_REASONER,
        "context_window_tokens": 131072,
        "eligible_roles": ["teacher", "student_base"],
        "supports_image_input": True,
        "thinking_toggle_mode": "qwen_enable_thinking",
        "thinking_toggle_support": "supported",
        "visual_budget_mode": "mm_processor_size",
        "visual_budget_support": "supported",
        "structured_generation_support": "supported",
        # The cosmos3-reasoner :1.7.0 NIM ships a silent profile default of
        # NIM_MAX_IMAGES_PER_PROMPT=5, but the local deploy LIFTS it via the
        # ``-e NIM_MAX_IMAGES_PER_PROMPT`` flag (built from this value) — so
        # seed the high cap (999, same as cosmos-reason2-{2b,8b}) the deploy
        # actually sets, keeping the NIM and the backend's ICL image-budget
        # pruner in agreement (proven recipe: run_icl_generalization_xl.sh).
        "max_images_per_request": 999,
        # July 2026 depth studies: CR3 is monotonic-up through the ~8-shot
        # band on in-capacity tasks (adaptive attach at K̄ 9–11 still paying
        # on visa); overshoot only ≥16-shot. Adaptive-K stays the per-query
        # mechanism — this default bounds it, not replaces it.
        "default_icl_max_examples": 8,
        "hosted_compatible": False,
        "local_deploy_metadata": {
            "nim_container_image": COSMOS3_REASONER_NIM_IMAGE,
            "nim_model_size": "nano",
            # Pin the nano fp8 profile id so the deploy never relies on the
            # fragile size-filter auto-selector, which silently falls back to
            # resident SUPER weights when the nano profile/weights are absent
            # (the silent-fallback footgun). This id is the RTX PRO 6000 /
            # fp8 / tp1 nano profile. Super omits this — it auto-selects
            # correctly off its cached default weights.
            "nim_model_profile": (
                "e2e00f3e555bb4fe0ef011faadd56a37441c7274e149d482cfeb67dbfb75b092"
            ),
            "nim_gpu_memory_minimum_gb": COSMOS3_NANO_REASONER_GPU_MIN_GB,
            "preferred_host_port": 8000,
        },
    },
    {
        "model_name": COSMOS3_SUPER_REASONER,
        "context_window_tokens": 131072,
        "eligible_roles": ["teacher", "student_base"],
        "supports_image_input": True,
        "thinking_toggle_mode": "qwen_enable_thinking",
        "thinking_toggle_support": "supported",
        "visual_budget_mode": "mm_processor_size",
        "visual_budget_support": "supported",
        "structured_generation_support": "supported",
        # See cosmos3-nano-reasoner note: the deploy lifts the :1.7.0 profile
        # default of 5 to this value via ``-e NIM_MAX_IMAGES_PER_PROMPT``.
        "max_images_per_request": 999,
        # See cosmos3-nano-reasoner depth note — same CR3 monotonic-up-to-~8
        # evidence (Super grocery 0.224 → 0.783 zero→8-shot, 0.727 at 16).
        "default_icl_max_examples": 8,
        "hosted_compatible": False,
        "local_deploy_metadata": {
            "nim_container_image": COSMOS3_REASONER_NIM_IMAGE,
            "nim_model_size": "super",
            "nim_gpu_memory_minimum_gb": COSMOS3_SUPER_REASONER_GPU_MIN_GB,
            "preferred_host_port": 8000,
            # ~30B tier needs BOTH tensor parallelism AND a bf16 optimizer
            # master to fit 8×80 GB. tp=8 × dp_shard=1 maxes activation/load
            # sharding; master_dtype=bfloat16 halves the AdamW master/m/v
            # state (the real hog — ~16 bytes/param at fp32, dp_shard-bound,
            # so tp alone doesn't relieve it). This config is live-verified
            # (~55 GB/GPU resident through a full epoch + checkpoint);
            # tp=1 and tp=4 both OOM at the 79 GB ceiling.
            # training_suite_service emits tao_train_parallelism as
            # policy.parallelism and tao_train_overrides into the train.* spec.
            "tao_train_parallelism": {
                "tp_size": 8,
                "dp_shard_size": 1,
                "cp_size": 1,
                "dp_replicate_size": 1,
                "pp_size": 1,
                "n_init_replicas": 1,
            },
            "tao_train_overrides": {"master_dtype": "bfloat16"},
        },
    },
    # Qwen 3.5 397B was removed from the catalog on 2026-07-23: NVIDIA retired
    # its hosted API on 2026-07-27 with no NVIDIA-hosted successor, and the endpoint
    # already 404s/DEGRADEs on the standard key. The QWEN_3_5 constant stays
    # in model_catalog_constants for the AutoRun-compare display map.
    {
        "model_name": MISTRAL_LARGE_3,
        "context_window_tokens": 262144,
        "eligible_roles": ["teacher"],
        "supports_image_input": True,
        "thinking_toggle_mode": "none",
        "thinking_toggle_support": "unsupported",
        "visual_budget_mode": "none",
        "visual_budget_support": "unsupported",
        "structured_generation_support": "supported",
        # Live-probed: hosted Mistral Large 3 returns
        # ``HTTP 400: At most 8 image(s) may be provided in one prompt``
        # at N=9 — the real cap is 8.
        "max_images_per_request": 8,
        "image_cap_support": "supported",
        # July 2026 depth studies: ceiling-class teachers are depth-
        # insensitive (Mistral Medium 3.5 scored 1.000 at k=2 and k=9 on
        # rps), so shallow is the cost floor — every exemplar past 2 buys
        # tokens/latency and nothing else.
        "default_icl_max_examples": 2,
        "local_deploy_metadata": None,
    },
    {
        "model_name": NEMOTRON_NANO_12B_VL,
        "context_window_tokens": 128000,
        "eligible_roles": ["teacher"],
        "supports_image_input": True,
        "thinking_toggle_mode": "none",
        "thinking_toggle_support": "unsupported",
        "visual_budget_mode": "mm_processor_tiles",
        "visual_budget_support": "supported",
        "structured_generation_support": "supported",
        # Live-probed: hosted Nemotron Nano VL returns
        # ``HTTP 400: At most 10 image(s) may be provided in one prompt``
        # at N=12 — the real cap is 10.
        "max_images_per_request": 10,
        "image_cap_support": "supported",
        # July 2026 depth studies: the Nemotron family overshoots past ~2
        # shots on EVERY task measured (rps k=2 0.993 → k=9 0.899; trashnet
        # k=2 0.731 → k=9 0.681 despite a rich 61-edit pool) — a model
        # property, not a task artifact.
        "default_icl_max_examples": 2,
        "local_deploy_metadata": None,
    },
    {
        # Vision+reasoning Teacher available through hosted NVIDIA NIM and the
        # official specialized local NIM. The local 1.7.0 runtime is the
        # measured quality leader on the Blueprint's long-horizon ICL matrix;
        # it is intentionally Teacher-only (Cosmos remains the trainable
        # Student base).
        "model_name": NEMOTRON_3_NANO_OMNI_REASONING,
        "context_window_tokens": 128000,
        "eligible_roles": ["teacher"],
        "supports_image_input": True,
        # NIM 1.7.0 accepts the Qwen chat-template enable_thinking switch.
        # Thinking defaults on at the project level and receives reasoning
        # headroom; operators can turn it off for the faster, schema-stabler
        # labeling regime used by the local ICL campaign.
        "thinking_toggle_mode": "qwen_enable_thinking",
        "thinking_toggle_support": "supported",
        "visual_budget_mode": "none",
        "visual_budget_support": "unsupported",
        "structured_generation_support": "supported",
        "max_images_per_request": 8,
        "image_cap_support": "unknown",
        # July 2026 held-out hero-story follow-up raised Omni's adaptive
        # ceiling to 4 so the model can receive a substantive demonstration
        # set. Adaptive-K still selects fewer examples when the relevance gap
        # warrants it, and explicit run/project overrides still win.
        "default_icl_max_examples": 4,
        "local_deploy_metadata": {
            "nim_container_image": NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
            "nim_gpu_memory_minimum_gb": NEMOTRON_3_NANO_OMNI_GPU_MIN_GB,
            "nim_compute_capability_minimum": (
                NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN
            ),
            "preferred_host_port": 8000,
        },
    },
    # ── 2026-07-21 additions: certified pv-2×2 campaign teachers ─────────
    # Seeded from the multi-domain guidance×rationale campaign and the
    # same-day hosted latency/binding study. All capability values below
    # are live-probed 2026-07-21 on hosted
    # build.nvidia.com; provider error strings quoted where they are the
    # evidence.
    {
        # Was the hosted default; replaced on 2026-07-23 after its hosted
        # endpoint regressed — the always-on
        # reasoning trace grew ~3-6× (≈700 chars in July → 2,000-4,200 on
        # 2026-07-23), tripling latency (steady p50 6.7 s → 21.0 s, p90 28.1 s,
        # one truncation). Certified accuracy stays strong (5-dataset
        # attempted-EM avg 0.865, second only to Qwen 3.5's 0.891), so it
        # stays a selectable alternate. Reasoning-by-default: emits
        # ``reasoning_content`` on every call and
        # ``chat_template_kwargs={"enable_thinking": false}`` is
        # accepted-but-ignored (reasoning persisted, live-probed) — so
        # always_on_reasoning, which buys it the
        # MODEL_REASONING_HEADROOM_TOKENS output budget.
        "model_name": STEP_3_7_FLASH,
        # Live-probed: max_tokens overflow error reports
        # ``max_model_len=max_total_tokens=262144``.
        "context_window_tokens": 262144,
        "eligible_roles": ["teacher"],
        "supports_image_input": True,
        "thinking_toggle_mode": "always_on_reasoning",
        "thinking_toggle_support": "unsupported",
        # Live-probed: mm_processor_kwargs rejected outright
        # (``Step3VLProcessingInfo.get_hf_processor() got an unexpected
        # keyword argument 'size'`` / ``'max_num_tiles'``).
        "visual_budget_mode": "none",
        "visual_budget_support": "unsupported",
        # Live-probed: strict json_schema response_format returns valid
        # schema-shaped JSON.
        "structured_generation_support": "supported",
        # Live-probed: ``At most 8 image(s) may be provided in one
        # prompt.`` at N=9.
        "max_images_per_request": 8,
        "image_cap_support": "supported",
        # Depth unmeasured for this model; near-ceiling teachers are
        # depth-insensitive so 2 is the measured cost floor.
        "default_icl_max_examples": 2,
        "local_deploy_metadata": None,
    },
    {
        # The hosted default Teacher since 2026-07-23. Deep-ICL-capable — the
        # only hosted Teacher measured past a 32-image request,
        # so it keeps benefiting as Verified Edits accumulate (every other
        # hosted Teacher is provider-capped at 8-10 images). 5-dataset
        # attempted-EM avg 0.823 — the top of the reachable field (Qwen 3.5
        # 397B's 0.891 was higher, but its hosted API is retired 2026-07-27
        # and it is removed from the seed). Re-measured
        # 2026-07-23 at 13.9 s cold / 15.0 s steady p50 (p90 19.9 s), down
        # from July's 27/36 s. Practical request ceiling is the provider's
        # ~5 MB body cap, not an image count (pv relaunch notes 2026-07-04).
        "model_name": MINIMAX_M3,
        # Live-probed floor: max_tokens=500000 accepted, 1M+ answered
        # with 200-and-empty-choices (MiniMax's over-limit shape). Seeded
        # at the proven floor rather than a vendor claim.
        "context_window_tokens": 500000,
        "eligible_roles": ["teacher"],
        "supports_image_input": True,
        # ``chat_template_kwargs`` neither errors nor acts as a real
        # toggle (enable_thinking true and false both emit the same
        # trace; baseline emits none) — no usable knob.
        "thinking_toggle_mode": "none",
        "thinking_toggle_support": "unsupported",
        # Live-probed: mm_processor_kwargs silently ignored
        # (prompt_tokens identical across budgets).
        "visual_budget_mode": "none",
        "visual_budget_support": "unsupported",
        # Live-probed: strict json_schema response_format returns valid
        # schema-shaped JSON.
        "structured_generation_support": "supported",
        # Live-probed ladder found NO 400 boundary through 33 images —
        # 32 is the pv-campaign-validated operating point, kept because
        # the ~5 MB body cap bites first on real photos; support stays
        # "unknown" (no provider-enforced count to pin).
        "max_images_per_request": 32,
        "image_cap_support": "unknown",
        # The certified 5-dataset campaign exercised adaptive depths above
        # two on every dataset (2.5-5.5 average). A same-pool VisA recheck
        # on 2026-07-27 recovered the prior operating point at cap 8
        # (5.97 average), while cap 2 left measurable ICL quality unused.
        # Adaptive-K still trims easy queries below this ceiling.
        "default_icl_max_examples": 8,
        "local_deploy_metadata": None,
    },
    {
        # Near-ceiling Mistral-family alternate (5-dataset attempted-EM
        # avg 0.821; 0.0% model-error on every campaign dataset; 1.000 at
        # k=2 and k=9 on rps). NOTE: measured 2026-07-21 at 62 s cold /
        # 63 s steady p50 — the same congestion band as Mistral Large 3,
        # so it is an accuracy alternate, not a latency escape.
        "model_name": MISTRAL_MEDIUM_3_5,
        # Live-probed: over-limit error names
        # ``maximum context length is 262144 tokens``.
        "context_window_tokens": 262144,
        "eligible_roles": ["teacher"],
        "supports_image_input": True,
        # Live-probed: ``chat_template is not supported for Mistral
        # tokenizers`` (400) — same no-thinking contract as Large.
        "thinking_toggle_mode": "none",
        "thinking_toggle_support": "unsupported",
        # Live-probed: mm_processor_kwargs silently ignored.
        "visual_budget_mode": "none",
        "visual_budget_support": "unsupported",
        # Live-probed: strict json_schema response_format returns valid
        # schema-shaped JSON.
        "structured_generation_support": "supported",
        # Live-probed: ``At most 10 image(s) may be provided in one
        # prompt.`` at N=11.
        "max_images_per_request": 10,
        "image_cap_support": "supported",
        # Measured ceiling-class: depth-insensitive, 2 is the cost floor.
        "default_icl_max_examples": 2,
        "local_deploy_metadata": None,
    },
]

# Default Teacher selection — MiniMax-M3 (2026-07-23, superseding Step 3.7
# Flash). The value is read
# from settings.DEFAULT_TEACHER_MODEL (config-overridable), NOT hardcoded
# elsewhere — the auth probe and the Confirm Defaults screen both resolve it
# from that setting / the /v1/environment default_teacher_model_name field.
# The reseat evidence:
#   * step regressed: its always-on reasoning trace grew ~3-6× (≈700 chars in
#     July → 2,000-4,200 now), tripling latency (steady p50 6.7 s → 21.0 s,
#     p90 28.1 s, one truncation) — the speed edge that won it the default is
#     gone
#   * accuracy: MiniMax-M3 is 0.823 avg attempted-EM across the certified
#     5-dataset pv 2×2 campaign — the top of the reachable field now that
#     Qwen 3.5 397B (0.891) is retired from the hosted API (2026-07-27) and
#     removed from the seed
#   * ICL-over-time: MiniMax-M3 is the ONLY hosted Teacher that uses a deep
#     ICL pool (>8 images; every other hosted Teacher is provider-capped at
#     8-10), so it keeps benefiting as Verified Edits accumulate
#   * latency: 15.0 s steady p50 / 19.9 s p90 (re-measured 2026-07-23, down
#     from July's 36 s) — lower and tighter than the regressed incumbent
# Step 3.7 Flash and the Mistral models remain reachable alternates via the
# top-bar Teacher picker. Local selection is separate and quality-first:
# Omni on supported ≥80 GB / cc≥9.0 GPUs, CR3-Nano at ≥56 GB when Omni is
# ineligible, and Cosmos Reason2 2B at 36–55 GB (see
# environment._pick_local_teacher_recommendation).
# Defaults that govern new-project creation. Module-level constants are
# kept for tests and operator tooling that import them directly; runtime project
# creation reads the *effective* values from Settings, which lets operators
# override in ~/.vlm_feedback_loop/config.yaml without editing source.
#
# DEFAULT_TEACHER_LOCAL_BASE_URL: when set, create_project probes that URL's
# /health/ready at create time. If healthy, a self_hosted NimEndpoint is
# seeded and the model_config matching settings.DEFAULT_TEACHER_MODEL is
# rebound to it — required when the chosen default Teacher (e.g.
# cosmos-reason2-8b) isn't served by the hosted catalog at build.nvidia.com.
#
# (The Teacher default itself is read from settings.DEFAULT_TEACHER_MODEL,
# not a module constant.)


# ── Internal helpers ─────────────────────────────────────────────────────────


def _probe_local_teacher(base_url: str) -> bool:
    """Best-effort GET on a local NIM /health/ready. Returns True iff 200.

    Synchronous by design: it runs inside the synchronous ``create_project``
    seeding path with a hard 1 s timeout against a loopback NIM. Uses httpx
    (the repo's one HTTP library) with the Blueprint source header rather
    than a parallel urllib mechanism.
    """
    health_url = base_url.rstrip("/") + "/health/ready"
    try:
        response = httpx.get(health_url, timeout=1.0, headers=NIM_DEFAULT_HEADERS)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _seed_local_teacher_endpoint(
    session: Session, project_id: str, base_url: str, model_name: str
) -> str:
    """Seed a self_hosted NimEndpoint pointing at a running local NIM."""
    endpoint = NimEndpoint(
        endpoint_id=generate_uuid4(),
        project_id=project_id,
        display_name=f"Local {model_name} ({base_url})",
        endpoint_mode="self_hosted",
        base_url=base_url,
        auth_mode="none",
        source_kind="user_configured",
        models_path="/models",
        health_ready_path="/health/ready",
        health_live_path="/health/live",
        metrics_path="/metrics",
    )
    session.add(endpoint)
    return endpoint.endpoint_id


def _seed_endpoint(session: Session, project_id: str, hosted_nim_base_url: str) -> str:
    """Create the default hosted NimEndpoint. Returns its endpoint_id."""
    endpoint = NimEndpoint(
        endpoint_id=generate_uuid4(),
        project_id=project_id,
        display_name="NVIDIA Hosted NIM",
        endpoint_mode="hosted",
        base_url=hosted_nim_base_url,
        auth_mode="bearer",
        source_kind="seeded_hosted",
        # base_url already includes /v1 (e.g. https://integrate.api.nvidia.com/v1)
        # so models_path must be /models, not /v1/models, to avoid double prefix.
        models_path="/models",
        health_ready_path="/health/ready",
        health_live_path="/health/live",
        metrics_path="/metrics",
    )
    session.add(endpoint)
    return endpoint.endpoint_id


def _seed_model_catalog(
    session: Session, project_id: str, endpoint_id: str
) -> dict[str, str]:
    """Insert the seeded catalog entries. Returns {model_name: model_config_id}."""
    name_to_id: dict[str, str] = {}
    for entry in SEEDED_MODEL_CATALOG:
        config_id = generate_uuid4()
        mc = ModelConfig(
            model_config_id=config_id,
            project_id=project_id,
            endpoint_id=endpoint_id,
            model_name=entry["model_name"],
            context_window_tokens=entry["context_window_tokens"],
            eligible_roles=entry["eligible_roles"],
            supports_image_input=entry["supports_image_input"],
            thinking_toggle_mode=entry["thinking_toggle_mode"],
            thinking_toggle_support=entry["thinking_toggle_support"],
            visual_budget_mode=entry["visual_budget_mode"],
            visual_budget_support=entry["visual_budget_support"],
            structured_generation_support=entry["structured_generation_support"],
            max_images_per_request=entry["max_images_per_request"],
            image_cap_support=entry.get("image_cap_support", "unknown"),
            # Absent = None = no per-model depth default (qwen unmeasured)
            # — selection behaves exactly as pre-column.
            default_icl_max_examples=entry.get("default_icl_max_examples"),
            local_deploy_metadata=entry["local_deploy_metadata"],
            hosted_compatible=entry.get("hosted_compatible", True),
        )
        session.add(mc)
        name_to_id[entry["model_name"]] = config_id
    return name_to_id


def _get_project_counts(session: Session, project_id: str) -> dict[str, int]:
    """Compute example state counts for a project."""
    rows = (
        session.query(Example.state, func.count())
        .filter(Example.project_id == project_id)
        .group_by(Example.state)
        .all()
    )
    state_counts: dict[str, int] = {row[0]: row[1] for row in rows}

    pending_relabel = (
        session.query(func.count())
        .filter(
            Example.project_id == project_id,
            Example.state == "Unlabeled",
            Example.prior_verified_label_ref.isnot(None),
        )
        .scalar()
    ) or 0

    # Verified examples that carry a prior_verified_label_ref are priors the
    # SME has already re-labeled after a semantic Core change. Summing
    # with ``pending_relabel`` gives the full "N of M re-labeled" denominator
    # the re-label progress strip renders.
    prior_relabeled = (
        session.query(func.count())
        .filter(
            Example.project_id == project_id,
            Example.state == "Verified",
            Example.prior_verified_label_ref.isnot(None),
        )
        .scalar()
    ) or 0

    return {
        "verified": state_counts.get("Verified", 0),
        "unlabeled": state_counts.get("Unlabeled", 0),
        "auto_labeled": state_counts.get("Auto-Labeled", 0),
        "omitted": state_counts.get("Omitted", 0),
        "pending_relabel": pending_relabel,
        "prior_relabeled": prior_relabeled,
    }


# ── Public API ───────────────────────────────────────────────────────────────


def create_project(
    name: str,
    description: str | None,
    settings: Settings,
    *,
    preferred_local_teacher_model_name: str | None = None,
) -> Project:
    """Create a project and adopt an exact running local Teacher when present."""
    project_id = generate_uuid4()
    project_dir = project_dir_path(settings.WORKSPACE_ROOT, project_id)

    # Create directory tree (auto-creates WORKSPACE_ROOT if missing)
    project_dir.mkdir(parents=True, exist_ok=True)
    for subdir in [
        "exports",
        "artifacts",
        "logs",
        "logs/operations",
        "logs/runs",
    ]:
        (project_dir / subdir).mkdir(exist_ok=True)

    # Lock and initialize project database (WAL, migrations, etc.)
    acquire_project_lock(project_dir)
    engine = open_project_db(project_dir)
    _engine_cache[project_id] = engine

    with Session(engine) as session:
        endpoint_id = _seed_endpoint(session, project_id, settings.HOSTED_NIM_BASE_URL)
        name_to_id = _seed_model_catalog(session, project_id, endpoint_id)

        # Effective default — overridable via ~/.vlm_feedback_loop/config.yaml.
        teacher_model = settings.DEFAULT_TEACHER_MODEL

        # Auto-bind the default Teacher to a running local NIM if a base URL
        # is configured and reachable. See DEFAULT_TEACHER_LOCAL_BASE_URL
        # docstring on Settings for rationale.
        local_url = settings.DEFAULT_TEACHER_LOCAL_BASE_URL
        if local_url and _probe_local_teacher(local_url):
            local_endpoint_id = _seed_local_teacher_endpoint(
                session, project_id, local_url, teacher_model
            )
            teacher_mc = (
                session.query(ModelConfig)
                .filter_by(project_id=project_id, model_name=teacher_model)
                .first()
            )
            if teacher_mc is not None:
                teacher_mc.endpoint_id = local_endpoint_id
                logger.info(
                    "Bound %s model_config to local NIM at %s",
                    teacher_model,
                    local_url,
                )
        elif local_url:
            logger.warning(
                "DEFAULT_TEACHER_LOCAL_BASE_URL=%s configured but health probe "
                "failed; default Teacher %s will resolve to the hosted endpoint. "
                "Deploy the local NIM container or unset the URL to silence this.",
                local_url,
                teacher_model,
            )

        project = Project(
            project_id=project_id,
            name=name,
            description=description,
            project_dir=str(project_dir),
            teacher_model_config_id=name_to_id[teacher_model],
            thinking_default_on=settings.THINKING_DEFAULT_ON,
        )
        session.add(project)
        session.commit()
        session.refresh(project)

        logger.info("Created project %s at %s", project_id, project_dir)

    # Function-level import avoids the module cycle: local_nim_service uses
    # get_project_engine/projects_root from this module. The Project row and
    # seeded catalog must exist before the host-wide scan can compare exact
    # model/image/profile identity and attach a project-local endpoint.
    try:
        from vlm_feedback_loop.services.local_nim_service import (
            reuse_first_compatible_running_teacher_for_project,
        )

        reused = reuse_first_compatible_running_teacher_for_project(
            project_id=project_id,
            workspace_root=settings.WORKSPACE_ROOT,
            preferred_model_name=preferred_local_teacher_model_name,
        )
        if reused is not None:
            model_config_id, resident = reused
            with Session(engine) as session:
                persisted_project = session.get(Project, project_id)
                if persisted_project is None:
                    raise RuntimeError(
                        f"Project {project_id} disappeared during Teacher adoption"
                    )
                persisted_project.teacher_model_config_id = model_config_id
                session.commit()
                session.refresh(persisted_project)
                logger.info(
                    "Selected reused local Teacher %s for new project %s "
                    "(owner_project=%s, deployment=%s)",
                    resident.model_name,
                    project_id,
                    resident.project_id,
                    resident.deployment_id,
                )
                return persisted_project
    except Exception as exc:
        # A host-inventory problem must not make project creation impossible.
        # The seeded hosted default remains usable and setup can retry/offer
        # the resident explicitly.
        logger.warning(
            "Could not adopt a running local Teacher for new project %s (%s: %s)",
            project_id,
            type(exc).__name__,
            str(exc) or "(no message)",
        )

    return project


def get_project(project_id: str, workspace_root: str) -> Project | None:
    """Load a single project by ID. Returns None if not found."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        return session.query(Project).filter_by(project_id=project_id).first()


def get_project_counts(project_id: str, workspace_root: str) -> dict[str, int] | None:
    """Compute example-state counts for a single project.

    Returns None when the project database is missing or the Project row
    has not been persisted. Used by ``GET/POST/PATCH /v1/projects/{id}`` to
    populate ``counts`` on the ``ProjectResponse`` — the Student Training
    screen depends on ``verified`` and ``auto_labeled`` for the Training
    Data card totals.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return None
        return _get_project_counts(session, project_id)


def update_project(
    project_id: str, updates: dict[str, Any], workspace_root: str
) -> Project | None:
    """Apply a partial update to a project. Returns updated Project or None.

    Raises ValueError if a selection pointer references a record not in
    this project (cross-project ref rejection).
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    # Fields that are selection pointers requiring same-project validation
    selection_fields = {
        "teacher_model_config_id",
        "active_student_model_config_id",
    }

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return None

        # Role requirements for model-selection pointers. Only applied when
        # the caller sets the corresponding selection pointer; project
        # seeding already validates defaults. Keyed by ProjectUpdate field.
        role_required = {
            "teacher_model_config_id": "teacher",
            "active_student_model_config_id": "student_base",
        }

        for key, value in updates.items():
            # Validate selection pointers exist in this project and declare
            # the appropriate role (invariant: "A model assigned as
            # `teacher_model_config_id` MUST have the `teacher` role and
            # `supports_image_input=true`"). Same role-gate applies to the
            # Student pointer via ``role_required``.
            if key in selection_fields and value is not None:
                config = (
                    session.query(ModelConfig)
                    .filter_by(model_config_id=value, project_id=project_id)
                    .first()
                )
                if config is None:
                    raise ValueError(
                        f"{key} '{value}' does not exist in project {project_id}"
                    )
                required_role = role_required.get(key)
                if required_role and required_role not in (config.eligible_roles or []):
                    raise ValueError(
                        f"{key} '{value}' is not eligible for role '{required_role}'"
                    )
                if key == "teacher_model_config_id" and not config.supports_image_input:
                    raise ValueError(
                        f"teacher_model_config_id '{value}' does not support image input"
                    )

            if key == "active_guidance_id":
                # This pointer is set once, during FTUE, to activate the
                # first Guidance (no labels or runs exist yet). After that,
                # the active Guidance only moves through the guidance edit
                # endpoint, which cancels in-flight runs and re-points the
                # existing labels. Switching — or clearing — it directly
                # here would orphan the whole corpus from the active
                # version and let a run keep writing under the old one.
                if project.active_guidance_id and value != project.active_guidance_id:
                    raise ValueError(
                        "active_guidance_id cannot be switched or cleared directly; "
                        "edit the guidance instead so in-flight runs are canceled "
                        "and existing labels are re-pointed to the new version"
                    )
                if value is not None:
                    exists = (
                        session.query(Guidance.guidance_id)
                        .filter_by(guidance_id=value, project_id=project_id)
                        .first()
                    )
                    if exists is None:
                        raise ValueError(
                            f"active_guidance_id '{value}' does not exist in project {project_id}"
                        )

            setattr(project, key, value)

        session.commit()
        session.refresh(project)
        return project


def _is_project_busy(session: Session, project_id: str) -> list[str]:
    """Return human-readable reasons the project has in-flight work.

    Empty list ⇒ project is idle and may be archived. The reentrant
    in-process file lock (``services/locks.py``) cannot detect this
    case — it returns the cached fd whenever the same process already
    holds the lock — so this application-level check is required before
    allowing ``archive_project``.
    """
    reasons: list[str] = []

    in_flight_runs = (
        session.query(func.count(RunRecord.run_id))
        .filter(
            RunRecord.project_id == project_id,
            RunRecord.status.notin_(_RUN_TERMINAL_STATUSES),
        )
        .scalar()
    ) or 0
    if in_flight_runs:
        reasons.append(f"{in_flight_runs} evaluation/batch run(s) still in progress")

    in_flight_exports = (
        session.query(func.count(DatasetExport.dataset_export_id))
        .filter(
            DatasetExport.project_id == project_id,
            DatasetExport.status == "running",
        )
        .scalar()
    ) or 0
    if in_flight_exports:
        reasons.append(f"{in_flight_exports} dataset export(s) still building")

    in_flight_tao = (
        session.query(func.count(TAOJob.tao_job_id))
        .filter(
            TAOJob.project_id == project_id,
            TAOJob.status.notin_(_TAO_TERMINAL_STATUSES),
        )
        .scalar()
    ) or 0
    if in_flight_tao:
        reasons.append(f"{in_flight_tao} TAO job(s) still in progress")

    in_flight_nim = (
        session.query(func.count(LocalNimDeployment.local_nim_deployment_id))
        .filter(
            LocalNimDeployment.project_id == project_id,
            LocalNimDeployment.status.in_(_LOCAL_NIM_BUSY_STATUSES),
        )
        .scalar()
    ) or 0
    if in_flight_nim:
        reasons.append(f"{in_flight_nim} local NIM deployment(s) still active")

    return reasons


def _archived_marker_path(workspace_root: str, project_id: str) -> Path:
    return project_dir_path(workspace_root, project_id) / ARCHIVED_MARKER_NAME


def archive_project(project_id: str, workspace_root: str) -> Project:
    """Set ``archived_at`` on a project, write the sentinel marker, and audit.

    Raises ``AlreadyArchivedError`` if already archived,
    ``ProjectBusyError`` if any RunRecord/TAOJob/LocalNimDeployment is
    non-terminal, and lets ``ProjectLockedError`` propagate when another
    *process* holds the file lock (mapped to 409 ``project_in_use``).
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        raise FileNotFoundError(project_id)

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            raise FileNotFoundError(project_id)
        if project.archived_at is not None:
            raise AlreadyArchivedError(project_id)

        reasons = _is_project_busy(session, project_id)
        if reasons:
            raise ProjectBusyError(reasons)

        now = utc_now()
        project.archived_at = now
        session.add(
            AuditEvent(
                project_id=project_id,
                event_type="project_archived",
                event_data={"archived_at": now},
            )
        )
        session.commit()
        session.refresh(project)

    # Marker is a lazy index — write AFTER the DB commit so the marker
    # never exists for a project the authoritative DB column does not
    # yet show as archived.
    marker = _archived_marker_path(workspace_root, project_id)
    try:
        marker.write_text(now)
    except OSError as exc:
        logger.warning(
            "Could not write archived marker for %s (%s); "
            "DB column is the source of truth, list/recovery scans will "
            "still skip via the column on next open.",
            project_id,
            exc,
        )

    logger.info("Project %s archived at %s", project_id, now)
    return project


def unarchive_project(project_id: str, workspace_root: str) -> Project:
    """Clear ``archived_at`` on a project, remove the sentinel marker, audit."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        raise FileNotFoundError(project_id)

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            raise FileNotFoundError(project_id)
        if project.archived_at is None:
            raise NotArchivedError(project_id)

        prev_archived_at = project.archived_at
        project.archived_at = None
        session.add(
            AuditEvent(
                project_id=project_id,
                event_type="project_unarchived",
                event_data={"prev_archived_at": prev_archived_at},
            )
        )
        session.commit()
        session.refresh(project)

    marker = _archived_marker_path(workspace_root, project_id)
    try:
        marker.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning(
            "Could not remove archived marker for %s (%s); "
            "next archive op will overwrite it.",
            project_id,
            exc,
        )

    logger.info(
        "Project %s unarchived (was archived at %s)", project_id, prev_archived_at
    )
    return project


def mark_setup_completed(
    project_id: str,
    workspace_root: str,
    *,
    auto_skip: bool,
    teacher_mode: str,
    embedding_mode: str,
    embedding_provider: str,
    local_deploy_queued: list[str] | None = None,
) -> tuple[Project, bool]:
    """Stamp ``setup_completed_at`` on first transition; idempotent.

    Returns ``(project, transitioned)`` where ``transitioned`` is True only
    on the first call (when ``setup_completed_at`` was previously null).
    Subsequent calls are no-ops: the project is returned unchanged and no
    additional AuditEvent is emitted. This guarantees redundant calls (e.g.
    NIMConnectionPage auto-skip immediately followed by ConfirmDefaultsPage
    auto-skip) collapse to a single audit entry and a stable timestamp.

    The AuditEvent payload captures the environment recommendation context
    at stamp time (``auto_skip``, ``teacher_mode``, ``embedding_mode``,
    ``embedding_provider``). The setup-completed event is the only forensic
    record of what the SME's first-run environment looked like; an empty
    payload would waste the row.

    Raises ``FileNotFoundError`` if the project does not exist.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        raise FileNotFoundError(project_id)

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            raise FileNotFoundError(project_id)

        if project.setup_completed_at is not None:
            return project, False

        now = utc_now()
        project.setup_completed_at = now
        session.add(
            AuditEvent(
                project_id=project_id,
                event_type="setup_completed",
                event_data={
                    "setup_completed_at": now,
                    "auto_skip": auto_skip,
                    "teacher_mode": teacher_mode,
                    "embedding_mode": embedding_mode,
                    "embedding_provider": embedding_provider,
                    "local_deploy_queued": local_deploy_queued or [],
                },
            )
        )
        session.commit()
        session.refresh(project)

    logger.info(
        "Project %s setup_completed_at stamped at %s (auto_skip=%s)",
        project_id,
        now,
        auto_skip,
    )
    return project, True


def is_project_archived(project_id: str, workspace_root: str) -> bool:
    """Return True iff the project's ``archived_at`` is non-null.

    Used by the cross-mutation guard. Marker file is consulted first as a
    cheap pre-check; the DB column is authoritative.
    """
    if _archived_marker_path(workspace_root, project_id).exists():
        # Confirm via the DB column — marker may be stale if a previous
        # crash left it on disk after a manual unarchive.
        engine = get_project_engine(project_id, workspace_root)
        if engine is None:
            return False
        with Session(engine) as session:
            row = (
                session.query(Project.archived_at)
                .filter_by(project_id=project_id)
                .first()
            )
            return row is not None and row[0] is not None
    return False


def has_archived_projects(workspace_root: str) -> bool:
    """True when any project directory carries the ``.archived`` marker.

    Pure filesystem scan — never opens a project DB, so it stays cheap on
    workspaces with hundreds of archived projects. The marker is written on
    archive/unarchive and reconciled by ``list_projects`` when drift is
    detected, so it is a trustworthy existence index even though the DB
    column stays authoritative for any single project.
    """
    projects_dir = projects_root(workspace_root)
    if not projects_dir.exists():
        return False
    return any(
        (entry / ARCHIVED_MARKER_NAME).exists()
        for entry in projects_dir.iterdir()
        if entry.is_dir()
    )


def list_projects(
    workspace_root: str,
    cursor: str | None = None,
    limit: int = 50,
    include_archived: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """List projects with counts and cursor pagination.

    Returns (items, next_cursor). Items sorted by created_at descending.

    When ``include_archived`` is False (default), projects whose directory
    contains the ``.archived`` marker file are skipped before opening the
    DB — keeping the list endpoint cheap when the workspace has many
    archived projects. The marker is reconciled with the DB column on each
    archive/unarchive op, so the only drift mode is a manual or partial
    failure; when the scan detects that, the DB column wins and the marker
    is rewritten (or removed) to match it.
    """
    projects_dir = projects_root(workspace_root)
    if not projects_dir.exists():
        return [], None

    # Collect project data from each project directory
    all_items: list[dict[str, Any]] = []
    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir():
            continue
        pid = entry.name

        marker_present = (entry / ARCHIVED_MARKER_NAME).exists()
        if marker_present and not include_archived:
            continue

        # Per-project isolation: a single corrupt project DB MUST NOT
        # 500 the entire LIST endpoint (the lifespan recovery functions
        # carry the same guard). Observed failure mode: a project whose
        # Alembic state has drifted (empty ``alembic_version`` row +
        # already-created tables) makes ``_run_migrations`` raise
        # ``DatabaseMigrationError``, which would propagate a 500 to
        # ``GET /v1/projects`` even when every sibling project is
        # healthy. Catch broadly, log a per-project warning with the
        # exception class name (so empty-str exceptions still produce
        # actionable signal — see ``services/background.py
        # _on_task_done`` for the same pattern), and continue.
        try:
            engine = get_project_engine(pid, workspace_root)
            if engine is None:
                continue

            with Session(engine) as session:
                proj = session.query(Project).filter_by(project_id=pid).first()
                if proj is None:
                    continue

                archived_at = proj.archived_at
                archived = archived_at is not None
                if archived != marker_present:
                    # Drift between DB and marker file. Trust the DB column
                    # and rewrite the marker to match it, so subsequent
                    # scans (and has_archived_projects) don't keep paying
                    # a DB open for this project.
                    logger.warning(
                        "Archive marker drift for project %s "
                        "(marker=%s, archived_at=%s); DB column is "
                        "authoritative — reconciling marker.",
                        pid,
                        marker_present,
                        archived_at,
                    )
                    marker_path = entry / ARCHIVED_MARKER_NAME
                    try:
                        if archived_at is not None:
                            marker_path.write_text(archived_at)
                        else:
                            marker_path.unlink(missing_ok=True)
                    except OSError as exc:
                        logger.warning(
                            "Could not reconcile archived marker for %s: %s",
                            pid,
                            exc,
                        )

                if archived and not include_archived:
                    continue

                counts = _get_project_counts(session, pid)
                all_items.append(
                    {
                        "project_id": proj.project_id,
                        "name": proj.name,
                        "description": proj.description,
                        "created_at": proj.created_at,
                        "updated_at": proj.updated_at,
                        "counts": counts,
                        "archived_at": proj.archived_at,
                        "setup_completed_at": proj.setup_completed_at,
                    }
                )
        except Exception as exc:
            # ``str(exc) or "(no message)"`` (NOT ``exc or "(no message)"``) —
            # an Exception instance with an empty-string message is still
            # truthy in Python, so the latter would render as
            # "RuntimeError: " with no fallback. Use ``str(exc)`` to get
            # the actual message. Same observability fallback as
            # ``services/background.py::_on_task_done``.
            logger.warning(
                "Skipping project %s in list (%s: %s)",
                pid,
                type(exc).__name__,
                str(exc) or "(no message)",
            )
            continue

    all_items.sort(key=lambda x: x["created_at"], reverse=True)

    # Apply cursor pagination
    start_idx = 0
    if cursor:
        for i, item in enumerate(all_items):
            if item["project_id"] == cursor:
                start_idx = i + 1
                break

    page = all_items[start_idx : start_idx + limit]
    next_cursor = page[-1]["project_id"] if len(all_items) > start_idx + limit else None

    return page, next_cursor

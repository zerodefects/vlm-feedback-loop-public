# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration system with five-level precedence.

Precedence (highest wins):
  1. Process environment variables
  2. Explicit env file (--env-file or VLM_FEEDBACK_LOOP_ENV_FILE)
  3. Default .env at ~/.vlm_feedback_loop/.env
  4. Config file values from ~/.vlm_feedback_loop/config.yaml
  5. Built-in defaults

The backend MUST NOT search for .env in CWD, WORKSPACE_ROOT, or
project directories — only the canonical location or an explicit override.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, cast, get_args, get_origin

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from vlm_feedback_loop._defaults import DEFAULTS

logger = logging.getLogger("vlm_feedback_loop.config")

# ── Path constants (monkeypatched in tests) ──────────────────────────────────

_CONFIG_DIR: Path = Path.home() / ".vlm_feedback_loop"
_CONFIG_FILE: Path = _CONFIG_DIR / "config.yaml"
_DEFAULT_ENV_FILE: Path = _CONFIG_DIR / ".env"

# The .env file this process actually loaded, captured at load_settings time
# (honors --env-file / VLM_FEEDBACK_LOOP_ENV_FILE). get_canonical_env_file_path
# returns this so a persisted secret is written to the SAME file the process
# read. None until the first load_settings.
_active_env_file: Path | None = None


# ── Settings model ───────────────────────────────────────────────────────────


class Settings(BaseModel):
    """Application settings validated by Pydantic.

    Every configuration field is declared here with its default value.
    Secrets have ``None`` as default and are populated from .env or env vars.
    ``extra="forbid"`` catches typos in config.yaml immediately.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Required (no default) ────────────────────────────────────────
    WORKSPACE_ROOT: str = Field(
        description=(
            "Absolute path to the workspace directory that holds per-project "
            "SQLite DBs, exports, artifacts, and logs."
        ),
    )

    # ── Embedding computation ────────────────────────────────────────
    EMBEDDING_PROVIDER: str = DEFAULTS["EMBEDDING_PROVIDER"]
    EMBEDDING_MODEL_ID: str = DEFAULTS["EMBEDDING_MODEL_ID"]
    EMBEDDING_DIM: int = DEFAULTS["EMBEDDING_DIM"]
    # NeMo Retriever asymmetric embedding models require an `input_type`
    # parameter (`"query"` vs `"passage"`).  NV-CLIP and other symmetric
    # CLIP-style models do not — set to None for those.  Default
    # `"passage"` matches the project's image-indexing usage.
    EMBEDDING_INPUT_TYPE: str | None = DEFAULTS["EMBEDDING_INPUT_TYPE"]
    EMBEDDINGS_AUTO_COMPUTE: bool = DEFAULTS["EMBEDDINGS_AUTO_COMPUTE"]
    # Provider-aware concurrency + batch shape for the background
    # embedding worker.  Hosted (build.nvidia.com) is shared/rate-limited
    # — keep concurrency at 1 and batch larger requests.  Self-hosted
    # local NIMs run small concurrent requests instead.
    EMBEDDING_CONCURRENCY_HOSTED: int = DEFAULTS["EMBEDDING_CONCURRENCY_HOSTED"]
    EMBEDDING_BATCH_SIZE_HOSTED: int = DEFAULTS["EMBEDDING_BATCH_SIZE_HOSTED"]
    EMBEDDING_CONCURRENCY_SELF_HOSTED: int = DEFAULTS[
        "EMBEDDING_CONCURRENCY_SELF_HOSTED"
    ]
    EMBEDDING_BATCH_SIZE_SELF_HOSTED: int = DEFAULTS["EMBEDDING_BATCH_SIZE_SELF_HOSTED"]

    # ── Generation Controls ──────────────────────────────────────────
    # (The default labeling/visual-budget preset KEYS are
    # per-project settings — Project columns, editable via the project
    # API — not process config. The preset VALUE tables below stay here.)
    THINKING_DEFAULT_ON: bool = DEFAULTS["THINKING_DEFAULT_ON"]
    LABELING_PRESETS: dict[str, dict[str, float]] = DEFAULTS["LABELING_PRESETS"]

    # ── Visual Budget Controls ───────────────────────────────────────
    VISUAL_BUDGET_PRESETS: dict[str, Any] = DEFAULTS["VISUAL_BUDGET_PRESETS"]

    @field_validator("VISUAL_BUDGET_PRESETS", mode="after")
    @classmethod
    def _normalize_visual_budget_presets(cls, v: dict[str, Any]) -> dict[str, Any]:
        """``native`` is a reserved, product-required preset.

        The Student NIM lifecycle depends on it (training-parity evals send
        no ``mm_processor_kwargs``), and an operator ``config.yaml`` that
        lists its own preset ladder REPLACES this dict wholesale — observed
        live: a deployment whose config predated the preset lost ``native``
        and every Student serving eval failed instantly on preset
        validation. Reserved presets are merged back, never trusted to the
        override.
        """
        normalized: dict[str, Any] = copy.deepcopy(v)
        if "native" not in normalized:
            normalized["native"] = {}
        return normalized

    IMAGE_TRANSPORT_MAX_LONGEST_EDGE: int | None = DEFAULTS[
        "IMAGE_TRANSPORT_MAX_LONGEST_EDGE"
    ]

    @field_validator("IMAGE_TRANSPORT_MAX_LONGEST_EDGE", mode="after")
    @classmethod
    def _normalize_transport_edge(cls, v: int | None) -> int | None:
        """0 (or any non-positive value) disables transport downscaling."""
        if v is not None and v <= 0:
            return None
        return v

    # ── Prompt budgets ───────────────────────────────────────────────
    RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE: int | None = DEFAULTS[
        "RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE"
    ]
    RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN: float = DEFAULTS[
        "RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN"
    ]
    BASE_OUTPUT_TOKENS_FLOOR: int = DEFAULTS["BASE_OUTPUT_TOKENS_FLOOR"]
    JSON_STRUCTURAL_OVERHEAD_TOKENS: int = DEFAULTS["JSON_STRUCTURAL_OVERHEAD_TOKENS"]
    MAX_OUTPUT_FRACTION: float = DEFAULTS["MAX_OUTPUT_FRACTION"]
    RATIONALE_NOTE_ESTIMATE_TOKENS: int = DEFAULTS["RATIONALE_NOTE_ESTIMATE_TOKENS"]
    DEFAULT_UNBOUNDED_STRING_BUDGET: int = DEFAULTS["DEFAULT_UNBOUNDED_STRING_BUDGET"]
    MODEL_REASONING_HEADROOM_TOKENS: int = DEFAULTS["MODEL_REASONING_HEADROOM_TOKENS"]

    # ── ICL selection ────────────────────────────────────────────────
    ICL_MAX_EXAMPLES: int | None = DEFAULTS["ICL_MAX_EXAMPLES"]
    ICL_SIM_GAP: float | None = DEFAULTS["ICL_SIM_GAP"]
    ICL_ABS_THRESHOLD: float | None = DEFAULTS["ICL_ABS_THRESHOLD"]

    # ── Batch labeling ───────────────────────────────────────────────
    BATCH_LABEL_RUN_LIMIT: int | None = DEFAULTS["BATCH_LABEL_RUN_LIMIT"]
    BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD: int = DEFAULTS[
        "BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD"
    ]
    BATCH_LABEL_CONCURRENCY_HOSTED: int = DEFAULTS["BATCH_LABEL_CONCURRENCY_HOSTED"]
    BATCH_LABEL_CONCURRENCY_SELF_HOSTED: int = DEFAULTS[
        "BATCH_LABEL_CONCURRENCY_SELF_HOSTED"
    ]

    # ── Student training ─────────────────────────────────────────────
    TAO_RELEASE_VERSION: str = DEFAULTS["TAO_RELEASE_VERSION"]
    COSMOS_RL_CONTAINER_TAG: str = DEFAULTS["COSMOS_RL_CONTAINER_TAG"]
    TAO_JOB_TIMEOUT_MINUTES: int = Field(
        default=DEFAULTS["TAO_JOB_TIMEOUT_MINUTES"],
        ge=1,
        description=(
            "TAO stale-heartbeat ceiling applied to every training-suite job. "
            "This is a dead-job reaper, not an expected training duration."
        ),
    )
    STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD: float = DEFAULTS[
        "STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD"
    ]
    MERGE_LORA_PYTHON: str | None = DEFAULTS["MERGE_LORA_PYTHON"]

    # ── NIM serving benchmark ────────────────────────────────────────
    NIM_STARTUP_TIMEOUT_S: int = DEFAULTS["NIM_STARTUP_TIMEOUT_S"]
    NIM_BENCHMARK_TIMEOUT_S: int = DEFAULTS["NIM_BENCHMARK_TIMEOUT_S"]
    STUDENT_LATENCY_TEST_CONCURRENCIES: list[int] = DEFAULTS[
        "STUDENT_LATENCY_TEST_CONCURRENCIES"
    ]

    # ── Student NIM deployment ───────────────────────────────────────
    NIM_GPU_MEMORY_8B_BF16_GB: int = DEFAULTS["NIM_GPU_MEMORY_8B_BF16_GB"]
    NIM_GPU_MEMORY_8B_FP8_GB: int = DEFAULTS["NIM_GPU_MEMORY_8B_FP8_GB"]
    NIM_GPU_MEMORY_2B_BF16_GB: int = DEFAULTS["NIM_GPU_MEMORY_2B_BF16_GB"]
    NIM_GPU_MEMORY_2B_FP8_GB: int = DEFAULTS["NIM_GPU_MEMORY_2B_FP8_GB"]
    NIM_STUDENT_PORT: int = DEFAULTS["NIM_STUDENT_PORT"]

    # ── Review selector ──────────────────────────────────────────────
    REVIEW_SELECTION_MODE: str = DEFAULTS["REVIEW_SELECTION_MODE"]
    REVIEW_RECENT_WINDOW_K: int = DEFAULTS["REVIEW_RECENT_WINDOW_K"]
    CLIP_SWITCHOVER_MIN_COUNT: int = DEFAULTS["CLIP_SWITCHOVER_MIN_COUNT"]

    # ── Schema refinement reminders ──────────────────────────────────
    SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1: int = DEFAULTS[
        "SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1"
    ]
    SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2: int = DEFAULTS[
        "SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2"
    ]

    # ── Runtime rejection fallbacks ──────────────────────────────────
    # Kill-switches for the per-invocation thinking-toggle and
    # visual-budget rejection fallbacks. Default True;
    # flip False to run in telemetry-only mode (the detector populates
    # ``OperationRecord.{thinking,visual_budget}_*_attempted`` but no
    # auto-retry / run-fail).
    ENABLE_THINKING_TOGGLE_FALLBACK: bool = DEFAULTS["ENABLE_THINKING_TOGGLE_FALLBACK"]
    ENABLE_VISUAL_BUDGET_FALLBACK: bool = DEFAULTS["ENABLE_VISUAL_BUDGET_FALLBACK"]

    # ── Evaluation ───────────────────────────────────────────────────
    EVAL_CONCURRENCY_HOSTED: int = DEFAULTS["EVAL_CONCURRENCY_HOSTED"]
    EVAL_CONCURRENCY_SELF_HOSTED: int = DEFAULTS["EVAL_CONCURRENCY_SELF_HOSTED"]
    FOREGROUND_HOLD_ENABLED: bool = DEFAULTS["FOREGROUND_HOLD_ENABLED"]
    EVAL_RATE_LIMIT_RETRY_MAX_PASSES: int = DEFAULTS["EVAL_RATE_LIMIT_RETRY_MAX_PASSES"]
    EVAL_RATE_LIMIT_RETRY_BACKOFF_S: float = DEFAULTS["EVAL_RATE_LIMIT_RETRY_BACKOFF_S"]
    HOSTED_GLOBAL_RPM: int = DEFAULTS["HOSTED_GLOBAL_RPM"]
    EVAL_FIRST_POOL_SIZE: int = DEFAULTS["EVAL_FIRST_POOL_SIZE"]

    # ── HTTP client ──────────────────────────────────────────────────
    HTTP_DEADLINE_INTERACTIVE_S: int = DEFAULTS["HTTP_DEADLINE_INTERACTIVE_S"]
    HTTP_DEADLINE_BACKGROUND_S: int = DEFAULTS["HTTP_DEADLINE_BACKGROUND_S"]
    HTTP_MAX_RETRIES: int = DEFAULTS["HTTP_MAX_RETRIES"]

    TAO_POLL_MIN_INTERVAL_S: int = DEFAULTS["TAO_POLL_MIN_INTERVAL_S"]
    TAO_POLL_MIN_INTERVAL_RUNNING_S: int = DEFAULTS["TAO_POLL_MIN_INTERVAL_RUNNING_S"]
    TAO_POLL_TICK_S: int = DEFAULTS["TAO_POLL_TICK_S"]
    TAO_AUTOEVAL_SKIP_BASES: list[str] = Field(
        default_factory=lambda: list(DEFAULTS["TAO_AUTOEVAL_SKIP_BASES"]),
        description=(
            "Exact ``ModelConfig.model_name`` values whose "
            "TAO ``evaluate`` action is currently broken upstream and should "
            "be auto-skipped during chain advancement. Default contains "
            "``nvidia/cosmos-reason2-8b`` (TAO ``cosmos-rl-evaluate`` fails "
            "on every Cosmos-Reason2-8B trained checkpoint with a weight-init "
            "loader gap) and both CR3 reasoners "
            "(``nvidia/cosmos3-{nano,super}-reasoner`` — cosmos-rl evaluates "
            "them via the generic HFModel/qwen3_vl fallback whose freeform "
            "decode is unreliable; NVIDIA engineering ticket open as of "
            "2026-07-14; quality routes through the §9.5 NIM-eval fallback "
            "instead). Clear entries once TAO ships an upstream "
            "fix. Each chain whose train job's base is on this list will "
            "see the post-train ``evaluate`` TAOJob marked ``canceled`` with "
            "``chain_halted_reason`` instead of being submitted; downstream "
            "``quantize`` siblings continue normally (chain-isolation rule). "
            'The Student lands at ``quality_status="pending"`` (the cold-start '
            "branch of the NIM-eval fallback) instead of ``failed``."
        ),
    )

    # ── Logging ──────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(
        default=DEFAULTS["LOG_LEVEL"],
        description=(
            "Logging verbosity. 'info' by default; set to 'debug' during "
            "development for ICL/token-budget/selector decision traces."
        ),
    )

    # ── Server ───────────────────────────────────────────────────────
    BIND_HOST: str = Field(
        default=DEFAULTS["BIND_HOST"],
        description=(
            "Uvicorn bind host. Loopback by default. When set to a non-"
            "loopback address, IMAGE_ROOT MUST be configured."
        ),
    )
    BIND_PORT: int = Field(
        default=DEFAULTS["BIND_PORT"],
        description="Uvicorn bind port.",
    )
    IMAGE_ROOT: str | None = Field(
        default=DEFAULTS["IMAGE_ROOT"],
        description=(
            "Single filesystem root exposed to image browsing, ingestion, "
            "serving, and path remapping. When unset on a loopback bind, '/' "
            "is used. Network-accessible deployments must set an absolute path."
        ),
    )
    HOSTED_NIM_BASE_URL: str = DEFAULTS["HOSTED_NIM_BASE_URL"]
    LOCAL_NIM_TEACHER_PORT: int = DEFAULTS["LOCAL_NIM_TEACHER_PORT"]
    LOCAL_NIM_NVCLIP_PORT: int = DEFAULTS["LOCAL_NIM_NVCLIP_PORT"]

    @field_validator("IMAGE_ROOT", mode="before")
    @classmethod
    def _validate_image_root(cls, value: Any) -> str | None:
        if value is None or value == "":
            return None
        path = Path(str(value))
        if not path.is_absolute():
            raise ValueError("IMAGE_ROOT must be an absolute path")
        return str(path)

    # ── New-project defaults ─────────────────────────────────────────
    DEFAULT_TEACHER_MODEL: str = Field(
        default=DEFAULTS["DEFAULT_TEACHER_MODEL"],
        description=(
            "Default Teacher model_name seeded as teacher_model_config_id on "
            "new projects. Must match one of the models in SEEDED_MODEL_CATALOG."
        ),
    )
    DEFAULT_TEACHER_LOCAL_BASE_URL: str | None = Field(
        default=DEFAULTS["DEFAULT_TEACHER_LOCAL_BASE_URL"],
        description=(
            "When set, create_project probes this URL's /health/ready and, if "
            "healthy, seeds a self_hosted NimEndpoint and rebinds the "
            "DEFAULT_TEACHER_MODEL's model_config to it. Required when the "
            "default Teacher is a model the hosted catalog doesn't serve "
            "(e.g. nvidia/cosmos-reason2-8b). Leave unset for hosted-only "
            "deployments."
        ),
    )

    # ── Secrets (from env only, never in config.yaml) ────────────────
    NVIDIA_API_KEY: str | None = Field(
        default=None,
        description=(
            "NVIDIA API Catalog key. Required for hosted NIM inference "
            "(Teacher) and hosted embeddings."
        ),
    )
    NGC_API_KEY: str | None = Field(
        default=None,
        description=(
            "NGC API key for pulling NIM container images. Required when "
            "deploying NIM locally via Docker."
        ),
    )
    HF_TOKEN: str | None = Field(
        default=None,
        description=(
            "Hugging Face access token. Required for the "
            "self-service base-experiment provisioning flow "
            "when the supported Cosmos checkpoints are gated on Hugging "
            "Face — the Blueprint propagates this through to the "
            "subprocess driver as both HF_TOKEN and HUGGING_FACE_HUB_TOKEN."
        ),
    )
    TAO_API_KEY: str | None = Field(
        default=None,
        description="TAO FTMS bearer token (typically a JWT).",
    )
    TAO_API_BASE_URL: str | None = Field(
        default=None,
        description="TAO FTMS base URL, e.g. https://<tao-host>/api/v2",
    )
    TAO_ORG_NAME: str | None = Field(
        default=None,
        description="TAO organization name used in API paths.",
    )

    # ── UI secret-persistence gate ───────────────────────────────────
    ALLOW_UI_SECRET_PERSIST: bool = Field(
        default=True,
        description=(
            "When True (default), the NIM Configuration UI's [Save to .env] "
            "checkbox is available and ``POST /v1/secrets:set`` accepts "
            "``persist=true`` to write the key into "
            "~/.vlm_feedback_loop/.env. Set False for "
            "container/production deployments where .env is managed "
            "externally — runtime application via the override layer "
            "still works, but disk persistence from the UI is refused "
            "with HTTP 403."
        ),
    )

    # ── TAO workspace + S3 ───────────────────────────────────────────
    # ``deployment.db.tao_deployment_configs`` is the persistent source
    # for workspace identity and non-secret S3 details. The two secret fields
    # (TAO_WORKSPACE_S3_ACCESS_KEY / _SECRET_KEY)
    # remain in ``.env`` — the daemon resolves them by env-var name at
    # runtime; the DB stores only the reference label.
    TAO_WORKSPACE_S3_ACCESS_KEY: str | None = Field(
        default=None,
        description="S3 access key for the workspace bucket (secret). Lives in .env.",
    )
    TAO_WORKSPACE_S3_SECRET_KEY: str | None = Field(
        default=None,
        description="S3 secret key for the workspace bucket (secret). Lives in .env.",
    )


# ── Loader ───────────────────────────────────────────────────────────────────


def get_canonical_env_file_path() -> Path:
    """Return the canonical ``.env`` path used for persisted secret writes.

    Public accessor over the test-monkeypatched ``_DEFAULT_ENV_FILE``
    constant. Honors ``VLM_FEEDBACK_LOOP_ENV_FILE`` as an explicit override
    (matches the precedence chain in :func:`load_settings`); otherwise
    returns the canonical ``~/.vlm_feedback_loop/.env`` location.

    Returns the path unconditionally — caller is responsible for checking
    whether the file exists.
    """
    # Prefer the file this process actually loaded (captured at load_settings,
    # honoring --env-file) so reads and persisted writes always target the
    # same .env. Falls back to the env-var override / default only when called
    # before any load (e.g. some tests).
    if _active_env_file is not None:
        return _active_env_file
    override = os.environ.get("VLM_FEEDBACK_LOOP_ENV_FILE")
    if override:
        return Path(override)
    return _DEFAULT_ENV_FILE


def get_config_file_path() -> Path:
    """Return the canonical ``config.yaml`` path (may not exist).

    Public accessor over ``_CONFIG_FILE`` for callers that must decide
    whether a full :func:`load_settings` reload is possible — the loader
    fail-fasts (``SystemExit``) when the file is missing, which is correct
    at process startup but must never escape a request handler.
    """
    return _CONFIG_FILE


def _resolve_env_file_path(cli_env_file: str | None = None) -> Path | None:
    """Determine which .env file to load.

    When an explicit path is given (CLI arg or VLM_FEEDBACK_LOOP_ENV_FILE),
    return it unconditionally — even if the file does not exist — so the
    canonical .env is never used as a fallback.
    """
    if cli_env_file:
        return Path(cli_env_file)

    from_var = os.environ.get("VLM_FEEDBACK_LOOP_ENV_FILE")
    if from_var:
        return Path(from_var)

    # Fall back to canonical location only if it exists
    if _DEFAULT_ENV_FILE.exists():
        return _DEFAULT_ENV_FILE

    return None


def _is_complex_field(field_name: str) -> bool:
    """Return True if the Settings field expects a list or dict."""
    field_info = Settings.model_fields.get(field_name)
    if field_info is None:
        return False
    annotation = field_info.annotation
    # Unwrap Optional (X | None)
    origin = get_origin(annotation)
    if origin is type(int | None):  # types.UnionType
        args = get_args(annotation)
        annotation = next((a for a in args if a is not type(None)), annotation)
        origin = get_origin(annotation)
    return origin in (list, dict)


def _coerce_string_values(merged: dict[str, Any]) -> dict[str, Any]:
    """JSON-parse string values destined for list/dict fields.

    Values from YAML arrive as native Python types (lists, dicts) and need
    no coercion.  Values from .env files and process environment variables
    arrive as strings.  For fields typed as list or dict, attempt
    ``json.loads``.  If parsing fails, leave the raw string — Pydantic will
    reject it with a type error (correct fail-fast behaviour).
    """
    for key, value in merged.items():
        if isinstance(value, str) and _is_complex_field(key):
            # Leave the raw string on parse failure — Pydantic will reject it.
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                merged[key] = json.loads(value)
    return merged


def _format_validation_error(exc: ValidationError) -> str:
    """Produce a human-readable message from a Pydantic ValidationError."""
    lines = ["Configuration error:"]
    for err in exc.errors():
        loc = " -> ".join(str(p) for p in err["loc"])
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)


def load_settings(cli_env_file: str | None = None) -> Settings:
    """Load settings with five-level precedence.

    Raises ``SystemExit`` on missing config file or validation failure
    so the backend fails fast with a clear message.
    """
    # Level 4: config.yaml (fail fast if missing)
    config_file = _CONFIG_FILE
    if not config_file.exists():
        print(
            f"Configuration file not found: {config_file}\n"
            f"Run 'vlm-feedback-loop init' to create it.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(config_file) as f:
        yaml_loaded: Any = yaml.safe_load(f) or {}
    yaml_values: dict[str, Any] = (
        cast("dict[str, Any]", yaml_loaded) if isinstance(yaml_loaded, dict) else {}
    )

    merged: dict[str, Any] = dict(yaml_values)

    # Record the write target for persisted secrets (get_canonical_env_file_
    # path). Unlike the load resolution below, this is always a concrete path
    # (persist may CREATE the file), honoring --env-file / the env override.
    global _active_env_file
    if cli_env_file:
        _active_env_file = Path(cli_env_file)
    elif os.environ.get("VLM_FEEDBACK_LOOP_ENV_FILE"):
        _active_env_file = Path(os.environ["VLM_FEEDBACK_LOOP_ENV_FILE"])
    else:
        _active_env_file = _DEFAULT_ENV_FILE

    # Level 2/3: .env file. Filter to known Settings fields — the same filter
    # process env uses. A .env is often shared/reused and may carry unrelated
    # shell vars; merging them unfiltered would crash startup under
    # extra="forbid". The app's own config.yaml stays strict so typos fail fast.
    known_env_keys = set(Settings.model_fields)
    env_file_path = _resolve_env_file_path(cli_env_file)
    if env_file_path is not None and env_file_path.exists():
        dot_values = dotenv_values(env_file_path)
        for k, v in dot_values.items():
            if v is not None and k in known_env_keys:
                merged[k] = v

    # Level 1: process environment variables (only for known fields)
    for field_name in known_env_keys:
        if field_name in os.environ:
            merged[field_name] = os.environ[field_name]

    # Coerce string values for complex fields before Pydantic validation
    merged = _coerce_string_values(merged)

    # Level 5 (defaults) are baked into the Settings field definitions.
    # Pydantic fills missing keys from field defaults automatically.
    try:
        return Settings(**merged)
    except ValidationError as exc:
        print(_format_validation_error(exc), file=sys.stderr)
        sys.exit(1)


# ── Cached accessor ──────────────────────────────────────────────────────────
#
# ``@lru_cache``
# wraps the no-arg default load. We also expose
# ``init_settings(cli_env_file)`` as an explicit-override escape hatch used
# by the CLI entry point (``main.py``) to honour ``--env-file`` before
# routers and services start consuming settings.


@lru_cache(maxsize=1)
def _cached_default_settings() -> Settings:
    """LRU-cached default load (no CLI env-file override)."""
    return load_settings()


_override_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings singleton, loading on first call.

    If :func:`init_settings` was previously called with an explicit
    env-file path, that instance is returned. Otherwise the default
    LRU-cached load is used (cheap on repeat calls).
    """
    if _override_settings is not None:
        return _override_settings
    return _cached_default_settings()


def init_settings(cli_env_file: str | None = None) -> Settings:
    """Load settings with an explicit env-file path and seed the cache.

    Subsequent :func:`get_settings` calls return this instance until
    :func:`reset_settings` is called.
    """
    global _override_settings
    _override_settings = load_settings(cli_env_file=cli_env_file)
    # Invalidate the default cache so a later reset + get_settings path
    # re-reads from disk rather than returning a stale default.
    _cached_default_settings.cache_clear()
    return _override_settings


def reset_settings() -> None:
    """Clear cached settings. For testing only."""
    global _override_settings
    _override_settings = None
    _cached_default_settings.cache_clear()

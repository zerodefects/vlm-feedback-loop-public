# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local NIM deployment infrastructure.

Docker container orchestration for deploying NIM models locally on the
backend host.  Covers: preflight checks (6 sequential), container lifecycle,
health polling, port allocation, GPU placement, auto-endpoint-registration,
and restart recovery.

All Docker commands use ``asyncio.create_subprocess_exec`` via the shared
``run_subprocess`` helper from ``environment.py`` — no shell injection risk.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db
from vlm_feedback_loop.db.models.local_nim_deployment import LocalNimDeployment
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.model_catalog_constants import (
    EMBEDDING_NIM_IMAGE,
    NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
)
from vlm_feedback_loop.services.background import background_manager
from vlm_feedback_loop.services.clip_embedding_service import resweep_embedding_tasks
from vlm_feedback_loop.services.environment import (
    check_docker_available,
    check_nvidia_toolkit,
    gpu_memory_meets_floor,
    probe_gpu_inventory,
    run_subprocess,
)
from vlm_feedback_loop.services.http_client import resilient_request
from vlm_feedback_loop.services.image_cap_resolver import (
    resolve_max_images_per_request,
)
from vlm_feedback_loop.services.nim_client import NIM_DEFAULT_HEADERS, create_embeddings
from vlm_feedback_loop.services.nim_endpoint_service import (
    NimEndpointConfigurationError,
    create_nim_endpoint,
    normalize_self_hosted_base_url,
)
from vlm_feedback_loop.services.project_service import (
    get_project_engine,
    projects_root,
)
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret

logger = logging.getLogger("vlm_feedback_loop.services.local_nim_service")


def release_version_from_image(image: str | None) -> str | None:
    """Return a registry image tag without mistaking a port for a tag."""
    if not image or "@" in image:
        return None
    final_component = image.rsplit("/", 1)[-1]
    if ":" not in final_component:
        return None
    return final_component.rsplit(":", 1)[-1] or None


# Upper bound for the preflight ``list-model-profiles`` probe (check 7). On
# current VLM NIM images the command performs a full engine init rather than a
# cheap manifest lookup, so a timeout here is treated as INCONCLUSIVE (the serve
# health poll is authoritative), never a deploy failure. Kept modest so a slow
# image fails fast to "inconclusive" instead of blocking the deploy for minutes.
PROFILE_PROBE_TIMEOUT_S = 120.0

# Host-side NIM cache root every deploy mounts at ``/opt/nim/.cache``
# (``_build_docker_run_command``). Model weights land under
# ``<root>/ngc/hub/models--nim--nvidia--<slug>/snapshots/<profile>/``.
NIM_CACHE_ROOT = "~/.cache/nim"

# NeMo Retriever 2.0 downloads its model to ``NIM_MODEL_PATH``.  The image
# default (``/model/embed``) lives in the disposable container layer, so every
# restart otherwise downloads the same ~3.2 GiB artifact again.  Keep it below
# the existing host-backed NIM cache mount instead.
EMBEDDING_MODEL_CACHE_PATH = "/opt/nim/.cache/models/llama-nemotron-embed-vl-1b-v2"


def _uses_declared_container_user(nim_container_image: str) -> bool:
    """Whether the image must keep its named in-container user.

    The Omni 1.7.0 image resolves the current user through ``getpwuid`` during
    startup. Replacing its declared ``nvs`` user with an arbitrary host UID
    therefore crashes before model loading when that UID has no image-local
    passwd entry.
    """

    return nim_container_image == NEMOTRON_3_NANO_OMNI_NIM_IMAGE


def _prepare_declared_user_cache(nim_container_image: str) -> None:
    """Make shared-cache parent directories writable to Omni's ``nvs`` user.

    Only the cache parents and NGC's shared download scratch directory are
    opened; existing model artifacts retain their ownership and permissions.
    This lets the declared user create its own model-specific subtree without
    recursively changing a developer's NIM cache.
    """

    if not _uses_declared_container_user(nim_container_image):
        return
    root = Path(NIM_CACHE_ROOT).expanduser()
    hub = root / "ngc" / "hub"
    vllm = root / "vllm"
    for path in (
        root,
        root / "ngc",
        hub,
        hub / "tmp",
        vllm,
        vllm / "modelinfos",
        vllm / "torch_compile_cache",
        root / "flashinfer",
        root / "tvm-ffi",
    ):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(path.stat().st_mode | 0o777)


# Weight-file extensions that prove a real model is cached (not a config-only
# stub). The silent-fallback footgun: a 52K cache with ZERO of these —
# only a ``config.json`` symlink — makes the NIM log
# ``Detected 0 compatible profile(s)`` and serve the cached SUPER weights
# under the requested NANO name. Presence of a non-trivial weight file for the
# requested model is the offline tell that the right weights are actually here.
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".engine")

# A weight file smaller than this is treated as a stub / index sidecar, not real
# weights (e.g. ``model.safetensors.index.json`` is JSON, and an empty/truncated
# shard from an interrupted pull is not a usable weight). Real VLM shards are
# multi-GB; 1 MiB is comfortably below any real shard yet above every metadata
# sidecar.
WEIGHT_FILE_MIN_BYTES = 1_048_576

# Env keys the docker-run builder itself owns. ``extra_container_env`` may not
# shadow them: the builder's values are contract-bearing (secret handling,
# image-cap agreement with the prompt pruner, served-model identity
# verification, shared-image size/profile selection).
RESERVED_CONTAINER_ENV_KEYS = frozenset(
    {
        "NGC_API_KEY",
        "NIM_MAX_IMAGES_PER_PROMPT",
        "NIM_MAX_VIDEOS_PER_PROMPT",
        "NIM_MODEL_SIZE",
        "NIM_MODEL_PROFILE",
        "NIM_SERVED_MODEL_NAME",
        "NIM_MODEL_NAME",
        "NIM_MODEL_PATH",
        "NIM_PRECISION",
        "NIM_ENABLE_KV_CACHE_REUSE",
    }
)

_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class PreflightCheckResult:
    """A single preflight check result."""

    check_name: str
    passed: bool
    diagnostic: str


@dataclass
class PreflightResult:
    """Aggregated preflight result with all checks and the docker run command."""

    all_passed: bool
    checks: list[PreflightCheckResult] = field(
        default_factory=list[PreflightCheckResult]
    )
    docker_run_command: str | None = None


@dataclass(frozen=True)
class ActiveNimResident:
    """Safe host-wide summary of one Blueprint-managed active NIM.

    Deployment rows live in per-project databases, but GPU occupancy is
    host-wide.  This projection gives placement, FTUE, and conflict UX one
    authoritative view without leaking credentials or Docker environment
    values.
    """

    project_id: str
    project_name: str
    deployment_id: str
    model_config_id: str
    role: str
    model_name: str | None
    nim_container_image: str
    gpu_assignment: str
    endpoint_url: str
    host_port: int
    status: str
    nim_model_size: str | None
    nim_model_profile: str | None
    extra_container_env: tuple[tuple[str, str], ...]

    def public_summary(self) -> dict[str, Any]:
        """Return the non-secret API shape used by setup and 409 responses."""

        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "local_nim_deployment_id": self.deployment_id,
            "role": self.role,
            "model_name": self.model_name,
            "nim_container_image": self.nim_container_image,
            "gpu_assignment": self.gpu_assignment,
            "status": self.status,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def extract_device_index(gpu_assignment: str) -> str:
    """Extract numeric device index from gpu_assignment like 'device=0'."""
    if "=" in gpu_assignment:
        return gpu_assignment.split("=", 1)[1]
    return gpu_assignment


def build_container_name(
    role: str, project_id: str, student_model_id: str | None = None
) -> str:
    """Deterministic container name for restart recovery.

    Teacher / embedding deployments are project-scoped (one active per
    project per role). Student deployments include the StudentModel id
    suffix so per-variant retries don't collide and the recovery loop
    can map a container back to its Student row deterministically.
    """
    if role == "student" and student_model_id:
        return f"vlm-student-{project_id[:8]}-{student_model_id[:8]}"
    return f"vlm-{role}-{project_id[:8]}"


def _build_endpoint_url(host_port: int) -> str:
    """Build the local endpoint URL from resolved port."""
    return f"http://localhost:{host_port}/v1"


def _build_docker_run_command(
    nim_container_image: str,
    container_name: str,
    gpu_assignment: str,
    host_port: int,
    role: str,
    *,
    checkpoint_mount: str | None = None,
    nim_model_name_path: str | None = None,
    nim_served_model_name: str | None = None,
    max_images_per_request: int | None = None,
    nim_model_size: str | None = None,
    nim_model_profile: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Build the docker run argument list.

    Most container roles use ``-u $(id -u)`` so the container runs as the
    host user's UID. This matches NVIDIA's published VLM NIM ``docker run``
    examples and keeps the cache mount (``~/.cache/nim``) writable:
    without the flag, the container falls back to a hardcoded internal
    UID (1000:1000 on the current NeMo Retriever VL 1B v2 image) which
    can't write to subdirectories created by the host user's UID, so
    the first model-weight download fails with ``Permission denied
    (os error 13)`` and the container exits 0 mid-startup.

    Omni 1.7.0 is the supported exception: its startup resolves the current
    user through ``getpwuid`` and crashes when an arbitrary host UID is not in
    the image's passwd database. It keeps its declared ``nvs`` user and the
    orchestrator opens only the shared cache parent directories for it.

    Student-specific extras (only applied when ``role == "student"`` AND
    the corresponding kwarg is provided): mounts the merged HF
    checkpoint directory at the in-container path passed via
    ``NIM_MODEL_NAME``, and tells NIM what served-model name clients use
    via ``NIM_SERVED_MODEL_NAME``. The Student NIM container picks up
    the checkpoint via these env vars.
    """
    device_index = extract_device_index(gpu_assignment)
    resolved_cache = f"{Path(NIM_CACHE_ROOT).expanduser()}:/opt/nim/.cache"

    args = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--runtime=nvidia",
        "--gpus",
        f'"device={device_index}"',
        "--shm-size=32GB",
        # ``--ulimit memlock=-1`` and ``--ulimit stack=67108864`` are
        # NVIDIA's documented prerequisites for VLM NIM containers (the
        # container itself emits a startup warning citing them when they
        # are absent). At the 60-image RPS workload exercised so far they
        # don't manifest as failures, but heavier workloads (large ICL
        # batches, multi-stream prefill) can hit pinned-memory or
        # thread-stack limits silently — symptom is a CUDA OOM or a
        # mid-inference container crash with no useful log line. They
        # are set defensively. ``--shm-size=32GB`` is kept (not switched
        # to ``--ipc=host``) so the container retains its own IPC
        # namespace.
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "-p",
        # System-managed NIMs are private implementation services consumed
        # through the loopback endpoint persisted below. Publishing on every
        # host interface would expose an unauthenticated model server even
        # when the Blueprint itself keeps its safe loopback bind default.
        f"127.0.0.1:{host_port}:8000",
        "-e",
        "NGC_API_KEY",
        "-v",
        resolved_cache,
    ]

    # Most images run as the host UID so the cache stays writable. Omni must
    # retain its declared ``nvs`` user; see the docstring and cache preparation
    # helper above.
    if not _uses_declared_container_user(nim_container_image):
        args.extend(["-u", str(os.getuid())])

    if role == "embedding" and nim_container_image == EMBEDDING_NIM_IMAGE:
        # NeMo Retriever 2.0.0's automatic SM120 selection currently enters a
        # cuDNN-plan path whose cache directory is never created, so the
        # server exits before readiness.  FP16 selects the image's working
        # ``experimental-vlm-avo-fp16`` Blackwell pipeline (also supported on
        # the older architectures in the NIM support matrix).  Persist the
        # downloaded weights under the host-backed cache at the same time.
        #
        # Live validation on RTX PRO 6000 Blackwell (2026-07-24): ready at
        # ~6.3 GiB and returned one finite 2,048-dimensional image embedding.
        args.extend(["-e", "NIM_PRECISION=fp16"])
        args.extend(["-e", f"NIM_MODEL_PATH={EMBEDDING_MODEL_CACHE_PATH}"])

    # Pin the per-prompt image cap so the NIM accepts exactly as many
    # images as the backend will send. VLM NIM profiles bake in a
    # ``NIM_MAX_IMAGES_PER_PROMPT`` default that is version-specific and
    # silently low on the cosmos ``:1.7.0`` family (5, down from ~999 on
    # ``:1.6.0`` — see ``services/image_cap_resolver.py``). Setting it
    # here from the resolved ``max_images_per_request`` makes the NIM and
    # the backend's ICL image-budget pruner (``prompt_service``) agree, so
    # an app-driven deploy can't 422 on a multi-image ICL request whose
    # size the backend itself considered legal.
    if max_images_per_request is not None:
        args.extend(["-e", f"NIM_MAX_IMAGES_PER_PROMPT={int(max_images_per_request)}"])

    # Image-only local NIMs: pin video capacity to 0. The cosmos image bakes in a
    # default of 1 (functionally harmless — the backend never emits ``video_url``
    # content parts, so zero videos ever reach the NIM), but reserving video
    # capacity can shift memory/serving behaviour, so set it explicitly.
    args.extend(["-e", "NIM_MAX_VIDEOS_PER_PROMPT=0"])

    # Multi-size shared-image selectors (cosmos3-reasoner nano/super).
    # Both Cosmos 3 sizes ship in ONE image (``cosmos3-reasoner:1.7.0``)
    # selected at deploy time by ``NIM_MODEL_SIZE``. Setting these env vars is
    # threaded ONLY for models whose ``local_deploy_metadata.nim_model_size``
    # is populated (cosmos3); single-image teachers (cosmos-reason2-{2b,8b})
    # leave them ``None`` so their docker command is byte-for-byte unchanged.
    #
    #   * ``NIM_MODEL_SIZE`` — picks nano vs super inside the shared image.
    #   * ``NIM_SERVED_MODEL_NAME`` — makes ``/v1/models`` report the
    #     size-specific catalog name (``nvidia/cosmos3-super-reasoner``)
    #     instead of the empty/cosmetic default; this is also the identity
    #     ``verify_served_model`` keys on.
    #   * ``NIM_MODEL_PROFILE`` — the footgun fix: pin the exact size-specific
    #     profile id (when known) so the NIM never relies on the fragile
    #     size-filter auto-selector that silently falls back to the resident
    #     SUPER weights when the requested profile/weights are absent
    #     (the silent-fallback footgun). ``None`` for sizes whose
    #     profile id isn't pinned (e.g. super, which auto-selects correctly off
    #     its cached default weights).
    if nim_model_size:
        args.extend(["-e", f"NIM_MODEL_SIZE={nim_model_size}"])
    # A base profile selects the image's bundled weights and must never be
    # combined with a custom Student checkpoint. Keep this guard here as a
    # final serialization invariant even though deploy/preflight resolution
    # already clears the incompatible profile.
    custom_student_checkpoint = bool(
        role == "student" and checkpoint_mount and nim_model_name_path
    )
    if nim_model_profile and not custom_student_checkpoint:
        args.extend(["-e", f"NIM_MODEL_PROFILE={nim_model_profile}"])
    # ``nim_served_model_name`` is the requested catalog model name for
    # teacher (shared-image) deploys and the custom ``student-<id>`` name for
    # student checkpoint mounts. The student branch below also emits it
    # alongside the mount; emitting it here covers the teacher path.
    if role == "teacher" and nim_served_model_name:
        args.extend(["-e", f"NIM_SERVED_MODEL_NAME={nim_served_model_name}"])

    if role == "student":
        # Canonical serving comparison policy. The benchmark replays the same
        # frozen Test Pool at multiple concurrencies; prefix-cache hits would
        # otherwise make later cells incomparable to the first.
        args.extend(["-e", "NIM_ENABLE_KV_CACHE_REUSE=0"])
        if checkpoint_mount and nim_model_name_path:
            # Mount host checkpoint dir at the in-container path.
            args.extend(["-v", f"{checkpoint_mount}:{nim_model_name_path}:ro"])
            args.extend(["-e", f"NIM_MODEL_NAME={nim_model_name_path}"])
        if nim_served_model_name:
            args.extend(["-e", f"NIM_SERVED_MODEL_NAME={nim_served_model_name}"])

    # Operator-supplied remediation env (validated by
    # _resolve_extra_container_env). Sorted for a deterministic command;
    # when absent the command stays byte-for-byte unchanged.
    if extra_env:
        for key in sorted(extra_env):
            args.extend(["-e", f"{key}={extra_env[key]}"])

    args.append(nim_container_image)
    return args


def docker_run_command_display(
    nim_container_image: str,
    container_name: str,
    gpu_assignment: str,
    host_port: int,
    role: str,
    *,
    checkpoint_mount: str | None = None,
    nim_model_name_path: str | None = None,
    nim_served_model_name: str | None = None,
    max_images_per_request: int | None = None,
    nim_model_size: str | None = None,
    nim_model_profile: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Build a copy-paste-safe docker run command.

    Rendered FROM :func:`_build_docker_run_command`'s argument list — the
    display can never drift from what the backend actually runs (they
    were previously hand-synced twins that drifted). Docker's name-only
    ``-e NGC_API_KEY`` form reads the exported value without putting it in
    argv. The numeric UID is rendered as ``$(id -u)`` for portability.
    """
    args = _build_docker_run_command(
        nim_container_image,
        container_name,
        gpu_assignment,
        host_port,
        role,
        checkpoint_mount=checkpoint_mount,
        nim_model_name_path=nim_model_name_path,
        nim_served_model_name=nim_served_model_name,
        max_images_per_request=max_images_per_request,
        nim_model_size=nim_model_size,
        nim_model_profile=nim_model_profile,
        extra_env=extra_env,
    )

    # Render one flag group per line: a token starting with "-" opens a
    # group, its value tokens follow on the same line; the trailing image
    # gets its own line.
    lines = ["docker run -d"]
    group: list[str] = []
    for token in args[3:-1]:  # skip "docker run -d"; image handled last
        if token.startswith("-") and group:
            lines.append("  " + " ".join(group))
            group = [token]
        else:
            group.append(token)
    if group:
        lines.append("  " + " ".join(group))
    lines.append(f"  {args[-1]}")

    uid_line = f"  -u {os.getuid()}"
    return " \\\n".join("  -u $(id -u)" if line == uid_line else line for line in lines)


def _resolve_deployment_image_cap(engine: Any, model_config_id: str) -> int | None:
    """Effective per-prompt image cap for a deployment's ModelConfig.

    Mirrors :func:`image_cap_resolver.resolve_max_images_per_request`
    (endpoint override > model_config). Returns ``None`` when the
    ModelConfig can't be loaded (older rows / test fixtures with a
    placeholder id) so the caller simply omits ``NIM_MAX_IMAGES_PER_PROMPT``
    and the NIM keeps its profile default.
    """
    if engine is None or not model_config_id:
        return None
    with Session(engine) as session:
        mc = session.get(ModelConfig, model_config_id)
        if mc is None:
            return None
        ep = None
        endpoint_id = getattr(mc, "endpoint_id", None)
        if endpoint_id:
            ep = session.get(NimEndpoint, endpoint_id)
        return resolve_max_images_per_request(model_config=mc, nim_endpoint=ep)


def _resolve_deployment_compute_capability_floor(
    engine: Any, model_config_id: str
) -> float | None:
    """Return a model's optional local-NIM architecture floor."""

    if engine is None or not model_config_id:
        return None
    with Session(engine) as session:
        mc = session.get(ModelConfig, model_config_id)
        if mc is None:
            return None
        raw = (mc.local_deploy_metadata or {}).get("nim_compute_capability_minimum")
        return float(raw) if raw is not None else None


def _resolve_shared_image_deploy_env(
    engine: Any, model_config_id: str
) -> tuple[str | None, str | None, str | None]:
    """Resolve shared-image deploy selectors for a base ModelConfig.

    Returns ``(nim_model_size, nim_model_profile, nim_served_model_name)`` for
    a model that ships in a multi-size shared NIM image (cosmos3-reasoner
    nano/super). ``nim_model_size`` comes from
    ``local_deploy_metadata.nim_model_size``; when it is absent (every
    single-image model, e.g. cosmos-reason2-{2b,8b}) ALL THREE values are
    ``None`` so the docker command for those models is byte-for-byte unchanged.

    For a shared-image model:

    * ``nim_model_size`` selects nano vs super inside the image.
    * ``nim_served_model_name`` is the catalog ``ModelConfig.model_name``
      (e.g. ``nvidia/cosmos3-super-reasoner``) so ``/v1/models`` reports the
      requested size-specific name rather than the empty/cosmetic default —
      and so ``verify_served_model`` has a stable identity to key on.
    * ``nim_model_profile`` is the size-specific profile id, pinned via
      ``local_deploy_metadata.nim_model_profile`` to bypass the fragile
      size-filter auto-selector that silently falls back to the resident SUPER
      weights (the silent-fallback footgun). ``None`` when no profile is pinned for the size
      (e.g. super, which auto-selects correctly off its cached default).
    """
    if engine is None or not model_config_id:
        return (None, None, None)
    with Session(engine) as session:
        mc = session.get(ModelConfig, model_config_id)
        if mc is None:
            return (None, None, None)
        metadata = mc.local_deploy_metadata or {}
        model_size = metadata.get("nim_model_size")
        profile = metadata.get("nim_model_profile") or None
        if not model_size:
            # Not a shared-image (size-selected) model: no size/name
            # selectors — but an operator-pinned profile must still be
            # honored. Single-image NIMs auto-select per hardware, and the
            # auto-choice can be unusable (observed 2026-07-14:
            # cosmos-reason2-2b:1.6.0 auto-selected an h100-nvl fp8 profile
            # whose engine dies with CUDA illegal-memory-access under
            # concurrent structured-gen multimodal load; the bf16 pin is the
            # remediation). With no pin set the docker command remains
            # byte-for-byte unchanged for single-image teachers.
            return (None, str(profile) if profile else None, None)
        served_name = mc.model_name or None
        return (str(model_size), str(profile) if profile else None, served_name)


def _resolve_runtime_deploy_env(
    engine: Any,
    model_config_id: str,
    *,
    role: str,
    custom_checkpoint: bool,
) -> tuple[str | None, str | None, str | None]:
    """Resolve selectors for the workload the container will actually serve.

    A base-model profile selects the weights bundled with the NIM image. That
    pin is required for a shared-image Teacher, but it is incompatible with a
    Student's read-only custom checkpoint mount: the profile selector wins
    before ``NIM_MODEL_NAME`` can generate/select a checkpoint-compatible
    runtime profile. Custom Students therefore retain the base model's size
    selector while allowing NIM to auto-select against the mounted weights.
    """

    model_size, model_profile, served_name = _resolve_shared_image_deploy_env(
        engine, model_config_id
    )
    if role == "student" and custom_checkpoint:
        model_profile = None
    return model_size, model_profile, served_name


def resolve_shared_image_preflight_env(
    project_id: str,
    model_config_id: str,
    workspace_root: str,
) -> tuple[str | None, str | None, str | None]:
    """Resolve the model selectors a Teacher or Student probe must receive."""
    engine = get_project_engine(project_id, workspace_root)
    return _resolve_shared_image_deploy_env(engine, model_config_id)


def resolve_student_checkpoint_preflight_env(
    project_id: str,
    model_config_id: str,
    workspace_root: str,
) -> tuple[str | None, None, str | None]:
    """Resolve size/name selectors for a custom Student checkpoint probe."""

    engine = get_project_engine(project_id, workspace_root)
    model_size, _model_profile, served_name = _resolve_runtime_deploy_env(
        engine,
        model_config_id,
        role="student",
        custom_checkpoint=True,
    )
    return model_size, None, served_name


def normalize_extra_container_env(raw: object) -> dict[str, str]:
    """Return only metadata entries the Docker builder will pass to NIM.

    Keys must be ``UPPER_SNAKE_CASE`` and may not shadow builder-owned
    environment. Values are rendered verbatim into the launch command and
    operator-visible handoffs, so this metadata must never contain secrets.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "extra_container_env ignored: expected an object, got %s",
            type(raw).__name__,
        )
        return {}
    resolved: dict[str, str] = {}
    for key, value in cast("dict[Any, Any]", raw).items():
        if not isinstance(key, str) or not _ENV_KEY_RE.match(key):
            logger.warning("extra_container_env: skipping invalid env key %r", key)
            continue
        if key in RESERVED_CONTAINER_ENV_KEYS:
            logger.warning(
                "extra_container_env: skipping reserved builder-owned env key %s", key
            )
            continue
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            logger.warning(
                "extra_container_env: skipping env key %s (value must be a "
                "string or number)",
                key,
            )
            continue
        resolved[key] = str(value)
    return resolved


def _resolve_extra_container_env(engine: Any, model_config_id: str) -> dict[str, str]:
    """Load and validate a model config's operator-supplied NIM environment."""
    if engine is None or not model_config_id:
        return {}
    with Session(engine) as session:
        mc = session.get(ModelConfig, model_config_id)
        if mc is None:
            return {}
        raw: Any = (mc.local_deploy_metadata or {}).get("extra_container_env")
    return normalize_extra_container_env(raw)


def resolve_extra_container_env(
    project_id: str,
    model_config_id: str,
    workspace_root: str,
) -> dict[str, str]:
    """Return validated non-secret deploy env for handoffs and bundles."""
    engine = get_project_engine(project_id, workspace_root)
    return _resolve_extra_container_env(engine, model_config_id)


# ── Served-model verification (anti-silent-fallback) ───────────────────────────
#
# A NIM that answers ``/v1/health/ready`` with 200 and reports a model name on
# ``/v1/models`` is NOT proof the requested weights are loaded. When the
# ``cosmos3-reasoner:1.7.0`` image is deployed with a model size whose
# profile/weights were never fetched (an NGC filemap fetch timeout leaves the
# cache a 52K config-only stub with ZERO weight files), the NIM logs
# ``Detected 0 compatible profile(s)``, SILENTLY falls back to the cached SUPER
# weights, and keeps reporting ``served_model_name=...nano...`` only at the
# /v1/models level — while ``/v1/metadata`` names the SUPER model that is
# actually loaded. Port + the cosmetic /v1/models name pass; the operator is
# unknowingly served the wrong (and far larger) model.
#
# This image hosts BOTH Cosmos 3 sizes (nano + super) under ONE container,
# selected at deploy time — so the requested model's expected served slug is
# the size-specific name from ``ModelConfig.model_name``
# (``nvidia/cosmos3-nano-reasoner`` → ``cosmos3-nano-reasoner``), NOT the shared
# image slug (``cosmos3-reasoner``). Verification keys on that expected served
# slug so a correct super deploy passes and a nano→super fallback fails.
# (Keying on the IMAGE slug — ``cosmos3-reasoner`` for the shared image — can
# never match the size-specific ``/v1/metadata`` slug, so it would false-FAIL
# a correct super deploy.)
#
# Two independent signals harden the healthy path so this can never recur:
#
#   1. LIVE ``/v1/metadata`` LOADED-MODEL SLUG (authoritative when reachable) —
#      the running NIM's ``modelInfo[].modelUrl`` (e.g.
#      ``ngc://nim/nvidia/cosmos3-super-reasoner:...``) names the model that is
#      ACTUALLY loaded, independent of the cosmetic ``/v1/models`` id. A
#      mismatch against the requested served slug is the direct silent-fallback
#      signature (requested nano, serving super).
#
#   2. CACHE WEIGHT-FILE PRESENCE (offline root-cause check) — the requested
#      model's NIM cache dir must contain at least one non-trivial weight file
#      (``*.safetensors`` / ``*.bin`` / …). This directly detects the "0 weight
#      files for the requested size" failure mode regardless of what the running
#      container later decides to serve.
#
# A mismatch on either signal fails the deployment with a specific, operator-
# actionable reason instead of marking it healthy. Inconclusive signals (no
# expected slug parseable, ``/v1/metadata`` unreachable AND weights present)
# never fail a working deploy — a signal must actively contradict the request.


@dataclass
class ServedModelVerification:
    """Outcome of verifying that a local NIM serves the requested model.

    ``ok`` is the only field a caller must branch on; the rest is evidence
    that gets persisted to ``LocalNimDeployment.status_reason`` (on failure)
    and logged (always) so the silent-fallback footgun leaves a paper trail.
    """

    ok: bool
    reason: str
    expected_slug: str | None = None
    served_slug: str | None = None
    weight_files_found: int = 0
    selected_profile_id: str | None = None


def _model_name_slug(model_name: str) -> str | None:
    """Extract the served-model slug from a catalog ``model_name``.

    ``nvidia/cosmos3-nano-reasoner`` → ``cosmos3-nano-reasoner``. This is the
    name the NIM reports for the requested model on ``/v1/models`` and
    ``/v1/metadata`` (size-specific even when several sizes share one image),
    so it is the correct identity to verify against — NOT the shared image
    slug (``cosmos3-reasoner``). Returns ``None`` when the name is empty so the
    caller treats verification as inconclusive.
    """
    if not model_name:
        return None
    last = model_name.rsplit("/", 1)[-1].strip()
    return last or None


def _ngc_model_slug(model_url: str) -> str | None:
    """Extract the model slug from an NGC model URL or short name.

    ``ngc://nim/nvidia/cosmos-reason2-2b:1208-fp8-dynamic-kv8`` →
    ``cosmos-reason2-2b``. Also accepts the bare ``shortName``
    (``cosmos-reason2-2b:1208-...``).
    """
    if not model_url:
        return None
    ref = model_url.split("://", 1)[-1]  # drop scheme if present
    ref = ref.split(":", 1)[0]  # drop the ``:profile-tag``
    last = ref.rsplit("/", 1)[-1].strip()
    return last or None


def _cache_hub_dir(model_slug: str, cache_root: str = NIM_CACHE_ROOT) -> Path:
    """Resolve the NIM cache hub directory for a model slug.

    Mirrors the HF-hub layout NIM writes under the mounted cache:
    ``<root>/ngc/hub/models--nim--nvidia--<slug>``.
    """
    expanded = Path(cache_root).expanduser()
    return expanded / "ngc" / "hub" / f"models--nim--nvidia--{model_slug}"


def count_cached_weight_files(model_slug: str, cache_root: str = NIM_CACHE_ROOT) -> int:
    """Count non-trivial weight files cached for *model_slug*.

    Walks the model's hub dir for files whose suffix is in
    :data:`WEIGHT_FILE_SUFFIXES` and whose resolved size is at least
    :data:`WEIGHT_FILE_MIN_BYTES`. Symlinks are resolved (the HF-hub
    layout stores blobs under ``blobs/`` and symlinks them into
    ``snapshots/<profile>/``) so a real shard counts and a dangling /
    truncated link does not. Returns 0 when the dir is absent — which is
    exactly the silent-fallback footgun's signature.
    """
    hub_dir = _cache_hub_dir(model_slug, cache_root)
    if not hub_dir.exists():
        return 0
    count = 0
    seen: set[str] = set()
    for path in hub_dir.rglob("*"):
        if path.suffix.lower() not in WEIGHT_FILE_SUFFIXES:
            continue
        try:
            resolved = path.resolve()
            # Dedupe: snapshots/<profile>/x.safetensors symlinks to the same
            # blob across profiles must not be triple-counted.
            key = str(resolved)
            if key in seen:
                continue
            if not resolved.is_file():
                continue
            if resolved.stat().st_size < WEIGHT_FILE_MIN_BYTES:
                continue
        except OSError:
            continue
        seen.add(key)
        count += 1
    return count


async def _probe_loaded_model(endpoint_url: str) -> tuple[str | None, str | None]:
    """Probe the running NIM's ``/v1/metadata`` for the LOADED model.

    Returns ``(loaded_model_slug, selected_profile_id)``. Both are
    ``None`` when the endpoint doesn't expose ``/v1/metadata`` or the
    response can't be parsed — verification then treats the live signal
    as unavailable (the cache check still applies).

    ``endpoint_url`` already carries the ``/v1`` prefix (e.g.
    ``http://localhost:8000/v1``), matching the health-poll convention.
    """
    metadata_url = f"{endpoint_url.rstrip('/')}/metadata"
    result = await resilient_request(
        "GET",
        metadata_url,
        deadline_s=10.0,
        max_retries=1,
    )
    raw_body: Any = result.body
    if result.status_code != 200 or not isinstance(raw_body, dict):
        return (None, None)
    body = cast("dict[str, Any]", raw_body)
    model_info: Any = body.get("modelInfo")
    selected_profile: Any = body.get("selectedModelProfileId")
    loaded_slug: str | None = None
    if isinstance(model_info, list) and model_info:
        entries = cast("list[Any]", model_info)
        first = entries[0]
        if isinstance(first, dict):
            entry = cast("dict[str, Any]", first)
            url_value: Any = entry.get("modelUrl") or entry.get("shortName") or ""
            loaded_slug = _ngc_model_slug(str(url_value))
    return (
        loaded_slug,
        str(selected_profile) if selected_profile is not None else None,
    )


async def verify_served_model(
    *,
    expected_model_name: str,
    endpoint_url: str,
    cache_root: str = NIM_CACHE_ROOT,
) -> ServedModelVerification:
    """Verify the local NIM is genuinely serving the requested model.

    *expected_model_name* is the catalog ``ModelConfig.model_name`` of the
    model the operator asked to deploy (e.g. ``nvidia/cosmos3-nano-reasoner``).
    Its size-specific slug is what the NIM reports for the requested model even
    when several sizes share one container image, so it is the correct identity
    to verify — NOT the shared image slug (``cosmos3-reasoner``), which can
    never match the size-specific ``/v1/metadata`` slug and so would
    false-FAIL a correct super deploy.

    Combines the live ``/v1/metadata`` loaded-model slug with the offline cache
    weight-file check. Policy:

    * If the live ``/v1/metadata`` slug is available AND differs from the
      expected slug → FAIL. This is the direct
      silent-fallback signature (requested nano, serving super).
    * Else if NO weight file is cached for the expected model → FAIL.
      The requested weights aren't present, so a
      "healthy" container is necessarily serving something else (or will
      crash) — the root cause of the fallback.
    * Else → PASS. Either the live slug matched, or weights are present and the
      live probe was unavailable / inconclusive.

    The function never fails on inconclusiveness alone (unparseable model name,
    missing ``/v1/metadata``): a signal must actively contradict the request.
    This mirrors the preflight policy — never reject a working deploy on a
    probe that simply couldn't answer.
    """
    expected_slug = _model_name_slug(expected_model_name)
    loaded_slug, selected_profile = await _probe_loaded_model(endpoint_url)

    # Scan the NIM cache once, off the event loop: count_cached_weight_files
    # does a recursive rglob + stat() over potentially thousands of files, and
    # verify_served_model is awaited from _on_healthy on the main loop.
    weight_files = (
        await asyncio.to_thread(count_cached_weight_files, expected_slug, cache_root)
        if expected_slug
        else 0
    )

    # Signal 1: live loaded-model slug contradicts the request.
    if expected_slug and loaded_slug and loaded_slug != expected_slug:
        return ServedModelVerification(
            ok=False,
            reason=(
                f"Served model mismatch: requested '{expected_slug}' but the "
                f"NIM reports loaded model '{loaded_slug}' via /v1/metadata. "
                "The container silently fell back to a different cached model "
                "(the requested profile/weights are unavailable — likely an "
                "NGC fetch failure or an uncached model size). Refusing to "
                "mark healthy so the wrong model is never served under the "
                "requested name."
            ),
            expected_slug=expected_slug,
            served_slug=loaded_slug,
            weight_files_found=weight_files,
            selected_profile_id=selected_profile,
        )

    # Signal 2: requested model's weights are absent from the cache.
    if expected_slug and weight_files == 0:
        return ServedModelVerification(
            ok=False,
            reason=(
                f"No model weights cached for requested model '{expected_slug}' "
                f"(0 weight files under {_cache_hub_dir(expected_slug, cache_root)}). "
                "The NIM cache holds only config stubs — a 'healthy' container "
                "is serving fallback weights, not the requested model (likely an "
                "NGC weight-fetch failure or uncached model size). Refusing to "
                "mark healthy. Pre-fetch the requested weights while NGC is "
                "reachable (run the image's CPU-only download-to-cache "
                "entrypoint as the host UID), or deploy a dedicated image / "
                "profile for this model size."
            ),
            expected_slug=expected_slug,
            served_slug=loaded_slug,
            weight_files_found=0,
            selected_profile_id=selected_profile,
        )

    # Verified (live slug matched, or weights present and live probe
    # unavailable) — or inconclusive model name (expected_slug is None), which
    # we do not penalise.
    return ServedModelVerification(
        ok=True,
        reason=(
            "Served model verified"
            if expected_slug
            else "Served-model verification inconclusive (unrecognised model "
            "name) — proceeding"
        ),
        expected_slug=expected_slug,
        served_slug=loaded_slug,
        weight_files_found=max(weight_files, 0),
        selected_profile_id=selected_profile,
    )


# ── Port allocation ───────────────────────────────────────────────────────────


async def _resolve_port(preferred: int, reserved_ports: set[int] | None = None) -> int:
    """Find an available port starting from *preferred*.

    Tries the preferred port first, then increments up to +100. Uses a
    socket bind test (faster and more reliable than ``ss``) and skips ports
    already persisted by queued deployments whose containers have not bound
    them yet.

    Raises ``RuntimeError`` if no port is available.
    """
    reserved_ports = reserved_ports or set()
    for port in range(preferred, preferred + 101):
        if port in reserved_ports:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available port found in range {preferred}–{preferred + 100}"
    )


# ── GPU placement ─────────────────────────────────────────────────────────────


class GpuExhaustedError(RuntimeError):
    """No GPU can take the deployment.

    Raised by ``resolve_gpu_placement`` when every GPU on the host is
    occupied by a ``starting | running`` ``LocalNimDeployment`` row, or
    when every free GPU falls below the caller's memory floor.
    The caller chooses whether to opt into replace semantics
    (``stop_gpu_residents`` + redeploy) or surface ``409 gpu_exhausted``
    to the SME.
    """


def _scan_active_deployment_placements(
    workspace_root: str,
) -> list[tuple[str, str, str, str, int]]:
    """``(device, role, project_id, deployment_id, host_port)`` per active row.

    Scans every non-archived project in the workspace and collects every
    ``LocalNimDeployment`` row whose ``status`` is ``starting`` or
    ``running``. Cross-project: the one-NIM-per-GPU invariant
    is host-wide, not project-scoped.

    Per-project isolation: a single corrupt project DB MUST NOT block
    placement for a healthy deploy — catch broadly, log a per-project
    warning, continue the scan.
    """
    placements: list[tuple[str, str, str, str, int]] = []
    projects_dir = projects_root(workspace_root)
    if not projects_dir.exists():
        return placements
    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir():
            continue
        db_path = entry / "project.db"
        if not db_path.exists():
            continue
        if (entry / ".archived").exists():
            continue
        try:
            engine = get_project_engine(entry.name, workspace_root)
            if engine is None:
                continue
            with Session(engine) as session:
                stmt = select(LocalNimDeployment).where(
                    LocalNimDeployment.status.in_(("starting", "running")),
                )
                for dep in session.execute(stmt).scalars().all():
                    placements.append(
                        (
                            extract_device_index(dep.gpu_assignment),
                            str(dep.role),
                            str(dep.project_id),
                            str(dep.local_nim_deployment_id),
                            int(dep.host_port),
                        )
                    )
        except Exception as exc:
            logger.warning(
                "Skipping project %s in GPU-placement scan (%s: %s)",
                entry.name,
                type(exc).__name__,
                str(exc) or "(no message)",
            )
            continue
    return placements


def list_active_nim_residents(workspace_root: str) -> list[ActiveNimResident]:
    """Describe every Blueprint-managed NIM in ``starting | running``.

    The scan crosses non-archived project databases because a container is
    host infrastructure even though its owning deployment record is
    project-scoped.  Corrupt project databases are isolated exactly like the
    placement scan: one bad project cannot hide healthy residents elsewhere.
    """

    residents: list[ActiveNimResident] = []
    root = projects_root(workspace_root)
    if not root.exists():
        return residents

    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or (entry / ".archived").exists():
            continue
        if not (entry / "project.db").exists():
            continue
        try:
            engine = get_project_engine(entry.name, workspace_root)
            if engine is None:
                continue
            with Session(engine) as session:
                project = session.get(Project, entry.name)
                stmt = select(LocalNimDeployment).where(
                    LocalNimDeployment.status.in_(("starting", "running")),
                )
                for dep in session.execute(stmt).scalars().all():
                    mc = session.get(ModelConfig, dep.model_config_id)
                    metadata = dict(mc.local_deploy_metadata or {}) if mc else {}
                    normalized_env = tuple(
                        sorted(
                            normalize_extra_container_env(
                                metadata.get("extra_container_env")
                            ).items()
                        )
                    )
                    model_name = mc.model_name if mc is not None else None
                    if model_name is None and dep.role == "student":
                        model_name = dep.nim_served_model_name
                    residents.append(
                        ActiveNimResident(
                            project_id=str(dep.project_id),
                            project_name=(
                                str(project.name)
                                if project is not None
                                else str(dep.project_id)
                            ),
                            deployment_id=str(dep.local_nim_deployment_id),
                            model_config_id=str(dep.model_config_id),
                            role=str(dep.role),
                            model_name=str(model_name) if model_name else None,
                            nim_container_image=str(dep.nim_container_image),
                            gpu_assignment=str(dep.gpu_assignment),
                            endpoint_url=str(dep.endpoint_url),
                            host_port=int(dep.host_port),
                            status=str(dep.status),
                            nim_model_size=(
                                str(metadata["nim_model_size"])
                                if metadata.get("nim_model_size")
                                else None
                            ),
                            nim_model_profile=(
                                str(metadata["nim_model_profile"])
                                if metadata.get("nim_model_profile")
                                else None
                            ),
                            extra_container_env=normalized_env,
                        )
                    )
        except Exception as exc:
            logger.warning(
                "Skipping project %s in active-resident scan (%s: %s)",
                entry.name,
                type(exc).__name__,
                str(exc) or "(no message)",
            )
    return sorted(
        residents,
        key=lambda resident: (
            int(extract_device_index(resident.gpu_assignment)),
            resident.project_id,
            resident.deployment_id,
        ),
    )


def _requested_teacher_identity(
    project_id: str,
    model_config_id: str,
    nim_container_image: str,
    workspace_root: str,
) -> tuple[str, str, str | None, str | None, tuple[tuple[str, str], ...]] | None:
    """Resolve the container-affecting identity of a Teacher request."""

    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None
    with Session(engine) as session:
        mc = session.get(ModelConfig, model_config_id)
        if mc is None or mc.project_id != project_id:
            return None
        metadata = dict(mc.local_deploy_metadata or {})
        normalized_env = tuple(
            sorted(
                normalize_extra_container_env(
                    metadata.get("extra_container_env")
                ).items()
            )
        )
        return (
            str(mc.model_name).casefold(),
            nim_container_image,
            str(metadata["nim_model_size"]) if metadata.get("nim_model_size") else None,
            str(metadata["nim_model_profile"])
            if metadata.get("nim_model_profile")
            else None,
            normalized_env,
        )


def teacher_resident_matches_request(
    resident: ActiveNimResident,
    *,
    project_id: str,
    model_config_id: str,
    nim_container_image: str,
    workspace_root: str,
) -> bool:
    """Whether *resident* is the exact Teacher runtime the request needs."""

    requested = _requested_teacher_identity(
        project_id,
        model_config_id,
        nim_container_image,
        workspace_root,
    )
    if requested is None or resident.role != "teacher" or resident.model_name is None:
        return False
    actual = (
        resident.model_name.casefold(),
        resident.nim_container_image,
        resident.nim_model_size,
        resident.nim_model_profile,
        resident.extra_container_env,
    )
    return actual == requested


def find_compatible_running_teacher_resident(
    *,
    project_id: str,
    model_config_id: str,
    nim_container_image: str,
    workspace_root: str,
) -> ActiveNimResident | None:
    """Find a healthy-adopted resident that can serve the requested Teacher."""

    for resident in list_active_nim_residents(workspace_root):
        if resident.status != "running":
            continue
        if teacher_resident_matches_request(
            resident,
            project_id=project_id,
            model_config_id=model_config_id,
            nim_container_image=nim_container_image,
            workspace_root=workspace_root,
        ):
            return resident
    return None


def _resident_endpoint_overrides(
    resident: ActiveNimResident,
    workspace_root: str,
) -> tuple[int | None, str | None]:
    """Read endpoint-level image-cap values from the resident owner."""

    engine = get_project_engine(resident.project_id, workspace_root)
    if engine is None:
        return (None, None)
    with Session(engine) as session:
        stmt = (
            select(NimEndpoint)
            .where(
                NimEndpoint.local_nim_deployment_id == resident.deployment_id,
                NimEndpoint.base_url == resident.endpoint_url,
            )
            .order_by(NimEndpoint.created_at.desc())
        )
        endpoint = session.execute(stmt).scalars().first()
        if endpoint is None:
            return (None, None)
        return (endpoint.max_images_per_request, endpoint.image_cap_support)


def reuse_compatible_running_teacher(
    *,
    project_id: str,
    model_config_id: str,
    nim_container_image: str,
    workspace_root: str,
) -> ActiveNimResident | None:
    """Attach a project's matching ModelConfig to an existing Teacher NIM.

    The container keeps one owning ``LocalNimDeployment`` row.  Each consumer
    project gets its own local-system-managed ``NimEndpoint`` pointing at that
    resident and carrying the owner's deployment id.  This preserves
    per-project configuration while avoiding a duplicate container or a
    pointless stop/restart of identical weights.
    """

    resident = find_compatible_running_teacher_resident(
        project_id=project_id,
        model_config_id=model_config_id,
        nim_container_image=nim_container_image,
        workspace_root=workspace_root,
    )
    if resident is None:
        return None

    owner_deployment = get_local_deployment(
        resident.project_id,
        resident.deployment_id,
        workspace_root,
    )
    if owner_deployment is None or owner_deployment.status != "running":
        return None

    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None
    image_cap, image_cap_support = _resident_endpoint_overrides(
        resident, workspace_root
    )
    now = utc_now()
    with Session(engine) as session:
        mc = session.get(ModelConfig, model_config_id)
        if mc is None or mc.project_id != project_id:
            return None

        stmt = (
            select(NimEndpoint)
            .where(
                NimEndpoint.local_nim_deployment_id == resident.deployment_id,
                NimEndpoint.base_url == resident.endpoint_url,
            )
            .order_by(NimEndpoint.created_at.desc())
        )
        endpoint = session.execute(stmt).scalars().first()
        if endpoint is None:
            endpoint = NimEndpoint(
                endpoint_id=generate_uuid4(),
                project_id=project_id,
                display_name=(
                    f"Local Teacher ({resident.endpoint_url}; "
                    f"shared from {resident.project_name})"
                ),
                endpoint_mode="local_system_managed",
                base_url=resident.endpoint_url,
                api_format="openai_compatible",
                auth_mode="none",
                models_path="/models",
                health_ready_path="/health/ready",
                health_live_path="/health/live",
                metrics_path="/metrics",
                is_enabled=True,
                last_probe_at=now,
                last_probe_status="healthy",
                last_probe_error_ref=None,
                source_kind="auto_registered_local",
                local_nim_deployment_id=resident.deployment_id,
                max_images_per_request=image_cap,
                image_cap_support=image_cap_support,
            )
            session.add(endpoint)
        else:
            endpoint.is_enabled = True
            endpoint.last_probe_at = now
            endpoint.last_probe_status = "healthy"
            endpoint.last_probe_error_ref = None
            endpoint.max_images_per_request = image_cap
            endpoint.image_cap_support = image_cap_support

        mc.endpoint_id = endpoint.endpoint_id
        session.commit()

    # The resident can be stopped from its owning project while this project's
    # endpoint is being attached. Do not report reuse if that happened.
    current_owner = get_local_deployment(
        resident.project_id,
        resident.deployment_id,
        workspace_root,
    )
    if current_owner is None or current_owner.status != "running":
        disable_teacher_resident_endpoints(workspace_root, resident.deployment_id)
        return None

    logger.info(
        "Reused local Teacher resident %s from project %s for project %s "
        "(model=%s, endpoint=%s)",
        resident.deployment_id,
        resident.project_id,
        project_id,
        resident.model_name,
        resident.endpoint_url,
    )
    return resident


def reattach_selected_teacher_consumers(
    workspace_root: str,
    deployment_id: str,
) -> int:
    """Repair former consumers when an exact Teacher resident returns.

    A shared resident stop correctly disables every project-local attachment.
    The same exact runtime may later be restored under a new deployment row,
    often on the same host port.  Projects whose *selected* Teacher still
    points at a disabled Blueprint-managed local attachment should follow that
    replacement resident automatically; otherwise the picker says unavailable
    while the stale URL may accidentally reach the new process.

    This deliberately does not adopt hosted configs, self-hosted endpoints, or
    a different model selection. Exact runtime identity is checked with the
    same model/image/size/profile/environment contract as explicit reuse.
    """

    resident = next(
        (
            candidate
            for candidate in list_active_nim_residents(workspace_root)
            if candidate.deployment_id == deployment_id
            and candidate.role == "teacher"
            and candidate.status == "running"
        ),
        None,
    )
    if resident is None:
        return 0

    candidates: list[tuple[str, str, str]] = []
    root = projects_root(workspace_root)
    if not root.exists():
        return 0

    for entry in sorted(root.iterdir()):
        if (
            not entry.is_dir()
            or (entry / ".archived").exists()
            or not (entry / "project.db").exists()
        ):
            continue
        try:
            engine = get_project_engine(entry.name, workspace_root)
            if engine is None:
                continue
            with Session(engine) as session:
                project = session.get(Project, entry.name)
                if project is None or not project.teacher_model_config_id:
                    continue
                model_config = session.get(ModelConfig, project.teacher_model_config_id)
                if (
                    model_config is None
                    or model_config.project_id != entry.name
                    or "teacher" not in (model_config.eligible_roles or [])
                    or not model_config.supports_image_input
                ):
                    continue
                endpoint = session.get(NimEndpoint, model_config.endpoint_id)
                if (
                    endpoint is None
                    or endpoint.endpoint_mode != "local_system_managed"
                    or endpoint.source_kind != "auto_registered_local"
                    or (
                        endpoint.is_enabled
                        and endpoint.last_probe_status
                        not in {"unhealthy", "auth_failed", "unreachable"}
                    )
                ):
                    continue
                metadata = dict(model_config.local_deploy_metadata or {})
                image = metadata.get("nim_container_image")
                if not image:
                    continue
                candidates.append(
                    (entry.name, str(model_config.model_config_id), str(image))
                )
        except Exception as exc:
            logger.warning(
                "Skipping former Teacher consumer %s during resident repair (%s: %s)",
                entry.name,
                type(exc).__name__,
                str(exc) or "(no message)",
            )

    reattached = 0
    for project_id, model_config_id, image in candidates:
        if not teacher_resident_matches_request(
            resident,
            project_id=project_id,
            model_config_id=model_config_id,
            nim_container_image=image,
            workspace_root=workspace_root,
        ):
            continue
        reused = reuse_compatible_running_teacher(
            project_id=project_id,
            model_config_id=model_config_id,
            nim_container_image=image,
            workspace_root=workspace_root,
        )
        if reused is not None:
            reattached += 1

    if reattached:
        logger.info(
            "Reattached %d selected Teacher consumer(s) to restored resident %s",
            reattached,
            deployment_id,
        )
    return reattached


def reuse_first_compatible_running_teacher_for_project(
    *,
    project_id: str,
    workspace_root: str,
    preferred_model_name: str | None = None,
) -> tuple[str, ActiveNimResident] | None:
    """Attach the preferred exact running Teacher match in a fresh project.

    Project creation seeds a project-local ``ModelConfig`` for every catalog
    entry.  A Blueprint-managed Teacher that is already running is host
    infrastructure, so a fresh project should use the matching seeded config
    immediately instead of first selecting the hosted default and waiting for
    the setup UI to rediscover the resident.

    When ``preferred_model_name`` is supplied, a different resident is not
    adopted merely because it is already using the GPU. Setup can then name
    that conflict and ask whether to keep or replace it. Residents are
    considered in the deterministic order returned by
    :func:`list_active_nim_residents` (lowest GPU index first).  Identity is
    exact: model name, image, size/profile selectors, and operator-supplied
    container environment must all match.
    """

    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        candidates: dict[str, tuple[str, str]] = {}
        stmt = select(ModelConfig).where(ModelConfig.project_id == project_id)
        for mc in session.execute(stmt).scalars().all():
            metadata = dict(mc.local_deploy_metadata or {})
            image = metadata.get("nim_container_image")
            if "teacher" not in (mc.eligible_roles or []) or not image:
                continue
            candidates[str(mc.model_name).casefold()] = (
                str(mc.model_config_id),
                str(image),
            )

    for resident in list_active_nim_residents(workspace_root):
        if (
            resident.role != "teacher"
            or resident.status != "running"
            or resident.model_name is None
            or (
                preferred_model_name is not None
                and resident.model_name.casefold() != preferred_model_name.casefold()
            )
        ):
            continue
        candidate = candidates.get(resident.model_name.casefold())
        if candidate is None:
            continue
        model_config_id, image = candidate
        if not teacher_resident_matches_request(
            resident,
            project_id=project_id,
            model_config_id=model_config_id,
            nim_container_image=image,
            workspace_root=workspace_root,
        ):
            continue
        reused = reuse_compatible_running_teacher(
            project_id=project_id,
            model_config_id=model_config_id,
            nim_container_image=image,
            workspace_root=workspace_root,
        )
        if reused is not None:
            return (model_config_id, reused)
    return None


def disable_teacher_resident_endpoints(
    workspace_root: str,
    deployment_id: str,
) -> int:
    """Disable every project-local endpoint attached to a stopped resident."""

    disabled = 0
    root = projects_root(workspace_root)
    if not root.exists():
        return disabled
    for entry in sorted(root.iterdir()):
        # Archived projects cannot own a live resident (the archive busy gate
        # rejects active deployments) and are not active endpoint consumers.
        # Skip them before opening SQLite so a retained pre-v1 database cannot
        # flood an otherwise successful stop with irrelevant migration errors.
        if (
            not entry.is_dir()
            or (entry / ".archived").exists()
            or not (entry / "project.db").exists()
        ):
            continue
        try:
            engine = get_project_engine(entry.name, workspace_root)
            if engine is None:
                continue
            with Session(engine) as session:
                stmt = select(NimEndpoint).where(
                    NimEndpoint.local_nim_deployment_id == deployment_id,
                    NimEndpoint.is_enabled.is_(True),
                )
                endpoints = list(session.execute(stmt).scalars().all())
                for endpoint in endpoints:
                    endpoint.is_enabled = False
                    endpoint.last_probe_at = utc_now()
                    endpoint.last_probe_status = "unreachable"
                    endpoint.last_probe_error_ref = "Local NIM resident stopped"
                if endpoints:
                    session.commit()
                    disabled += len(endpoints)
        except Exception as exc:
            logger.warning(
                "Could not disable resident endpoint attachments in project %s "
                "(%s: %s)",
                entry.name,
                type(exc).__name__,
                str(exc) or "(no message)",
            )
    return disabled


def scan_active_residents_by_device(
    workspace_root: str,
) -> dict[str, list[tuple[str, str]]]:
    """Return ``{device_index: [(project_id, local_nim_deployment_id), …]}``."""
    residents: dict[str, list[tuple[str, str]]] = {}
    for (
        device,
        _role,
        project_id,
        deployment_id,
        _host_port,
    ) in _scan_active_deployment_placements(workspace_root):
        residents.setdefault(device, []).append((project_id, deployment_id))
    return residents


def scan_active_resident_roles_by_device(
    workspace_root: str,
) -> dict[str, set[str]]:
    """Return ``{device_index: {role, …}}`` for active deployments.

    The role-shaped view of the same scan
    ``scan_active_residents_by_device`` serves. The environment
    assessment uses it to work out which GPU the embedding NIM would
    actually get: a device holding an active Teacher / Student is not a
    candidate, while a device already running the embedding NIM is.
    """
    roles: dict[str, set[str]] = {}
    for (
        device,
        role,
        _project_id,
        _deployment_id,
        _host_port,
    ) in _scan_active_deployment_placements(workspace_root):
        roles.setdefault(device, set()).add(role)
    return roles


def scan_active_host_ports(workspace_root: str) -> set[int]:
    """Return host ports reserved by queued or running local NIMs."""
    return {
        host_port
        for _device, _role, _project_id, _deployment_id, host_port in (
            _scan_active_deployment_placements(workspace_root)
        )
    }


# Host-wide lock serializing GPU placement + deployment reservation. The
# one-NIM-per-GPU invariant is host-scoped, but project DB locks are
# per-project, so two concurrent cross-project deploys could both resolve the
# same free GPU and both start a container on it. deploy_local_nim holds this
# lock across the occupancy re-check and the whole reservation.
_GPU_DEPLOY_LOCK = asyncio.Lock()


def resolve_deploy_params(
    project_id: str,
    role: str,
    model_config_id: str | None,
    nim_container_image: str | None,
    gpu_assignment: str | None,
    preferred_port: int | None,
    settings: Settings,
) -> dict[str, Any]:
    """Resolve deployment parameters from request + catalog/config defaults.

    Raises ``ValueError`` with a ``map_service_error``-compatible message for
    client-fixable problems (unknown project/model config → "not found",
    missing/invalid role inputs → domain validation) and ``RuntimeError``
    when the deployment-scoped EmbeddingDeploymentConfig singleton is
    missing — a server-side invariant violation, not a client error.
    """
    if role == "teacher":
        if not model_config_id:
            raise ValueError("model_config_id is required for teacher role")
        engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
        if engine is None:
            raise ValueError("Project not found")

        with Session(engine) as session:
            mc = session.get(ModelConfig, model_config_id)
            if mc is None or mc.project_id != project_id:
                raise ValueError(f"ModelConfig {model_config_id} not found in project")
            local_meta = mc.local_deploy_metadata
            if not local_meta:
                raise ValueError(
                    f"Model {mc.model_name} does not support local deployment "
                    "(no local_deploy_metadata)"
                )

        image = nim_container_image or local_meta.get("nim_container_image", "")
        gpu_min = local_meta.get("nim_gpu_memory_minimum_gb", 0)
        compute_min = local_meta.get("nim_compute_capability_minimum")
        port = preferred_port or local_meta.get(
            "preferred_host_port", settings.LOCAL_NIM_TEACHER_PORT
        )

    elif role == "embedding":
        deploy_engine = init_deployment_db(Path(settings.WORKSPACE_ROOT))
        with Session(deploy_engine) as session:
            stmt = select(EmbeddingDeploymentConfig).limit(1)
            edc = session.execute(stmt).scalar_one_or_none()

        if edc is None:
            raise RuntimeError("EmbeddingDeploymentConfig not found")

        image = nim_container_image or edc.nim_container_image
        gpu_min = edc.gpu_memory_minimum_gb
        compute_min = None
        port = preferred_port or edc.preferred_host_port
        model_config_id = model_config_id or "embedding"

    else:
        raise ValueError(f"Invalid role: {role}")

    return {
        "model_config_id": model_config_id,
        "nim_container_image": image,
        "gpu_memory_minimum_gb": gpu_min,
        "gpu_compute_capability_minimum": compute_min,
        "preferred_port": port,
        "gpu_assignment": gpu_assignment,
    }


def activate_teacher_model_config(
    project_id: str, model_config_id: str, workspace_root: str
) -> bool:
    """Select a verified/reused local Teacher for one project.

    The caller is responsible for proving the endpoint is usable first. New
    deployments call this only from their verified-healthy transition; the
    router calls it after exact resident reuse has already attached a healthy
    endpoint. Returns False when the project or config disappeared.
    """

    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return False
    with Session(engine) as session:
        project = session.get(Project, project_id)
        model_config = session.get(ModelConfig, model_config_id)
        if (
            project is None
            or model_config is None
            or model_config.project_id != project_id
            or "teacher" not in (model_config.eligible_roles or [])
            or not model_config.supports_image_input
        ):
            return False
        project.teacher_model_config_id = model_config_id
        session.commit()
    logger.info(
        "Activated verified local Teacher %s for project %s",
        model_config_id,
        project_id,
    )
    return True


def update_embedding_deployment_config(
    workspace_root: str, updates: dict[str, Any]
) -> EmbeddingDeploymentConfig | None:
    """Apply a partial update to the deployment-scoped singleton.

    Returns the refreshed row, or ``None`` when the singleton is missing
    (deployment.db seeding invariant violated — callers surface a 500).
    """
    engine = init_deployment_db(workspace_root)
    with Session(engine) as session:
        config = session.execute(select(EmbeddingDeploymentConfig)).scalar_one_or_none()
        if config is None:
            return None
        for key, value in updates.items():
            setattr(config, key, value)
        session.commit()
        session.refresh(config)
        return config


async def probe_embedding_endpoint(
    base_url: str,
    auth_headers: dict[str, str],
    settings: Settings,
) -> EmbeddingDeploymentConfig:
    """Verify one embedding endpoint without mutating deployment state.

    A generic ``GET /models`` probe can go green against a Teacher-only NIM.
    The embedding configuration path therefore sends the real NeMo Retriever
    request and requires one finite vector at the deployment's exact seeded
    dimension before the URL is eligible for Save.
    """
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    with Session(engine) as session:
        config = session.execute(select(EmbeddingDeploymentConfig)).scalar_one_or_none()
        if config is None:
            raise NimEndpointConfigurationError(
                500,
                "embedding_config_missing",
                "Embedding deployment configuration is unavailable.",
            )
        session.expunge(config)

    result = await create_embeddings(
        base_url=base_url,
        auth_headers=auth_headers,
        model=config.model_name,
        input_items=["VLM Feedback Loop embedding connection test"],
        deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
        max_retries=1,
        input_type=settings.EMBEDDING_INPUT_TYPE,
    )
    if not result.success or not result.embeddings:
        message = (
            result.error or "Could not obtain an embedding from this NIM endpoint."
        )
        if result.status_code == 404:
            message = (
                "This endpoint does not expose the required /embeddings operation "
                "for the configured NeMo Retriever model."
            )
        raise NimEndpointConfigurationError(
            400,
            "embedding_probe_failed",
            message,
        )
    if len(result.embeddings) != 1:
        raise NimEndpointConfigurationError(
            400,
            "embedding_response_invalid",
            "The endpoint returned an unexpected number of embedding vectors.",
        )
    vector = result.embeddings[0]
    if len(vector) != config.embedding_dim or not all(
        math.isfinite(value) for value in vector
    ):
        raise NimEndpointConfigurationError(
            400,
            "embedding_response_invalid",
            f"The endpoint must return one finite {config.embedding_dim}-dimensional vector.",
        )
    return config


async def probe_self_hosted_embedding(
    base_url: str,
    settings: Settings,
) -> tuple[str, EmbeddingDeploymentConfig]:
    """Normalize and verify one credential-free embedding NIM."""
    normalized_url = normalize_self_hosted_base_url(base_url)
    config = await probe_embedding_endpoint(normalized_url, {}, settings)
    return normalized_url, config


async def configure_self_hosted_embedding(
    base_url: str,
    settings: Settings,
) -> EmbeddingDeploymentConfig:
    """Live-verify and durably select one external embedding NIM.

    The network call completes before any write. A concurrent or already
    active Blueprint-managed embedding resident blocks the switch so a Save
    cannot orphan a GPU-consuming container behind an external URL.
    """
    if settings.EMBEDDING_PROVIDER not in (
        "auto",
        "self_hosted_nvclip",
        "local_nvclip",
    ):
        raise NimEndpointConfigurationError(
            409,
            "embedding_provider_pinned",
            "The operator-pinned EMBEDDING_PROVIDER setting does not allow a self-hosted endpoint.",
        )

    normalized_url, _config = await probe_self_hosted_embedding(base_url, settings)

    async with _GPU_DEPLOY_LOCK:
        active_embedding_placements = [
            (project_id, deployment_id)
            for (
                _device,
                role,
                project_id,
                deployment_id,
                _host_port,
            ) in _scan_active_deployment_placements(settings.WORKSPACE_ROOT)
            if role == "embedding"
        ]
        exact_managed_gpu: str | None = None
        for owner_project_id, deployment_id in active_embedding_placements:
            owner_engine = get_project_engine(owner_project_id, settings.WORKSPACE_ROOT)
            if owner_engine is None:
                continue
            with Session(owner_engine) as session:
                deployment = session.get(LocalNimDeployment, deployment_id)
                if (
                    deployment is not None
                    and deployment.status in ("starting", "running")
                    and deployment.endpoint_url == normalized_url
                ):
                    exact_managed_gpu = deployment.gpu_assignment
                    break
        current_engine = init_deployment_db(settings.WORKSPACE_ROOT)
        with Session(current_engine) as session:
            current = session.execute(
                select(EmbeddingDeploymentConfig)
            ).scalar_one_or_none()
            current_endpoint_url = current.endpoint_url if current is not None else None
        if active_embedding_placements and (
            current_endpoint_url != normalized_url or exact_managed_gpu is None
        ):
            raise NimEndpointConfigurationError(
                409,
                "local_embedding_active",
                "Stop the Blueprint-managed local embedding NIM before switching to a different self-hosted endpoint.",
            )
        updates: dict[str, Any] = {
            "provider": "self_hosted_nvclip",
            "endpoint_url": normalized_url,
        }
        # Re-saving the exact active managed endpoint is a harmless live
        # verification and must preserve its placement identity. A genuinely
        # external endpoint has no Blueprint GPU assignment.
        updates["gpu_assignment"] = exact_managed_gpu
        updated = update_embedding_deployment_config(
            settings.WORKSPACE_ROOT,
            updates,
        )

    if updated is None:
        raise NimEndpointConfigurationError(
            500,
            "embedding_config_missing",
            "Embedding deployment configuration is unavailable.",
        )
    await resweep_embedding_tasks(settings)
    return updated


def _reset_embedding_deployment_config(
    workspace_root: str, endpoint_url: str | None
) -> bool:
    """Reset the singleton to the no-local-provider state.

    Runs when a local embedding NIM stops (user stop, displacement via
    ``stop_gpu_residents``), fails its startup health window, or is
    found gone/unhealthy by restart recovery, so ``GET /v1/environment``
    and the probe cascade stop advertising a dead endpoint. Crash paths
    the backend never observes are covered by the probe's live verify.

    Scoped to the dying deployment: the reset fires only when the
    config still points at *endpoint_url* (the deployment's own
    endpoint, the one its healthy transition stamped) or points at
    nothing. A replacement deployment may have re-stamped the config
    since — stopping the displaced container must not un-register its
    live successor. Returns True when the config was reset.
    """
    engine = init_deployment_db(Path(workspace_root))
    with Session(engine) as session:
        config = session.execute(select(EmbeddingDeploymentConfig)).scalar_one_or_none()
        if config is None:
            return False
        if config.endpoint_url is not None and config.endpoint_url != endpoint_url:
            return False
        config.provider = "none"
        config.endpoint_url = None
        config.updated_at = utc_now()
        session.commit()
        return True


async def resolve_gpu_placement(
    role: str,
    explicit_gpu: str | None,
    workspace_root: str,
    min_gpu_memory_gb: int | None = None,
    min_compute_capability: float | None = None,
) -> str:
    """Resolve which GPU device to use for a local NIM deployment.

    One-NIM-per-GPU
    invariant: placement is deterministic — claim the lowest-indexed
    GPU whose ``LocalNimDeployment`` rows are all terminal
    (``stopped`` / ``failed``). The auto-placer never returns a GPU
    whose residents are active. Co-location on the same GPU is not
    supported in v1; to start a new NIM on an occupied GPU the caller
    MUST explicitly opt into replace semantics
    (``replace_resident=true`` on ``POST :deploy``, or via the Student
    NIM lifecycle's pre-docker_run acquire-GPU step).

    - If *explicit_gpu* is provided, return it as-is (operator/caller
      assumes responsibility for invoking the replace path if needed).
    - Else: scan all non-archived projects, find the lowest free GPU
      index, return ``"device=N"``.
    - When *min_gpu_memory_gb* is given, free GPUs below that floor are
      skipped — embedding deploys pass the config floor so a
      heterogeneous host (say 80 GB + 16 GB) never lands the embedding
      NIM on a device that cannot hold it. Teacher callers also pass their
      floor so automatic placement cannot select an undersized free GPU.
      Student callers pass ``None`` (fit is enforced by their lifecycle).
    - If no free GPU qualifies → raise ``GpuExhaustedError``.

    *role* is a diagnostic label included in the ``gpu_exhausted``
    error message; it does not drive placement (all roles use the
    same lowest-free-index policy).
    """
    if explicit_gpu is not None:
        return explicit_gpu

    gpus = await probe_gpu_inventory()
    if not gpus:
        raise ValueError("No GPUs detected. Cannot auto-place local NIM deployment.")

    residents = scan_active_residents_by_device(workspace_root)

    free_below_floor: list[str] = []
    for i, gpu in enumerate(gpus):
        if str(i) in residents:
            continue
        if min_gpu_memory_gb is not None and not gpu_memory_meets_floor(
            gpu.memory_total_gb, min_gpu_memory_gb
        ):
            free_below_floor.append(f"device={i} ({gpu.memory_total_gb:g} GB)")
            continue
        if min_compute_capability is not None and (
            gpu.compute_capability is None
            or gpu.compute_capability < min_compute_capability
        ):
            actual = (
                f"cc {gpu.compute_capability:g}"
                if gpu.compute_capability is not None
                else "unknown compute capability"
            )
            free_below_floor.append(f"device={i} ({actual})")
            continue
        return f"device={i}"

    if free_below_floor:
        # Free GPUs exist but none meets the floor. Replace semantics
        # only help when some occupied GPU meets it — say which.
        largest_gb = max(gpu.memory_total_gb for gpu in gpus)
        if min_gpu_memory_gb is not None and largest_gb >= min_gpu_memory_gb:
            remedy = (
                f"The largest GPU on this host ({largest_gb:g} GB) meets the "
                f"floor but is occupied — pass replace_resident=true to "
                f"displace its resident, or fall back to the hosted provider "
                f"(NVIDIA_API_KEY)."
            )
        else:
            remedy = (
                f"No GPU on this host meets the floor (largest is "
                f"{largest_gb:g} GB) — use a host with a larger GPU, or fall "
                f"back to the hosted provider (NVIDIA_API_KEY)."
            )
        raise GpuExhaustedError(
            f"No free GPU meets the {min_gpu_memory_gb} GB memory floor "
            f"for role={role}: {', '.join(free_below_floor)} free but too "
            f"small. {remedy}"
        )

    # Every GPU is occupied by an active resident. Replace semantics are
    # the caller's opt-in path; surface that explicitly via
    # GpuExhaustedError so the router can return 409 gpu_exhausted and
    # the Student lifecycle can decide whether to fall back to
    # ``device=0`` + replace.
    raise GpuExhaustedError(
        f"All {len(gpus)} GPU(s) occupied by active NIM deployments "
        f"(one-NIM-per-GPU invariant). Pass replace_resident=true to "
        f"stop a resident and reuse its GPU. role={role}"
    )


async def resolve_replace_target(
    role: str,
    min_gpu_memory_gb: int | None = None,
    min_compute_capability: float | None = None,
) -> str:
    """Pick the GPU a ``replace_resident=true`` deploy should land on.

    The router falls back here when ``resolve_gpu_placement`` found no
    free qualifying GPU and the caller opted into replace semantics:
    occupancy no longer disqualifies a device (its resident will be
    displaced), but the memory floor still does — replacing a resident
    cannot grow the GPU. Returns the lowest-indexed device meeting the
    floor; raises ``GpuExhaustedError`` when no device on the host does.
    With no floor (Teacher/Student callers) this is ``device=0``, the
    same fall-back the Student lifecycle uses on single-GPU hosts.
    """
    gpus = await probe_gpu_inventory()
    if not gpus:
        raise ValueError("No GPUs detected. Cannot auto-place local NIM deployment.")

    for i, gpu in enumerate(gpus):
        if min_gpu_memory_gb is not None and not gpu_memory_meets_floor(
            gpu.memory_total_gb, min_gpu_memory_gb
        ):
            continue
        if min_compute_capability is not None and (
            gpu.compute_capability is None
            or gpu.compute_capability < min_compute_capability
        ):
            continue
        return f"device={i}"

    largest_gb = max(gpu.memory_total_gb for gpu in gpus)
    raise GpuExhaustedError(
        f"No GPU on this host meets the {min_gpu_memory_gb} GB memory floor "
        f"for role={role} (largest is {largest_gb:g} GB) — replacing a "
        f"resident cannot help. Use a host with a larger GPU, or fall back "
        f"to the hosted provider (NVIDIA_API_KEY)."
    )


async def stop_gpu_residents(
    workspace_root: str,
    device_index: str,
    *,
    displaced_by_deployment_id: str | None = None,
    settings: Settings | None = None,
) -> list[LocalNimDeployment]:
    """Stop every active ``LocalNimDeployment`` on *device_index*.

    Enforces the one-NIM-per-GPU invariant by stopping every
    deployment whose ``status`` is ``starting | running`` and whose
    ``gpu_assignment`` resolves to *device_index*. Scans all
    non-archived projects (the invariant is host-wide).

    Each stopped row's ``displaced_by_deployment_id`` (if provided) and
    ``displaced_at`` are persisted on the same row for audit.
    The container is removed via the standard ``stop_local_nim`` path
    so the GPU is actually freed (not just marked stopped in the DB).

    Idempotent: no residents → returns ``[]``. Per-project isolation:
    one project's broken DB does not block the scan.

    *settings* is threaded to ``stop_local_nim`` so a displaced
    embedding NIM proactively resweeps projects onto the fallback
    provider; ``None`` skips that hook.

    Returns the list of ``LocalNimDeployment`` rows that were stopped.
    """
    residents_by_device = scan_active_residents_by_device(workspace_root)
    targets = residents_by_device.get(extract_device_index(device_index), [])
    if not targets:
        return []

    stopped: list[LocalNimDeployment] = []
    for project_id, deployment_id in targets:
        try:
            dep = await stop_local_nim(
                deployment_id=deployment_id,
                project_id=project_id,
                workspace_root=workspace_root,
                settings=settings,
            )
        except Exception as exc:
            logger.warning(
                "acquire-GPU stop failed for %s/%s (%s: %s); continuing",
                project_id,
                deployment_id,
                type(exc).__name__,
                str(exc) or "(no message)",
            )
            continue
        if dep is None:
            continue

        # Persist displacement audit on the displaced row. stop_local_nim
        # has already committed status="stopped" + status_reason +
        # stopped_at; we tack on the displacement-audit fields in a
        # follow-up write so
        # the helper stays a single-purpose primitive.
        engine = get_project_engine(project_id, workspace_root)
        if engine is not None and displaced_by_deployment_id is not None:
            with Session(engine) as session:
                row = session.get(LocalNimDeployment, deployment_id)
                if row is not None:
                    row.displaced_by_deployment_id = displaced_by_deployment_id
                    row.displaced_at = utc_now()
                    row.status_reason = "displaced_by_replace"
                    session.commit()
                    session.refresh(row)
                    session.expunge(row)
                    stopped.append(row)
                    continue
        stopped.append(dep)

    if stopped:
        logger.info(
            "acquire-GPU stopped %d resident(s) on %s (displaced_by=%s)",
            len(stopped),
            device_index,
            displaced_by_deployment_id,
        )
    return stopped


# ── Preflight checks ─────────────────────────────────────────────────────────

# Container registry that serves the NIM images. A configured NGC key is
# necessary but NOT sufficient to pull from here: the docker *daemon* must
# also be authenticated, or a ``docker pull`` runs anonymously and nvcr.io
# returns "Access Denied" even for a key with full pull entitlement.
_NVCR_REGISTRY = "nvcr.io"
# NVIDIA's fixed registry username convention — the literal string
# ``$oauthtoken`` (NOT a shell variable; run_subprocess uses exec, so it is
# passed verbatim), with the NGC API key as the password.
_NVCR_REGISTRY_USERNAME = "$oauthtoken"


async def docker_login_nvcr(settings: Settings) -> tuple[bool, str]:
    """Authenticate the docker daemon to nvcr.io with the effective NGC key.

    The NIM container images on ``nvcr.io`` are private. A bare
    ``docker pull`` (preflight check 5) and the implicit pull inside
    ``docker run`` (deploy) both run under whatever credentials the docker
    daemon holds — nothing about passing ``NGC_API_KEY`` as a *container*
    env var (for runtime weight download) authenticates the *image* pull.
    On a fresh host with no ``~/.docker/config.json`` the pull is anonymous
    and nvcr.io denies it, surfacing as an opaque ``Access Denied`` at
    deploy time despite a green NGC-key credential test.

    The canonical NIM workflow closes that gap with
    ``docker login nvcr.io -u '$oauthtoken' --password-stdin``. We run it
    here — key on stdin, never argv — before pulling. It is idempotent
    (writes/refreshes ``~/.docker/config.json``) and covers every
    subsequent pull for the life of the host.

    Returns ``(ok, diagnostic)``. ``ok=False`` when no NGC key is
    configured or the login command fails.
    """
    key = get_effective_secret("NGC_API_KEY", settings)
    if not key:
        return False, "NGC_API_KEY is not configured; cannot authenticate to nvcr.io."
    rc, _stdout, stderr = await run_subprocess(
        "docker",
        "login",
        _NVCR_REGISTRY,
        "--username",
        _NVCR_REGISTRY_USERNAME,
        "--password-stdin",
        timeout_s=30.0,
        stdin_input=key,
    )
    if rc == 0:
        return True, f"Authenticated docker to {_NVCR_REGISTRY}."
    return (
        False,
        f"docker login {_NVCR_REGISTRY} failed: {stderr or 'unknown error'}. "
        "The NGC key must have the NGC Catalog and Private Registry scopes.",
    )


async def run_preflight_checks(
    nim_container_image: str,
    gpu_memory_minimum_gb: int,
    gpu_assignment: str,
    role: str,
    settings: Settings,
    *,
    gpu_compute_capability_minimum: float | None = None,
    nim_model_size: str | None = None,
    nim_model_profile: str | None = None,
    nim_served_model_name: str | None = None,
) -> PreflightResult:
    """Run the 7 sequential preflight checks for local NIM deployment.

    Checks 1-5 short-circuit on failure (later checks depend on earlier ones).
    Checks 6-7 always run if 1-5 pass.

    Returns a ``PreflightResult`` with per-check diagnostics and the exact
    docker run command (always populated for Action Request on failure).
    """
    # Mock-mode hook (Student role only): when
    # LOCAL_NIM_MOCK_ENDPOINT_URL is set, return an all-pass preflight so
    # the closing-smoke can drive the lifecycle end-to-end against the
    # MockNIMServer fixture without needing real Docker / GPU. Production
    # is unaffected when the env var is unset.
    if role == "student" and os.environ.get("LOCAL_NIM_MOCK_ENDPOINT_URL"):
        return PreflightResult(
            all_passed=True,
            checks=[
                PreflightCheckResult(
                    check_name=name,
                    passed=True,
                    diagnostic="(mock-mode bypass — test fixture)",
                )
                for name in (
                    "docker",
                    "nvidia_toolkit",
                    "gpu_memory",
                    "ngc_api_key",
                    "registry_auth",
                    "image_pullable",
                    "checkpoint_validated",
                )
            ],
            docker_run_command=docker_run_command_display(
                nim_container_image=nim_container_image,
                container_name=build_container_name(role, "mock"),
                gpu_assignment=gpu_assignment,
                host_port=settings.NIM_STUDENT_PORT,
                role=role,
            ),
        )

    checks: list[PreflightCheckResult] = []
    device_index = extract_device_index(gpu_assignment)
    container_name = build_container_name(role, "preflight")

    # Always build docker_run_command for Action Request.  Student
    # deployments use a third role-specific default port (NIM_STUDENT_PORT).
    # The orchestrator may rebuild this command with Student-specific
    # checkpoint mount / served-model-name kwargs before passing it to the
    # ``student_nim_deploy`` Action Request generator.
    if role == "teacher":
        host_port = settings.LOCAL_NIM_TEACHER_PORT
    elif role == "student":
        host_port = settings.NIM_STUDENT_PORT
    else:
        host_port = settings.LOCAL_NIM_NVCLIP_PORT
    docker_run_command = docker_run_command_display(
        nim_container_image=nim_container_image,
        container_name=container_name,
        gpu_assignment=gpu_assignment,
        host_port=host_port,
        role=role,
        nim_model_size=nim_model_size,
        nim_model_profile=nim_model_profile,
        nim_served_model_name=nim_served_model_name,
    )

    # ── Check 1: Docker available ─────────────────────────────────────
    docker_ok, docker_err = await check_docker_available()
    checks.append(
        PreflightCheckResult(
            check_name="docker",
            passed=docker_ok,
            diagnostic="Docker is available"
            if docker_ok
            else (docker_err or "Docker not available"),
        )
    )
    if not docker_ok:
        return PreflightResult(
            all_passed=False, checks=checks, docker_run_command=docker_run_command
        )

    # ── Check 2: NVIDIA Container Toolkit ─────────────────────────────
    toolkit_ok, toolkit_err = await check_nvidia_toolkit()
    checks.append(
        PreflightCheckResult(
            check_name="nvidia_toolkit",
            passed=toolkit_ok,
            diagnostic="NVIDIA Container Toolkit is available"
            if toolkit_ok
            else (toolkit_err or "Toolkit not available"),
        )
    )
    if not toolkit_ok:
        return PreflightResult(
            all_passed=False, checks=checks, docker_run_command=docker_run_command
        )

    # ── Check 3: GPU memory vs minimum ────────────────────────────────
    rc, stdout, stderr = await run_subprocess(
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
        f"--id={device_index}",
        timeout_s=10.0,
    )
    gpu_check_passed = False
    gpu_diagnostic = ""
    if rc == 0 and stdout:
        parts = [p.strip() for p in stdout.split(",")]
        if len(parts) >= 2:
            try:
                gpu_name = parts[0]
                gpu_mem_mb = int(float(parts[1]))
                gpu_mem_gb = gpu_mem_mb / 1024
                gpu_compute_capability = (
                    float(parts[2]) if len(parts) >= 3 and parts[2] else None
                )
                memory_ok = gpu_memory_meets_floor(gpu_mem_gb, gpu_memory_minimum_gb)
                compute_ok = gpu_compute_capability_minimum is None or (
                    gpu_compute_capability is not None
                    and gpu_compute_capability >= gpu_compute_capability_minimum
                )
                if memory_ok and compute_ok:
                    gpu_check_passed = True
                    gpu_diagnostic = (
                        f"GPU {device_index} ({gpu_name}): "
                        f"{gpu_mem_gb:.0f} GB available, need "
                        f">={gpu_memory_minimum_gb} GB"
                    )
                    if gpu_compute_capability_minimum is not None:
                        gpu_diagnostic += (
                            f"; compute capability {gpu_compute_capability:g}, "
                            f"need >={gpu_compute_capability_minimum:g}"
                        )
                else:
                    gpu_diagnostic = (
                        f"GPU {device_index} ({gpu_name}): "
                        f"{gpu_mem_gb:.0f} GB available, need "
                        f">={gpu_memory_minimum_gb} GB"
                    )
                    if gpu_compute_capability_minimum is not None:
                        actual_compute = (
                            f"{gpu_compute_capability:g}"
                            if gpu_compute_capability is not None
                            else "unknown"
                        )
                        gpu_diagnostic += (
                            f"; compute capability {actual_compute}, "
                            f"need >={gpu_compute_capability_minimum:g}"
                        )
            except (ValueError, IndexError):
                gpu_diagnostic = f"Failed to parse GPU memory from nvidia-smi: {stdout}"
        else:
            gpu_diagnostic = f"Unexpected nvidia-smi output: {stdout}"
    else:
        gpu_diagnostic = (
            f"nvidia-smi failed for device {device_index}: {stderr or 'unknown error'}"
        )

    checks.append(
        PreflightCheckResult(
            check_name="gpu_memory",
            passed=gpu_check_passed,
            diagnostic=gpu_diagnostic,
        )
    )
    if not gpu_check_passed:
        return PreflightResult(
            all_passed=False, checks=checks, docker_run_command=docker_run_command
        )

    # ── Check 4: NGC_API_KEY configured ───────────────────────────────
    # The runtime secret override is honored — preflight passes
    # immediately after a UI-applied NGC key, no restart required.
    ngc_ok = bool(get_effective_secret("NGC_API_KEY", settings))
    checks.append(
        PreflightCheckResult(
            check_name="ngc_api_key",
            passed=ngc_ok,
            diagnostic="NGC_API_KEY is configured"
            if ngc_ok
            else (
                "NGC_API_KEY is not configured. "
                "Add it to ~/.vlm_feedback_loop/.env or paste in the NIM "
                "Configuration screen to apply it for this session."
            ),
        )
    )
    if not ngc_ok:
        return PreflightResult(
            all_passed=False, checks=checks, docker_run_command=docker_run_command
        )

    # ── Check 5: Docker daemon authenticated to nvcr.io ───────────────
    # A configured NGC key (check 4) is necessary but not sufficient: the
    # docker DAEMON must be logged in to nvcr.io or the private-image pull
    # below runs anonymously and is denied. Log in now (key on stdin) so
    # both this pull and the deploy-time ``docker run`` pull carry
    # credentials. See docker_login_nvcr for the full rationale.
    login_ok, login_diag = await docker_login_nvcr(settings)
    checks.append(
        PreflightCheckResult(
            check_name="registry_auth",
            passed=login_ok,
            diagnostic=login_diag,
        )
    )
    if not login_ok:
        return PreflightResult(
            all_passed=False, checks=checks, docker_run_command=docker_run_command
        )

    # ── Check 6: Container image pullable ────────────────────────────
    # Pull the image BEFORE running list-model-profiles (check 7).
    # On a cold cache, docker run implicitly pulls the image, which can
    # easily exceed the list-model-profiles timeout.  An explicit pull
    # here gives clear progress and a generous timeout (10 min).
    pull_rc, _pull_stdout, pull_stderr = await run_subprocess(
        "docker",
        "pull",
        nim_container_image,
        timeout_s=600.0,
    )
    image_ok = pull_rc == 0
    checks.append(
        PreflightCheckResult(
            check_name="image_pullable",
            passed=image_ok,
            diagnostic=(
                f"Container image {nim_container_image} is available"
                if image_ok
                else f"Cannot pull container image {nim_container_image}: {pull_stderr or 'unknown error'}"
            ),
        )
    )
    if not image_ok:
        return PreflightResult(
            all_passed=False, checks=checks, docker_run_command=docker_run_command
        )

    # ── Check 7: Model profile compatible ─────────────────────────────
    # Image is now cached, so list-model-profiles runs without a pull.
    #
    # Embedding NIMs (legacy NV-CLIP, current NeMo Retriever VL) do not
    # support list-model-profiles as a standalone probe. For embedding
    # containers we skip this check; the GPU memory gate (check 3) is
    # sufficient.
    if role == "embedding":
        checks.append(
            PreflightCheckResult(
                check_name="model_profile",
                passed=True,
                diagnostic="Skipped for embedding role (GPU memory validated in check 3)",
            )
        )
    else:
        # NIM images commonly use a shell-form entrypoint such as
        # ``/bin/bash -c $SERVER_START_SCRIPT_PATH``. Appending
        # ``list-model-profiles`` after the image makes it bash's ``$0`` and
        # the wrapper ignores it, starting the full inference server instead.
        # Override the entrypoint so the standalone probe binary is actually
        # invoked. Keep the bounded timeout and cleanup as defense in depth:
        #   1. False FAILURE — a GPU where the model serves fine is rejected
        #      with "No compatible model profile ... Command timed out after
        #      120.0s", because the probe simply hasn't finished initialising.
        #   2. GPU-holding ORPHAN — ``run_subprocess`` kills only the docker
        #      *client* on timeout; the detached container keeps running and
        #      grabs ~90% of VRAM (NIM's gpu_memory_utilization floor), so
        #      ``--rm`` never fires. The next deploy then hits
        #      "Detected 0 compatible profile(s)" (one-NIM-per-GPU collision).
        # Hence: deterministic ``--name`` + force-remove before AND after so a
        # timed-out probe can't leak; and treat a timeout as INCONCLUSIVE (not
        # a deploy failure) — the serve health poll (NIM_STARTUP_TIMEOUT_S) is
        # the authoritative profile gate. A real rc!=0 error still fails.
        preflight_name = f"vlm-preflight-profile-{device_index}"
        await run_subprocess("docker", "rm", "-f", preflight_name, timeout_s=30.0)
        cache_root = Path(NIM_CACHE_ROOT).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        _prepare_declared_user_cache(nim_container_image)
        profile_runtime = [
            "--shm-size=32GB",
            "-v",
            f"{cache_root}:/opt/nim/.cache",
        ]
        if not _uses_declared_container_user(nim_container_image):
            profile_runtime.extend(["-u", str(os.getuid())])
        profile_env = ["-e", "NGC_API_KEY"]
        if nim_model_size:
            profile_env.extend(["-e", f"NIM_MODEL_SIZE={nim_model_size}"])
        if nim_model_profile:
            profile_env.extend(["-e", f"NIM_MODEL_PROFILE={nim_model_profile}"])
        if nim_served_model_name:
            profile_env.extend(["-e", f"NIM_SERVED_MODEL_NAME={nim_served_model_name}"])
        profile_rc, profile_stdout, profile_stderr = await run_subprocess(
            "docker",
            "run",
            "--rm",
            "--name",
            preflight_name,
            "--entrypoint",
            "/usr/local/bin/list-model-profiles",
            "--runtime=nvidia",
            "--gpus",
            f'"device={device_index}"',
            *profile_runtime,
            *profile_env,
            nim_container_image,
            timeout_s=PROFILE_PROBE_TIMEOUT_S,
            secret_env={
                "NGC_API_KEY": get_effective_secret("NGC_API_KEY", settings) or ""
            },
        )
        # Always force-remove: on timeout the container is still running and
        # holding the GPU (see note above); --rm only fires on clean exit.
        await run_subprocess("docker", "rm", "-f", preflight_name, timeout_s=30.0)
        timed_out = profile_rc == -1 and "timed out" in (profile_stderr or "")
        probe_output = f"{profile_stderr or ''}\n{profile_stdout or ''}"
        # Some published single-model NIMs (live-confirmed with Cosmos
        # Reason2 8B 1.6.0) do not ship the optional standalone probe binary.
        # That says nothing about whether their one model can serve on this
        # GPU. Keep shared-image selection strict: Nano/Super must retain the
        # probe because otherwise the image can validate its default sibling.
        probe_unavailable_for_single_image = (
            nim_model_size is None
            and re.search(
                r"/usr/local/bin/list-model-profiles.*(?:no such file|not found)",
                probe_output,
                re.IGNORECASE,
            )
            is not None
        )
        if profile_rc == 0:
            profile_ok = True
            profile_diagnostic = f"Compatible model profile found on GPU {device_index}"
        elif timed_out:
            profile_ok = True
            profile_diagnostic = (
                f"Profile probe inconclusive on GPU {device_index}: "
                f"list-model-profiles did not return within "
                f"{int(PROFILE_PROBE_TIMEOUT_S)}s. Proceeding — the serve "
                f"health check is the authoritative profile gate."
            )
        elif probe_unavailable_for_single_image:
            profile_ok = True
            profile_diagnostic = (
                f"Profile probe unavailable for this single-model image on GPU "
                f"{device_index}: the image does not ship list-model-profiles. "
                f"Proceeding — the serve health and served-model checks are the "
                f"authoritative compatibility gates."
            )
        else:
            profile_ok = False
            profile_diagnostic = (
                f"No compatible model profile on GPU {device_index}: "
                f"{profile_stderr or profile_stdout or 'unknown error'}"
            )
        checks.append(
            PreflightCheckResult(
                check_name="model_profile",
                passed=profile_ok,
                diagnostic=profile_diagnostic,
            )
        )

    all_passed = all(c.passed for c in checks)
    return PreflightResult(
        all_passed=all_passed,
        checks=checks,
        docker_run_command=docker_run_command,
    )


# ── Deploy ────────────────────────────────────────────────────────────────────


async def deploy_local_nim(
    project_id: str,
    model_config_id: str,
    role: str,
    nim_container_image: str,
    gpu_assignment: str,
    gpu_memory_minimum_gb: int,
    preferred_port: int,
    settings: Settings,
    workspace_root: str,
    *,
    student_model_id: str | None = None,
    checkpoint_mount: str | None = None,
    nim_model_name_path: str | None = None,
    nim_served_model_name: str | None = None,
    precision_method: str | None = None,
    replace_resident: bool = False,
    activate_on_success: bool = False,
    background: bool = False,
) -> dict[str, Any]:
    """Deploy a local NIM, serializing GPU reservation host-wide.

    Thin lock wrapper over :func:`_deploy_local_nim_impl`. The lock closes the
    TOCTOU between placement and the ``starting`` record commit: without it two
    concurrent cross-project deploys (project DB locks are per-project) could
    both resolve the same free GPU and both start a container on it. Under the
    lock we re-check occupancy (a resident may have appeared since the caller's
    placement) and, unless the caller opted into replace semantics, refuse with
    ``GpuExhaustedError`` instead of co-locating.

    ``background=True`` reserves the GPU with a persisted ``starting`` record
    and returns immediately. Replacement, preflight, image pull, container
    startup, and health polling then run in tracked background tasks. Direct
    service callers keep the original synchronous orchestration by default.
    """
    async with _GPU_DEPLOY_LOCK:
        if not replace_resident:
            device_index = extract_device_index(gpu_assignment)
            residents = scan_active_residents_by_device(workspace_root)
            if device_index in residents:
                raise GpuExhaustedError(
                    f"GPU {gpu_assignment} became occupied by another "
                    f"deployment (one-NIM-per-GPU invariant, §1.5). Pass "
                    f"replace_resident=true to reuse it. role={role}"
                )
        if background:
            return await _queue_local_nim_deploy(
                project_id=project_id,
                model_config_id=model_config_id,
                role=role,
                nim_container_image=nim_container_image,
                gpu_assignment=gpu_assignment,
                gpu_memory_minimum_gb=gpu_memory_minimum_gb,
                preferred_port=preferred_port,
                settings=settings,
                workspace_root=workspace_root,
                student_model_id=student_model_id,
                checkpoint_mount=checkpoint_mount,
                nim_model_name_path=nim_model_name_path,
                nim_served_model_name=nim_served_model_name,
                precision_method=precision_method,
                replace_resident=replace_resident,
                activate_on_success=activate_on_success,
            )
        return await _deploy_local_nim_impl(
            project_id=project_id,
            model_config_id=model_config_id,
            role=role,
            nim_container_image=nim_container_image,
            gpu_assignment=gpu_assignment,
            gpu_memory_minimum_gb=gpu_memory_minimum_gb,
            preferred_port=preferred_port,
            settings=settings,
            workspace_root=workspace_root,
            student_model_id=student_model_id,
            checkpoint_mount=checkpoint_mount,
            nim_model_name_path=nim_model_name_path,
            nim_served_model_name=nim_served_model_name,
            precision_method=precision_method,
            replace_resident=replace_resident,
            activate_on_success=activate_on_success,
        )


def _update_reserved_deployment(
    engine: Any,
    deployment_id: str,
    **values: Any,
) -> tuple[bool, LocalNimDeployment | None]:
    """Update a queued deployment only while it still owns ``starting``.

    The status guard prevents a background startup from resurrecting a
    deployment the user stopped while preflight or an image pull was in
    progress. The detached row is returned even when no update was applied.
    """
    with Session(engine) as session:
        deployment = session.get(LocalNimDeployment, deployment_id)
        updated = deployment is not None and deployment.status == "starting"
        if updated and deployment is not None:
            for key, value in values.items():
                setattr(deployment, key, value)
            session.commit()
            session.refresh(deployment)
        if deployment is not None:
            session.expunge(deployment)
        return updated, deployment


async def _queue_local_nim_deploy(
    project_id: str,
    model_config_id: str,
    role: str,
    nim_container_image: str,
    gpu_assignment: str,
    gpu_memory_minimum_gb: int,
    preferred_port: int,
    settings: Settings,
    workspace_root: str,
    *,
    student_model_id: str | None = None,
    checkpoint_mount: str | None = None,
    nim_model_name_path: str | None = None,
    nim_served_model_name: str | None = None,
    precision_method: str | None = None,
    replace_resident: bool = False,
    activate_on_success: bool = False,
) -> dict[str, Any]:
    """Persist a GPU reservation and schedule the slow deployment workflow."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        raise ValueError(f"Project {project_id} not found")

    deployment_id = generate_uuid4()
    displaced: list[LocalNimDeployment] = []
    if replace_resident:
        # Stop the old resident before inserting the new reservation so the
        # scan inside stop_gpu_residents cannot select the reservation itself.
        displaced = await stop_gpu_residents(
            workspace_root=workspace_root,
            device_index=gpu_assignment,
            displaced_by_deployment_id=deployment_id,
            settings=settings,
        )

    host_port = await _resolve_port(
        preferred_port,
        scan_active_host_ports(workspace_root),
    )
    container_name = build_container_name(role, project_id, student_model_id)
    endpoint_url = _build_endpoint_url(host_port)
    deployment = LocalNimDeployment(
        local_nim_deployment_id=deployment_id,
        project_id=project_id,
        model_config_id=model_config_id,
        role=role,
        nim_container_image=nim_container_image,
        container_name=container_name,
        host_port=host_port,
        endpoint_url=endpoint_url,
        gpu_assignment=gpu_assignment,
        status="starting",
        status_reason="Preflight is running in the background",
        activate_on_success=activate_on_success,
        student_model_id=student_model_id,
        checkpoint_mount_path=checkpoint_mount,
        nim_served_model_name=nim_served_model_name,
        nim_model_name_path=nim_model_name_path,
        precision_method=precision_method,
    )
    with Session(engine) as session:
        session.add(deployment)
        session.commit()
        session.refresh(deployment)
        session.expunge(deployment)

    nim_model_size: str | None = None
    nim_model_profile: str | None = None
    shared_image_served_name: str | None = None
    if role in ("teacher", "student"):
        (
            nim_model_size,
            nim_model_profile,
            shared_image_served_name,
        ) = _resolve_runtime_deploy_env(
            engine,
            model_config_id,
            role=role,
            custom_checkpoint=bool(checkpoint_mount and nim_model_name_path),
        )
    queued_preflight = PreflightResult(
        all_passed=True,
        checks=[
            PreflightCheckResult(
                check_name="deployment_queued",
                passed=True,
                diagnostic=(
                    "Deployment accepted; preflight and container startup "
                    "are running in the background."
                ),
            )
        ],
        docker_run_command=docker_run_command_display(
            nim_container_image=nim_container_image,
            container_name=container_name,
            gpu_assignment=gpu_assignment,
            host_port=host_port,
            role=role,
            checkpoint_mount=checkpoint_mount,
            nim_model_name_path=nim_model_name_path,
            nim_served_model_name=(
                shared_image_served_name if role == "teacher" else nim_served_model_name
            ),
            max_images_per_request=(
                _resolve_deployment_image_cap(engine, model_config_id)
                if role in ("teacher", "student")
                else None
            ),
            nim_model_size=nim_model_size,
            nim_model_profile=nim_model_profile,
            extra_env=_resolve_extra_container_env(engine, model_config_id),
        ),
    )

    task_id = f"local-nim-deploy-{deployment_id}"
    coro = _run_queued_local_nim_deploy(
        project_id=project_id,
        model_config_id=model_config_id,
        role=role,
        nim_container_image=nim_container_image,
        gpu_assignment=gpu_assignment,
        gpu_memory_minimum_gb=gpu_memory_minimum_gb,
        preferred_port=preferred_port,
        settings=settings,
        workspace_root=workspace_root,
        student_model_id=student_model_id,
        checkpoint_mount=checkpoint_mount,
        nim_model_name_path=nim_model_name_path,
        nim_served_model_name=nim_served_model_name,
        precision_method=precision_method,
        deployment_id=deployment_id,
        host_port=host_port,
        displaced=displaced,
        activate_on_success=activate_on_success,
    )
    try:
        background_manager.register(task_id, coro)
    except RuntimeError as exc:
        coro.close()
        _updated, failed = _update_reserved_deployment(
            engine,
            deployment_id,
            status="failed",
            status_reason=f"Could not start background deployment task: {exc}",
        )
        if failed is not None:
            deployment = failed
        queued_preflight = PreflightResult(
            all_passed=False,
            checks=[
                PreflightCheckResult(
                    check_name="background_start",
                    passed=False,
                    diagnostic="Could not start the background deployment task.",
                )
            ],
            docker_run_command=queued_preflight.docker_run_command,
        )
        if displaced:
            await restore_displaced_deployments(
                displaced,
                workspace_root=workspace_root,
                settings=settings,
            )

    return {
        "deployment": deployment,
        "preflight": queued_preflight,
        "displaced": displaced,
    }


async def _run_queued_local_nim_deploy(
    *,
    project_id: str,
    model_config_id: str,
    role: str,
    nim_container_image: str,
    gpu_assignment: str,
    gpu_memory_minimum_gb: int,
    preferred_port: int,
    settings: Settings,
    workspace_root: str,
    student_model_id: str | None,
    checkpoint_mount: str | None,
    nim_model_name_path: str | None,
    nim_served_model_name: str | None,
    precision_method: str | None,
    deployment_id: str,
    host_port: int,
    displaced: list[LocalNimDeployment],
    activate_on_success: bool,
) -> None:
    """Run queued preflight/startup and restore residents on failure."""
    try:
        result = await _deploy_local_nim_impl(
            project_id=project_id,
            model_config_id=model_config_id,
            role=role,
            nim_container_image=nim_container_image,
            gpu_assignment=gpu_assignment,
            gpu_memory_minimum_gb=gpu_memory_minimum_gb,
            preferred_port=preferred_port,
            settings=settings,
            workspace_root=workspace_root,
            student_model_id=student_model_id,
            checkpoint_mount=checkpoint_mount,
            nim_model_name_path=nim_model_name_path,
            nim_served_model_name=nim_served_model_name,
            precision_method=precision_method,
            reserved_deployment_id=deployment_id,
            reserved_host_port=host_port,
            pre_displaced=displaced,
            activate_on_success=activate_on_success,
        )
        deployment = result["deployment"]
        if deployment.status == "failed" and displaced:
            await restore_displaced_deployments(
                displaced,
                workspace_root=workspace_root,
                settings=settings,
            )
    except Exception as exc:
        logger.exception(
            "Queued local NIM deployment failed for %s/%s (%s)",
            project_id,
            deployment_id,
            type(exc).__name__,
        )
        engine = get_project_engine(project_id, workspace_root)
        if engine is not None:
            _update_reserved_deployment(
                engine,
                deployment_id,
                status="failed",
                status_reason=(
                    f"Background deployment failed: {str(exc) or type(exc).__name__}"
                ),
            )
        if displaced:
            await restore_displaced_deployments(
                displaced,
                workspace_root=workspace_root,
                settings=settings,
            )


async def restore_displaced_deployments(
    displaced: list[LocalNimDeployment],
    *,
    workspace_root: str,
    settings: Settings,
) -> None:
    """Best-effort background restore after a failed Teacher replacement.

    Replacement must stop the resident before its model-profile preflight can
    inspect genuinely free GPU memory. If that preflight, container start, or
    health verification fails, leaving the GPU empty is a destructive
    surprise. Requeue each displaced Teacher/embedding with its original
    project, model, image, GPU, and preferred port. The original stopped row
    retains its displacement audit; the restoration gets a fresh deployment
    row and follows the normal verification lifecycle.
    """

    for row in displaced:
        if row.role not in ("teacher", "embedding"):
            continue
        try:
            params = resolve_deploy_params(
                project_id=str(row.project_id),
                role=str(row.role),
                model_config_id=str(row.model_config_id),
                nim_container_image=str(row.nim_container_image),
                gpu_assignment=str(row.gpu_assignment),
                preferred_port=int(row.host_port),
                settings=settings,
            )
            await deploy_local_nim(
                project_id=str(row.project_id),
                model_config_id=params["model_config_id"],
                role=str(row.role),
                nim_container_image=params["nim_container_image"],
                gpu_assignment=params["gpu_assignment"],
                gpu_memory_minimum_gb=params["gpu_memory_minimum_gb"],
                preferred_port=params["preferred_port"],
                settings=settings,
                workspace_root=workspace_root,
                replace_resident=False,
                activate_on_success=False,
                background=True,
            )
            logger.info(
                "Queued restoration of displaced %s/%s on %s",
                row.project_id,
                row.local_nim_deployment_id,
                row.gpu_assignment,
            )
        except Exception as exc:
            logger.warning(
                "Could not restore displaced %s/%s (%s: %s)",
                row.project_id,
                row.local_nim_deployment_id,
                type(exc).__name__,
                str(exc) or "(no message)",
            )


def find_deployments_displaced_by(
    workspace_root: str, deployment_id: str
) -> list[LocalNimDeployment]:
    """Find durable residents stopped by one replacement deployment."""

    rows: list[LocalNimDeployment] = []
    root = projects_root(workspace_root)
    if not root.exists():
        return rows
    for entry in sorted(root.iterdir()):
        if (
            not entry.is_dir()
            or (entry / ".archived").exists()
            or not (entry / "project.db").exists()
        ):
            continue
        try:
            engine = get_project_engine(entry.name, workspace_root)
            if engine is None:
                continue
            with Session(engine) as session:
                matches = list(
                    session.execute(
                        select(LocalNimDeployment).where(
                            LocalNimDeployment.displaced_by_deployment_id
                            == deployment_id
                        )
                    )
                    .scalars()
                    .all()
                )
                for row in matches:
                    session.expunge(row)
                rows.extend(matches)
        except Exception as exc:
            logger.warning(
                "Skipping project %s in displacement restore scan (%s: %s)",
                entry.name,
                type(exc).__name__,
                str(exc) or "(no message)",
            )
    return rows


async def _deploy_local_nim_impl(
    project_id: str,
    model_config_id: str,
    role: str,
    nim_container_image: str,
    gpu_assignment: str,
    gpu_memory_minimum_gb: int,
    preferred_port: int,
    settings: Settings,
    workspace_root: str,
    *,
    student_model_id: str | None = None,
    checkpoint_mount: str | None = None,
    nim_model_name_path: str | None = None,
    nim_served_model_name: str | None = None,
    precision_method: str | None = None,
    replace_resident: bool = False,
    reserved_deployment_id: str | None = None,
    reserved_host_port: int | None = None,
    pre_displaced: list[LocalNimDeployment] | None = None,
    activate_on_success: bool = False,
) -> dict[str, Any]:
    """Deploy a local NIM container.

    1. If ``replace_resident=True``: stop any active resident on the
       target GPU first (one-NIM-per-GPU
       invariant). Idempotent — no-op when the GPU is free.
    2. Run preflight checks. (Re-runs AFTER step 1 so the GPU-memory
       check reflects post-displacement free memory.)
    3. If preflight fails: create LocalNimDeployment(status="failed"), return.
    4. If passes: resolve port, start container, create record, kick off
       background health poll.

    Student-specific kwargs (``student_model_id``, ``checkpoint_mount``,
    ``nim_model_name_path``, ``nim_served_model_name``, ``precision_method``)
    are persisted onto the LocalNimDeployment row when ``role == "student"``
    and threaded into the docker run command. They are ignored for
    Teacher / embedding deployments.

    Returns ``{"deployment": LocalNimDeployment, "preflight": PreflightResult,
    "displaced": [LocalNimDeployment, ...]}``. The ``displaced`` list is
    empty when ``replace_resident=False`` or when the GPU was already
    free.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        raise ValueError(f"Project {project_id} not found")

    # Pre-generate the new deployment_id so we can stamp the displaced
    # rows' displaced_by_deployment_id before the new container starts.
    deployment_id = reserved_deployment_id or generate_uuid4()

    # Step 1: replace semantics.
    displaced: list[LocalNimDeployment] = list(pre_displaced or [])
    if replace_resident and reserved_deployment_id is None:
        displaced = await stop_gpu_residents(
            workspace_root=workspace_root,
            device_index=gpu_assignment,
            displaced_by_deployment_id=deployment_id,
            settings=settings,
        )

    # Resolve shared-image selectors before preflight. The profile probe must
    # inspect the exact requested size/profile, not the image's default model.
    nim_model_size: str | None = None
    nim_model_profile: str | None = None
    shared_image_served_name: str | None = None
    if role in ("teacher", "student"):
        (
            nim_model_size,
            nim_model_profile,
            shared_image_served_name,
        ) = _resolve_runtime_deploy_env(
            engine,
            model_config_id,
            role=role,
            custom_checkpoint=bool(checkpoint_mount and nim_model_name_path),
        )

    # Run preflight
    preflight = await run_preflight_checks(
        nim_container_image=nim_container_image,
        gpu_memory_minimum_gb=gpu_memory_minimum_gb,
        gpu_assignment=gpu_assignment,
        role=role,
        settings=settings,
        gpu_compute_capability_minimum=(
            _resolve_deployment_compute_capability_floor(engine, model_config_id)
            if role in ("teacher", "student")
            else None
        ),
        nim_model_size=nim_model_size,
        nim_model_profile=nim_model_profile,
        nim_served_model_name=shared_image_served_name,
    )

    if not preflight.all_passed:
        # Create a failed deployment record (deployment_id pre-generated
        # above so step-1 displacement audit links match this row's id).
        container_name = build_container_name(role, project_id, student_model_id)
        failure_reason = "; ".join(
            c.diagnostic for c in preflight.checks if not c.passed
        )
        if reserved_deployment_id is not None:
            _updated, deployment = _update_reserved_deployment(
                engine,
                deployment_id,
                status="failed",
                status_reason=failure_reason,
                activate_on_success=activate_on_success,
            )
            if deployment is None:
                raise RuntimeError(f"Reserved deployment {deployment_id} disappeared")
        else:
            deployment = LocalNimDeployment(
                local_nim_deployment_id=deployment_id,
                project_id=project_id,
                model_config_id=model_config_id,
                role=role,
                nim_container_image=nim_container_image,
                container_name=container_name,
                host_port=preferred_port,
                endpoint_url=_build_endpoint_url(preferred_port),
                gpu_assignment=gpu_assignment,
                status="failed",
                status_reason=failure_reason,
                activate_on_success=activate_on_success,
                student_model_id=student_model_id,
                checkpoint_mount_path=checkpoint_mount,
                nim_served_model_name=nim_served_model_name,
                nim_model_name_path=nim_model_name_path,
                precision_method=precision_method,
            )
            with Session(engine) as session:
                session.add(deployment)
                session.commit()
                session.refresh(deployment)
                session.expunge(deployment)

        logger.warning(
            "Local NIM preflight failed for %s/%s: %s",
            project_id,
            role,
            deployment.status_reason,
        )
        return {
            "deployment": deployment,
            "preflight": preflight,
            "displaced": displaced,
        }

    # Preflight passed — resolve port and start container.
    # deployment_id pre-generated at function entry (see step 1 above).
    host_port = reserved_host_port or await _resolve_port(preferred_port)
    container_name = build_container_name(role, project_id, student_model_id)
    endpoint_url = _build_endpoint_url(host_port)

    if reserved_deployment_id is not None:
        with Session(engine) as session:
            reserved = session.get(LocalNimDeployment, deployment_id)
            if reserved is None:
                raise RuntimeError(f"Reserved deployment {deployment_id} disappeared")
            if reserved.status != "starting":
                session.expunge(reserved)
                return {
                    "deployment": reserved,
                    "preflight": preflight,
                    "displaced": displaced,
                }

    # Test seam (closing-smoke mock-mode integration): when
    # ``LOCAL_NIM_MOCK_ENDPOINT_URL`` is set and ``role == "student"``, the
    # docker subprocess and pre-cleanup are skipped and the deployment
    # endpoint_url is overridden to the mock URL. Health polling, smoke
    # inference, evaluation, and benchmark all then run against the
    # in-process MockNIMServer fixture. Production code is unaffected when
    # the env var is unset (the standard path runs verbatim). This hook
    # exists ONLY for closing-smoke mock validation.
    mock_endpoint_url = os.environ.get("LOCAL_NIM_MOCK_ENDPOINT_URL")
    use_mock = bool(mock_endpoint_url) and role == "student"
    if use_mock and mock_endpoint_url is not None:
        endpoint_url = mock_endpoint_url
        # Keep docker_args computed below for evidence; no subprocess runs.
    else:
        # Stop any existing container with the same name (idempotent cleanup)
        await run_subprocess("docker", "rm", "-f", container_name, timeout_s=15.0)

    # Resolve the per-prompt image cap for VLM roles so the NIM is started
    # with NIM_MAX_IMAGES_PER_PROMPT matching the backend's effective
    # ``max_images_per_request`` (endpoint override > model_config). Without
    # this, a cosmos ``:1.7.0`` deploy keeps the profile default of 5 and
    # silently 400s the moment ICL sends a 6th image. Embedding NIMs take a
    # single image, so the cap is moot for them.
    image_cap: int | None = (
        _resolve_deployment_image_cap(engine, model_config_id)
        if role in ("teacher", "student")
        else None
    )

    # Build and execute docker run — the runtime secret override wins
    # so a freshly-applied NGC key powers the deploy.
    docker_args = _build_docker_run_command(
        nim_container_image=nim_container_image,
        container_name=container_name,
        gpu_assignment=gpu_assignment,
        host_port=host_port,
        role=role,
        checkpoint_mount=checkpoint_mount,
        nim_model_name_path=nim_model_name_path,
        # Teacher shared-image deploys serve under the requested catalog name;
        # student deploys keep their custom ``student-<id>`` served name.
        nim_served_model_name=(
            shared_image_served_name if role == "teacher" else nim_served_model_name
        ),
        max_images_per_request=image_cap,
        nim_model_size=nim_model_size,
        nim_model_profile=nim_model_profile,
        extra_env=_resolve_extra_container_env(engine, model_config_id),
    )

    logger.info(
        "Starting local NIM container: %s (role=%s, gpu=%s, port=%d, mock=%s)",
        container_name,
        role,
        gpu_assignment,
        host_port,
        use_mock,
    )

    if use_mock:
        rc, stdout, stderr = 0, f"mock-{deployment_id[:12]}", ""
    else:
        try:
            _prepare_declared_user_cache(nim_container_image)
        except OSError as exc:
            rc, stdout, stderr = (
                1,
                "",
                f"Unable to prepare the NIM cache for the container user: {exc}",
            )
        else:
            rc, stdout, stderr = await run_subprocess(
                *docker_args,
                timeout_s=120.0,
                secret_env={
                    "NGC_API_KEY": (get_effective_secret("NGC_API_KEY", settings) or "")
                },
            )

    if rc != 0:
        failure_reason = f"docker run failed: {stderr or stdout or 'unknown error'}"
        if reserved_deployment_id is not None:
            _updated, deployment = _update_reserved_deployment(
                engine,
                deployment_id,
                status="failed",
                status_reason=failure_reason,
            )
            if deployment is None:
                raise RuntimeError(f"Reserved deployment {deployment_id} disappeared")
        else:
            deployment = LocalNimDeployment(
                local_nim_deployment_id=deployment_id,
                project_id=project_id,
                model_config_id=model_config_id,
                role=role,
                nim_container_image=nim_container_image,
                container_name=container_name,
                host_port=host_port,
                endpoint_url=endpoint_url,
                gpu_assignment=gpu_assignment,
                status="failed",
                status_reason=failure_reason,
                activate_on_success=activate_on_success,
                student_model_id=student_model_id,
                checkpoint_mount_path=checkpoint_mount,
                nim_served_model_name=nim_served_model_name,
                nim_model_name_path=nim_model_name_path,
                precision_method=precision_method,
            )
            with Session(engine) as session:
                session.add(deployment)
                session.commit()
                session.refresh(deployment)
                session.expunge(deployment)
        return {
            "deployment": deployment,
            "preflight": preflight,
            "displaced": displaced,
        }

    # Container started — create record with status="starting"
    container_id = stdout.strip()[:12] if stdout else None
    if reserved_deployment_id is not None:
        updated, deployment = _update_reserved_deployment(
            engine,
            deployment_id,
            container_id=container_id,
            host_port=host_port,
            endpoint_url=endpoint_url,
            status_reason=None,
        )
        if deployment is None:
            await _teardown_deployment_container(container_name, container_id)
            raise RuntimeError(f"Reserved deployment {deployment_id} disappeared")
        if not updated:
            # The user stopped the queued deployment while docker run was in
            # flight. Remove the just-started container and leave the terminal
            # database state untouched.
            await _teardown_deployment_container(container_name, container_id)
            return {
                "deployment": deployment,
                "preflight": preflight,
                "displaced": displaced,
            }
    else:
        deployment = LocalNimDeployment(
            local_nim_deployment_id=deployment_id,
            project_id=project_id,
            model_config_id=model_config_id,
            role=role,
            nim_container_image=nim_container_image,
            container_name=container_name,
            container_id=container_id,
            host_port=host_port,
            endpoint_url=endpoint_url,
            gpu_assignment=gpu_assignment,
            status="starting",
            activate_on_success=activate_on_success,
            student_model_id=student_model_id,
            checkpoint_mount_path=checkpoint_mount,
            nim_served_model_name=nim_served_model_name,
            nim_model_name_path=nim_model_name_path,
            precision_method=precision_method,
        )
        with Session(engine) as session:
            session.add(deployment)
            session.commit()
            session.refresh(deployment)
            session.expunge(deployment)

    logger.info(
        "Container %s started (id=%s). Polling health at %s",
        container_name,
        container_id,
        endpoint_url,
    )

    # Kick off background health polling
    task_id = f"local-nim-health-{deployment_id}"
    coro = _poll_health(
        deployment_id=deployment_id,
        endpoint_url=endpoint_url,
        timeout_s=settings.NIM_STARTUP_TIMEOUT_S,
        project_id=project_id,
        model_config_id=model_config_id,
        role=role,
        workspace_root=workspace_root,
        settings=settings,
    )
    try:
        background_manager.register(task_id, coro)
    except RuntimeError:
        coro.close()

    return {"deployment": deployment, "preflight": preflight, "displaced": displaced}


# ── Health polling ────────────────────────────────────────────────────────────


async def _container_startup_liveness(
    container_name: str,
) -> tuple[str, int | None, str]:
    """Classify a starting container as running / exited / missing.

    Returns ``(state, exit_code, log_tail)``. ``state`` is ``"running"``
    (keep polling), ``"exited"`` (dead — exit code available), or
    ``"missing"`` (removed out-of-band). ``log_tail`` carries an actionable
    root-cause line plus the last few shutdown lines for dead containers; empty
    when running. Inspect errors other than not-found are treated as
    ``"running"`` — a transient docker CLI hiccup must not fail a healthy
    startup.
    """
    rc, stdout, stderr = await run_subprocess(
        "docker",
        "inspect",
        "-f",
        "{{.State.Running}} {{.State.ExitCode}}",
        container_name,
        timeout_s=10.0,
    )
    if rc != 0:
        if "no such" in (stderr or "").lower():
            return "missing", None, ""
        return "running", None, ""
    parts = stdout.strip().split()
    if not parts:
        return "running", None, ""
    if parts[0].lower() == "true":
        return "running", None, ""
    exit_code = (
        int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None
    )
    _, logs_out, logs_err = await run_subprocess(
        "docker", "logs", "--tail", "200", container_name, timeout_s=10.0
    )
    tail_lines = ((logs_err or "") + "\n" + (logs_out or "")).strip().splitlines()
    normalized = [line.strip() for line in tail_lines if line.strip()]
    root_cause = next(
        (
            line
            for line in reversed(normalized)
            if re.search(
                r"(?:ValueError:|CUDA out of memory|out of memory|NIM_MODEL_|No compatible profile)",
                line,
                re.IGNORECASE,
            )
        ),
        None,
    )
    final_lines = normalized[-3:]
    if root_cause is not None and root_cause not in final_lines:
        final_lines.insert(0, root_cause)
    log_tail = " | ".join(final_lines)
    return "exited", exit_code, log_tail[:600]


async def _commit_with_lock_retry(
    fn: Callable[[], None],
    *,
    what: str,
    attempts: int = 4,
    delay_s: float = 1.5,
) -> None:
    """Run a short read-modify-write transaction, retrying lock contention.

    SQLite WAL's ``busy_timeout`` does NOT cover snapshot-upgrade
    conflicts: a session that reads first and then writes fails
    immediately with ``database is locked`` when any other writer
    committed between its read and its write — regardless of the pragma.
    Under write-heavy phases (TAO polling + chain advancement + lifecycle
    transitions all on one project DB) that race killed a health-poll
    task mid-commit and a perfectly healthy NIM was reported failed
    (observed live 2026-07-15). Each retry calls ``fn`` again with
    a FRESH session so guards re-read current state (e.g. the
    only-a-starting-deployment-may-promote check stays correct).
    Non-lock errors and lock errors past the final attempt re-raise.
    """
    for attempt in range(1, attempts + 1):
        try:
            fn()
            return
        except OperationalError as exc:
            if "database is locked" not in str(exc) or attempt == attempts:
                raise
            logger.warning(
                "SQLite lock contention on %s (attempt %d/%d) — retrying",
                what,
                attempt,
                attempts,
            )
            await asyncio.sleep(delay_s)


async def _teardown_deployment_container(
    container_name: str,
    container_id: str | None,
) -> None:
    """Best-effort stop + remove of a terminal deployment's container.

    A deployment row in a terminal state is invisible to the
    one-NIM-per-GPU placement scan (only ``starting``/``running`` rows
    count as residents), so every transition to ``failed``/``stopped``
    for a deployment whose container may still exist must tear the
    container down — otherwise a live NIM keeps holding the GPU's VRAM
    and port while the placement math hands the GPU out as free.

    Targets the docker id recorded at start time when available:
    container names are project+role scoped and reused across deployment
    generations, so a name-based stop could hit a successor container.
    Errors are ignored — the container may already have exited or been
    removed.
    """
    ref = container_id or container_name
    await run_subprocess("docker", "stop", ref, timeout_s=30.0)
    await run_subprocess("docker", "rm", "-f", ref, timeout_s=15.0)


# Deadline for the post-health inference probe. Generous relative to the
# ~1-token completion it requests because a first completion on a freshly
# ready NIM can queue behind warmup; a probe timeout is treated as
# inconclusive-PASS, so this bound only caps how long adoption waits.
INFERENCE_PROBE_DEADLINE_S = 60.0


async def _probe_inference_ready(
    *,
    endpoint_url: str,
    model_name: str,
) -> tuple[bool, str | None]:
    """One minimal real completion — health/ready 200 is not proof of life.

    A NIM whose vLLM EngineCore has died keeps serving
    ``/v1/health/ready``, ``/v1/models`` and ``/v1/metadata`` 200s from
    the surviving HTTP front-end (observed live: CUDA
    illegal-memory-access engine death under load), so adoption — the
    startup health poll and the restart-recovery rebind — must exercise
    the inference path before trusting the container. Returns
    ``(ok, failure_detail)``.

    Only an affirmative endpoint-level failure (HTTP error, connection
    refusal) fails the probe. A timeout is an inconclusive PASS: a busy
    but healthy NIM queues completions for tens of seconds under load,
    and a probe must never tear down a serving deployment on slowness
    alone. A missing model name skips the probe — never fail on
    inconclusiveness (mirrors ``verify_served_model``).
    """
    if not model_name:
        return (True, None)
    url = f"{endpoint_url.rstrip('/')}/chat/completions"
    result = await resilient_request(
        "POST",
        url,
        deadline_s=INFERENCE_PROBE_DEADLINE_S,
        max_retries=1,
        headers=dict(NIM_DEFAULT_HEADERS),
        json_body={
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )
    if result.status_code == 200:
        return (True, None)
    if result.error_class == "timeout":
        logger.warning(
            "Inference probe against %s timed out — treating as inconclusive "
            "(a busy NIM queues completions under load)",
            url,
        )
        return (True, None)
    detail = result.error_detail or f"HTTP {result.status_code}"
    return (False, detail)


async def _fail_active_deployment(
    *,
    deployment_id: str,
    project_id: str,
    workspace_root: str,
    endpoint_url: str,
    role: str,
    reason: str,
    settings: Settings,
    from_statuses: tuple[str, ...] = ("starting",),
) -> None:
    """Transition a deployment still in one of *from_statuses* to ``failed``.

    Only fails a deployment still in *from_statuses*; a concurrent stop
    may already have moved it to "stopped"/"failed". The embedding-config
    reset and the container teardown are gated on the transition actually
    happening here: when a concurrent stop already moved the row, that
    stop path owns both (and a replacement deployment may have re-stamped
    the config / reused the container name since).
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return
    transitioned = False
    container_name: str | None = None
    container_id: str | None = None

    def _mark_failed() -> None:
        nonlocal transitioned, container_name, container_id
        with Session(engine) as session:
            dep = session.get(LocalNimDeployment, deployment_id)
            if dep and dep.status in from_statuses:
                dep.status = "failed"
                dep.status_reason = reason
                container_name = dep.container_name
                container_id = dep.container_id
                session.commit()
                transitioned = True

    # Same lock-contention retry as the promote path — a failure
    # transition lost to a lock leaves the row active forever.
    await _commit_with_lock_retry(
        _mark_failed, what=f"deployment {deployment_id} mark-failed"
    )
    # The failed row no longer counts as a GPU resident, so a container
    # that survived the failure (health timeout) must not keep the GPU;
    # for the container-death callers this is a harmless no-op cleanup.
    if transitioned and container_name:
        await _teardown_deployment_container(container_name, container_id)
    # A reset resweeps immediately so open projects flip to the fallback
    # provider instead of waiting for the next worker trigger.
    if (
        role == "embedding"
        and transitioned
        and _reset_embedding_deployment_config(workspace_root, endpoint_url)
    ):
        await resweep_embedding_tasks(settings)
    if role == "teacher" and transitioned:
        disable_teacher_resident_endpoints(workspace_root, deployment_id)
    if transitioned:
        displaced = find_deployments_displaced_by(workspace_root, deployment_id)
        if displaced:
            await restore_displaced_deployments(
                displaced,
                workspace_root=workspace_root,
                settings=settings,
            )


async def _poll_health(
    deployment_id: str,
    endpoint_url: str,
    timeout_s: int,
    project_id: str,
    model_config_id: str,
    role: str,
    workspace_root: str,
    settings: Settings,
) -> None:
    """Poll ``/v1/health/ready`` until healthy, container death, or timeout.

    On healthy: update deployment to "running", auto-register endpoint.
    On container death: update deployment to "failed" immediately with the
    exit code and a log tail — a NIM container can die during startup
    (engine-init crashes such as unbuildable attention kernels for the
    selected profile; the exit-0 cache-permission case documented on
    ``build_docker_run_args``) and health polling alone would otherwise
    hold the deployment in "starting" for the full NIM_STARTUP_TIMEOUT_S
    (default 20 min) with the GPU marked occupied and zero diagnostic.
    On timeout: update deployment to "failed".

    Every failure transition also best-effort stops+removes the container
    (``_fail_active_deployment``): a failed row no longer counts as a
    GPU resident, so its container must not keep holding the GPU.
    """
    import time

    poll_interval = 10
    # Measure elapsed on the wall clock, not by summing sleeps: each poll adds
    # up to deadline_s (10s) of request time, so a sleep-only counter under-
    # counts and lets this loop outlive the lifecycle's real-clock watcher —
    # the window in which a late "healthy" could resurrect a torn-down
    # deployment (guarded again in _on_healthy).
    start = time.monotonic()
    health_url = f"{endpoint_url}/health/ready"

    # Container name for liveness checks, read once from the deployment row
    # (committed before this task is registered).
    container_name: str | None = None
    engine = get_project_engine(project_id, workspace_root)
    if engine:
        with Session(engine) as session:
            dep = session.get(LocalNimDeployment, deployment_id)
            container_name = dep.container_name if dep else None

    while time.monotonic() - start < timeout_s:
        if background_manager.is_shutting_down():
            logger.info(
                "Shutdown requested — stopping health poll for %s", deployment_id
            )
            return

        result = await resilient_request(
            "GET",
            health_url,
            deadline_s=10.0,
            max_retries=1,
        )

        if result.status_code == 200:
            logger.info(
                "Local NIM %s is healthy after %.0fs",
                deployment_id,
                time.monotonic() - start,
            )
            await _on_healthy(
                deployment_id=deployment_id,
                project_id=project_id,
                model_config_id=model_config_id,
                role=role,
                endpoint_url=endpoint_url,
                workspace_root=workspace_root,
                settings=settings,
            )
            return

        if container_name:
            state, exit_code, log_tail = await _container_startup_liveness(
                container_name
            )
            if state != "running":
                reason = (
                    f"Container exited during startup (exit code {exit_code})"
                    if state == "exited"
                    else "Container removed during startup"
                )
                if log_tail:
                    reason = f"{reason}. Last log lines: {log_tail}"
                logger.warning(
                    "Local NIM %s container %s died during startup: %s",
                    deployment_id,
                    container_name,
                    reason,
                )
                await _fail_active_deployment(
                    deployment_id=deployment_id,
                    project_id=project_id,
                    workspace_root=workspace_root,
                    endpoint_url=endpoint_url,
                    role=role,
                    reason=reason,
                    settings=settings,
                )
                return

        await asyncio.sleep(poll_interval)

    # Timeout
    logger.warning(
        "Local NIM %s health check timed out after %ds",
        deployment_id,
        timeout_s,
    )
    await _fail_active_deployment(
        deployment_id=deployment_id,
        project_id=project_id,
        workspace_root=workspace_root,
        endpoint_url=endpoint_url,
        role=role,
        reason=f"Health check timed out after {timeout_s}s",
        settings=settings,
    )


async def _on_healthy(
    deployment_id: str,
    project_id: str,
    model_config_id: str,
    role: str,
    endpoint_url: str,
    workspace_root: str,
    settings: Settings,
    *,
    rebind_running: bool = False,
) -> None:
    """Handle a newly healthy local NIM deployment.

    Verifies the requested model is genuinely the one being served
    (anti-silent-fallback) and that the inference path actually answers
    (anti-dead-engine), then marks the deployment ``running`` and
    auto-registers the endpoint. Restart recovery may explicitly rebind an
    already-``running`` deployment after re-verifying its surviving container;
    normal health polling may only promote ``starting`` deployments. A
    verification or probe failure marks
    the deployment ``failed`` with a specific reason, stops the container,
    and does NOT auto-register — the operator is never silently handed a
    wrong or dead model.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return

    # ── Served-model verification (Teacher role only) ─────────────────────
    # The silent-fallback footgun (cosmos3-nano → super) is
    # a TEACHER concern: a stock NGC model is deployed by size, and when that
    # size's weights weren't fetched the NIM silently serves a cached sibling
    # under the requested name. Teacher weights come from the NGC cache and
    # the loaded model is named on /v1/metadata, so both verification signals
    # apply directly.
    #
    # The expected served identity is the catalog ``ModelConfig.model_name``
    # (size-specific even when several Cosmos 3 sizes share one image), NOT the
    # shared image slug — so a correct super deploy passes and a nano→super
    # fallback fails. (Keying on the image slug — ``cosmos3-reasoner`` — can
    # never match the size-specific ``/v1/metadata`` slug and would false-FAIL
    # a correct super deploy.)
    #
    # STUDENT deployments are deliberately NOT gated here: they mount a
    # fine-tuned checkpoint via NIM_MODEL_NAME (weights come from the mount,
    # not the NGC cache) and serve it under a custom ``student-<id>`` name, so
    # neither the cache-weight check nor the NGC-slug comparison applies — and
    # the lifecycle orchestrator already runs its own smoke-inference gate.
    # EMBEDDING NIMs serve a single image, use a different cache layout, and
    # have not exhibited the fallback; their healthy signal stays the port +
    # health-ready check.
    if role == "teacher":
        with Session(engine) as session:
            mc_for_verify = session.get(ModelConfig, model_config_id)
            expected_model_name = mc_for_verify.model_name if mc_for_verify else ""
        verification = await verify_served_model(
            expected_model_name=expected_model_name,
            endpoint_url=endpoint_url,
        )
        logger.info(
            "Served-model verification for %s: ok=%s expected=%s served=%s "
            "weights=%d profile=%s",
            deployment_id,
            verification.ok,
            verification.expected_slug,
            verification.served_slug,
            verification.weight_files_found,
            verification.selected_profile_id,
        )
        failure_reason: str | None = None
        if not verification.ok:
            failure_reason = verification.reason
        else:
            # Ready + verified identity still doesn't prove the engine is
            # alive — a dead EngineCore leaves the HTTP front-end serving
            # health/metadata 200s. Gate adoption (startup poll AND the
            # restart-recovery rebind, which both land here) on one real
            # completion.
            probe_ok, probe_detail = await _probe_inference_ready(
                endpoint_url=endpoint_url,
                model_name=expected_model_name,
            )
            if not probe_ok:
                failure_reason = (
                    f"Inference probe failed after health-ready: {probe_detail}. "
                    "The container answers health checks but cannot serve "
                    "inference (engine dead) — container stopped; redeploy "
                    "locally."
                )
        if failure_reason is not None:
            logger.error(
                "Local NIM %s FAILED post-health verification: %s",
                deployment_id,
                failure_reason,
            )
            # "running" is included because the restart-recovery rebind
            # re-verifies an already-running row (dead-engine re-adoption).
            # The status guard still skips rows a concurrent stop moved to a
            # terminal state during the verify/probe awaits — a user-stopped
            # deployment must not be re-failed or re-torn-down.
            await _fail_active_deployment(
                deployment_id=deployment_id,
                project_id=project_id,
                workspace_root=workspace_root,
                endpoint_url=endpoint_url,
                role=role,
                reason=failure_reason,
                settings=settings,
                from_statuses=("starting", "running"),
            )
            return

    ready_to_bind = False

    def _promote_or_adopt_running() -> None:
        nonlocal ready_to_bind
        with Session(engine) as session:
            dep = session.get(LocalNimDeployment, deployment_id)
            if dep is None:
                return
            # Only a deployment still in "starting" may transition to
            # "running". A benchmark/timeout watcher on another task can tear
            # this container down (status → "stopped"/"failed") while a
            # health poll is mid-flight; a late "healthy" must not resurrect
            # it into "running" and re-register a dead endpoint. See
            # _poll_health's wall-clock note.
            if dep.status == "starting":
                dep.status = "running"
                dep.deployed_at = utc_now()
                session.commit()
                ready_to_bind = True
                return
            if rebind_running and dep.status == "running":
                # Backend restart recovery has already health-checked,
                # identity-verified, and inference-probed the surviving
                # container. Keep the original deployment timestamp and
                # continue into the idempotent endpoint repair below.
                ready_to_bind = True
                return
            else:
                logger.info(
                    "Local NIM %s became healthy but is now %s — not "
                    "binding it (deployment was torn down concurrently)",
                    deployment_id,
                    dep.status,
                )
                return

    # Retry lock contention — a healthy NIM must not be lost to a
    # transient snapshot-upgrade conflict with a concurrent writer.
    await _commit_with_lock_retry(
        _promote_or_adopt_running,
        what=f"deployment {deployment_id} promote-or-adopt-running",
    )
    if not ready_to_bind:
        return

    # Auto-register endpoint. Recovery is idempotent: reuse and repair the
    # endpoint already stamped with this deployment identity instead of
    # appending one duplicate endpoint per backend restart.
    if role == "teacher":
        activate_requested = False
        nim_endpoint: NimEndpoint | None = None
        if rebind_running:
            with Session(engine) as session:
                stmt = (
                    select(NimEndpoint)
                    .where(
                        NimEndpoint.local_nim_deployment_id == deployment_id,
                        NimEndpoint.base_url == endpoint_url,
                    )
                    .order_by(NimEndpoint.created_at.desc())
                )
                existing_endpoint = session.execute(stmt).scalars().first()
                if existing_endpoint is not None:
                    existing_endpoint.is_enabled = True
                    existing_endpoint.last_probe_at = utc_now()
                    existing_endpoint.last_probe_status = "healthy"
                    existing_endpoint.last_probe_error_ref = None
                    session.commit()
                    session.refresh(existing_endpoint)
                    session.expunge(existing_endpoint)
                    nim_endpoint = existing_endpoint
        if nim_endpoint is None:
            nim_endpoint = await create_nim_endpoint(
                project_id=project_id,
                data={
                    "display_name": f"Local Teacher ({endpoint_url})",
                    "endpoint_mode": "local_system_managed",
                    "base_url": endpoint_url,
                    "auth_mode": "none",
                    "source_kind": "auto_registered_local",
                },
                workspace_root=workspace_root,
                settings=settings,
            )
        if nim_endpoint:
            with Session(engine) as session:
                # Re-verify the deployment is still running before repointing
                # the ModelConfig. create_nim_endpoint above is an await
                # boundary; a concurrent benchmark/timeout teardown could have
                # stopped the container in the meantime, and repointing the
                # loop's Teacher at a dead endpoint is the damaging outcome.
                # If it was torn down, leave the just-created endpoint
                # unlinked (harmless — no ModelConfig references it) and skip.
                dep2 = session.get(LocalNimDeployment, deployment_id)
                if dep2 is None or dep2.status != "running":
                    logger.info(
                        "Local NIM %s was torn down during endpoint "
                        "registration — not repointing ModelConfig %s",
                        deployment_id,
                        model_config_id,
                    )
                    return
                activate_requested = bool(dep2.activate_on_success)

                ep = session.get(NimEndpoint, nim_endpoint.endpoint_id)
                if ep:
                    ep.local_nim_deployment_id = deployment_id
                    session.commit()

                # Update ModelConfig to use the local endpoint
                mc = session.get(ModelConfig, model_config_id)
                if mc:
                    mc.endpoint_id = nim_endpoint.endpoint_id
                    session.commit()

            # Pin the endpoint's per-prompt image cap from the served NIM's
            # effective NIM_MAX_IMAGES_PER_PROMPT so deep ICL isn't silently
            # truncated. The deploy path started the container with
            # ``NIM_MAX_IMAGES_PER_PROMPT=<resolved cap>``; surface that same
            # value as the per-endpoint override (image_cap_resolver prefers
            # the endpoint value) so the backend's ICL pruner and the NIM
            # agree without the operator hand-setting it. Falls back to the
            # deploy-time resolved cap when the NIM exposes no live cap probe.
            await _auto_set_endpoint_image_cap(
                engine=engine,
                endpoint_id=nim_endpoint.endpoint_id,
                model_config_id=model_config_id,
            )

            # A prior generation of this exact shared Teacher may have been
            # displaced and disabled across consumer projects. Once the new
            # resident has passed health, served-model verification, and a
            # real inference probe, repair those still-selected attachments.
            reattach_selected_teacher_consumers(workspace_root, deployment_id)

            # NIM Configuration's model-change intent becomes active only
            # after every adoption gate above passed and the ModelConfig now
            # points at the verified local endpoint. FTUE hybrid deploys leave
            # this false so a background alternate never steals the active
            # Teacher.
            if activate_requested:
                activate_teacher_model_config(
                    project_id, model_config_id, workspace_root
                )

            logger.info(
                "Auto-registered local teacher endpoint %s for model %s",
                nim_endpoint.endpoint_id,
                model_config_id,
            )

    elif role == "embedding":
        # Update EmbeddingDeploymentConfig
        with Session(engine) as session:
            deployment = session.get(LocalNimDeployment, deployment_id)
            if deployment is None or deployment.status != "running":
                logger.info(
                    "Embedding NIM %s is no longer running — not stamping deployment config",
                    deployment_id,
                )
                return
            embedding_gpu_assignment = deployment.gpu_assignment
        deploy_engine = init_deployment_db(Path(workspace_root))
        with Session(deploy_engine) as session:
            stmt = select(EmbeddingDeploymentConfig).limit(1)
            config = session.execute(stmt).scalar_one_or_none()
            if config:
                config.provider = "self_hosted_nvclip"
                config.endpoint_url = endpoint_url
                config.gpu_assignment = embedding_gpu_assignment
                config.updated_at = utc_now()
                session.commit()

        logger.info(
            "Updated EmbeddingDeploymentConfig: provider=self_hosted_nvclip, endpoint=%s",
            endpoint_url,
        )

        # Flip open projects to the local provider and drain any pending
        # embedding work now — nothing else re-probes projects until the
        # next ingest or backend restart.
        await resweep_embedding_tasks(settings)

    elif role == "student":
        # Student deployments deliberately do NOT auto-register a
        # project-level NimEndpoint here: the lifecycle orchestrator
        # (services/student_nim_lifecycle.py) owns that step so it can
        # set ``is_enabled=False`` + ``source_kind="auto_registered_student"``
        # — keeping the temp endpoint out of the Teacher picker / model
        # config dropdowns. We just log the healthy transition.
        logger.info(
            "Local Student NIM container healthy (deployment %s, endpoint %s) — "
            "lifecycle orchestrator will register the temporary endpoint.",
            deployment_id,
            endpoint_url,
        )


async def _auto_set_endpoint_image_cap(
    *,
    engine: Any,
    endpoint_id: str,
    model_config_id: str,
) -> None:
    """Set ``NimEndpoint.max_images_per_request`` from the served NIM's cap.

    Without a per-endpoint override, the backend's ICL image-budget
    pruner falls back to ``ModelConfig.max_images_per_request`` (seeded
    conservatively, e.g. 5), so a deep-ICL request that the local NIM
    would happily accept gets silently truncated to the per-model
    default. The container was started with
    ``NIM_MAX_IMAGES_PER_PROMPT=<resolved cap>``; record that same cap
    as the endpoint override so the resolver (which prefers the endpoint
    value) lets the backend send exactly what the NIM accepts.

    The value is the deploy-time resolved cap
    (``_resolve_deployment_image_cap``), which is what the container was
    actually started with. NIM does not announce its effective
    ``NIM_MAX_IMAGES_PER_PROMPT`` on ``/v1/metadata`` or ``/v1/models``
    (the cap is enforced at inference, not advertised), so the value the
    container was launched with is the authoritative cap — there is no
    separate live value to read back.

    Idempotent: the value written equals what the NIM was launched with,
    so re-running on recovery is a no-op write.
    """
    cap = _resolve_deployment_image_cap(engine, model_config_id)
    if cap is None:
        return
    with Session(engine) as session:
        ep = session.get(NimEndpoint, endpoint_id)
        if ep is None:
            return
        ep.max_images_per_request = int(cap)
        ep.updated_at = utc_now()
        session.commit()
    logger.info(
        "Auto-set endpoint %s max_images_per_request=%d from served NIM cap",
        endpoint_id,
        int(cap),
    )


# ── Stop ──────────────────────────────────────────────────────────────────────


async def stop_local_nim(
    deployment_id: str,
    project_id: str,
    workspace_root: str,
    settings: Settings | None = None,
) -> LocalNimDeployment | None:
    """Stop a running local NIM container and update the record.

    When *settings* is provided and stopping an embedding NIM reset the
    deployment config, projects are reswept immediately — open
    projects flip to the fallback provider (hosted when a key is
    present) without waiting for the next worker trigger. ``None``
    keeps the stop side-effect-local (callers without a Settings in
    hand; the worker's self-heal re-probe still covers them).
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        dep = session.get(LocalNimDeployment, deployment_id)
        if dep is None:
            return None

        if dep.status in ("stopped", "failed"):
            # Already terminal — a failed deployment's container was torn
            # down on the failure transition. Re-issuing `docker stop` by
            # name here could hit a SUCCESSOR that reused the name, so
            # never touch docker for a terminal row.
            session.expunge(dep)
            return dep

        container_name = dep.container_name
        container_id = dep.container_id
        dep_role = dep.role
        dep_endpoint_url = dep.endpoint_url

    # Stop and remove the container. In mock mode (deploy
    # short-circuited), there's no real container to stop — skip the
    # docker subprocess so the mock-integration test doesn't need docker
    # installed.
    mock_endpoint_url = os.environ.get("LOCAL_NIM_MOCK_ENDPOINT_URL")
    if not (mock_endpoint_url and dep_role == "student"):
        await _teardown_deployment_container(container_name, container_id)

    stopped_dep: LocalNimDeployment | None = None
    with Session(engine) as session:
        dep = session.get(LocalNimDeployment, deployment_id)
        if dep:
            dep.status = "stopped"
            dep.status_reason = "user_stopped"
            dep.stopped_at = utc_now()
            session.commit()
            session.refresh(dep)
            session.expunge(dep)
            stopped_dep = dep

    # Every stop path funnels through here (user stop, displacement via
    # stop_gpu_residents), so the config stops advertising the dead
    # local endpoint the moment the container goes away — scoped to
    # this deployment's endpoint so a re-stamped successor survives.
    if stopped_dep is not None and dep_role == "embedding":
        did_reset = _reset_embedding_deployment_config(workspace_root, dep_endpoint_url)
        if did_reset and settings is not None:
            await resweep_embedding_tasks(settings)
    if stopped_dep is not None and dep_role == "teacher":
        disable_teacher_resident_endpoints(workspace_root, deployment_id)

    return stopped_dep


# ── CRUD ──────────────────────────────────────────────────────────────────────


def list_local_deployments(
    project_id: str,
    workspace_root: str,
) -> list[LocalNimDeployment]:
    """List all LocalNimDeployment records for a project."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return []

    with Session(engine) as session:
        stmt = (
            select(LocalNimDeployment)
            .where(LocalNimDeployment.project_id == project_id)
            .order_by(LocalNimDeployment.created_at)
        )
        deployments = list(session.execute(stmt).scalars().all())
        for dep in deployments:
            session.expunge(dep)
        return deployments


def matches_active_role_config(
    project_id: str,
    workspace_root: str,
    deployments: list[LocalNimDeployment],
) -> dict[str, bool]:
    """Map deployment_id → whether the project's ACTIVE config for the
    deployment's role still references the deployment's model config.

    A ``failed`` teacher deploy whose model config is no longer the
    project's active Teacher is stale evidence: the SME has since switched
    Teachers (hosted, self-hosted, or a different local model) and labeling
    works — the failure banner outlived its truth. The check is config
    IDENTITY, not endpoint mode, because on a fresh FTUE failure the
    ModelConfig is only repointed to the ``local_system_managed`` endpoint
    on deploy *success* — an endpoint-mode check would suppress exactly the
    fresh failure the banner exists for.

    Non-teacher roles (embedding is deployment-scoped, not a project
    config; student lifecycles manage their own status surface) always map
    True — never suppressed on this axis.
    """
    engine = get_project_engine(project_id, workspace_root)
    active_teacher_id: str | None = None
    if engine is not None:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            if project is not None:
                active_teacher_id = project.teacher_model_config_id
    return {
        dep.local_nim_deployment_id: (
            dep.role != "teacher" or dep.model_config_id == active_teacher_id
        )
        for dep in deployments
    }


def get_local_deployment(
    project_id: str,
    deployment_id: str,
    workspace_root: str,
) -> LocalNimDeployment | None:
    """Get a single LocalNimDeployment record."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        dep = session.get(LocalNimDeployment, deployment_id)
        if dep and dep.project_id == project_id:
            session.expunge(dep)
            return dep
    return None


# ── Restart recovery ──────────────────────────────────────────────────────────


async def recover_local_deployments(
    workspace_root: str,
    settings: Settings,
) -> None:
    """Recover local NIM deployments after backend restart.

    Inspects only containers whose names are persisted in
    LocalNimDeployment records — NO generic Docker orphan discovery.

    For each active deployment:
    - Running + health-ready → rebind endpoint (teacher rebinds are
      additionally gated on served-model verification + an inference
      probe; a probe failure marks the row failed and stops the
      container)
    - Starting + container running but not health-ready → resume the
      background health poll
    - Running but not health-ready → stop container, mark stopped
    - Not running → mark stopped
    """
    projects_dir = projects_root(workspace_root)
    if not projects_dir.exists():
        return

    recovery_batches: list[tuple[str, list[LocalNimDeployment]]] = []
    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir():
            continue
        db_path = entry / "project.db"
        if not db_path.exists():
            continue
        # Archived projects don't own live containers (busy gate ensures
        # no LocalNimDeployment is starting/running at archive time), so
        # there's nothing to recover.
        if (entry / ".archived").exists():
            continue

        project_id = entry.name
        # Per-project isolation: a single corrupt project DB MUST NOT
        # prevent backend startup. Un-isolated recovery lets one
        # project's drifted Alembic state ("table projects already
        # exists") propagate through the FastAPI lifespan and block
        # every healthy project from serving. Catch broadly here, log a
        # per-project warning with the exception class name (so
        # empty-str exceptions still produce actionable signal — see
        # services/background.py ``_on_task_done`` for the same
        # pattern), and continue.
        try:
            engine = get_project_engine(project_id, workspace_root)
            if engine is None:
                continue

            with Session(engine) as session:
                stmt = select(LocalNimDeployment).where(
                    LocalNimDeployment.status.in_(("starting", "running"))
                )
                active_deps = list(session.execute(stmt).scalars().all())
                for dep in active_deps:
                    session.expunge(dep)

            recovery_batches.append((project_id, active_deps))
        except Exception as exc:
            logger.warning(
                "Skipping local-NIM recovery for project %s (%s: %s)",
                project_id,
                type(exc).__name__,
                exc or "(no message)",
            )
            continue

    # Freeze the startup population before any recovery handler runs. A
    # Student can requeue a displaced Teacher or embedding owned by another
    # project; that fresh ``starting`` row belongs to its new background task
    # and must not be mistaken for pre-restart residue later in this sweep.
    for project_id, active_deps in recovery_batches:
        try:
            for dep in active_deps:
                await _recover_single_deployment(
                    dep, project_id, workspace_root, settings
                )
        except Exception as exc:
            logger.warning(
                "Skipping local-NIM recovery for project %s (%s: %s)",
                project_id,
                type(exc).__name__,
                exc or "(no message)",
            )
            continue


async def _recover_single_deployment(
    dep: LocalNimDeployment,
    project_id: str,
    workspace_root: str,
    settings: Settings,
) -> None:
    """Recover a single local NIM deployment.

    Student deployments are NEVER rebound: the in-process orchestrator
    state (current concurrency level, partial benchmark results) is lost
    on restart, so an honest "failed" status beats half-state. We remove the
    container, mark the row stopped, and flip the paired StudentModel
    serving_status to "failed" with failure_stage="interrupted_by_restart".
    If removal cannot be confirmed, the deployment row stays active to keep
    its GPU reserved. Only after confirmed teardown commits do we requeue
    residents durably displaced by that Student.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return

    # Student deployments: never rebind, always tear down.
    if dep.role == "student":
        removed = await _recover_student_deployment(dep, project_id, workspace_root)
        if not removed:
            return
        displaced = find_deployments_displaced_by(
            workspace_root, dep.local_nim_deployment_id
        )
        if displaced:
            await restore_displaced_deployments(
                displaced,
                workspace_root=workspace_root,
                settings=settings,
            )
        return

    # Inspect the container by its persisted name
    rc, stdout, _stderr = await run_subprocess(
        "docker",
        "inspect",
        "--format",
        "{{.State.Status}}",
        dep.container_name,
        timeout_s=10.0,
    )

    if rc == 0 and stdout.strip().lower() == "running":
        # Container is running — check health
        health_url = f"{dep.endpoint_url}/health/ready"
        result = await resilient_request(
            "GET",
            health_url,
            deadline_s=10.0,
            max_retries=1,
        )

        if result.status_code == 200:
            logger.info(
                "Recovery: container %s is running and healthy — rebinding",
                dep.container_name,
            )
            await _on_healthy(
                deployment_id=dep.local_nim_deployment_id,
                project_id=project_id,
                model_config_id=dep.model_config_id,
                role=dep.role,
                endpoint_url=dep.endpoint_url,
                workspace_root=workspace_root,
                settings=settings,
                rebind_running=True,
            )
        elif dep.status == "starting":
            # The deploy's in-process health watcher disappears with the
            # backend process, but the named NIM container keeps initializing.
            # Re-register the same authoritative poller instead of treating
            # one not-ready response as a terminal failure. Give the recovered
            # container a fresh configured startup window: the deployment row
            # predates docker start (it includes background preflight and image
            # pull time), so created_at cannot provide a truthful remaining
            # health deadline without a new persisted container-start field.
            logger.info(
                "Recovery: container %s is still starting — resuming health poll",
                dep.container_name,
            )
            task_id = f"local-nim-health-{dep.local_nim_deployment_id}"
            background_manager.try_register(
                task_id,
                _poll_health(
                    deployment_id=dep.local_nim_deployment_id,
                    endpoint_url=dep.endpoint_url,
                    timeout_s=settings.NIM_STARTUP_TIMEOUT_S,
                    project_id=project_id,
                    model_config_id=dep.model_config_id,
                    role=dep.role,
                    workspace_root=workspace_root,
                    settings=settings,
                ),
                no_loop_warning=(
                    "Could not resume local NIM health polling during "
                    f"restart recovery for {dep.local_nim_deployment_id}"
                ),
            )
        else:
            logger.warning(
                "Recovery: container %s is running but unhealthy — stopping",
                dep.container_name,
            )
            # Teardown before the status write: a stopped/failed row is
            # invisible to the one-NIM-per-GPU placement scan, so the
            # container must not outlive the transition as an unmanaged
            # GPU resident. Persisted ``starting`` containers take the branch
            # above and retain their GPU reservation while polling resumes.
            await _teardown_deployment_container(dep.container_name, dep.container_id)
            with Session(engine) as session:
                d = session.get(LocalNimDeployment, dep.local_nim_deployment_id)
                if d:
                    d.status = "stopped"
                    d.status_reason = (
                        "Container running but unhealthy after backend restart — "
                        "stopped; redeploy locally"
                    )
                    d.stopped_at = utc_now()
                    session.commit()
            if dep.role == "embedding":
                _reset_embedding_deployment_config(workspace_root, dep.endpoint_url)
            elif dep.role == "teacher":
                disable_teacher_resident_endpoints(
                    workspace_root, dep.local_nim_deployment_id
                )
            displaced = find_deployments_displaced_by(
                workspace_root, dep.local_nim_deployment_id
            )
            if displaced:
                await restore_displaced_deployments(
                    displaced,
                    workspace_root=workspace_root,
                    settings=settings,
                )
    else:
        # Container not found or not running
        logger.info(
            "Recovery: container %s not running — marking stopped",
            dep.container_name,
        )
        with Session(engine) as session:
            d = session.get(LocalNimDeployment, dep.local_nim_deployment_id)
            if d:
                d.status = "stopped"
                d.status_reason = "Container not running after backend restart"
                d.stopped_at = utc_now()
                session.commit()
        # No resweep here: startup runs recover_embedding_tasks right
        # after this, and the worker's self-heal re-probe covers the
        # provider flip for projects with pending work.
        if dep.role == "embedding":
            _reset_embedding_deployment_config(workspace_root, dep.endpoint_url)
        elif dep.role == "teacher":
            disable_teacher_resident_endpoints(
                workspace_root, dep.local_nim_deployment_id
            )
        displaced = find_deployments_displaced_by(
            workspace_root, dep.local_nim_deployment_id
        )
        if displaced:
            await restore_displaced_deployments(
                displaced,
                workspace_root=workspace_root,
                settings=settings,
            )


async def _recover_student_deployment(
    dep: LocalNimDeployment,
    project_id: str,
    workspace_root: str,
) -> bool:
    """Tear down a Student deployment after backend restart.

    A confirmed removal marks the row stopped. If Docker cannot confirm the
    container is gone, the row remains active so placement still reserves its
    GPU and a later startup can retry cleanup. In either case the paired
    StudentModel becomes failed with failure_stage="interrupted_by_restart":
    the lifecycle orchestrator's in-memory benchmark state cannot be resumed.

    Returns whether the container is confirmed absent and resident restoration
    may safely begin.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return False

    container_ref = dep.container_id or dep.container_name
    remove_rc, _stdout, remove_stderr = await run_subprocess(
        "docker", "rm", "-f", container_ref, timeout_s=15.0
    )
    remove_error = remove_stderr.lower()
    container_absent = remove_rc == 0 or any(
        marker in remove_error for marker in ("no such container", "no such object")
    )

    # Local import to avoid circular import at module load.
    from vlm_feedback_loop.db.models.student_model import StudentModel

    with Session(engine) as session:
        d = session.get(LocalNimDeployment, dep.local_nim_deployment_id)
        if d is not None:
            if container_absent:
                d.status = "stopped"
                d.status_reason = "student_recovery_no_resume"
                d.stopped_at = utc_now()
            elif d.status in ("starting", "running"):
                d.status_reason = "student_recovery_teardown_failed"

        if dep.student_model_id:
            student = session.get(StudentModel, dep.student_model_id)
            if student is not None:
                student.serving_status = "failed"
                # Build a fresh dict so SQLAlchemy notices the JSON column
                # change (in-place mutation isn't tracked by default).
                details = dict(student.nim_preflight_details or {})
                details["failure_stage"] = "interrupted_by_restart"
                student.nim_preflight_details = details
                student.nim_endpoint_url = None
                student.nim_container_id = None

        session.commit()

    if not container_absent:
        logger.error(
            "Recovery: student deployment %s container removal was not "
            "confirmed (container=%s, docker_rc=%s); row remains active and "
            "resident restoration is deferred",
            dep.local_nim_deployment_id,
            container_ref,
            remove_rc,
        )
        return False

    logger.warning(
        "Recovery: student deployment %s torn down (container=%s, "
        "student_model_id=%s) — serving_status flipped to failed",
        dep.local_nim_deployment_id,
        container_ref,
        dep.student_model_id,
    )
    return True


# ── Public docker-run builders for handoff generation ─────────────────────────
#
# These are the only sanctioned entry points for serializing the docker run
# command that ``:deploy_nim`` would actually execute. The deployment_handoff
# AR generator and the closing-smoke (re-execution proof)
# MUST use these wrappers — emitting a bespoke docker-run string from the
# StudentModel fields would diverge from production behavior the moment any
# of the canonical builder's flags change (e.g. ``-u $(id -u)``, the read-only
# checkpoint mount, ``--gpus "device={n}"``).


def build_student_docker_run_args(
    *,
    deployment: LocalNimDeployment,
    settings: Settings,
) -> list[str]:
    """Canonical docker run argv list for a Student deployment.

    Byte-equivalent to what ``deploy_local_nim()`` actually executed. Every
    consumer gets the same safe ``-e NGC_API_KEY`` shape; the Docker client
    reads the exported value from its private child environment. Action
    Request payloads therefore cannot contain a credential or a second,
    unsafe builder mode.
    """
    # Mirror the per-prompt image cap and operator env the deploy path
    # resolved so this argv stays byte-equivalent to what
    # ``deploy_local_nim`` actually executed.
    engine = get_project_engine(deployment.project_id, settings.WORKSPACE_ROOT)
    image_cap = _resolve_deployment_image_cap(engine, deployment.model_config_id)
    nim_model_size, nim_model_profile, _base_served_name = _resolve_runtime_deploy_env(
        engine,
        deployment.model_config_id,
        role=deployment.role,
        custom_checkpoint=bool(
            deployment.checkpoint_mount_path and deployment.nim_model_name_path
        ),
    )
    return _build_docker_run_command(
        nim_container_image=deployment.nim_container_image,
        container_name=deployment.container_name,
        gpu_assignment=deployment.gpu_assignment,
        host_port=deployment.host_port,
        role=deployment.role,
        checkpoint_mount=deployment.checkpoint_mount_path,
        nim_model_name_path=deployment.nim_model_name_path,
        nim_served_model_name=deployment.nim_served_model_name,
        max_images_per_request=image_cap,
        nim_model_size=nim_model_size,
        nim_model_profile=nim_model_profile,
        extra_env=_resolve_extra_container_env(engine, deployment.model_config_id),
    )


def build_student_docker_run_display(
    *, deployment: LocalNimDeployment, settings: Settings
) -> str:
    """Copy-pastable shell command for a Student deployment.

    Emits name-only ``-e NGC_API_KEY``; the caller must export the value.
    """
    engine = get_project_engine(deployment.project_id, settings.WORKSPACE_ROOT)
    image_cap = _resolve_deployment_image_cap(engine, deployment.model_config_id)
    nim_model_size, nim_model_profile, _base_served_name = _resolve_runtime_deploy_env(
        engine,
        deployment.model_config_id,
        role=deployment.role,
        custom_checkpoint=bool(
            deployment.checkpoint_mount_path and deployment.nim_model_name_path
        ),
    )
    return docker_run_command_display(
        nim_container_image=deployment.nim_container_image,
        container_name=deployment.container_name,
        gpu_assignment=deployment.gpu_assignment,
        host_port=deployment.host_port,
        role=deployment.role,
        checkpoint_mount=deployment.checkpoint_mount_path,
        nim_model_name_path=deployment.nim_model_name_path,
        nim_served_model_name=deployment.nim_served_model_name,
        max_images_per_request=image_cap,
        nim_model_size=nim_model_size,
        nim_model_profile=nim_model_profile,
        extra_env=_resolve_extra_container_env(engine, deployment.model_config_id),
    )

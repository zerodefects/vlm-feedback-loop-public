# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment assessment service.

Probes the local environment for hardware, credentials, Docker, and GPU
availability.  The expensive machine capability snapshot (Docker, NVIDIA
Container Toolkit, and GPU inventory) is cached for the backend process
lifetime. Cheap deployment state such as credentials, configured embeddings,
and active NIM residents is composed fresh for every response.

Real deployment preflight remains authoritative and re-checks its hardware
requirements before launching a container. Operators who change machine
prerequisites without restarting the backend can explicitly invalidate and
refresh the cached snapshot through ``GET /v1/environment?refresh_hardware=true``.

Shell commands use ``asyncio.create_subprocess_exec`` (safe, no injection).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS3_SUPER_REASONER,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
    EMBEDDING_MODEL_ID,
    EMBEDDING_NIM_GPU_MIN_GB,
    EMBEDDING_NIM_IMAGE,
    EMBEDDING_NIM_SUPPORTED_GPU_NAMES,
    NEMOTRON_3_NANO_OMNI_REASONING,
)
from vlm_feedback_loop.services.logging_config import redact_exact_secrets
from vlm_feedback_loop.services.project_service import SEEDED_MODEL_CATALOG
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret
from vlm_feedback_loop.services.subprocess_utils import communicate_with_timeout

logger = logging.getLogger("vlm_feedback_loop.services.environment")


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class GpuInfo:
    """A detected GPU from nvidia-smi."""

    name: str
    memory_total_mb: int
    compute_capability: float | None = None

    @property
    def memory_total_gb(self) -> float:
        return round(self.memory_total_mb / 1024, 1)


@dataclass(frozen=True)
class MachineAssessment:
    """Stable host capabilities cached for one backend process lifetime."""

    docker_available: bool
    nvidia_toolkit_available: bool
    gpu_inventory: tuple[GpuInfo, ...]


@dataclass
class LocalDeployableModel:
    """A seeded model that supports local NIM deployment."""

    model_name: str
    nim_container_image: str
    gpu_memory_minimum_gb: int
    compute_capability_minimum: float | None
    fits: bool


def gpu_memory_meets_floor(
    memory_total_gb: float, minimum_memory_gb: int | float
) -> bool:
    """Match nominal GPU tiers despite small ``nvidia-smi`` reporting gaps.

    NVIDIA's nominal 80 GB cards can report about 79.6 GiB because product
    capacity and ``nvidia-smi`` use different units/reservations. A one-percent
    tolerance recognizes the documented hardware tier without making a
    meaningfully smaller card eligible.
    """

    return memory_total_gb >= float(minimum_memory_gb) * 0.99


@dataclass
class MissingPrerequisite:
    """A missing prerequisite for local NIM deployment."""

    check: str
    install_hint: str


@dataclass
class EmbeddingDeploymentSummary:
    """Embedding NIM deployment metadata from EmbeddingDeploymentConfig."""

    model_name: str
    nim_container_image: str
    gpu_memory_minimum_gb: int
    fits: bool
    provider: str


# ── Subprocess helpers (monkeypatchable in tests) ───────────────────────────


async def run_subprocess(
    *args: str,
    timeout_s: float = 10.0,
    stdin_input: str | None = None,
    secret_env: Mapping[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr).

    Returns (-1, "", error_message) on timeout or failure to start.

    When ``stdin_input`` is provided it is written to the child's stdin and
    the stream is closed — this is how secrets are handed to commands that
    read from stdin (e.g. ``docker login --password-stdin``) so they never
    appear in the process argv / ``ps`` output.

    ``secret_env`` adds private values to a copy of the current environment.
    It and ``stdin_input`` are treated as private: their values are redacted
    from stdout, stderr, and spawn diagnostics before they leave this boundary.
    The parent environment is never mutated.
    """
    try:
        subprocess_kwargs: dict[str, Any] = {
            "stdin": asyncio.subprocess.PIPE if stdin_input is not None else None,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "start_new_session": True,
        }
        if secret_env is not None:
            child_env = dict(os.environ)
            child_env.update(secret_env)
            subprocess_kwargs["env"] = child_env
        proc = await asyncio.create_subprocess_exec(
            *args,
            **subprocess_kwargs,
        )
        stdout_bytes, stderr_bytes = await communicate_with_timeout(
            proc,
            timeout_s=timeout_s,
            stdin=stdin_input.encode("utf-8") if stdin_input is not None else None,
        )
        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        return (
            proc.returncode or 0,
            _redact_private_values(stdout, secret_env, stdin_input),
            _redact_private_values(stderr, secret_env, stdin_input),
        )
    except TimeoutError:
        return (-1, "", f"Command timed out after {timeout_s}s")
    except FileNotFoundError:
        return (-1, "", f"Command not found: {args[0]}")
    except Exception as exc:
        return (-1, "", _redact_private_values(str(exc), secret_env, stdin_input))


def _redact_private_values(
    text: str,
    secret_env: Mapping[str, str] | None,
    stdin_input: str | None = None,
) -> str:
    """Remove child-only environment and stdin values from diagnostics."""
    values: set[str | None] = set(secret_env.values()) if secret_env else set()
    if stdin_input:
        values.add(stdin_input)
    return redact_exact_secrets(text, values)


# ── Individual checks ───────────────────────────────────────────────────────

# Docker creates this file inside every container; its presence means the
# backend itself is containerized (docker compose mode).
_DOCKERENV_PATH = Path("/.dockerenv")


def is_containerized() -> bool:
    """True when the backend runs inside a container.

    In containerized mode the backend has no docker CLI or socket, so
    system-managed local NIMs are unavailable by design — prerequisite
    hints must say that instead of "install Docker" on a host that is
    already running Docker.
    """
    return _DOCKERENV_PATH.exists()


async def check_docker_available() -> tuple[bool, str | None]:
    """Check if Docker is installed and running."""
    rc, stdout, stderr = await run_subprocess(
        "docker",
        "info",
        "--format",
        "{{.ServerVersion}}",
        timeout_s=10.0,
    )
    if rc == 0 and stdout:
        return (True, None)
    return (False, f"Docker not available: {stderr or 'unknown error'}")


async def check_nvidia_toolkit() -> tuple[bool, str | None]:
    """Check if NVIDIA Container Toolkit is installed and GPU passthrough works."""
    # The FIRST run pulls the CUDA base image (~hundreds of MB); a 30 s cap
    # spuriously reported the toolkit "failed" on a cold cache when the pull
    # alone exceeded it. A genuinely-missing runtime errors out fast (docker
    # rejects --runtime=nvidia immediately, before any pull), so the larger
    # budget only affects the cold-pull-then-success case, not the failure one.
    rc, _stdout, stderr = await run_subprocess(
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--gpus",
        "all",
        "nvidia/cuda:12.6.3-base-ubuntu24.04",
        "nvidia-smi",
        "-L",
        timeout_s=180.0,
    )
    if rc == 0:
        return (True, None)
    return (
        False,
        f"NVIDIA Container Toolkit check failed: {stderr or 'unknown error'}",
    )


async def probe_gpu_inventory() -> list[GpuInfo]:
    """Detect GPUs via nvidia-smi. Returns empty list on failure."""
    rc, stdout, _stderr = await run_subprocess(
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
        timeout_s=10.0,
    )
    if rc != 0 or not stdout:
        return []

    gpus: list[GpuInfo] = []
    for line in stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                name = parts[0]
                memory_mb = int(float(parts[1]))
                compute_capability = (
                    float(parts[2]) if len(parts) >= 3 and parts[2] else None
                )
                gpus.append(
                    GpuInfo(
                        name=name,
                        memory_total_mb=memory_mb,
                        compute_capability=compute_capability,
                    )
                )
            except (ValueError, IndexError):
                continue
    return gpus


# ── Assessment logic ────────────────────────────────────────────────────────


def _assess_local_deployable_models(
    gpus: list[GpuInfo],
    catalog_entries: list[dict[str, Any]] | None = None,
) -> list[LocalDeployableModel]:
    """Cross-reference catalog entries with local_deploy_metadata against GPU inventory."""
    if catalog_entries is None:
        catalog_entries = SEEDED_MODEL_CATALOG

    results: list[LocalDeployableModel] = []
    for entry in catalog_entries:
        meta = entry.get("local_deploy_metadata")
        if meta is None:
            continue
        min_gb = meta.get("nim_gpu_memory_minimum_gb", 0)
        min_compute = meta.get("nim_compute_capability_minimum")
        results.append(
            LocalDeployableModel(
                model_name=entry["model_name"],
                nim_container_image=meta["nim_container_image"],
                gpu_memory_minimum_gb=min_gb,
                compute_capability_minimum=min_compute,
                fits=any(
                    _gpu_meets_model_floor(gpu, min_gb, min_compute) for gpu in gpus
                ),
            )
        )
    return results


def _gpu_meets_model_floor(
    gpu: GpuInfo,
    minimum_memory_gb: int | float,
    minimum_compute_capability: float | None,
) -> bool:
    """Whether one physical GPU satisfies a local model's declared floors."""

    if not gpu_memory_meets_floor(gpu.memory_total_gb, minimum_memory_gb):
        return False
    if minimum_compute_capability is None:
        return True
    return (
        gpu.compute_capability is not None
        and gpu.compute_capability >= minimum_compute_capability
    )


# Quality-first preference for local Teacher recommendation. Hardware floors
# remain eligibility gates, but do not double as a quality ranking. This keeps
# a larger model from silently becoming the default just because it consumes
# more memory. The order is backed by the completed long-horizon ICL matrix:
# Omni has the highest absolute labeling accuracy when its cc>=9.0 NIM is
# supported; CR3-Nano is the robust fallback and beats Super on two of the
# three final campaign datasets. Models absent from the map sort last so a new
# catalog row never becomes the automatic default without evidence.
_LOCAL_TEACHER_PREFERENCE_RANK: dict[str, int] = {
    NEMOTRON_3_NANO_OMNI_REASONING: 500,
    COSMOS3_NANO_REASONER: 400,
    COSMOS_REASON2_8B: 300,
    COSMOS3_SUPER_REASONER: 200,
    COSMOS_REASON2_2B: 100,
}


def _pick_local_teacher_recommendation(
    gpus: list[GpuInfo],
    catalog_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the recommended teacher-eligible locally-deployable model that fits.

    On a GPU machine the FTUE surfaces the highest-quality validated Teacher
    whose memory and architecture floors are both met: Omni on supported
    compute-capability >=9.0 GPUs with >=80 GB, CR3-Nano on >=56 GB, and
    CR2-2B on 36–55 GB. CR3-Super and CR2-8B remain selectable alternates.

    Sort is by curated quality rank first, then memory floor and model name
    for deterministic fallback. Hardware size only decides eligibility.

    Filters to entries with ``"teacher"`` in ``eligible_roles`` so the
    embedding NIM and student-only bases don't accidentally win. Returns
    the catalog entry dict (caller picks the fields it needs) or ``None``
    when no teacher-eligible local model fits.
    """
    if catalog_entries is None:
        catalog_entries = SEEDED_MODEL_CATALOG

    if not gpus:
        return None
    eligible: list[tuple[int, int, str, dict[str, Any]]] = []
    for entry in catalog_entries:
        meta = entry.get("local_deploy_metadata")
        if meta is None:
            continue
        if "teacher" not in entry.get("eligible_roles", []):
            continue
        min_gb = meta.get("nim_gpu_memory_minimum_gb", 0)
        min_compute = meta.get("nim_compute_capability_minimum")
        if not any(_gpu_meets_model_floor(gpu, min_gb, min_compute) for gpu in gpus):
            continue
        rank = _LOCAL_TEACHER_PREFERENCE_RANK.get(entry.get("model_name", ""), 0)
        model_name = str(entry.get("model_name", ""))
        eligible.append((rank, min_gb, model_name, entry))

    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return eligible[0][3]


def pick_local_teacher_recommendation(
    gpus: list[GpuInfo],
    catalog_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Public entry point for the curated local-Teacher policy."""

    return _pick_local_teacher_recommendation(gpus, catalog_entries)


def _pick_running_teacher_resident_entry(
    active_residents: list[Any],
    catalog_entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the seeded catalog entry for the first exact running Teacher.

    A verified resident is stronger evidence than a theoretical GPU-fit
    recommendation.  Matching the container-affecting identity is important
    for shared images such as Cosmos 3, where Nano and Super use the same image
    but different size/profile selectors.
    """

    # Imported here because local_nim_service imports this module while it
    # initializes. Runtime identity must use the same validation as launch.
    from vlm_feedback_loop.services.local_nim_service import (
        normalize_extra_container_env,
    )

    if catalog_entries is None:
        catalog_entries = SEEDED_MODEL_CATALOG

    entries_by_name = {
        str(entry.get("model_name", "")).casefold(): entry
        for entry in catalog_entries
        if "teacher" in entry.get("eligible_roles", [])
        and entry.get("local_deploy_metadata")
    }
    for resident in active_residents:
        if (
            resident.role != "teacher"
            or resident.status != "running"
            or resident.model_name is None
        ):
            continue
        entry = entries_by_name.get(str(resident.model_name).casefold())
        if entry is None:
            continue
        metadata: dict[str, Any] = entry["local_deploy_metadata"]
        requested_env = tuple(
            sorted(
                normalize_extra_container_env(
                    metadata.get("extra_container_env")
                ).items()
            )
        )
        if (
            str(metadata.get("nim_container_image", "")) == resident.nim_container_image
            and (
                str(metadata["nim_model_size"])
                if metadata.get("nim_model_size")
                else None
            )
            == resident.nim_model_size
            and (
                str(metadata["nim_model_profile"])
                if metadata.get("nim_model_profile")
                else None
            )
            == resident.nim_model_profile
            and requested_env == resident.extra_container_env
        ):
            return entry
    return None


def _embedding_gpu_candidates(
    gpus: list[GpuInfo],
    occupied_roles_by_device: dict[str, set[str]],
    reserve_lowest_free_for_teacher: bool,
) -> list[GpuInfo]:
    """Return GPUs the embedding NIM could actually claim.

    Mirrors ``local_nim_service.resolve_gpu_placement``: a device with
    an active non-embedding resident (Teacher / Student) is unavailable,
    while a device already running the embedding NIM stays a candidate —
    the local provider is live there. When a local Teacher is
    recommended but not yet deployed, the lowest fully-free device is
    reserved for it (the auto-placer's deterministic pick), so a
    single-GPU host that will run the Teacher locally reports no
    embedding candidate.
    """
    candidates: dict[int, GpuInfo] = {}
    for i, gpu in enumerate(gpus):
        roles = occupied_roles_by_device.get(str(i), set())
        if roles - {"embedding"}:
            continue
        candidates[i] = gpu

    if reserve_lowest_free_for_teacher:
        fully_free = [i for i in candidates if not occupied_roles_by_device.get(str(i))]
        if fully_free:
            del candidates[min(fully_free)]

    return list(candidates.values())


def _normalized_gpu_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


_SUPPORTED_EMBEDDING_GPU_NAMES = frozenset(
    _normalized_gpu_name(name) for name in EMBEDDING_NIM_SUPPORTED_GPU_NAMES
)


def embedding_gpu_is_supported(gpu: GpuInfo) -> bool:
    """Whether NIM 2.0.0 validates the pinned VLM model on this exact SKU."""
    return _normalized_gpu_name(gpu.name) in _SUPPORTED_EMBEDDING_GPU_NAMES


def _get_embedding_deployment_summary(
    settings: Settings, candidate_gpus: list[GpuInfo]
) -> EmbeddingDeploymentSummary:
    """Read EmbeddingDeploymentConfig from deployment.db and summarize.

    ``candidate_gpus`` contains devices the embedding NIM could actually
    claim. ``fits`` is true only when one device both meets the configured
    memory floor and is a validated SKU for the pinned model/NIM release.
    """
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    with Session(engine) as session:
        config = session.execute(select(EmbeddingDeploymentConfig)).scalar_one_or_none()

    if config is None:
        return EmbeddingDeploymentSummary(
            model_name=EMBEDDING_MODEL_ID,
            nim_container_image=EMBEDDING_NIM_IMAGE,
            gpu_memory_minimum_gb=EMBEDDING_NIM_GPU_MIN_GB,
            fits=False,
            provider="none",
        )

    supported_memory_gb = [
        gpu.memory_total_gb for gpu in candidate_gpus if embedding_gpu_is_supported(gpu)
    ]
    best_gpu_gb = max(supported_memory_gb, default=0.0)
    return EmbeddingDeploymentSummary(
        model_name=config.model_name,
        nim_container_image=config.nim_container_image,
        gpu_memory_minimum_gb=config.gpu_memory_minimum_gb,
        fits=best_gpu_gb >= config.gpu_memory_minimum_gb,
        provider=config.provider,
    )


def _build_missing_prerequisites(
    docker_available: bool,
    nvidia_toolkit_available: bool,
    nvidia_api_key_configured: bool,
    ngc_api_key_configured: bool,
    containerized: bool,
) -> list[MissingPrerequisite]:
    """Build list of missing prerequisites with install hints."""
    missing: list[MissingPrerequisite] = []

    if not docker_available:
        if containerized:
            # The backend container has no docker CLI/socket — telling the
            # user to "install Docker" on a host that is already running
            # this stack under Docker would be wrong.
            missing.append(
                MissingPrerequisite(
                    check="Docker",
                    install_hint=(
                        "The backend is running inside a container without "
                        "access to the host Docker daemon. System-managed "
                        "local NIMs require local-source mode "
                        "(./scripts/dev.sh); in containerized mode use "
                        "hosted NIM endpoints instead."
                    ),
                )
            )
        else:
            missing.append(
                MissingPrerequisite(
                    check="Docker",
                    install_hint="Install Docker: https://docs.docker.com/engine/install/ or run ./scripts/setup-local.sh",
                )
            )

    if docker_available and not nvidia_toolkit_available:
        missing.append(
            MissingPrerequisite(
                check="NVIDIA Container Toolkit",
                install_hint="Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html or run ./scripts/setup-local.sh",
            )
        )

    if not nvidia_api_key_configured and not ngc_api_key_configured:
        missing.append(
            MissingPrerequisite(
                check="API credentials",
                install_hint="Configure NVIDIA_API_KEY (for hosted NIM) or NGC_API_KEY (for local NIM) in ~/.vlm_feedback_loop/.env",
            )
        )

    return missing


async def _probe_machine_assessment() -> MachineAssessment:
    """Probe stable host capabilities.

    This is the expensive portion of the environment assessment: the toolkit
    check starts a temporary CUDA container. Callers serving ordinary reads
    should use :func:`get_cached_environment`, not this helper directly.
    """
    docker_result, gpu_inventory = await asyncio.gather(
        check_docker_available(),
        probe_gpu_inventory(),
    )
    docker_available, _docker_error = docker_result

    nvidia_toolkit_available = False
    if docker_available:
        toolkit_result = await check_nvidia_toolkit()
        nvidia_toolkit_available = toolkit_result[0]

    return MachineAssessment(
        docker_available=docker_available,
        nvidia_toolkit_available=nvidia_toolkit_available,
        gpu_inventory=tuple(gpu_inventory),
    )


def _compose_environment(
    settings: Settings,
    machine: MachineAssessment,
) -> dict[str, Any]:
    """Combine cached host capabilities with current deployment state."""
    # Runtime override wins over the .env-loaded
    # value so a UI-applied key is reflected in the next env response —
    # the recommendation surface can flip from "hosted only" to "hybrid"
    # once an NGC key is pasted without a backend restart.
    nvidia_api_key_configured = bool(get_effective_secret("NVIDIA_API_KEY", settings))
    ngc_api_key_configured = bool(get_effective_secret("NGC_API_KEY", settings))
    docker_available = machine.docker_available
    nvidia_toolkit_available = machine.nvidia_toolkit_available
    gpu_inventory = list(machine.gpu_inventory)

    local_deploy_available = (
        docker_available and nvidia_toolkit_available and len(gpu_inventory) > 0
    )

    local_deployable_models = _assess_local_deployable_models(gpu_inventory)

    missing_prerequisites = _build_missing_prerequisites(
        docker_available,
        nvidia_toolkit_available,
        nvidia_api_key_configured,
        ngc_api_key_configured,
        containerized=is_containerized(),
    )

    # Function-level import: local_nim_service imports this module at load
    # time. Resident summaries let the FTUE distinguish "needs deployment"
    # from "the exact Teacher is already running for another project."
    from vlm_feedback_loop.services.local_nim_service import (
        list_active_nim_residents,
        scan_active_resident_roles_by_device,
    )

    active_residents = list_active_nim_residents(settings.WORKSPACE_ROOT)
    running_teacher_entry = _pick_running_teacher_resident_entry(active_residents)
    quality_teacher_entry = (
        _pick_local_teacher_recommendation(gpu_inventory)
        if local_deploy_available
        else None
    )
    # Reuse is automatic only when the resident is the model the current
    # quality policy would choose on this hardware. A different resident is
    # still reported to the FTUE, which asks the user whether to keep it or
    # explicitly replace it with the recommendation.
    resident_teacher_entry = (
        running_teacher_entry
        if running_teacher_entry is not None
        and quality_teacher_entry is not None
        and running_teacher_entry["model_name"] == quality_teacher_entry["model_name"]
        else None
    )

    # Local teacher recommendation.
    # Populated whenever a teacher-eligible Cosmos Reason2
    # variant fits the GPU, regardless of whether ``NVIDIA_API_KEY`` is
    # configured. The frontend uses these fields for:
    #   * Case A (no key + GPU + fit): primary "Run Cosmos Reason2
    #     locally" CTA on the NVIDIA API key setup page.
    #   * Case B (key + GPU + fit): "Also deploy Cosmos Reason2 locally?"
    #     hybrid peer card on that same page.
    # Returns None when no teacher-eligible local model fits the GPU,
    # which collapses both cases to today's hosted-only flow.
    local_teacher_entry = resident_teacher_entry or quality_teacher_entry

    # Recommendation logic: when NVIDIA_API_KEY is not configured and
    # local deployment is available, recommend a local Teacher.
    #
    # Teacher:
    #   * Exact running Blueprint Teacher → "local", regardless of a hosted
    #     key. Fresh projects attach and select that resident at creation.
    #   * Key configured and no reusable resident → "hosted". Step 3.7 Flash is the seeded
    #     default and is the strongest hosted Teacher on live end-to-end
    #     validation runs. The local Teacher recommendation is still
    #     populated so the frontend can offer hybrid.
    #   * No key + GPU + teacher-eligible local model fits → "local".
    #     Cosmos Reason2 lights up as the primary path on a GPU box —
    #     the SME can run the loop offline without an API key.
    #   * Otherwise → "hosted". Not a dead end: the NVIDIA API key setup
    #     page prompts for a key.
    if resident_teacher_entry is not None:
        recommended_teacher_mode = "local"
    elif nvidia_api_key_configured:
        recommended_teacher_mode = "hosted"
    elif local_teacher_entry is not None:
        recommended_teacher_mode = "local"
    else:
        recommended_teacher_mode = "hosted"

    # Embedding: the local embedding NIM is the DEFAULT provider
    # whenever a suitable GPU will actually be free for it — hosted is
    # the fallback. Local embedding avoids hosted rate limits and
    # doesn't compete with the Teacher for hosted-NIM quota. "Suitable"
    # is placement-aware: devices holding an active Teacher / Student
    # NIM are excluded, and when a local Teacher is recommended but not
    # yet deployed, the auto-placer's pick is reserved for it. A host
    # whose GPUs are all below every Teacher floor but at/above the
    # embedding floor still recommends local (hosted Teacher + local
    # embedding NIM).
    occupied_roles_by_device: dict[str, set[str]] = {}
    if gpu_inventory:
        occupied_roles_by_device = scan_active_resident_roles_by_device(
            settings.WORKSPACE_ROOT
        )
    teacher_already_resident = resident_teacher_entry is not None or any(
        "teacher" in roles for roles in occupied_roles_by_device.values()
    )
    embedding_deployment = _get_embedding_deployment_summary(
        settings,
        _embedding_gpu_candidates(
            gpu_inventory,
            occupied_roles_by_device,
            reserve_lowest_free_for_teacher=(
                recommended_teacher_mode == "local" and not teacher_already_resident
            ),
        ),
    )
    embedding_fits_locally = local_deploy_available and embedding_deployment.fits
    if settings.EMBEDDING_PROVIDER == "none":
        # An explicit operator opt-out is a real supported mode: the review
        # selector remains useful through pHash diversity. Do not recommend a
        # hosted endpoint (or prompt/probe its key) after the operator chose
        # that local-only contract.
        recommended_embedding_mode = "none"
    else:
        recommended_embedding_mode = "local" if embedding_fits_locally else "hosted"

    # Concrete fields for the local-teacher peer card. Null when no
    # local teacher fits; the frontend collapses the peer card in that
    # case. Image is the pinned ``nvcr.io/...`` ref from
    # local_deploy_metadata so the FTUE can name the exact container.
    # ``_pick_local_teacher_recommendation`` only returns entries that
    # have ``local_deploy_metadata`` populated with both keys, so the
    # indexing below is safe when ``local_teacher_entry`` is not None.
    recommended_local_teacher_model_name: str | None
    recommended_local_teacher_image: str | None
    recommended_local_teacher_gpu_memory_minimum_gb: int | None
    if local_teacher_entry is not None:
        local_teacher_meta: dict[str, Any] = local_teacher_entry[
            "local_deploy_metadata"
        ]
        recommended_local_teacher_model_name = str(local_teacher_entry["model_name"])
        recommended_local_teacher_image = str(local_teacher_meta["nim_container_image"])
        recommended_local_teacher_gpu_memory_minimum_gb = int(
            local_teacher_meta["nim_gpu_memory_minimum_gb"]
        )
    else:
        recommended_local_teacher_model_name = None
        recommended_local_teacher_image = None
        recommended_local_teacher_gpu_memory_minimum_gb = None

    return {
        "hosted_nim_available": nvidia_api_key_configured,
        "local_deploy_available": local_deploy_available,
        "docker_available": docker_available,
        "nvidia_toolkit_available": nvidia_toolkit_available,
        "nvidia_api_key_configured": nvidia_api_key_configured,
        "ngc_api_key_configured": ngc_api_key_configured,
        "gpus": [
            {
                "name": g.name,
                "memory_total_gb": g.memory_total_gb,
                "compute_capability": g.compute_capability,
            }
            for g in gpu_inventory
        ],
        "local_deployable_models": [
            {
                "model_name": m.model_name,
                "nim_container_image": m.nim_container_image,
                "gpu_memory_minimum_gb": m.gpu_memory_minimum_gb,
                "compute_capability_minimum": m.compute_capability_minimum,
                "fits": m.fits,
            }
            for m in local_deployable_models
        ],
        "embedding_deployment": {
            "model_name": embedding_deployment.model_name,
            "nim_container_image": embedding_deployment.nim_container_image,
            "gpu_memory_minimum_gb": embedding_deployment.gpu_memory_minimum_gb,
            "fits": embedding_deployment.fits,
            "provider": embedding_deployment.provider,
        },
        "missing_prerequisites": [
            {"check": p.check, "install_hint": p.install_hint}
            for p in missing_prerequisites
        ],
        "nim_startup_timeout_s": settings.NIM_STARTUP_TIMEOUT_S,
        "student_latency_test_concurrencies": list(
            settings.STUDENT_LATENCY_TEST_CONCURRENCIES
        ),
        "default_teacher_model_name": settings.DEFAULT_TEACHER_MODEL,
        "recommended_teacher_mode": recommended_teacher_mode,
        "recommended_embedding_mode": recommended_embedding_mode,
        "recommended_local_teacher_model_name": recommended_local_teacher_model_name,
        "recommended_local_teacher_image": recommended_local_teacher_image,
        "recommended_local_teacher_gpu_memory_minimum_gb": (
            recommended_local_teacher_gpu_memory_minimum_gb
        ),
        "active_local_nim_residents": [
            resident.public_summary() for resident in active_residents
        ],
        "allow_secret_persist": _resolve_allow_secret_persist(settings),
    }


# ── Machine assessment cache ────────────────────────────────────────────────

_machine_assessment_cache: MachineAssessment | None = None
_machine_assessment_task: asyncio.Task[MachineAssessment] | None = None


async def assess_environment(settings: Settings) -> dict[str, Any]:
    """Run a fully fresh assessment, including machine subprocess probes.

    Deployment preflight and focused tests can use this uncached entry point.
    Ordinary API reads should use :func:`get_cached_environment`.
    """
    return _compose_environment(settings, await _probe_machine_assessment())


async def get_cached_machine_assessment() -> MachineAssessment:
    """Return the process-wide machine snapshot with single-flight cold start."""
    global _machine_assessment_cache, _machine_assessment_task
    if _machine_assessment_cache is not None:
        return _machine_assessment_cache

    task = _machine_assessment_task
    if task is None:
        task = asyncio.create_task(_probe_machine_assessment())
        _machine_assessment_task = task

    cache_result = _machine_assessment_task is task
    try:
        result = await task
    finally:
        cache_result = cache_result and _machine_assessment_task is task
        if cache_result:
            _machine_assessment_task = None

    # An explicit invalidation while this task was running detaches it from
    # ``_machine_assessment_task``. Return its result to the original caller,
    # but do not repopulate the cache after that invalidation.
    if cache_result and _machine_assessment_cache is None:
        _machine_assessment_cache = result
    return result


async def get_cached_environment(settings: Settings) -> dict[str, Any]:
    """Return fresh deployment state over the cached machine snapshot."""
    machine = await get_cached_machine_assessment()
    return _compose_environment(settings, machine)


def invalidate_machine_assessment_cache() -> None:
    """Force the next cached read to re-probe Docker, toolkit, and GPUs."""
    global _machine_assessment_cache, _machine_assessment_task
    _machine_assessment_cache = None
    # Do not cancel a probe already serving a caller. Detaching it prevents
    # that stale result from repopulating the cache; the next read starts a
    # replacement probe.
    _machine_assessment_task = None


def _resolve_allow_secret_persist(settings: Settings) -> bool:
    """Compute the effective ``allow_secret_persist`` flag.

    True when BOTH the deployment-level ``ALLOW_UI_SECRET_PERSIST`` flag
    is on (default true; container/production deployments set it false)
    AND the canonical ``.env`` parent directory is writable from this
    process. The directory-writable check matters because a misconfigured
    home directory permission would cause ``POST /v1/secrets:set`` with
    ``persist=true`` to 500 at write time — better to surface the
    "can't persist" state in the env response so the UI hides the
    checkbox cleanly.
    """
    if not settings.ALLOW_UI_SECRET_PERSIST:
        return False
    from vlm_feedback_loop.config import get_canonical_env_file_path

    env_path = get_canonical_env_file_path()
    parent = env_path.parent
    # Writable when the parent dir exists and passes a W_OK probe. When
    # the parent does not exist yet, return True optimistically — the
    # first persist mkdirs it (0o700), and a mkdir failure is rare
    # enough that surfacing it as a write-time error is acceptable.
    if not parent.exists():
        return True  # We'll mkdir(0o700) on first write
    try:
        # Probe: os.access W_OK is the cheapest portable signal.
        import os as _os

        return _os.access(parent, _os.W_OK)
    except OSError:
        return False

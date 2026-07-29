# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP embedding computation, storage, and in-memory cache.

Orchestrates the full embedding workflow (default model: NeMo Retriever
VL 1B v2; the *_nvclip provider labels are retained for back-compat):
  - Embedding endpoint probe at project creation / first ingest
  - Background batched computation in deterministic seeded-random order
  - Vector persistence in the ``ClipEmbedding`` table
  - In-memory cache for fast similarity lookups
  - SSE progress/completion events
  - Restart recovery for interrupted computation
  - Provider resweep when the local embedding NIM comes or goes
"""

from __future__ import annotations

import asyncio
import logging
import random
import struct
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db
from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services import nim_client
from vlm_feedback_loop.services.background import (
    background_manager,
    run_in_low_priority_thread,
    run_in_thread,
)
from vlm_feedback_loop.services.image_transport import (
    read_and_normalize,
    to_base64_data_url,
)
from vlm_feedback_loop.services.priority import priority_dispatch
from vlm_feedback_loop.services.project_service import (
    get_project_engine,
    projects_root,
)
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret
from vlm_feedback_loop.services.sse import sse_manager

logger = logging.getLogger("vlm_feedback_loop.clip_embedding")

PROBE_TEXT_INPUT = "hello"
INTER_BATCH_SLEEP_S = 0.05


def _prepare_embedding_input(storage_ref: str) -> str:
    """Read, normalize, and base64-encode one embedding image off-loop.

    Base64 encoding previously ran on the asyncio event loop after the image
    read returned from the low-priority pool. A hosted batch can carry eight
    multi-megabyte images, so those repeated encodes delayed unrelated
    response streaming and interactive request completion despite the worker
    being nominally background-only.
    """
    img_bytes, mime = read_and_normalize(storage_ref)
    return to_base64_data_url(img_bytes, mime)


def _resolve_worker_shape(provider: str, settings: Settings) -> tuple[int, int]:
    """Pick effective ``(concurrency, batch_size)`` for the worker.

    Hosted endpoints (``hosted_nvclip``) keep concurrency at 1 with
    larger batches to stay polite under shared rate limits.  Self-hosted
    or local NIMs (``self_hosted_nvclip``, ``local_nvclip``) dispatch
    smaller-batch requests concurrently to saturate the GPU pipeline.
    See ``_defaults.py`` for the configurable values and the rationale.
    ``probe_nvclip`` emits ``self_hosted_nvclip`` whenever a healthy
    local embedding NIM is deployed; ``local_nvclip`` is a legacy config
    value that is accepted but never emitted.
    """
    if provider == "hosted_nvclip":
        return (
            settings.EMBEDDING_CONCURRENCY_HOSTED,
            settings.EMBEDDING_BATCH_SIZE_HOSTED,
        )
    # self_hosted_nvclip, local_nvclip — both treated as self-hosted
    return (
        settings.EMBEDDING_CONCURRENCY_SELF_HOSTED,
        settings.EMBEDDING_BATCH_SIZE_SELF_HOSTED,
    )


def _self_hosted_stamp_out_of_sync(provider: str, settings: Settings) -> bool:
    """True when a project's self-hosted stamp lost its config backing.

    ``_resolve_embedding_endpoint`` only requires a recorded endpoint
    URL, but the deploy flow's healthy transition is what vouches for
    that URL by setting ``provider="self_hosted_nvclip"`` alongside it.
    A config whose provider flag moved off self-hosted while a stale
    URL lingers (partial config update) must count as unserveable so
    the worker re-probes instead of computing against an endpoint
    nothing vouches for.
    """
    if provider not in ("self_hosted_nvclip", "local_nvclip"):
        return False
    config = _local_embedding_deployment(settings)
    return config is None or config.provider != "self_hosted_nvclip"


def _local_embedding_deployment(
    settings: Settings,
) -> EmbeddingDeploymentConfig | None:
    """Read the deployment-scoped ``EmbeddingDeploymentConfig`` singleton.

    Returns a detached row, or ``None`` when the singleton is missing.
    """
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    with Session(engine) as session:
        config = session.execute(select(EmbeddingDeploymentConfig)).scalar_one_or_none()
        if config is not None:
            session.expunge(config)
        return config


def _resolve_embedding_endpoint(
    provider: str,
    settings: Settings,
) -> tuple[str, dict[str, str], str] | None:
    """Resolve ``(base_url, auth_headers, model_id)`` for a provider.

    Self-hosted NIMs (``self_hosted_nvclip``; legacy ``local_nvclip``)
    are unauthenticated — requests carry no bearer header
    (``nim_client`` still adds the Blueprint source header) — and live
    at the endpoint recorded on ``EmbeddingDeploymentConfig``. The
    hosted arm needs ``NVIDIA_API_KEY``; the runtime override layer
    wins over the .env-loaded value so a UI-applied key takes effect on
    the next worker spawn without a backend restart. Returns ``None``
    when the provider cannot be served (no recorded endpoint for
    self-hosted, missing key for hosted).
    """
    if provider in ("self_hosted_nvclip", "local_nvclip"):
        config = _local_embedding_deployment(settings)
        if config is None or not config.endpoint_url:
            return None
        return config.endpoint_url, {}, config.model_name

    if provider == "hosted_nvclip":
        try:
            auth = nim_client.build_auth_headers(
                auth_mode="bearer",
                credential=get_effective_secret("NVIDIA_API_KEY", settings),
            )
        except ValueError:
            return None
        return settings.HOSTED_NIM_BASE_URL, auth, settings.EMBEDDING_MODEL_ID

    return None


# ── Vector serialization ────────────────────────────────────────


def serialize_vector(vector: list[float], dim: int) -> bytes:
    """Pack a float vector into a binary blob (float32)."""
    if len(vector) != dim:
        raise ValueError(f"Vector length {len(vector)} does not match dimension {dim}")
    return struct.pack(f"{dim}f", *vector)


def deserialize_vector(blob: bytes, dim: int) -> list[float]:
    """Unpack a binary blob into a float vector."""
    expected = dim * 4
    if len(blob) != expected:
        raise ValueError(
            f"Blob length {len(blob)} does not match expected {expected} "
            f"for dimension {dim}"
        )
    return list(struct.unpack(f"{dim}f", blob))


# ── pHash hamming distance ──────────────────────────────────────


def phash_hamming_distance(a: str, b: str) -> int:
    """Compute hamming distance between two 16-char hex pHash strings."""
    val_a = int(a, 16)
    val_b = int(b, 16)
    return bin(val_a ^ val_b).count("1")


def _embedding_order(
    examples: list[tuple[str, str | None]],
    project_id: str,
) -> list[str]:
    """Order the pending examples for CLIP embedding.

    A deterministic, ``project_id``-seeded shuffle. The worker embeds in
    this order and the review selector can only pick diversely among
    examples that *already have* an embedding, so the prefix of this order
    decides what the SME reviews during the (often long) embedding warmup.
    A uniform random prefix is a representative sample of the whole dataset
    at every coverage level, so early picks span all classes.

    The alternatives both cluster: ingest order is class-blocked (datasets
    are stored one directory per class), which traps the SME on the first
    class until embeddings catch up; ordering by pHash diversity tracks
    visual layout rather than semantic class and front-loads whichever
    class spans the widest pHash range. Seeding on ``project_id`` keeps the
    order reproducible for a given project.
    """
    keys = [key for key, _phash in examples]
    random.Random(project_id).shuffle(keys)
    return keys


# ── In-memory embedding cache ───────────────────────────────────


class _EmbeddingCache:
    """Project-scoped in-memory cache for CLIP embedding vectors."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, list[float]]] = {}
        # Memoized L2-normalized matrix + key→row index per project, built
        # lazily from _store for vectorized cosine scoring (the review
        # selector scores every eligible candidate against the recent
        # window each pick). Dropped on any store mutation so it never
        # goes stale; rebuilt on next request.
        self._normalized: dict[str, tuple[dict[str, int], NDArray[np.float32]]] = {}

    def load(self, project_id: str, engine: Any, dim: int) -> None:
        """Bulk-load all embeddings for a project from the DB."""
        temp: dict[str, list[float]] = {}
        with Session(engine) as session:
            rows = session.execute(
                select(
                    ClipEmbedding.example_key,
                    ClipEmbedding.vector_blob_f32,
                ).where(ClipEmbedding.project_id == project_id)
            ).all()
            for key, blob in rows:
                temp[key] = deserialize_vector(blob, dim)
        self._store[project_id] = temp
        self._normalized.pop(project_id, None)
        logger.debug(
            "Loaded %d embeddings into cache for project %s",
            len(temp),
            project_id,
        )

    def get(self, project_id: str, example_key: str) -> list[float] | None:
        project_cache = self._store.get(project_id)
        if project_cache is None:
            return None
        return project_cache.get(example_key)

    def put(self, project_id: str, example_key: str, vector: list[float]) -> None:
        if project_id not in self._store:
            self._store[project_id] = {}
        self._store[project_id][example_key] = vector
        self._normalized.pop(project_id, None)

    def count(self, project_id: str) -> int:
        return len(self._store.get(project_id, {}))

    def get_all(self, project_id: str) -> dict[str, list[float]]:
        return dict(self._store.get(project_id, {}))

    def get_normalized_matrix(
        self, project_id: str
    ) -> tuple[dict[str, int], NDArray[np.float32]]:
        """Return ``(key→row index, L2-normalized (N, D) matrix)``.

        Memoized view of the project's cached vectors for vectorized
        cosine scoring: with every row a unit vector, a plain matrix
        multiply yields cosine similarities directly. Zero-magnitude
        vectors stay zero (cosine 0.0 against everything), matching
        ``sim_clip``'s zero-norm guard. Built once and reused until the
        store mutates. Empty projects yield an empty index and a ``(0, 0)``
        matrix.
        """
        cached = self._normalized.get(project_id)
        if cached is not None:
            return cached
        store = self._store.get(project_id, {})
        keys = list(store)
        index = {key: row for row, key in enumerate(keys)}
        if keys:
            matrix = np.array([store[k] for k in keys], dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, norms, out=matrix, where=norms != 0.0)
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)
        result = (index, matrix)
        self._normalized[project_id] = result
        return result

    def invalidate(self, project_id: str) -> None:
        """Drop a project's cached embeddings (tests only).

        No production caller — tests use this to reset the module-global
        cache between cases, mirroring ``locks.clear_lock_state``.
        """
        self._store.pop(project_id, None)
        self._normalized.pop(project_id, None)

    def is_loaded(self, project_id: str) -> bool:
        return project_id in self._store


embedding_cache = _EmbeddingCache()


# ── Embedding probe ────────────────────────────────────────────


async def _verify_local_embedding_nim(
    config: EmbeddingDeploymentConfig,
    settings: Settings,
) -> tuple[bool, str, str | None, int | None]:
    """Live-verify a locally deployed embedding NIM before selecting it.

    One real ``create_embeddings`` call against the recorded endpoint —
    no bearer auth (local NIMs are unauthenticated; ``nim_client`` still
    adds the Blueprint source header). The live check is what makes a
    stale config safe: nothing clears ``EmbeddingDeploymentConfig`` when
    the container stops or is displaced, so a dead endpoint must fail
    here and let the caller fall back rather than stamping projects
    against an endpoint that cannot serve.
    """
    if not config.endpoint_url:
        return False, "none", None, None

    result = await nim_client.create_embeddings(
        base_url=config.endpoint_url,
        auth_headers={},
        model=config.model_name,
        input_items=[PROBE_TEXT_INPUT],
        deadline_s=30.0,
        max_retries=2,
        input_type=settings.EMBEDDING_INPUT_TYPE,
    )

    if result.success and result.embeddings:
        actual_dim = len(result.embeddings[0])
        if actual_dim == config.embedding_dim:
            return (
                True,
                "self_hosted_nvclip",
                config.model_name,
                config.embedding_dim,
            )
        logger.warning(
            "Local embedding NIM probe dimension mismatch: expected %d, got %d",
            config.embedding_dim,
            actual_dim,
        )
        return False, "none", None, None

    logger.warning("Local embedding NIM probe failed: %s", result.error)
    return False, "none", None, None


async def probe_nvclip(
    settings: Settings,
) -> tuple[bool, str, str | None, int | None]:
    """Probe embedding endpoints and resolve the effective provider.

    Returns ``(success, provider, model_id, dim)``.

    ``EMBEDDING_PROVIDER="auto"`` is a cascade: a healthy local
    embedding NIM (recorded on ``EmbeddingDeploymentConfig`` by the
    deploy flow) is the default, the hosted endpoint (``NVIDIA_API_KEY``)
    is the fallback, and "none" — pHash-diversity mode — is the last
    resort. Explicit provider values pin a single arm and never fall
    through.
    """
    provider_setting = settings.EMBEDDING_PROVIDER

    if provider_setting == "none":
        return False, "none", None, None

    if provider_setting in ("auto", "self_hosted_nvclip", "local_nvclip"):
        config = _local_embedding_deployment(settings)
        # "auto" tries only an endpoint the deploy flow marked healthy;
        # an explicit local pin tries whatever endpoint is recorded.
        # Either way the live verify is the real gate against a stale
        # config (see ``_verify_local_embedding_nim``).
        if config is not None and (
            provider_setting != "auto" or config.provider == "self_hosted_nvclip"
        ):
            local = await _verify_local_embedding_nim(config, settings)
            if local[0]:
                return local
        if provider_setting != "auto":
            return False, "none", None, None

    # Hosted arm — "auto" reaches here when no healthy local NIM
    # answered. We consult the runtime override layer first so a
    # UI-applied key takes effect on the next probe without a backend
    # restart. The caller-supplied ``settings`` is the fallback —
    # preserves test ergonomics that construct an ad-hoc Settings here.
    nvidia_api_key = get_effective_secret("NVIDIA_API_KEY", settings)
    if provider_setting == "auto":
        if not nvidia_api_key:
            return False, "none", None, None
        provider_setting = "hosted_nvclip"

    if provider_setting == "hosted_nvclip":
        try:
            auth = nim_client.build_auth_headers(
                auth_mode="bearer",
                credential=nvidia_api_key,
            )
        except ValueError:
            return False, "none", None, None

        result = await nim_client.create_embeddings(
            base_url=settings.HOSTED_NIM_BASE_URL,
            auth_headers=auth,
            model=settings.EMBEDDING_MODEL_ID,
            input_items=[PROBE_TEXT_INPUT],
            deadline_s=30.0,
            max_retries=2,
            input_type=settings.EMBEDDING_INPUT_TYPE,
        )

        if result.success and result.embeddings:
            actual_dim = len(result.embeddings[0])
            if actual_dim == settings.EMBEDDING_DIM:
                return (
                    True,
                    "hosted_nvclip",
                    settings.EMBEDDING_MODEL_ID,
                    settings.EMBEDDING_DIM,
                )
            logger.warning(
                "Embedding probe dimension mismatch: expected %d, got %d",
                settings.EMBEDDING_DIM,
                actual_dim,
            )
            return False, "none", None, None

        logger.warning("Hosted embedding probe failed: %s", result.error)
        return False, "none", None, None

    # Unrecognized EMBEDDING_PROVIDER value — treat as disabled.
    return False, "none", None, None


# ── Project-level probe integration ─────────────────────────────


async def probe_and_set_embedding_provider(
    project_id: str,
    engine: Any,
    settings: Settings,
) -> Project | None:
    """Run the embedding endpoint probe and update the Project record.

    Returns the updated Project object, or ``None`` if the project
    was not found.
    """
    _success, provider, model_id, dim = await probe_nvclip(settings)

    with Session(engine) as session:
        project = session.execute(
            select(Project).where(Project.project_id == project_id)
        ).scalar_one_or_none()
        if project is None:
            return None

        project.embedding_provider = provider
        project.embedding_model_id = model_id
        project.embedding_dim = dim
        session.commit()
        session.refresh(project)
        session.expunge(project)
        return project


# ── Background embedding worker ─────────────────────────────────────────────


async def _embedding_worker(
    project_id: str,
    workspace_root: str,
    settings: Settings,
) -> None:
    """Background coroutine that computes CLIP embeddings for a project.

    Multi-pass: after each batch sweep completes, the worker re-queries
    the project DB for newly-ingested examples that arrived while the
    previous sweep was running, and processes those too. Loops until a
    sweep finds nothing pending. This closes the multi-batch UI ingest
    race: ``trigger_embedding_computation``'s dedup guard silently drops
    trigger calls while a worker is already registered, so a single-pass
    worker that snapshotted its pending set up front would never see the
    additional batches.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        logger.warning("Embedding worker: project %s not found", project_id)
        return

    # Gate: feature flag
    if not settings.EMBEDDINGS_AUTO_COMPUTE:
        logger.info(
            "Embedding worker: EMBEDDINGS_AUTO_COMPUTE=False, skipping %s",
            project_id,
        )
        return

    # Read current provider from project
    with Session(engine) as session:
        project = session.execute(
            select(Project).where(Project.project_id == project_id)
        ).scalar_one_or_none()
        if project is None:
            return
        provider = project.embedding_provider
        model_id = project.embedding_model_id
        dim = project.embedding_dim

    # First-ingest re-probe: if provider is "none" but settings say we should
    # have embeddings, attempt probe again
    if provider == "none" and settings.EMBEDDING_PROVIDER != "none":
        updated = await probe_and_set_embedding_provider(project_id, engine, settings)
        if updated is None:
            return
        provider = updated.embedding_provider
        model_id = updated.embedding_model_id
        dim = updated.embedding_dim

    if provider == "none" or dim is None or model_id is None:
        logger.info(
            "Embedding worker: no embedding provider for project %s",
            project_id,
        )
        return

    # Per-provider endpoint resolution: a self-hosted NIM is served at
    # the endpoint recorded on EmbeddingDeploymentConfig with no auth;
    # the hosted endpoint requires NVIDIA_API_KEY.
    resolved = _resolve_embedding_endpoint(provider, settings)

    # Self-heal: a stamped provider that can no longer be served — the
    # local embedding NIM died and its config was reset (or drifted)
    # since this project was stamped. Re-run the probe cascade and
    # continue with whatever it lands on (hosted when a key is present)
    # instead of stranding the project on a dead stamp forever.
    if resolved is None or _self_hosted_stamp_out_of_sync(provider, settings):
        logger.info(
            "Embedding worker: stamped provider %s is no longer serveable "
            "for %s — re-running the probe cascade",
            provider,
            project_id,
        )
        updated = await probe_and_set_embedding_provider(project_id, engine, settings)
        if updated is None:
            return
        provider = updated.embedding_provider
        model_id = updated.embedding_model_id
        dim = updated.embedding_dim
        if provider == "none" or dim is None or model_id is None:
            logger.info(
                "Embedding worker: no embedding provider for project %s after re-probe",
                project_id,
            )
            return
        resolved = _resolve_embedding_endpoint(provider, settings)
        if resolved is None:
            logger.warning(
                "Embedding worker: cannot resolve endpoint for provider %s on %s",
                provider,
                project_id,
            )
            return
    base_url, auth, model_id = resolved

    # Provider-aware concurrency + batch shape.
    concurrency, batch_size = _resolve_worker_shape(provider, settings)

    # Cumulative counters across passes (for the final completion log).
    cumulative_processed = 0
    cumulative_total = 0
    pass_index = 0

    # ``failure`` is shared across passes — one hard NIM API failure stops
    # the whole worker so the operator sees a single ``run_failed`` SSE.
    failure: dict[str, str] = {}

    # Keys we've already attempted in this worker invocation. Used to
    # exclude examples that didn't write a ClipEmbedding row (unreadable
    # files, dim mismatches) from the rescan, otherwise they'd appear
    # in every subsequent pass and the loop would never terminate.
    # On worker restart they get a fresh attempt automatically.
    attempted_keys: set[str] = set()

    while True:
        if background_manager.is_shutting_down() or failure:
            break

        # Re-query each pass: catches examples ingested while the previous
        # pass was running (the dispatch race described in the docstring).
        with Session(engine) as session:
            all_examples = session.execute(
                select(
                    Example.example_key,
                    Example.phash,
                    Example.storage_ref,
                ).where(Example.project_id == project_id)
            ).all()

            embedded_keys: set[str] = set(
                session.execute(
                    select(ClipEmbedding.example_key).where(
                        ClipEmbedding.project_id == project_id
                    )
                )
                .scalars()
                .all()
            )

        example_map: dict[str, str] = {}  # key → storage_ref
        remaining: list[tuple[str, str | None]] = []
        for key, phash, storage_ref in all_examples:
            if key in embedded_keys or key in attempted_keys:
                continue
            remaining.append((key, phash))
            example_map[key] = storage_ref

        if not remaining:
            if pass_index == 0:
                logger.info(
                    "Embedding worker: all examples already embedded for %s",
                    project_id,
                )
            break

        pass_index += 1
        pass_total = len(remaining)
        cumulative_total += pass_total
        if pass_index == 1:
            logger.info(
                "Embedding worker: %d examples to embed for project %s",
                pass_total,
                project_id,
            )
            logger.info(
                "Embedding worker: provider=%s, concurrency=%d, batch_size=%d",
                provider,
                concurrency,
                batch_size,
            )
        else:
            logger.info(
                "Embedding worker: pass %d picked up %d more examples for %s "
                "(ingested while previous pass ran)",
                pass_index,
                pass_total,
                project_id,
            )

        # Embed in a deterministic, project-seeded random order over THIS
        # pass's snapshot. A random prefix is a representative sample of the
        # dataset at every coverage level, so the review selector's
        # diversity picks span all classes during the embedding warmup
        # instead of being trapped on whichever class ingest loaded first
        # (see ``_embedding_order``).
        ordered_keys = await run_in_thread(_embedding_order, remaining, project_id)

        # Ensure cache is initialized
        await run_in_thread(ensure_embedding_cache_loaded, project_id, engine, dim)

        # Pre-slice batches once.
        batches: list[list[str]] = [
            ordered_keys[i : i + batch_size]
            for i in range(0, len(ordered_keys), batch_size)
        ]

        semaphore = asyncio.Semaphore(concurrency)
        # Cross-worker per-project write lock from
        # ``services.project_db_locks``: serializes this worker's own
        # concurrent batches AND writes from the :ingest endpoint and
        # the background ingest sweeper. A worker-local asyncio.Lock is not
        # enough — concurrent writers on those other paths still trip
        # SQLite's "OperationalError: database is locked" under
        # multi-worker load.
        from vlm_feedback_loop.services.project_db_locks import (
            get_project_write_lock,
        )

        write_lock = get_project_write_lock(project_id)
        progress_lock = asyncio.Lock()  # serialize the processed counter
        progress = {"processed": 0}

        async def _process_batch(batch_keys: list[str]) -> None:
            if background_manager.is_shutting_down() or failure:
                return
            async with semaphore:
                if background_manager.is_shutting_down() or failure:
                    return

                # Mark these keys attempted before any I/O. If reads fail
                # or dims mismatch, the rescan still excludes them so the
                # multi-pass loop terminates.
                attempted_keys.update(batch_keys)

                # Prepare images as base64 data URLs (per-batch local state).
                input_items: list[str] = []
                valid_keys: list[str] = []
                local_skip = 0
                for key in batch_keys:
                    storage_ref = example_map.get(key)
                    if storage_ref is None:
                        local_skip += 1
                        continue
                    try:
                        data_url = await run_in_low_priority_thread(
                            _prepare_embedding_input,
                            storage_ref,
                        )
                        input_items.append(data_url)
                        valid_keys.append(key)
                    except Exception:
                        logger.warning(
                            "Embedding worker: failed to read %s, skipping",
                            key,
                            exc_info=True,
                        )
                        local_skip += 1

                if not input_items:
                    if local_skip:
                        async with progress_lock:
                            progress["processed"] += local_skip
                    return

                # Embedding is a background workload class: hold this NEW
                # HTTP dispatch while a foreground request (interactive
                # proposal / retry / rationale regeneration) is in flight.
                # Already-in-flight batches complete normally — same
                # integration point as evaluation and batch labeling.
                await priority_dispatch.wait_for_background()
                if background_manager.is_shutting_down() or failure:
                    return

                # Call NIM embeddings API (HTTP request, no DB lock held).
                result = await nim_client.create_embeddings(
                    base_url=base_url,
                    auth_headers=auth,
                    model=model_id,
                    input_items=input_items,
                    deadline_s=settings.HTTP_DEADLINE_BACKGROUND_S,
                    input_type=settings.EMBEDDING_INPUT_TYPE,
                )

                if not result.success or result.embeddings is None:
                    logger.error(
                        "Embedding worker: NIM API failed for batch (%d items): %s",
                        len(input_items),
                        result.error,
                    )
                    if not failure:
                        failure["summary"] = result.error or "Embedding API call failed"
                    return

                # Persist embeddings — short write transaction under shared
                # lock so concurrent batches don't trip SQLite busy_timeout.
                async with write_lock:
                    with Session(engine) as session:
                        for i, key in enumerate(valid_keys):
                            if i >= len(result.embeddings):
                                break
                            vec = result.embeddings[i]
                            if len(vec) != dim:
                                logger.warning(
                                    "Embedding dim mismatch for %s: expected %d, got %d",
                                    key,
                                    dim,
                                    len(vec),
                                )
                                continue

                            blob = serialize_vector(vec, dim)

                            existing = session.execute(
                                select(ClipEmbedding).where(
                                    ClipEmbedding.project_id == project_id,
                                    ClipEmbedding.example_key == key,
                                )
                            ).scalar_one_or_none()

                            if existing is not None:
                                existing.vector_blob_f32 = blob
                                existing.embedding_provider = provider
                                existing.clip_embedding_model_id = model_id
                                existing.clip_embedding_dim = dim
                            else:
                                session.add(
                                    ClipEmbedding(
                                        project_id=project_id,
                                        example_key=key,
                                        embedding_provider=provider,
                                        clip_embedding_model_id=model_id,
                                        clip_embedding_dim=dim,
                                        vector_blob_f32=blob,
                                    )
                                )

                            ex = session.execute(
                                select(Example).where(
                                    Example.project_id == project_id,
                                    Example.example_key == key,
                                )
                            ).scalar_one_or_none()
                            if ex is not None:
                                ex.clip_embedding_present = True
                                ex.clip_embedding_dim = dim
                                ex.clip_embedding_model_id = model_id
                                ex.embedding_provider = provider

                            embedding_cache.put(project_id, key, vec)

                        session.commit()

                async with progress_lock:
                    progress["processed"] += len(batch_keys)
                    processed_now = progress["processed"]

                await sse_manager.emit(
                    project_id,
                    "embedding_progress",
                    {"processed": min(processed_now, pass_total), "total": pass_total},
                )

                await asyncio.sleep(INTER_BATCH_SLEEP_S)

        await asyncio.gather(*(_process_batch(b) for b in batches))
        cumulative_processed += progress["processed"]
        # Loop top will rescan for newly-arrived examples.

    if failure:
        await sse_manager.emit(
            project_id,
            "run_failed",
            {"run_type": "embedding", "error_summary": failure["summary"]},
        )
        return

    # Completion event (single emit covering all passes)
    if not background_manager.is_shutting_down():
        await sse_manager.emit(
            project_id,
            "embedding_completed",
            {"processed": cumulative_processed, "total": cumulative_total},
        )
        if pass_index == 0:
            logger.info(
                "Embedding worker: completed 0/0 for project %s (nothing pending)",
                project_id,
            )
        elif pass_index == 1:
            logger.info(
                "Embedding worker: completed %d/%d for project %s",
                cumulative_processed,
                cumulative_total,
                project_id,
            )
        else:
            logger.info(
                "Embedding worker: completed %d/%d for project %s "
                "across %d passes (caught examples ingested mid-run)",
                cumulative_processed,
                cumulative_total,
                project_id,
                pass_index,
            )


# ── Trigger and recovery ────────────────────────────────────────────────────


def trigger_embedding_computation(
    project_id: str,
    workspace_root: str,
    settings: Settings,
) -> None:
    """Trigger background CLIP embedding computation for a project.

    Non-blocking — returns immediately.  Deduplicates by task ID.
    """
    if not settings.EMBEDDINGS_AUTO_COMPUTE:
        return

    background_manager.try_register(
        f"clip-embed-{project_id}",
        _embedding_worker(project_id, workspace_root, settings),
        no_loop_warning=(
            f"Embedding worker NOT scheduled for {project_id}: caller has no "
            "running event loop. Async route handler required for embedding "
            "trigger to work; worker will recover on next backend restart."
        ),
    )


def get_embedding_status(
    project_id: str, workspace_root: str, settings: Settings
) -> dict[str, Any] | None:
    """Embedding-completion status for a project (the REST counterpart to the
    ``embedding_progress`` / ``embedding_completed`` SSE events).

    Returns ``None`` when the project DB is missing. Otherwise a dict::

        {
          "total_examples": int,    # rows in ``examples``
          "embedded": int,          # rows in ``clip_embeddings``
          "pending": int,           # max(0, total - embedded)
          "worker_active": bool,    # the clip-embed-<id> background task is running
          "auto_compute": bool,     # settings.EMBEDDINGS_AUTO_COMPUTE
          "provider": str | None,   # project.embedding_provider (None until probed)
          "model_id": str | None,
          "dim": int | None,
          "complete": bool,         # see below
        }

    ``complete`` is True when every example is embedded and no worker is still
    running (``embedded >= total_examples and not worker_active`` — an empty
    project, ``total_examples == 0``, is trivially complete). It is also True
    when embeddings are not going to be computed at all per OPERATOR settings —
    ``EMBEDDINGS_AUTO_COMPUTE`` off, or ``EMBEDDING_PROVIDER == "none"`` — so an
    embeddings-first barrier (``autorun --wait-embeddings``) treats a pHash-only
    deployment as "done waiting" instead of hanging until timeout. (The
    *project row's* ``embedding_provider`` is reported as ``provider`` for
    diagnostics but is NOT used for the disabled decision — it is ``"none"``
    transiently until the worker probes it, which would otherwise make the
    barrier proceed before the first probe.)
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return None
        total = session.query(Example).filter_by(project_id=project_id).count()
        embedded = session.query(ClipEmbedding).filter_by(project_id=project_id).count()
        provider = project.embedding_provider
        model_id = project.embedding_model_id
        dim = project.embedding_dim

    worker_active = f"clip-embed-{project_id}" in background_manager.active_task_ids
    # "Disabled" is an OPERATOR-intent decision read from settings — NOT from
    # ``project.embedding_provider``, which is ``"none"`` transiently until the
    # worker probes and sets it (basing the barrier on the project row would
    # make it proceed prematurely, before the first probe). Mirrors
    # ``recover_embedding_tasks``'s ``operator_wants_embeddings`` test.
    embeddings_disabled = (not settings.EMBEDDINGS_AUTO_COMPUTE) or (
        settings.EMBEDDING_PROVIDER == "none"
    )
    all_embedded = embedded >= total and not worker_active
    complete = bool(embeddings_disabled or all_embedded)

    return {
        "total_examples": total,
        "embedded": embedded,
        "pending": max(0, total - embedded),
        "worker_active": worker_active,
        "auto_compute": bool(settings.EMBEDDINGS_AUTO_COMPUTE),
        "provider": provider,
        "model_id": model_id,
        "dim": dim,
        "complete": complete,
    }


# A permanently failing provider must be retried politely, not hot-looped:
# the autorun ``--wait-embeddings`` barrier polls the status endpoint every
# few seconds, and each restarted worker fires real HTTP batches before
# giving up. One restart per cooldown window per project caps that.
_SELF_HEAL_REKICK_COOLDOWN_S = 60.0
_last_self_heal_rekick: dict[str, float] = {}  # project_id → time.monotonic()


def maybe_restart_dead_embedding_worker(
    project_id: str,
    workspace_root: str,
    settings: Settings,
    status: dict[str, Any],
) -> bool:
    """Re-arm the embedding worker when a drain died mid-flight.

    The worker treats one hard provider failure as terminal for the whole
    drain (a single ``run_failed`` SSE), and every other recovery trigger
    is event-coupled — backend restart, a new ingest, or a local embedding
    NIM lifecycle change. A hosted-provider drain that dies mid-flight has
    none of those, so the project would otherwise sit indefinitely with a
    partially embedded pool. Called from the ``:embedding_status`` route,
    making any status poll (the autorun ``--wait-embeddings`` barrier, an
    operator curl, a UI refresh) the recovery path.

    ``status`` is the dict returned by ``get_embedding_status``. Restarts
    are skipped when nothing is pending, a worker is already running, the
    operator disabled embeddings (same intent test as
    ``recover_embedding_tasks``), or a restart already happened within the
    cooldown window. Returns True when a worker restart was triggered.
    """
    if status["pending"] <= 0 or status["worker_active"]:
        return False
    if not settings.EMBEDDINGS_AUTO_COMPUTE or settings.EMBEDDING_PROVIDER == "none":
        return False

    now = time.monotonic()
    last = _last_self_heal_rekick.get(project_id)
    if last is not None and now - last < _SELF_HEAL_REKICK_COOLDOWN_S:
        return False
    _last_self_heal_rekick[project_id] = now

    logger.info(
        "Embedding self-heal: %d pending examples and no worker for project "
        "%s — restarting the embedding worker",
        status["pending"],
        project_id,
    )
    trigger_embedding_computation(project_id, workspace_root, settings)
    return True


def _pending_embedding_count(session: Session, project_id: str) -> int:
    """Examples without a ``ClipEmbedding`` row — the worker's pending set."""
    total = session.query(Example).filter_by(project_id=project_id).count()
    embedded = session.query(ClipEmbedding).filter_by(project_id=project_id).count()
    return max(0, total - embedded)


def _walk_embedding_projects(
    workspace_root: str,
    per_project: Callable[[str], None],
) -> None:
    """Run ``per_project(project_id)`` for every non-archived project.

    Shared walk for startup recovery and the provider resweep. Archived
    projects are skipped — pictures don't change while a project is
    paused, and the marker-file gate keeps the scan O(active projects).
    Per-project isolation: a single corrupt project DB MUST NOT prevent
    backend startup — it is logged and skipped, never aborting the walk.
    See services/local_nim_service.py ``recover_local_deployments`` for
    the same pattern.
    """
    projects_dir = projects_root(workspace_root)
    if not projects_dir.exists():
        return

    for entry in sorted(projects_dir.iterdir()):
        if not entry.is_dir():
            continue
        if not (entry / "project.db").exists():
            continue
        if (entry / ".archived").exists():
            continue

        project_id = entry.name
        try:
            per_project(project_id)
        except Exception as exc:
            logger.warning(
                "Skipping embedding recovery for project %s (%s: %s)",
                project_id,
                type(exc).__name__,
                exc or "(no message)",
            )
            continue


async def recover_embedding_tasks(settings: Settings) -> None:
    """Recover incomplete CLIP embedding tasks on startup.

    Scans all projects and triggers embedding workers for projects
    that have unembedded examples and a valid provider.
    """
    workspace_root = settings.WORKSPACE_ROOT

    # Operator-level intent: if EMBEDDING_PROVIDER is "none" the operator
    # explicitly disabled embeddings, so don't second-guess that.
    operator_wants_embeddings = (
        settings.EMBEDDING_PROVIDER != "none" and settings.EMBEDDINGS_AUTO_COMPUTE
    )

    def _recover_one(project_id: str) -> None:
        engine = get_project_engine(project_id, workspace_root)
        if engine is None:
            return

        with Session(engine) as session:
            project = session.execute(
                select(Project).where(Project.project_id == project_id)
            ).scalar_one_or_none()
            if project is None:
                return

            # Skip only when the project provider is "none" AND the operator
            # disabled embeddings. When the project provider is "none" but
            # the operator wants embeddings, fall through and trigger the
            # worker so it re-probes (see ``_embedding_worker``'s
            # first-ingest re-probe). Retry-on-restart makes the probe
            # self-healing: a probe that failed because the upstream
            # endpoint was degraded at project creation is retried on the
            # next backend startup while the operator's images still need
            # embedding.
            if project.embedding_provider == "none" and not operator_wants_embeddings:
                return

            pending = _pending_embedding_count(session, project_id)

        if pending > 0:
            logger.info(
                "Recovery: %d unembedded examples in project %s, triggering worker",
                pending,
                project_id,
            )
            trigger_embedding_computation(project_id, workspace_root, settings)

    _walk_embedding_projects(workspace_root, _recover_one)


async def resweep_embedding_tasks(settings: Settings) -> None:
    """Re-resolve the embedding provider for every non-archived project.

    Runs after a deployment-level provider change — a local embedding
    NIM turning healthy, or its endpoint dying. Probes once (the outcome
    is project-independent), re-stamps projects whose provider drifted,
    and restarts workers where examples are still unembedded. Stored
    ``ClipEmbedding`` rows are never touched here: invalidation is keyed
    on model identity, never on provider, and the hosted and local
    endpoints serve the same model at the same dimension — existing
    vectors stay valid across an endpoint switch.
    """
    workspace_root = settings.WORKSPACE_ROOT
    _success, provider, model_id, dim = await probe_nvclip(settings)

    def _resweep_one(project_id: str) -> None:
        engine = get_project_engine(project_id, workspace_root)
        if engine is None:
            return

        with Session(engine) as session:
            project = session.execute(
                select(Project).where(Project.project_id == project_id)
            ).scalar_one_or_none()
            if project is None:
                return

            if project.embedding_provider != provider:
                logger.info(
                    "Resweep: project %s embedding provider %s -> %s",
                    project_id,
                    project.embedding_provider,
                    provider,
                )
                project.embedding_provider = provider
                project.embedding_model_id = model_id
                project.embedding_dim = dim
                session.commit()

            pending = _pending_embedding_count(session, project_id)

        if provider != "none" and pending > 0:
            trigger_embedding_computation(project_id, workspace_root, settings)

    _walk_embedding_projects(workspace_root, _resweep_one)


def ensure_embedding_cache_loaded(
    project_id: str,
    engine: Any,
    dim: int | None,
) -> None:
    """Idempotent guard-then-load for the per-project embedding cache.

    The ONE loader every consumer routes through (review selector, pool
    router, ICL selection, and the embedding worker) — the guard+load
    pair used to be inlined at each site. No-op when the cache is
    already resident or the project has no ``embedding_dim`` yet.
    Synchronous (bulk DB read); async callers wrap it in
    ``run_in_thread``.
    """
    if dim is None or embedding_cache.is_loaded(project_id):
        return
    embedding_cache.load(project_id, engine, dim)

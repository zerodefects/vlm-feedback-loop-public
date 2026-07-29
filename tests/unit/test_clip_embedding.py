# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLIP embedding computation, storage, and cache."""

from __future__ import annotations

import socket
import struct
from pathlib import Path
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import (
    create_project_via_api,
    make_api_client,
    make_settings,
    make_test_image,
    open_project_workspace,
)
from mock_nim_server import MockNIMServer
from vlm_feedback_loop.model_catalog_constants import (
    EMBEDDING_MODEL_ID,
)
from vlm_feedback_loop.services.clip_embedding_service import (
    _EmbeddingCache,
    deserialize_vector,
    phash_hamming_distance,
    serialize_vector,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _fake_embeddings_result(dim: int = 2048, count: int = 1):
    """Build a mock NimEmbeddingsResult."""
    from vlm_feedback_loop.services.nim_client import NimEmbeddingsResult

    embeddings = [[float(i + j) / 1000 for j in range(dim)] for i in range(count)]
    return NimEmbeddingsResult(
        success=True,
        embeddings=embeddings,
        model=EMBEDDING_MODEL_ID,
        usage={"prompt_tokens": 10, "total_tokens": 10},
    )


def _fake_probe_fail():
    from vlm_feedback_loop.services.nim_client import NimEmbeddingsResult

    return NimEmbeddingsResult(
        success=False,
        error="Connection refused",
        status_code=503,
    )


def _seed_embedding_deployment_config(
    workspace_root: Path | str,
    *,
    endpoint_url: str | None,
    provider: str = "self_hosted_nvclip",
    model_name: str = EMBEDDING_MODEL_ID,
    dim: int = 2048,
) -> None:
    """Stamp the deployment.db singleton the way the deploy flow's
    healthy transition does (provider + endpoint_url)."""
    from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig
    from vlm_feedback_loop.db.engine import init_deployment_db

    engine = init_deployment_db(workspace_root)
    with Session(engine) as session:
        config = session.execute(select(EmbeddingDeploymentConfig)).scalar_one()
        config.provider = provider
        config.endpoint_url = endpoint_url
        config.model_name = model_name
        config.embedding_dim = dim
        session.commit()


def _dead_endpoint_url() -> str:
    """A localhost URL that refuses connections (bound-then-released port)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def _lower_headers(request: dict) -> dict[str, str]:
    """Captured wire headers with case-insensitive (lowercase) names."""
    return {k.lower(): v for k, v in request["headers"].items()}


async def _noop_backoff(attempt_index: int, max_wait: float | None = None) -> None:
    """No-op the retry backoff so dead-endpoint probes fail fast."""
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Section A: Vector serialization
# ═══════════════════════════════════════════════════════════════════════════


class TestSerializeVector:
    def test_round_trip(self):
        vec = [float(i) for i in range(1024)]
        blob = serialize_vector(vec, 1024)
        assert len(blob) == 1024 * 4
        recovered = deserialize_vector(blob, 1024)
        assert recovered == vec

    def test_small_vector(self):
        vec = [1.0, 2.0, 3.0]
        blob = serialize_vector(vec, 3)
        assert len(blob) == 12
        assert deserialize_vector(blob, 3) == [1.0, 2.0, 3.0]

    def test_dimension_mismatch_serialize(self):
        with pytest.raises(ValueError, match="does not match"):
            serialize_vector([1.0, 2.0], 3)

    def test_dimension_mismatch_deserialize(self):
        blob = struct.pack("2f", 1.0, 2.0)
        with pytest.raises(ValueError, match="does not match"):
            deserialize_vector(blob, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Section B: pHash hamming distance and diverse ordering
# ═══════════════════════════════════════════════════════════════════════════


class TestPHashHammingDistance:
    def test_identical_hashes(self):
        assert phash_hamming_distance("0000000000000000", "0000000000000000") == 0

    def test_one_bit_different(self):
        assert phash_hamming_distance("0000000000000000", "0000000000000001") == 1

    def test_all_bits_different(self):
        assert phash_hamming_distance("0000000000000000", "ffffffffffffffff") == 64

    def test_symmetric(self):
        a = "abcdef0123456789"
        b = "1234567890abcdef"
        assert phash_hamming_distance(a, b) == phash_hamming_distance(b, a)


# ═══════════════════════════════════════════════════════════════════════════
# Section C: In-memory embedding cache
# ═══════════════════════════════════════════════════════════════════════════


class TestEmbeddingOrder:
    """The embedding feed order spreads across the dataset so the review
    selector's diversity picks span all classes during the embedding
    warmup, rather than being trapped on whichever class ingest loaded
    first (datasets are stored one directory per class)."""

    def test_class_clustered_ingest_is_spread(self):
        from vlm_feedback_loop.services.clip_embedding_service import _embedding_order

        # Ingest order = class-blocked: all of "a", then "b", then "c".
        examples = (
            [(f"a/img{i}", "0" * 16) for i in range(100)]
            + [(f"b/img{i}", "0" * 16) for i in range(100)]
            + [(f"c/img{i}", "0" * 16) for i in range(100)]
        )
        order = _embedding_order(examples, "proj-1")
        # Ingest order would make the first 100 entirely class "a"; the
        # spread order pulls from all three classes early.
        first_hundred = {k.split("/")[0] for k in order[:100]}
        assert first_hundred == {"a", "b", "c"}

    def test_deterministic_per_project(self):
        from vlm_feedback_loop.services.clip_embedding_service import _embedding_order

        examples = [(f"k{i}", None) for i in range(50)]
        assert _embedding_order(list(examples), "p") == _embedding_order(
            list(examples), "p"
        )

    def test_different_project_different_order(self):
        from vlm_feedback_loop.services.clip_embedding_service import _embedding_order

        examples = [(f"k{i}", None) for i in range(50)]
        assert _embedding_order(list(examples), "p1") != _embedding_order(
            list(examples), "p2"
        )

    def test_preserves_all_keys(self):
        from vlm_feedback_loop.services.clip_embedding_service import _embedding_order

        examples = [(f"k{i}", "0" * 16) for i in range(30)]
        order = _embedding_order(examples, "p")
        assert sorted(order) == sorted(k for k, _ in examples)


class TestEmbeddingCache:
    def test_put_and_get(self):
        cache = _EmbeddingCache()
        cache.put("p1", "k1", [1.0, 2.0])
        assert cache.get("p1", "k1") == [1.0, 2.0]

    def test_get_missing(self):
        cache = _EmbeddingCache()
        assert cache.get("p1", "k1") is None

    def test_count(self):
        cache = _EmbeddingCache()
        assert cache.count("p1") == 0
        cache.put("p1", "k1", [1.0])
        cache.put("p1", "k2", [2.0])
        assert cache.count("p1") == 2

    def test_get_all(self):
        cache = _EmbeddingCache()
        cache.put("p1", "k1", [1.0])
        cache.put("p1", "k2", [2.0])
        all_vecs = cache.get_all("p1")
        assert len(all_vecs) == 2
        assert all_vecs["k1"] == [1.0]

    def test_get_normalized_matrix(self):
        """The matrix view returns L2-normalized rows keyed by a row index;
        a zero vector stays zero so its cosine is 0.0 against everything."""
        cache = _EmbeddingCache()
        cache.put("p1", "a", [3.0, 4.0])  # norm 5 → [0.6, 0.8]
        cache.put("p1", "b", [0.0, 0.0])  # zero-magnitude → stays zero
        index, matrix = cache.get_normalized_matrix("p1")
        assert set(index) == {"a", "b"}
        assert matrix.shape == (2, 2)
        row_a = matrix[index["a"]]
        assert row_a[0] == pytest.approx(0.6)
        assert row_a[1] == pytest.approx(0.8)
        row_b = matrix[index["b"]]
        assert row_b[0] == 0.0 and row_b[1] == 0.0

    def test_get_normalized_matrix_invalidated_on_put(self):
        """A put after the matrix is built serves a rebuilt matrix, never a
        stale one — otherwise a freshly embedded image would be invisible to
        diversity scoring."""
        cache = _EmbeddingCache()
        cache.put("p1", "a", [1.0, 0.0])
        index1, _ = cache.get_normalized_matrix("p1")
        assert set(index1) == {"a"}
        cache.put("p1", "b", [0.0, 1.0])
        index2, matrix2 = cache.get_normalized_matrix("p1")
        assert set(index2) == {"a", "b"}
        assert matrix2.shape == (2, 2)

    def test_get_normalized_matrix_empty(self):
        """An unknown project yields an empty index and a (0, 0) matrix
        rather than raising."""
        cache = _EmbeddingCache()
        index, matrix = cache.get_normalized_matrix("nope")
        assert index == {}
        assert matrix.shape == (0, 0)

    def test_is_loaded(self):
        cache = _EmbeddingCache()
        assert not cache.is_loaded("p1")
        cache.put("p1", "k1", [1.0])
        assert cache.is_loaded("p1")

    def test_load_from_db(self, tmp_path: Path):
        """Verify load() reads from ClipEmbedding table."""
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
        from vlm_feedback_loop.db.models.example import Example

        project_dir = tmp_path / "projects" / "test-proj"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        dim = 4
        vec = [1.0, 2.0, 3.0, 4.0]
        blob = serialize_vector(vec, dim)

        with Session(engine) as session:
            session.add(
                Example(
                    example_key="k1",
                    project_id="test-proj",
                    storage_ref="/fake",
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                )
            )
            # In production, ingestion commits the Example before the
            # background worker derives its embedding.
            session.flush()
            session.add(
                ClipEmbedding(
                    project_id="test-proj",
                    example_key="k1",
                    embedding_provider="hosted_nvclip",
                    clip_embedding_model_id=EMBEDDING_MODEL_ID,
                    clip_embedding_dim=dim,
                    vector_blob_f32=blob,
                )
            )
            session.commit()

        cache = _EmbeddingCache()
        cache.load("test-proj", engine, dim)
        assert cache.is_loaded("test-proj")
        assert cache.count("test-proj") == 1
        loaded = cache.get("test-proj", "k1")
        assert loaded == vec


# ═══════════════════════════════════════════════════════════════════════════
# Section C.5: Provider-aware worker shape
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveWorkerShape:
    """``_resolve_worker_shape`` picks (concurrency, batch_size) by provider."""

    def _settings(self, **overrides: object):
        from vlm_feedback_loop.config import Settings

        defaults = {
            "WORKSPACE_ROOT": "/tmp/wsroot",
            "EMBEDDING_CONCURRENCY_HOSTED": 1,
            "EMBEDDING_BATCH_SIZE_HOSTED": 8,
            "EMBEDDING_CONCURRENCY_SELF_HOSTED": 4,
            "EMBEDDING_BATCH_SIZE_SELF_HOSTED": 1,
        }
        defaults.update(overrides)
        return Settings(**defaults)  # type: ignore[arg-type]

    def test_hosted_uses_hosted_pair(self):
        from vlm_feedback_loop.services.clip_embedding_service import (
            _resolve_worker_shape,
        )

        concurrency, batch_size = _resolve_worker_shape(
            "hosted_nvclip", self._settings()
        )
        assert (concurrency, batch_size) == (1, 8)

    def test_self_hosted_uses_self_hosted_pair(self):
        from vlm_feedback_loop.services.clip_embedding_service import (
            _resolve_worker_shape,
        )

        concurrency, batch_size = _resolve_worker_shape(
            "self_hosted_nvclip", self._settings()
        )
        assert (concurrency, batch_size) == (4, 1)

    def test_local_nvclip_treated_as_self_hosted(self):
        from vlm_feedback_loop.services.clip_embedding_service import (
            _resolve_worker_shape,
        )

        concurrency, batch_size = _resolve_worker_shape(
            "local_nvclip", self._settings()
        )
        assert (concurrency, batch_size) == (4, 1)

    def test_operator_overrides_respected(self):
        """Operator can tune both knobs independently per provider."""
        from vlm_feedback_loop.services.clip_embedding_service import (
            _resolve_worker_shape,
        )

        s = self._settings(
            EMBEDDING_CONCURRENCY_HOSTED=2,
            EMBEDDING_BATCH_SIZE_HOSTED=16,
            EMBEDDING_CONCURRENCY_SELF_HOSTED=8,
            EMBEDDING_BATCH_SIZE_SELF_HOSTED=2,
        )
        assert _resolve_worker_shape("hosted_nvclip", s) == (2, 16)
        assert _resolve_worker_shape("self_hosted_nvclip", s) == (8, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Section D: NV-CLIP probe
# ═══════════════════════════════════════════════════════════════════════════


class TestProbeNvclip:
    @pytest.mark.asyncio
    async def test_probe_success_with_api_key(self, tmp_path: Path):
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        settings = make_settings(
            tmp_path, NVIDIA_API_KEY="nvapi-test", EMBEDDING_PROVIDER="auto"
        )
        mock_result = _fake_embeddings_result(dim=2048, count=1)

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(return_value=mock_result),
        ):
            success, provider, model_id, dim = await probe_nvclip(settings)

        assert success is True
        assert provider == "hosted_nvclip"
        assert model_id == EMBEDDING_MODEL_ID
        assert dim == 2048

    @pytest.mark.asyncio
    async def test_probe_no_api_key(self, tmp_path: Path):
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        settings = make_settings(
            tmp_path, NVIDIA_API_KEY=None, EMBEDDING_PROVIDER="auto"
        )
        success, provider, _, _ = await probe_nvclip(settings)
        assert success is False
        assert provider == "none"

    @pytest.mark.asyncio
    async def test_probe_failure(self, tmp_path: Path):
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        settings = make_settings(
            tmp_path, NVIDIA_API_KEY="nvapi-test", EMBEDDING_PROVIDER="auto"
        )

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(return_value=_fake_probe_fail()),
        ):
            success, provider, _, _ = await probe_nvclip(settings)

        assert success is False
        assert provider == "none"

    @pytest.mark.asyncio
    async def test_probe_provider_none_skips(self, tmp_path: Path):
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        settings = make_settings(tmp_path, EMBEDDING_PROVIDER="none")
        success, provider, _, _ = await probe_nvclip(settings)
        assert success is False
        assert provider == "none"

    @pytest.mark.asyncio
    async def test_probe_dimension_mismatch(self, tmp_path: Path):
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        settings = make_settings(
            tmp_path,
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDING_PROVIDER="auto",
            EMBEDDING_DIM=2048,
        )
        # Return embeddings with wrong dimension
        mock_result = _fake_embeddings_result(dim=512, count=1)

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(return_value=mock_result),
        ):
            success, provider, _, _ = await probe_nvclip(settings)

        assert success is False
        assert provider == "none"


# ═══════════════════════════════════════════════════════════════════════════
# Section D.5: Probe cascade — local embedding NIM
# ═══════════════════════════════════════════════════════════════════════════


class TestProbeCascadeLocalNim:
    """``EMBEDDING_PROVIDER="auto"`` prefers a healthy local embedding NIM,
    falls back to the hosted endpoint (NVIDIA_API_KEY), and lands on
    "none" (pHash diversity) as the last resort."""

    @pytest.mark.asyncio
    async def test_auto_prefers_healthy_local_nim(self, tmp_path: Path):
        """A healthy local deployment wins over a configured NVIDIA key,
        and the local probe request carries no bearer auth."""
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        with MockNIMServer() as local_mock:
            _seed_embedding_deployment_config(
                tmp_path, endpoint_url=local_mock.base_url
            )
            settings = make_settings(
                tmp_path, NVIDIA_API_KEY="nvapi-test", EMBEDDING_PROVIDER="auto"
            )
            success, provider, model_id, dim = await probe_nvclip(settings)
            requests = local_mock.embeddings_requests

        assert success is True
        assert provider == "self_hosted_nvclip"
        assert model_id == EMBEDDING_MODEL_ID
        assert dim == 2048
        assert len(requests) == 1
        headers = _lower_headers(requests[0])
        assert "authorization" not in headers
        assert headers.get("source") == "vlm-feedback-loop"

    @pytest.mark.asyncio
    async def test_auto_dead_local_falls_back_to_hosted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A stale config pointing at a dead endpoint degrades to hosted:
        the live verify fails and the key-based hosted probe wins."""
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        monkeypatch.setattr(
            "vlm_feedback_loop.services.http_client._backoff", _noop_backoff
        )
        with MockNIMServer() as hosted_mock:
            _seed_embedding_deployment_config(
                tmp_path, endpoint_url=_dead_endpoint_url()
            )
            settings = make_settings(
                tmp_path,
                NVIDIA_API_KEY="nvapi-test",
                EMBEDDING_PROVIDER="auto",
                HOSTED_NIM_BASE_URL=hosted_mock.base_url,
            )
            success, provider, model_id, dim = await probe_nvclip(settings)
            requests = hosted_mock.embeddings_requests

        assert success is True
        assert provider == "hosted_nvclip"
        assert model_id == EMBEDDING_MODEL_ID
        assert dim == 2048
        assert _lower_headers(requests[0]).get("authorization") == "Bearer nvapi-test"

    @pytest.mark.asyncio
    async def test_auto_dead_local_without_key_resolves_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """No healthy local NIM and no NVIDIA key → provider none."""
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        monkeypatch.setattr(
            "vlm_feedback_loop.services.http_client._backoff", _noop_backoff
        )
        _seed_embedding_deployment_config(tmp_path, endpoint_url=_dead_endpoint_url())
        settings = make_settings(
            tmp_path, NVIDIA_API_KEY=None, EMBEDDING_PROVIDER="auto"
        )
        success, provider, _, _ = await probe_nvclip(settings)
        assert success is False
        assert provider == "none"

    @pytest.mark.asyncio
    async def test_auto_ignores_endpoint_not_marked_healthy(self, tmp_path: Path):
        """ "auto" trusts only a config the deploy flow marked healthy: a
        recorded endpoint_url with provider!=self_hosted_nvclip is never
        probed."""
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        with MockNIMServer() as local_mock:
            _seed_embedding_deployment_config(
                tmp_path, endpoint_url=local_mock.base_url, provider="none"
            )
            settings = make_settings(
                tmp_path, NVIDIA_API_KEY=None, EMBEDDING_PROVIDER="auto"
            )
            success, provider, _, _ = await probe_nvclip(settings)
            assert local_mock.embeddings_requests == []

        assert success is False
        assert provider == "none"

    @pytest.mark.asyncio
    async def test_auto_local_dim_mismatch_falls_back(self, tmp_path: Path):
        """A local NIM serving the wrong dimension fails the live verify."""
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        with MockNIMServer() as local_mock:
            local_mock.set_embedding_dim(512)
            _seed_embedding_deployment_config(
                tmp_path, endpoint_url=local_mock.base_url, dim=2048
            )
            settings = make_settings(
                tmp_path, NVIDIA_API_KEY=None, EMBEDDING_PROVIDER="auto"
            )
            success, provider, _, _ = await probe_nvclip(settings)

        assert success is False
        assert provider == "none"

    @pytest.mark.asyncio
    async def test_explicit_self_hosted_pin_works_without_key(self, tmp_path: Path):
        """An explicit local pin needs no NVIDIA key at all."""
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        with MockNIMServer() as local_mock:
            _seed_embedding_deployment_config(
                tmp_path, endpoint_url=local_mock.base_url
            )
            settings = make_settings(
                tmp_path, NVIDIA_API_KEY=None, EMBEDDING_PROVIDER="self_hosted_nvclip"
            )
            success, provider, model_id, dim = await probe_nvclip(settings)

        assert success is True
        assert provider == "self_hosted_nvclip"
        assert model_id == EMBEDDING_MODEL_ID
        assert dim == 2048

    @pytest.mark.asyncio
    async def test_explicit_self_hosted_pin_dead_local_resolves_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """An explicit local pin never falls back to hosted on failure,
        even when a key is configured."""
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        monkeypatch.setattr(
            "vlm_feedback_loop.services.http_client._backoff", _noop_backoff
        )
        _seed_embedding_deployment_config(tmp_path, endpoint_url=_dead_endpoint_url())
        settings = make_settings(
            tmp_path,
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDING_PROVIDER="self_hosted_nvclip",
        )
        success, provider, _, _ = await probe_nvclip(settings)
        assert success is False
        assert provider == "none"

    @pytest.mark.asyncio
    async def test_explicit_hosted_pin_ignores_local(self, tmp_path: Path):
        """An explicit hosted pin never touches the local endpoint."""
        from vlm_feedback_loop.services.clip_embedding_service import probe_nvclip

        with MockNIMServer() as local_mock, MockNIMServer() as hosted_mock:
            _seed_embedding_deployment_config(
                tmp_path, endpoint_url=local_mock.base_url
            )
            settings = make_settings(
                tmp_path,
                NVIDIA_API_KEY="nvapi-test",
                EMBEDDING_PROVIDER="hosted_nvclip",
                HOSTED_NIM_BASE_URL=hosted_mock.base_url,
            )
            success, provider, _, _ = await probe_nvclip(settings)
            assert local_mock.embeddings_requests == []
            assert len(hosted_mock.embeddings_requests) == 1

        assert success is True
        assert provider == "hosted_nvclip"


# ═══════════════════════════════════════════════════════════════════════════
# Section E: Probe at project creation (cross-phase modification)
# ═══════════════════════════════════════════════════════════════════════════


class TestProjectCreationProbe:
    def test_creation_with_api_key_sets_provider(self, tmp_path: Path):
        """Project creation with NVIDIA_API_KEY probes NV-CLIP → hosted_nvclip."""
        mock_result = _fake_embeddings_result(dim=2048, count=1)

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(return_value=mock_result),
        ):
            client = make_api_client(
                tmp_path, NVIDIA_API_KEY="nvapi-test", EMBEDDING_PROVIDER="auto"
            )
            project = create_project_via_api(client)

        assert project["embedding_provider"] == "hosted_nvclip"
        assert project["embedding_model_id"] == EMBEDDING_MODEL_ID
        assert project["embedding_dim"] == 2048

    def test_creation_without_api_key_stays_none(self, tmp_path: Path):
        """Project creation without NVIDIA_API_KEY → embedding_provider=none."""
        client = make_api_client(
            tmp_path, NVIDIA_API_KEY=None, EMBEDDING_PROVIDER="auto"
        )
        project = create_project_via_api(client)
        assert project["embedding_provider"] in (None, "none")

    def test_creation_probe_failure_stays_none(self, tmp_path: Path):
        """Failed probe at creation → embedding_provider=none."""
        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(return_value=_fake_probe_fail()),
        ):
            client = make_api_client(
                tmp_path, NVIDIA_API_KEY="nvapi-test", EMBEDDING_PROVIDER="auto"
            )
            project = create_project_via_api(client)

        assert project["embedding_provider"] in (None, "none")


# ═══════════════════════════════════════════════════════════════════════════
# Section F: Background worker
# ═══════════════════════════════════════════════════════════════════════════


class TestBackgroundWorker:
    @pytest.mark.asyncio
    async def test_worker_processes_examples(self, tmp_path: Path):
        """Full worker run: creates ClipEmbedding rows and updates Example flags."""
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )

        # Setup project directory and DB
        project_dir = tmp_path / "workspace" / "projects" / "proj1"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        # Create test image
        img_path = tmp_path / "img.jpg"
        make_test_image(img_path)

        # Seed project and example
        with Session(engine) as session:
            session.add(
                Project(
                    project_id="proj1",
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="hosted_nvclip",
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    embedding_dim=2048,
                )
            )
            session.add(
                Example(
                    example_key="k1",
                    project_id="proj1",
                    storage_ref=str(img_path),
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    phash="abcdef0123456789",
                )
            )
            session.commit()

        # Cache the engine
        from vlm_feedback_loop.services.project_service import set_project_engine

        set_project_engine("proj1", engine)

        settings = make_settings(
            tmp_path / "workspace",
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDINGS_AUTO_COMPUTE=True,
        )
        mock_result = _fake_embeddings_result(dim=2048, count=1)

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(return_value=mock_result),
        ):
            await _embedding_worker("proj1", str(tmp_path / "workspace"), settings)

        # Verify ClipEmbedding row
        with Session(engine) as session:
            clip = session.execute(
                select(ClipEmbedding).where(
                    ClipEmbedding.project_id == "proj1",
                    ClipEmbedding.example_key == "k1",
                )
            ).scalar_one_or_none()
            assert clip is not None
            assert clip.clip_embedding_dim == 2048
            assert clip.embedding_provider == "hosted_nvclip"
            assert len(clip.vector_blob_f32) == 2048 * 4

            # Verify Example flags
            ex = session.execute(
                select(Example).where(
                    Example.project_id == "proj1",
                    Example.example_key == "k1",
                )
            ).scalar_one()
            assert ex.clip_embedding_present is True
            assert ex.clip_embedding_dim == 2048

    @pytest.mark.asyncio
    async def test_worker_gated_on_feature_flag(self, tmp_path: Path):
        """Worker returns at the EMBEDDINGS_AUTO_COMPUTE gate — no provider
        read, no NIM call. Needs a REAL project: the worker checks
        project-not-found BEFORE the flag, so the old nonexistent-project
        version returned early without ever reaching the gate under test.
        """
        from unittest.mock import AsyncMock
        from unittest.mock import patch as _patch

        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )
        from vlm_feedback_loop.services.project_service import set_project_engine

        project_dir = tmp_path / "projects" / "gate-proj"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)
        set_project_engine("gate-proj", engine)
        with Session(engine) as session:
            session.add(
                Project(
                    project_id="gate-proj",
                    name="gate",
                    project_dir=str(project_dir),
                    embedding_provider="hosted_nvclip",
                    embedding_dim=4,
                )
            )
            session.add(
                Example(
                    example_key="k1",
                    project_id="gate-proj",
                    storage_ref="/fake",
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                )
            )
            session.commit()

        settings = make_settings(
            tmp_path, EMBEDDINGS_AUTO_COMPUTE=False, NVIDIA_API_KEY="nvapi-test"
        )

        with _patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(),
        ) as embed_mock:
            await _embedding_worker("gate-proj", str(tmp_path), settings)
        # The gate held: the pending example never reached the provider.
        embed_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_worker_per_image_failure_continues(self, tmp_path: Path):
        """Per-image failure doesn't crash the batch."""
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )

        project_dir = tmp_path / "workspace" / "projects" / "proj2"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        good_img = tmp_path / "good.jpg"
        make_test_image(good_img)

        with Session(engine) as session:
            session.add(
                Project(
                    project_id="proj2",
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="hosted_nvclip",
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    embedding_dim=2048,
                )
            )
            # One good image, one missing
            session.add(
                Example(
                    example_key="good",
                    project_id="proj2",
                    storage_ref=str(good_img),
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    phash="0000000000000001",
                )
            )
            session.add(
                Example(
                    example_key="missing",
                    project_id="proj2",
                    storage_ref="/nonexistent/path.jpg",
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    phash="0000000000000002",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.project_service import (
            set_project_engine,
        )

        set_project_engine("proj2", engine)

        settings = make_settings(
            tmp_path / "workspace",
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDINGS_AUTO_COMPUTE=True,
        )
        mock_result = _fake_embeddings_result(dim=2048, count=1)

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(return_value=mock_result),
        ):
            # Should not raise
            await _embedding_worker("proj2", str(tmp_path / "workspace"), settings)

        # Good image should be embedded
        with Session(engine) as session:
            clip = session.execute(
                select(ClipEmbedding).where(
                    ClipEmbedding.project_id == "proj2",
                    ClipEmbedding.example_key == "good",
                )
            ).scalar_one_or_none()
            assert clip is not None

    @pytest.mark.asyncio
    async def test_worker_resumable(self, tmp_path: Path):
        """Worker only processes unembedded examples — already-embedded skipped."""
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding as CE
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )

        project_dir = tmp_path / "workspace" / "projects" / "proj3"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        img1 = tmp_path / "img1.jpg"
        img2 = tmp_path / "img2.jpg"
        make_test_image(img1)
        make_test_image(img2)

        dim = 2048
        vec = [0.0] * dim
        blob = serialize_vector(vec, dim)

        with Session(engine) as session:
            session.add(
                Project(
                    project_id="proj3",
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="hosted_nvclip",
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    embedding_dim=dim,
                )
            )
            # Examples are ingested before the background worker derives
            # embeddings. Seed that real persistence order explicitly.
            session.add(
                Example(
                    example_key="k1",
                    project_id="proj3",
                    storage_ref=str(img1),
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    phash="0000000000000000",
                    clip_embedding_present=True,
                )
            )
            session.add(
                Example(
                    example_key="k2",
                    project_id="proj3",
                    storage_ref=str(img2),
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    phash="ffffffffffffffff",
                )
            )
            session.flush()
            session.add(
                CE(
                    project_id="proj3",
                    example_key="k1",
                    embedding_provider="hosted_nvclip",
                    clip_embedding_model_id=EMBEDDING_MODEL_ID,
                    clip_embedding_dim=dim,
                    vector_blob_f32=blob,
                )
            )
            session.commit()

        from vlm_feedback_loop.services.project_service import (
            set_project_engine,
        )

        set_project_engine("proj3", engine)

        settings = make_settings(
            tmp_path / "workspace",
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDINGS_AUTO_COMPUTE=True,
        )
        mock_result = _fake_embeddings_result(dim=dim, count=1)
        call_count = 0

        async def counting_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_result

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=counting_create,
        ):
            await _embedding_worker("proj3", str(tmp_path / "workspace"), settings)

        # Should only have been called once (for k2, not k1)
        assert call_count == 1

        # Both should now be embedded
        with Session(engine) as session:
            count = len(
                session.execute(
                    select(CE.example_key).where(CE.project_id == "proj3")
                ).all()
            )
            assert count == 2

    @pytest.mark.asyncio
    async def test_worker_picks_up_examples_ingested_mid_run(self, tmp_path: Path):
        """Multi-pass: examples ingested while a worker is running get
        picked up by a subsequent rescan, not silently dropped.

        Regression for the dispatch race: ``trigger_embedding_computation``
        dedups by task ID, so concurrent ingests during a running worker
        are no-ops. A worker that snapshots its pending set once and exits
        leaves every later batch unembedded until backend restart — the
        worker must re-query after each sweep.
        """
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )

        project_dir = tmp_path / "workspace" / "projects" / "race_proj"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        # Two image batches: 3 ingested before the worker starts, 4 more
        # ingested mid-flight (simulated via the NIM mock side-effect).
        first_batch_paths = [tmp_path / f"first_{i}.jpg" for i in range(3)]
        second_batch_paths = [tmp_path / f"second_{i}.jpg" for i in range(4)]
        for p in first_batch_paths + second_batch_paths:
            make_test_image(p)

        with Session(engine) as session:
            session.add(
                Project(
                    project_id="race_proj",
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="hosted_nvclip",
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    embedding_dim=2048,
                )
            )
            for i, p in enumerate(first_batch_paths):
                session.add(
                    Example(
                        example_key=f"first_{i}",
                        project_id="race_proj",
                        storage_ref=str(p),
                        ingested_at="2026-01-01T00:00:00Z",
                        source_metadata={},
                        state="Unlabeled",
                        # Distinct pHashes so diverse-ordering has work to do
                        phash=f"{i:016x}",
                    )
                )
            session.commit()

        from vlm_feedback_loop.services.project_service import (
            set_project_engine,
        )

        set_project_engine("race_proj", engine)

        settings = make_settings(
            tmp_path / "workspace",
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDINGS_AUTO_COMPUTE=True,
            EMBEDDING_BATCH_SIZE_HOSTED=8,
        )

        # NIM mock that, on its FIRST call, also commits the second batch
        # to the DB — modeling concurrent ingest POSTs that arrive while
        # the worker's first sweep is in flight.
        call_count = 0

        async def mock_create_embeddings(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            input_items = kwargs.get("input_items") or args[3]
            n = len(input_items)
            if call_count == 1:
                # Mid-flight: commit the second batch as if a concurrent
                # ingest just landed. The worker's running snapshot does
                # NOT include these — only a rescan after this sweep can.
                with Session(engine) as session:
                    for i, p in enumerate(second_batch_paths):
                        session.add(
                            Example(
                                example_key=f"second_{i}",
                                project_id="race_proj",
                                storage_ref=str(p),
                                ingested_at="2026-01-01T00:00:01Z",
                                source_metadata={},
                                state="Unlabeled",
                                phash=f"{(i + 100):016x}",
                            )
                        )
                    session.commit()
            return _fake_embeddings_result(dim=2048, count=n)

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=mock_create_embeddings,
        ):
            await _embedding_worker("race_proj", str(tmp_path / "workspace"), settings)

        # All 7 examples — first batch + second batch — should be embedded.
        with Session(engine) as session:
            embedded = (
                session.execute(
                    select(ClipEmbedding.example_key).where(
                        ClipEmbedding.project_id == "race_proj"
                    )
                )
                .scalars()
                .all()
            )
            assert set(embedded) == {
                "first_0",
                "first_1",
                "first_2",
                "second_0",
                "second_1",
                "second_2",
                "second_3",
            }, (
                "Multi-pass worker must rescan and pick up examples "
                "ingested while the previous sweep was running."
            )

        # Two passes: 1 NIM call for the 3-image first batch, 1 for the
        # 4-image second batch (both fit in batch_size=8).
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_worker_state_independent(self, tmp_path: Path):
        """Verified examples still get embeddings (state-independent)."""
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )

        project_dir = tmp_path / "workspace" / "projects" / "proj4"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        img = tmp_path / "img.jpg"
        make_test_image(img)

        with Session(engine) as session:
            session.add(
                Project(
                    project_id="proj4",
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="hosted_nvclip",
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    embedding_dim=2048,
                )
            )
            session.add(
                Example(
                    example_key="verified_ex",
                    project_id="proj4",
                    storage_ref=str(img),
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Verified",  # Not Unlabeled!
                    phash="abcdef0123456789",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.project_service import (
            set_project_engine,
        )

        set_project_engine("proj4", engine)

        settings = make_settings(
            tmp_path / "workspace",
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDINGS_AUTO_COMPUTE=True,
        )

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(return_value=_fake_embeddings_result(dim=2048)),
        ):
            await _embedding_worker("proj4", str(tmp_path / "workspace"), settings)

        with Session(engine) as session:
            ex = session.execute(
                select(Example).where(Example.example_key == "verified_ex")
            ).scalar_one()
            assert ex.clip_embedding_present is True

    @pytest.mark.asyncio
    async def test_worker_local_mode_keyless_end_to_end(self, tmp_path: Path):
        """The headline Mode C case: a keyless box with a healthy local
        embedding NIM computes embeddings end-to-end over the wire — the
        worker hits the endpoint recorded on EmbeddingDeploymentConfig
        with the self-hosted shape (batch size 1) and no Authorization
        header."""
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )
        from vlm_feedback_loop.services.project_service import set_project_engine

        workspace = tmp_path / "workspace"
        project_dir = workspace / "projects" / "local-proj"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        img1 = tmp_path / "img1.jpg"
        img2 = tmp_path / "img2.jpg"
        make_test_image(img1)
        make_test_image(img2)

        with Session(engine) as session:
            session.add(
                Project(
                    project_id="local-proj",
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="self_hosted_nvclip",
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    embedding_dim=2048,
                )
            )
            for i, img in enumerate((img1, img2)):
                session.add(
                    Example(
                        example_key=f"k{i}",
                        project_id="local-proj",
                        storage_ref=str(img),
                        ingested_at="2026-01-01T00:00:00Z",
                        source_metadata={},
                        state="Unlabeled",
                        phash=f"{i:016x}",
                    )
                )
            session.commit()

        set_project_engine("local-proj", engine)

        settings = make_settings(
            workspace,
            NVIDIA_API_KEY=None,
            EMBEDDINGS_AUTO_COMPUTE=True,
        )

        with MockNIMServer() as local_mock:
            _seed_embedding_deployment_config(
                workspace, endpoint_url=local_mock.base_url
            )
            await _embedding_worker("local-proj", str(workspace), settings)
            requests = local_mock.embeddings_requests

        with Session(engine) as session:
            rows = (
                session.execute(
                    select(ClipEmbedding).where(
                        ClipEmbedding.project_id == "local-proj"
                    )
                )
                .scalars()
                .all()
            )
            assert {r.example_key for r in rows} == {"k0", "k1"}
            for row in rows:
                assert row.embedding_provider == "self_hosted_nvclip"
                assert row.clip_embedding_model_id == EMBEDDING_MODEL_ID
                assert len(row.vector_blob_f32) == 2048 * 4

            ex = session.execute(
                select(Example).where(Example.example_key == "k0")
            ).scalar_one()
            assert ex.clip_embedding_present is True
            assert ex.embedding_provider == "self_hosted_nvclip"

        # Wire shape: self-hosted batch size is 1 (one image per request),
        # every request unauthenticated against the recorded endpoint.
        assert len(requests) == 2
        for request in requests:
            assert request["path"] == "/v1/embeddings"
            assert len(request["body"]["input"]) == 1
            assert request["body"]["model"] == EMBEDDING_MODEL_ID
            headers = _lower_headers(request)
            assert "authorization" not in headers
            assert headers.get("source") == "vlm-feedback-loop"

    @pytest.mark.asyncio
    async def test_worker_hosted_mode_without_key_computes_nothing(
        self, tmp_path: Path
    ):
        """Hosted mode still requires NVIDIA_API_KEY — the worker exits
        without sending any request rather than dispatching
        unauthenticated traffic to the hosted endpoint."""
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )
        from vlm_feedback_loop.services.project_service import set_project_engine

        workspace = tmp_path / "workspace"
        project_dir = workspace / "projects" / "keyless-hosted"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        img = tmp_path / "img.jpg"
        make_test_image(img)

        with Session(engine) as session:
            session.add(
                Project(
                    project_id="keyless-hosted",
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="hosted_nvclip",
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    embedding_dim=2048,
                )
            )
            session.add(
                Example(
                    example_key="k1",
                    project_id="keyless-hosted",
                    storage_ref=str(img),
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    phash="abcdef0123456789",
                )
            )
            session.commit()

        set_project_engine("keyless-hosted", engine)

        with MockNIMServer() as hosted_mock:
            settings = make_settings(
                workspace,
                NVIDIA_API_KEY=None,
                EMBEDDINGS_AUTO_COMPUTE=True,
                HOSTED_NIM_BASE_URL=hosted_mock.base_url,
            )
            await _embedding_worker("keyless-hosted", str(workspace), settings)
            assert hosted_mock.embeddings_requests == []

        with Session(engine) as session:
            rows = (
                session.execute(
                    select(ClipEmbedding.example_key).where(
                        ClipEmbedding.project_id == "keyless-hosted"
                    )
                )
                .scalars()
                .all()
            )
            assert rows == []


# ═══════════════════════════════════════════════════════════════════════════
# Foreground priority: embedding as a background workload class
# ═══════════════════════════════════════════════════════════════════════════


def _seed_hosted_worker_project(
    tmp_path: Path, project_id: str, n_examples: int
) -> Path:
    """Project stamped ``hosted_nvclip`` plus ``n_examples`` unembedded
    examples backed by real image files. Returns the workspace root."""
    from vlm_feedback_loop.db.engine import open_project_db
    from vlm_feedback_loop.db.models.example import Example
    from vlm_feedback_loop.db.models.project import Project
    from vlm_feedback_loop.services.project_service import set_project_engine

    workspace = tmp_path / "workspace"
    project_dir = workspace / "projects" / project_id
    project_dir.mkdir(parents=True)
    engine = open_project_db(project_dir)
    set_project_engine(project_id, engine)

    with Session(engine) as session:
        session.add(
            Project(
                project_id=project_id,
                name="Test",
                project_dir=str(project_dir),
                embedding_provider="hosted_nvclip",
                embedding_model_id=EMBEDDING_MODEL_ID,
                embedding_dim=2048,
            )
        )
        for i in range(n_examples):
            img = make_test_image(tmp_path / f"img_{i}.jpg")
            session.add(
                Example(
                    example_key=f"k{i}",
                    project_id=project_id,
                    storage_ref=str(img),
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    phash=f"{i:016x}",
                )
            )
        session.commit()
    return workspace


class TestForegroundPriorityWithEmbeddingBackground:
    """The embedding worker honors the foreground-priority dispatch.

    The ``ForegroundPriorityDispatch`` primitive is unit-tested in
    isolation; these verify the *embedding worker* actually awaits
    ``priority_dispatch.wait_for_background()`` before each NIM batch
    dispatch, so embedding batches never race a live interactive
    proposal for the shared endpoint and account budget.
    """

    @pytest.mark.asyncio
    async def test_worker_waits_for_background_per_batch_dispatch(
        self, tmp_path: Path
    ) -> None:
        """Structural: one gate wait per NIM batch dispatch.

        If the call is missing, embedding batches keep firing while an
        SME waits on an interactive proposal — the latency the
        foreground-priority hold exists to prevent.
        """
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )
        from vlm_feedback_loop.services.priority import priority_dispatch

        workspace = _seed_hosted_worker_project(tmp_path, "prio-count", 2)
        settings = make_settings(
            workspace,
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDINGS_AUTO_COMPUTE=True,
            EMBEDDING_BATCH_SIZE_HOSTED=1,  # 2 examples → 2 batch dispatches
        )

        embed_mock = AsyncMock(return_value=_fake_embeddings_result(dim=2048, count=1))
        wait_mock = AsyncMock()
        with (
            patch(
                "vlm_feedback_loop.services.clip_embedding_service."
                "nim_client.create_embeddings",
                new=embed_mock,
            ),
            patch.object(priority_dispatch, "wait_for_background", new=wait_mock),
        ):
            await _embedding_worker("prio-count", str(workspace), settings)

        assert embed_mock.await_count == 2
        assert wait_mock.await_count == 2, (
            "every embedding batch dispatch must await the foreground-priority gate"
        )

    @pytest.mark.asyncio
    async def test_worker_holds_dispatch_while_foreground_active(
        self, tmp_path: Path
    ) -> None:
        """While a foreground request is in flight the worker parks before
        its NIM dispatch; the batch goes out once foreground demand clears
        (dispatch hold, not preemption)."""
        import asyncio

        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )
        from vlm_feedback_loop.services.priority import priority_dispatch

        workspace = _seed_hosted_worker_project(tmp_path, "prio-hold", 1)
        settings = make_settings(
            workspace,
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDINGS_AUTO_COMPUTE=True,
        )
        embed_mock = AsyncMock(return_value=_fake_embeddings_result(dim=2048, count=1))
        with patch(
            "vlm_feedback_loop.services.clip_embedding_service."
            "nim_client.create_embeddings",
            new=embed_mock,
        ):
            await priority_dispatch.enter_foreground()
            try:
                worker = asyncio.create_task(
                    _embedding_worker("prio-hold", str(workspace), settings)
                )
                # Generous window: an ungated worker completes its dispatch
                # well within this; a gated one parks at the gate.
                await asyncio.sleep(0.3)
                dispatched_during_hold = embed_mock.await_count
            finally:
                await priority_dispatch.exit_foreground()
            await asyncio.wait_for(worker, timeout=15)

        assert dispatched_during_hold == 0, (
            "embedding batch dispatched while a foreground request was active"
        )
        assert embed_mock.await_count == 1  # resumed after the hold cleared


# ═══════════════════════════════════════════════════════════════════════════
# Section G: Trigger function
# ═══════════════════════════════════════════════════════════════════════════


class TestTriggerFunction:
    def test_trigger_is_nonblocking(self, tmp_path: Path):
        """trigger_embedding_computation returns immediately (doesn't await)."""
        from vlm_feedback_loop.services.clip_embedding_service import (
            trigger_embedding_computation,
        )

        settings = make_settings(
            tmp_path, EMBEDDINGS_AUTO_COMPUTE=True, NVIDIA_API_KEY="nvapi-test"
        )

        # Should return without error even with a non-existent project
        # (the background task will fail gracefully). Patch register (the
        # scheduling boundary) so the real try_register dedupe/close logic
        # still runs.
        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.background_manager.register"
        ) as mock_register:
            mock_register.side_effect = lambda task_id, coro: (
                coro.close() if hasattr(coro, "close") else None
            )
            trigger_embedding_computation("proj", str(tmp_path), settings)
            mock_register.assert_called_once()

    def test_trigger_gated_on_flag(self, tmp_path: Path):
        from vlm_feedback_loop.services.clip_embedding_service import (
            trigger_embedding_computation,
        )

        settings = make_settings(tmp_path, EMBEDDINGS_AUTO_COMPUTE=False)

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.background_manager.register"
        ) as mock_register:
            trigger_embedding_computation("proj", str(tmp_path), settings)
            mock_register.assert_not_called()

    def test_trigger_deduplicates(self, tmp_path: Path):
        from vlm_feedback_loop.services.clip_embedding_service import (
            trigger_embedding_computation,
        )

        settings = make_settings(
            tmp_path, EMBEDDINGS_AUTO_COMPUTE=True, NVIDIA_API_KEY="nvapi-test"
        )

        from vlm_feedback_loop.services.background import BackgroundTaskManager

        with (
            patch.object(
                BackgroundTaskManager,
                "active_task_ids",
                new_callable=PropertyMock,
                return_value=["clip-embed-proj"],
            ),
            patch(
                "vlm_feedback_loop.services.clip_embedding_service.background_manager.register"
            ) as mock_register,
        ):
            trigger_embedding_computation("proj", str(tmp_path), settings)
        mock_register.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# First-ingest re-probe
# ═══════════════════════════════════════════════════════════════════════════


class TestFirstIngestReprobe:
    @pytest.mark.asyncio
    async def test_worker_reprobes_on_none_provider(self, tmp_path: Path):
        """Worker re-attempts probe when project.embedding_provider=none."""
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )

        project_dir = tmp_path / "workspace" / "projects" / "reprobe"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        img = tmp_path / "img.jpg"
        make_test_image(img)

        with Session(engine) as session:
            session.add(
                Project(
                    project_id="reprobe",
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="none",  # Failed at creation
                )
            )
            session.add(
                Example(
                    example_key="k1",
                    project_id="reprobe",
                    storage_ref=str(img),
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    phash="abcdef0123456789",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.project_service import (
            set_project_engine,
        )

        set_project_engine("reprobe", engine)

        settings = make_settings(
            tmp_path / "workspace",
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDING_PROVIDER="auto",
            EMBEDDINGS_AUTO_COMPUTE=True,
        )

        mock_embed = _fake_embeddings_result(dim=2048, count=1)

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.nim_client.create_embeddings",
            new=AsyncMock(return_value=mock_embed),
        ):
            await _embedding_worker("reprobe", str(tmp_path / "workspace"), settings)

        # Probe should have succeeded and updated the project
        with Session(engine) as session:
            proj = session.execute(
                select(Project).where(Project.project_id == "reprobe")
            ).scalar_one()
            assert proj.embedding_provider == "hosted_nvclip"

            # And the example should be embedded
            ex = session.execute(
                select(Example).where(Example.example_key == "k1")
            ).scalar_one()
            assert ex.clip_embedding_present is True


# ═══════════════════════════════════════════════════════════════════════════
# Worker provider self-heal
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkerProviderSelfHeal:
    """A project stamped onto a provider that can no longer be served
    re-runs the probe cascade instead of stranding: the worker heals to
    the fallback provider (hosted when a key is present) or exits
    cleanly on "none"."""

    def _seed_stranded_project(self, tmp_path: Path, project_id: str):
        """Project stamped self_hosted with one unembedded example.

        Returns ``(engine, workspace)``.
        """
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.project_service import set_project_engine

        workspace = tmp_path / "workspace"
        project_dir = workspace / "projects" / project_id
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)

        img = tmp_path / "img.jpg"
        make_test_image(img)

        with Session(engine) as session:
            session.add(
                Project(
                    project_id=project_id,
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="self_hosted_nvclip",
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    embedding_dim=2048,
                )
            )
            session.add(
                Example(
                    example_key="k1",
                    project_id=project_id,
                    storage_ref=str(img),
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    phash="abcdef0123456789",
                )
            )
            session.commit()

        set_project_engine(project_id, engine)
        return engine, workspace

    @pytest.mark.asyncio
    async def test_stranded_self_hosted_project_heals_to_hosted(self, tmp_path: Path):
        """Project stamped self_hosted + config reset to none (the local
        NIM died) + NVIDIA_API_KEY present → the worker re-probes,
        re-stamps the project hosted_nvclip, and computes embeddings via
        the hosted endpoint (authenticated) instead of stranding."""
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )

        engine, workspace = self._seed_stranded_project(tmp_path, "stranded")
        _seed_embedding_deployment_config(workspace, endpoint_url=None, provider="none")

        with MockNIMServer() as hosted_mock:
            settings = make_settings(
                workspace,
                NVIDIA_API_KEY="nvapi-test",
                EMBEDDINGS_AUTO_COMPUTE=True,
                HOSTED_NIM_BASE_URL=hosted_mock.base_url,
            )
            await _embedding_worker("stranded", str(workspace), settings)
            requests = hosted_mock.embeddings_requests

        with Session(engine) as session:
            proj = session.execute(
                select(Project).where(Project.project_id == "stranded")
            ).scalar_one()
            assert proj.embedding_provider == "hosted_nvclip"

            row = session.execute(
                select(ClipEmbedding).where(ClipEmbedding.project_id == "stranded")
            ).scalar_one()
            assert row.example_key == "k1"
            assert row.embedding_provider == "hosted_nvclip"

        # Probe + batch both hit the hosted endpoint with the key.
        assert len(requests) >= 2
        for request in requests:
            headers = _lower_headers(request)
            assert headers.get("authorization") == "Bearer nvapi-test"

    @pytest.mark.asyncio
    async def test_stranded_self_hosted_keyless_exits_cleanly(self, tmp_path: Path):
        """The keyless variant: with no fallback available the worker
        re-stamps the project "none" and exits cleanly — no embeddings,
        no run_failed SSE (a dead stamp must not turn into a failure
        storm on every trigger)."""
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )

        engine, workspace = self._seed_stranded_project(tmp_path, "stranded-keyless")
        _seed_embedding_deployment_config(workspace, endpoint_url=None, provider="none")

        settings = make_settings(
            workspace,
            NVIDIA_API_KEY=None,
            EMBEDDINGS_AUTO_COMPUTE=True,
            HOSTED_NIM_BASE_URL=_dead_endpoint_url(),
        )

        emit = AsyncMock()
        with patch(
            "vlm_feedback_loop.services.clip_embedding_service.sse_manager.emit",
            new=emit,
        ):
            await _embedding_worker("stranded-keyless", str(workspace), settings)

        with Session(engine) as session:
            proj = session.execute(
                select(Project).where(Project.project_id == "stranded-keyless")
            ).scalar_one()
            assert proj.embedding_provider == "none"

            rows = (
                session.execute(
                    select(ClipEmbedding.example_key).where(
                        ClipEmbedding.project_id == "stranded-keyless"
                    )
                )
                .scalars()
                .all()
            )
            assert rows == []

        emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_config_provider_flag_triggers_reprobe(self, tmp_path: Path):
        """A config whose provider flag moved off self-hosted while a
        stale endpoint URL lingers is treated as unserveable: the worker
        re-probes instead of computing against an endpoint the deploy
        flow no longer vouches for."""
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            _embedding_worker,
        )

        engine, workspace = self._seed_stranded_project(tmp_path, "stale-flag")
        # provider flag reset, endpoint URL left behind (partial update).
        _seed_embedding_deployment_config(
            workspace, endpoint_url=_dead_endpoint_url(), provider="none"
        )

        with MockNIMServer() as hosted_mock:
            settings = make_settings(
                workspace,
                NVIDIA_API_KEY="nvapi-test",
                EMBEDDINGS_AUTO_COMPUTE=True,
                HOSTED_NIM_BASE_URL=hosted_mock.base_url,
            )
            await _embedding_worker("stale-flag", str(workspace), settings)
            requests = hosted_mock.embeddings_requests

        with Session(engine) as session:
            proj = session.execute(
                select(Project).where(Project.project_id == "stale-flag")
            ).scalar_one()
            assert proj.embedding_provider == "hosted_nvclip"
        assert len(requests) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Section H: Provider resweep
# ═══════════════════════════════════════════════════════════════════════════


class TestResweepEmbeddingTasks:
    """``resweep_embedding_tasks`` re-resolves the provider after a
    deployment-level change and restarts pending embedding work."""

    @pytest.mark.asyncio
    async def test_resweep_flips_provider_preserves_vectors_and_triggers(
        self, tmp_path: Path
    ):
        """When the local NIM becomes the effective provider, a project
        stamped hosted is re-stamped in place: stored ClipEmbedding rows
        survive (same model + dim means the vectors stay valid) and the
        project's pending work is re-triggered."""
        from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            resweep_embedding_tasks,
        )

        engine, project_dir, workspace = open_project_workspace(
            tmp_path, "resweep-proj", register_engine=True
        )

        dim = 2048
        blob = serialize_vector([0.0] * dim, dim)
        with Session(engine) as session:
            session.add(
                Project(
                    project_id="resweep-proj",
                    name="Test",
                    project_dir=str(project_dir),
                    embedding_provider="hosted_nvclip",
                    embedding_model_id=EMBEDDING_MODEL_ID,
                    embedding_dim=dim,
                )
            )
            # Examples are ingested before the background worker derives
            # embeddings. Seed that real persistence order explicitly.
            session.add(
                Example(
                    example_key="k1",
                    project_id="resweep-proj",
                    storage_ref="/fake1",
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                    clip_embedding_present=True,
                )
            )
            session.add(
                Example(
                    example_key="k2",
                    project_id="resweep-proj",
                    storage_ref="/fake2",
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                )
            )
            session.flush()
            session.add(
                ClipEmbedding(
                    project_id="resweep-proj",
                    example_key="k1",
                    embedding_provider="hosted_nvclip",
                    clip_embedding_model_id=EMBEDDING_MODEL_ID,
                    clip_embedding_dim=dim,
                    vector_blob_f32=blob,
                )
            )
            session.commit()

        with MockNIMServer() as local_mock:
            _seed_embedding_deployment_config(
                workspace, endpoint_url=local_mock.base_url
            )
            settings = make_settings(
                workspace,
                NVIDIA_API_KEY=None,
                EMBEDDING_PROVIDER="auto",
                EMBEDDINGS_AUTO_COMPUTE=True,
            )
            with patch(
                "vlm_feedback_loop.services.clip_embedding_service."
                "trigger_embedding_computation"
            ) as trigger:
                await resweep_embedding_tasks(settings)

        with Session(engine) as session:
            proj = session.execute(
                select(Project).where(Project.project_id == "resweep-proj")
            ).scalar_one()
            assert proj.embedding_provider == "self_hosted_nvclip"
            assert proj.embedding_model_id == EMBEDDING_MODEL_ID
            assert proj.embedding_dim == dim

            # The hosted-computed vector was NOT deleted or recomputed —
            # its descriptive provider label keeps recording who made it.
            rows = (
                session.execute(
                    select(ClipEmbedding).where(
                        ClipEmbedding.project_id == "resweep-proj"
                    )
                )
                .scalars()
                .all()
            )
            assert len(rows) == 1
            assert rows[0].example_key == "k1"
            assert rows[0].embedding_provider == "hosted_nvclip"
            assert rows[0].vector_blob_f32 == blob

        trigger.assert_called_once_with("resweep-proj", str(workspace), settings)

    @pytest.mark.asyncio
    async def test_resweep_probes_once_and_skips_fully_embedded_projects(
        self, tmp_path: Path
    ):
        """The probe outcome is project-independent, so one probe covers
        the whole walk; projects with nothing pending spawn no worker."""
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.clip_embedding_service import (
            resweep_embedding_tasks,
        )

        for pid in ("proj-a", "proj-b"):
            engine, project_dir, workspace = open_project_workspace(
                tmp_path, pid, register_engine=True
            )
            with Session(engine) as session:
                session.add(
                    Project(
                        project_id=pid,
                        name="Test",
                        project_dir=str(project_dir),
                        embedding_provider="hosted_nvclip",
                        embedding_model_id=EMBEDDING_MODEL_ID,
                        embedding_dim=2048,
                    )
                )
                session.commit()

        settings = make_settings(
            tmp_path / "workspace",
            NVIDIA_API_KEY="nvapi-test",
            EMBEDDING_PROVIDER="auto",
            EMBEDDINGS_AUTO_COMPUTE=True,
        )

        with (
            patch(
                "vlm_feedback_loop.services.clip_embedding_service.probe_nvclip",
                new=AsyncMock(
                    return_value=(True, "hosted_nvclip", EMBEDDING_MODEL_ID, 2048)
                ),
            ) as probe,
            patch(
                "vlm_feedback_loop.services.clip_embedding_service."
                "trigger_embedding_computation"
            ) as trigger,
        ):
            await resweep_embedding_tasks(settings)

        probe.assert_awaited_once()
        trigger.assert_not_called()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``services.tao_base_experiment_provisioning_service``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from conftest import make_settings, make_tao_settings
from support import FakeS3Client
from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db, open_project_db
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS3_SUPER_REASONER,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_2B_HF_PATH,
    COSMOS_REASON2_8B,
    COSMOS_REASON2_8B_HF_PATH,
)
from vlm_feedback_loop.services import tao_auth as tao_auth_mod
from vlm_feedback_loop.services.project_service import create_project
from vlm_feedback_loop.services.tao_base_experiment_provisioning_service import (
    PULL_REQUIREMENTS_PATH,
    PULL_SCRIPT_PATH,
    _default_subprocess_runner,
    _Target,
    _upload_stage_tree_sync,
    _write_csv,
    provision_base_experiments,
)

# ── Shared fixtures and helpers ──────────────────────────────────────────────


def _settings_for(workspace_root: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "TAO_API_BASE_URL": "https://tao.example/api/v2",
        "TAO_API_KEY": "jwt-token",
    }
    base.update(overrides)
    return make_tao_settings(workspace_root, **base)


def _seed_bootstrapped_workspace(
    workspace_root: Path, *, bucket: str = "bucket"
) -> None:
    """Mark the singleton as bootstrapped so the service does not bail."""
    engine = init_deployment_db(workspace_root)
    with Session(engine) as session:
        cfg = session.query(TAODeploymentConfig).first()
        assert cfg is not None
        cfg.tao_workspace_id = "ws-1"
        cfg.tao_workspace_name = "ws-1"
        cfg.tao_workspace_bucket = bucket
        cfg.tao_workspace_cloud_type = "seaweedfs"
        cfg.tao_workspace_s3_endpoint_url_external = "http://127.0.0.1:8333"
        cfg.tao_workspace_s3_endpoint_url_internal = "http://seaweedfs-s3:8333"
        cfg.tao_workspace_s3_access_key_ref = "TAO_WORKSPACE_S3_ACCESS_KEY"
        cfg.tao_workspace_s3_secret_key_ref = "TAO_WORKSPACE_S3_SECRET_KEY"
        cfg.bootstrap_status = "bootstrapped"
        session.commit()


@pytest.fixture()
def workspace(tmp_workspace):
    init_deployment_db(tmp_workspace)
    _seed_bootstrapped_workspace(tmp_workspace)
    return tmp_workspace


@pytest.fixture(autouse=True)
def _noop_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speed up tests by zeroing HTTP retry backoff."""

    async def _noop(_attempt_index: int) -> None:
        pass

    monkeypatch.setattr("vlm_feedback_loop.services.http_client._backoff", _noop)


@pytest.fixture(autouse=True)
def _no_settle_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the post-load_airgapped settle sleep so tests run fast."""

    async def _noop(_seconds: float) -> None:
        pass

    monkeypatch.setattr(
        "vlm_feedback_loop.services.tao_base_experiment_provisioning_service.asyncio.sleep",
        _noop,
    )


@pytest.fixture(autouse=True)
def _clear_tao_bearer_cache() -> None:
    """Ensure each test starts with a clean bearer cache."""
    tao_auth_mod._token_cache.clear()  # type: ignore[attr-defined]
    yield
    tao_auth_mod._token_cache.clear()  # type: ignore[attr-defined]


# ── Subprocess runner / TAO transport stubs ─────────────────────────────────


def _ok_subprocess_runner(
    stage_seeded_files: list[tuple[str, bytes]] | None = None,
):
    """Build a runner that pretends pretrained_models succeeded.

    Optionally writes ``stage_seeded_files`` into the stage dir so the
    S3 upload step has something to walk. ``--shared-folder-path`` is
    parsed out of ``cmd`` to find the stage dir.
    """
    captured: dict[str, Any] = {}

    async def runner(cmd, *, env, timeout_s):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        captured["timeout_s"] = timeout_s
        idx = cmd.index("--shared-folder-path")
        stage = Path(cmd[idx + 1])
        csv_idx = cmd.index("--csv")
        captured["csv"] = Path(cmd[csv_idx + 1]).read_text(encoding="utf-8")
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "ptm_metadatas.json").write_text(
            json.dumps({"Cosmos Reason2 2B": {}}), encoding="utf-8"
        )
        if stage_seeded_files:
            for rel, payload in stage_seeded_files:
                p = stage / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(payload)
        return {
            "ok": True,
            "stdout": json.dumps(
                {
                    "ptm_metadatas_path": str(stage / "ptm_metadatas.json"),
                    "registered": ["Cosmos Reason2 2B"],
                }
            ),
            "stderr": "",
            "returncode": 0,
        }

    runner.captured = captured  # type: ignore[attr-defined]
    return runner


def _failing_subprocess_runner(stderr: str = "boom", rc: int = 7):
    captured: dict[str, Any] = {}

    async def runner(cmd, *, env, timeout_s):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        return {"ok": False, "stdout": "", "stderr": stderr, "returncode": rc}

    runner.captured = captured  # type: ignore[attr-defined]
    return runner


def _tao_transport(
    *,
    pre_pass_experiments: list[dict[str, Any]] | None = None,
    confirm_experiments: list[dict[str, Any]] | None = None,
    load_airgapped_responses: list[tuple[int, dict[str, Any]]] | None = None,
):
    """Build a MockTransport that answers list_base_experiments + load_airgapped.

    Realistic TAO semantics: ``:list_base_experiments?network_arch=cosmos-rl``
    returns ALL experiments registered on the workspace. The Blueprint
    then filters client-side by ``name_substring``. So the mock returns
    the full list per call, and we have two phases:

    - ``pre_pass_experiments`` is what TAO reports BEFORE :load_airgapped
      runs (default: empty — fresh workspace).
    - ``confirm_experiments`` is what TAO reports AFTER :load_airgapped
      runs (default: empty too — for tests that don't reach confirmation).

    ``load_airgapped_responses`` is consumed by index per POST. Default
    a single 200 with ``experiments_failed=0``.
    """
    pre_pass = list(pre_pass_experiments or [])
    confirm = list(confirm_experiments or [])
    la_calls = {"n": 0}
    list_calls = {"n": 0}
    load_airgapped_responses = load_airgapped_responses or [
        (200, {"success": True, "experiments_loaded": 1, "experiments_failed": 0}),
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith("/jobs:list_base_experiments"):
            list_calls["n"] += 1
            payload = confirm if la_calls["n"] >= 1 else pre_pass
            return httpx.Response(200, json={"experiments": payload})
        if path.endswith("/jobs:load_airgapped"):
            i = la_calls["n"]
            la_calls["n"] = i + 1
            status, body = load_airgapped_responses[
                min(i, len(load_airgapped_responses) - 1)
            ]
            return httpx.Response(status, json=body)
        return httpx.Response(404, json={"error": f"unexpected path {path}"})

    transport = httpx.MockTransport(handler)
    transport.requests = requests  # type: ignore[attr-defined]
    transport.list_calls = list_calls  # type: ignore[attr-defined]
    transport.la_calls = la_calls  # type: ignore[attr-defined]
    return transport


# ── Workspace readiness ──────────────────────────────────────────────────────


class TestWorkspaceReadiness:
    @pytest.mark.asyncio
    async def test_unbootstrapped_workspace_names_the_canonical_state_store(
        self, tmp_workspace: Path
    ):
        """Recovery guidance points to deployment.db, not a retired env key."""
        init_deployment_db(tmp_workspace)

        result = await provision_base_experiments(_settings_for(tmp_workspace))

        assert result.failed == [
            (
                "workspace",
                "TAO workspace identity is not bootstrapped in deployment.db; "
                "run `vlm-feedback-loop tao-bootstrap` first.",
            )
        ]


# ── #1: CSV writer ───────────────────────────────────────────────────────────


class TestCsvWriter:
    def test_csv_writer_emits_4col_header_and_hf_model_rows(self, tmp_path: Path):
        targets = [
            _Target(
                model_config_id="mc-2b",
                model_name=COSMOS_REASON2_2B,
                hf_path=COSMOS_REASON2_2B_HF_PATH,
                display_name="Cosmos Reason2 2B",
            ),
            _Target(
                model_config_id="mc-8b",
                model_name=COSMOS_REASON2_8B,
                hf_path=COSMOS_REASON2_8B_HF_PATH,
                display_name="Cosmos Reason2 8B",
            ),
        ]
        csv_path = tmp_path / "pretrained_models.csv"
        _write_csv(csv_path, targets)
        content = csv_path.read_text(encoding="utf-8")
        # Header + 2 rows + trailing newline.
        assert content == (
            "displayName,ngc_path,network_arch,is_backbone\n"
            "Cosmos Reason2 2B,hf_model://nvidia/Cosmos-Reason2-2B,cosmos-rl,True\n"
            "Cosmos Reason2 8B,hf_model://nvidia/Cosmos-Reason2-8B,cosmos-rl,True\n"
        )


# ── #2: subprocess invocation shape ──────────────────────────────────────────


class TestSubprocessInvocationShape:
    @pytest.mark.asyncio
    async def test_subprocess_invocation_shape(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The NGC key reaches the child ONLY via env (``PTM_API_KEY``) —
        an argv copy would be readable by any co-tenant process via
        ``ps`` / ``/proc/<pid>/cmdline`` for the subprocess lifetime."""
        ngc_sentinel = "SENTINEL_OUTER_NGC"
        hf_sentinel = "SENTINEL_OUTER_HF"
        settings = _settings_for(workspace, TAO_API_KEY=ngc_sentinel)
        monkeypatch.setenv("HARMLESS_INHERITED", "present")
        parent_before = dict(os.environ)
        runner = _ok_subprocess_runner()
        s3 = FakeS3Client()
        # No targets in TAO yet (pre-pass empty) → full chain runs.
        transport = _tao_transport(
            confirm_experiments=[
                {"id": "uuid-2b", "name": "Cosmos-Reason2-2B"},
                {"id": "uuid-8b", "name": "Cosmos-Reason2-8B"},
            ]
        )
        result = await provision_base_experiments(
            settings,
            student_base_model_config_ids=None,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=lambda: s3,
            hf_token=hf_sentinel,
        )
        # The default discovery loop hits load_airgapped + a confirm
        # call. We don't assert on result.registered here — confirmed
        # by other tests. We DO assert the subprocess invocation shape.
        cmd = runner.captured["cmd"]  # type: ignore[attr-defined]
        env = runner.captured["env"]  # type: ignore[attr-defined]
        assert cmd[:4] == ["uv", "run", "--isolated", "--no-project"]
        assert "--with-requirements" in cmd
        assert str(PULL_REQUIREMENTS_PATH) in cmd
        assert "--python" in cmd
        assert sys.executable in cmd
        assert str(PULL_SCRIPT_PATH) in cmd
        assert "--csv" in cmd
        assert "--shared-folder-path" in cmd
        assert "--ngc-key" not in cmd
        assert ngc_sentinel not in " ".join(cmd)
        assert hf_sentinel not in " ".join(cmd)
        assert env["AIRGAPPED_MODE"] == "true"
        assert env["PTM_API_KEY"] == ngc_sentinel
        assert env["HF_TOKEN"] == hf_sentinel
        assert env["HUGGING_FACE_HUB_TOKEN"] == hf_sentinel
        assert env["HARMLESS_INHERITED"] == "present"
        assert "HF_HOME" in env  # default seeded by service when not pre-set
        assert dict(os.environ) == parent_before
        del result

    @pytest.mark.asyncio
    async def test_target_model_names_limits_the_download_roster(self, workspace: Path):
        """A single-base validation run must not pull unrelated optional bases."""
        settings = _settings_for(workspace, TAO_API_KEY="ngc-key")
        runner = _ok_subprocess_runner()
        result = await provision_base_experiments(
            settings,
            target_model_names=[COSMOS_REASON2_2B],
            _transport=_tao_transport(),
            _subprocess_runner=runner,
            _dry_run=True,
        )

        assert result.failed == [(COSMOS_REASON2_2B, "dry_run: not uploaded")]
        assert "Cosmos Reason2 2B" in runner.captured["csv"]  # type: ignore[attr-defined]
        assert "Cosmos Reason2 8B" not in runner.captured["csv"]  # type: ignore[attr-defined]
        assert "Cosmos3" not in runner.captured["csv"]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_default_runner_redacts_spawn_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The real outer spawn boundary cannot return an echoed credential."""
        sentinel = "SENTINEL_OUTER_SPAWN_FAILURE"

        async def fail_spawn(*_args, **_kwargs):
            raise OSError(f"spawn failed around {sentinel}")

        monkeypatch.setattr("asyncio.create_subprocess_exec", fail_spawn)
        result = await _default_subprocess_runner(
            ["python", "pull_base_experiments.py"],
            env={
                "PTM_API_KEY": sentinel,
                "HF_TOKEN": "",
                "HUGGING_FACE_HUB_TOKEN": "",
            },
            timeout_s=1.0,
        )

        assert result["ok"] is False
        assert result["returncode"] == -1
        assert "[REDACTED]" in result["stderr"]
        assert sentinel not in result["stderr"]

    @pytest.mark.asyncio
    async def test_default_runner_cancellation_uses_shared_cleanup_and_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import asyncio
        from unittest.mock import AsyncMock

        from vlm_feedback_loop.services import (
            tao_base_experiment_provisioning_service as provisioning,
        )

        class FakeProcess:
            returncode = None

        async def fake_exec(*_args, **_kwargs):
            return FakeProcess()

        communication = AsyncMock(side_effect=asyncio.CancelledError)
        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr(
            provisioning,
            "communicate_with_timeout",
            communication,
        )

        with pytest.raises(asyncio.CancelledError):
            await provisioning._default_subprocess_runner(
                ["python", "pull_base_experiments.py"],
                env={},
                timeout_s=1.0,
            )

        communication.assert_awaited_once()


# ── #3: subprocess stdout parsing propagation ────────────────────────────────


class TestSubprocessStdoutParsing:
    @pytest.mark.asyncio
    async def test_subprocess_stdout_parsing_propagates(self, workspace: Path):
        """Happy path: subprocess stdout reports registered models, S3
        upload runs, load_airgapped succeeds, registered list populated."""
        settings = _settings_for(workspace, TAO_API_KEY="ngc-key")
        runner = _ok_subprocess_runner(
            stage_seeded_files=[
                ("ptm_metadatas.json", b"{}"),  # already seeded by runner; overwritten
                ("Cosmos-Reason2-2B/config.json", b"{}"),
                ("Cosmos-Reason2-2B/tiny.bin", b"x" * 4096),
            ]
        )
        s3 = FakeS3Client()
        transport = _tao_transport(
            confirm_experiments=[
                {"id": "uuid-2b", "name": "Cosmos-Reason2-2B"},
                {"id": "uuid-8b", "name": "Cosmos-Reason2-8B"},
                {"id": "uuid-nano", "name": "Cosmos3-Nano-Reasoner"},
                {"id": "uuid-super", "name": "Cosmos3-Super-Reasoner"},
            ],
        )
        result = await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=lambda: s3,
        )
        assert result.failed == [], result.failed
        assert sorted(result.registered) == [
            COSMOS_REASON2_2B,
            COSMOS_REASON2_8B,
            COSMOS3_NANO_REASONER,
            COSMOS3_SUPER_REASONER,
        ]


# ── #4: subprocess nonzero exit surfaces stderr and skips S3 ────────────────


class TestSubprocessNonzeroExit:
    @pytest.mark.asyncio
    async def test_subprocess_nonzero_exit_surfaces_stderr_and_skips_s3(
        self, workspace: Path
    ):
        ngc_sentinel = "SENTINEL_FAILED_OUTER_NGC"
        hf_sentinel = "SENTINEL_FAILED_OUTER_HF"
        settings = _settings_for(workspace, TAO_API_KEY=ngc_sentinel)
        runner = _failing_subprocess_runner(
            stderr=f"catalog 503: {ngc_sentinel} {hf_sentinel}", rc=2
        )

        def s3_factory_must_not_be_called():
            pytest.fail("S3 client factory must not be called when subprocess fails")

        transport = _tao_transport()  # no experiments registered, none expected
        result = await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=s3_factory_must_not_be_called,
            hf_token=hf_sentinel,
        )
        assert result.registered == []
        assert result.failed, "expected failures recorded"
        # Each target carries the subprocess error string + rc.
        joined = " ".join(err for _, err in result.failed)
        assert "catalog 503" in joined
        assert "rc=2" in joined
        assert "[REDACTED]" in joined
        assert ngc_sentinel not in joined
        assert hf_sentinel not in joined
        # No load_airgapped HTTP traffic either.
        paths = [r.url.path for r in transport.requests]  # type: ignore[attr-defined]
        assert not any(p.endswith("/jobs:load_airgapped") for p in paths)


# ── #5: S3 upload tree walk + threshold + idempotency ──────────────────────


class TestS3UploadTree:
    def test_upload_remains_bound_to_opened_file_after_path_replacement(
        self, tmp_path: Path
    ):
        stage_dir = tmp_path / "stage"
        stage_dir.mkdir()
        staged_path = stage_dir / "weights.bin"
        original_bytes = b"original checkpoint bytes"
        replacement_bytes = b"replacement checkpoint bytes"
        staged_path.write_bytes(original_bytes)

        class ReplacingHeadClient(FakeS3Client):
            def head_object(
                self,
                *,
                Bucket: str,
                Key: str,  # noqa: N803
            ) -> dict[str, Any]:
                replacement = stage_dir / "replacement.bin"
                replacement.write_bytes(replacement_bytes)
                os.replace(replacement, staged_path)
                return super().head_object(Bucket=Bucket, Key=Key)

        s3 = ReplacingHeadClient()
        _upload_stage_tree_sync(s3, bucket="bucket", stage_dir=stage_dir)

        uploaded = s3.objects[("bucket", "shared-storage/models/weights.bin")]
        assert uploaded["Body"] == original_bytes
        assert staged_path.read_bytes() == replacement_bytes

    @pytest.mark.asyncio
    async def test_s3_upload_walks_tree_threshold_and_idempotent(self, workspace: Path):
        settings = _settings_for(
            workspace,
            TAO_API_KEY="ngc-key",
            TAO_WORKSPACE_S3_ACCESS_KEY="ak",
            TAO_WORKSPACE_S3_SECRET_KEY="sk",
        )
        s3 = FakeS3Client()

        # Pre-seed S3 with one already-uploaded file (matching SHA-256 metadata)
        # to verify the idempotency short-circuit. The runner will write
        # this same file into the stage dir.
        import hashlib

        already_payload = b"already" * 100
        already_sha = hashlib.sha256(already_payload).hexdigest()
        s3.objects[("bucket", "shared-storage/models/Already/skip.bin")] = {
            "Body": already_payload,
            "Metadata": {"dataset-export-sha256": already_sha},
        }

        # Stage will get: 5 KiB (single PUT), 9 MiB (multipart), and the
        # 700-byte already-uploaded file.
        small_payload = b"x" * 5120
        large_payload = b"y" * (9 * 1024 * 1024)
        runner = _ok_subprocess_runner(
            stage_seeded_files=[
                ("Cosmos-Reason2-2B/small.bin", small_payload),
                ("Cosmos-Reason2-2B/large.bin", large_payload),
                ("Already/skip.bin", already_payload),
                ("_hf_cache/hub/blobs/cache.bin", b"cache-only"),
            ]
        )
        transport = _tao_transport(
            confirm_experiments=[
                {"id": "uuid-2b", "name": "Cosmos-Reason2-2B"},
                {"id": "uuid-8b", "name": "Cosmos-Reason2-8B"},
            ],
        )
        result = await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=lambda: s3,
        )
        del result
        methods = [m for m, _ in s3.calls]
        # head_object hit for every regular file in the stage dir.
        # exact count depends on rglob order + ptm_metadatas.json (also a file).
        assert methods.count("head_object") >= 4
        # small + ptm_metadatas use put_object.
        put_keys = [k["Key"] for m, k in s3.calls if m == "put_object"]
        assert any(k.endswith("small.bin") for k in put_keys)
        # large uses multipart.
        mp_keys = [k["Key"] for m, k in s3.calls if m == "create_multipart_upload"]
        assert any(k.endswith("large.bin") for k in mp_keys)
        # already-uploaded file is NOT re-uploaded.
        assert not any(k.endswith("Already/skip.bin") for k in put_keys)
        assert not any(k.endswith("Already/skip.bin") for k in mp_keys)
        assert not any("_hf_cache" in kwargs.get("Key", "") for _, kwargs in s3.calls)


# ── #6: load_airgapped request shape ────────────────────────────────────────


class TestLoadAirgappedRequestShape:
    @pytest.mark.asyncio
    async def test_load_airgapped_request_shape(self, workspace: Path):
        settings = _settings_for(workspace, TAO_API_KEY="jwt-token")
        s3 = FakeS3Client()
        runner = _ok_subprocess_runner()
        transport = _tao_transport(
            confirm_experiments=[
                {"id": "uuid-2b", "name": "Cosmos-Reason2-2B"},
                {"id": "uuid-8b", "name": "Cosmos-Reason2-8B"},
            ],
        )
        await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=lambda: s3,
        )
        la = next(
            (
                r
                for r in transport.requests
                if r.url.path.endswith("/jobs:load_airgapped")
            ),  # type: ignore[attr-defined]
            None,
        )
        assert la is not None, "load_airgapped POST not observed"
        assert la.method == "POST"
        assert la.url.path == "/api/v2/orgs/example-org/jobs:load_airgapped"
        body = json.loads(la.content.decode("utf-8"))
        assert body == {"workspace_id": "ws-1"}
        assert la.headers["Authorization"].startswith("Bearer ")


# ── #7: 401 retry triggers bearer invalidation ──────────────────────────────


class TestLoadAirgapped401Retry:
    @pytest.mark.asyncio
    async def test_load_airgapped_401_triggers_bearer_invalidate_and_retry(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ):
        settings = _settings_for(workspace, TAO_API_KEY="jwt-token")

        # Spy on invalidate_tao_bearer at its home module — the single-401-
        # retry lives in the shared ``tao_auth.retry_once_on_401``, which
        # resolves the invalidator through ``tao_auth``'s namespace.
        invalidate_calls = {"n": 0}
        original_invalidate = tao_auth_mod.invalidate_tao_bearer

        def spy_invalidate(s):
            invalidate_calls["n"] += 1
            original_invalidate(s)

        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_auth.invalidate_tao_bearer",
            spy_invalidate,
        )

        s3 = FakeS3Client()
        runner = _ok_subprocess_runner()
        transport = _tao_transport(
            confirm_experiments=[
                {"id": "uuid-2b", "name": "Cosmos-Reason2-2B"},
                {"id": "uuid-8b", "name": "Cosmos-Reason2-8B"},
            ],
            load_airgapped_responses=[
                (401, {"error": "expired"}),
                (
                    200,
                    {"success": True, "experiments_loaded": 1, "experiments_failed": 0},
                ),
            ],
        )
        result = await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=lambda: s3,
        )
        # Two load_airgapped POSTs were made.
        la_requests = [
            r
            for r in transport.requests  # type: ignore[attr-defined]
            if r.url.path.endswith("/jobs:load_airgapped")
        ]
        assert len(la_requests) == 2
        # invalidate_tao_bearer was called exactly once between the
        # two POSTs.
        assert invalidate_calls["n"] == 1
        # No load_airgapped failures recorded after retry.
        assert all("load_airgapped" not in err for _, err in result.failed)


# ── #8: experiments_failed > 0 raises ───────────────────────────────────────


class TestLoadAirgappedExperimentsFailed:
    @pytest.mark.asyncio
    async def test_load_airgapped_experiments_failed_raises(self, workspace: Path):
        settings = _settings_for(workspace, TAO_API_KEY="jwt-token")
        s3 = FakeS3Client()
        runner = _ok_subprocess_runner()
        transport = _tao_transport(
            load_airgapped_responses=[
                (
                    200,
                    {
                        "success": False,
                        "experiments_loaded": 0,
                        "experiments_failed": 2,
                    },
                ),
            ],
        )
        result = await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=lambda: s3,
        )
        assert result.failed
        joined = " ".join(err for _, err in result.failed)
        assert "experiments_failed=2" in joined


# ── #9: find_base_experiment_by_arch called per target ─────────────────────


class TestFindBaseExperimentPerTarget:
    @pytest.mark.asyncio
    async def test_find_base_experiment_called_per_target_after_success(
        self, workspace: Path
    ):
        """At least one list-base-experiments call per target after success
        (one in pre-pass + at least one in confirm settle loop)."""
        settings = _settings_for(workspace, TAO_API_KEY="jwt-token")
        s3 = FakeS3Client()
        runner = _ok_subprocess_runner()
        transport = _tao_transport(
            confirm_experiments=[
                {"id": "uuid-2b", "name": "Cosmos-Reason2-2B"},
                {"id": "uuid-8b", "name": "Cosmos-Reason2-8B"},
            ],
        )
        await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=lambda: s3,
        )
        # >= 4 list calls: 2 pre-pass + 2 confirm (one per target).
        assert transport.list_calls["n"] >= 4  # type: ignore[attr-defined]


# ── #10: default enumeration of seeded student_base ────────────────────────


class TestDefaultEnumeration:
    @pytest.mark.asyncio
    async def test_provision_base_experiments_default_enumerates_seeded_student_base(
        self, workspace: Path
    ):
        settings = _settings_for(workspace, TAO_API_KEY="jwt-token")

        captured_csv: dict[str, list[list[str]]] = {"rows": []}

        async def runner(cmd, *, env, timeout_s):
            # Read back the CSV the service wrote.
            idx = cmd.index("--csv")
            csv_path = Path(cmd[idx + 1])
            content = csv_path.read_text(encoding="utf-8")
            captured_csv["rows"] = [line.split(",") for line in content.splitlines()]
            stage_idx = cmd.index("--shared-folder-path")
            stage = Path(cmd[stage_idx + 1])
            stage.mkdir(parents=True, exist_ok=True)
            (stage / "ptm_metadatas.json").write_text(
                json.dumps({"Cosmos Reason2 2B": {}, "Cosmos Reason2 8B": {}}),
                encoding="utf-8",
            )
            return {"ok": True, "stdout": "{}", "stderr": "", "returncode": 0}

        s3 = FakeS3Client()
        transport = _tao_transport(
            confirm_experiments=[
                {"id": "uuid-2b", "name": "Cosmos-Reason2-2B"},
                {"id": "uuid-8b", "name": "Cosmos-Reason2-8B"},
                {"id": "uuid-nano", "name": "Cosmos3-Nano-Reasoner"},
                {"id": "uuid-super", "name": "Cosmos3-Super-Reasoner"},
            ],
        )
        result = await provision_base_experiments(
            settings,
            student_base_model_config_ids=None,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=lambda: s3,
        )
        rows = captured_csv["rows"]
        # Header + 4 rows (CR2 2B/8B + Cosmos 3 nano/super reasoner).
        assert rows[0] == ["displayName", "ngc_path", "network_arch", "is_backbone"]
        assert sorted(row[0] for row in rows[1:]) == [
            "Cosmos 3 Nano (Reasoner)",
            "Cosmos 3 Super (Reasoner)",
            "Cosmos Reason2 2B",
            "Cosmos Reason2 8B",
        ]
        ngc_paths = [r[1] for r in rows[1:]]
        assert sorted(ngc_paths) == [
            "hf_model://nvidia/Cosmos-Reason2-2B",
            "hf_model://nvidia/Cosmos-Reason2-8B",
            "hf_model://nvidia/Cosmos3-Nano-Reasoner",
            "hf_model://nvidia/Cosmos3-Super-Reasoner",
        ]
        # All four rows registered.
        assert sorted(result.registered) == [
            COSMOS_REASON2_2B,
            COSMOS_REASON2_8B,
            COSMOS3_NANO_REASONER,
            COSMOS3_SUPER_REASONER,
        ]


# ── #11: already-registered short-circuits everything ──────────────────────


class TestAlreadyRegisteredShortCircuit:
    @pytest.mark.asyncio
    async def test_already_registered_short_circuits_no_load_airgapped_call(
        self, workspace: Path
    ):
        settings = _settings_for(workspace, TAO_API_KEY="jwt-token")

        def s3_factory_must_not_be_called():
            pytest.fail("S3 must not be touched when already registered")

        async def runner_must_not_be_called(*args, **kwargs):
            pytest.fail("subprocess must not be spawned when already registered")

        transport = _tao_transport(
            # All targets already in TAO on the pre-pass.
            pre_pass_experiments=[
                {"id": "uuid-2b", "name": "Cosmos-Reason2-2B"},
                {"id": "uuid-8b", "name": "Cosmos-Reason2-8B"},
                {"id": "uuid-nano", "name": "Cosmos3-Nano-Reasoner"},
                {"id": "uuid-super", "name": "Cosmos3-Super-Reasoner"},
            ],
        )
        result = await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner_must_not_be_called,
            _s3_client_factory=s3_factory_must_not_be_called,
        )
        assert result.registered == []
        assert sorted(result.already_registered) == [
            COSMOS_REASON2_2B,
            COSMOS_REASON2_8B,
            COSMOS3_NANO_REASONER,
            COSMOS3_SUPER_REASONER,
        ]
        # No load_airgapped POST issued.
        assert not any(
            r.url.path.endswith("/jobs:load_airgapped")
            for r in transport.requests  # type: ignore[attr-defined]
        )

    @pytest.mark.asyncio
    async def test_already_registered_still_patches_project_dbs(self, workspace: Path):
        """Regression: a fresh project DB created since the prior provisioning
        run must still receive ``tao_base_experiment_id`` even when every
        target is in ``already_registered``. Without this, the operator
        sees the CLI report "already_registered" but ``training_preflight``
        fails because the project DB has nulls.
        """
        settings = _settings_for(workspace, TAO_API_KEY="jwt-token")
        # Create a fresh project so we have ModelConfig rows whose
        # tao_base_experiment_id is still null.
        seed_settings = make_settings(workspace)
        proj = create_project(name="P", description=None, settings=seed_settings)
        project_dir = workspace / "projects" / proj.project_id

        transport = _tao_transport(
            pre_pass_experiments=[
                {"id": "uuid-2b", "name": "Cosmos Reason2 2B"},
                {"id": "uuid-8b", "name": "Cosmos Reason2 8B"},
                {"id": "uuid-nano", "name": "Cosmos3-Nano-Reasoner"},
                {"id": "uuid-super", "name": "Cosmos3-Super-Reasoner"},
            ],
        )

        async def runner_must_not_be_called(*args, **kwargs):
            pytest.fail("subprocess must not run when already registered")

        def s3_factory_must_not_be_called():
            pytest.fail("S3 must not be touched when already registered")

        result = await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner_must_not_be_called,
            _s3_client_factory=s3_factory_must_not_be_called,
        )

        assert result.registered == []
        assert sorted(result.already_registered) == [
            COSMOS_REASON2_2B,
            COSMOS_REASON2_8B,
            COSMOS3_NANO_REASONER,
            COSMOS3_SUPER_REASONER,
        ]
        # Project DB rows MUST be patched even on the
        # all-already-registered short-circuit path.
        engine = open_project_db(project_dir)
        with Session(engine) as session:
            for model_name, expected_uuid in [
                (COSMOS_REASON2_2B, "uuid-2b"),
                (COSMOS_REASON2_8B, "uuid-8b"),
                (COSMOS3_NANO_REASONER, "uuid-nano"),
                (COSMOS3_SUPER_REASONER, "uuid-super"),
            ]:
                row = (
                    session.query(ModelConfig)
                    .filter_by(project_id=proj.project_id, model_name=model_name)
                    .one()
                )
                assert row.tao_base_experiment_id == expected_uuid
                assert row.tao_base_experiment_pull_status == "pull_complete"


# ── #12: failures leave ModelConfig rows untouched ─────────────────────────


class TestFailuresLeaveDbsUntouched:
    @pytest.mark.asyncio
    async def test_s3_upload_failure_leaves_modelconfig_rows_untouched(
        self, workspace: Path
    ):
        settings = _settings_for(
            workspace,
            TAO_API_KEY="jwt-token",
            TAO_WORKSPACE_S3_ACCESS_KEY="ak",
            TAO_WORKSPACE_S3_SECRET_KEY="sk",
        )
        # Create a project so we have a ModelConfig row to inspect.
        seed_settings = make_settings(workspace)
        proj = create_project(name="P", description=None, settings=seed_settings)
        project_dir = workspace / "projects" / proj.project_id

        # S3 client whose put_object raises.
        s3 = FakeS3Client(raise_on_put=True)
        runner = _ok_subprocess_runner(
            stage_seeded_files=[("Cosmos-Reason2-2B/tiny.bin", b"x" * 64)]
        )
        transport = _tao_transport()  # no load_airgapped traffic expected
        result = await provision_base_experiments(
            settings,
            _transport=transport,
            _subprocess_runner=runner,
            _s3_client_factory=lambda: s3,
        )
        assert result.failed
        # No load_airgapped POST.
        assert not any(
            r.url.path.endswith("/jobs:load_airgapped")
            for r in transport.requests  # type: ignore[attr-defined]
        )
        # Re-open project DB; tao_base_experiment_id stays null on the 2B row.
        engine = open_project_db(project_dir)
        with Session(engine) as session:
            row = (
                session.query(ModelConfig)
                .filter_by(project_id=proj.project_id, model_name=COSMOS_REASON2_2B)
                .one()
            )
            assert row.tao_base_experiment_id is None


# ── #13: FastAPI import boundary ───────────────────────────────────────────


class TestFastApiImportBoundary:
    """#13 — service module must not pull in transformers/peft/etc."""

    def test_provisioning_service_import_does_not_pull_ml_libs(self):
        code = (
            "import sys, json\n"
            "from vlm_feedback_loop.services.tao_base_experiment_provisioning_service "
            "import provision_base_experiments  # noqa\n"
            "loaded = list(sys.modules.keys())\n"
            "banned = ['nvidia_tao_core','transformers','peft','huggingface_hub']\n"
            "found = [m for m in banned if any(x == m or x.startswith(m + '.') "
            "for x in loaded)]\n"
            "print(json.dumps(found))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
        found = json.loads(proc.stdout.strip().splitlines()[-1])
        assert found == [], f"forbidden modules pulled into FastAPI process: {found}"

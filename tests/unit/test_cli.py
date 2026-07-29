# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bootstrap CLI (``vlm-feedback-loop init``)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import stat

import pytest
import yaml

from vlm_feedback_loop._defaults import DEFAULTS, SECRET_KEYS
from vlm_feedback_loop.cli import cmd_init
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
    EMBEDDING_MODEL_ID,
)


@pytest.fixture()
def run_init(patch_cli_paths, tmp_config_dir, tmp_workspace, monkeypatch):
    """Run ``cmd_init`` with a canned WORKSPACE_ROOT answer."""
    monkeypatch.setattr("builtins.input", lambda _: str(tmp_workspace))
    cmd_init(argparse.Namespace())
    return tmp_config_dir


class TestInitWorkspaceRootFlag:
    def test_flag_skips_interactive_prompt(
        self, patch_cli_paths, tmp_config_dir, tmp_workspace, monkeypatch
    ):
        """--workspace-root makes init non-interactive so the README's
        copy-paste quick start works without a hidden input() prompt."""

        def _no_input(_prompt: str) -> str:
            raise AssertionError("init must not prompt when --workspace-root is given")

        monkeypatch.setattr("builtins.input", _no_input)
        cmd_init(argparse.Namespace(workspace_root=str(tmp_workspace)))
        config = yaml.safe_load((tmp_config_dir / "config.yaml").read_text())
        assert config["WORKSPACE_ROOT"] == str(tmp_workspace)

    def test_flag_expands_tilde(
        self, patch_cli_paths, tmp_config_dir, tmp_workspace, monkeypatch
    ):
        """A leading ~ is expanded so `--workspace-root ~/vlm-workspace`
        (the documented example) resolves instead of failing the
        absolute-path check."""
        monkeypatch.setenv("HOME", str(tmp_workspace))
        cmd_init(argparse.Namespace(workspace_root="~/ws"))
        config = yaml.safe_load((tmp_config_dir / "config.yaml").read_text())
        assert config["WORKSPACE_ROOT"] == str(tmp_workspace / "ws")


class TestInitCreatesFiles:
    def test_config_yaml_created(self, run_init):
        assert (run_init / "config.yaml").exists()

    def test_env_created(self, run_init):
        assert (run_init / ".env").exists()


class TestInitPermissions:
    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="chmod semantics differ on Windows",
    )
    def test_directory_permissions_700(self, run_init):
        mode = stat.S_IMODE(os.stat(run_init).st_mode)
        assert mode == 0o700

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="chmod semantics differ on Windows",
    )
    def test_env_permissions_600(self, run_init):
        mode = stat.S_IMODE(os.stat(run_init / ".env").st_mode)
        assert mode == 0o600


class TestInitConfigContent:
    def test_workspace_root_set(self, run_init, tmp_workspace):
        data = yaml.safe_load(open(run_init / "config.yaml"))
        assert data["WORKSPACE_ROOT"] == str(tmp_workspace)

    def test_workspace_root_is_the_only_active_key(self, run_init, tmp_workspace):
        """Init must not snapshot defaults as active keys: any key present
        in config.yaml is an operator override that pins its install-date
        value forever, silently shadowing every later default change (a
        default-Teacher reseat would never reach an init'd deployment)."""
        data = yaml.safe_load(open(run_init / "config.yaml"))
        assert data == {"WORKSPACE_ROOT": str(tmp_workspace)}

    def test_all_non_secret_defaults_documented_as_comments(self, run_init):
        """The generated file stays teachable: every non-secret default is
        present as a commented line the operator can uncomment to override."""
        content = (run_init / "config.yaml").read_text()
        for key in DEFAULTS:
            if key not in SECRET_KEYS:
                assert f"# {key}:" in content, f"Missing documented default: {key}"

    def test_commented_defaults_uncomment_to_shipped_values(self, run_init):
        """The documentation block round-trips: uncommenting everything
        below the defaults marker parses to exactly the shipped non-secret
        defaults — the file is a faithful reference, not a stale snapshot."""
        content = (run_init / "config.yaml").read_text()
        marker = "uncomment to override"
        assert marker in content
        block = content.split(marker, 1)[1].splitlines()[1:]
        uncommented = [
            line[2:] if line.startswith("# ") else line
            for line in block
            if line.strip()
        ]
        data = yaml.safe_load("\n".join(uncommented))
        expected = {k: v for k, v in DEFAULTS.items() if k not in SECRET_KEYS}
        assert data == expected

    def test_no_secret_keys_in_config(self, run_init):
        """Secrets belong in .env only — not even as commented documentation
        lines that could tempt an operator to paste a key here."""
        content = (run_init / "config.yaml").read_text()
        for key in SECRET_KEYS:
            assert key not in content, f"Secret key should not be in config.yaml: {key}"


class TestInitEnvContent:
    def test_env_has_no_active_variables(self, run_init):
        """Skeleton .env should contain only comments (no uncommented lines)."""
        content = (run_init / ".env").read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                pytest.fail(f"Active variable found in skeleton .env: {stripped}")

    def test_env_documents_secrets(self, run_init):
        """Skeleton .env should mention the key secret variable names."""
        content = (run_init / ".env").read_text()
        assert "NVIDIA_API_KEY" in content
        assert "NGC_API_KEY" in content
        assert "TAO_API_KEY" in content


class TestInitDeploymentDb:
    def test_deployment_db_created(self, run_init, tmp_workspace):
        """Verify cmd_init creates deployment.db with EmbeddingDeploymentConfig singleton."""
        deployment_db = tmp_workspace / "deployment.db"
        assert deployment_db.exists()

    def test_deployment_db_singleton_seeded(self, run_init, tmp_workspace):
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig
        from vlm_feedback_loop.db.engine import init_deployment_db

        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            configs = session.query(EmbeddingDeploymentConfig).all()
            assert len(configs) == 1
            assert configs[0].model_name == EMBEDDING_MODEL_ID


class TestInitNoClobber:
    """`init` must never destroy an operator's configuration silently:
    config.yaml requires --force to regenerate; .env (API keys) is never
    overwritten at all."""

    def test_second_init_refuses_without_force(
        self, run_init, tmp_workspace, monkeypatch, capsys
    ):
        monkeypatch.setattr("builtins.input", lambda _: str(tmp_workspace))
        with pytest.raises(SystemExit):
            cmd_init(argparse.Namespace())
        assert "--force" in capsys.readouterr().err

    def test_force_regenerates_config_but_preserves_env(
        self, run_init, tmp_workspace, monkeypatch
    ):
        env_file = run_init / ".env"
        env_file.write_text("NVIDIA_API_KEY=nvapi-precious\n")
        monkeypatch.setattr("builtins.input", lambda _: str(tmp_workspace))
        cmd_init(argparse.Namespace(force=True))
        assert "nvapi-precious" in env_file.read_text()
        assert (run_init / "config.yaml").exists()


class TestInitValidation:
    def test_rejects_empty_path(self, patch_cli_paths, tmp_config_dir, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        with pytest.raises(SystemExit):
            cmd_init(argparse.Namespace())

    def test_rejects_relative_path(self, patch_cli_paths, tmp_config_dir, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "relative/path")
        with pytest.raises(SystemExit):
            cmd_init(argparse.Namespace())


# ── tao-bootstrap CLI ────────────────────────────────────────────────────────


def _build_patched_workspace_service(
    *,
    request_tracker: dict[str, int],
    credential_tracker: dict[str, str | None] | None = None,
    workspace_id: str = "ws-uuid-1",
    already_provisioned: bool = False,
):
    """Return a stand-in module object compatible with _run_tao_bootstrap.

    ``request_tracker['count']`` increments on every call so the
    idempotency test can confirm re-runs make ZERO new calls.
    """
    from vlm_feedback_loop.services.tao_workspace_service import WorkspaceResult

    class _StubService:
        @staticmethod
        async def create_or_get_workspace(settings, **kwargs):
            request_tracker["count"] += 1
            if credential_tracker is not None:
                credential_tracker["access_key"] = kwargs.get("access_key")
                credential_tracker["secret_key"] = kwargs.get("secret_key")
            # Mirror the real service's DB-write side effect so tests can
            # verify that bootstrap persists workspace state to deployment.db
            # (the canonical source of truth for workspace identity).
            if not already_provisioned and settings.WORKSPACE_ROOT:
                from sqlalchemy.orm import Session

                from vlm_feedback_loop.db.deployment_models import (
                    TAODeploymentConfig,
                )
                from vlm_feedback_loop.db.engine import init_deployment_db

                engine = init_deployment_db(settings.WORKSPACE_ROOT)
                with Session(engine) as session:
                    cfg = session.query(TAODeploymentConfig).first()
                    assert cfg is not None
                    cfg.tao_workspace_id = workspace_id
                    cfg.tao_workspace_name = kwargs.get("workspace_name")
                    cfg.tao_workspace_cloud_type = kwargs.get("cloud_type")
                    cfg.tao_workspace_bucket = kwargs.get("bucket")
                    cfg.tao_workspace_s3_endpoint_url_internal = kwargs.get(
                        "endpoint_url_internal"
                    )
                    cfg.tao_workspace_s3_endpoint_url_external = kwargs.get(
                        "endpoint_url_external"
                    )
                    cfg.tao_workspace_s3_access_key_ref = "TAO_WORKSPACE_S3_ACCESS_KEY"
                    cfg.tao_workspace_s3_secret_key_ref = "TAO_WORKSPACE_S3_SECRET_KEY"
                    session.commit()
            return WorkspaceResult(
                success=True,
                workspace_id=workspace_id,
                workspace_detail={"id": workspace_id},
                already_provisioned=already_provisioned,
            )

    return _StubService()


def _make_settings_loader(
    workspace_root,
    *,
    s3_access_key: str = "SENTINEL_S3_ACCESS",
    s3_secret_key: str = "SENTINEL_S3_SECRET",
):
    """Return an argument-free settings loader for _run_tao_bootstrap.

    Bypasses config.yaml / .env resolution so tests don't need to patch
    config module paths — only the CLI paths (for .env writes) matter.
    """
    from conftest import make_settings

    def _load():
        return make_settings(
            workspace_root,
            TAO_API_BASE_URL="https://tao.example/api/v2",
            TAO_API_KEY="test-jwt",
            TAO_ORG_NAME="test-org",
            TAO_WORKSPACE_S3_ACCESS_KEY=s3_access_key,
            TAO_WORKSPACE_S3_SECRET_KEY=s3_secret_key,
        )

    return _load


class TestTaoBootstrapCli:
    """tao-bootstrap orchestration with mocked TAO.

    The core flow creates/adopts the workspace using configured secrets —
    base-experiment provisioning belongs exclusively to the
    ``--self-service`` / ``--admin-managed`` branches (stock FTMS
    exposes no client-driven ``:pull_from_ngc`` route)."""

    @pytest.mark.asyncio
    async def test_end_to_end_uses_configured_secrets_without_persisting(
        self,
        run_init,
        tmp_workspace,
        monkeypatch,
    ):
        """Workspace creation uses Settings and leaves .env byte-stable."""
        from vlm_feedback_loop import cli as cli_mod

        tracker: dict[str, int] = {"count": 0}
        credentials: dict[str, str | None] = {}
        stub = _build_patched_workspace_service(
            request_tracker=tracker,
            credential_tracker=credentials,
        )

        env_path = run_init / ".env"
        env_before = env_path.read_bytes()

        # Run the orchestration core directly (avoids asyncio.run nesting).
        await cli_mod._run_tao_bootstrap(
            workspace_name="my-ws",
            cloud_type="seaweedfs",
            bucket="my-bucket",
            s3_endpoint_url_external="http://127.0.0.1:8333",
            s3_endpoint_url_internal="http://seaweedfs-s3:8333",
            _workspace_service=stub,
            _settings_loader=_make_settings_loader(tmp_workspace),
        )

        assert env_path.read_bytes() == env_before
        # Exactly one workspace create — no base-experiment pulls (stock
        # FTMS has no :pull_from_ngc route; provisioning is a mode branch).
        assert tracker["count"] == 1
        assert credentials == {
            "access_key": "SENTINEL_S3_ACCESS",
            "secret_key": "SENTINEL_S3_SECRET",
        }

        # TAODeploymentConfig singleton flipped to "bootstrapped" — and
        # holds the non-secret workspace state (never echoed to .env).
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
        from vlm_feedback_loop.db.engine import init_deployment_db

        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg.bootstrap_status == "bootstrapped"
            assert cfg.bootstrap_last_run_at is not None
            assert cfg.tao_workspace_id == "ws-uuid-1"
            assert cfg.tao_workspace_bucket == "my-bucket"

    @pytest.mark.asyncio
    async def test_idempotent_rerun_makes_zero_tao_calls(self, run_init, tmp_workspace):
        """Re-run against already-provisioned deployment → 1 GET only."""
        from vlm_feedback_loop import cli as cli_mod

        # First run: normal bootstrap.
        tracker: dict[str, int] = {"count": 0}
        first_stub = _build_patched_workspace_service(
            request_tracker=tracker, already_provisioned=False
        )
        await cli_mod._run_tao_bootstrap(
            workspace_name="my-ws",
            cloud_type="seaweedfs",
            bucket="my-bucket",
            s3_endpoint_url_external="http://127.0.0.1:8333",
            s3_endpoint_url_internal="http://seaweedfs-s3:8333",
            _workspace_service=first_stub,
            _settings_loader=_make_settings_loader(tmp_workspace),
        )
        assert tracker["count"] == 1  # create only

        # Second run: stub signals already_provisioned=True.
        tracker2: dict[str, int] = {"count": 0}
        second_stub = _build_patched_workspace_service(
            request_tracker=tracker2, already_provisioned=True
        )
        await cli_mod._run_tao_bootstrap(
            workspace_name="my-ws",
            cloud_type="seaweedfs",
            bucket="my-bucket",
            s3_endpoint_url_external="http://127.0.0.1:8333",
            s3_endpoint_url_internal="http://seaweedfs-s3:8333",
            _workspace_service=second_stub,
            _settings_loader=_make_settings_loader(tmp_workspace),
        )
        # Only the one workspace-lookup (already_provisioned stub's
        # create_or_get_workspace call).
        assert tracker2["count"] == 1

    @pytest.mark.asyncio
    async def test_missing_s3_secret_fails_before_db_or_tao_mutation(
        self, tmp_workspace
    ):
        """Missing environment configuration cannot start a partial bootstrap."""
        from vlm_feedback_loop import cli as cli_mod

        tracker: dict[str, int] = {"count": 0}
        stub = _build_patched_workspace_service(request_tracker=tracker)
        with pytest.raises(RuntimeError, match="TAO_WORKSPACE_S3_SECRET_KEY"):
            await cli_mod._run_tao_bootstrap(
                workspace_name="my-ws",
                cloud_type="seaweedfs",
                bucket="my-bucket",
                s3_endpoint_url_external="http://127.0.0.1:8333",
                s3_endpoint_url_internal="http://seaweedfs-s3:8333",
                _workspace_service=stub,
                _settings_loader=_make_settings_loader(tmp_workspace, s3_secret_key=""),
            )
        assert tracker["count"] == 0
        assert not (tmp_workspace / "deployment.db").exists()

    @pytest.mark.asyncio
    async def test_fresh_bootstrap_completes_against_ftms_without_pull_route(
        self, run_init, tmp_workspace
    ):
        """Stock FTMS (6.25.11 / 6.26.3 per the live OpenAPI probes)
        exposes no client-driven ``:pull_from_ngc`` route — a fresh
        bootstrap over the real workspace-service wire path must complete
        without ever requesting one."""
        import httpx
        from sqlalchemy.orm import Session

        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
        from vlm_feedback_loop.db.engine import init_deployment_db
        from vlm_feedback_loop.services import tao_workspace_service as ws_svc

        seen_paths: list[str] = []

        def _ftms(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            if request.method == "GET" and request.url.path.endswith("/workspaces"):
                return httpx.Response(200, json={"workspaces": []})
            if request.method == "POST" and request.url.path.endswith("/workspaces"):
                return httpx.Response(201, json={"id": "ws-wire-1"})
            return httpx.Response(404, json={"error": "no such route"})

        transport = httpx.MockTransport(_ftms)

        class _WireService:
            """The real tao_workspace_service bound to the emulated FTMS."""

            @staticmethod
            async def create_or_get_workspace(settings, **kwargs):
                return await ws_svc.create_or_get_workspace(
                    settings, **kwargs, _transport=transport
                )

        await cli_mod._run_tao_bootstrap(
            workspace_name="wire-ws",
            cloud_type="seaweedfs",
            bucket="wire-bucket",
            s3_endpoint_url_external="http://127.0.0.1:8333",
            s3_endpoint_url_internal="http://seaweedfs-s3:8333",
            _workspace_service=_WireService(),
            _settings_loader=_make_settings_loader(tmp_workspace),
        )

        assert not [p for p in seen_paths if "pull_from_ngc" in p]
        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg.bootstrap_status == "bootstrapped"
            assert cfg.tao_workspace_id == "ws-wire-1"

    @pytest.mark.asyncio
    async def test_workspace_create_failure_sets_failed_status(
        self, run_init, tmp_workspace
    ):
        """A failed workspace create stamps bootstrap_status='failed' with
        an actionable error_ref (the deployment stays re-runnable)."""
        from sqlalchemy.orm import Session

        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
        from vlm_feedback_loop.db.engine import init_deployment_db
        from vlm_feedback_loop.services.tao_workspace_service import WorkspaceResult

        class FailingStub:
            @staticmethod
            async def create_or_get_workspace(settings, **kwargs):  # noqa: ARG004
                return WorkspaceResult(
                    success=False,
                    error="FTMS rejected the workspace body",
                )

        with pytest.raises(RuntimeError, match="workspace create failed"):
            await cli_mod._run_tao_bootstrap(
                workspace_name="my-ws",
                cloud_type="seaweedfs",
                bucket="my-bucket",
                s3_endpoint_url_external="http://127.0.0.1:8333",
                s3_endpoint_url_internal="http://seaweedfs-s3:8333",
                _workspace_service=FailingStub(),
                _settings_loader=_make_settings_loader(tmp_workspace),
            )

        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg.bootstrap_status == "failed"
            assert cfg.bootstrap_error_ref is not None
            assert "rejected" in cfg.bootstrap_error_ref

    @pytest.mark.asyncio
    async def test_workspace_error_echoing_s3_secret_is_redacted(self, tmp_workspace):
        """An FTMS error body cannot enter an exception or deployment.db with a key."""
        import httpx
        from sqlalchemy.orm import Session

        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
        from vlm_feedback_loop.db.engine import init_deployment_db
        from vlm_feedback_loop.services import tao_workspace_service as ws_svc

        sentinel = "SENTINEL_TAO_ECHOED_S3_SECRET"

        def _ftms(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path.endswith("/workspaces"):
                return httpx.Response(200, json={"workspaces": []})
            if request.method == "POST" and request.url.path.endswith("/workspaces"):
                return httpx.Response(
                    400,
                    json={"error": {"message": f"invalid secret {sentinel}"}},
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(_ftms)

        class _WireService:
            @staticmethod
            async def create_or_get_workspace(settings, **kwargs):
                return await ws_svc.create_or_get_workspace(
                    settings, **kwargs, _transport=transport
                )

        with pytest.raises(RuntimeError) as exc_info:
            await cli_mod._run_tao_bootstrap(
                workspace_name="wire-ws",
                cloud_type="seaweedfs",
                bucket="wire-bucket",
                s3_endpoint_url_external="http://127.0.0.1:8333",
                s3_endpoint_url_internal="http://seaweedfs-s3:8333",
                _workspace_service=_WireService(),
                _settings_loader=_make_settings_loader(
                    tmp_workspace,
                    s3_secret_key=sentinel,
                ),
            )

        assert sentinel not in str(exc_info.value)
        assert "[REDACTED]" in str(exc_info.value)
        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg is not None
            assert cfg.bootstrap_error_ref is not None
            assert sentinel not in cfg.bootstrap_error_ref
            assert "[REDACTED]" in cfg.bootstrap_error_ref


# ── tao-pull-base-experiments + tao-bootstrap mode flags ────────────────────


def _seed_tao_workspace(tmp_workspace, workspace_id: str = "ws-1") -> None:
    """Mark deployment.db as bootstrapped so cmd_tao_pull_base_experiments
    finds a valid TAO workspace.

    The gate in `cmd_tao_pull_base_experiments` reads from
    deployment.db.tao_deployment_configs, not Settings.
    """
    from sqlalchemy.orm import Session

    from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
    from vlm_feedback_loop.db.engine import init_deployment_db

    engine = init_deployment_db(tmp_workspace)
    with Session(engine) as session:
        cfg = session.query(TAODeploymentConfig).first()
        assert cfg is not None
        cfg.tao_workspace_id = workspace_id
        cfg.bootstrap_status = "bootstrapped"
        session.commit()


def _ns_pull_args(**overrides):
    """Build a Namespace shaped like argparse would for tao-pull-base-experiments."""
    base = {
        "model_config_id": [],
        "skip_install": True,  # tests skip the pip install by default
        "dry_run": False,
        "timeout_download_s": 1800.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _ns_bootstrap_args(**overrides):
    """Build a Namespace shaped like argparse would for tao-bootstrap."""
    base = {
        "workspace_name": "my-ws",
        "cloud_type": "seaweedfs",
        "bucket": "my-bucket",
        "s3_endpoint_url_external": "http://127.0.0.1:8333",
        "s3_endpoint_url_internal": "http://seaweedfs-s3:8333",
        "base_experiment_id_8b": None,
        "base_experiment_id_2b": None,
        "admin_managed": False,
        "eager_bases": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestTaoPullBaseExperimentsCli:
    """Mocked tests for the ``tao-pull-base-experiments`` subcommand."""

    def test_cli_pull_happy_path(
        self, run_init, patch_config_paths, tmp_workspace, monkeypatch, capsys
    ):
        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.services import (
            tao_base_experiment_provisioning_service as prov_mod,
        )

        captured: dict[str, object] = {}

        async def fake_provision(settings, student_base_model_config_ids=None, **kw):
            captured["tao_api_key"] = settings.TAO_API_KEY
            captured["hf_token"] = kw.get("hf_token")
            return prov_mod.ProvisioningResult(
                registered=[COSMOS_REASON2_2B],
                already_registered=[],
                failed=[],
            )

        monkeypatch.setattr(prov_mod, "provision_base_experiments", fake_provision)
        monkeypatch.setenv("NGC_API_KEY", "SENTINEL_NGC")
        monkeypatch.setenv("HF_TOKEN", "SENTINEL_HF")
        env_before = (run_init / ".env").read_bytes()
        _seed_tao_workspace(tmp_workspace)

        cli_mod.cmd_tao_pull_base_experiments(_ns_pull_args())
        out = capsys.readouterr().out
        body = json.loads(out)
        assert body["registered"] == [COSMOS_REASON2_2B]
        assert body["dry_run"] is False
        assert captured == {
            "tao_api_key": "SENTINEL_NGC",
            "hf_token": "SENTINEL_HF",
        }
        assert (run_init / ".env").read_bytes() == env_before

    def test_cli_pull_falls_back_to_canonical_env_tao_key(
        self, run_init, patch_config_paths, tmp_workspace, monkeypatch
    ):
        """TAO_API_KEY in canonical .env remains the NGC-key fallback."""
        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.services import (
            tao_base_experiment_provisioning_service as prov_mod,
        )

        sentinel = "SENTINEL_CANONICAL_TAO_KEY"
        env_path = run_init / ".env"
        env_path.write_text(env_path.read_text() + f"\nTAO_API_KEY={sentinel}\n")
        env_before = env_path.read_bytes()
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        monkeypatch.delenv("TAO_API_KEY", raising=False)
        monkeypatch.delenv("VLM_FEEDBACK_LOOP_ENV_FILE", raising=False)
        captured: dict[str, object] = {}

        async def fake_provision(settings, student_base_model_config_ids=None, **kw):
            captured["tao_api_key"] = settings.TAO_API_KEY
            return prov_mod.ProvisioningResult()

        monkeypatch.setattr(prov_mod, "provision_base_experiments", fake_provision)
        _seed_tao_workspace(tmp_workspace)

        cli_mod.cmd_tao_pull_base_experiments(_ns_pull_args())

        assert captured["tao_api_key"] == sentinel
        assert env_path.read_bytes() == env_before

    def test_cli_pull_dry_run_passes_dry_run_kwarg(
        self, run_init, patch_config_paths, tmp_workspace, monkeypatch
    ):
        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.services import (
            tao_base_experiment_provisioning_service as prov_mod,
        )

        captured: dict[str, object] = {}

        async def spy(settings, student_base_model_config_ids=None, **kw):
            captured["dry_run"] = kw.get("_dry_run")
            return prov_mod.ProvisioningResult()

        monkeypatch.setattr(prov_mod, "provision_base_experiments", spy)
        monkeypatch.setenv("NGC_API_KEY", "nvapi-key")
        _seed_tao_workspace(tmp_workspace)

        cli_mod.cmd_tao_pull_base_experiments(_ns_pull_args(dry_run=True))
        assert captured["dry_run"] is True

    def test_cli_pull_skip_install_uses_current_interpreter(
        self, run_init, patch_config_paths, tmp_workspace, monkeypatch
    ):
        """--skip-install selects the operator-managed helper environment."""
        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.services import (
            tao_base_experiment_provisioning_service as prov_mod,
        )

        captured: dict[str, object] = {}

        async def fake_provision(settings, student_base_model_config_ids=None, **kw):
            captured["use_isolated_helper"] = kw.get("use_isolated_helper")
            return prov_mod.ProvisioningResult()

        monkeypatch.setattr(prov_mod, "provision_base_experiments", fake_provision)
        monkeypatch.setenv("NGC_API_KEY", "nvapi-key")
        _seed_tao_workspace(tmp_workspace)

        cli_mod.cmd_tao_pull_base_experiments(_ns_pull_args(skip_install=True))
        assert captured["use_isolated_helper"] is False

    def test_cli_pull_missing_ngc_key_clear_error(
        self, run_init, patch_config_paths, monkeypatch, capsys
    ):
        from vlm_feedback_loop import cli as cli_mod

        # Clear all NGC-key sources.
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        monkeypatch.delenv("TAO_API_KEY", raising=False)

        with pytest.raises(SystemExit):
            cli_mod.cmd_tao_pull_base_experiments(_ns_pull_args())
        err = capsys.readouterr().err
        assert "NGC_API_KEY" in err
        assert "TAO_API_KEY" in err


class TestTaoBootstrapModeFlags:
    """Mocked tests for the tao-bootstrap --self-service / --admin-managed mode flags."""

    def test_cli_bootstrap_self_service_defers_base_provisioning(
        self, run_init, patch_config_paths, tmp_workspace, monkeypatch
    ):
        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.services import (
            tao_base_experiment_provisioning_service as prov_mod,
        )

        monkeypatch.setenv("TAO_WORKSPACE_S3_ACCESS_KEY", "SENTINEL_S3_ACCESS")
        monkeypatch.setenv("TAO_WORKSPACE_S3_SECRET_KEY", "SENTINEL_S3_SECRET")

        # Stub the workspace service so _run_tao_bootstrap completes cleanly.
        tracker: dict[str, int] = {"count": 0}
        stub = _build_patched_workspace_service(request_tracker=tracker)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_workspace_service.create_or_get_workspace",
            stub.create_or_get_workspace,
        )

        provision_calls = {"n": 0}

        async def fake_provision(settings, student_base_model_config_ids=None, **kw):
            provision_calls["n"] += 1
            return prov_mod.ProvisioningResult(registered=[COSMOS_REASON2_2B])

        monkeypatch.setattr(prov_mod, "provision_base_experiments", fake_provision)

        # Run synchronously (cmd_tao_bootstrap uses asyncio.run internally).
        cli_mod.cmd_tao_bootstrap(_ns_bootstrap_args())
        assert provision_calls["n"] == 0

    def test_cli_bootstrap_eager_bases_invokes_provision(
        self, run_init, patch_config_paths, tmp_workspace, monkeypatch
    ):
        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.services import (
            tao_base_experiment_provisioning_service as prov_mod,
        )

        monkeypatch.setenv("TAO_WORKSPACE_S3_ACCESS_KEY", "SENTINEL_S3_ACCESS")
        monkeypatch.setenv("TAO_WORKSPACE_S3_SECRET_KEY", "SENTINEL_S3_SECRET")
        stub = _build_patched_workspace_service(request_tracker={"count": 0})
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_workspace_service.create_or_get_workspace",
            stub.create_or_get_workspace,
        )
        provision_calls = {"n": 0}

        async def fake_provision(settings, student_base_model_config_ids=None, **kw):
            provision_calls["n"] += 1
            return prov_mod.ProvisioningResult()

        monkeypatch.setattr(prov_mod, "provision_base_experiments", fake_provision)

        cli_mod.cmd_tao_bootstrap(_ns_bootstrap_args(eager_bases=True))
        assert provision_calls["n"] == 1

    def test_cli_bootstrap_admin_managed_skips_provision(
        self, run_init, patch_config_paths, tmp_workspace, monkeypatch, capsys
    ):
        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop.services import (
            tao_base_experiment_provisioning_service as prov_mod,
        )

        monkeypatch.setenv("TAO_WORKSPACE_S3_ACCESS_KEY", "SENTINEL_S3_ACCESS")
        monkeypatch.setenv("TAO_WORKSPACE_S3_SECRET_KEY", "SENTINEL_S3_SECRET")

        tracker: dict[str, int] = {"count": 0}
        stub = _build_patched_workspace_service(request_tracker=tracker)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_workspace_service.create_or_get_workspace",
            stub.create_or_get_workspace,
        )

        async def fail_if_called(*a, **k):
            pytest.fail(
                "provision_base_experiments must not be called in admin-managed mode"
            )

        monkeypatch.setattr(prov_mod, "provision_base_experiments", fail_if_called)

        cli_mod.cmd_tao_bootstrap(_ns_bootstrap_args(admin_managed=True))
        out = capsys.readouterr().out
        assert "docs/tao-ftms-install.md" in out

    def test_cli_bootstrap_rejects_eager_bases_in_admin_mode(self, monkeypatch, capsys):
        """The eager self-service modifier cannot silently alter admin mode."""
        from vlm_feedback_loop import cli as cli_mod

        monkeypatch.setattr(
            cli_mod,
            "_run_tao_bootstrap",
            lambda **_kwargs: pytest.fail("workspace must not be mutated"),
        )

        with pytest.raises(SystemExit) as exc_info:
            cli_mod.cmd_tao_bootstrap(
                _ns_bootstrap_args(admin_managed=True, eager_bases=True)
            )

        assert exc_info.value.code == 2
        assert "--eager-bases is only valid in self-service mode" in (
            capsys.readouterr().err
        )

    def test_cli_bootstrap_admin_managed_with_uuid_patches_dbs(
        self, run_init, patch_config_paths, tmp_workspace, monkeypatch
    ):
        """--admin-managed --base-experiment-id-2b=<UUID> writes UUIDs into project DBs."""
        from sqlalchemy.orm import Session

        from vlm_feedback_loop import cli as cli_mod
        from vlm_feedback_loop._defaults import DEFAULTS, SECRET_KEYS
        from vlm_feedback_loop.config import Settings
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.services import (
            tao_base_experiment_provisioning_service as prov_mod,
        )
        from vlm_feedback_loop.services.project_service import create_project

        monkeypatch.setenv("TAO_WORKSPACE_S3_ACCESS_KEY", "SENTINEL_S3_ACCESS")
        monkeypatch.setenv("TAO_WORKSPACE_S3_SECRET_KEY", "SENTINEL_S3_SECRET")

        non_secret = {k: v for k, v in DEFAULTS.items() if k not in SECRET_KEYS}
        settings = Settings(WORKSPACE_ROOT=str(tmp_workspace), **non_secret)
        proj = create_project(name="P", description=None, settings=settings)

        tracker: dict[str, int] = {"count": 0}
        stub = _build_patched_workspace_service(request_tracker=tracker)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_workspace_service.create_or_get_workspace",
            stub.create_or_get_workspace,
        )

        async def fail_if_called(*a, **k):
            pytest.fail(
                "provision_base_experiments must not be called in admin-managed mode"
            )

        monkeypatch.setattr(prov_mod, "provision_base_experiments", fail_if_called)

        cli_mod.cmd_tao_bootstrap(
            _ns_bootstrap_args(
                admin_managed=True,
                base_experiment_id_2b="admin-uuid-2b",
                base_experiment_id_8b="admin-uuid-8b",
            )
        )

        engine = open_project_db(tmp_workspace / "projects" / proj.project_id)
        with Session(engine) as session:
            two_b = (
                session.query(ModelConfig)
                .filter_by(project_id=proj.project_id, model_name=COSMOS_REASON2_2B)
                .one()
            )
            assert two_b.tao_base_experiment_id == "admin-uuid-2b"
            assert two_b.tao_base_experiment_pull_status == "pull_complete"
            eight_b = (
                session.query(ModelConfig)
                .filter_by(project_id=proj.project_id, model_name=COSMOS_REASON2_8B)
                .one()
            )
            assert eight_b.tao_base_experiment_id == "admin-uuid-8b"

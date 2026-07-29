# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TAO validation drivers' canonical workspace resolver."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db

# ``scripts/`` is not a package; load the shared validation module explicitly.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

tao_validation = importlib.import_module("tao_validation")
resolve_workspace_state = tao_validation.resolve_workspace_state
ResolvedWorkspaceState = tao_validation.ResolvedWorkspaceState


class _FakeSettings:
    """Minimal stand-in; the resolver only needs the workspace root."""

    def __init__(self, workspace_root: Path) -> None:
        self.WORKSPACE_ROOT = workspace_root


def _seed_deployment_config(
    workspace_root: Path,
    *,
    tao_workspace_id: str | None,
    bootstrap_status: str = "bootstrapped",
    bucket: str | None = "blueprint",
    endpoint_external: str | None = "http://127.0.0.1:8333",
    endpoint_internal: str | None = "http://seaweedfs-s3:8333",
) -> None:
    """Stand up deployment.db and write a TAODeploymentConfig row."""
    engine = init_deployment_db(workspace_root)
    with Session(engine) as session:
        cfg = session.query(TAODeploymentConfig).first()
        # init_deployment_db pre-creates the singleton row.
        assert cfg is not None
        cfg.tao_workspace_id = tao_workspace_id
        cfg.tao_workspace_bucket = bucket
        cfg.tao_workspace_cloud_type = "seaweedfs"
        cfg.tao_workspace_s3_endpoint_url_external = endpoint_external
        cfg.tao_workspace_s3_endpoint_url_internal = endpoint_internal
        cfg.tao_workspace_s3_access_key_ref = "TAO_WORKSPACE_S3_ACCESS_KEY"
        cfg.tao_workspace_s3_secret_key_ref = "TAO_WORKSPACE_S3_SECRET_KEY"
        cfg.bootstrap_status = bootstrap_status
        cfg.bootstrap_last_run_at = utc_now()
        session.commit()


class TestResolveWorkspaceState:
    """The build-mode resolver has one non-secret configuration source."""

    def test_bootstrapped_workspace_uses_deployment_db(
        self, tmp_workspace: Path
    ) -> None:
        _seed_deployment_config(
            tmp_workspace, tao_workspace_id="8fcbea27-5220-451d-bdcb-494c8be428bb"
        )
        settings = _FakeSettings(workspace_root=tmp_workspace)

        state = resolve_workspace_state(settings)

        assert isinstance(state, ResolvedWorkspaceState)
        assert state.tao_workspace_id == "8fcbea27-5220-451d-bdcb-494c8be428bb"
        assert state.tao_workspace_bucket == "blueprint"
        assert state.tao_workspace_s3_endpoint_url_external == "http://127.0.0.1:8333"
        assert (
            state.tao_workspace_s3_endpoint_url_internal == "http://seaweedfs-s3:8333"
        )

    def test_unbootstrapped_workspace_names_canonical_remediation(
        self, tmp_workspace: Path
    ) -> None:
        _seed_deployment_config(
            tmp_workspace,
            tao_workspace_id=None,
            bootstrap_status="not_bootstrapped",
        )
        settings = _FakeSettings(workspace_root=tmp_workspace)

        with pytest.raises(SystemExit) as excinfo:
            resolve_workspace_state(settings)

        msg = str(excinfo.value)
        assert "tao-bootstrap" in msg
        assert "deployment.db.tao_deployment_configs" in msg

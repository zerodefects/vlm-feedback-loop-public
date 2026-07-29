# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `services.tao_bootstrap_service`."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from conftest import make_settings
from vlm_feedback_loop.db.engine import init_deployment_db, open_project_db
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.services.project_service import create_project
from vlm_feedback_loop.services.tao_bootstrap_service import (
    iter_project_dirs,
    patch_model_configs_across_projects,
)


@pytest.fixture()
def workspace(tmp_workspace):
    """Bare workspace with deployment.db seeded — local equivalent of
    ``test_cli.run_init`` minus the .env / config.yaml writes."""
    init_deployment_db(tmp_workspace)
    return tmp_workspace


class TestIterProjectDirs:
    """Yield every project dir that has a project.db; skip incomplete ones."""

    def test_iter_project_dirs_yields_only_dirs_with_project_db(self, workspace):
        # No projects yet → empty.
        assert list(iter_project_dirs(workspace)) == []

        # Create one real project.
        settings = make_settings(workspace)
        p = create_project(name="P1", description=None, settings=settings)

        # Create a sibling directory missing project.db (simulates a
        # partial / aborted creation).
        (workspace / "projects" / "garbage").mkdir(parents=True, exist_ok=True)

        result = list(iter_project_dirs(workspace))
        names = sorted(r.name for r in result)
        assert names == [p.project_id]


class TestPatchModelConfigsAcrossProjects:
    """patch_model_configs_across_projects writes the right rows + status."""

    def test_no_projects_returns_empty_list(self, workspace):
        # Workspace exists (deployment.db seeded) but contains no projects.
        result = patch_model_configs_across_projects(
            workspace,
            base_experiment_map={"nvidia/cosmos-reason2-2b": "uuid-2b"},
        )
        assert result == []

    def test_updates_only_matching_model_names(self, workspace):
        settings = make_settings(workspace)
        p1 = create_project(name="P1", description=None, settings=settings)
        p2 = create_project(name="P2", description=None, settings=settings)

        # Patch only the 2B row across both projects (8B unaffected).
        patched = patch_model_configs_across_projects(
            workspace,
            base_experiment_map={"nvidia/cosmos-reason2-2b": "uuid-2b-from-tao"},
        )
        per_project = {pdir.name: count for pdir, count in patched}
        assert per_project[p1.project_id] == 1
        assert per_project[p2.project_id] == 1

        for pid in (p1.project_id, p2.project_id):
            engine = open_project_db(workspace / "projects" / pid)
            with Session(engine) as session:
                two_b = (
                    session.query(ModelConfig)
                    .filter_by(project_id=pid, model_name="nvidia/cosmos-reason2-2b")
                    .one()
                )
                assert two_b.tao_base_experiment_id == "uuid-2b-from-tao"
                assert two_b.tao_base_experiment_pull_status == "pull_complete"
                eight_b = (
                    session.query(ModelConfig)
                    .filter_by(project_id=pid, model_name="nvidia/cosmos-reason2-8b")
                    .one()
                )
                # 8B was NOT in the patch map.
                assert eight_b.tao_base_experiment_id is None
                # Default pull-status is null/unknown until a patch fires.
                assert eight_b.tao_base_experiment_pull_status in (None, "unknown")

    def test_idempotent_rerun_writes_same_values(self, workspace):
        settings = make_settings(workspace)
        p = create_project(name="P", description=None, settings=settings)

        base_map = {"nvidia/cosmos-reason2-2b": "uuid-stable"}
        patch_model_configs_across_projects(workspace, base_experiment_map=base_map)
        # Second run: same input, no errors, same final state.
        patch_model_configs_across_projects(workspace, base_experiment_map=base_map)
        engine = open_project_db(workspace / "projects" / p.project_id)
        with Session(engine) as session:
            row = (
                session.query(ModelConfig)
                .filter_by(
                    project_id=p.project_id, model_name="nvidia/cosmos-reason2-2b"
                )
                .one()
            )
            assert row.tao_base_experiment_id == "uuid-stable"
            assert row.tao_base_experiment_pull_status == "pull_complete"

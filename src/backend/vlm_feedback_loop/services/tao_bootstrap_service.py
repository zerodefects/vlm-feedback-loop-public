# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared TAO bootstrap helpers.

The deployment-level TAO workspace + base-experiment provisioning flow
needs the same per-project ``ModelConfig`` patch from two callers:

- ``cli.cmd_tao_bootstrap`` (the admin-managed path: receives
  admin-supplied UUIDs and writes them across every project DB).
- ``services.tao_base_experiment_provisioning_service.provision_base_experiments``
  (the self-service path: writes the UUIDs returned by
  ``find_base_experiment_by_arch`` after ``POST :load_airgapped``).

Housing the helper here rather than in ``cli.py`` lets both call sites
share the same iteration + write logic without one importing CLI
argparse code.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.orm import Session

from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.services.project_service import projects_root


def iter_project_dirs(workspace_root: Path) -> Iterator[Path]:
    """Yield every ``{workspace_root}/projects/<uuid>`` directory.

    A directory is yielded only when ``project.db`` exists inside it,
    so partial / aborted project creations are skipped.
    """
    projects_dir = projects_root(workspace_root)
    if not projects_dir.exists():
        return
    for entry in sorted(projects_dir.iterdir()):
        if entry.is_dir() and (entry / "project.db").exists():
            yield entry


def patch_model_configs_across_projects(
    workspace_root: Path,
    *,
    base_experiment_map: dict[str, str],
) -> list[tuple[Path, int]]:
    """Patch ``tao_base_experiment_id`` + pull status across every project DB.

    For each project DB under ``workspace_root``, find every
    :class:`ModelConfig` row whose ``model_name`` appears as a key in
    ``base_experiment_map`` and set:

    - ``tao_base_experiment_id`` = ``base_experiment_map[model_name]``
    - ``tao_base_experiment_pull_status`` = ``"pull_complete"``

    Returns ``[(project_dir, rows_patched), …]`` so callers can report
    which projects were touched. Idempotent: rerunning with the same
    map is a no-op (rows already at the target value, but the writes
    do execute — keep the helper simple).
    """
    results: list[tuple[Path, int]] = []
    for project_dir in iter_project_dirs(workspace_root):
        engine = open_project_db(project_dir)
        patched = 0
        with Session(engine) as session:
            rows = (
                session.query(ModelConfig)
                .filter(ModelConfig.model_name.in_(list(base_experiment_map)))
                .all()
            )
            for row in rows:
                row.tao_base_experiment_id = base_experiment_map[row.model_name]
                row.tao_base_experiment_pull_status = "pull_complete"
                patched += 1
            session.commit()
        results.append((project_dir, patched))
    return results


def patch_model_pull_status_across_projects(
    workspace_root: Path,
    *,
    model_names: list[str],
    pull_status: str,
    preserve_pull_complete: bool,
) -> list[tuple[Path, int]]:
    """Patch the observable pull lifecycle for matching seeded bases.

    The provisioning run is deployment-scoped, while ModelConfig rows are
    project-scoped. Mirroring ``pulling`` / ``failed`` to every matching row
    lets any project's Student Training screen reconcile after navigation or
    restart. A completed row is never regressed when
    ``preserve_pull_complete`` is true.
    """
    names = list(dict.fromkeys(model_names))
    results: list[tuple[Path, int]] = []
    if not names:
        return results

    for project_dir in iter_project_dirs(workspace_root):
        engine = open_project_db(project_dir)
        patched = 0
        with Session(engine) as session:
            rows = (
                session.query(ModelConfig)
                .filter(ModelConfig.model_name.in_(names))
                .all()
            )
            for row in rows:
                if (
                    preserve_pull_complete
                    and row.tao_base_experiment_id
                    and row.tao_base_experiment_pull_status == "pull_complete"
                ):
                    continue
                row.tao_base_experiment_pull_status = pull_status
                patched += 1
            if patched:
                session.commit()
        results.append((project_dir, patched))
    return results


__all__ = [
    "iter_project_dirs",
    "patch_model_configs_across_projects",
    "patch_model_pull_status_across_projects",
]

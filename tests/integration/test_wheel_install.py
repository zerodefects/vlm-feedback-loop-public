# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-distribution smoke tests for runtime package resources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(command)}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


@pytest.mark.timeout(240)
def test_built_distributions_contain_runnable_tao_pull_resources(
    tmp_path: Path,
) -> None:
    """The sdist-built wheel must run outside the source tree."""
    dist_dir = tmp_path / "dist"
    _run(
        ["uv", "build", "--out-dir", str(dist_dir)],
        cwd=REPO_ROOT,
    )
    source_distributions = list(dist_dir.glob("vlm_feedback_loop-*.tar.gz"))
    wheels = list(dist_dir.glob("vlm_feedback_loop-*.whl"))
    assert len(source_distributions) == 1
    assert len(wheels) == 1

    venv_dir = tmp_path / "venv"
    _run(
        ["uv", "venv", str(venv_dir), "--python", sys.executable],
        cwd=tmp_path,
    )
    venv_python = venv_dir / "bin" / "python"
    _run(
        ["uv", "pip", "install", "--python", str(venv_python), str(wheels[0])],
        cwd=tmp_path,
    )

    workspace = tmp_path / "workspace"
    probe = f"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import vlm_feedback_loop
from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.services import (
    tao_base_experiment_provisioning_run_service as run_service,
)
from vlm_feedback_loop.services import tao_base_experiment_provisioning_service as service

package_root = Path(vlm_feedback_loop.__file__).resolve().parent
assert service.PULL_SCRIPT_PATH.is_file()
assert service.PULL_REQUIREMENTS_PATH.is_file()
assert service.PULL_SCRIPT_PATH.is_relative_to(package_root)
assert service.PULL_REQUIREMENTS_PATH.is_relative_to(package_root)
assert "nvidia-tao-core" in service.PULL_REQUIREMENTS_PATH.read_text()

settings = Settings(
    WORKSPACE_ROOT={str(workspace)!r},
    TAO_WORKSPACE_S3_ACCESS_KEY="access",
    TAO_WORKSPACE_S3_SECRET_KEY="secret",
    HF_TOKEN="hf-test",
)
run_service.read_tao_deployment_config = lambda _settings: SimpleNamespace(
    bootstrap_status="bootstrapped",
    tao_workspace_id="workspace-id",
    tao_workspace_bucket="bucket",
    tao_workspace_s3_endpoint_url_external="https://s3.example.test",
)
run_service.shutil.which = lambda _name: "/usr/bin/uv"
assert run_service._provisioning_prerequisite_error(settings) is None

command = service._build_subprocess_command(
    Path("input.csv"), Path("stage"), use_isolated_helper=True
)
assert str(service.PULL_SCRIPT_PATH) in command
assert str(service.PULL_REQUIREMENTS_PATH) in command

subprocess.run([sys.executable, str(service.PULL_SCRIPT_PATH), "--help"], check=True)
cli = Path(sys.executable).parent / "vlm-feedback-loop"
subprocess.run([str(cli), "tao-pull-base-experiments", "--help"], check=True)
"""
    _run([str(venv_python), "-I", "-c", probe], cwd=tmp_path)

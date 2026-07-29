# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential-boundary tests for the standalone TAO pull driver."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def pull_script() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts" / "pull_base_experiments.py"
    spec = importlib.util.spec_from_file_location(
        "pull_base_experiments_under_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_grandchild_receives_ngc_only_through_environment(
    pull_script, tmp_path, monkeypatch
):
    """The final nvidia-tao-core process cannot expose the key through argv."""
    sentinel = "SENTINEL_FINAL_GRANDCHILD_NGC"
    hf_sentinel = "SENTINEL_FINAL_GRANDCHILD_HF"
    monkeypatch.setenv("UNCHANGED_PARENT_VALUE", "harmless")
    monkeypatch.setenv("HF_TOKEN", hf_sentinel)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    parent_before = dict(pull_script.os.environ)
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(pull_script.subprocess, "run", fake_run)
    result = pull_script._run_pretrained_models(
        shared_folder=tmp_path, ngc_key=sentinel
    )

    assert result.returncode == 0
    command = captured["cmd"]
    assert isinstance(command, list)
    assert "--ngc-key" not in command
    assert not any(sentinel in str(token) for token in command)
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["PTM_API_KEY"] == sentinel
    assert child_env["HF_TOKEN"] == hf_sentinel
    assert child_env["HUGGING_FACE_HUB_TOKEN"] == hf_sentinel
    assert child_env["UNCHANGED_PARENT_VALUE"] == "harmless"
    assert dict(pull_script.os.environ) == parent_before


def test_grandchild_failure_diagnostic_redacts_environment_secrets(
    pull_script, tmp_path, monkeypatch, capsys
):
    """A child that echoes its environment cannot leak through driver stderr."""
    ngc_sentinel = "SENTINEL_FAILED_NGC"
    hf_sentinel = "SENTINEL_FAILED_HF"
    monkeypatch.setenv("PTM_API_KEY", ngc_sentinel)
    monkeypatch.setenv("HF_TOKEN", hf_sentinel)
    csv_path = tmp_path / "models.csv"
    csv_path.write_text(
        "displayName,ngc_path,network_arch,is_backbone\n"
        "Example,hf_model://example,cosmos-rl,True\n"
    )
    monkeypatch.setattr(pull_script, "_swap_in_operator_csv", lambda _path: _path)
    monkeypatch.setattr(
        pull_script,
        "_run_pretrained_models",
        lambda **_kwargs: subprocess.CompletedProcess(
            [],
            9,
            stdout="",
            stderr=f"failed for {ngc_sentinel} and {hf_sentinel}",
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        pull_script.main(
            [
                "--csv",
                str(csv_path),
                "--shared-folder-path",
                str(tmp_path / "stage"),
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "pretrained_models exited rc=9" in captured.err
    assert "[REDACTED]" in captured.err
    assert ngc_sentinel not in captured.out + captured.err
    assert hf_sentinel not in captured.out + captured.err


def test_grandchild_failure_redacts_before_truncating(
    pull_script, tmp_path, monkeypatch, capsys
):
    """A credential crossing the diagnostic limit cannot leak a partial value."""
    sentinel = "SENTINEL_BOUNDARY_CREDENTIAL_" + ("X" * 40)
    padding = "p" * (1024 - (len(sentinel) - 1))
    monkeypatch.setenv("PTM_API_KEY", sentinel)
    csv_path = tmp_path / "models.csv"
    csv_path.write_text(
        "displayName,ngc_path,network_arch,is_backbone\n"
        "Example,hf_model://example,cosmos-rl,True\n"
    )
    monkeypatch.setattr(pull_script, "_swap_in_operator_csv", lambda _path: _path)
    monkeypatch.setattr(
        pull_script,
        "_run_pretrained_models",
        lambda **_kwargs: subprocess.CompletedProcess(
            [],
            9,
            stdout="",
            stderr=padding + sentinel,
        ),
    )

    with pytest.raises(SystemExit):
        pull_script.main(
            [
                "--csv",
                str(csv_path),
                "--shared-folder-path",
                str(tmp_path / "stage"),
            ]
        )

    captured = capsys.readouterr()
    assert "[REDACTED]" in captured.err
    assert sentinel[:-1] not in captured.out + captured.err


def test_grandchild_spawn_failure_redacts_environment_secrets(
    pull_script, tmp_path, monkeypatch, capsys
):
    """A spawn exception cannot bypass the standalone driver's redactor."""
    sentinel = "SENTINEL_FINAL_SPAWN_FAILURE"
    monkeypatch.setenv("PTM_API_KEY", sentinel)
    csv_path = tmp_path / "models.csv"
    csv_path.write_text(
        "displayName,ngc_path,network_arch,is_backbone\n"
        "Example,hf_model://example,cosmos-rl,True\n"
    )
    monkeypatch.setattr(pull_script, "_swap_in_operator_csv", lambda _path: _path)

    def fail_spawn(**_kwargs):
        raise OSError(f"spawn failed around {sentinel}")

    monkeypatch.setattr(pull_script, "_run_pretrained_models", fail_spawn)

    with pytest.raises(SystemExit) as exc_info:
        pull_script.main(
            [
                "--csv",
                str(csv_path),
                "--shared-folder-path",
                str(tmp_path / "stage"),
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "pretrained_models failed to spawn" in captured.err
    assert "[REDACTED]" in captured.err
    assert sentinel not in captured.out + captured.err

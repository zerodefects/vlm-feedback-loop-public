# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the offline public-release structure and link validator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "validate_public_release.py"
)


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_public_release_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _write(root: Path, relative: str, content: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_local_link_check_covers_all_shipping_markdown(tmp_path: Path) -> None:
    """Links are resolved relative to each document, including image links."""

    _write(tmp_path, "README.md", "[Overview](docs/Overview.md)\n")
    _write(
        tmp_path,
        "docs/Overview.md",
        "[deployment](deployment.md)\n![screen](images/screen.png)\n",
    )
    _write(tmp_path, "docs/deployment.md", "Deployment\n")
    _write(tmp_path, "docs/images/screen.png", "not-a-real-png")

    assert validator.find_local_link_issues(tmp_path) == []

    (tmp_path / "docs/deployment.md").unlink()
    issues = validator.find_local_link_issues(tmp_path)
    assert [issue.message for issue in issues] == ["broken local link: deployment.md"]


def test_readme_smoke_contract_pins_anonymous_compose_walkthrough(
    tmp_path: Path,
) -> None:
    """The README check catches drift in acquisition and first-run commands."""

    _write(
        tmp_path,
        "README.md",
        "\n".join(
            (
                validator.PUBLIC_REPOSITORY_URL,
                "docker compose up --build",
                "docker compose down",
                "Open 127.0.0.1",
                "Use bundled sample",
                "Rock, paper, scissors",
                "Use at least 150 images",
            )
        ),
    )
    assert validator.find_readme_issues(tmp_path) == []

    _write(tmp_path, "README.md", "docker compose up --build\n")
    messages = {issue.message for issue in validator.find_readme_issues(tmp_path)}
    assert "public clone URL is missing" in messages
    assert "Compose stop command is missing" in messages

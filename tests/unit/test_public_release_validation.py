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


def _write_migration(
    root: Path,
    sequence: int,
    name: str,
    *,
    revision: str,
    down_revision: str | None,
) -> None:
    _write(
        root,
        f"src/backend/vlm_feedback_loop/migrations/versions/{sequence:03d}_{name}.py",
        f"revision = {revision!r}\ndown_revision = {down_revision!r}\n",
    )


def test_public_migration_check_accepts_linked_post_v1_upgrades(
    tmp_path: Path,
) -> None:
    """A public release may add sequential upgrades after its v1 baseline."""

    _write_migration(
        tmp_path,
        1,
        "public_v1_schema",
        revision="v1_0001",
        down_revision=None,
    )
    _write_migration(
        tmp_path,
        2,
        "student_training_suite_lineage",
        revision="v1_0002",
        down_revision="v1_0001",
    )

    assert validator.find_public_migration_issues(tmp_path) == []


def test_public_migration_check_rejects_a_numbering_gap(tmp_path: Path) -> None:
    """An omitted public upgrade cannot produce a seemingly valid snapshot."""

    _write_migration(
        tmp_path,
        1,
        "public_v1_schema",
        revision="v1_0001",
        down_revision=None,
    )
    _write_migration(
        tmp_path,
        3,
        "later_change",
        revision="v1_0003",
        down_revision="v1_0002",
    )

    messages = [
        issue.message for issue in validator.find_public_migration_issues(tmp_path)
    ]

    assert (
        "public migration filenames must be contiguous from 001; found 001, 003"
        in messages
    )


def test_public_migration_check_rejects_a_broken_revision_link(
    tmp_path: Path,
) -> None:
    """Every shipped upgrade must point to the immediately preceding revision."""

    _write_migration(
        tmp_path,
        1,
        "public_v1_schema",
        revision="v1_0001",
        down_revision=None,
    )
    _write_migration(
        tmp_path,
        2,
        "student_training_suite_lineage",
        revision="v1_0002",
        down_revision=None,
    )

    issues = validator.find_public_migration_issues(tmp_path)

    assert [(issue.path, issue.message) for issue in issues] == [
        (
            "src/backend/vlm_feedback_loop/migrations/versions/002_student_training_suite_lineage.py",
            "expected down_revision 'v1_0001'",
        )
    ]


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


def test_local_link_check_ignores_installed_dependency_markdown(
    tmp_path: Path,
) -> None:
    """Running the validator after uv/pnpm install does not scan dependencies."""

    _write(tmp_path, "README.md", "[Overview](docs/Overview.md)\n")
    _write(tmp_path, "docs/Overview.md", "Overview\n")
    _write(
        tmp_path,
        ".venv/lib/python3.12/site-packages/example/README.md",
        "[package contributor guide](CONTRIBUTING.md)\n",
    )
    _write(
        tmp_path,
        "src/ui/node_modules/example/README.md",
        "[package contributor guide](CONTRIBUTING.md)\n",
    )

    assert validator.find_local_link_issues(tmp_path) == []


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
                "Ingest Images normally opens the bundled `/data/images` sample.",
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


def test_version_check_requires_all_public_declarations_to_agree(
    tmp_path: Path,
) -> None:
    """Release metadata cannot advertise different product versions."""

    _write(tmp_path, "pyproject.toml", '[project]\nversion = "1.0.0"\n')
    _write(
        tmp_path,
        "src/backend/vlm_feedback_loop/__init__.py",
        '__version__ = "1.0.0"\n',
    )
    _write(tmp_path, "src/ui/package.json", '{"version": "1.0.0"}\n')
    _write(tmp_path, "docs/Overview.md", "**Version 1.0.0**\n")
    _write(tmp_path, "sonar-project.properties", "sonar.projectVersion=1.0.0\n")

    assert validator.find_version_issues(tmp_path) == []

    _write(tmp_path, "src/ui/package.json", '{"version": "0.0.0"}\n')
    issues = validator.find_version_issues(tmp_path)
    assert [issue.path for issue in issues] == ["version declarations"]
    assert "UI package=0.0.0" in issues[0].message


def test_source_readability_rejects_minified_first_party_scripts(
    tmp_path: Path,
) -> None:
    """Teaching source stays readable while explicitly vendored code is allowed."""

    _write(tmp_path, "src/ui/src/App.tsx", "export function App() { return null; }\n")
    _write(tmp_path, "src/ui/src/app.min.js", "const x=1;\n")
    _write(
        tmp_path,
        "src/ui/src/components/nebula/dist/vendor.min.js",
        "const bundled='allowed library artifact';\n",
    )

    issues = validator.find_source_readability_issues(tmp_path)
    assert [issue.path for issue in issues] == ["src/ui/src/app.min.js"]


def test_agent_instruction_check_rejects_private_machine_assumptions_and_drift(
    tmp_path: Path,
) -> None:
    """Public agent twins stay self-contained and byte-identical."""

    _write(tmp_path, "AGENTS.md", "Portable Blueprint guidance.\n")
    _write(tmp_path, "CLAUDE.md", "Portable Blueprint guidance.\n")
    assert validator.find_agent_instruction_issues(tmp_path) == []

    _write(
        tmp_path,
        "AGENTS.md",
        "Full sudo on this machine. Use ~/vlm-ui-inspector.\n",
    )
    issues = validator.find_agent_instruction_issues(tmp_path)
    assert {issue.path for issue in issues} == {
        "AGENTS.md",
        "AGENTS.md / CLAUDE.md",
    }

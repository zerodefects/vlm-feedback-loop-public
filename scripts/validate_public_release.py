#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the structure and local documentation links of a public export.

Run this against the directory produced by ``export_public_snapshot.py``:

    uv run python scripts/validate_public_release.py ~/release-vlm-feedback-loop

The check is intentionally offline. Anonymous repository access, secret
scanning, dependency tests, and runtime smoke tests remain explicit release
handoff steps because they require network access or installed services.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

PUBLIC_REPOSITORY_URL = "https://github.com/zerodefects/vlm-feedback-loop-public"
PRIVATE_EVIDENCE_PATH = Path("docs") / "internal"
PRIVATE_SOURCE_HOST = "gitlab-master." + "nvidia.com"
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)\n]+)\)")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
SCRIPT_SUFFIXES = frozenset(
    {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}
)
VENDORED_SCRIPT_PREFIXES = (
    Path("src/ui/assets"),
    Path("src/ui/src/components/nebula/dist"),
)
GENERATED_MARKDOWN_PARTS = frozenset({".git", ".venv", "node_modules"})
PUBLIC_MIGRATION_NAME_RE = re.compile(r"(?P<sequence>\d{3})_[a-z0-9_]+\.py")

REQUIRED_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("README.md"),
    Path("LICENSE"),
    Path("LICENSE-3rd-party.txt"),
    Path("pyproject.toml"),
    Path("sonar-project.properties"),
    Path("docker-compose.yml"),
    Path(".env.example"),
    Path("config.yaml.example"),
    Path("docs/Overview.md"),
    Path("docs/deployment.md"),
    Path("docs/tao-ftms-install.md"),
    Path("deploy/example-images/LICENSE.DATA"),
    Path("scripts/export_public_snapshot.py"),
    Path("src/backend/vlm_feedback_loop/main.py"),
    Path("src/backend/vlm_feedback_loop/__init__.py"),
    Path("src/ui/package.json"),
    Path("src/ui/src/components/nebula/dist/index.d.mts"),
    Path("src/ui/src/components/nebula/dist/index.mjs"),
    Path("src/ui/src/components/nebula/dist/styles.css"),
    Path("src/backend/vlm_feedback_loop/migrations/versions/001_public_v1_schema.py"),
)


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def render(self) -> str:
        return f"{self.path}: {self.message}"


def _markdown_target(raw_target: str) -> str:
    """Return the path portion from a Markdown destination."""

    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        # A non-angle Markdown destination may have a quoted title after it.
        target = target.split(maxsplit=1)[0]
    return unquote(target.split("#", 1)[0])


def find_local_link_issues(root: Path) -> list[ValidationIssue]:
    """Find broken relative file links in every shipping Markdown document."""

    issues: list[ValidationIssue] = []
    for document in sorted(root.rglob("*.md")):
        if not document.is_file():
            continue
        relative_document = document.relative_to(root)
        if any(part in GENERATED_MARKDOWN_PARTS for part in relative_document.parts):
            continue
        text = document.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            target = _markdown_target(match.group(1))
            if (
                not target
                or target.startswith(("#", "/", "mailto:"))
                or "://" in target
            ):
                continue
            resolved = (document.parent / target).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                issues.append(
                    ValidationIssue(
                        relative_document.as_posix(),
                        f"local link escapes the release root: {target}",
                    )
                )
                continue
            if not resolved.exists():
                issues.append(
                    ValidationIssue(
                        relative_document.as_posix(),
                        f"broken local link: {target}",
                    )
                )
    return issues


def find_structure_issues(root: Path) -> list[ValidationIssue]:
    """Validate the release profile's required and prohibited paths."""

    issues: list[ValidationIssue] = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            issues.append(
                ValidationIssue(relative.as_posix(), "required path is missing")
            )

    prohibited = (
        Path(".claude"),
        Path(".codex"),
        PRIVATE_EVIDENCE_PATH,
        Path("docs/fixtures/pinned"),
        Path("scripts/research"),
        Path("src/backend/vlm_feedback_loop/cli_autorun.py"),
    )
    for relative in prohibited:
        if (root / relative).exists():
            issues.append(
                ValidationIssue(relative.as_posix(), "private/optional path is present")
            )

    issues.extend(find_public_migration_issues(root))

    sample_root = root / "deploy/example-images"
    image_count = (
        sum(
            1
            for path in sample_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if sample_root.is_dir()
        else 0
    )
    if image_count != 15:
        issues.append(
            ValidationIssue(
                "deploy/example-images",
                f"expected 15 bundled sample images; found {image_count}",
            )
        )
    return issues


def find_public_migration_issues(root: Path) -> list[ValidationIssue]:
    """Require an inspectable, contiguous public v1 Alembic lineage."""

    relative_root = Path("src/backend/vlm_feedback_loop/migrations/versions")
    migration_root = root / relative_root
    migrations = sorted(migration_root.glob("*.py"))
    issues: list[ValidationIssue] = []
    numbered: list[tuple[int, Path]] = []

    for migration in migrations:
        match = PUBLIC_MIGRATION_NAME_RE.fullmatch(migration.name)
        if match is None:
            issues.append(
                ValidationIssue(
                    (relative_root / migration.name).as_posix(),
                    "public migration filename must use NNN_lowercase_name.py",
                )
            )
            continue
        numbered.append((int(match.group("sequence")), migration))

    sequences = [sequence for sequence, _migration in numbered]
    if sequences != list(range(1, len(numbered) + 1)):
        rendered = ", ".join(f"{sequence:03d}" for sequence in sequences) or "none"
        issues.append(
            ValidationIssue(
                relative_root.as_posix(),
                f"public migration filenames must be contiguous from 001; found {rendered}",
            )
        )

    for sequence, migration in numbered:
        relative = (relative_root / migration.name).as_posix()
        try:
            tree = ast.parse(migration.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError, UnicodeError) as exc:
            issues.append(
                ValidationIssue(relative, f"cannot inspect public migration: {exc}")
            )
            continue

        assignments: dict[str, object] = {}
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "revision",
                    "down_revision",
                }:
                    try:
                        assignments[target.id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        assignments[target.id] = object()

        expected_revision = f"v1_{sequence:04d}"
        expected_parent = None if sequence == 1 else f"v1_{sequence - 1:04d}"
        if assignments.get("revision") != expected_revision:
            issues.append(
                ValidationIssue(
                    relative,
                    f"expected revision {expected_revision!r}",
                )
            )
        if assignments.get("down_revision", object()) != expected_parent:
            issues.append(
                ValidationIssue(
                    relative,
                    f"expected down_revision {expected_parent!r}",
                )
            )

    return issues


def find_readme_issues(root: Path) -> list[ValidationIssue]:
    """Pin the documented first-run contract and public acquisition URL."""

    readme = root / "README.md"
    if not readme.is_file():
        return []
    text = readme.read_text(encoding="utf-8")
    expectations = (
        (PUBLIC_REPOSITORY_URL, "public clone URL is missing"),
        ("docker compose up --build", "primary Compose launch command is missing"),
        ("docker compose down", "Compose stop command is missing"),
        ("127.0.0.1", "loopback-only launch boundary is missing"),
        ("bundled `/data/images` sample", "bundled-sample walkthrough is missing"),
        ("Rock, paper, scissors", "RPS Guidance step is missing"),
        ("150", "150-image recommendation is missing"),
    )
    issues = [
        ValidationIssue("README.md", message)
        for expected, message in expectations
        if expected not in text
    ]
    if PRIVATE_SOURCE_HOST in text:
        issues.append(
            ValidationIssue("README.md", "private source-repository host is present")
        )
    return issues


def find_version_issues(root: Path) -> list[ValidationIssue]:
    """Require every public product-version declaration to agree."""

    declarations: list[tuple[str, Path, str | None]] = []

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            value = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
                "version"
            ]
        except (KeyError, TypeError, tomllib.TOMLDecodeError):
            value = None
        declarations.append(("Python package", pyproject, value))

    backend_package = root / "src/backend/vlm_feedback_loop/__init__.py"
    if backend_package.is_file():
        match = re.search(
            r'^__version__\s*=\s*["\']([^"\']+)["\']',
            backend_package.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        declarations.append(
            ("backend module", backend_package, match.group(1) if match else None)
        )

    ui_package = root / "src/ui/package.json"
    if ui_package.is_file():
        try:
            value = json.loads(ui_package.read_text(encoding="utf-8"))["version"]
        except (json.JSONDecodeError, KeyError, TypeError):
            value = None
        declarations.append(("UI package", ui_package, value))

    overview = root / "docs/Overview.md"
    if overview.is_file():
        match = re.search(
            r"^\*\*Version\s+([^*]+)\*\*$",
            overview.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        declarations.append(
            ("product overview", overview, match.group(1).strip() if match else None)
        )

    sonar_properties = root / "sonar-project.properties"
    if sonar_properties.is_file():
        match = re.search(
            r"^sonar\.projectVersion\s*=\s*(\S+)\s*$",
            sonar_properties.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        declarations.append(
            ("SonarQube project", sonar_properties, match.group(1) if match else None)
        )

    issues: list[ValidationIssue] = []
    for label, path, value in declarations:
        if not isinstance(value, str) or not value.strip():
            issues.append(
                ValidationIssue(
                    path.relative_to(root).as_posix(),
                    f"{label} version declaration is missing or invalid",
                )
            )

    valid_values = {value for _, _, value in declarations if isinstance(value, str)}
    if len(valid_values) > 1:
        rendered = ", ".join(
            f"{label}={value or '<missing>'}" for label, _, value in declarations
        )
        issues.append(
            ValidationIssue(
                "version declarations",
                f"public product versions disagree: {rendered}",
            )
        )
    return issues


def find_source_readability_issues(root: Path) -> list[ValidationIssue]:
    """Reject minified first-party scripts while allowing vendored libraries."""

    source_root = root / "src" / "ui" / "src"
    if not source_root.is_dir():
        return []

    issues: list[ValidationIssue] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix not in SCRIPT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(relative.is_relative_to(prefix) for prefix in VENDORED_SCRIPT_PREFIXES):
            continue

        lowered_name = path.name.lower()
        if ".min." in lowered_name or ".bundle." in lowered_name:
            issues.append(
                ValidationIssue(
                    relative.as_posix(),
                    "minified or bundled first-party script is not publishable source",
                )
            )
            continue

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            issues.append(
                ValidationIssue(
                    relative.as_posix(),
                    "first-party script is not readable UTF-8 source",
                )
            )
            continue
        if any(len(line) > 1_000 for line in lines):
            issues.append(
                ValidationIssue(
                    relative.as_posix(),
                    "first-party script contains a minification-like line over 1000 characters",
                )
            )
    return issues


def find_agent_instruction_issues(root: Path) -> list[ValidationIssue]:
    """Keep public agent guidance portable, self-contained, and in sync."""

    twins = (root / "AGENTS.md", root / "CLAUDE.md")
    existing = [path for path in twins if path.is_file()]
    issues: list[ValidationIssue] = []
    if len(existing) == 2 and twins[0].read_bytes() != twins[1].read_bytes():
        issues.append(
            ValidationIssue(
                "AGENTS.md / CLAUDE.md",
                "public agent-instruction twins differ",
            )
        )

    source_only_pattern = re.compile(
        r"Full sudo on this machine|Unrestricted filesystem access"
        r"|~/vlm-ui-inspector|~/Retail-Agentic-Commerce"
        r"|developer-machine resources|public-export:exclude-source-only"
    )
    for path in existing:
        if source_only_pattern.search(path.read_text(encoding="utf-8")):
            issues.append(
                ValidationIssue(
                    path.relative_to(root).as_posix(),
                    (
                        "contains private-machine authority, dataset, companion-"
                        "checkout instructions, or unprocessed export markers"
                    ),
                )
            )
    return issues


def validate(root: Path) -> list[ValidationIssue]:
    root = root.resolve()
    if not root.is_dir():
        return [ValidationIssue(str(root), "release directory does not exist")]
    return [
        *find_structure_issues(root),
        *find_readme_issues(root),
        *find_version_issues(root),
        *find_source_readability_issues(root),
        *find_agent_instruction_issues(root),
        *find_local_link_issues(root),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_directory", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    issues = validate(args.release_directory)
    if issues:
        print(f"Public release validation failed with {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue.render()}")
        return 1
    print(f"Public release structure and local links passed: {args.release_directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

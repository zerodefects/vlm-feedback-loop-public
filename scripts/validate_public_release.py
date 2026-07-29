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
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

PUBLIC_REPOSITORY_URL = "https://github.com/zerodefects/vlm-feedback-loop-public"
PRIVATE_EVIDENCE_PATH = Path("docs") / "internal"
PRIVATE_SOURCE_HOST = "gitlab-master." + "nvidia.com"
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)\n]+)\)")
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})

REQUIRED_PATHS = (
    Path("README.md"),
    Path("LICENSE"),
    Path("LICENSE-3rd-party.txt"),
    Path("docker-compose.yml"),
    Path(".env.example"),
    Path("config.yaml.example"),
    Path("docs/Overview.md"),
    Path("docs/deployment.md"),
    Path("docs/tao-ftms-install.md"),
    Path("deploy/example-images/LICENSE.DATA"),
    Path("scripts/export_public_snapshot.py"),
    Path("src/backend/vlm_feedback_loop/main.py"),
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
                        document.relative_to(root).as_posix(),
                        f"local link escapes the release root: {target}",
                    )
                )
                continue
            if not resolved.exists():
                issues.append(
                    ValidationIssue(
                        document.relative_to(root).as_posix(),
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

    migrations = sorted(
        path.name
        for path in (root / "src/backend/vlm_feedback_loop/migrations/versions").glob(
            "*.py"
        )
    )
    if migrations != ["001_public_v1_schema.py"]:
        issues.append(
            ValidationIssue(
                "src/backend/vlm_feedback_loop/migrations/versions",
                f"expected only the public v1 baseline; found {migrations}",
            )
        )

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
        ("Use bundled sample", "bundled-sample walkthrough is missing"),
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


def validate(root: Path) -> list[ValidationIssue]:
    root = root.resolve()
    if not root.is_dir():
        return [ValidationIssue(str(root), "release directory does not exist")]
    return [
        *find_structure_issues(root),
        *find_readme_issues(root),
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

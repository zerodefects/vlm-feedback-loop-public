#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a curated public-repository snapshot from a committed Git revision.

The exporter intentionally starts from ``git archive`` rather than copying the
working directory. That keeps local credentials, virtual environments,
``node_modules``, runtime databases, agent transcripts, and other ignored or
untracked state out of the destination.

By default the command:

* requires a clean source checkout;
* verifies every tracked top-level path is classified;
* removes private evidence and private agent configuration;
* excludes the optional AutoRun operator feature and its dedicated tests;
* runs public-readiness checks before changing the destination; and
* refuses to write into a non-empty destination.

Use ``--replace-existing`` only for an existing public Git checkout. It
preserves that checkout's ``.git`` entry and replaces every other top-level
entry with the newly generated snapshot.

Examples:

    # Validate HEAD and show what would be exported.
    uv run python scripts/export_public_snapshot.py /tmp/vlm-public --dry-run

    # Create a new snapshot directory.
    uv run python scripts/export_public_snapshot.py ~/vlm-feedback-loop-public

    # Include the optional AutoRun operator feature.
    uv run python scripts/export_public_snapshot.py \
      ~/vlm-feedback-loop-public --include-autorun

    # Refresh an existing public Git checkout, preserving only its .git data.
    uv run python scripts/export_public_snapshot.py \
      ~/vlm-feedback-loop-public --replace-existing
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
SCRIPT_RELATIVE_PATH: Final = PurePosixPath("scripts/export_public_snapshot.py")
PUBLIC_REPOSITORY_URL: Final = "https://github.com/zerodefects/vlm-feedback-loop-public"

# The public export is allowlisted at the top level. A new root path must be
# classified here deliberately instead of silently leaking into the snapshot.
PUBLIC_TOP_LEVEL: Final = frozenset(
    {
        ".dockerignore",
        ".env.example",
        ".github",
        ".gitleaks.toml",
        ".gitignore",
        ".pre-commit-config.yaml",
        "AGENTS.md",
        "CHANGELOG.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "LICENSE-3rd-party.txt",
        "README.md",
        "SECURITY.md",
        "config.yaml.example",
        "deploy",
        "docker-compose.yml",
        "docs",
        "nginx.conf",
        "pyproject.toml",
        "scripts",
        "sonar-project.properties",
        "src",
        "tests",
        "uv.lock",
    }
)
PRIVATE_TOP_LEVEL: Final = frozenset({".claude", ".codex"})
EXCLUDED_PATHS: Final = (
    PurePosixPath(".claude"),
    PurePosixPath(".codex"),
    PurePosixPath("docs/internal"),
    PurePosixPath("docs/fixtures/pinned"),
    PurePosixPath("scripts/research"),
)
AUTORUN_EXCLUDED_PATHS: Final = (
    PurePosixPath("src/backend/vlm_feedback_loop/cli_autorun.py"),
    PurePosixPath("tests/autorun_support.py"),
    PurePosixPath("tests/unit/test_autorun_ci_mode.py"),
    PurePosixPath("tests/unit/test_autorun_decide.py"),
    PurePosixPath("tests/unit/test_autorun_embedding.py"),
    PurePosixPath("tests/unit/test_autorun_evaluation.py"),
    PurePosixPath("tests/unit/test_autorun_local_nim.py"),
    PurePosixPath("tests/unit/test_autorun_manifest.py"),
    PurePosixPath("tests/unit/test_autorun_rpm_limiter.py"),
    PurePosixPath("tests/unit/test_autorun_selector_loop.py"),
    PurePosixPath("tests/unit/test_autorun_trajectory.py"),
    PurePosixPath("tests/unit/test_autorun_validation_guards.py"),
    PurePosixPath("tests/unit/test_autorun_warmup_ci.py"),
)
AUTORUN_BLOCK_START: Final = "<!-- public-export:exclude-autorun:start -->"
AUTORUN_BLOCK_END: Final = "<!-- public-export:exclude-autorun:end -->"
AUTORUN_CONDITIONAL_TEXT_PATHS: Final = (
    PurePosixPath("AGENTS.md"),
    PurePosixPath("CLAUDE.md"),
    PurePosixPath("README.md"),
    PurePosixPath("docs/AdvancedTests.md"),
    PurePosixPath("docs/Engineering_Spec.md"),
    PurePosixPath("docs/Engineering_Spec_Brief.md"),
    PurePosixPath("docs/Overview.md"),
)


class ExportError(RuntimeError):
    """A user-actionable export failure."""


@dataclass(frozen=True, order=True)
class ReadinessIssue:
    code: str
    path: str
    line: int
    message: str

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.code}] {location}: {self.message}"


@dataclass(frozen=True)
class SnapshotStats:
    file_count: int
    byte_count: int


def _run_git(
    repo_root: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def resolve_commit(repo_root: Path, ref: str) -> str:
    try:
        result = _run_git(repo_root, "rev-parse", "--verify", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise ExportError(f"Git ref does not resolve to a commit: {ref}") from exc
    return result.stdout.strip()


def source_status(repo_root: Path) -> list[str]:
    result = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    return [line for line in result.stdout.splitlines() if line]


def tracked_top_level(repo_root: Path, commit: str) -> set[str]:
    result = _run_git(repo_root, "ls-tree", "--name-only", commit)
    return {line for line in result.stdout.splitlines() if line}


def validate_top_level_classification(repo_root: Path, commit: str) -> None:
    tracked = tracked_top_level(repo_root, commit)
    classified = PUBLIC_TOP_LEVEL | PRIVATE_TOP_LEVEL
    unknown = sorted(tracked - classified)
    if unknown:
        joined = ", ".join(unknown)
        raise ExportError(
            "Unclassified tracked top-level path(s): "
            f"{joined}. Add each path to PUBLIC_TOP_LEVEL or PRIVATE_TOP_LEVEL."
        )


def _archive_commit(repo_root: Path, commit: str, archive_path: Path) -> None:
    with archive_path.open("wb") as stream:
        try:
            subprocess.run(
                ["git", "archive", "--format=tar", commit],
                cwd=repo_root,
                check=True,
                stdout=stream,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise ExportError(f"git archive failed: {detail or exc}") from exc


def _validate_tar_member(destination: Path, member: tarfile.TarInfo) -> None:
    member_path = (destination / member.name).resolve()
    if not member_path.is_relative_to(destination):
        raise ExportError(f"Archive member escapes the snapshot root: {member.name}")
    if member.issym():
        link_path = (member_path.parent / member.linkname).resolve()
        if not link_path.is_relative_to(destination):
            raise ExportError(f"Archive link escapes the snapshot root: {member.name}")
    if member.islnk():
        link_path = (destination / member.linkname).resolve()
        if not link_path.is_relative_to(destination):
            raise ExportError(f"Archive link escapes the snapshot root: {member.name}")


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_tar_member(destination, member)
        archive.extractall(destination, members=members, filter="data")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def remove_private_paths(snapshot_root: Path) -> None:
    for relative in EXCLUDED_PATHS:
        target = snapshot_root.joinpath(*relative.parts)
        if target.exists() or target.is_symlink():
            _remove_path(target)


def remove_autorun_paths(snapshot_root: Path) -> None:
    for relative in AUTORUN_EXCLUDED_PATHS:
        target = snapshot_root.joinpath(*relative.parts)
        if target.exists() or target.is_symlink():
            _remove_path(target)


def process_autorun_documentation(
    snapshot_root: Path,
    *,
    exclude_autorun: bool,
) -> None:
    for relative in AUTORUN_CONDITIONAL_TEXT_PATHS:
        target = snapshot_root.joinpath(*relative.parts)
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8")
        start_count = text.count(AUTORUN_BLOCK_START)
        end_count = text.count(AUTORUN_BLOCK_END)
        if start_count != end_count:
            raise ExportError(
                f"Unbalanced AutoRun export markers in {relative.as_posix()}"
            )
        if exclude_autorun:
            pattern = re.compile(
                rf"\n?{re.escape(AUTORUN_BLOCK_START)}.*?"
                rf"{re.escape(AUTORUN_BLOCK_END)}\n?",
                flags=re.DOTALL,
            )
            text = pattern.sub("\n", text)
        else:
            text = text.replace(f"{AUTORUN_BLOCK_START}\n", "")
            text = text.replace(f"{AUTORUN_BLOCK_END}\n", "")
        target.write_text(text, encoding="utf-8")


def _is_excluded_from_readiness_scan(relative: PurePosixPath) -> bool:
    return relative == SCRIPT_RELATIVE_PATH


def _read_text_files(snapshot_root: Path) -> list[tuple[PurePosixPath, str]]:
    files: list[tuple[PurePosixPath, str]] = []
    for path in sorted(snapshot_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = PurePosixPath(path.relative_to(snapshot_root).as_posix())
        if _is_excluded_from_readiness_scan(relative):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files.append((relative, text))
    return files


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _first_pattern_issue(
    relative: PurePosixPath,
    text: str,
    *,
    code: str,
    pattern: re.Pattern[str],
    message: str,
) -> ReadinessIssue | None:
    match = pattern.search(text)
    if match is None:
        return None
    return ReadinessIssue(
        code=code,
        path=relative.as_posix(),
        line=_line_number(text, match.start()),
        message=message,
    )


def find_readiness_issues(snapshot_root: Path) -> list[ReadinessIssue]:
    """Return public-cut blockers found in an extracted candidate snapshot."""

    issues: list[ReadinessIssue] = []
    for excluded in EXCLUDED_PATHS:
        path = snapshot_root.joinpath(*excluded.parts)
        if path.exists() or path.is_symlink():
            issues.append(
                ReadinessIssue(
                    code="private-path-present",
                    path=excluded.as_posix(),
                    line=0,
                    message="private path must not exist in the public snapshot",
                )
            )

    general_rules = (
        (
            "private-repository-url",
            re.compile(
                r"github\.com/zerodefects/vlm-feedback-loop"
                r"(?:\.git)?(?:[/?#\s]|$)"
                r"|gitlab-master\.nvidia\.com"
            ),
            "replace the private repository URL with the final public URL",
        ),
        (
            "private-evidence-reference",
            re.compile(r"docs/internal/|\]\(internal/"),
            "make the explanation self-contained; excluded evidence will not ship",
        ),
        (
            "private-preserve-branch",
            re.compile(r"\bpreserve/path1-prompt-package\b"),
            "do not promise an internal-only branch to public readers",
        ),
        (
            "personal-git-identity",
            re.compile(r"jclohessy@nvidia\.com|James W\. Clohessy"),
            "remove the private developer Git identity from public instructions",
        ),
    )

    for relative, text in _read_text_files(snapshot_root):
        for code, pattern, message in general_rules:
            issue = _first_pattern_issue(
                relative,
                text,
                code=code,
                pattern=pattern,
                message=message,
            )
            if issue is not None:
                issues.append(issue)

    focused_rules: tuple[tuple[PurePosixPath, str, re.Pattern[str], str], ...] = (
        (
            PurePosixPath("README.md"),
            "readme-merge-artifact",
            re.compile(r"Student\s+The server-side"),
            "repair the TAO timeout paragraph merge artifact",
        ),
        (
            PurePosixPath("docs/changelog.md"),
            "private-development-changelog",
            re.compile(r"A note on internal references"),
            "replace the private development diary with a public release history",
        ),
        (
            PurePosixPath(".gitleaks.toml"),
            "broad-secret-allowlist",
            re.compile(
                r"'''README\\\.md'''|'''docs/changelog\\\.md'''"
                r"|'''docs/TAO_FTMS_INSTALL_GUIDE_v8\\\.md'''"
            ),
            "remove broad public-prose allowlists from the public scanner config",
        ),
        (
            PurePosixPath(".pre-commit-config.yaml"),
            "global-precommit-exclusion",
            re.compile(r"(?m)^exclude:\s*\|"),
            "make formatting exclusions hook-specific so gitleaks scans all text",
        ),
        (
            PurePosixPath(".github/workflows/sonarqube.yml"),
            "sonarqube-test-scope-todo",
            re.compile(r"configure the template to run tests/unit only"),
            "resolve and validate the official-org SonarQube test invocation",
        ),
    )
    for relative, code, pattern, message in focused_rules:
        path = snapshot_root.joinpath(*relative.parts)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        issue = _first_pattern_issue(
            relative,
            text,
            code=code,
            pattern=pattern,
            message=message,
        )
        if issue is not None:
            issues.append(issue)

    tao_topology_pattern = re.compile(
        r"\bnvidian\b|ubuntu@brev|\.brev\.local\b|\bShadeform\b|\bNetbird\b",
        re.IGNORECASE,
    )
    docs_root = snapshot_root / "docs"
    tao_guides = (
        guide
        for guide in docs_root.glob("*.md")
        if "tao" in guide.stem.lower()
        and any(word in guide.stem.lower() for word in ("install", "setup"))
    )
    for guide in sorted(tao_guides):
        relative = PurePosixPath(guide.relative_to(snapshot_root).as_posix())
        issue = _first_pattern_issue(
            relative,
            guide.read_text(encoding="utf-8"),
            code="tao-guide-private-topology",
            pattern=tao_topology_pattern,
            message="replace private organization, host, key, and mesh assumptions",
        )
        if issue is not None:
            issues.append(issue)

    legacy_tao_guide = docs_root / "TAO_FTMS_INSTALL_GUIDE_v8.md"
    if legacy_tao_guide.exists():
        issues.append(
            ReadinessIssue(
                code="versioned-tao-guide-name",
                path="docs/TAO_FTMS_INSTALL_GUIDE_v8.md",
                line=1,
                message="rename the shipping guide to a stable public filename",
            )
        )

    return sorted(set(issues))


def snapshot_stats(snapshot_root: Path) -> SnapshotStats:
    file_count = 0
    byte_count = 0
    for path in snapshot_root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            file_count += 1
            byte_count += path.stat().st_size
    return SnapshotStats(file_count=file_count, byte_count=byte_count)


def _format_size(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return f"{value:.1f} {suffix}"
        value /= 1024.0
    return f"{value:.1f} GiB"


def validate_destination(
    repo_root: Path,
    destination: Path,
    *,
    replace_existing: bool,
) -> Path:
    resolved_repo = repo_root.resolve()
    resolved = destination.expanduser().resolve()
    home = Path.home().resolve()

    if resolved in {Path(resolved.anchor), home, resolved_repo}:
        raise ExportError(f"Refusing unsafe destination: {resolved}")
    if resolved.is_relative_to(resolved_repo):
        raise ExportError("Destination must be outside the private source repository")
    if resolved.exists() and resolved.is_symlink():
        raise ExportError("Destination must not be a symlink")
    if resolved.exists() and not resolved.is_dir():
        raise ExportError("Destination exists and is not a directory")

    if resolved.exists():
        entries = list(resolved.iterdir())
        if entries and not replace_existing:
            raise ExportError(
                "Destination is not empty. Use --replace-existing only for the "
                "intended public Git checkout."
            )
        if entries and replace_existing and not (resolved / ".git").exists():
            raise ExportError(
                "--replace-existing requires an existing .git entry as a "
                "destination safety marker"
            )
    return resolved


def _copy_snapshot_contents(snapshot_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in snapshot_root.iterdir():
        target = destination / source.name
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        elif source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        else:
            shutil.copy2(source, target)


def install_snapshot(
    snapshot_root: Path,
    destination: Path,
    *,
    replace_existing: bool,
) -> None:
    """Install a prepared snapshot, preserving only an existing ``.git``."""

    if not destination.exists() or not any(destination.iterdir()):
        destination_existed = destination.exists()
        try:
            _copy_snapshot_contents(snapshot_root, destination)
        except Exception:
            if destination.exists():
                for current in list(destination.iterdir()):
                    _remove_path(current)
                if not destination_existed:
                    destination.rmdir()
            raise
        return
    if not replace_existing:
        raise ExportError("Refusing to replace a non-empty destination")

    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.public-export-backup-",
            dir=destination.parent,
        )
    )
    moved: list[Path] = []
    old_worktree_fully_moved = False
    try:
        for current in list(destination.iterdir()):
            if current.name == ".git":
                continue
            target = backup / current.name
            current.rename(target)
            moved.append(target)
        old_worktree_fully_moved = True
        _copy_snapshot_contents(snapshot_root, destination)
    except Exception as install_error:
        try:
            if old_worktree_fully_moved:
                for current in list(destination.iterdir()):
                    if current.name != ".git":
                        _remove_path(current)
            for saved in moved:
                saved.rename(destination / saved.name)
        except Exception as rollback_error:
            raise ExportError(
                "Snapshot installation and rollback both failed. Existing files "
                f"remain recoverable in {backup}: {rollback_error}"
            ) from install_error
        else:
            shutil.rmtree(backup)
        raise
    else:
        shutil.rmtree(backup)


def build_snapshot(
    repo_root: Path,
    commit: str,
    working_dir: Path,
    *,
    exclude_autorun: bool = True,
) -> Path:
    validate_top_level_classification(repo_root, commit)
    archive_path = working_dir / "source.tar"
    snapshot_root = working_dir / "snapshot"
    _archive_commit(repo_root, commit, archive_path)
    _extract_archive(archive_path, snapshot_root)
    remove_private_paths(snapshot_root)
    if exclude_autorun:
        remove_autorun_paths(snapshot_root)
    process_autorun_documentation(
        snapshot_root,
        exclude_autorun=exclude_autorun,
    )
    return snapshot_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Export a curated public snapshot from a committed Git revision.")
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="New directory, or an existing public Git checkout with --replace-existing.",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Committed Git ref to export (default: HEAD).",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=(
            "Replace every destination top-level entry except .git. "
            "Requires an existing .git safety marker."
        ),
    )
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Allow a dirty private checkout. Dirty/untracked changes are still "
            "not exported; only --ref is archived."
        ),
    )
    parser.add_argument(
        "--allow-unready",
        action="store_true",
        help="Write a candidate snapshot even when public-readiness checks fail.",
    )
    autorun_group = parser.add_mutually_exclusive_group()
    autorun_group.add_argument(
        "--exclude-autorun",
        dest="exclude_autorun",
        action="store_true",
        default=True,
        help=("Exclude AutoRun and its dedicated test support (default)."),
    )
    autorun_group.add_argument(
        "--include-autorun",
        dest="exclude_autorun",
        action="store_false",
        help="Include the optional AutoRun operator feature and dedicated tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the candidate without changing the destination.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = validate_destination(
            REPO_ROOT,
            args.destination,
            replace_existing=args.replace_existing,
        )
        commit = resolve_commit(REPO_ROOT, args.ref)
        dirty = source_status(REPO_ROOT)
        if dirty and not args.allow_dirty_source:
            preview = "\n".join(f"  {line}" for line in dirty[:20])
            remainder = len(dirty) - 20
            suffix = f"\n  ... and {remainder} more" if remainder > 0 else ""
            raise ExportError(
                "Source checkout is dirty. Commit or remove the changes first, "
                "or pass --allow-dirty-source to export only the committed ref:\n"
                f"{preview}{suffix}"
            )

        with tempfile.TemporaryDirectory(prefix="vlm-public-export-") as tmp:
            working_dir = Path(tmp)
            snapshot_root = build_snapshot(
                REPO_ROOT,
                commit,
                working_dir,
                exclude_autorun=args.exclude_autorun,
            )
            issues = find_readiness_issues(snapshot_root)
            stats = snapshot_stats(snapshot_root)

            print(f"Source commit: {commit}")
            print(
                f"Candidate: {stats.file_count} files, {_format_size(stats.byte_count)}"
            )
            excluded = ", ".join(f"{path.as_posix()}/**" for path in EXCLUDED_PATHS)
            print(f"Excluded: {excluded}")
            print(
                "AutoRun: "
                + ("excluded (default)" if args.exclude_autorun else "included")
            )

            if issues:
                print(
                    f"Public-readiness issues ({len(issues)}):",
                    file=sys.stderr,
                )
                for issue in issues:
                    print(f"  {issue.render()}", file=sys.stderr)
                if not args.allow_unready:
                    raise ExportError(
                        "Public-readiness checks failed; destination was not changed. "
                        "Fix the findings or use --allow-unready for a non-publishable "
                        "candidate."
                    )

            if args.dry_run:
                print(f"Dry run: destination not changed ({destination})")
                return 0

            install_snapshot(
                snapshot_root,
                destination,
                replace_existing=args.replace_existing,
            )
            print(f"Exported public snapshot to: {destination}")
            if issues:
                print("WARNING: candidate was exported with readiness issues.")
            return 0
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        print(f"error: Git command failed: {detail or exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

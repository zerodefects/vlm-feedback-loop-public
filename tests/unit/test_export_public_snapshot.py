# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the curated public-snapshot exporter."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "export_public_snapshot.py"
)


def _load_exporter() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "export_public_snapshot_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = _load_exporter()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write(repo: Path, relative: str, content: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _committed_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "private"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Snapshot Test")
    _git(repo, "config", "user.email", "snapshot@example.com")
    for relative, content in files.items():
        _write(repo, relative, content)
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo


def test_build_snapshot_uses_committed_files_and_removes_private_paths(
    tmp_path: Path,
) -> None:
    """The candidate contains the product, never private acceptance or local state."""

    internal_report = "docs" + "/internal/report.md"
    repo = _committed_repo(
        tmp_path,
        {
            "README.md": (
                "Public readme\n"
                f"{exporter.AUTORUN_BLOCK_START}\n"
                "AutoRun docs\n"
                f"{exporter.AUTORUN_BLOCK_END}\n"
            ),
            "docs/API.md": "API\n",
            "docs/live-release-acceptance.md": "private run ledger\n",
            internal_report: "private evidence\n",
            "docs/fixtures/pinned/study.json": "{}\n",
            ".github/workflows/sonarqube.yml": (
                "jobs:\n"
                "  sonarqube:\n"
                "    uses: NVIDIA-AI-Blueprints/sonarqube-workflows/"
                ".github/workflows/sonarqube-reusable-template.yml@main\n"
            ),
            ".github/workflows/ci.yml": "jobs:\n  test:\n    script: pytest\n",
            "scripts/prepare_training_clone.py": "acceptance = True\n",
            "scripts/research/experiment.py": "research = True\n",
            "tests/unit/test_prepare_training_clone.py": (
                "def test_acceptance_helper(): pass\n"
            ),
            ".claude/settings.json": "{}\n",
            ".codex/config.toml": "model = 'private'\n",
        },
    )
    _write(repo, ".env", "TOKEN=secret\n")
    _write(repo, "local-notes.txt", "not tracked\n")

    working_dir = tmp_path / "work"
    working_dir.mkdir()
    commit = exporter.resolve_commit(repo, "HEAD")
    snapshot = exporter.build_snapshot(repo, commit, working_dir)

    assert (snapshot / "README.md").read_text(encoding="utf-8") == "Public readme\n"
    assert (snapshot / "docs/API.md").is_file()
    assert not (snapshot / "docs/live-release-acceptance.md").exists()
    assert not (snapshot / "docs" / "internal").exists()
    assert not (snapshot / "docs/fixtures/pinned").exists()
    assert not (snapshot / ".github/workflows/ci.yml").exists()
    assert not (snapshot / ".github/workflows/sonarqube.yml").exists()
    assert not (snapshot / "scripts/prepare_training_clone.py").exists()
    assert not (snapshot / "scripts/research").exists()
    assert not (snapshot / "tests/unit/test_prepare_training_clone.py").exists()
    assert not (snapshot / ".claude").exists()
    assert not (snapshot / ".codex").exists()
    assert not (snapshot / ".env").exists()
    assert not (snapshot / "local-notes.txt").exists()


def test_build_snapshot_rewrites_source_clone_url_for_public_readers(
    tmp_path: Path,
) -> None:
    """The source README uses NVRetail while its export stays anonymous-cloneable."""

    repo = _committed_repo(
        tmp_path,
        {
            "README.md": (
                f"git clone {exporter.SOURCE_REPOSITORY_URL}\n"
                f"Source: {exporter.SOURCE_REPOSITORY_URL}\n"
            )
        },
    )
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    snapshot = exporter.build_snapshot(
        repo,
        exporter.resolve_commit(repo, "HEAD"),
        working_dir,
    )
    readme = (snapshot / "README.md").read_text(encoding="utf-8")

    assert exporter.SOURCE_REPOSITORY_URL not in readme
    assert readme.count(f"{exporter.PUBLIC_REPOSITORY_URL}.git") == 2
    assert not any(
        issue.code == "private-repository-url"
        for issue in exporter.find_readiness_issues(snapshot)
    )


def test_build_snapshot_excludes_autorun_by_default(tmp_path: Path) -> None:
    """The default public profile removes AutoRun and every dedicated test."""

    repo = _committed_repo(
        tmp_path,
        {
            "README.md": "Public readme\n",
            "src/backend/vlm_feedback_loop/cli.py": "core = True\n",
            "src/backend/vlm_feedback_loop/cli_autorun.py": "optional = True\n",
            "tests/unit/test_autorun_decide.py": "def test_optional(): pass\n",
            "tests/unit/test_embedding_status.py": "def test_core(): pass\n",
        },
    )
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    snapshot = exporter.build_snapshot(
        repo,
        exporter.resolve_commit(repo, "HEAD"),
        working_dir,
    )

    assert (snapshot / "src/backend/vlm_feedback_loop/cli.py").is_file()
    assert not (snapshot / "src/backend/vlm_feedback_loop/cli_autorun.py").exists()
    assert not (snapshot / "tests/unit/test_autorun_decide.py").exists()
    assert (snapshot / "tests/unit/test_embedding_status.py").is_file()
    assert "AutoRun docs" not in (snapshot / "README.md").read_text(encoding="utf-8")
    assert exporter.AUTORUN_BLOCK_START not in (snapshot / "README.md").read_text(
        encoding="utf-8"
    )


def test_build_snapshot_can_include_autorun(tmp_path: Path) -> None:
    """The explicit opt-in preserves the optional AutoRun surface."""

    repo = _committed_repo(
        tmp_path,
        {
            "README.md": (
                "Public readme\n"
                f"{exporter.AUTORUN_BLOCK_START}\n"
                "AutoRun docs\n"
                f"{exporter.AUTORUN_BLOCK_END}\n"
            ),
            "src/backend/vlm_feedback_loop/cli_autorun.py": "optional = True\n",
            "tests/unit/test_autorun_decide.py": "def test_optional(): pass\n",
        },
    )
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    snapshot = exporter.build_snapshot(
        repo,
        exporter.resolve_commit(repo, "HEAD"),
        working_dir,
        exclude_autorun=False,
    )

    assert (snapshot / "src/backend/vlm_feedback_loop/cli_autorun.py").is_file()
    assert (snapshot / "tests/unit/test_autorun_decide.py").is_file()
    readme = (snapshot / "README.md").read_text(encoding="utf-8")
    assert "AutoRun docs" in readme
    assert exporter.AUTORUN_BLOCK_START not in readme


def test_build_snapshot_removes_private_machine_agent_instructions(
    tmp_path: Path,
) -> None:
    """Public agent twins retain product guidance without source-host assumptions."""

    agent_text = (
        "# Agent instructions\n"
        f"{exporter.SOURCE_ONLY_BLOCK_START}\n"
        "Full sudo on this machine. Use ~/vlm-ui-inspector.\n"
        f"{exporter.SOURCE_ONLY_BLOCK_END}\n"
        "Run ./scripts/ci-local.sh before committing.\n"
    )
    repo = _committed_repo(
        tmp_path,
        {
            "AGENTS.md": agent_text,
            "CLAUDE.md": agent_text,
            "README.md": "Public readme\n",
        },
    )
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    snapshot = exporter.build_snapshot(
        repo,
        exporter.resolve_commit(repo, "HEAD"),
        working_dir,
    )

    public_agents = (snapshot / "AGENTS.md").read_text(encoding="utf-8")
    assert public_agents == (snapshot / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Full sudo" not in public_agents
    assert "vlm-ui-inspector" not in public_agents
    assert exporter.SOURCE_ONLY_BLOCK_START not in public_agents
    assert "Run ./scripts/ci-local.sh before committing." in public_agents


def test_unbalanced_source_only_markers_stop_export(tmp_path: Path) -> None:
    """A malformed source-only block cannot silently leak private assumptions."""

    repo = _committed_repo(
        tmp_path,
        {
            "AGENTS.md": (
                f"{exporter.SOURCE_ONLY_BLOCK_START}\nunterminated private docs\n"
            ),
            "README.md": "Public readme\n",
        },
    )
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    with pytest.raises(exporter.ExportError, match="Unbalanced source-only"):
        exporter.build_snapshot(
            repo,
            exporter.resolve_commit(repo, "HEAD"),
            working_dir,
        )


def test_unbalanced_autorun_documentation_markers_stop_export(
    tmp_path: Path,
) -> None:
    """A malformed conditional block cannot silently leak or erase docs."""

    repo = _committed_repo(
        tmp_path,
        {
            "README.md": (
                "Public readme\n"
                f"{exporter.AUTORUN_BLOCK_START}\n"
                "unterminated optional docs\n"
            )
        },
    )
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    with pytest.raises(exporter.ExportError, match="Unbalanced AutoRun"):
        exporter.build_snapshot(
            repo,
            exporter.resolve_commit(repo, "HEAD"),
            working_dir,
        )


def test_parser_excludes_autorun_by_default_and_supports_opt_in() -> None:
    """The CLI default is exclusion and --include-autorun is its inverse."""

    assert exporter._parser().parse_args(["/tmp/public"]).exclude_autorun is True
    assert (
        exporter._parser()
        .parse_args(["/tmp/public", "--exclude-autorun"])
        .exclude_autorun
        is True
    )
    assert (
        exporter._parser()
        .parse_args(["/tmp/public", "--include-autorun"])
        .exclude_autorun
        is False
    )


def test_unknown_top_level_path_stops_the_export(tmp_path: Path) -> None:
    """A newly tracked root path must receive an explicit public/private decision."""

    repo = _committed_repo(
        tmp_path,
        {
            "README.md": "Public readme\n",
            "unclassified-area/file.txt": "surprise\n",
        },
    )
    commit = exporter.resolve_commit(repo, "HEAD")

    with pytest.raises(exporter.ExportError, match="unclassified-area"):
        exporter.validate_top_level_classification(repo, commit)


def test_dirty_source_requires_explicit_committed_only_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dirty files are rejected by default and excluded even with the override."""

    repo = _committed_repo(tmp_path, {"README.md": "Public readme\n"})
    _write(repo, "local-notes.txt", "not committed\n")
    monkeypatch.setattr(exporter, "REPO_ROOT", repo)

    rejected_destination = tmp_path / "rejected"
    assert exporter.main([str(rejected_destination)]) == 2
    assert not rejected_destination.exists()
    assert "Source checkout is dirty" in capsys.readouterr().err

    destination = tmp_path / "public"
    assert (
        exporter.main(
            [
                str(destination),
                "--allow-dirty-source",
            ]
        )
        == 0
    )
    assert (destination / "README.md").is_file()
    assert not (destination / "local-notes.txt").exists()


def test_readiness_failure_does_not_change_destination_without_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication blockers prevent writes unless a candidate is requested."""

    repo = _committed_repo(
        tmp_path,
        {
            "README.md": (
                "Source: https://github.com/" + "zero" + "defects/vlm-feedback-loop\n"
            )
        },
    )
    monkeypatch.setattr(exporter, "REPO_ROOT", repo)
    destination = tmp_path / "public"

    assert exporter.main([str(destination)]) == 2
    assert not destination.exists()

    assert exporter.main([str(destination), "--allow-unready"]) == 0
    assert (destination / "README.md").is_file()


def test_public_mirror_name_does_not_match_private_repository_prefix(
    tmp_path: Path,
) -> None:
    """The `-public` mirror must not be rejected as the old private repo."""

    snapshot = tmp_path / "snapshot"
    _write(
        snapshot,
        "README.md",
        f"Source: {exporter.PUBLIC_REPOSITORY_URL}.git\n",
    )

    assert not any(
        issue.code == "private-repository-url"
        for issue in exporter.find_readiness_issues(snapshot)
    )


def test_vendored_nebula_distribution_is_stageable_in_a_fresh_public_repo(
    tmp_path: Path,
) -> None:
    """The generic dist ignore must not drop the UI's vendored dependency."""

    public_repo = tmp_path / "public"
    public_repo.mkdir()
    _git(public_repo, "init", "--quiet")
    _write(
        public_repo,
        ".gitignore",
        (exporter.REPO_ROOT / ".gitignore").read_text(encoding="utf-8"),
    )
    for relative in (
        "src/ui/src/components/nebula/dist/index.d.mts",
        "src/ui/src/components/nebula/dist/index.mjs",
        "src/ui/src/components/nebula/dist/styles.css",
    ):
        _write(public_repo, relative, "vendored distribution fixture\n")
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", relative],
            cwd=public_repo,
            check=False,
        )
        assert result.returncode == 1, f"{relative} is still ignored"


def test_readiness_checks_any_shipping_tao_guide_name(tmp_path: Path) -> None:
    """Renaming the TAO guide cannot bypass checks for private topology."""

    snapshot = tmp_path / "snapshot"
    _write(
        snapshot,
        "docs/TAO_FTMS_INSTALL_GUIDE.md",
        "Connect with Netbird before using the service.\n",
    )

    issues = exporter.find_readiness_issues(snapshot)

    assert any(issue.code == "tao-guide-private-topology" for issue in issues)
    assert not any(issue.code == "versioned-tao-guide-name" for issue in issues)

    stable = tmp_path / "stable"
    _write(
        stable,
        "docs/tao-ftms-install.md",
        "Connect with Netbird before using the service.\n",
    )
    assert any(
        issue.code == "tao-guide-private-topology"
        for issue in exporter.find_readiness_issues(stable)
    )


def test_readiness_scans_public_migrations(tmp_path: Path) -> None:
    """The public baseline cannot hide private references from release checks."""

    snapshot = tmp_path / "snapshot"
    _write(
        snapshot,
        "src/backend/vlm_feedback_loop/migrations/versions/001_public_v1_schema.py",
        "# See docs/" + "internal/private-schema-notes.md\n",
    )

    issues = exporter.find_readiness_issues(snapshot)

    assert any(issue.code == "private-evidence-reference" for issue in issues)


def test_replace_existing_preserves_git_and_removes_stale_files(
    tmp_path: Path,
) -> None:
    """Refreshing a public checkout preserves history and replaces its worktree."""

    snapshot = tmp_path / "snapshot"
    _write(snapshot, "README.md", "new snapshot\n")
    _write(snapshot, "src/app.py", "new = True\n")

    destination = tmp_path / "public"
    _write(destination, ".git/HEAD", "ref: refs/heads/main\n")
    _write(destination, "README.md", "old snapshot\n")
    _write(destination, "stale.txt", "remove me\n")

    exporter.install_snapshot(snapshot, destination, replace_existing=True)

    assert (destination / ".git/HEAD").read_text(encoding="utf-8") == (
        "ref: refs/heads/main\n"
    )
    assert (destination / "README.md").read_text(encoding="utf-8") == ("new snapshot\n")
    assert (destination / "src/app.py").is_file()
    assert not (destination / "stale.txt").exists()


def test_failed_new_export_removes_partial_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy failure cannot leave a partial new public snapshot behind."""

    snapshot = tmp_path / "snapshot"
    _write(snapshot, "README.md", "new snapshot\n")
    destination = tmp_path / "public"

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated copy failure")

    monkeypatch.setattr(exporter.shutil, "copy2", fail_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        exporter.install_snapshot(snapshot, destination, replace_existing=False)

    assert not destination.exists()


def test_failed_replace_restores_existing_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed refresh rolls the public checkout back to its previous files."""

    snapshot = tmp_path / "snapshot"
    _write(snapshot, "README.md", "new snapshot\n")

    destination = tmp_path / "public"
    _write(destination, ".git/HEAD", "ref: refs/heads/main\n")
    _write(destination, "README.md", "old snapshot\n")
    _write(destination, "keep.txt", "keep me\n")

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated copy failure")

    monkeypatch.setattr(exporter.shutil, "copy2", fail_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        exporter.install_snapshot(snapshot, destination, replace_existing=True)

    assert (destination / "README.md").read_text(encoding="utf-8") == ("old snapshot\n")
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep me\n"
    assert (destination / ".git/HEAD").is_file()
    assert not list(tmp_path.glob(".public.public-export-backup-*"))


def test_failed_move_does_not_delete_unmoved_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure while backing up the old worktree leaves every old file intact."""

    snapshot = tmp_path / "snapshot"
    _write(snapshot, "README.md", "new snapshot\n")

    destination = tmp_path / "public"
    _write(destination, ".git/HEAD", "ref: refs/heads/main\n")
    _write(destination, "README.md", "old snapshot\n")
    _write(destination, "keep.txt", "keep me\n")

    real_rename = exporter.Path.rename

    def fail_keep_move(path: Path, target: Path) -> Path:
        if path == destination / "keep.txt":
            raise OSError("simulated move failure")
        return real_rename(path, target)

    monkeypatch.setattr(exporter.Path, "rename", fail_keep_move)

    with pytest.raises(OSError, match="simulated move failure"):
        exporter.install_snapshot(snapshot, destination, replace_existing=True)

    assert (destination / "README.md").read_text(encoding="utf-8") == ("old snapshot\n")
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep me\n"
    assert (destination / ".git/HEAD").is_file()

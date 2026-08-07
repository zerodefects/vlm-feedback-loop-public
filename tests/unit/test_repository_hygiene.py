# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository-wide publication hygiene checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RETIRED_REPOSITORY_TERMS = ("git" + "hub", "zero" + "defects")


def test_repository_has_no_retired_host_or_account_references() -> None:
    """Current source paths and readable files omit retired publication terms."""

    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    offenders: list[str] = []
    for raw_relative in result.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = raw_relative.decode("utf-8")
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink():
            continue
        if any(term in relative.lower() for term in RETIRED_REPOSITORY_TERMS):
            offenders.append(f"{relative}: path")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(term in text.lower() for term in RETIRED_REPOSITORY_TERMS):
            offenders.append(f"{relative}: text")

    assert offenders == []

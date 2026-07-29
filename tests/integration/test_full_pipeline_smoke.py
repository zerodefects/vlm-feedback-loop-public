# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test for the final integration checkpoint.

Drives ``scripts/full_pipeline_smoke.run_full_pipeline_smoke`` end-to-end
against the real FastAPI backend (started by the session-scoped
``backend_server`` fixture in ``conftest.py``). The test is **live-NIM
only** — it skips when ``NVIDIA_API_KEY`` is not configured because the
labeling step issues real ``/proposals`` calls to hosted Mistral Large 3.

The test that runs here is the same script the closing smoke calls into
as a library, so a pass here is direct evidence the
final-integration-checkpoint code path is healthy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from full_pipeline_smoke import run_full_pipeline_smoke  # noqa: E402

# RPS canonical dataset (372 PNGs, dir-per-class at ~/rps-test-set on every
# dev/GPU box — see the CLAUDE.md dataset table) plus the repo's bundled
# 15-image sample as fallback. ``Path.home()`` keeps this host-agnostic:
# the team moves between boxes with different usernames, so hardcoded
# ``/home/<user>/...`` roots silently skip everywhere but the box they
# were written on. A root can exist but be unreadable when tests run as a
# different user than the directory owner; such roots are skipped — and
# the test then skips with a clear message — rather than failing.
_RPS_ROOT_CANDIDATES = (
    Path.home() / "rps-test-set",
    REPO_ROOT / "deploy" / "example-images",
)


def _resolve_image_paths(min_count: int) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for root in _RPS_ROOT_CANDIDATES:
        try:
            if not root.exists():
                continue
            entries = list(root.iterdir())
        except (PermissionError, OSError):
            # Parent dir permission-restricted (typical for /home/ubuntu/...
            # when the test runs as another user).
            continue
        # Two layouts supported: flat files OR class-named subdirs (rock/, paper/, scissors/).
        flat = sorted(
            p for p in entries if p.is_file() and p.suffix.lower() in suffixes
        )
        if len(flat) >= min_count:
            return flat[:min_count]
        nested: list[Path] = []
        for cls in ("rock", "paper", "scissors"):
            sub = root / cls
            if not sub.is_dir():
                continue
            try:
                nested.extend(
                    sorted(
                        p
                        for p in sub.iterdir()
                        if p.is_file() and p.suffix.lower() in suffixes
                    )[: min_count // 3 + 1]
                )
            except PermissionError:
                continue
        if len(nested) >= min_count:
            return nested[:min_count]
    return []


@pytest.fixture(scope="session")
def hosted_nim_available() -> bool:
    return bool(os.environ.get("NVIDIA_API_KEY"))


@pytest.mark.asyncio
async def test_full_pipeline_smoke_against_live_backend(
    backend_server, hosted_nim_available, tmp_path
):
    """Full-pipeline smoke passes end-to-end against the live backend."""
    if not hosted_nim_available:
        pytest.skip("NVIDIA_API_KEY not set; full-pipeline smoke is hosted-NIM only")
    images = _resolve_image_paths(min_count=10)
    if not images:
        pytest.skip(
            "no usable image directory found for the smoke "
            f"(tried {[str(p) for p in _RPS_ROOT_CANDIDATES]})"
        )

    evidence_dir = tmp_path / "evidence"
    result = await run_full_pipeline_smoke(
        backend_url="http://127.0.0.1:8000",
        image_paths=images,
        label_count=10,
        export_field_mode="all",
        keep_project=False,
        evidence_dir=evidence_dir,
        project_name="full-pipeline-smoke-pytest",
    )

    # Acceptance file should always be written.
    out = evidence_dir / "full_pipeline_smoke_acceptance.json"
    assert out.exists(), "acceptance evidence file not written"

    # Critical steps (every one except batch_label, which can legitimately
    # be gate-blocked at small scale) must all pass.
    by_name = {s.name: s for s in result.steps}
    for required in (
        "create_project",
        "verify_teacher_endpoint",
        "ingest",
        "save_guidance",
        "label_loop",
        "evaluate",
        "check_gate",
        "dataset_export",
        "validate_cosmos_rl_format",
    ):
        assert required in by_name, f"missing pipeline step: {required}"
        assert by_name[required].ok, (
            f"step {required!r} failed: {by_name[required].detail}"
        )

    # Cosmos-RL format MUST validate — this is the training wire-format
    # contract gate.
    assert result.cosmos_rl_format_validated is True
    assert result.label_count == 10
    assert result.dataset_example_count > 0
    assert result.overall_ok is True


@pytest.mark.asyncio
async def test_full_pipeline_smoke_skips_cleanly_without_key(
    backend_server, hosted_nim_available, tmp_path
):
    """Negative path: no NVIDIA_API_KEY → script aborts at proposal step.

    Confirms the script's failure behavior is a graceful per-step report,
    not a Python traceback. When NVIDIA_API_KEY *is* set the live success
    path covers this, so the test is gated on the absence of the key.
    """
    if hosted_nim_available:
        pytest.skip("hosted NIM available; live success path covers this")
    images = _resolve_image_paths(min_count=10)
    if not images:
        pytest.skip("no usable image directory")
    evidence_dir = tmp_path / "evidence"
    result = await run_full_pipeline_smoke(
        backend_url="http://127.0.0.1:8000",
        image_paths=images,
        label_count=2,  # short — fail fast
        export_field_mode="all",
        keep_project=False,
        evidence_dir=evidence_dir,
        project_name="full-pipeline-smoke-no-key",
    )
    # Without an API key, the proposal step fails — but the script must
    # complete cleanly, write the evidence file, and return overall_ok=False
    # (not raise).
    assert (evidence_dir / "full_pipeline_smoke_acceptance.json").exists()
    assert result.overall_ok is False
    by_name = {s.name: s for s in result.steps}
    # Earlier steps still passed; failure must be at label_loop (or earlier
    # configuration step that depends on hosted NIM reachability).
    assert by_name.get("create_project") and by_name["create_project"].ok
    assert by_name.get("save_guidance") and by_name["save_guidance"].ok

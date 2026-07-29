# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live TAO fixture checks: idempotent re-capture and live-vs-committed drift.

Validates the committed ``tests/fixtures/tao/*.json`` files match what
live TAO FTMS returns. Running the capture script a second time MUST
produce byte-identical output.

These tests skip unless ``RUN_LIVE_TAO_FIXTURE=1`` AND a reachable TAO
endpoint is configured in the operator's ``~/.vlm_feedback_loop/.env``.
This keeps CI green while still allowing operators to run the full
fixture check on Profile D when the TAO tunnel is up.

The non-live shape invariants of the committed fixtures (which endpoints
and fields the Blueprint consumes) are CI-gated in
``tests/unit/test_tao_wire_fixtures.py``.

Per-sample-predictions capture from an actual completed evaluate job
is not yet covered — it requires a TAO deployment with a completed
training chain. The capture script (:mod:`scripts.capture_tao_fixtures`)
is ready to pick up that fixture when a real evaluate job exists; the
ambient fixture set captured here pins the wire-format contract the
TAO re-scoring code is built against.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "tao"
_CAPTURE_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "capture_tao_fixtures.py"
)


def _skip_unless_live() -> None:
    if os.environ.get("RUN_LIVE_TAO_FIXTURE") != "1":
        pytest.skip(
            "Live TAO fixture check disabled. "
            "Set RUN_LIVE_TAO_FIXTURE=1 + configure TAO tunnel to enable."
        )


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Idempotency — re-running the capture produces identical bytes
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_capture_script_is_idempotent(tmp_path):
    """Running the capture script twice yields byte-identical fixtures.

    The core invariant: captures are deterministic, so fixture drift
    always reflects a real wire-format change on TAO rather than
    capture noise.
    """
    _skip_unless_live()

    # Snapshot the current fixture bytes.
    before = {p.name: _hash_file(p) for p in sorted(_FIXTURE_DIR.glob("*.json"))}
    assert before, "fixture directory is empty — run the capture script first"

    # Run the capture script a second time (picks up the same live TAO
    # endpoint configured in ~/.vlm_feedback_loop/.env).
    result = subprocess.run(
        [sys.executable, str(_CAPTURE_SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    after = {p.name: _hash_file(p) for p in sorted(_FIXTURE_DIR.glob("*.json"))}
    assert before == after, (
        "capture script is not idempotent — fixture hashes drifted:\n"
        f"  before: {before}\n"
        f"  after:  {after}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Live match — live TAO responses match committed fixtures (opt-in)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_tao_openapi_matches_fixture():
    """The committed OpenAPI fixture matches what the live TAO endpoint
    returns. Any drift flags that the Blueprint's TAO code must adapt.
    """
    _skip_unless_live()
    from vlm_feedback_loop.config import get_settings
    from vlm_feedback_loop.services.http_client import resilient_request

    s = get_settings()
    if not s.TAO_API_BASE_URL:
        pytest.skip("TAO_API_BASE_URL not configured")

    url = f"{s.TAO_API_BASE_URL.rstrip('/')}/openapi.json"
    result = await resilient_request("GET", url, deadline_s=30.0, max_retries=1)
    assert result.error_class is None, result.error_detail
    assert isinstance(result.body, dict)

    # Compare structure — same TAO version → byte-identical after
    # pretty-print normalization.
    live_norm = json.dumps(result.body, indent=2, sort_keys=True) + "\n"
    committed_raw = (_FIXTURE_DIR / "openapi_v2.json").read_text("utf-8")
    assert live_norm == committed_raw, (
        "live TAO OpenAPI spec diverged from committed fixture — re-run "
        "scripts/capture_tao_fixtures.py to refresh"
    )

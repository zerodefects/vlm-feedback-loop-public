# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``scripts/probe_image_caps_sweep.py`` core functions.

The probe script is the answer to "what's the real ``max_images_per_request``
for cosmos-reason2 NIM running locally?" The
``_probe_image_cap_support`` in ``services/model_config_service.py``
only verifies "is the seeded cap exactly right?" via N + N+1; this script
sweeps a wide ladder to find the true cap and capture the failure-error
string for layer attribution.

These tests pin the contract:

* on a 200 ladder, ``cap == max(ladder)`` and ``first_failure is None``
* on a 4xx at N=K, ``cap == K-1`` and the rejection text is captured
  as ``failure_excerpt`` (the layer-attribution signal)
* on a transport error (ConnectError/Timeout), the sweep stops with
  ``http_status=-1`` so the operator can distinguish "cap" from
  "endpoint unreachable"
* the request body contains exactly N image_url content parts plus the
  text instruction (no schema_response, no max_tokens drift) — vital
  because the probe is supposed to measure the *raw* image cap, not
  cap-when-also-requesting-json_schema
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

# The probe is a stand-alone script under ``scripts/``; load it as a
# module so we can call ``_probe_one_count`` and ``_sweep`` directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE_SCRIPT = _REPO_ROOT / "scripts" / "probe_image_caps_sweep.py"

# Insert backend on path so the script's own ``from vlm_feedback_loop...`` import works.
sys.path.insert(0, str(_REPO_ROOT / "src" / "backend"))

_spec = importlib.util.spec_from_file_location("probe_image_caps_sweep", _PROBE_SCRIPT)
assert _spec and _spec.loader
probe_mod = importlib.util.module_from_spec(_spec)
# Register before exec — Python 3.13's @dataclass walks sys.modules to
# resolve cls.__module__; without registration it crashes mid-decoration.
sys.modules["probe_image_caps_sweep"] = probe_mod
_spec.loader.exec_module(probe_mod)


def _make_client(handler) -> httpx.AsyncClient:
    """An AsyncClient backed by httpx.MockTransport — repo-standard pattern (test_http_client.py)."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def probe_image_data_url():
    return probe_mod.generate_probe_image_data_url()


class TestProbeOneCount:
    @pytest.mark.asyncio
    async def test_200_returns_clean_attempt(self, probe_image_data_url):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = request.read()
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )

        async with _make_client(handler) as client:
            attempt = await probe_mod._probe_one_count(
                client,
                "http://probe-target.invalid/v1",
                "test-model",
                n_images=4,
                auth={"Content-Type": "application/json"},
                probe_image_data_url=probe_image_data_url,
            )

        assert attempt.http_status == 200
        assert attempt.error_excerpt is None
        assert attempt.n_images == 4
        # Exact path: base_url is normalized (rstrip /) and probe targets /chat/completions.
        assert captured["url"] == "http://probe-target.invalid/v1/chat/completions"
        # Body has exactly 4 image_url parts + 1 text part (verifies probe is not
        # silently double-counting or attaching extra content parts).
        import json

        body = json.loads(captured["body"])
        content = body["messages"][0]["content"]
        image_parts = [c for c in content if c.get("type") == "image_url"]
        text_parts = [c for c in content if c.get("type") == "text"]
        assert len(image_parts) == 4
        assert len(text_parts) == 1
        # max_tokens must be set (NIM default of 16 would cap output unexpectedly)
        # and response_format must be ABSENT (the cap-when-also-requesting-json_schema
        # confounder must not pollute the cap-attribution signal).
        assert body["max_tokens"] == 4
        assert "response_format" not in body

    @pytest.mark.asyncio
    async def test_4xx_captures_error_excerpt_for_layer_attribution(
        self, probe_image_data_url
    ):
        # The hosted-Mistral live-probe rejection at N=9 was literally
        # "At most 8 image(s) may be provided in one prompt" — that's
        # the build.nvidia.com gateway pattern.
        # A NIM/vLLM rejection looks different. The probe MUST surface
        # the exact text so the operator can classify which layer rejects.
        rejection_body = (
            '{"object":"error","message":"At most 8 image(s) may be '
            'provided in one prompt","type":"invalid_request_error"}'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text=rejection_body)

        async with _make_client(handler) as client:
            attempt = await probe_mod._probe_one_count(
                client,
                "http://probe-target.invalid/v1",
                "test-model",
                n_images=9,
                auth={"Content-Type": "application/json"},
                probe_image_data_url=probe_image_data_url,
            )

        assert attempt.http_status == 400
        assert attempt.error_excerpt is not None
        # Layer-attribution requires the literal cap-rejection text be
        # captured; anything that erases it makes the probe useless.
        assert "At most 8 image(s)" in attempt.error_excerpt

    @pytest.mark.asyncio
    async def test_transport_error_marked_negative_one(self, probe_image_data_url):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        async with _make_client(handler) as client:
            attempt = await probe_mod._probe_one_count(
                client,
                "http://probe-target.invalid/v1",
                "test-model",
                n_images=1,
                auth={"Content-Type": "application/json"},
                probe_image_data_url=probe_image_data_url,
            )

        # http_status=-1 is the "I couldn't even reach the endpoint"
        # signal — distinct from a 4xx cap rejection so the sweep loop
        # can treat it as inconclusive (don't claim cap=N-1 when the
        # network ate the request).
        assert attempt.http_status == -1
        assert attempt.error_excerpt is not None
        assert "ConnectError" in attempt.error_excerpt


class TestSweep:
    @pytest.mark.asyncio
    async def test_all_succeed_cap_equals_top_of_ladder(self, monkeypatch):
        # All N succeed → cap = ladder[-1]; first_failure is None; the
        # operator should be advised to extend the ladder upward.
        monkeypatch.setattr(
            probe_mod.httpx, "AsyncClient", _AsyncClientFactory(_always_200_handler)
        )
        result = await probe_mod._sweep(
            "http://probe-target.invalid/v1",
            "test-model",
            ladder=[1, 4, 8, 16],
            auth={"Content-Type": "application/json"},
        )
        assert result.cap == 16
        assert result.first_failure is None
        assert result.failure_excerpt is None
        assert len(result.attempts) == 4
        assert all(a.http_status == 200 for a in result.attempts)

    @pytest.mark.asyncio
    async def test_4xx_at_K_yields_cap_K_minus_1(self, monkeypatch):
        # Ladder 1, 4, 8, 12, 16 — endpoint accepts 1, 4, 8 then rejects 12.
        # Probe MUST stop at first 4xx (assume monotonic) and return
        # cap=8, first_failure=12.
        def handler(request: httpx.Request) -> httpx.Response:
            import json as _j

            body = _j.loads(request.read())
            n_images = sum(
                1
                for c in body["messages"][0]["content"]
                if c.get("type") == "image_url"
            )
            if n_images <= 8:
                return httpx.Response(
                    200, json={"choices": [{"message": {"content": "ok"}}]}
                )
            return httpx.Response(
                400,
                text='{"object":"error","message":"max_images exceeded","type":"invalid_request_error"}',
            )

        monkeypatch.setattr(
            probe_mod.httpx, "AsyncClient", _AsyncClientFactory(handler)
        )
        result = await probe_mod._sweep(
            "http://probe-target.invalid/v1",
            "test-model",
            ladder=[1, 4, 8, 12, 16],
            auth={"Content-Type": "application/json"},
        )
        assert result.cap == 8
        assert result.first_failure == 12
        assert result.failure_status == 400
        assert "max_images exceeded" in (result.failure_excerpt or "")
        # Crucially: the sweep did NOT continue past the first 4xx (no probe at N=16).
        assert len(result.attempts) == 4
        assert [a.n_images for a in result.attempts] == [1, 4, 8, 12]

    @pytest.mark.asyncio
    async def test_to_dict_serializes_attempts(self, monkeypatch):
        monkeypatch.setattr(
            probe_mod.httpx, "AsyncClient", _AsyncClientFactory(_always_200_handler)
        )
        result = await probe_mod._sweep(
            "http://probe-target.invalid/v1",
            "test-model",
            ladder=[1, 2],
            auth={"Content-Type": "application/json"},
        )
        d = result.to_dict()
        # The JSON file written via --json-out must contain enough to
        # rerun the analysis off-line: cap, ladder, per-attempt status,
        # error excerpt for layer attribution.
        assert set(d.keys()) >= {
            "base_url",
            "model_name",
            "ladder",
            "cap",
            "first_failure",
            "failure_status",
            "failure_excerpt",
            "attempts",
            "total_wall_ms",
        }
        assert isinstance(d["attempts"], list)
        assert all("n_images" in a and "http_status" in a for a in d["attempts"])


# ── helpers ──────────────────────────────────────────────────────────────────


def _always_200_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})


class _AsyncClientFactory:
    """Drop-in replacement for httpx.AsyncClient that uses MockTransport.

    The probe's _sweep calls ``async with httpx.AsyncClient() as client:``
    with no kwargs, so we replace the class itself. Each call returns a
    fresh client backed by MockTransport(handler).

    ``_RealAsyncClient`` is captured at import time (before any
    monkeypatching), so the factory's own client construction doesn't
    recurse back into itself when ``httpx.AsyncClient`` is patched.
    """

    _RealAsyncClient = httpx.AsyncClient  # snapshot before patching

    def __init__(self, handler):
        self._handler = handler

    def __call__(self, *args, **kwargs):
        return self._RealAsyncClient(transport=httpx.MockTransport(self._handler))

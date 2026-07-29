# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NIM benchmark adapter.

Covers:
  - ``select_adapter`` returns ``HttpxAdapter`` when ``genai_perf`` is not
    importable (fallback path).
  - ``select_adapter`` returns ``GenaiPerfAdapter`` when ``genai_perf``
    *is* importable.
  - Both adapters' ``BenchmarkResult.to_json()`` emit the IDENTICAL key
    set (identical artifact schema).
  - ``HttpxAdapter`` percentile math is correct on a known sample.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import httpx

from vlm_feedback_loop.services.benchmark_adapter import (
    BenchmarkResult,
    GenaiPerfAdapter,
    HttpxAdapter,
    _percentiles,
    select_adapter,
)

# ── select_adapter ────────────────────────────────────────────────────────────


class TestAdapterSelection:
    def test_select_returns_httpx_when_genai_perf_missing(self, monkeypatch):
        # Force the import to fail: a None sys.modules entry makes
        # ``import genai_perf`` raise ImportError.
        monkeypatch.setitem(sys.modules, "genai_perf", None)
        adapter = select_adapter()
        assert isinstance(adapter, HttpxAdapter)

    def test_select_returns_genai_perf_when_present(self, monkeypatch):
        # Inject a stub module so the import succeeds.
        monkeypatch.setitem(sys.modules, "genai_perf", MagicMock())
        adapter = select_adapter()
        assert isinstance(adapter, GenaiPerfAdapter)


# ── BenchmarkResult schema parity ───────────────────────────────────────


class TestArtifactSchemaParity:
    def test_both_adapters_emit_identical_keys(self):
        """Both drivers MUST produce the same artifact
        shape. The router/UI consume only ``BenchmarkResult.to_json()``.
        """
        # Build a minimal BenchmarkResult to assert the dataclass field set.
        sample = BenchmarkResult(
            concurrency=1,
            latency_p50_ms=10.0,
            latency_p90_ms=12.0,
            latency_p99_ms=14.0,
            ttft_p50_ms=None,
            ttft_p90_ms=None,
            itl_p50_ms=None,
            itl_p90_ms=None,
            request_count=100,
            error_count=0,
            prometheus={},
            artifact_dir="/tmp",
            driver="httpx",
        )
        keyset = set(sample.to_json().keys())
        # The contract: both drivers emit exactly this keyset.
        expected = {
            "concurrency",
            "latency_p50_ms",
            "latency_p90_ms",
            "latency_p99_ms",
            "ttft_p50_ms",
            "ttft_p90_ms",
            "itl_p50_ms",
            "itl_p90_ms",
            "request_count",
            "error_count",
            "prometheus",
            "artifact_dir",
            "driver",
            "failed",
        }
        assert keyset == expected


# ── _percentiles math ────────────────────────────────────────────────────────


class TestPercentileMath:
    def test_empty_returns_zeros(self):
        p50, p90, p99 = _percentiles([])
        assert (p50, p90, p99) == (0.0, 0.0, 0.0)

    def test_small_sample_falls_back_to_max(self):
        # Fewer than 4 samples — quantiles rejects, helper falls back.
        p50, p90, p99 = _percentiles([10.0, 20.0, 30.0])
        # Median (p50) is the middle element 20.0; tail percentiles use max.
        assert p50 == 20.0
        assert p90 == 30.0
        assert p99 == 30.0

    def test_uniform_distribution(self):
        samples = [float(i) for i in range(1, 101)]
        p50, p90, p99 = _percentiles(samples)
        # Inclusive method on 100 samples → p50 ≈ 50.5, p90 ≈ 90.1, p99 ≈ 99.01
        assert 49.0 <= p50 <= 52.0
        assert 89.0 <= p90 <= 91.0
        assert 98.0 <= p99 <= 100.0


# ── HttpxAdapter end-to-end with mocked NIM ──────────────────────────────────


class TestHttpxAdapter:
    def test_run_writes_artifact_and_returns_result(self, tmp_path, monkeypatch):
        """Mock the NIM endpoint; assert the adapter calls it the right
        number of times, emits a BenchmarkResult, and writes result.json.
        """
        adapter = HttpxAdapter()

        async def _fake_post(self, url, json=None):
            # Simulate a successful chat completion.
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"ok": true}'}}]},
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        async def _go():
            return await adapter.run(
                base_url="http://localhost:8002/v1",
                model="student-foo",
                concurrency=2,
                project_dir=str(tmp_path),
                student_model_id="abcdef0123456789",
                request_count=10,
            )

        result: BenchmarkResult = asyncio.run(_go())
        assert result.driver == "httpx"
        assert result.concurrency == 2
        assert result.request_count == 10
        assert result.error_count == 0
        assert result.failed is False  # real measurement
        # TTFT/ITL not measured by httpx adapter — None.
        assert result.ttft_p50_ms is None
        # Artifact file exists.
        artifact_path = Path(result.artifact_dir) / "result.json"
        assert artifact_path.exists()
        text = artifact_path.read_text()
        # The driver field round-trips through the JSON serialization.
        assert '"driver": "httpx"' in text

    def test_errors_count_distinct_from_latencies(self, tmp_path, monkeypatch):
        """Failed responses (e.g. 500) should increment error_count without
        contaminating the latency samples.
        """
        adapter = HttpxAdapter()

        call_index = {"i": 0}

        async def _fake_post(self, url, json=None):
            i = call_index["i"]
            call_index["i"] += 1
            if i % 2 == 0:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        async def _go():
            return await adapter.run(
                base_url="http://localhost:8002/v1",
                model="student-foo",
                concurrency=1,
                project_dir=str(tmp_path),
                student_model_id="abcdef0123456789",
                request_count=4,
            )

        result = asyncio.run(_go())
        assert result.request_count == 2  # Only successful calls
        assert result.error_count == 2

    def test_all_requests_error_marks_failed(self, tmp_path, monkeypatch):
        """Every request failing → zero successful samples is a
        non-measurement, not a genuine zero-latency result. failed=True so
        the sweep records the concurrency skipped instead of persisting
        fake zeros the ServingMatrix would render as valid (the httpx
        counterpart to genai-perf's process-failure path)."""
        adapter = HttpxAdapter()

        async def _fake_post(self, url, json=None):
            return httpx.Response(500, json={"error": "down"})

        monkeypatch.setattr(httpx.AsyncClient, "post", _fake_post)

        async def _go():
            return await adapter.run(
                base_url="http://localhost:8002/v1",
                model="student-foo",
                concurrency=1,
                project_dir=str(tmp_path),
                student_model_id="abcdef0123456789",
                request_count=4,
            )

        result = asyncio.run(_go())
        assert result.request_count == 0
        assert result.error_count == 4
        assert result.failed is True
        assert result.latency_p50_ms == 0.0

    def test_requests_carry_blueprint_source_header(self, tmp_path, monkeypatch):
        """Every backend-constructed outbound NIM request carries the
        {"source": "vlm-feedback-loop"} usage-tracking header — including
        the benchmark sweep's chat-completions traffic, which bypasses the
        shared nim_client for latency-measurement fidelity."""
        adapter = HttpxAdapter()
        captured_headers = []

        # Patch send (not post) so the client-level default headers are
        # merged into the built request before capture.
        async def _fake_send(self, request, **kwargs):
            captured_headers.append(request.headers)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
                request=request,
            )

        monkeypatch.setattr(httpx.AsyncClient, "send", _fake_send)

        async def _go():
            return await adapter.run(
                base_url="http://localhost:8002/v1",
                model="student-foo",
                concurrency=1,
                project_dir=str(tmp_path),
                student_model_id="abcdef0123456789",
                request_count=2,
            )

        result = asyncio.run(_go())
        assert result.request_count == 2
        assert captured_headers
        for headers in captured_headers:
            assert headers.get("source") == "vlm-feedback-loop"


# ── GenaiPerfAdapter parser ──────────────────────────────────────────────────


class TestGenaiPerfParser:
    def test_parse_succeeds_with_full_output(self, tmp_path):
        """The parser handles a typical genai-perf JSON output shape."""
        from vlm_feedback_loop.services.benchmark_adapter import (
            _parse_genai_perf_output,
        )

        # Simulate genai-perf's output file.
        output = {
            "request_latency": {"p50": 100.0, "p90": 200.0, "p99": 300.0},
            "time_to_first_token": {"p50": 50.0, "p90": 80.0},
            "inter_token_latency": {"p50": 12.0, "p90": 18.0},
            "request_count": 100,
            "error_count": 1,
        }
        import json

        artifact_dir = tmp_path / "bench"
        artifact_dir.mkdir()
        (artifact_dir / "profile_export_genai_perf.json").write_text(json.dumps(output))

        result = _parse_genai_perf_output(
            artifact_dir=artifact_dir,
            concurrency=8,
            return_code=0,
        )
        assert result.driver == "genai-perf"
        assert result.concurrency == 8
        assert result.latency_p50_ms == 100.0
        assert result.latency_p90_ms == 200.0
        assert result.latency_p99_ms == 300.0
        assert result.ttft_p50_ms == 50.0
        assert result.itl_p90_ms == 18.0
        assert result.request_count == 100
        assert result.error_count == 1
        assert result.failed is False

    def test_parse_handles_missing_file(self, tmp_path):
        from vlm_feedback_loop.services.benchmark_adapter import (
            _parse_genai_perf_output,
        )

        artifact_dir = tmp_path / "bench"
        artifact_dir.mkdir()
        result = _parse_genai_perf_output(
            artifact_dir=artifact_dir,
            concurrency=8,
            return_code=1,
        )
        # Returns a zeroed BenchmarkResult; does not crash. failed=True marks
        # it as no-measurement so the sweep records the concurrency skipped
        # instead of persisting fake zeros the UI would render as valid.
        assert result.latency_p50_ms == 0.0
        assert result.ttft_p50_ms is None
        assert result.failed is True

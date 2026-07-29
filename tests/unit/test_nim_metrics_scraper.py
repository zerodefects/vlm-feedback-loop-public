# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NIM Prometheus metrics scraper.

Covers:
  - _build_metrics_url strips a trailing /v1 (and only that) before
    appending /metrics.
  - _parse_prom_text extracts all three TRACKED_METRICS when present.
  - Missing metric → 0.0 default.
  - Non-200 / transport error → all-zeros dict.
"""

from __future__ import annotations

import asyncio

from vlm_feedback_loop.services.http_client import HttpResult
from vlm_feedback_loop.services.nim_metrics_scraper import (
    TRACKED_METRICS,
    _build_metrics_url,
    _parse_prom_text,
    scrape_prometheus,
)

# ── _build_metrics_url ────────────────────────────────────────────────────────


class TestBuildMetricsUrl:
    def test_strips_trailing_v1(self):
        assert (
            _build_metrics_url("http://localhost:8002/v1")
            == "http://localhost:8002/metrics"
        )

    def test_strips_trailing_v1_with_slash(self):
        assert (
            _build_metrics_url("http://localhost:8002/v1/")
            == "http://localhost:8002/metrics"
        )

    def test_no_v1_suffix(self):
        assert (
            _build_metrics_url("http://localhost:8002")
            == "http://localhost:8002/metrics"
        )


# ── _parse_prom_text ─────────────────────────────────────────────────────────


SAMPLE_METRICS = """
# HELP request_failure_total Total failed requests
# TYPE request_failure_total counter
request_failure_total{model="student"} 42.0
# HELP request_success_total Total successful requests
# TYPE request_success_total counter
request_success_total{model="student"} 1234.0
# HELP gpu_cache_usage_perc KV cache utilization
# TYPE gpu_cache_usage_perc gauge
gpu_cache_usage_perc 0.78
# HELP unrelated_metric noise
# TYPE unrelated_metric gauge
unrelated_metric 999.99
""".strip()


class TestParseText:
    def test_extracts_all_tracked_metrics(self):
        result = _parse_prom_text(SAMPLE_METRICS)
        assert result["request_failure_total"] == 42.0
        assert result["request_success_total"] == 1234.0
        assert result["gpu_cache_usage_perc"] == 0.78

    def test_returns_zero_for_missing_metrics(self):
        text = "request_success_total{model='x'} 99.0"
        result = _parse_prom_text(text)
        assert result["request_success_total"] == 99.0
        assert result["request_failure_total"] == 0.0
        assert result["gpu_cache_usage_perc"] == 0.0

    def test_keyset_always_complete(self):
        result = _parse_prom_text("")
        assert set(result.keys()) == set(TRACKED_METRICS)

    def test_ignores_help_and_type_lines(self):
        text = """
# HELP request_failure_total this should not parse
# TYPE request_failure_total counter
request_failure_total 5
"""
        result = _parse_prom_text(text)
        assert result["request_failure_total"] == 5.0


# ── scrape_prometheus ────────────────────────────────────────────────────────


class TestScrape:
    def test_returns_zeros_on_non_200(self, monkeypatch):
        async def _fake(*args, **kwargs):
            return HttpResult(status_code=503, body="Service unavailable")

        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_metrics_scraper.resilient_request",
            _fake,
        )
        result = asyncio.run(scrape_prometheus("http://x/v1"))
        assert all(value == 0.0 for value in result.values())
        assert set(result.keys()) == set(TRACKED_METRICS)

    def test_returns_zeros_on_transport_error(self, monkeypatch):
        async def _fake(*args, **kwargs):
            return HttpResult(
                status_code=None,
                body=None,
                error_class="endpoint_error",
                error_detail="connection refused",
            )

        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_metrics_scraper.resilient_request",
            _fake,
        )
        result = asyncio.run(scrape_prometheus("http://x/v1"))
        assert all(value == 0.0 for value in result.values())

    def test_parses_real_response_text(self, monkeypatch):
        async def _fake(*args, **kwargs):
            return HttpResult(status_code=200, body=SAMPLE_METRICS)

        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_metrics_scraper.resilient_request",
            _fake,
        )
        result = asyncio.run(scrape_prometheus("http://x/v1"))
        assert result["request_failure_total"] == 42.0
        assert result["request_success_total"] == 1234.0
        assert result["gpu_cache_usage_perc"] == 0.78

    def test_url_is_metrics_not_v1_metrics(self, monkeypatch):
        captured = {}

        async def _fake(method, url, **kwargs):
            captured["url"] = url
            return HttpResult(status_code=200, body="")

        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_metrics_scraper.resilient_request",
            _fake,
        )
        asyncio.run(scrape_prometheus("http://localhost:8002/v1"))
        assert captured["url"] == "http://localhost:8002/metrics"

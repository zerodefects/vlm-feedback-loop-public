# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM Prometheus metrics scraper.

Reads ``request_failure_total``, ``request_success_total``, and
``gpu_cache_usage_perc`` from a deployed NIM endpoint's ``/metrics`` path
(Prometheus text exposition format). Used by the Student NIM benchmarking
pipeline to attach per-concurrency Prometheus snapshots to the
``BenchmarkResult`` records.

The scraper is best-effort:

  - Missing metric → ``0.0`` for that key (NIM endpoints that don't expose
    Prometheus metrics still produce a usable ``BenchmarkResult``).
  - Non-200 response or transport error → all-zeros dict.
  - Single GET with a tight 10s deadline; one retry only (the benchmark is
    the heavy operation, not this scrape).

This is intentionally NOT a full Prometheus parser — it extracts only the
three metric names the Blueprint tracks and ignores everything else
(including labels). If a metric appears multiple times (per-instance, per-
endpoint), the first numeric sample wins.
"""

from __future__ import annotations

import logging
from typing import Final

from vlm_feedback_loop.services.http_client import resilient_request

logger = logging.getLogger("vlm_feedback_loop.services.nim_metrics_scraper")


# Metric names tracked for the serving benchmark.
TRACKED_METRICS: Final[tuple[str, ...]] = (
    "request_failure_total",
    "request_success_total",
    "gpu_cache_usage_perc",
)


def _build_metrics_url(base_url: str) -> str:
    """Convert an OpenAI-compatible base URL into the Prometheus metrics URL.

    NIM exposes ``/metrics`` at the root, NOT under ``/v1``. So we strip a
    trailing ``/v1`` (or ``/v1/``) from the base URL before appending
    ``/metrics``. Examples:

      - ``http://localhost:8002/v1`` → ``http://localhost:8002/metrics``
      - ``http://localhost:8002/v1/`` → ``http://localhost:8002/metrics``
      - ``http://localhost:8002`` → ``http://localhost:8002/metrics``
    """
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3].rstrip("/")
    return f"{cleaned}/metrics"


def _parse_prom_text(body: str) -> dict[str, float]:
    """Parse Prometheus exposition-format text and extract tracked metrics.

    Returns a dict keyed on the metric name with float values. Missing
    metrics are filled with 0.0 so downstream consumers always see the full
    keyset. The first numeric sample wins when a metric appears multiple
    times (per-label/per-instance variants are intentionally collapsed).
    """
    result: dict[str, float] = dict.fromkeys(TRACKED_METRICS, 0.0)
    seen: set[str] = set()

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Sample line format: ``metric_name{labels...} 12.34`` or
        # ``metric_name 12.34``. Split off any ``{labels}`` chunk first
        # so we can match the bare metric name.
        head = line.split("{", 1)[0].strip()
        # Re-split to peel off the metric name from the rest after labels.
        parts = line.split()
        if len(parts) < 2:
            continue

        # The metric name is everything before either "{" or whitespace.
        name = head if head and " " not in head else parts[0].split("{", 1)[0]

        if name not in TRACKED_METRICS or name in seen:
            continue

        try:
            value = float(parts[-1])
        except ValueError:
            continue

        result[name] = value
        seen.add(name)
        if len(seen) == len(TRACKED_METRICS):
            break

    return result


async def scrape_prometheus(
    base_url: str,
    *,
    deadline_s: float = 10.0,
) -> dict[str, float]:
    """Scrape the NIM Prometheus metrics endpoint.

    Returns a dict with all three TRACKED_METRICS keys. Missing data —
    metric absent, non-200, transport error — produces 0.0 defaults so
    callers never see a partial dict.
    """
    metrics_url = _build_metrics_url(base_url)

    result = await resilient_request(
        "GET",
        metrics_url,
        deadline_s=deadline_s,
        max_retries=2,
    )

    if result.status_code != 200 or not isinstance(result.body, str):
        if result.status_code is not None:
            logger.debug(
                "Prometheus scrape returned status=%s for %s",
                result.status_code,
                metrics_url,
            )
        else:
            logger.debug(
                "Prometheus scrape transport error for %s: %s",
                metrics_url,
                result.error_detail,
            )
        return dict.fromkeys(TRACKED_METRICS, 0.0)

    return _parse_prom_text(result.body)

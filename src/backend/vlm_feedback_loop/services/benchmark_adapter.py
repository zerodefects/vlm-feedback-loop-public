# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned AIPerf adapter for production-representative Student benchmarks."""

from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

from vlm_feedback_loop.services.subprocess_utils import communicate_with_timeout

AIPERF_VERSION = "0.10.0"


@dataclass
class BenchmarkResult:
    concurrency: int
    status: str = "passed"
    latency_p50_ms: float | None = None
    latency_p90_ms: float | None = None
    latency_p99_ms: float | None = None
    request_throughput_rps: float | None = None
    benchmark_duration_s: float | None = None
    attempted_request_count: int = 0
    successful_request_count: int = 0
    failed_request_count: int = 0
    request_count: int = 0
    error_count: int = 0
    failure_rate: float | None = None
    input_tokens_mean: float | None = None
    output_tokens_mean: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p90_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p90_ms: float | None = None
    prometheus: dict[str, float | None] = field(default_factory=dict[str, float | None])
    prometheus_available: bool = False
    artifact_dir: str = ""
    driver: str = "aiperf"
    driver_version: str = AIPERF_VERSION
    export_schema_version: str | None = None
    error_summary: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    failure_reason: str | None = None
    failed: bool = False

    def __post_init__(self) -> None:
        # Compatibility for injected test adapters and historic callers while
        # the persisted v1 shape carries the explicit successful/failed names.
        if self.successful_request_count == 0 and self.request_count:
            self.successful_request_count = self.request_count
        elif self.request_count == 0 and self.successful_request_count:
            self.request_count = self.successful_request_count
        if self.failed_request_count == 0 and self.error_count:
            self.failed_request_count = self.error_count
        elif self.error_count == 0 and self.failed_request_count:
            self.error_count = self.failed_request_count
        if self.attempted_request_count == 0:
            self.attempted_request_count = self.request_count + self.error_count
        if self.failure_rate is None and self.attempted_request_count:
            self.failure_rate = self.error_count / self.attempted_request_count

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkAdapter(Protocol):
    async def run(
        self,
        *,
        base_url: str,
        model: str,
        concurrency: int,
        input_file: Path,
        artifact_dir: Path,
        request_count: int,
        auth_headers: dict[str, str] | None = None,
        deadline_s: float = 1200.0,
    ) -> BenchmarkResult: ...


def _metric_value(
    raw: dict[str, Any], metric: str, statistic: str, expected_unit: str
) -> float | None:
    raw_node = raw.get(metric)
    if not isinstance(raw_node, dict):
        return None
    node = cast("dict[str, Any]", raw_node)
    if node.get("unit") != expected_unit:
        return None
    value = node.get(statistic)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _duration_seconds(raw: dict[str, Any]) -> float | None:
    start = raw.get("start_time")
    end = raw.get("end_time")
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    try:
        duration = (
            datetime.fromisoformat(end.replace("Z", "+00:00"))
            - datetime.fromisoformat(start.replace("Z", "+00:00"))
        ).total_seconds()
    except ValueError:
        return None
    return duration if duration >= 0 and math.isfinite(duration) else None


def parse_aiperf_output(
    *,
    artifact_dir: Path,
    concurrency: int,
    expected_request_count: int,
    return_code: int | None,
) -> BenchmarkResult:
    """Parse AIPerf's versioned summary without inventing zero metrics."""
    candidates = sorted(artifact_dir.rglob("profile_export_aiperf.json"))
    if return_code != 0 or len(candidates) != 1:
        reason = (
            f"aiperf_exit_{return_code}"
            if return_code != 0
            else f"expected_one_summary_found_{len(candidates)}"
        )
        return BenchmarkResult(
            concurrency=concurrency,
            status="failed",
            artifact_dir=str(artifact_dir),
            failure_reason=reason,
            failed=True,
        )
    try:
        raw_value = json.loads(candidates[0].read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return BenchmarkResult(
            concurrency=concurrency,
            status="failed",
            artifact_dir=str(artifact_dir),
            failure_reason=f"invalid_aiperf_summary:{type(exc).__name__}",
            failed=True,
        )
    if not isinstance(raw_value, dict):
        return BenchmarkResult(
            concurrency=concurrency,
            status="failed",
            artifact_dir=str(artifact_dir),
            failure_reason="invalid_aiperf_summary_root",
            failed=True,
        )
    raw = cast("dict[str, Any]", raw_value)

    successful = int(_metric_value(raw, "request_count", "avg", "requests") or 0)
    failures = int(_metric_value(raw, "error_request_count", "avg", "requests") or 0)
    attempted = successful + failures
    p50 = _metric_value(raw, "request_latency", "p50", "ms")
    p90 = _metric_value(raw, "request_latency", "p90", "ms")
    p99 = _metric_value(raw, "request_latency", "p99", "ms")
    throughput = _metric_value(raw, "request_throughput", "avg", "requests/sec")
    error_summary_raw = raw.get("error_summary")
    error_summary: list[dict[str, Any]] = []
    if isinstance(error_summary_raw, list):
        for item in cast("list[Any]", error_summary_raw):
            if isinstance(item, dict):
                error_summary.append(cast("dict[str, Any]", item))
    reasons: list[str] = []
    reported_driver_version = raw.get("aiperf_version")
    if reported_driver_version != AIPERF_VERSION:
        reasons.append(
            f"driver_version_{reported_driver_version}_expected_{AIPERF_VERSION}"
        )
    if raw.get("was_cancelled") is True:
        reasons.append("cancelled")
    if attempted != expected_request_count:
        reasons.append(f"request_count_{attempted}_expected_{expected_request_count}")
    if failures:
        reasons.append(f"failed_requests_{failures}")
    if any(value is None for value in (p50, p90, p99, throughput)):
        reasons.append("missing_or_non_finite_required_metric")

    passed = not reasons
    return BenchmarkResult(
        concurrency=concurrency,
        status="passed" if passed else "failed",
        latency_p50_ms=p50,
        latency_p90_ms=p90,
        latency_p99_ms=p99,
        request_throughput_rps=throughput,
        benchmark_duration_s=_duration_seconds(raw),
        attempted_request_count=attempted,
        successful_request_count=successful,
        failed_request_count=failures,
        request_count=successful,
        error_count=failures,
        failure_rate=(failures / attempted) if attempted else None,
        input_tokens_mean=_metric_value(raw, "input_sequence_length", "avg", "tokens"),
        output_tokens_mean=_metric_value(
            raw, "output_sequence_length", "avg", "tokens"
        ),
        artifact_dir=str(artifact_dir),
        driver_version=str(reported_driver_version or AIPERF_VERSION),
        export_schema_version=(
            str(raw["schema_version"])
            if raw.get("schema_version") is not None
            else None
        ),
        error_summary=error_summary,
        failure_reason=";".join(reasons) if reasons else None,
        failed=not passed,
    )


class AIPerfAdapter:
    """Run the mandatory pinned NVIDIA AIPerf load driver as a subprocess."""

    driver_name = "aiperf"

    async def run(
        self,
        *,
        base_url: str,
        model: str,
        concurrency: int,
        input_file: Path,
        artifact_dir: Path,
        request_count: int,
        auth_headers: dict[str, str] | None = None,
        deadline_s: float = 1200.0,
    ) -> BenchmarkResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        authorization = (auth_headers or {}).get("Authorization")
        endpoint_config: dict[str, Any] = {
            "urls": [base_url.rstrip("/")],
            "type": "chat",
            "timeout": deadline_s,
            "use_server_token_count": True,
            "headers": {"source": "vlm-feedback-loop"},
        }
        if authorization:
            scheme, _, credential = authorization.partition(" ")
            if scheme.lower() != "bearer" or not credential:
                return BenchmarkResult(
                    concurrency=concurrency,
                    status="failed",
                    artifact_dir=str(artifact_dir),
                    failure_reason="unsupported_benchmark_auth_header",
                    failed=True,
                )
            env["VLM_BENCHMARK_API_KEY"] = credential
            endpoint_config["api_key"] = "${VLM_BENCHMARK_API_KEY}"

        # A complete v2 config is required: AIPerf validates config files
        # before applying any CLI overrides. JSON is valid YAML and prevents
        # model names, URLs, or paths from becoming YAML syntax. The credential
        # stays an environment placeholder, never plaintext on disk or argv.
        config_file = input_file.parent / f"aiperf-c{concurrency}.yaml"
        config_file.write_text(
            json.dumps(
                {
                    "schemaVersion": "2.0",
                    "randomSeed": 17,
                    "benchmark": {
                        "models": {"items": [{"name": model}]},
                        "endpoint": endpoint_config,
                        "datasets": [
                            {
                                "name": "main",
                                "type": "file",
                                "path": str(input_file),
                                "format": "raw_payload",
                                "sampling": "sequential",
                                "entries": request_count,
                                "random_seed": 17,
                            }
                        ],
                        "phases": [
                            {
                                "name": "profiling",
                                "type": "concurrency",
                                "requests": request_count,
                                "concurrency": concurrency,
                            }
                        ],
                        "artifacts": {
                            "dir": str(artifact_dir),
                            "records": False,
                            "raw": False,
                            "auto_plot": False,
                            "plot_required": False,
                        },
                        "tokenizer": {"name": "builtin"},
                        "gpu_telemetry": {"enabled": False},
                        "server_metrics": {"enabled": False},
                        "runtime": {"ui": "none"},
                    },
                },
                indent=2,
            )
            + "\n"
        )
        config_file.chmod(0o600)
        cmd = ["aiperf", "profile", "--config", str(config_file)]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = await communicate_with_timeout(
                process,
                timeout_s=deadline_s,
            )
        except TimeoutError:
            return BenchmarkResult(
                concurrency=concurrency,
                status="failed",
                artifact_dir=str(artifact_dir),
                failure_reason="aiperf_timeout",
                failed=True,
            )
        (artifact_dir / "driver_stdout.log").write_bytes(stdout[-1_000_000:])
        (artifact_dir / "driver_stderr.log").write_bytes(stderr[-1_000_000:])
        return parse_aiperf_output(
            artifact_dir=artifact_dir,
            concurrency=concurrency,
            expected_request_count=request_count,
            return_code=process.returncode,
        )


def select_adapter() -> BenchmarkAdapter:
    """Return the required driver; there is intentionally no fallback."""
    return AIPerfAdapter()


__all__ = [
    "AIPERF_VERSION",
    "AIPerfAdapter",
    "BenchmarkAdapter",
    "BenchmarkResult",
    "parse_aiperf_output",
    "select_adapter",
]

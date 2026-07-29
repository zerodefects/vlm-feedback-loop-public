# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM benchmark adapter.

Isolates the load-driver integration (request construction, concurrency
control, artifact parsing) behind a Protocol so a future migration from
``genai-perf`` to AIPerf is a one-file swap without redesigning the
benchmarking pipeline.

Two implementations ship today:

  - ``GenaiPerfAdapter`` — primary. Subprocess invocation of ``genai-perf
    profile --service-kind openai --endpoint-type chat ...``. Parses
    ``profile_export_genai_perf.json``.
  - ``HttpxAdapter`` — fallback (CI, unit tests, and environments without
    ``genai-perf``). Concurrent ``httpx.AsyncClient`` runner with the same
    OpenAI-compatible request shape. Produces the IDENTICAL output schema,
    so callers only see ``BenchmarkResult`` regardless of which path ran.

``select_adapter()`` try-imports ``genai_perf`` and falls back. The
``BenchmarkResult.driver`` field captures which path ran, for audit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import httpx

from vlm_feedback_loop.services.nim_client import NIM_DEFAULT_HEADERS

logger = logging.getLogger("vlm_feedback_loop.services.benchmark_adapter")


# ── Output artifact schema ────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Per-concurrency benchmark result.

    Both adapters MUST produce this exact shape. The ``driver`` field
    records which path ran. ``prometheus`` is left empty by the adapter
    itself; the lifecycle orchestrator scrapes ``/metrics`` post-run and
    merges the dict in.
    """

    concurrency: int
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    ttft_p50_ms: float | None
    ttft_p90_ms: float | None
    itl_p50_ms: float | None
    itl_p90_ms: float | None
    request_count: int
    error_count: int
    prometheus: dict[str, float] = field(default_factory=dict[str, float])
    artifact_dir: str = ""
    driver: str = ""  # "genai-perf" | "httpx"
    # True when the driver produced no real measurement (process failure /
    # missing output). Distinguishes a failed run from a genuine
    # zero-latency one so the sweep can mark the concurrency skipped instead
    # of persisting fake zeros the UI would render as valid.
    failed: bool = False

    def to_json(self) -> dict[str, Any]:
        """Stable dict serialization — the persisted benchmark shape.

        Consumed by ``student_nim_lifecycle`` (the per-Student
        ``benchmarks`` list) and written to each run's ``result.json``
        artifact; both adapters must emit identical keys.
        """
        return asdict(self)


# ── Adapter Protocol ──────────────────────────────────────────────────────────


class BenchmarkAdapter(Protocol):
    """Async load-driver. Implementations MUST be deterministic-keyed
    (the same ``project_dir`` + ``concurrency`` writes to a stable artifact
    path so reruns are idempotent).
    """

    async def run(
        self,
        *,
        base_url: str,
        model: str,
        concurrency: int,
        project_dir: str,
        student_model_id: str,
        request_count: int = 100,
        deadline_s: float = 1200.0,
    ) -> BenchmarkResult: ...


# ── Default prompt payload ────────────────────────────────────────────────────


def _default_prompt_payload(model: str) -> dict[str, Any]:
    """Default benchmark payload — text-only OpenAI chat completion.

    Apples-to-apples rule: every variant of the same Student
    is benchmarked with identical decoding params. We use a deterministic
    short prompt to keep prefill cost stable across concurrencies.
    """
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    'Return the single JSON object {"ok": true} and nothing else.'
                ),
            }
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def _percentiles(samples: list[float]) -> tuple[float, float, float]:
    """Return (p50, p90, p99) in milliseconds.

    For N < 4 samples the standard ``quantiles`` API rejects, so we fall
    back to the max value for tail percentiles (degenerate but explicit).
    """
    if not samples:
        return 0.0, 0.0, 0.0
    sorted_samples = sorted(samples)
    if len(sorted_samples) < 4:
        return (
            sorted_samples[len(sorted_samples) // 2],
            sorted_samples[-1],
            sorted_samples[-1],
        )
    quantiles = statistics.quantiles(sorted_samples, n=100, method="inclusive")
    # quantiles returns 99 cut points: q[i] is the (i+1)-th percentile.
    return quantiles[49], quantiles[89], quantiles[98]


# ── HttpxAdapter (fallback) ───────────────────────────────────────────────────


class HttpxAdapter:
    """Concurrent httpx runner.

    Sends ``concurrency`` parallel POSTs of the prompt payload using an
    ``asyncio.Semaphore`` to bound in-flight calls, then computes
    percentiles via ``statistics.quantiles``. Streaming is disabled so
    TTFT/ITL fields are ``None`` — explicitly documented in the
    BenchmarkResult schema. Writes the full result.json artifact under
    ``{project_dir}/benchmarks/{student_model_id}/{concurrency}/``.
    """

    driver_name = "httpx"

    async def run(
        self,
        *,
        base_url: str,
        model: str,
        concurrency: int,
        project_dir: str,
        student_model_id: str,
        request_count: int = 100,
        deadline_s: float = 1200.0,
    ) -> BenchmarkResult:
        artifact_dir = (
            Path(project_dir) / "benchmarks" / student_model_id / str(concurrency)
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        payload = _default_prompt_payload(model)

        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        sem = asyncio.Semaphore(concurrency)
        latencies_ms: list[float] = []
        errors = 0

        async def _one_request(client: httpx.AsyncClient) -> None:
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    response = await client.post(endpoint, json=payload)
                    elapsed = (time.perf_counter() - t0) * 1000.0
                    if 200 <= response.status_code < 400:
                        latencies_ms.append(elapsed)
                    else:
                        errors += 1
                except Exception as exc:  # bench tolerates any exception
                    errors += 1
                    logger.debug("httpx benchmark request failed: %s", exc)

        # Deliberately bypasses nim_client/resilient_request (retries and
        # pacing would corrupt latency percentiles) but still carries the
        # Blueprint source header every outbound NIM request must have.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(deadline_s), headers=NIM_DEFAULT_HEADERS
        ) as client:
            tasks = [_one_request(client) for _ in range(request_count)]
            await asyncio.gather(*tasks, return_exceptions=False)

        p50, p90, p99 = _percentiles(latencies_ms)
        # Zero successful requests (every request errored / timed out) is a
        # non-measurement, not a genuine zero-latency result — mark it failed
        # so the sweep records the concurrency skipped instead of persisting
        # fake zeros, matching the genai-perf failure path.
        result = BenchmarkResult(
            concurrency=concurrency,
            latency_p50_ms=p50,
            latency_p90_ms=p90,
            latency_p99_ms=p99,
            ttft_p50_ms=None,
            ttft_p90_ms=None,
            itl_p50_ms=None,
            itl_p90_ms=None,
            request_count=len(latencies_ms),
            error_count=errors,
            prometheus={},
            artifact_dir=str(artifact_dir),
            driver=self.driver_name,
            failed=not latencies_ms,
        )

        # Persist the full result.json so reruns are idempotent and the
        # benchmark UI / audit trail can read the raw numbers.
        (artifact_dir / "result.json").write_text(
            json.dumps(result.to_json(), indent=2, sort_keys=True)
        )
        return result


# ── GenaiPerfAdapter (primary) ────────────────────────────────────────────────


class GenaiPerfAdapter:
    """Subprocess invocation of ``genai-perf``.

    Command:
      genai-perf profile \\
        --service-kind openai \\
        --endpoint-type chat \\
        --url {base_url} \\
        --model {model} \\
        --concurrency {c} \\
        --request-count {n} \\
        --output-format json \\
        --artifact-dir {artifact_dir}

    Parses ``profile_export_genai_perf.json`` (genai-perf's standard
    output filename). The parser is forgiving: any missing field defaults
    to 0 / None so the ``BenchmarkResult`` schema is always populated.
    """

    driver_name = "genai-perf"

    async def run(
        self,
        *,
        base_url: str,
        model: str,
        concurrency: int,
        project_dir: str,
        student_model_id: str,
        request_count: int = 100,
        deadline_s: float = 1200.0,
    ) -> BenchmarkResult:
        artifact_dir = (
            Path(project_dir) / "benchmarks" / student_model_id / str(concurrency)
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "genai-perf",
            "profile",
            "--service-kind",
            "openai",
            "--endpoint-type",
            "chat",
            "--url",
            base_url.rstrip("/"),
            "--model",
            model,
            "--concurrency",
            str(concurrency),
            "--request-count",
            str(request_count),
            "--output-format",
            "json",
            "--artifact-dir",
            str(artifact_dir),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=deadline_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        return _parse_genai_perf_output(
            artifact_dir=artifact_dir,
            concurrency=concurrency,
            return_code=proc.returncode,
        )


def _parse_genai_perf_output(
    artifact_dir: Path,
    concurrency: int,
    return_code: int | None,
) -> BenchmarkResult:
    """Parse genai-perf's ``profile_export_genai_perf.json`` output.

    The exact schema varies between genai-perf releases but consistently
    exposes a ``request_latency`` block with ``p50``/``p90``/``p99``
    (milliseconds) and a ``time_to_first_token`` / ``inter_token_latency``
    block with the same shape. Missing fields → 0.0 / None.
    """
    output_path = artifact_dir / "profile_export_genai_perf.json"
    if return_code != 0 or not output_path.exists():
        logger.warning(
            "genai-perf produced no usable output at concurrency=%d "
            "(return_code=%s, output_exists=%s)",
            concurrency,
            return_code,
            output_path.exists(),
        )
        return BenchmarkResult(
            concurrency=concurrency,
            latency_p50_ms=0.0,
            latency_p90_ms=0.0,
            latency_p99_ms=0.0,
            ttft_p50_ms=None,
            ttft_p90_ms=None,
            itl_p50_ms=None,
            itl_p90_ms=None,
            request_count=0,
            error_count=0,
            prometheus={},
            artifact_dir=str(artifact_dir),
            driver=GenaiPerfAdapter.driver_name,
            failed=True,
        )

    raw = json.loads(output_path.read_text())

    def _ms(node: dict[str, Any] | None, key: str) -> float | None:
        if not isinstance(node, dict):
            return None
        value = node.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    latency_raw: Any = raw.get("request_latency") or raw.get("e2e_latency") or {}
    ttft_raw: Any = raw.get("time_to_first_token") or {}
    itl_raw: Any = raw.get("inter_token_latency") or {}
    latency: dict[str, Any] = (
        cast("dict[str, Any]", latency_raw) if isinstance(latency_raw, dict) else {}
    )
    ttft: dict[str, Any] = (
        cast("dict[str, Any]", ttft_raw) if isinstance(ttft_raw, dict) else {}
    )
    itl: dict[str, Any] = (
        cast("dict[str, Any]", itl_raw) if isinstance(itl_raw, dict) else {}
    )

    p50 = _ms(latency, "p50") or 0.0
    p90 = _ms(latency, "p90") or 0.0
    p99 = _ms(latency, "p99") or 0.0

    request_count = int(raw.get("request_count", 0) or 0)
    error_count = int(raw.get("error_count", 0) or 0)

    return BenchmarkResult(
        concurrency=concurrency,
        latency_p50_ms=p50,
        latency_p90_ms=p90,
        latency_p99_ms=p99,
        ttft_p50_ms=_ms(ttft, "p50"),
        ttft_p90_ms=_ms(ttft, "p90"),
        itl_p50_ms=_ms(itl, "p50"),
        itl_p90_ms=_ms(itl, "p90"),
        request_count=request_count,
        error_count=error_count,
        prometheus={},
        artifact_dir=str(artifact_dir),
        driver=GenaiPerfAdapter.driver_name,
    )


# ── Adapter selection ─────────────────────────────────────────────────────────


def select_adapter() -> BenchmarkAdapter:
    """Choose the best available benchmark adapter.

    Returns ``GenaiPerfAdapter`` when ``genai_perf`` is importable; falls
    back to ``HttpxAdapter`` otherwise. Tests monkeypatch this function or
    inject ``sys.modules["genai_perf"]`` to exercise both paths.
    """
    try:
        import genai_perf  # noqa: F401  # pyright: ignore[reportUnusedImport, reportMissingImports] — optional dep, capability probe only
    except ImportError:
        logger.info(
            "genai-perf not available; using httpx fallback for NIM benchmarks. "
            "Install via `pip install vlm-feedback-loop[benchmarking]`."
        )
        return HttpxAdapter()
    return GenaiPerfAdapter()

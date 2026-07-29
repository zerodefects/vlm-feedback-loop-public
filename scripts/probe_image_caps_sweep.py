#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-cap sweep probe — find the real ``max_images_per_request`` per endpoint.

Complements (does not replace) ``scripts/probe_hosted_image_caps.py``:
the hosted probe is the seeded-catalog regression check, while this sweep
discovers caps against *any* OpenAI-compatible chat/completions endpoint —
hosted NIM, local NIM (deployed via
``POST /v1/projects/{id}/local_nim/deploy``), self-hosted NIM, or anything
else that takes ``image_url`` content parts.

Why this script exists
======================

The in-process ``_probe_image_cap_support`` (see
``services/model_config_service.py``) only verifies "is the seeded
``max_images_per_request`` value correct?" by sending N + N+1. When both
succeed (the seeded value is conservative), it logs a hint and returns
``"supported"`` — but doesn't sweep upward to find the true cap. Migration
012 is the precedent: live probing turned up that hosted Mistral's real
cap was 8 (seeded 5), Nemotron's was 10 (seeded 5), and Qwen's was 8
(seeded 10 — actively wrong).

Cosmos Reason2 2B and 8B are both seeded ``max_images_per_request: 8``
with ``image_cap_support: "unknown"`` — that 8 was inferred, never
live-probed (the hosted Cosmos endpoints are NVCF-account-gated so
``probe_hosted_image_caps.py`` couldn't reach them). When Cosmos runs
locally, the cap could reflect any of three layers:

* ``build.nvidia.com`` gateway clamp — only applies to hosted (typically
  emits ``"At most N image(s) may be provided in one prompt"``).
* NIM container clamp — would persist regardless of where NIM runs
  (typically references a NIM env var or NIM server log line).
* Cosmos Reason2 model clamp — would persist regardless of NIM (typically
  emits a vLLM ``MaxImagesExceeded`` style message, or a context-window
  rejection from the multimodal preprocessor).

Sweeping a wide ladder against local NIM, then comparing the failure
error string against the hosted error string (when both are reachable),
attributes the cap to its layer.

Method
======

For each model, POST chat/completions with N copies of the same 512×512
synthetic probe image and a trivial text instruction. Walk N upwards and
record:

* the largest N that returned HTTP 200 (``cap``)
* the smallest N that failed (``first_failure``) plus the HTTP status
  and the first 240 chars of the error body (the layer-attribution
  signal)
* per-N latency

Stops on the first 4xx (assume monotonic). Treats network errors as
inconclusive (records and continues). Like the hosted probe, runs with
``response_format`` OFF so the cap reported is the raw image-input cap,
not "image cap when also requesting structured generation."

Usage
=====

::

    # Local NIM (no auth — localhost only):
    uv run python scripts/probe_image_caps_sweep.py \\
      --base-url http://127.0.0.1:8001/v1 \\
      --model nvidia/cosmos-reason2-2b

    # Hosted NIM (NVIDIA_API_KEY from env):
    uv run python scripts/probe_image_caps_sweep.py \\
      --base-url https://integrate.api.nvidia.com/v1 \\
      --model nvidia/cosmos-reason2-8b \\
      --api-key-env NVIDIA_API_KEY

    # Custom ladder + output JSON:
    uv run python scripts/probe_image_caps_sweep.py \\
      --base-url http://127.0.0.1:8001/v1 \\
      --model nvidia/cosmos-reason2-2b \\
      --ladder 1,4,8,12,16,24,32,48,64 \\
      --json-out /path/to/sweep_result.json

Exits 0 on a clean sweep (any N succeeded), 1 on no successes (every N
failed — typically auth or wrong base URL), 2 on argument validation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

# Make backend package importable from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "backend"))

from vlm_feedback_loop.services.model_config_service import (  # noqa: E402
    generate_probe_image_data_url,
)

DEFAULT_LADDER = [1, 4, 8, 12, 16, 24, 32, 48, 64]
DEADLINE_S = 120.0  # local NIM may be cold; allow generous deadline


@dataclass
class ProbeAttempt:
    n_images: int
    http_status: int  # -1 on transport error
    latency_ms: int
    error_excerpt: str | None  # first 240 chars of error body if non-2xx


@dataclass
class SweepResult:
    base_url: str
    model_name: str
    ladder: list[int]
    cap: int  # largest N with HTTP 200; 0 if none succeeded
    first_failure: int | None  # smallest N with non-2xx
    failure_status: int | None
    failure_excerpt: str | None
    attempts: list[ProbeAttempt]
    total_wall_ms: int

    def to_dict(self) -> dict:
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "ladder": self.ladder,
            "cap": self.cap,
            "first_failure": self.first_failure,
            "failure_status": self.failure_status,
            "failure_excerpt": self.failure_excerpt,
            "attempts": [asdict(a) for a in self.attempts],
            "total_wall_ms": self.total_wall_ms,
        }


def _resolve_api_key(api_key_env: str | None) -> str | None:
    """Read the API key from the named env var, then ~/.vlm_feedback_loop/.env.

    Returns None when ``api_key_env`` is None (anonymous local NIM).
    Exits 2 when ``api_key_env`` is named but unset everywhere.
    """
    if api_key_env is None:
        return None
    key = os.environ.get(api_key_env)
    if key:
        return key
    env_path = Path.home() / ".vlm_feedback_loop" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{api_key_env}="):
                return line.split("=", 1)[1].strip()
    print(
        f"ERROR: {api_key_env} not found in env or ~/.vlm_feedback_loop/.env",
        file=sys.stderr,
    )
    sys.exit(2)


async def _probe_one_count(
    client: httpx.AsyncClient,
    base_url: str,
    model_name: str,
    n_images: int,
    auth: dict[str, str],
    probe_image_data_url: str,
) -> ProbeAttempt:
    """Send a chat/completions request with ``n_images`` images.

    ``response_format`` is intentionally omitted — we want the raw
    image-input cap, not the cap-when-also-requesting-json_schema.
    """
    content: list[dict] = [{"type": "text", "text": "Respond with: ok"}]
    for _ in range(n_images):
        content.append(
            {"type": "image_url", "image_url": {"url": probe_image_data_url}}
        )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=auth,
            timeout=DEADLINE_S,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        if r.status_code == 200:
            return ProbeAttempt(
                n_images=n_images,
                http_status=200,
                latency_ms=latency_ms,
                error_excerpt=None,
            )
        excerpt = (r.text or "")[:240].replace("\n", " ")
        return ProbeAttempt(
            n_images=n_images,
            http_status=r.status_code,
            latency_ms=latency_ms,
            error_excerpt=excerpt,
        )
    except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ProbeAttempt(
            n_images=n_images,
            http_status=-1,
            latency_ms=latency_ms,
            error_excerpt=f"{type(exc).__name__}: {exc}"[:240],
        )


async def _sweep(
    base_url: str,
    model_name: str,
    ladder: list[int],
    auth: dict[str, str],
) -> SweepResult:
    print(f"\n→ {model_name} via {base_url}", flush=True)
    print(f"   ladder: {ladder}", flush=True)
    probe_image_data_url = generate_probe_image_data_url()
    attempts: list[ProbeAttempt] = []
    cap = 0
    first_failure: int | None = None
    failure_status: int | None = None
    failure_excerpt: str | None = None
    t_start = time.monotonic()

    async with httpx.AsyncClient() as client:
        for n in ladder:
            attempt = await _probe_one_count(
                client, base_url, model_name, n, auth, probe_image_data_url
            )
            attempts.append(attempt)
            verdict = (
                "OK" if attempt.http_status == 200 else f"FAIL {attempt.http_status}"
            )
            excerpt_str = (
                f"  {attempt.error_excerpt[:120]}" if attempt.error_excerpt else ""
            )
            print(
                f"   N={n:>3}  {verdict}  ({attempt.latency_ms} ms){excerpt_str}",
                flush=True,
            )
            if attempt.http_status == 200:
                cap = n
            elif attempt.http_status == -1:
                # Transport error — inconclusive. Stop, but don't classify
                # as a "first_failure" (we can't say cap is N-1 if the
                # network ate the request).
                if first_failure is None:
                    first_failure = n
                    failure_status = -1
                    failure_excerpt = attempt.error_excerpt
                break
            else:
                if first_failure is None:
                    first_failure = n
                    failure_status = attempt.http_status
                    failure_excerpt = attempt.error_excerpt
                # Once we hit a 4xx, stop — assume monotonic.
                break

    total_wall_ms = int((time.monotonic() - t_start) * 1000)
    return SweepResult(
        base_url=base_url,
        model_name=model_name,
        ladder=ladder,
        cap=cap,
        first_failure=first_failure,
        failure_status=failure_status,
        failure_excerpt=failure_excerpt,
        attempts=attempts,
        total_wall_ms=total_wall_ms,
    )


def _print_summary(result: SweepResult) -> None:
    print("\n" + "─" * 80)
    print(f"Image-cap sweep — {result.model_name}")
    print(f"endpoint: {result.base_url}")
    print(f"ladder: {result.ladder}")
    print(f"cap (largest N with 200): {result.cap}")
    if result.first_failure is not None:
        print(
            f"first_failure: N={result.first_failure}, status={result.failure_status}"
        )
        if result.failure_excerpt:
            print(f"failure_excerpt: {result.failure_excerpt}")
    else:
        print(
            "no failures — every N in ladder succeeded; consider extending ladder upward."
        )
    print(f"total wall: {result.total_wall_ms / 1000:.1f}s")
    print("─" * 80)


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="OpenAI-compatible chat/completions base URL (e.g. http://127.0.0.1:8001/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name as the endpoint expects it (e.g. nvidia/cosmos-reason2-2b)",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Env var holding the bearer token. Omit for unauthenticated local NIM.",
    )
    parser.add_argument(
        "--ladder",
        type=str,
        default=",".join(str(n) for n in DEFAULT_LADDER),
        help=(
            "Comma list of image counts to probe (ascending). Default: "
            + ",".join(str(n) for n in DEFAULT_LADDER)
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the full SweepResult as JSON to this path.",
    )
    args = parser.parse_args(argv)

    ladder = [int(n) for n in args.ladder.split(",") if n.strip()]
    if not ladder or sorted(ladder) != ladder or ladder[0] < 1:
        print("ERROR: --ladder must be ascending positive integers", file=sys.stderr)
        return 2

    auth: dict[str, str] = {"Content-Type": "application/json"}
    if args.api_key_env:
        api_key = _resolve_api_key(args.api_key_env)
        if api_key:
            auth["Authorization"] = f"Bearer {api_key}"

    result = await _sweep(args.base_url, args.model, ladder, auth)
    _print_summary(result)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nWrote {args.json_out}")

    return 0 if result.cap > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(sys.argv[1:])))

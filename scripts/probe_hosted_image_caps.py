#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live multi-image probe — find the real ``max_images_per_request`` per hosted Teacher.

Background
==========

The seeded ``model_configs.max_images_per_request`` values started as
conservative guesses that were never live-tested, and hosted vendors
can change their limits. This script runs an N-image probe against
every hosted Teacher to verify each Teacher's seeded cap still matches
the live API and to size any future correction migration if a vendor
changes their limit.

Method
======

For each Teacher with ``supports_image_input=True``, POST chat/completions
with N copies of the same 512×512 synthetic probe image and a trivial
text instruction. Walk N upwards (1, 2, 4, 6, 7, 8, 9, 10, 12) and record
the largest N that returns HTTP 200. The first N that returns HTTP 400
(or any 4xx) is treated as ``cap = N - 1``.

We probe with **structured generation OFF** so the cap we report is the
raw image-input cap, not "image cap when also requesting json_schema."
The ICL pruning cap subtracts 1 for the query image, so the
seeded `max_images_per_request` value should be the image budget per
request (not "max ICL examples").

Usage
=====

::

    NVIDIA_API_KEY=nvapi-... uv run python scripts/probe_hosted_image_caps.py
    uv run python scripts/probe_hosted_image_caps.py --models qwen,mistral
    uv run python scripts/probe_hosted_image_caps.py --ladder 1,5,8,10

Exits 0 with a one-line-per-model report. Mismatches against the
currently-seeded values (derived from
``services.project_service.SEEDED_MODEL_CATALOG``, the canonical model
catalog that the project-creation path consumes) are flagged as
``MISMATCH`` so the next correction migration can be sized accurately.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

# Make backend package importable from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "backend"))

from vlm_feedback_loop.services.model_config_service import (  # noqa: E402
    generate_probe_image_data_url,
)
from vlm_feedback_loop.services.project_service import (  # noqa: E402
    SEEDED_MODEL_CATALOG,
)

HOSTED_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEADLINE_S = 60.0

# Currently-seeded ``max_images_per_request`` per model, derived from the
# canonical model catalog (``services/project_service.py``). Single
# source of truth — when ``SEEDED_MODEL_CATALOG`` adds a model, corrects
# a cap, or removes a model, this dict tracks automatically and the
# probe verdict stays accurate.
CURRENT_SEEDED_CAPS: dict[str, int] = {
    entry["model_name"]: entry["max_images_per_request"]
    for entry in SEEDED_MODEL_CATALOG
}

# Default models to probe — the hosted Teachers with `supports_image_input=True`.
# Cosmos Reason 2 is NVCF-gated on hosted (per README "Current Status"),
# so it's omitted by default; pass ``--models cosmos_8b`` to include it
# (will report 404 if your account doesn't have access).
DEFAULT_MODEL_LABELS = ["qwen", "kimi", "mistral", "nemotron"]

MODEL_LOOKUP = {
    "qwen": "qwen/qwen3.5-397b-a17b",
    "kimi": "moonshotai/kimi-k2-thinking",
    "mistral": "mistralai/mistral-large-3-675b-instruct-2512",
    "nemotron": "nvidia/nemotron-nano-12b-v2-vl",
    "cosmos_8b": "nvidia/cosmos-reason2-8b",
    "cosmos_2b": "nvidia/cosmos-reason2-2b",
}

# Default probe ladder. Picks values around the seeded caps so the cap
# is bracketed for every seeded Teacher (5, 8, 10).
DEFAULT_LADDER = [1, 2, 4, 6, 7, 8, 9, 10, 12]


@dataclass
class CapResult:
    model_name: str
    cap: int  # largest N with HTTP 200
    first_failure: int | None  # smallest N with HTTP 400+
    failure_status: int | None
    failure_excerpt: str | None
    seeded: int | None
    timings_ms: dict[int, int]  # N -> latency for that probe

    @property
    def matches_seeded(self) -> bool:
        return self.seeded is None or self.cap == self.seeded


def _resolve_api_key() -> str:
    """Read NVIDIA_API_KEY from env or ~/.vlm_feedback_loop/.env."""
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key
    env_path = Path.home() / ".vlm_feedback_loop" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("NVIDIA_API_KEY="):
                return line.split("=", 1)[1].strip()
    print(
        "ERROR: NVIDIA_API_KEY not found in env or ~/.vlm_feedback_loop/.env",
        file=sys.stderr,
    )
    sys.exit(2)


async def _probe_one_count(
    client: httpx.AsyncClient,
    model_name: str,
    n_images: int,
    auth: dict[str, str],
    probe_image_data_url: str,
) -> tuple[int, str | None, int]:
    """Send a chat/completions request with ``n_images`` images.

    Returns ``(http_status, error_excerpt or None, latency_ms)``.
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
            f"{HOSTED_BASE_URL}/chat/completions",
            json=payload,
            headers=auth,
            timeout=DEADLINE_S,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        if r.status_code == 200:
            return 200, None, latency_ms
        excerpt = (r.text or "")[:160].replace("\n", " ")
        return r.status_code, excerpt, latency_ms
    except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        return -1, f"{type(exc).__name__}: {exc}", latency_ms


async def _probe_model(
    client: httpx.AsyncClient,
    model_name: str,
    ladder: list[int],
    auth: dict[str, str],
    probe_image_data_url: str,
) -> CapResult:
    print(f"\n→ {model_name}", flush=True)
    cap = 0
    first_failure = None
    failure_status = None
    failure_excerpt = None
    timings: dict[int, int] = {}

    for n in ladder:
        status, excerpt, latency_ms = await _probe_one_count(
            client, model_name, n, auth, probe_image_data_url
        )
        timings[n] = latency_ms
        verdict = "OK" if status == 200 else f"FAIL {status}"
        excerpt_str = f"  {excerpt[:80]}" if excerpt else ""
        print(f"   N={n:>2}  {verdict}  ({latency_ms} ms){excerpt_str}", flush=True)
        if status == 200:
            cap = n
        else:
            if first_failure is None:
                first_failure = n
                failure_status = status
                failure_excerpt = excerpt
            # Once we hit a failure, stop — assume monotonic.
            break

    return CapResult(
        model_name=model_name,
        cap=cap,
        first_failure=first_failure,
        failure_status=failure_status,
        failure_excerpt=failure_excerpt,
        seeded=CURRENT_SEEDED_CAPS.get(model_name),
        timings_ms=timings,
    )


def _print_summary(results: list[CapResult]) -> bool:
    """Render the report. Returns True if ALL models match seeded values."""
    print("\n" + "─" * 80)
    print("Hosted-NIM image-cap probe — summary")
    print(f"{'model':<48} {'observed':>9} {'seeded':>7} {'verdict':>9}")
    print("─" * 80)
    all_match = True
    for r in results:
        if r.seeded is None:
            verdict = "—"
        elif r.cap == r.seeded:
            verdict = "OK"
        else:
            verdict = "MISMATCH"
            all_match = False
        print(
            f"{r.model_name:<48} {r.cap:>9} "
            f"{(r.seeded if r.seeded is not None else '—'):>7} "
            f"{verdict:>9}"
        )
    print("─" * 80)
    print(
        f"Overall: {'OK' if all_match else 'MISMATCH (a correction migration may be needed)'}"
    )
    print("─" * 80)
    return all_match


async def amain(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_LABELS),
        help=(
            "Comma list of model labels. Default: "
            + ",".join(DEFAULT_MODEL_LABELS)
            + ". Available: "
            + ",".join(MODEL_LOOKUP.keys())
        ),
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
    args = parser.parse_args(argv)

    requested = [t.strip() for t in args.models.split(",") if t.strip()]
    unknown = [t for t in requested if t not in MODEL_LOOKUP]
    if unknown:
        print(f"ERROR: unknown model labels: {unknown}", file=sys.stderr)
        return 2
    model_names = [MODEL_LOOKUP[t] for t in requested]

    ladder = [int(n) for n in args.ladder.split(",") if n.strip()]
    if not ladder or sorted(ladder) != ladder:
        print(
            "ERROR: --ladder must be ascending positive integers",
            file=sys.stderr,
        )
        return 2

    api_key = _resolve_api_key()
    auth = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    probe_image_data_url = generate_probe_image_data_url()

    results: list[CapResult] = []
    async with httpx.AsyncClient() as client:
        for name in model_names:
            results.append(
                await _probe_model(client, name, ladder, auth, probe_image_data_url)
            )

    return 0 if _print_summary(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain(sys.argv[1:])))

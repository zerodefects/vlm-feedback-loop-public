#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture live-TAO wire-format fixtures.

Idempotent capture: running this script twice produces byte-identical
output files (the OpenAPI spec, empty-state responses, and any
per-sample predictions from completed evaluate jobs).

Usage::

    # Requires tunnel established; TAO config in ~/.vlm_feedback_loop/.env.
    python scripts/capture_tao_fixtures.py [--tao-job-id UUID]

Without ``--tao-job-id``: captures the OpenAPI spec, empty-state
responses, and version — the ambient wire-format fixtures that pin
the TAO contract the Blueprint is built against.

With ``--tao-job-id``: additionally fetches per-sample-predictions from
that completed evaluate TAOJob and writes them under
``tests/fixtures/tao/per_sample_predictions/{job_id}.json``. This is
the full fixture set, complete only when a real training chain has
produced an evaluate job.

All writes are atomic (write to ``.part`` then rename) and idempotent
(running twice yields the same content). Exits non-zero if live TAO
is unreachable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("capture_tao_fixtures")


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tao"


def _atomic_write(path: Path, data: bytes) -> None:
    """Atomic on-disk write (write to ``.part``, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


async def _download_tao_file_bytes(
    tao_external_job_id: str,
    file_name: str,
    local_path: Path,
    *,
    settings,
) -> dict:
    """Download one TAO artifact by name via ``:download_selective_files``.

    Fixture-capture only. The production artifact path reads bytes directly
    from the workspace S3 bucket instead: FTMS 6.26.3 rejects POST on
    ``:download_selective_files`` (405) and its GET variant requires
    ``best_model``/``latest_model`` aliases that cosmos-rl does not produce.

    Posts a single-element ``files`` list. Accepts either an inline binary
    response (streamed to disk) or a JSON response carrying a signed URL
    that is followed with a second GET. Returns ``{"success": bool,
    "local_path": str | None, "error": str | None, "bytes_written": int}``.
    """
    import httpx

    from vlm_feedback_loop.services.nim_client import NIM_DEFAULT_HEADERS
    from vlm_feedback_loop.services.runtime_secrets import get_effective_secret
    from vlm_feedback_loop.services.tao_auth import get_tao_bearer

    if not (
        settings.TAO_API_BASE_URL
        and get_effective_secret("TAO_API_KEY", settings)
        and settings.TAO_ORG_NAME
    ):
        return {
            "success": False,
            "local_path": None,
            "error": "TAO configuration incomplete",
            "bytes_written": 0,
        }

    base = settings.TAO_API_BASE_URL.rstrip("/")
    url = (
        f"{base}/orgs/{settings.TAO_ORG_NAME}/jobs/{tao_external_job_id}"
        f":download_selective_files"
    )
    try:
        bearer = await get_tao_bearer(settings)
    except RuntimeError as exc:
        return {
            "success": False,
            "local_path": None,
            "error": f"TAO authentication failed: {exc}",
            "bytes_written": 0,
        }
    headers = {
        **NIM_DEFAULT_HEADERS,
        "Authorization": f"Bearer {bearer}",
    }
    body = {"files": [file_name]}

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    deadline_s = float(settings.HTTP_DEADLINE_BACKGROUND_S)

    try:
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(deadline_s)) as client,
            client.stream("POST", url, headers=headers, json=body) as response,
        ):
            if response.status_code >= 400:
                err_text = await response.aread()
                return {
                    "success": False,
                    "local_path": None,
                    "error": f"download failed HTTP {response.status_code}: "
                    f"{err_text[:256]!r}",
                    "bytes_written": 0,
                }
            content_type = (response.headers.get("content-type") or "").lower()

            # Branch 1: JSON with a signed URL.
            if "application/json" in content_type:
                payload_bytes = await response.aread()
                try:
                    payload = json.loads(payload_bytes)
                except Exception:
                    return {
                        "success": False,
                        "local_path": None,
                        "error": "download: JSON response malformed",
                        "bytes_written": 0,
                    }
                signed_url = None
                if isinstance(payload, dict):
                    signed_url = (
                        payload.get("url")
                        or payload.get("download_url")
                        or payload.get("signed_url")
                    )
                if not signed_url:
                    return {
                        "success": False,
                        "local_path": None,
                        "error": "download: JSON response missing signed URL",
                        "bytes_written": 0,
                    }
                # The signed URL carries its own signature — plain GET.
                written = 0
                async with client.stream("GET", signed_url) as signed_resp:
                    if signed_resp.status_code >= 400:
                        err_text = await signed_resp.aread()
                        return {
                            "success": False,
                            "local_path": None,
                            "error": (
                                f"signed URL download failed "
                                f"HTTP {signed_resp.status_code}: "
                                f"{err_text[:256]!r}"
                            ),
                            "bytes_written": 0,
                        }
                    with open(tmp_path, "wb") as fh:
                        async for chunk in signed_resp.aiter_bytes():
                            fh.write(chunk)
                            written += len(chunk)
                tmp_path.replace(local_path)
                return {
                    "success": True,
                    "local_path": str(local_path),
                    "error": None,
                    "bytes_written": written,
                }

            # Branch 2: inline binary stream.
            written = 0
            with open(tmp_path, "wb") as fh:
                async for chunk in response.aiter_bytes():
                    fh.write(chunk)
                    written += len(chunk)
            tmp_path.replace(local_path)
            return {
                "success": True,
                "local_path": str(local_path),
                "error": None,
                "bytes_written": written,
            }
    except httpx.TimeoutException:
        tmp_path.unlink(missing_ok=True)
        return {
            "success": False,
            "local_path": None,
            "error": "download: request timed out",
            "bytes_written": 0,
        }
    except httpx.HTTPError as exc:
        tmp_path.unlink(missing_ok=True)
        return {
            "success": False,
            "local_path": None,
            "error": f"download: HTTP error: {exc}",
            "bytes_written": 0,
        }
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        return {
            "success": False,
            "local_path": None,
            "error": f"download: file IO error: {exc}",
            "bytes_written": 0,
        }


async def _capture_ambient(settings) -> None:
    """Capture OpenAPI spec, version, and empty-state endpoints."""
    from vlm_feedback_loop.services.http_client import resilient_request
    from vlm_feedback_loop.services.nim_client import NIM_DEFAULT_HEADERS

    base = settings.TAO_API_BASE_URL.rstrip("/")
    headers = {
        **NIM_DEFAULT_HEADERS,
        "Authorization": f"Bearer {settings.TAO_API_KEY}",
    }

    targets = [
        (f"{base}/openapi.json", "openapi_v2.json"),
        (f"{base}/version", "ftms_version.json"),
        (
            f"{base}/orgs/{settings.TAO_ORG_NAME}/jobs?limit=5",
            "jobs_empty_response.json",
        ),
        (
            f"{base}/orgs/{settings.TAO_ORG_NAME}/workspaces",
            "workspaces_empty_response.json",
        ),
        (
            f"{base}/orgs/{settings.TAO_ORG_NAME}/datasets",
            "datasets_empty_response.json",
        ),
    ]
    for url, name in targets:
        logger.info("capturing %s → %s", url, name)
        result = await resilient_request(
            "GET",
            url,
            deadline_s=30.0,
            max_retries=1,
            headers=headers,
        )
        if result.error_class is not None:
            logger.warning(
                "skipping %s: %s (status=%s)",
                name,
                result.error_detail,
                result.status_code,
            )
            continue
        if isinstance(result.body, dict | list):
            content = (
                json.dumps(result.body, indent=2, sort_keys=True).encode("utf-8")
                + b"\n"
            )
        else:
            content = str(result.body).encode("utf-8")
        _atomic_write(FIXTURE_DIR / name, content)


async def _capture_predictions(settings, tao_job_id: str) -> None:
    """Capture per-sample-predictions from a completed evaluate TAOJob.

    NOTE: The live TAO FTMS v6.25.11 ``:download_selective_files`` endpoint
    is GET-only and returns a single binary artifact bundle — NOT the
    POST + JSON-metadata shape the Blueprint's internal code expects.
    Newer TAO releases (the Blueprint pins 6.26.3) may accept POST.
    This script probes both shapes so the capture succeeds across
    versions.
    """
    out_dir = FIXTURE_DIR / "per_sample_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tao_job_id}.bin"

    result = await _download_tao_file_bytes(
        tao_job_id,
        "per_sample_predictions",
        out_path,
        settings=settings,
    )
    if result["success"]:
        logger.info(
            "captured predictions: %s (%d bytes)",
            out_path,
            result["bytes_written"],
        )
    else:
        logger.error("prediction capture failed: %s", result["error"])


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tao-job-id",
        help="Capture per-sample predictions from this evaluate job id.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    from vlm_feedback_loop.config import get_settings

    settings = get_settings()
    if not (
        settings.TAO_API_BASE_URL and settings.TAO_API_KEY and settings.TAO_ORG_NAME
    ):
        logger.error(
            "TAO not configured — set TAO_API_BASE_URL / TAO_API_KEY / "
            "TAO_ORG_NAME via ~/.vlm_feedback_loop/.env"
        )
        return 2

    await _capture_ambient(settings)
    if args.tao_job_id:
        await _capture_predictions(settings, args.tao_job_id)

    logger.info("capture complete; fixtures in %s", FIXTURE_DIR)
    return 0


def main() -> int:
    args = _cli()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

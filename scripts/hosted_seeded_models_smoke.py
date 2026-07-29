#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live hosted-NIM smoke — verifies the seeded hosted models reach
``https://integrate.api.nvidia.com/v1``.

This script hits the real hosted API — requires a valid
``NVIDIA_API_KEY`` in the environment or in
``~/.vlm_feedback_loop/.env``.

For each model in ``SEEDED_MODELS`` below:

1. **list_models** probe — confirms the model appears in the hosted catalog.
2. **Text-only chat/completions** — minimal ``Reply with OK.`` prompt;
   asserts HTTP 200 + non-empty content.
3. **Image-enabled chat/completions** — for ``supports_image_input=true``
   models, sends the same prompt plus a 512×512 RGB PNG probe image;
   asserts HTTP 200.

Then runs the three capability probes against ``nvidia/cosmos-reason2-8b``
and ``nvidia/cosmos-reason2-2b`` (the user's primary target) and reports
resolved capability status.

Exit 0 on all-green. Exit 1 on any failure. The script prints a concise
one-line summary per model plus a final verdict line.

Usage::

    NVIDIA_API_KEY=nvapi-... uv run python scripts/hosted_seeded_models_smoke.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the backend package importable when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src" / "backend"))

from vlm_feedback_loop.services import nim_client  # noqa: E402
from vlm_feedback_loop.services.model_config_service import (  # noqa: E402
    generate_probe_image_data_url,
    probe_structured_generation,
    probe_thinking_toggle,
    probe_visual_budget,
)
from vlm_feedback_loop.services.project_service import (  # noqa: E402
    SEEDED_MODEL_CATALOG,
)

HOSTED_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEADLINE_S = 60.0


@dataclass
class SeededModel:
    name: str
    supports_image_input: bool
    thinking_toggle_mode: (
        str  # none | always_on_reasoning | qwen_enable_thinking | kimi_thinking
    )
    visual_budget_mode: (
        str  # none | mm_processor_size | mm_processor_tiles | mm_processor_pixels
    )


# Which catalog models this smoke covers (a smoke-scope decision); the
# capability fields for each are derived from the canonical
# ``services/project_service.py::SEEDED_MODEL_CATALOG`` so they can never
# drift from the backend.
_COVERED_MODEL_NAMES = (
    "nvidia/cosmos-reason2-8b",
    "nvidia/cosmos-reason2-2b",
    "minimaxai/minimax-m3",
    "stepfun-ai/step-3.7-flash",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "nvidia/nemotron-nano-12b-v2-vl",
)
_CATALOG_BY_NAME = {entry["model_name"]: entry for entry in SEEDED_MODEL_CATALOG}
_MISSING_FROM_CATALOG = [
    name for name in _COVERED_MODEL_NAMES if name not in _CATALOG_BY_NAME
]
if _MISSING_FROM_CATALOG:
    raise SystemExit(
        "hosted_seeded_models_smoke: models no longer in SEEDED_MODEL_CATALOG "
        f"(update _COVERED_MODEL_NAMES): {_MISSING_FROM_CATALOG}"
    )
SEEDED_MODELS: list[SeededModel] = [
    SeededModel(
        name=name,
        supports_image_input=bool(_CATALOG_BY_NAME[name]["supports_image_input"]),
        thinking_toggle_mode=str(_CATALOG_BY_NAME[name]["thinking_toggle_mode"]),
        visual_budget_mode=str(_CATALOG_BY_NAME[name]["visual_budget_mode"]),
    )
    for name in _COVERED_MODEL_NAMES
]


def _load_api_key() -> str | None:
    """Resolve NVIDIA_API_KEY from env or ~/.vlm_feedback_loop/.env."""
    env_key = os.environ.get("NVIDIA_API_KEY")
    if env_key:
        return env_key

    dotenv_path = Path.home() / ".vlm_feedback_loop" / ".env"
    if not dotenv_path.exists():
        return None

    for line in dotenv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("NVIDIA_API_KEY="):
            val = line.split("=", 1)[1].strip()
            # Strip matching surrounding quotes, if any.
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            return val or None
    return None


async def probe_list(auth_headers: dict[str, str]) -> set[str]:
    """Return the set of model IDs returned by GET {base}/models."""
    result = await nim_client.list_models(
        base_url=HOSTED_BASE_URL,
        auth_headers=auth_headers,
        deadline_s=DEADLINE_S,
        max_retries=1,
    )
    if not result.success or result.models is None:
        print(f"  list_models failed: status={result.status_code} err={result.error}")
        return set()
    return set(result.models)


async def probe_text(model: str, auth_headers: dict[str, str]) -> tuple[bool, str]:
    # max_tokens=128 gives thinking models (Cosmos Reason2, always-on
    # reasoners like Step 3.7 Flash) room to finish their reasoning +
    # answer. Non-thinking models ignore the surplus budget.
    result = await nim_client.chat_completions(
        base_url=HOSTED_BASE_URL,
        auth_headers=auth_headers,
        model=model,
        messages=[{"role": "user", "content": "Reply with OK."}],
        deadline_s=DEADLINE_S,
        max_retries=1,
        max_tokens=128,
    )
    if not result.success:
        return False, (result.error or f"status={result.status_code}")[:80]
    # Thinking models may return content=None with finish_reason="length"
    # when reasoning_content fills the budget. Treat HTTP 200 as reachable; the
    # smoke's purpose is connectivity, not full answer quality.
    if result.content:
        return True, result.content.strip()[:40]
    if result.finish_reason == "length":
        return True, "(thinking model exhausted budget)"
    return False, f"empty content; finish={result.finish_reason}"


async def probe_image(model: str, auth_headers: dict[str, str]) -> tuple[bool, str]:
    """Image-enabled probe — sends a 512×512 gray PNG plus a text prompt."""
    image_data_url = generate_probe_image_data_url()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Reply with OK."},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]
    result = await nim_client.chat_completions(
        base_url=HOSTED_BASE_URL,
        auth_headers=auth_headers,
        model=model,
        messages=messages,
        deadline_s=DEADLINE_S,
        max_retries=1,
        max_tokens=128,
    )
    if not result.success:
        return False, (result.error or f"status={result.status_code}")[:80]
    if result.content:
        return True, result.content.strip()[:40]
    if result.finish_reason == "length":
        return True, "(thinking model exhausted budget)"
    return False, f"empty content; finish={result.finish_reason}"


async def run_cosmos_capability_probes(
    model: str,
    auth_headers: dict[str, str],
    thinking_mode: str,
    visual_mode: str,
) -> dict[str, str]:
    """Runs the three capability probes against a Cosmos Reason2 model."""
    structured = await probe_structured_generation(
        base_url=HOSTED_BASE_URL,
        auth_headers=auth_headers,
        model_name=model,
        deadline_s=DEADLINE_S,
    )
    thinking = await probe_thinking_toggle(
        base_url=HOSTED_BASE_URL,
        auth_headers=auth_headers,
        model_name=model,
        thinking_toggle_mode=thinking_mode,
        deadline_s=DEADLINE_S,
    )
    visual = await probe_visual_budget(
        base_url=HOSTED_BASE_URL,
        auth_headers=auth_headers,
        model_name=model,
        visual_budget_mode=visual_mode,
        deadline_s=DEADLINE_S,
    )
    return {
        "structured_generation": structured,
        "thinking_toggle": thinking,
        "visual_budget": visual,
    }


async def main() -> int:
    # No flags — but parse argv so ``--help`` prints usage and an argv typo
    # errors out instead of silently launching the paid live smoke.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args()

    api_key = _load_api_key()
    if not api_key:
        print(
            "ERROR: NVIDIA_API_KEY not found in environment or "
            "~/.vlm_feedback_loop/.env",
            file=sys.stderr,
        )
        return 1

    auth_headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Hosted NIM base URL: {HOSTED_BASE_URL}")
    print(f"Testing {len(SEEDED_MODELS)} seeded models…\n")

    # 1. One list_models call, reused across all models.
    catalog_ids = await probe_list(auth_headers)
    print(f"Catalog list_models returned {len(catalog_ids)} model IDs\n")

    results: list[tuple[str, bool, str]] = []
    for m in SEEDED_MODELS:
        listed = m.name in catalog_ids
        text_ok, text_summary = await probe_text(m.name, auth_headers)
        if m.supports_image_input:
            image_ok, _image_summary = await probe_image(m.name, auth_headers)
        else:
            image_ok = True

        summary_parts = [
            f"list={'✓' if listed else '✗'}",
            f"text={'✓' if text_ok else '✗'}",
            f"image={'✓' if image_ok else '✗' if m.supports_image_input else '-'}",
        ]
        status = "✓" if (listed and text_ok and image_ok) else "✗"
        line = (
            f"{status} {m.name:<55}  "
            + "  ".join(summary_parts)
            + f"  reply={text_summary!r}"
        )
        print(line)
        results.append((m.name, (listed and text_ok and image_ok), text_summary))

    # Capability probes — Cosmos Reason2 only (user's primary target).
    print("\nCapability probes — Cosmos Reason2 (primary target):")
    for m in SEEDED_MODELS:
        if not m.name.startswith("nvidia/cosmos-reason2-"):
            continue
        caps = await run_cosmos_capability_probes(
            m.name, auth_headers, m.thinking_toggle_mode, m.visual_budget_mode
        )
        print(
            f"  {m.name}: structured={caps['structured_generation']} "
            f"thinking={caps['thinking_toggle']} "
            f"visual_budget={caps['visual_budget']}"
        )

    # Final verdict.
    print()
    green = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    cosmos_ok = all(
        ok for name, ok, _ in results if name.startswith("nvidia/cosmos-reason2-")
    )
    if green == total:
        print(f"✓ ALL GREEN — {green}/{total} models reachable.")
        return 0
    if cosmos_ok:
        print(
            f"⚠ Cosmos Reason2 2B+8B OK, but {total - green}/{total} other model(s) failed. "
            "Cosmos is the primary target — non-zero exit to flag the partial miss."
        )
    else:
        print(
            f"✗ Cosmos Reason2 FAILURE — {green}/{total} total green; "
            "investigate before declaring the fix complete."
        )
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

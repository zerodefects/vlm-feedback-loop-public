#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged subprocess driver for self-service base-experiment provisioning.

Standalone helper invoked as a subprocess by
``services.tao_base_experiment_provisioning_service``. Kept out of the
backend import path so the main FastAPI process does not pull in
``nvidia-tao-core``, ``huggingface_hub``, ``transformers``, ``peft`` or
the rest of NVIDIA's PTM packaging tooling. Mirrors the contract of
``scripts/merge_lora.py``.

Operators normally invoke this helper through
``vlm-feedback-loop tao-pull-base-experiments``. It lives inside the backend
package so source, container, and installed-wheel executions use the same
resource.

What it does
------------

1. Locates ``nvidia_tao_core.microservices.pretrained_models``'s bundled
   ``pretrained_models.csv``, backs it up next to itself as
   ``pretrained_models.csv.bak`` (idempotent), and overwrites it with
   the operator-supplied CSV. The bundled-CSV swap is what feeds Cosmos
   Reason2 ``hf_model://`` rows into the airgapped pipeline.
2. Sets ``AIRGAPPED_MODE=true`` and ``PTM_API_KEY`` in the subprocess env
   so ``pretrained_models`` writes ``ptm_metadatas.json`` directly under
   ``--shared-folder-path`` without contacting MongoDB.
3. Spawns ``python -m nvidia_tao_core.microservices.pretrained_models
   --shared-folder-path <stage> --use-csv``. Captures stdout/stderr.
4. On success: parses ``<stage>/ptm_metadatas.json`` and prints a JSON
   summary to stdout, exit 0:
   ``{"ptm_metadatas_path": "...", "registered": ["<display name>", ...]}``.
5. On failure: prints ``{"error": "..."}`` to stderr (single line), exits
   non-zero.

This script is exercised end-to-end by
``scripts/tao_live_smoke.py --auto-provision-base-experiments``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn, cast


def _redact_environment_secrets(text: str) -> str:
    """Remove credential values inherited by this standalone driver."""
    redacted = text
    values = {
        os.environ.get("PTM_API_KEY", ""),
        os.environ.get("HF_TOKEN", ""),
        os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
    }
    for value in sorted({value for value in values if value}, key=len, reverse=True):
        redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def _die(msg: str, code: int = 2) -> NoReturn:
    print(json.dumps({"error": _redact_environment_secrets(msg)}), file=sys.stderr)
    sys.exit(code)


def _locate_bundled_csv() -> Path:
    """Find the ``pretrained_models.csv`` shipped inside nvidia_tao_core."""
    try:
        # Imported here so help/validation runs do not require the lib.
        package = importlib.import_module("nvidia_tao_core.microservices")
    except ImportError as exc:
        _die(
            f"nvidia-tao-core is not installed ({exc}). "
            "Use the Blueprint provisioning command's default isolated helper, "
            "or install the packaged TAO pull requirements into an "
            "operator-managed environment and pass --skip-install."
        )
    package_file = package.__file__
    if package_file is None:
        _die("nvidia_tao_core.microservices has no filesystem package path")
    bundled = Path(package_file).parent / "pretrained_models.csv"
    if not bundled.is_file():
        _die(f"bundled pretrained_models.csv not found at {bundled}")
    return bundled


def _swap_in_operator_csv(operator_csv: Path) -> Path:
    """Back up the bundled CSV (once) and overwrite it with the operator's."""
    bundled = _locate_bundled_csv()
    backup = bundled.with_suffix(bundled.suffix + ".bak")
    if not backup.exists():
        try:
            shutil.copy2(bundled, backup)
        except OSError as exc:
            _die(f"could not back up bundled CSV at {bundled}: {exc}")
    try:
        shutil.copy2(operator_csv, bundled)
    except OSError as exc:
        _die(f"could not write operator CSV to {bundled}: {exc}")
    return bundled


def _validate_csv(operator_csv: Path) -> None:
    if not operator_csv.is_file():
        _die(f"--csv not found: {operator_csv}")
    head = operator_csv.read_text(encoding="utf-8").splitlines()[:1]
    if not head:
        _die(f"--csv is empty: {operator_csv}")
    expected_header = "displayName,ngc_path,network_arch,is_backbone"
    if head[0].strip() != expected_header:
        _die(f"--csv header must be exactly {expected_header!r}, got {head[0]!r}")


def _run_pretrained_models(
    *, shared_folder: Path, ngc_key: str
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "nvidia_tao_core.microservices.pretrained_models",
        "--shared-folder-path",
        str(shared_folder),
        "--use-csv",
    ]
    env = os.environ.copy()
    env["AIRGAPPED_MODE"] = "true"
    if ngc_key:
        env["PTM_API_KEY"] = ngc_key
    hf_token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if "HF_HOME" not in env:
        env["HF_HOME"] = str(shared_folder / "_hf_cache")

    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)

    p = argparse.ArgumentParser(description="Self-service base-experiment driver.")
    p.add_argument(
        "--csv", required=True, help="Operator-supplied pretrained_models CSV."
    )
    p.add_argument(
        "--shared-folder-path",
        required=True,
        help="Stage directory; pretrained_models writes ptm_metadatas.json + checkpoints here.",
    )
    args = p.parse_args(cli_argv)

    csv_path = Path(args.csv).resolve()
    stage = Path(args.shared_folder_path).resolve()

    _validate_csv(csv_path)
    stage.mkdir(parents=True, exist_ok=True)

    ngc_key = (os.environ.get("PTM_API_KEY", "") or "").strip()

    _swap_in_operator_csv(csv_path)

    try:
        proc = _run_pretrained_models(shared_folder=stage, ngc_key=ngc_key)
    except OSError as exc:
        _die(f"pretrained_models failed to spawn: {exc}")
    if proc.returncode != 0:
        safe_stderr = _redact_environment_secrets((proc.stderr or "").strip())[:1024]
        _die(f"pretrained_models exited rc={proc.returncode}: {safe_stderr}")

    metadatas = stage / "ptm_metadatas.json"
    if not metadatas.is_file():
        _die(
            f"missing ptm_metadatas.json under {stage}; "
            "pretrained_models did not produce expected output"
        )
    try:
        body = json.loads(metadatas.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _die(f"ptm_metadatas.json malformed: {exc}")

    metadata = cast("dict[str, object]", body) if isinstance(body, dict) else {}
    registered = sorted(metadata)

    print(json.dumps({"ptm_metadatas_path": str(metadatas), "registered": registered}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge a LoRA adapter checkpoint into its base model.

Standalone helper invoked as a subprocess by the checkpoint packaging
service. Kept out of the backend import path so the main
FastAPI process does not pull in ``transformers`` + ``peft`` (~2 GB of
ML tooling).

Usage::

    python scripts/merge_lora.py \\
        --adapter /path/to/lora/output \\
        --base    /path/to/hf/base_model \\
        --out     /path/to/merged_hf_checkpoint

On success: prints a JSON summary to stdout and exits 0. On failure:
prints a single-line error message to stderr and exits non-zero.

Requires ``transformers>=4.41`` and ``peft>=0.11``. See
``scripts/merge_lora_requirements.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def _validate_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.is_dir():
        _die(f"adapter path is not a directory: {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").is_file():
        _die(f"missing adapter_config.json in {adapter_dir}")
    have_safetensors = (adapter_dir / "adapter_model.safetensors").is_file()
    have_bin = (adapter_dir / "adapter_model.bin").is_file()
    if not (have_safetensors or have_bin):
        _die(f"missing adapter_model.safetensors or adapter_model.bin in {adapter_dir}")


def _validate_base_path(base_path: str) -> None:
    if not base_path.strip():
        _die("base path is empty")
    # Accept both local paths and HF Hub identifiers (e.g., "org/model").
    local = Path(base_path)
    if local.exists() and not local.is_dir():
        _die(f"base path exists but is not a directory: {base_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter).resolve()
    out_dir = Path(args.out).resolve()

    _validate_adapter_dir(adapter_dir)
    _validate_base_path(args.base)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Import locally so module import doesn't fail for help/validation runs.
    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        _die(
            f"merge_lora: missing required libraries ({exc}). "
            "Install via 'pip install -r scripts/merge_lora_requirements.txt'"
        )

    # The Cosmos student bases are vision-language models (Qwen3-VL
    # configs); ``AutoModelForCausalLM`` rejects their config class
    # outright ("Unrecognized configuration class … for this kind of
    # AutoModel", observed live on the first real adapter
    # merge). Try the VLM auto-classes first — the text-only class stays
    # as the final fallback for any pure-LLM base an operator registers.
    loader_classes = []
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore

        loader_classes.append(AutoModelForImageTextToText)
    except ImportError:  # pragma: no cover — older transformers
        pass
    try:
        from transformers import AutoModelForVision2Seq  # type: ignore

        loader_classes.append(AutoModelForVision2Seq)
    except ImportError:  # pragma: no cover
        pass
    loader_classes.append(AutoModelForCausalLM)

    base_model = None
    load_errors: list[str] = []
    for loader in loader_classes:
        try:
            base_model = loader.from_pretrained(args.base)
            break
        except Exception as exc:  # pragma: no cover — depends on real models
            load_errors.append(f"{loader.__name__}: {exc}")
    if base_model is None:
        _die(
            "merge_lora: failed to load base model: "
            + " | ".join(e[:300] for e in load_errors)
        )

    # cosmos-rl exports its OWN LoraConfig dump as adapter_config.json —
    # not a peft-parseable one (``alpha_pattern: null`` where peft wants a
    # dict, ``r_pattern`` vs peft's ``rank_pattern``, cosmos-only keys such
    # as ``lora_names``, and a container path in base_model_name_or_path).
    # peft 0.19 chokes with "'NoneType' object has no attribute 'keys'"
    # (observed live on the first real cosmos-rl adapter). Build a
    # peft-shaped adapter dir: normalized config + the original weights.
    import shutil
    import tempfile

    raw_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    normalized = {
        "peft_type": "LORA",
        "task_type": None,
        "bias": "none",
        "r": raw_cfg.get("r", 16),
        "lora_alpha": raw_cfg.get("lora_alpha", 32),
        "lora_dropout": raw_cfg.get("lora_dropout", 0.0),
        "target_modules": raw_cfg.get("target_modules") or ["q_proj", "v_proj"],
        "use_rslora": raw_cfg.get("use_rslora", False),
        "modules_to_save": raw_cfg.get("modules_to_save") or None,
        "alpha_pattern": raw_cfg.get("alpha_pattern") or {},
        "rank_pattern": raw_cfg.get("rank_pattern") or raw_cfg.get("r_pattern") or {},
        "init_lora_weights": raw_cfg.get("init_lora_weights", True),
        "base_model_name_or_path": args.base,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="merge-lora-") as tmp:
            tmp_adapter = Path(tmp) / "adapter"
            tmp_adapter.mkdir()
            (tmp_adapter / "adapter_config.json").write_text(json.dumps(normalized))
            for weights_name in ("adapter_model.safetensors", "adapter_model.bin"):
                src = adapter_dir / weights_name
                if src.exists():
                    shutil.copy2(src, tmp_adapter / weights_name)
            merged = PeftModel.from_pretrained(
                base_model, str(tmp_adapter)
            ).merge_and_unload()
    except Exception as exc:  # pragma: no cover
        _die(f"merge_lora: adapter merge failed: {exc}")

    try:
        merged.save_pretrained(str(out_dir), safe_serialization=True)
    except Exception as exc:  # pragma: no cover
        _die(f"merge_lora: failed to write merged checkpoint: {exc}")

    # Tokenizer + processor aux files: PREFER the adapter dir's original
    # files (cosmos-rl copies the base's alongside the adapter) and
    # OVERWRITE anything this environment's save_pretrained produced.
    # Two live failures drove this: the merged dir served
    # without preprocessor_config.json ("Can't load image processor"),
    # and this venv's newer transformers re-serialized
    # tokenizer/special-tokens files in a format the NIM image's older
    # transformers cannot parse ("'list' object has no attribute
    # 'keys'"). Byte-copying the originals is deterministic,
    # offline-friendly, and version-skew-proof. AutoTokenizer save is
    # only the fallback for adapters that ship no tokenizer files.
    _AUX_FILES = (
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
    )
    copied_any = False
    for aux_name in _AUX_FILES:
        src = adapter_dir / aux_name
        if src.exists():
            shutil.copy2(src, out_dir / aux_name)
            copied_any = True
    if not copied_any:
        try:
            tok = AutoTokenizer.from_pretrained(args.base)
            tok.save_pretrained(str(out_dir))
        except Exception:  # pragma: no cover — models without tokenizers
            pass

    # Summary for the caller (stdout only; stderr is reserved for errors).
    summary = {
        "status": "ok",
        "out": str(out_dir),
        "files": sorted(p.name for p in out_dir.iterdir()),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

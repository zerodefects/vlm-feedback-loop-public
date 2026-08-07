# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical model identifiers and deployment constants — single source of truth.

Every model name, NIM container image ref, Hugging Face model-card path,
TAO base-experiment display name, embedding-NIM identity value, and
GPU-minimum figure that the Blueprint seeds or matches against lives here.
Services, routers, DB seeding, and CLIs import these constants instead of
re-typing wire values.

Rules for this module:

* **Dependency-free.** It imports nothing from the package (no
  ``services/``, no ``db/``, no ``config``), so any layer — including
  Alembic migration helpers and the CLIs — can import it without cycles.
* **Values are wire/persisted contracts.** These strings appear in
  project databases (``ModelConfig.model_name``), in the
  deployment DB, and on the wire to NIM/TAO. Changing a value
  here is a data migration, not a refactor.
* Alembic revisions under ``migrations/versions/`` intentionally keep
  their own literals — committed migrations are frozen and must not
  drift when these constants move.

Pinning tests (drift guards) keep asserting the raw literals:
``test_projects.py`` (seed catalog model names + NIM images),
``test_database.py`` (embedding model/dim/image), and
``test_training_preset.py`` (per-model epoch tables).
"""

from __future__ import annotations

# ── Cosmos family (local NIM Teachers + TAO-trainable Student bases) ────────

COSMOS_REASON2_2B = "nvidia/cosmos-reason2-2b"
COSMOS_REASON2_8B = "nvidia/cosmos-reason2-8b"
COSMOS3_NANO_REASONER = "nvidia/cosmos3-nano-reasoner"
COSMOS3_SUPER_REASONER = "nvidia/cosmos3-super-reasoner"

# ── Hosted Teacher identities (current seed + historical records) ──────────

MISTRAL_MEDIUM_3_5 = "mistralai/mistral-medium-3.5-128b"
NEMOTRON_NANO_12B_VL = "nvidia/nemotron-nano-12b-v2-vl"
NEMOTRON_3_NANO_OMNI_REASONING = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
STEP_3_7_FLASH = "stepfun-ai/step-3.7-flash"
# Historical/operator-created records only. MiniMax M3 is intentionally not
# part of the fresh-project seed because its published terms are
# non-commercial. The constant remains a wire-value pin for old project DBs
# and retained experiment tooling.
MINIMAX_M3 = "minimaxai/minimax-m3"

# ── NIM container images (pinned tags; compose + local_nim deploys) ─────────
#
# Both Cosmos 3 Reasoner sizes ship in ONE shared image selected by
# ``NIM_MODEL_SIZE`` (see project_service seed metadata and
# local_nim_service), hence a single COSMOS3_REASONER_NIM_IMAGE.

COSMOS_REASON2_2B_NIM_IMAGE = "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0"
COSMOS_REASON2_8B_NIM_IMAGE = "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0"
COSMOS3_REASONER_NIM_IMAGE = "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0"
NEMOTRON_3_NANO_OMNI_NIM_IMAGE = (
    "nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant"
)

# ── GPU minimums (GB, per local_deploy_metadata.nim_gpu_memory_minimum_gb) ──
#
# Measured floors, not vendor VRAM math: CR2-2B fits the 36–55 GB
# tier, CR2-8B and CR3-Nano share the ≥56 GB tier, CR3-Super (~30B class)
# needs ≥88 GB.

COSMOS_REASON2_2B_GPU_MIN_GB = 36
COSMOS_REASON2_8B_GPU_MIN_GB = 56
COSMOS3_NANO_REASONER_GPU_MIN_GB = 56
COSMOS3_SUPER_REASONER_GPU_MIN_GB = 88
NEMOTRON_3_NANO_OMNI_GPU_MIN_GB = 80
NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN = 9.0

# ── Hugging Face model-card paths (TAO training wire format) ────────────────
#
# NVIDIA-published HF repo names for the seeded student_base entries. The
# Cosmos 3 trainable VLM is the ``-Reasoner`` repo (plain ``qwen3_vl``),
# NOT the omni generator ``nvidia/Cosmos3-{Nano,Super}`` (diffusion + VAE,
# untrainable by cosmos-rl SFT). cosmos-rl has no dense ``qwen3_vl`` class
# and trains it via its generic HFModel + HFVLMDataPacker fallback —
# verified end-to-end on TAO FTMS for both Nano and Super.

COSMOS_REASON2_2B_HF_PATH = "nvidia/Cosmos-Reason2-2B"
COSMOS_REASON2_8B_HF_PATH = "nvidia/Cosmos-Reason2-8B"
COSMOS3_NANO_REASONER_HF_PATH = "nvidia/Cosmos3-Nano-Reasoner"
COSMOS3_SUPER_REASONER_HF_PATH = "nvidia/Cosmos3-Super-Reasoner"

# The one model_name → HF-path roster. Consumed by cosmos-rl training-spec
# emission (``training_suite_service``) and self-service base-experiment
# provisioning (``tao_base_experiment_provisioning_service``); a new
# trainable base is added here once and both flows pick it up.
HF_MODEL_PATHS: dict[str, str] = {
    COSMOS_REASON2_2B: COSMOS_REASON2_2B_HF_PATH,
    COSMOS_REASON2_8B: COSMOS_REASON2_8B_HF_PATH,
    COSMOS3_NANO_REASONER: COSMOS3_NANO_REASONER_HF_PATH,
    COSMOS3_SUPER_REASONER: COSMOS3_SUPER_REASONER_HF_PATH,
}

# TAO uses these display names both when registering air-gapped experiments
# and when finding the resulting records. They are external lookup identities,
# not UI presentation labels.
TAO_BASE_EXPERIMENT_DISPLAY_NAMES: dict[str, str] = {
    COSMOS_REASON2_2B: "Cosmos Reason2 2B",
    COSMOS_REASON2_8B: "Cosmos Reason2 8B",
    COSMOS3_NANO_REASONER: "Cosmos 3 Nano (Reasoner)",
    COSMOS3_SUPER_REASONER: "Cosmos 3 Super (Reasoner)",
}

# ── Embedding NIM (NeMo Retriever VL 1B v2) ─────────────────────────────────
#
# NIM 2.0.0 keeps the model/API contract while replacing the legacy runtime
# with automatic architecture-aware kernels. Its support matrix validates
# specific GPU SKUs; the smallest listed devices (L4 and A10G) have 24 GB.
# Memory alone does not establish compatibility. ``db/engine.py`` upgrades
# deployment DBs that still carry a shipped 1.x image pin or older floor.

EMBEDDING_MODEL_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBEDDING_DIM = 2048
EMBEDDING_INPUT_TYPE = "passage"
EMBEDDING_NIM_IMAGE = "nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0"
EMBEDDING_NIM_GPU_MIN_GB = 24
# Exact display names listed for the pinned VLM embedding model in NVIDIA's
# NIM 2.0.0 support matrix. Environment recommendations stay conservative on
# unlisted hardware; an operator may still try the documented fallback path
# manually because final NIM preflight remains authoritative.
EMBEDDING_NIM_SUPPORTED_GPU_NAMES = (
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA B200",
    "NVIDIA GB200",
    "NVIDIA H200",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 NVL",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA L4",
    "NVIDIA L40S",
    "NVIDIA A10G",
)

__all__ = [
    "COSMOS3_NANO_REASONER",
    "COSMOS3_NANO_REASONER_GPU_MIN_GB",
    "COSMOS3_NANO_REASONER_HF_PATH",
    "COSMOS3_REASONER_NIM_IMAGE",
    "COSMOS3_SUPER_REASONER",
    "COSMOS3_SUPER_REASONER_GPU_MIN_GB",
    "COSMOS3_SUPER_REASONER_HF_PATH",
    "COSMOS_REASON2_2B",
    "COSMOS_REASON2_2B_GPU_MIN_GB",
    "COSMOS_REASON2_2B_HF_PATH",
    "COSMOS_REASON2_2B_NIM_IMAGE",
    "COSMOS_REASON2_8B",
    "COSMOS_REASON2_8B_GPU_MIN_GB",
    "COSMOS_REASON2_8B_HF_PATH",
    "COSMOS_REASON2_8B_NIM_IMAGE",
    "EMBEDDING_DIM",
    "EMBEDDING_INPUT_TYPE",
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_NIM_GPU_MIN_GB",
    "EMBEDDING_NIM_IMAGE",
    "EMBEDDING_NIM_SUPPORTED_GPU_NAMES",
    "HF_MODEL_PATHS",
    "MINIMAX_M3",
    "MISTRAL_MEDIUM_3_5",
    "NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN",
    "NEMOTRON_3_NANO_OMNI_GPU_MIN_GB",
    "NEMOTRON_3_NANO_OMNI_NIM_IMAGE",
    "NEMOTRON_3_NANO_OMNI_REASONING",
    "NEMOTRON_NANO_12B_VL",
    "STEP_3_7_FLASH",
    "TAO_BASE_EXPERIMENT_DISPLAY_NAMES",
]

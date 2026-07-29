# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference configuration defaults — single source of truth.

Every configuration key is represented here with its Python-native
default value. The config loader and bootstrap CLI both import from
this module so the defaults are never duplicated.
"""

from __future__ import annotations

from typing import Any

from vlm_feedback_loop.model_catalog_constants import (
    EMBEDDING_DIM,
    EMBEDDING_INPUT_TYPE,
    EMBEDDING_MODEL_ID,
    MINIMAX_M3,
)

# Keys that are deployment-scoped secrets and belong ONLY in .env,
# never in config.yaml.
SECRET_KEYS: set[str] = {
    "NVIDIA_API_KEY",
    "NGC_API_KEY",
    "HF_TOKEN",
    "TAO_API_KEY",
    "TAO_API_BASE_URL",
    "TAO_ORG_NAME",
    "TAO_WORKSPACE_S3_ACCESS_KEY",
    "TAO_WORKSPACE_S3_SECRET_KEY",
}

# Complete configuration defaults. WORKSPACE_ROOT is intentionally absent —
# it has no default and is required.
DEFAULTS: dict[str, Any] = {
    # ── Embedding computation ────────────────────────────────────────
    "EMBEDDING_PROVIDER": "auto",
    "EMBEDDING_MODEL_ID": EMBEDDING_MODEL_ID,
    "EMBEDDING_DIM": EMBEDDING_DIM,
    "EMBEDDING_INPUT_TYPE": EMBEDDING_INPUT_TYPE,
    "EMBEDDINGS_AUTO_COMPUTE": True,
    # ── Generation Controls ──────────────────────────────────────────
    "THINKING_DEFAULT_ON": True,
    "LABELING_PRESETS": {
        "precise": {"temperature": 0.0, "top_p": 1.0},
        "explore": {"temperature": 0.3, "top_p": 0.9},
    },
    # ── Visual Budget Controls ───────────────────────────────────────
    # The default is the documented Cosmos-max pixel-area budget
    # (high_detail) for all tunable models. ``size.shortest_edge`` /
    # ``size.longest_edge`` are Hugging Face processor *pixel-area* bounds
    # despite their names; they are not linear image dimensions. The former
    # 672/1344 → 1568/4096 ladder treated them as dimensions and collapsed
    # every Cosmos 3 preset to roughly text-only perception. The 64K/128K/
    # 256K ladder below was live-validated on Cosmos 3 Super and preserves
    # progressively larger visual-token budgets. More tiles/pixels = finer
    # patches.
    "VISUAL_BUDGET_PRESETS": {
        # ``native``: no mm_processor_kwargs at all — the model's processor
        # sees the image at its natural size. This is the REQUIRED preset
        # for fine-tuned Student invocations (established 2026-07-15): §9.3
        # training consumes images at native size, so any serve-time
        # resize puts the Student off its training distribution. Measured
        # live on a freiburg (256×256) CR2-2B student, same checkpoint,
        # same 120 keys: EM 0.95 native vs 0.367 with the former high_detail
        # resize override. Teacher invocations keep the corrected area-budget
        # ladder — more visual tokens genuinely help un-tuned perception.
        "native": {},
        "fast": {
            "mm_processor_size": {
                "size": {"shortest_edge": 1568, "longest_edge": 65536},
            },
            "mm_processor_pixels": {
                "images_kwargs": {"min_pixels": 1568, "max_pixels": 65536},
            },
            "mm_processor_tiles": {"max_num_tiles": 8},
        },
        "balanced": {
            "mm_processor_size": {
                "size": {"shortest_edge": 1568, "longest_edge": 131072},
            },
            "mm_processor_pixels": {
                "images_kwargs": {"min_pixels": 1568, "max_pixels": 131072},
            },
            "mm_processor_tiles": {"max_num_tiles": 16},
        },
        # Cosmos-documented maximum visual detail (the default). The
        # capability probe uses the same 1568/262144 area bounds. A live
        # rationale sweep on Cosmos 3 Super confirmed this restores physical
        # item perception that the legacy 1568/4096 cap erased.
        "high_detail": {
            "mm_processor_size": {
                "size": {"shortest_edge": 1568, "longest_edge": 262144},
            },
            "mm_processor_pixels": {
                "images_kwargs": {"min_pixels": 1568, "max_pixels": 262144},
            },
            "mm_processor_tiles": {"max_num_tiles": 32},
        },
    },
    # Client-side transport downscale: images whose longest edge exceeds
    # this many pixels are resized (aspect-preserving, LANCZOS) before
    # base64 encoding. The visual-budget presets above are server-side
    # (vLLM mm_processor_kwargs) and only work where the endpoint's
    # capability probe reports support; most hosted build.nvidia.com
    # models ignore them, so oversized source images ship at full
    # resolution and can blow the provider's ~300s gateway ceiling on
    # multi-image ICL requests (observed: 4 full-res
    # 4032x3024 images → HTTP 504 at 302s on qwen3.5; the same request
    # at 2016px answers in 3-14s; JPEG recompression alone does NOT fix
    # it — the constraint is pixels, not bytes).
    #
    # Default 2090 is the transport study's high-fidelity ceiling. A 7-model
    # acuity study found 5 of 7 hosted teachers plateau at
    # or below this, and it keeps the worst-case 8-image payload inside
    # every measured provider envelope (~36 MP pixel ceiling, ~5 MB
    # request-body cap) at any aspect ratio. Set to 0 (or null) to
    # disable and transmit source images as-is.
    "IMAGE_TRANSPORT_MAX_LONGEST_EDGE": 2090,
    # ── Prompt budgets ───────────────────────────────────────────────
    "RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE": None,
    "RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN": 0.85,
    "BASE_OUTPUT_TOKENS_FLOOR": 256,
    "JSON_STRUCTURAL_OVERHEAD_TOKENS": 48,
    "MAX_OUTPUT_FRACTION": 0.25,
    "RATIONALE_NOTE_ESTIMATE_TOKENS": 160,
    "DEFAULT_UNBOUNDED_STRING_BUDGET": 200,
    "MODEL_REASONING_HEADROOM_TOKENS": 16384,
    # ── ICL selection ────────────────────────────────────────────────
    # Selection is always relevance-ranked (per-query CLIP-nearest Edits —
    # proven-best: 1.6-4x macro-F1 over zero-shot on Cosmos), falling back
    # to newest-first when embeddings are unavailable.
    "ICL_MAX_EXAMPLES": None,
    # Adaptive-K (similarity-gap stopping). Per-query depth: after relevance
    # ranking, keep the leading prefix of neighbors whose cosine similarity is
    # within ICL_SIM_GAP of the best neighbor; stop at the first that drops
    # further (top-1 always kept). This prunes the dissimilar tail that drives
    # the ICL depth-decline at high K — the proven cross-model max_icl fix
    # (gap 0.05 rescues +0.10-0.11 mean EM on Cosmos Reason, up to +0.41 on
    # Nemotron Omni). ICL_ABS_THRESHOLD is an optional absolute similarity
    # floor (off by default). When BOTH are None this is a no-op (fixed-K).
    # In the pHash / no-CLIP regime relevance falls back to newest_first, so
    # adaptive-K is automatically a clean no-op.
    "ICL_SIM_GAP": 0.05,
    "ICL_ABS_THRESHOLD": None,
    # ── Batch labeling ───────────────────────────────────────────────
    "BATCH_LABEL_RUN_LIMIT": None,
    "BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD": 10,
    "BATCH_LABEL_CONCURRENCY_HOSTED": 1,
    "BATCH_LABEL_CONCURRENCY_SELF_HOSTED": 8,
    # ── Student training ─────────────────────────────────────────────
    "TAO_RELEASE_VERSION": "6.26.3",
    "COSMOS_RL_CONTAINER_TAG": "6.26.3-cosmos-rl",
    # FTMS kills any job whose status heartbeat goes stale for
    # ``timeout_minutes`` (default 60 when the field is absent). cosmos-rl
    # emits NO status updates during training, so every train longer than
    # about an hour is otherwise SIGTERM'd at +60 min. The v8 TAO install
    # guide patches FTMS 6.26.3's v2 schema to accept this field. The
    # Blueprint now requires that capability in training preflight and sends
    # a generous 24-hour stale-job ceiling on every suite job. This remains
    # an operator-overridable dead-job backstop, not a runtime estimate.
    "TAO_JOB_TIMEOUT_MINUTES": 1440,
    # Partial quality_status threshold. A NIM eval that
    # finishes ``incomplete`` but produces parseable output on at least this
    # fraction of examples promotes the paired StudentModel to
    # quality_status="partial" (informational; deployment_handoff still
    # requires ``validated``). Default 0.90 matches the observed 8B
    # baseline near-miss case (135/148 = 91% parseable).
    "STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD": 0.90,
    # Interpreter for the LoRA-merge subprocess. None → use the
    # provisioned {WORKSPACE_ROOT}/merge-lora-venv when present, else the
    # backend interpreter (which needs scripts/merge_lora_requirements.txt
    # installed).
    "MERGE_LORA_PYTHON": None,
    # Non-secret TAO workspace identity lives in deployment.db. S3 credential
    # settings are declared with default=None in Settings because they are
    # secrets loaded from the process environment or canonical .env.
    # ── NIM serving benchmark ────────────────────────────────────────
    # Cosmos Reason2 8B first-deploy cold-start measured at ~693 s on a
    # single A100-SXM4-80GB host (NGC pull ~5 min for ~16 GB of weights +
    # weights load + torch.compile 53 s + KV cache + CUDA graphs ~3 min +
    # uvicorn boot). 600 s is too tight: both the backend's
    # ``_poll_health`` task (services/local_nim_service.py) and AutoRun's
    # deploy poll (cli_autorun.py) time out before NIM becomes healthy,
    # leaving the deployment stuck at status="failed" while the
    # container itself is minutes away from serving. 1200 s
    # gives cosmos-8b ~500 s of margin and is still a sane upper bound
    # for cosmos-2b (whose cold-start is ~9 min on cold cache, ~5 min on
    # warm cache). Operators with slower NGC pulls can override via
    # NIM_STARTUP_TIMEOUT_S in ~/.vlm_feedback_loop/.env.
    "NIM_STARTUP_TIMEOUT_S": 1200,
    "NIM_BENCHMARK_TIMEOUT_S": 1200,
    "STUDENT_LATENCY_TEST_CONCURRENCIES": [1, 8, 24],
    # ── Student NIM deployment ───────────────────────────────────────
    # Per-precision GPU memory minima used by the Student preflight
    # check. Cosmos Reason2 8B BF16 needs >56 GB, FP8 >48 GB; 2B BF16
    # needs >36 GB, FP8 >24 GB. W4A16 reuses the FP8 floor for the same
    # base model. The orchestrator's _resolve_gpu_memory_minimum() helper
    # selects the right value from base model size + quantization_method.
    "NIM_GPU_MEMORY_8B_BF16_GB": 56,
    "NIM_GPU_MEMORY_8B_FP8_GB": 48,
    "NIM_GPU_MEMORY_2B_BF16_GB": 36,
    "NIM_GPU_MEMORY_2B_FP8_GB": 24,
    # Third role-specific default port; Teacher=8000, embedding=8001,
    # Student=8002. Falls back via _resolve_port() up to +100 if occupied.
    "NIM_STUDENT_PORT": 8002,
    # ── Review selector ──────────────────────────────────────────────
    "REVIEW_SELECTION_MODE": "auto",
    "REVIEW_RECENT_WINDOW_K": 20,
    "CLIP_SWITCHOVER_MIN_COUNT": 50,
    # ── Schema refinement reminders ──────────────────────────────────
    "SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1": 10,
    "SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2": 35,
    # ── Runtime rejection fallbacks ──────────────────────────────────
    # Kill-switches for the per-invocation thinking-toggle and
    # visual-budget rejection fallbacks. Default True; flip False to run
    # in telemetry-only mode where the detector still populates
    # ``OperationRecord.{thinking,visual_budget}_toggle_attempted`` but
    # neither auto-retries (interactive) nor fails the run (eval/batch)
    # — useful for one release of observation before trusting the
    # heuristic. Mirrors the rollout pattern used for
    # ``structured_generation_rejected``.
    "ENABLE_THINKING_TOGGLE_FALLBACK": True,
    "ENABLE_VISUAL_BUDGET_FALLBACK": True,
    # ── Evaluation concurrency ───────────────────────────────────────
    # Provider-aware tuning, mirrors the embedding worker pattern below.
    # Hosted endpoints (build.nvidia.com) enforce per-account RPM caps
    # (e.g., Mistral Large 3 = 40 RPM at the time of writing) and are
    # shared with foreground SME traffic — concurrency=1 keeps eval polite
    # and well under the cap. Self-hosted NIMs have no shared rate limit
    # and benefit from concurrent dispatch to saturate the GPU pipeline.
    # Effective concurrency for a given run is picked from the run's
    # Teacher endpoint mode (hosted / self_hosted / local_system_managed).
    # See evaluation_service._resolve_eval_concurrency.
    "EVAL_CONCURRENCY_HOSTED": 1,
    "EVAL_CONCURRENCY_SELF_HOSTED": 8,
    # Foreground-priority dispatch hold (services/priority.py). True = shipped
    # interactive behavior: an in-flight SME proposal holds new background
    # HTTP dispatches. False = long AutoRun/batch runs: background evals
    # dispatch freely (a continuous AutoRun proposal stream otherwise starves
    # them indefinitely — measured 2026-07-14).
    "FOREGROUND_HOLD_ENABLED": True,
    "EVAL_FIRST_POOL_SIZE": 5,
    # ── Rate-limit-aware eval completion (Phase C multi-pass retry) ──
    # After the single Phase C retry pass, examples still failing with a
    # hosted 429 (``rate_limited``) get up to this many extra retry passes,
    # each preceded by a backoff. 429 is transient (quota recovers), so this
    # lets a temporary hosted-quota crunch still yield a COMPLETE Test Pool —
    # the Returning-vs-New regression signal depends on the same images being
    # scored each run. Bounded so a true sustained outage still finalizes
    # ``incomplete``. Only ``rate_limited`` is
    # retried this way; ``schema_invalid`` (real model failure) and
    # ``timeout``/``endpoint_error`` (possible dead endpoint) are not.
    "EVAL_RATE_LIMIT_RETRY_MAX_PASSES": 4,
    "EVAL_RATE_LIMIT_RETRY_BACKOFF_S": 30,
    # ── Global hosted RPM ceiling (cross-model) ─────────────────────
    # Some accounts enforce a single GLOBAL request-per-minute cap shared
    # across ALL hosted models (not the per-model caps the adaptive pacer
    # assumes). Under such a cap, concurrent eval streams of different
    # models collectively overrun the budget and slow models get starved
    # into 429/timeout storms. When >0, a process-global token bucket
    # (services/hosted_rate_limiter) paces every hosted build.nvidia.com
    # request — incl. retries — to this ceiling. 0 = disabled (default;
    # the per-model adaptive pacer still applies). Set a few under the
    # real cap for headroom (e.g. 38 for a 40-RPM account).
    "HOSTED_GLOBAL_RPM": 0,
    # ── Embedding worker concurrency + batch shape ───────────────────
    # Provider-aware tuning for the background embedding worker.
    #
    # Hosted endpoints (build.nvidia.com) are rate-limited and shared
    # across users — keep concurrency at 1 so we don't compete with the
    # operator's own foreground traffic or other tenants.  Batching
    # multiple images per request is more efficient under rate-limits.
    #
    # Self-hosted local NIMs have no rate-limit constraint and benefit
    # from concurrent single-image requests: the GPU pipeline saturates
    # at lower per-request memory pressure than large server-side batches.
    "EMBEDDING_CONCURRENCY_HOSTED": 1,
    "EMBEDDING_BATCH_SIZE_HOSTED": 8,
    "EMBEDDING_CONCURRENCY_SELF_HOSTED": 4,
    "EMBEDDING_BATCH_SIZE_SELF_HOSTED": 1,
    # ── HTTP client ──────────────────────────────────────────────────
    "HTTP_DEADLINE_INTERACTIVE_S": 180,
    "HTTP_DEADLINE_BACKGROUND_S": 300,
    "HTTP_MAX_RETRIES": 3,
    # ── TAO polling (on-demand refresh + background loop) ────────────
    "TAO_POLL_MIN_INTERVAL_S": 30,
    "TAO_POLL_MIN_INTERVAL_RUNNING_S": 60,
    "TAO_POLL_TICK_S": 10,
    # ── TAO auto-eval skip blocklist ─────────────────────────────────
    # Auto-skip chain advancement of TAO ``evaluate`` jobs whose parent
    # train job's base model is on this blocklist. Default contains
    # the empirically-blocked 8B path: TAO
    # ``cosmos-rl-evaluate`` hits a weight-init loader gap on
    # every Cosmos-Reason2-8B trained checkpoint (the same checkpoint
    # NIM 1.6.0's vLLM serves cleanly). Routing around the doomed
    # evaluate saves ~3 min of TAO compute per chain and lands the
    # Student at ``quality_status="pending"`` (the cleaner cold-start
    # NIM-eval-fallback path) instead of ``failed`` (which requires the
    # weight-init failure pattern-match branch). Operator clears this
    # list once TAO ships an upstream fix. Identification uses the
    # trained ``ModelConfig.model_name`` — match is exact-string against
    # the seeded names (``nvidia/cosmos-reason2-2b`` / ``-8b``).
    "TAO_AUTOEVAL_SKIP_BASES": [
        "nvidia/cosmos-reason2-8b",
        "nvidia/cosmos3-nano-reasoner",
        "nvidia/cosmos3-super-reasoner",
    ],
    # ── Operational logging ──────────────────────────────────────────
    "LOG_LEVEL": "info",
    # ── Workspace, server, and NIM endpoint quickstart ────────────────
    "BIND_HOST": "127.0.0.1",
    "BIND_PORT": 8000,
    # Local loopback development defaults to "/" when unset. Network-
    # accessible deployments must set one explicit root; Compose supplies
    # "/data/images".
    "IMAGE_ROOT": None,
    "HOSTED_NIM_BASE_URL": "https://integrate.api.nvidia.com/v1",
    "LOCAL_NIM_TEACHER_PORT": 8000,
    "LOCAL_NIM_NVCLIP_PORT": 8001,
    # ── New-project defaults (overridable in ~/.vlm_feedback_loop/config.yaml) ─
    # These are read at create_project time. Operators on rentals with a
    # locally-deployed Teacher NIM typically point DEFAULT_TEACHER_MODEL at the
    # locally-available model (e.g. nvidia/cosmos-reason2-8b) and set
    # DEFAULT_TEACHER_LOCAL_BASE_URL to its base URL — the project service
    # then probes the URL on create and rebinds the matching model_config to a
    # self_hosted endpoint, so the new project's first proposal stays local
    # rather than 404'ing at the hosted catalog.
    "DEFAULT_TEACHER_MODEL": MINIMAX_M3,
    "DEFAULT_TEACHER_LOCAL_BASE_URL": None,
}

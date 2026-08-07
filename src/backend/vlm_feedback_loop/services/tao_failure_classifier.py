# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Classify TAO eval failures by signature.

The Blueprint's NIM-eval-as-quality-fallback
fires only when the TAO failure signature matches a known
**upstream model-loader / runtime incompatibility**. Other TAO failures
(dataset shape mismatch, OOM, transient infra failure, schema mismatch
between training and eval, etc.) leave ``quality_status="failed"`` —
NIM eval is NOT a generic "TAO failed → use NIM instead" rescue, it's a
narrow workaround for the specific class of upstream bugs where the
trained checkpoint is HuggingFace-compatible but the cosmos-rl-bundled
vLLM cannot register its architecture.

Two known failure classes are documented:

* **Qwen3-VL-dense gap** — cosmos-rl 6.26.3 + Cosmos-Reason2:
  cosmos-rl's bundled vLLM lacks ``Qwen3VLForConditionalGeneration``,
  falls back through ``qwen2_5_vl.py``, hits the
  ``vocab_parallel_embedding.py`` assertion.
* **Weight-init loader gap** — cosmos-rl-evaluate vLLM raises
  ``ValueError: Following weights were not initialized from checkpoint``
  on Cosmos-Reason2-8B trained checkpoints, listing weight names from
  the deeper visual blocks and language-model layers. NIM 1.6.0's vLLM
  loads the same checkpoint cleanly.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from vlm_feedback_loop.db.models.tao_job import TAOJob

# Conservative pattern list. **Order matters** — the most-specific tokens
# go first so a casual ``AssertionError`` somewhere in a stack does not
# match. Plain ``"AssertionError"`` is intentionally excluded — it's too
# broad and would mis-trigger on legitimate dataset-shape failures or
# config-validation crashes.
#
# Add a new entry here when a new upstream-loader gap is reproduced
# end-to-end against TAO + a known-good fallback path. **Do not add**
# patterns that match generic Python exception types.
MODEL_LOADER_FAILURE_PATTERNS: tuple[str, ...] = (
    # Architecture name not registered in the cosmos-rl-bundled vLLM —
    # the dense Qwen3-VL family that powers Cosmos-Reason2 2B/8B.
    "Qwen3VLForConditionalGeneration",
    # The ``qwen2_5_vl`` fallback path that vLLM routes through when
    # the actual architecture (qwen3_vl) is unregistered.
    "qwen2_5_vl.py",
    # The vocab-parallel-embedding shape assertion that fires when the
    # nested Qwen3-VL key path is mis-routed through the flat Qwen2.5
    # loader.
    "vocab_parallel_embedding.py",
    # The Qwen3-VL nested embedding key path that the trained safetensors
    # carry but the qwen2_5_vl loader can't parse.
    "model.language_model.embed_tokens.weight",
    # Defensive: exact transformers/vLLM error wording variants.
    "Unsupported model type: qwen3_vl",
    # LoRA adapter-only checkpoints (the cosmos-rl default): the
    # in-chain evaluate feeds the adapter layout (``base_model.model.*``
    # safetensors keys) to the flat vLLM loader, which cannot resolve the
    # prefix. The checkpoint itself is healthy — the merged adapter serves
    # and scores through the §9.5 NIM path (observed live 2026-07-15).
    "There is no module or parameter named 'base_model'",
    # The vLLM checkpoint loader's post-load weight-init
    # validation fails on Cosmos-Reason2-8B trained checkpoints. The
    # accompanying weight-name list always includes ``visual.blocks.*``
    # and ``language_model.model.layers.*`` paths from the deeper layers
    # the loader's registered model class doesn't enumerate. NIM 1.6.0's
    # vLLM does NOT exhibit this gap on the same checkpoint (16.78 GiB
    # loaded in 1.86 s, 83/84 NIM eval success @ 83.1 % exact-match).
    # The phrase is vLLM-specific and
    # cannot fire on a Blueprint-context model corruption — every
    # Blueprint Student checkpoint comes from a TAO ``train`` action
    # whose ``succeeded`` status implies the checkpoint was fully
    # written.
    "Following weights were not initialized from checkpoint",
)


def first_matching_pattern(error_text: str | None) -> str | None:
    """Return the first loader-gap pattern found in ``error_text``, or None.

    ``error_text`` may come from any of: ``TAOJob.error_ref``,
    ``TAOJob.status_reason``, the contents of an attached ``tao_logs_ref``
    file, or a concatenation thereof. Pattern matching is case-sensitive
    on purpose — the loader / vLLM exception strings are stable. Patterns
    are tried in declaration order (most-specific first).

    Empty/None input returns None (cannot classify; remain conservative).
    """
    if not error_text:
        return None
    return next((p for p in MODEL_LOADER_FAILURE_PATTERNS if p in error_text), None)


def collect_failure_evidence(tao_job: TAOJob, *, max_chars: int = 65_536) -> str:
    """Concatenate every available failure-text field for signature matching.

    Returns a single string joining (in order): ``error_ref``,
    ``poll_error_ref``, ``chain_halted_reason``, ``outputs.tao_logs_text``
    (if present inline), ``outputs.tao_events_text``, and the contents of
    ``outputs.tao_logs_ref`` if it points at a readable file. Truncated to
    ``max_chars`` so signature matching stays bounded.

    Truncation keeps the **tail** of oversized log parts (and of the final
    join): container logs put the terminal traceback at the end, so a
    head-keeping cut discards exactly the text the signatures live in. The
    default covers the complete 64 KB TAO log payload that the polling service
    persists. A smaller 32 KB window missed the Qwen3-VL architecture marker
    at byte 32 175 of a live 65 536-byte FP8 evaluate log even though NIM
    loaded and evaluated the same checkpoint cleanly.
    """
    parts: list[str] = []

    for field in ("error_ref", "poll_error_ref", "chain_halted_reason"):
        v = getattr(tao_job, field, None)
        if v:
            parts.append(str(v))

    outputs = tao_job.outputs or {}
    for key in ("tao_logs_text", "tao_events_text"):
        v = outputs.get(key)
        if isinstance(v, str) and v:
            parts.append(v[-max_chars:])

    ref = outputs.get("tao_logs_ref")
    if isinstance(ref, str) and ref:
        try:
            with open(ref, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
                parts.append(content[-max_chars:])
        except OSError:
            pass  # best-effort; absence is non-fatal

    joined = "\n---\n".join(parts)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined


def find_failed_evaluate_job_for_student(
    session: Session,
    *,
    project_id: str,
    student_artifact_tao_job_id: str,
) -> TAOJob | None:
    """Return the most recent failed ``evaluate`` TAOJob whose parent produced
    the Student artifact, or ``None`` if no such job exists.

    Baseline Students point at their ``train`` job. Quantized Students point at
    their ``quantize`` job, whose paired evaluate is not parented by ``train``.
    Following the artifact parent keeps classification scoped to the exact
    checkpoint that NIM served.

    Used by the NIM-eval-as-quality-fallback gate to decide
    whether the prior TAO failure was a known upstream-loader gap.
    """
    return (
        session.query(TAOJob)
        .filter(
            TAOJob.project_id == project_id,
            TAOJob.parent_tao_job_id == student_artifact_tao_job_id,
            TAOJob.action == "evaluate",
            TAOJob.status.in_(("failed", "canceled")),
        )
        .order_by(TAOJob.completed_at.desc().nullslast())
        .first()
    )


def matches_known_loader_gap(
    session: Session,
    *,
    project_id: str,
    student_artifact_tao_job_id: str,
) -> tuple[bool, str | None]:
    """High-level helper: ``(matched, summary)`` describing whether the prior
    TAO eval failure for this Student matches a known upstream loader gap.

    Returns ``(False, None)`` when no failed TAO eval exists or evidence
    is empty. Returns ``(True, "<matched_pattern>")`` when at least one
    pattern fires. Caller decides what to do with the verdict.
    """
    failed = find_failed_evaluate_job_for_student(
        session,
        project_id=project_id,
        student_artifact_tao_job_id=student_artifact_tao_job_id,
    )
    if failed is None:
        return False, None
    evidence = collect_failure_evidence(failed)
    if not evidence:
        return False, None
    matched = first_matching_pattern(evidence)
    return matched is not None, matched


__all__ = [
    "MODEL_LOADER_FAILURE_PATTERNS",
    "collect_failure_evidence",
    "find_failed_evaluate_job_for_student",
    "first_matching_pattern",
    "matches_known_loader_gap",
]

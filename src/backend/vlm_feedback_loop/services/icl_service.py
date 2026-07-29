# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ICL selection and budget pruning service (Selective-K).

Queries eligible corrective examples, ranks them by CLIP relevance to the
query image, trims per query via the adaptive similarity gap, caps at the
model's depth default, and prunes the relevance tail when the token or
image budget binds.  Pure computation plus database reads — no external
HTTP calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.clip_embedding_service import (
    embedding_cache,
    ensure_embedding_cache_loaded,
)
from vlm_feedback_loop.services.review_selector_service import sim_clip
from vlm_feedback_loop.services.token_budget_service import (
    estimate_icl_example_tokens,
)

logger = logging.getLogger("vlm_feedback_loop.icl")

# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class ICLExample:
    """A single ICL example with metadata for selection and pruning."""

    example_key: str
    label_json: dict[str, Any]
    labeled_at: str
    phash: str | None = None
    clip_embedding: list[float] | None = None
    # Absolute filesystem path of the ICL example's image. Required for
    # inline ICL image injection — `prompt_service.invoke_teacher`
    # batch-prepares this alongside the query image so the model receives
    # `[icl_text, icl_image, …]` content-part pairs in the user message.
    # Optional only because callers and unit tests that do not exercise the
    # image-transport path may construct ICLExamples without plumbing a
    # path. `query_icl_candidates` populates it from the joined Example row.
    storage_ref: str | None = None


@dataclass
class ICLSelectionResult:
    """Result of the ICL selection + pruning pipeline."""

    examples: list[ICLExample] = field(default_factory=list[ICLExample])
    candidate_pool_size: int = 0
    selected_keys: list[str] = field(default_factory=list[str])
    total_count: int = 0
    pruned_count: int = 0
    pruned_keys: list[str] = field(default_factory=list[str])


def bookend_icl_examples(examples: list[ICLExample]) -> list[ICLExample]:
    """Place the two strongest selected examples at the prompt edges.

    Selection and budget pruning keep examples in descending relevance order.
    Bookending is a final presentation-only permutation: top-1 stays first,
    top-2 moves last, and the remaining examples keep their relative order in
    the middle. Applying it after pruning preserves relevance-tail budget
    semantics while giving the strongest examples both primacy and recency.
    """
    if len(examples) <= 2:
        return list(examples)
    return [examples[0], *examples[2:], examples[1]]


# ── ICL eligibility query ──────────────────────────────────────────────────


def count_icl_eligible_edits(
    session: Session, project_id: str, guidance_id: str | None
) -> int:
    """Count the Labels matching the ``ICL_ELIGIBLE`` predicate.

    Single home for the domain definition (see ``query_icl_candidates``):
    verified, not in the Test Pool, ``verified_outcome='Edit'``, and belonging
    to the given Guidance version. Evaluation triggers, the ``:icl_count``
    endpoint, and run bookkeeping all count against this one predicate. A
    ``None`` guidance_id matches nothing (count 0), same as the SQL it replaces.
    """
    return (
        session.execute(
            select(func.count())
            .select_from(Label)
            .where(
                Label.project_id == project_id,
                Label.label_status == "verified",
                Label.pool_assignment.is_(None),
                Label.verified_outcome == "Edit",
                Label.guidance_id == guidance_id,
            )
        ).scalar()
        or 0
    )


def query_icl_candidates(
    session: Session,
    project_id: str,
    active_guidance_id: str,
) -> list[ICLExample]:
    """Query eligible ICL examples.

    ``ICL_ELIGIBLE = Label WHERE label_status='verified'
    AND pool_assignment IS NULL AND verified_outcome='Edit'
    AND guidance_id = project.active_guidance_id``

    Returns candidates ordered by ``labeled_at DESC`` (newest first).
    """
    rows = session.execute(
        select(Label, Example.phash, Example.storage_ref)
        .join(
            Example,
            and_(
                Example.project_id == Label.project_id,
                Example.example_key == Label.example_key,
            ),
        )
        .where(
            Label.project_id == project_id,
            Label.label_status == "verified",
            Label.pool_assignment.is_(None),
            Label.verified_outcome == "Edit",
            Label.guidance_id == active_guidance_id,
        )
        # ``labeled_at`` is second-resolution. Use example_key as an explicit
        # tie-break so pool-growth prefixes are stable when a fast Teacher saves
        # multiple corrections in the same second.
        .order_by(Label.labeled_at.desc(), Label.example_key.asc())
    ).all()

    # Hydrate the in-memory embedding cache from the DB if this process hasn't
    # loaded it yet (e.g., after a backend restart — the cache is otherwise only
    # populated as a side effect of computing embeddings). Without this,
    # get_all() returns {} and relevance-ranked ICL selection silently
    # degrades to the newest-first fallback. Guarded by is_loaded so the bulk
    # DB read runs at most once per process per project.
    if not embedding_cache.is_loaded(project_id):
        proj = session.get(Project, project_id)
        dim = getattr(proj, "embedding_dim", None) if proj else None
        ensure_embedding_cache_loaded(project_id, session.get_bind(), dim)

    clip_vectors = embedding_cache.get_all(project_id)

    candidates: list[ICLExample] = []
    for label_row, phash, storage_ref in rows:
        candidates.append(
            ICLExample(
                example_key=label_row.example_key,
                label_json=label_row.label_json,
                labeled_at=label_row.labeled_at,
                phash=phash,
                clip_embedding=clip_vectors.get(label_row.example_key),
                storage_ref=storage_ref,
            )
        )
    return candidates


# ── ICL selection algorithm ────────────────────────────────────────────────


def adaptive_cutoff(
    sorted_sims: list[float],
    *,
    icl_sim_gap: float | None,
    icl_abs_threshold: float | None,
    icl_max: int | None,
) -> int:
    """Adaptive per-query ICL depth (similarity-gap stopping).

    Given the query→candidate CLIP cosine similarities ``sorted_sims`` in
    DESCENDING order (``sorted_sims[0]`` is the most-similar neighbor), return
    how many leading neighbors to keep. Neighbor ``i`` (``i >= 1``) is kept
    iff it is "close enough" to the query: when set, its similarity must satisfy
    BOTH the relative gap to the best neighbor and the absolute floor::

        sim_i >= sim_0 - icl_sim_gap     (when icl_sim_gap is not None)
        sim_i >= icl_abs_threshold       (when icl_abs_threshold is not None)

    Stop at the FIRST neighbor that fails either active rule. The top-1
    neighbor is always kept (``return >= 1`` whenever any candidate exists).
    The result is finally capped at ``icl_max``.

    Backward-compatibility contract: when BOTH ``icl_sim_gap`` and
    ``icl_abs_threshold`` are None this is a no-op — it returns ``icl_max``
    (or ``len(sorted_sims)`` when ``icl_max`` is None), so the caller's fixed-K
    behavior is unchanged.

    Args:
        sorted_sims: Query→candidate cosine similarities, descending.
        icl_sim_gap: Max allowed drop below ``sorted_sims[0]``; None disables.
        icl_abs_threshold: Absolute similarity floor; None disables.
        icl_max: Hard cap on the returned count; None means no cap.

    Returns:
        Number of leading neighbors to keep (``>= 1`` when non-empty, ``0``
        only when ``sorted_sims`` is empty), capped at ``icl_max``.
    """
    n = len(sorted_sims)
    if n == 0:
        return 0

    # No-op when neither rule is active: keep everything (capped by icl_max).
    if icl_sim_gap is None and icl_abs_threshold is None:
        return n if icl_max is None else min(n, icl_max)

    sim_best = sorted_sims[0]
    kept = 1  # always keep the top-1 neighbor
    for i in range(1, n):
        sim_i = sorted_sims[i]
        if icl_sim_gap is not None and sim_i < sim_best - icl_sim_gap:
            break
        if icl_abs_threshold is not None and sim_i < icl_abs_threshold:
            break
        kept = i + 1

    return kept if icl_max is None else min(kept, icl_max)


def select_icl_examples(
    candidates: list[ICLExample],
    *,
    icl_max_examples: int | None = None,
    query_clip_embedding: list[float] | None = None,
    icl_sim_gap: float | None = None,
    icl_abs_threshold: float | None = None,
) -> ICLSelectionResult:
    """Select ICL examples: relevance rank → adaptive cutoff → depth cap.

    Selective-K, established by the July 2026 depth studies: a small number
    of highly relevant exemplars beats context-filling, so selection is
    always relevance-ranked (query→candidate CLIP similarity, degrading to
    the incoming newest-first order when embeddings are unavailable — see
    :func:`_relevance_order`).

    Args:
        candidates: Already sorted by ``labeled_at DESC`` from
            :func:`query_icl_candidates`.
        icl_max_examples: Effective depth cap (per-run override or the
            model's ``default_icl_max_examples``, resolved by the caller).
            None means no cap.
        query_clip_embedding: CLIP embedding of the QUERY image; None
            degrades ranking and disables the adaptive cutoff.
        icl_sim_gap: Adaptive-K relative gap — after relevance ranking,
            keep the prefix of neighbors within ``sim_best - icl_sim_gap``.
            None disables.
        icl_abs_threshold: Adaptive-K absolute similarity floor; a neighbor
            must clear BOTH rules when both are set. None disables.

    Returns:
        An :class:`ICLSelectionResult` with selected examples.
    """
    if not candidates:
        return ICLSelectionResult()

    result = ICLSelectionResult(candidate_pool_size=len(candidates))

    # Step 1: Relevance rank (newest-first degrade inside the helper).
    ordered = _relevance_order(candidates, query_clip_embedding)

    # Step 2: Adaptive per-query ICL depth (similarity-gap stopping). The
    # relevance-ordered list is already descending by query similarity; keep
    # only the prefix of neighbors close enough to the query (gap-to-best
    # AND/OR absolute floor), always keeping >= 1. ``None/None`` for both
    # knobs makes ``adaptive_cutoff`` a no-op.
    if (icl_sim_gap is not None or icl_abs_threshold is not None) and (
        query_clip_embedding
    ):
        sorted_sims = [
            sim_clip(query_clip_embedding, ex.clip_embedding)
            if ex.clip_embedding is not None
            else -1.0
            for ex in ordered
        ]
        keep_n = adaptive_cutoff(
            sorted_sims,
            icl_sim_gap=icl_sim_gap,
            icl_abs_threshold=icl_abs_threshold,
            icl_max=None,  # the depth cap is applied by Step 3 below
        )
        ordered = ordered[:keep_n]

    # Step 3: Depth cap — keep the head (most relevant).
    selected = ordered[:icl_max_examples] if icl_max_examples is not None else ordered

    result.examples = list(selected)
    result.selected_keys = [ex.example_key for ex in selected]
    result.total_count = len(selected)

    return result


def _relevance_order(
    candidates: list[ICLExample],
    query_clip_embedding: list[float] | None,
) -> list[ICLExample]:
    """Order candidates by descending CLIP similarity to the QUERY image.

    Surfaces the Edits whose images are most similar to the query, so the
    depth cap and tail-dropping budget pruners retain the most *relevant*
    corrections. This targets ICL's intended use — "a corrected image teaches
    a visually-similar query its label" (Overview S3) — which a query-agnostic
    ordering cannot. Candidates without a CLIP embedding (or when the query
    embedding is unavailable) fall back to the incoming newest-first order.
    Deterministic ascending-``example_key`` tie-break.
    """
    if not query_clip_embedding or len(candidates) <= 1:
        return list(candidates)
    query_vec: list[float] = query_clip_embedding
    with_emb = [ex for ex in candidates if ex.clip_embedding is not None]
    without_emb = [ex for ex in candidates if ex.clip_embedding is None]
    if not with_emb:
        return list(candidates)

    def _key(ex: ICLExample) -> tuple[float, str]:
        emb = ex.clip_embedding
        score = -sim_clip(query_vec, emb) if emb is not None else 0.0
        return (score, ex.example_key)

    scored = sorted(with_emb, key=_key)
    # Embedding-less candidates keep their incoming order, appended last so the
    # budget pruner prefers the ranked-relevant ones.
    return scored + without_emb


# ── Budget pruning (relevance-tail dropping) ───────────────────────────────


def prune_icl_by_budget(
    selected: list[ICLExample],
    max_icl_tokens: int,
    fields: list[dict[str, Any]],
    generation_order: list[str],
    icl_field_mode: Literal["all", "aux_and_core", "core_only"],
) -> tuple[list[ICLExample], list[str]]:
    """Drop from the END of the selection-ordered list until tokens fit.

    ``selected`` arrives relevance-ordered from :func:`select_icl_examples`
    (newest-first in the embedding-less degrade), so the tail is always the
    least query-relevant exemplar — the correct victim under Selective-K.
    May empty the list entirely; the caller's cold-start render handles that.

    Returns ``(retained, dropped_keys)``.
    """
    retained = list(selected)
    dropped: list[str] = []

    while (
        retained
        and _estimate_total_icl_tokens(
            retained, fields, generation_order, icl_field_mode
        )
        > max_icl_tokens
    ):
        dropped.append(retained.pop().example_key)

    return retained, dropped


def prune_icl_by_image_budget(
    selected: list[ICLExample],
    max_icl_images: int,
) -> tuple[list[ICLExample], list[str]]:
    """Cap retained ICL at ``max_icl_images`` for inline image injection.

    Keeps the selection-order head (the most query-relevant exemplars) and
    drops the tail. ``max_images_per_request`` is a hard wire-format
    constraint; the image-budget invariant
    (``icl_images_attached_count == len(retained)`` post-pruning) is why
    the cap MUST be enforced by dropping whole examples rather than
    degrading them to text-only labels.

    Special case: ``max_icl_images <= 0`` (model with
    ``max_images_per_request <= 1`` — no room beyond the query image)
    drops everything. Caller falls through to the cold-start render path.

    Returns ``(retained, dropped_keys)``.
    """
    if max_icl_images <= 0:
        return [], [ex.example_key for ex in selected]

    return (
        list(selected[:max_icl_images]),
        [ex.example_key for ex in selected[max_icl_images:]],
    )


def _estimate_total_icl_tokens(
    examples: list[ICLExample],
    fields: list[dict[str, Any]],
    generation_order: list[str],
    icl_field_mode: Literal["all", "aux_and_core", "core_only"],
) -> int:
    """Estimate total tokens for all ICL examples."""
    total = 0
    for ex in examples:
        total += estimate_icl_example_tokens(
            ex.label_json, fields, generation_order, icl_field_mode
        )
    return total


# ── Logging helpers ────────────────────────────────────────────────────────


def log_icl_selection(
    project_id: str,
    result: ICLSelectionResult,
) -> None:
    """Emit the structured ICL-selection log line.

    DEBUG by default; escalates to INFO when pruning drops examples.
    """
    details: dict[str, Any] = {
        "candidate_pool_size": result.candidate_pool_size,
        "selected_keys": result.selected_keys,
        "total_count": result.total_count,
        "pruned_count": result.pruned_count,
    }
    if result.pruned_keys:
        details["pruned_keys"] = result.pruned_keys

    level = logging.INFO if result.pruned_count > 0 else logging.DEBUG
    logger.log(
        level,
        "ICL selection: %d candidates → %d selected%s",
        result.candidate_pool_size,
        result.total_count,
        f" ({result.pruned_count} pruned)" if result.pruned_count > 0 else "",
        extra={
            "component": "icl_selection",
            "project_id": project_id,
            "details": details,
        },
    )

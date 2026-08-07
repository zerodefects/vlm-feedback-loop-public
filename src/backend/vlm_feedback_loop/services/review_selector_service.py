# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review selector service.

Greedy max-min diversity algorithm that chooses the next image to present
to the SME.  Two similarity tiers (pHash hamming, CLIP cosine) with
automatic switchover.  Persists scheduler state on the Project record
for reproducibility.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.clip_embedding_service import (
    embedding_cache,
    ensure_embedding_cache_loaded,
    phash_hamming_distance,
)
from vlm_feedback_loop.services.greedy_max_min import greedy_max_min_indices

logger = logging.getLogger("vlm_feedback_loop.review_selector")

# ── Constants ──────────────────────────────────────────────────────────────

HASH_BITS = 64  # 64-bit pHash

# Track last mode per project for log-point-7 escalation. This is an
# unsynchronized module-global read-modify-write, which is safe here because
# the architecture is single-user with one active session per project:
# review selections for a given project are serialized by the SME's
# save→next click loop, so no two select_next calls race the same key. The
# value is a mode HINT for switchover hysteresis, not correctness state — the
# authoritative scheduler state lives on the Project row inside the DB
# transaction. A future multi-reviewer mode would need to move this into that
# per-project row (or a lock) alongside the rest of the scheduler state.
_last_mode_by_project: dict[str, str] = {}


# ── Similarity functions ───────────────────────────────────────────────────


def sim_phash(a: str | None, b: str | None) -> float:
    """pHash similarity: ``1 - hamming_distance / hash_bits``.

    Returns 0.0 when either hash is None.
    """
    if a is None or b is None:
        return 0.0
    return 1.0 - phash_hamming_distance(a, b) / HASH_BITS


def sim_clip(a: list[float], b: list[float]) -> float:
    """CLIP cosine similarity: ``dot(a,b) / (||a|| * ||b||)``.

    ``strict=True`` surfaces a dimension mismatch (raises ``ValueError``)
    instead of silently truncating to the shorter vector and returning a
    meaningless score that would misorder the whole review/ICL queue.
    Embeddings are dimension-locked to ``EMBEDDING_DIM`` and invalidated on
    a model switch, so a mismatch here is a real data-integrity bug worth
    failing loudly.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Core algorithm ─────────────────────────────────────────────────────────


def _determine_selection_mode(
    eligible_clip_count: int,
    review_selection_mode: str,
    clip_switchover_min_count: int,
) -> str:
    """Determine the active selection mode.

    Returns ``"clip_diverse"`` or ``"phash_diverse"``.
    """
    if review_selection_mode == "phash_diverse":
        return "phash_diverse"
    if review_selection_mode == "clip_diverse":
        # Force CLIP but fall back to pHash below threshold
        if eligible_clip_count >= clip_switchover_min_count:
            return "clip_diverse"
        return "phash_diverse"
    # auto: prefer CLIP when threshold met
    if eligible_clip_count >= clip_switchover_min_count:
        return "clip_diverse"
    return "phash_diverse"


def _window_max_similarity(
    candidates: list[dict[str, Any]],
    window: list[dict[str, Any]],
    mode: str,
    clip_matrix: tuple[dict[str, int], NDArray[np.float32]] | None = None,
) -> list[float]:
    """Max similarity of each candidate to the reference ``window``.

    Computes ``max(sim(candidate, w) for w in window)`` over every
    candidate — the quantity the greedy max-min selector minimizes. Both
    lists are non-empty and carry the active signal for ``mode`` (the
    caller guarantees this). The result is a list aligned with
    ``candidates``.

    ``clip_diverse``: when ``clip_matrix`` — the ``(key→row, L2-normalized
    matrix)`` view from the embedding cache — is supplied, the whole N×M
    cosine table is one matrix multiply, turning seconds of per-pair
    Python cosine at real pool sizes into milliseconds. Without it, the
    exact per-pair ``sim_clip`` is used (bit-identical, for callers with
    no cache-backed matrix). ``phash_diverse``: pHash hamming via numpy
    popcount — bit-identical to ``sim_phash`` (integer distance, and
    ``k / 64`` is exact in float64).
    """
    if mode == "clip_diverse":
        if clip_matrix is None:
            return [
                max(sim_clip(c["clip"], h["clip"]) for h in window) for c in candidates
            ]
        index, matrix = clip_matrix
        cand = matrix[[index[c["key"]] for c in candidates]]
        ref = matrix[[index[w["key"]] for w in window]]
        sims = cand @ ref.T
    else:
        cand = np.array([int(c["phash"], 16) for c in candidates], dtype=np.uint64)
        ref = np.array([int(w["phash"], 16) for w in window], dtype=np.uint64)
        hamming = np.bitwise_count(cand[:, None] ^ ref[None, :])
        sims = 1.0 - hamming.astype(np.float64) / HASH_BITS
    return [float(x) for x in sims.max(axis=1)]


def _greedy_select_next(
    candidates: list[dict[str, Any]],
    recent_window: list[dict[str, Any]],
    mode: str,
    clip_matrix: tuple[dict[str, int], NDArray[np.float32]] | None = None,
) -> tuple[str | None, float | None]:
    """Greedy max-min diversity selection.

    Each candidate/recent dict has:
      - ``key``: example_key
      - ``phash``: hex string or None
      - ``clip``: float list or None  (populated only in clip_diverse mode)

    Returns ``(selected_example_key, diversity_score)`` where
    *diversity_score* is the max-similarity score of the chosen candidate
    (lower is more diverse), or None when no scoring occurred.
    """
    if not candidates:
        return None, None

    # Keep only candidates that carry the active signal (CLIP vector
    # or pHash, depending on mode).
    if mode == "clip_diverse":
        c_with_signal = [c for c in candidates if c["clip"] is not None]
    else:
        c_with_signal = [c for c in candidates if c["phash"] is not None]

    # No candidate carries the signal — deterministic fallback to
    # ascending example_key.
    if not c_with_signal:
        candidates_sorted = sorted(candidates, key=lambda c: c["key"])
        return candidates_sorted[0]["key"], None

    # Restrict the recent window to entries that carry the same signal.
    if mode == "clip_diverse":
        h_sig = [h for h in recent_window if h["clip"] is not None]
    else:
        h_sig = [h for h in recent_window if h["phash"] is not None]

    # Empty recent window: nothing to diversify against, so select the
    # first candidate.  Candidates arrive pre-sorted by the caller
    # (Tier 1 priority order or natural order).  Among those with the
    # active signal, pick the first; if there's a tie in position, fall
    # back to ascending key.
    if not h_sig:
        return c_with_signal[0]["key"], None

    # Greedy max-min via the shared core. Argmin of the max
    # similarity to the window ≡ argmax of min(-similarity); tie-break
    # by ascending key. A single pick against a fixed reference set, so
    # no pairwise candidate distance is needed.
    #
    # The per-candidate max similarity is computed once, vectorized;
    # ``init_dist`` then only indexes that array, leaving the shared
    # greedy loop's deterministic ascending-key tie-break untouched.
    max_sim = _window_max_similarity(c_with_signal, h_sig, mode, clip_matrix)

    picked = greedy_max_min_indices(
        [c["key"] for c in c_with_signal],
        init_dist=lambda i: -max_sim[i],
        n_select=1,
    )
    best_i = picked[0]
    return c_with_signal[best_i]["key"], max_sim[best_i]


def _log_selection(
    project_id: str,
    mode: str,
    candidate_count: int,
    selected_key: str | None,
    diversity_score: float | None,
    mode_switched: bool = False,
) -> None:
    """Log point 7: review_selector.

    Emits at DEBUG by default.  Escalates to INFO when the selection mode
    switches between CLIP-diverse and pHash-diverse.
    """
    details: dict[str, Any] = {
        "selection_mode": mode,
        "candidate_set_size": candidate_count,
        "selected_example_key": selected_key,
    }
    if diversity_score is not None:
        details["diversity_score"] = round(diversity_score, 6)

    level = logging.INFO if mode_switched else logging.DEBUG
    suffix = "; mode switched" if mode_switched else ""
    logger.log(
        level,
        "Review selector: selected %s (%s mode, %d candidates%s)",
        selected_key,
        mode,
        candidate_count,
        suffix,
        extra={
            "component": "review_selector",
            "project_id": project_id,
            "details": details,
        },
    )


# ── Public API ─────────────────────────────────────────────────────────────


class ReviewSelectorResult:
    """Result of a review selector selection."""

    __slots__ = (
        "example_key",
        "example_state",
        "has_existing_label",
        "selection_mode",
        "queue_empty",
        "storage_ref",
        "prior_verified_label_ref",
    )

    def __init__(
        self,
        example_key: str | None,
        example_state: str | None,
        has_existing_label: bool,
        selection_mode: str,
        queue_empty: bool,
        storage_ref: str | None = None,
        prior_verified_label_ref: str | None = None,
    ) -> None:
        self.example_key = example_key
        self.example_state = example_state
        self.has_existing_label = has_existing_label
        self.selection_mode = selection_mode
        self.queue_empty = queue_empty
        # Selected Example's disk path and prior-label snapshot JSON —
        # carried on the response so the labeling screen does not re-scan
        # the examples list for the record it was just handed.
        self.storage_ref = storage_ref
        self.prior_verified_label_ref = prior_verified_label_ref


def select_next(
    engine: Any,
    project: Project,
    review_selection_mode: str = "auto",
    review_recent_window_k: int = 20,
    clip_switchover_min_count: int = 50,
) -> ReviewSelectorResult:
    """Select the next example for the SME to review.

    Implements the full selection algorithm:
    1. Query eligible examples (Unlabeled, Auto-Labeled)
    2. Partition into Tier 1 (prior labels) and Tier 2 (standard)
    3. Apply prior-label priority within Tier 1
    4. Select using greedy max-min diversity
    5. Persist scheduler state on Project record
    """
    project_id = project.project_id

    with Session(engine) as session:
        # ── Load eligible examples ────────────────────────────────────
        eligible = (
            session.execute(
                select(Example).where(
                    Example.project_id == project_id,
                    Example.state.in_(["Unlabeled", "Auto-Labeled"]),
                )
            )
            .scalars()
            .all()
        )

        if not eligible:
            return ReviewSelectorResult(
                example_key=None,
                example_state=None,
                has_existing_label=False,
                selection_mode="phash_diverse",
                queue_empty=True,
            )

        # ── Count eligible examples with CLIP embeddings ──────────────
        eligible_clip_count = sum(1 for e in eligible if e.clip_embedding_present)

        # ── Determine selection mode ──────────────────────────────────
        mode = _determine_selection_mode(
            eligible_clip_count,
            review_selection_mode,
            clip_switchover_min_count,
        )

        # ── Detect mode switch for log escalation ─────────────────────
        previous_mode = _last_mode_by_project.get(project_id)
        mode_switched = previous_mode is not None and previous_mode != mode

        # ── Load scheduler state ──────────────────────────────────────
        sched_state = project.review_selector_scheduler_state or {}
        recent_keys: list[str] = sched_state.get("recent_window", [])

        # ── Build lookup and load CLIP vectors ────────────────────────
        eligible_by_key = {e.example_key: e for e in eligible}

        clip_vectors: dict[str, list[float]] = {}
        clip_matrix: tuple[dict[str, int], NDArray[np.float32]] | None = None
        if mode == "clip_diverse":
            ensure_embedding_cache_loaded(project_id, engine, project.embedding_dim)
            clip_vectors = embedding_cache.get_all(project_id)
            clip_matrix = embedding_cache.get_normalized_matrix(project_id)

        def _make_entry(ex: Example) -> dict[str, Any]:
            return {
                "key": ex.example_key,
                "phash": ex.phash,
                "clip": (
                    clip_vectors.get(ex.example_key) if mode == "clip_diverse" else None
                ),
            }

        # ── Build recent window entries (batched query) ───────────────
        recent_window_keys = recent_keys[-review_recent_window_k:]
        non_eligible_keys = [
            rk for rk in recent_window_keys if rk not in eligible_by_key
        ]

        non_eligible_map: dict[str, Example] = {}
        if non_eligible_keys:
            non_eligible_rows = (
                session.execute(
                    select(Example).where(
                        Example.project_id == project_id,
                        Example.example_key.in_(non_eligible_keys),
                    )
                )
                .scalars()
                .all()
            )
            non_eligible_map = {ex.example_key: ex for ex in non_eligible_rows}

        recent_entries: list[dict[str, Any]] = []
        for rk in recent_window_keys:
            if rk in eligible_by_key:
                recent_entries.append(_make_entry(eligible_by_key[rk]))
            elif rk in non_eligible_map:
                recent_entries.append(_make_entry(non_eligible_map[rk]))

        # ── Prior-label priority ──────────────────────────────────────
        tier1: list[Example] = []
        tier2: list[Example] = []
        for ex in eligible:
            if ex.prior_verified_label_ref is not None:
                tier1.append(ex)
            else:
                tier2.append(ex)

        context_key = project.schema_change_context_example_key

        def _tier1_sort_key(ex: Example) -> tuple[int, int, str]:
            is_context = 0 if (context_key and ex.example_key == context_key) else 1
            is_edit = 0 if ex.prior_verified_outcome == "Edit" else 1
            return (is_context, is_edit, ex.example_key)

        tier1.sort(key=_tier1_sort_key)
        tier2.sort(key=lambda ex: ex.example_key)

        # Exhaust Tier 1 before Tier 2
        active_tier = tier1 if tier1 else tier2
        candidates = [_make_entry(ex) for ex in active_tier]

        # ── Run selection algorithm ───────────────────────────────────
        selected_key, diversity_score = _greedy_select_next(
            candidates, recent_entries, mode, clip_matrix
        )

        # ── Log point 7: review_selector ──────────────────────────────
        _log_selection(
            project_id,
            mode,
            len(candidates),
            selected_key,
            diversity_score,
            mode_switched=mode_switched,
        )
        _last_mode_by_project[project_id] = mode

        if selected_key is None:
            return ReviewSelectorResult(
                example_key=None,
                example_state=None,
                has_existing_label=False,
                selection_mode=mode,
                queue_empty=True,
            )

        # ── Look up state and existing label ──────────────────────────
        # Capture scalar columns now: the session commits (and expires
        # instances) before the return below.
        selected_example = eligible_by_key[selected_key]
        example_state = selected_example.state
        storage_ref = selected_example.storage_ref
        prior_verified_label_ref = selected_example.prior_verified_label_ref

        has_existing_label = False
        if example_state == "Auto-Labeled":
            existing_label = session.execute(
                select(Label).where(
                    Label.project_id == project_id,
                    Label.example_key == selected_key,
                    Label.label_status == "auto_labeled",
                )
            ).scalar_one_or_none()
            has_existing_label = existing_label is not None

        # ── Update scheduler state ────────────────────────────────────
        new_recent = list(recent_keys)
        new_recent.append(selected_key)
        if len(new_recent) > review_recent_window_k:
            new_recent = new_recent[-review_recent_window_k:]

        new_state = {"recent_window": new_recent}

        # ── Persist state and clear context key if presented ──────────
        proj = session.get(Project, project_id)
        if proj is not None:
            proj.review_selector_scheduler_state = new_state
            if context_key and selected_key == context_key:
                proj.schema_change_context_example_key = None
            session.commit()

    return ReviewSelectorResult(
        example_key=selected_key,
        example_state=example_state,
        has_existing_label=has_existing_label,
        selection_mode=mode,
        queue_empty=False,
        storage_ref=storage_ref,
        prior_verified_label_ref=prior_verified_label_ref,
    )

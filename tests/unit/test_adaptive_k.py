# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for adaptive per-query ICL depth (similarity-gap stopping).

Covers:
  - ``adaptive_cutoff`` pure helper: gap-only, threshold-only, both
    (intersection), always-keep->=1, icl_max cap, and the None/None no-op.
  - ``select_icl_examples`` integration: the cutoff trims the
    relevance-ranked prefix and is a no-op for the None/None default and
    when no query embedding is available.
"""

from __future__ import annotations

import pytest

from vlm_feedback_loop.services.icl_service import (
    ICLExample,
    adaptive_cutoff,
    select_icl_examples,
)
from vlm_feedback_loop.services.review_selector_service import sim_clip

# ── Pure helper: adaptive_cutoff ────────────────────────────────────────────


class TestAdaptiveCutoff:
    """``adaptive_cutoff(sorted_sims, gap, threshold, icl_max) -> n``.

    ``sorted_sims`` is DESCENDING (sorted_sims[0] is the most-similar
    neighbor). Keep the top-1, then keep neighbor i (i>=1) iff it clears
    every active rule; stop at the first failure; finally cap at icl_max.
    """

    def test_gap_only_keeps_prefix_within_gap(self):
        # best=0.90; gap=0.10 -> floor 0.80. 0.85,0.82 pass; 0.70 fails -> stop.
        sims = [0.90, 0.85, 0.82, 0.70, 0.60]
        n = adaptive_cutoff(
            sims, icl_sim_gap=0.10, icl_abs_threshold=None, icl_max=None
        )
        assert n == 3  # 0.90, 0.85, 0.82

    def test_gap_only_stops_at_first_failure_not_best_match(self):
        # A later neighbor back inside the gap window is NOT recovered: we stop
        # at the FIRST failure (0.70 < 0.80), even though 0.81 would pass.
        sims = [0.90, 0.85, 0.70, 0.81]
        n = adaptive_cutoff(
            sims, icl_sim_gap=0.10, icl_abs_threshold=None, icl_max=None
        )
        assert n == 2  # 0.90, 0.85 — 0.70 fails, stop before 0.81

    def test_threshold_only_keeps_those_above_floor(self):
        # floor=0.50; 0.90,0.60,0.55 pass; 0.40 fails -> stop.
        sims = [0.90, 0.60, 0.55, 0.40, 0.45]
        n = adaptive_cutoff(
            sims, icl_sim_gap=None, icl_abs_threshold=0.50, icl_max=None
        )
        assert n == 3

    def test_both_is_intersection(self):
        # best=0.90; gap=0.20 -> gap-floor 0.70. abs floor=0.75.
        # Effective floor = max(0.70, 0.75) = 0.75 for each neighbor (it must
        # clear BOTH). 0.80 passes both; 0.72 clears the gap (>=0.70) but fails
        # the abs floor (<0.75) -> stop. Intersection is stricter than either.
        sims = [0.90, 0.80, 0.72, 0.71]
        n_both = adaptive_cutoff(
            sims, icl_sim_gap=0.20, icl_abs_threshold=0.75, icl_max=None
        )
        assert n_both == 2  # 0.90, 0.80
        # Gap-only would keep 0.90,0.80,0.72,0.71 (all >= 0.70).
        n_gap = adaptive_cutoff(
            sims, icl_sim_gap=0.20, icl_abs_threshold=None, icl_max=None
        )
        assert n_gap == 4
        # Abs-only would keep 0.90,0.80 (>=0.75), stop at 0.72.
        n_abs = adaptive_cutoff(
            sims, icl_sim_gap=None, icl_abs_threshold=0.75, icl_max=None
        )
        assert n_abs == 2

    def test_always_keeps_at_least_one(self):
        # Even when neighbor #2 immediately fails both rules, keep the top-1.
        sims = [0.30, 0.10, 0.05]
        n = adaptive_cutoff(
            sims, icl_sim_gap=0.05, icl_abs_threshold=0.90, icl_max=None
        )
        assert n == 1

    def test_top1_kept_even_below_abs_threshold(self):
        # The top-1 is unconditional: it is kept even if it is itself below the
        # absolute floor (the rule only ever filters neighbor i>=2).
        sims = [0.40]
        n = adaptive_cutoff(
            sims, icl_sim_gap=None, icl_abs_threshold=0.90, icl_max=None
        )
        assert n == 1

    def test_capped_at_icl_max(self):
        # All five neighbors pass the gap rule, but icl_max=2 caps the result.
        sims = [0.90, 0.89, 0.88, 0.87, 0.86]
        n = adaptive_cutoff(sims, icl_sim_gap=0.10, icl_abs_threshold=None, icl_max=2)
        assert n == 2

    def test_none_none_is_noop_returns_icl_max(self):
        # No knobs set -> keep everything, capped by icl_max (fixed-K behavior).
        sims = [0.90, 0.50, 0.10, 0.05]
        assert (
            adaptive_cutoff(sims, icl_sim_gap=None, icl_abs_threshold=None, icl_max=3)
            == 3
        )
        # icl_max=None with no knobs keeps the full list length.
        assert (
            adaptive_cutoff(
                sims, icl_sim_gap=None, icl_abs_threshold=None, icl_max=None
            )
            == 4
        )

    def test_empty_returns_zero(self):
        assert (
            adaptive_cutoff([], icl_sim_gap=0.1, icl_abs_threshold=0.5, icl_max=5) == 0
        )

    def test_cap_applies_below_adaptive_count(self):
        # Adaptive keeps 4 by the gap rule but icl_max=1 wins (min of the two).
        sims = [0.99, 0.98, 0.97, 0.96]
        n = adaptive_cutoff(sims, icl_sim_gap=0.5, icl_abs_threshold=None, icl_max=1)
        assert n == 1


# ── Integration: select_icl_examples ────────────────────────────────────────


class TestSelectIclAdaptiveK:
    """The cutoff is wired into ``select_icl_examples`` for the relevance
    policy only, between relevance rank and the ``icl_max_examples`` cap.
    """

    def _ex(self, key: str, clip: list[float]) -> ICLExample:
        return ICLExample(
            example_key=key,
            label_json={"damage_type": "crack"},
            labeled_at="2026-01-01T00:00:00Z",
            clip_embedding=clip,
        )

    def _relevance_cands(self) -> list[ICLExample]:
        # Orthogonal one-hot embeddings so each candidate's similarity to the
        # query is independent. ``sim_clip`` normalizes both vectors, so the
        # actual sims are the query components divided by ||query||; relevance
        # ranking is therefore e0 > e1 > e2 > e3 (by descending query weight).
        return [
            self._ex("e0", [1.0, 0.0, 0.0, 0.0]),
            self._ex("e1", [0.0, 1.0, 0.0, 0.0]),
            self._ex("e2", [0.0, 0.0, 1.0, 0.0]),
            self._ex("e3", [0.0, 0.0, 0.0, 1.0]),
        ]

    # Descending query weights -> descending relevance e0..e3. The exact sim
    # VALUES depend on ``sim_clip``'s cosine normalization, so the tests derive
    # gap/threshold from the real sims rather than hard-coding magnitudes.
    _QUERY = [0.90, 0.60, 0.30, 0.05]

    def _sims(self, cands: list[ICLExample]) -> list[float]:
        return [sim_clip(self._QUERY, ex.clip_embedding) for ex in cands]

    def test_gap_truncates_low_relevance_tail(self):
        cands = self._relevance_cands()
        s = self._sims(cands)  # descending: s[0] > s[1] > s[2] > s[3]
        # Gap window admits e1 (within s0-gap) but excludes e2: pick a gap that
        # straddles s[1] and s[2] -> keep e0, e1; stop at e2.
        gap = (s[0] - s[1] + s[0] - s[2]) / 2.0
        r = select_icl_examples(
            cands,
            icl_max_examples=4,
            query_clip_embedding=self._QUERY,
            icl_sim_gap=gap,
        )
        assert r.selected_keys == ["e0", "e1"]

    def test_threshold_truncates(self):
        cands = self._relevance_cands()
        s = self._sims(cands)
        # Floor between s[1] and s[2] -> keep e0, e1; e2 falls below the floor.
        floor = (s[1] + s[2]) / 2.0
        r = select_icl_examples(
            cands,
            icl_max_examples=4,
            query_clip_embedding=self._QUERY,
            icl_abs_threshold=floor,
        )
        assert r.selected_keys == ["e0", "e1"]

    def test_icl_max_caps_below_adaptive_count(self):
        # Wide gap keeps all 4 adaptively, but icl_max_examples=1 wins.
        cands = self._relevance_cands()
        r = select_icl_examples(
            cands,
            icl_max_examples=1,
            query_clip_embedding=self._QUERY,
            icl_sim_gap=2.0,  # >> any sim drop -> adaptive keeps all 4
        )
        assert r.selected_keys == ["e0"]

    def test_always_keeps_top1(self):
        # A floor above the BEST sim rejects every neighbor i>=2; the top-1
        # (e0) is unconditional and is still returned.
        cands = self._relevance_cands()
        s = self._sims(cands)
        r = select_icl_examples(
            cands,
            icl_max_examples=4,
            query_clip_embedding=self._QUERY,
            icl_abs_threshold=s[0] + 0.01,
        )
        assert r.selected_keys == ["e0"]

    def test_none_none_is_fixed_k_identical(self):
        # No knobs -> identical to the existing fixed-K relevance selection.
        cands = self._relevance_cands()
        baseline = select_icl_examples(
            cands,
            icl_max_examples=4,
            query_clip_embedding=self._QUERY,
        )
        adaptive_off = select_icl_examples(
            cands,
            icl_max_examples=4,
            query_clip_embedding=self._QUERY,
            icl_sim_gap=None,
            icl_abs_threshold=None,
        )
        assert adaptive_off.selected_keys == baseline.selected_keys
        assert baseline.selected_keys == ["e0", "e1", "e2", "e3"]

    def test_noop_without_query_embedding(self):
        # Relevance with knobs but no query embedding -> falls back to incoming
        # order with no cutoff (cannot compute per-candidate sims).
        cands = self._relevance_cands()
        r = select_icl_examples(
            cands,
            icl_max_examples=4,
            query_clip_embedding=None,
            icl_sim_gap=0.01,
        )
        assert r.selected_keys == ["e0", "e1", "e2", "e3"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Determinism equivalence guard for the shared greedy max-min core.

The greedy max-min selection loop feeds persisted state — Test-Pool
membership, the review order, and the pHash-diverse embedding order —
so its tie-breaking and float-comparison semantics must never drift. Each
oracle below is a **verbatim transcription of the per-site
implementation that ``services/greedy_max_min.py`` replaced**; the
tests replay tie-heavy fixed and seeded-random inputs through both the
oracle and the refactored production function and assert identical
output order.

If one of these fails after a change to ``greedy_max_min_indices`` (or
a call-site mapping), the change altered selection order for at least
one input — treat that as a behavioral regression, not a test to
update.
"""

from __future__ import annotations

import random
from typing import Any

from vlm_feedback_loop.services.pool_service import _greedy_max_min_select
from vlm_feedback_loop.services.review_selector_service import (
    _greedy_select_next,
    sim_clip,
    sim_phash,
)

# ── Verbatim oracles (pre-consolidation implementations) ───────────────────


def _oracle_pool_greedy_max_min_select(
    candidates: list[dict[str, Any]],
    pool_ref: list[dict[str, Any]],
    mode: str,
    n_needed: int,
) -> list[str]:
    """pool_service._greedy_max_min_select, pre-refactor."""
    if n_needed <= 0 or not candidates:
        return []

    if mode == "clip_diverse":
        c_with = [c for c in candidates if c["clip"] is not None]
        c_without = [c for c in candidates if c["clip"] is None]
        p_ref = [p for p in pool_ref if p["clip"] is not None]
    else:
        c_with = [c for c in candidates if c["phash"] is not None]
        c_without = [c for c in candidates if c["phash"] is None]
        p_ref = [p for p in pool_ref if p["phash"] is not None]

    selected: list[str] = []

    for _ in range(n_needed):
        if not c_with:
            break

        if not p_ref:
            c_with.sort(key=lambda c: c["key"])
            chosen = c_with[0]
        else:
            best_key: str | None = None
            best_score = float("inf")
            chosen = c_with[0]

            for c in c_with:
                if mode == "clip_diverse":
                    score = max(sim_clip(c["clip"], r["clip"]) for r in p_ref)
                else:
                    score = max(sim_phash(c["phash"], r["phash"]) for r in p_ref)

                if score < best_score or (
                    score == best_score and (best_key is None or c["key"] < best_key)
                ):
                    best_score = score
                    best_key = c["key"]
                    chosen = c

        selected.append(chosen["key"])
        p_ref.append(chosen)
        c_with.remove(chosen)

    remaining = n_needed - len(selected)
    if remaining > 0 and c_without:
        c_without.sort(key=lambda c: c["key"])
        for c in c_without[:remaining]:
            selected.append(c["key"])

    return selected


def _oracle_review_greedy_select_next(
    candidates: list[dict[str, Any]],
    recent_window: list[dict[str, Any]],
    mode: str,
) -> tuple[str | None, float | None]:
    """review_selector_service._greedy_select_next, pre-refactor (A.3)."""
    if not candidates:
        return None, None

    if mode == "clip_diverse":
        c_with_signal = [c for c in candidates if c["clip"] is not None]
    else:
        c_with_signal = [c for c in candidates if c["phash"] is not None]

    if not c_with_signal:
        candidates_sorted = sorted(candidates, key=lambda c: c["key"])
        return candidates_sorted[0]["key"], None

    if mode == "clip_diverse":
        h_sig = [h for h in recent_window if h["clip"] is not None]
    else:
        h_sig = [h for h in recent_window if h["phash"] is not None]

    if not h_sig:
        return c_with_signal[0]["key"], None

    best_key: str | None = None
    best_score = float("inf")

    for c in c_with_signal:
        if mode == "clip_diverse":
            score = max(sim_clip(c["clip"], h["clip"]) for h in h_sig)
        else:
            score = max(sim_phash(c["phash"], h["phash"]) for h in h_sig)

        if score < best_score or (
            score == best_score and (best_key is None or c["key"] < best_key)
        ):
            best_score = score
            best_key = c["key"]

    return best_key, best_score


# ── Input generators (tie-heavy on purpose) ─────────────────────────────────

# Hand-picked pHashes with deliberate duplicates (exact ties) and
# symmetric distances (score ties across distinct hashes).
_PHASHES = [
    "0000000000000000",
    "0000000000000000",  # duplicate → hamming 0 tie
    "ffffffffffffffff",
    "00000000ffffffff",  # equidistant (32) from both extremes
    "ffffffff00000000",  # its mirror — same distances, tie on score
    "0f0f0f0f0f0f0f0f",
]


def _rand_phash(rng: random.Random) -> str:
    # Small alphabet of nibbles keeps hamming-distance ties frequent.
    return "".join(rng.choice("0f3c") for _ in range(16))


def _rand_vec(rng: random.Random) -> list[float]:
    # Coordinates from a tiny discrete set force exact cosine ties.
    return [float(rng.choice((0, 1))) for _ in range(4)]


def _entry(key: str, phash: str | None, clip: list[float] | None) -> dict[str, Any]:
    return {"key": key, "phash": phash, "clip": clip}


# ── Equivalence tests ───────────────────────────────────────────────────────


class TestPoolGreedyMaxMinEquivalence:
    """Pool rebalancing selection matches the pre-refactor loop."""

    def test_fixed_clip_mode_with_reference(self):
        candidates = [
            _entry("c1", None, [1.0, 0.0, 0.0, 0.0]),
            _entry("c2", None, [1.0, 0.0, 0.0, 0.0]),  # exact tie with c1
            _entry("c3", None, [0.0, 1.0, 0.0, 0.0]),
            _entry("c4", None, None),  # no signal → step-5 fill
        ]
        pool_ref = [_entry("p1", None, [1.0, 0.0, 0.0, 0.0])]
        for n_needed in (1, 2, 3, 4, 10):
            assert _greedy_max_min_select(
                [dict(c) for c in candidates],
                [dict(p) for p in pool_ref],
                "clip_diverse",
                n_needed,
            ) == _oracle_pool_greedy_max_min_select(
                [dict(c) for c in candidates],
                [dict(p) for p in pool_ref],
                "clip_diverse",
                n_needed,
            )

    def test_fixed_phash_mode_empty_reference(self):
        candidates = [
            _entry("b", _PHASHES[2], None),
            _entry("a", _PHASHES[0], None),
            _entry("c", _PHASHES[3], None),
        ]
        for n_needed in (1, 2, 3):
            assert _greedy_max_min_select(
                [dict(c) for c in candidates], [], "phash_diverse", n_needed
            ) == _oracle_pool_greedy_max_min_select(
                [dict(c) for c in candidates], [], "phash_diverse", n_needed
            )

    def test_seeded_random_sweep_both_modes(self):
        rng = random.Random(99)
        for _ in range(60):
            mode = rng.choice(("clip_diverse", "phash_diverse"))
            n_cand = rng.randint(0, 7)
            n_ref = rng.randint(0, 4)

            def _maybe_entry(idx: int, prefix: str, mode: str = mode) -> dict[str, Any]:
                has_signal = rng.random() > 0.25
                if mode == "clip_diverse":
                    return _entry(
                        f"{prefix}{idx:02d}",
                        None,
                        _rand_vec(rng) if has_signal else None,
                    )
                return _entry(
                    f"{prefix}{idx:02d}",
                    _rand_phash(rng) if has_signal else None,
                    None,
                )

            candidates = [_maybe_entry(i, "c") for i in range(n_cand)]
            pool_ref = [_maybe_entry(i, "p") for i in range(n_ref)]
            n_needed = rng.randint(0, n_cand + 2)
            got = _greedy_max_min_select(
                [dict(c) for c in candidates],
                [dict(p) for p in pool_ref],
                mode,
                n_needed,
            )
            want = _oracle_pool_greedy_max_min_select(
                [dict(c) for c in candidates],
                [dict(p) for p in pool_ref],
                mode,
                n_needed,
            )
            assert got == want, (
                f"diverged: mode={mode} n_needed={n_needed} "
                f"candidates={candidates!r} pool_ref={pool_ref!r}"
            )


class TestReviewSelectorEquivalence:
    """The next-review pick matches the pre-refactor loop."""

    def test_fixed_tie_candidates(self):
        window = [_entry("w1", None, [1.0, 0.0, 0.0, 0.0])]
        candidates = [
            _entry("b", None, [0.0, 1.0, 0.0, 0.0]),
            _entry("a", None, [0.0, 1.0, 0.0, 0.0]),  # exact tie → key wins
            _entry("z", None, [1.0, 0.0, 0.0, 0.0]),
        ]
        assert _greedy_select_next(
            list(candidates), list(window), "clip_diverse"
        ) == _oracle_review_greedy_select_next(
            list(candidates), list(window), "clip_diverse"
        )

    def test_fallback_paths(self):
        # No candidate carries the signal → sorted-key fallback.
        candidates = [_entry("m", None, None), _entry("k", None, None)]
        window = [_entry("w", None, [1.0, 0.0, 0.0, 0.0])]
        assert _greedy_select_next(
            list(candidates), list(window), "clip_diverse"
        ) == _oracle_review_greedy_select_next(
            list(candidates), list(window), "clip_diverse"
        )
        # Empty window → first candidate in caller order (NOT lowest key).
        candidates = [
            _entry("z", "0000000000000000", None),
            _entry("a", "ffffffffffffffff", None),
        ]
        assert _greedy_select_next(
            list(candidates), [], "phash_diverse"
        ) == _oracle_review_greedy_select_next(list(candidates), [], "phash_diverse")

    def test_seeded_random_sweep(self):
        rng = random.Random(7)
        for _ in range(60):
            mode = rng.choice(("clip_diverse", "phash_diverse"))
            n_cand = rng.randint(0, 6)
            n_win = rng.randint(0, 4)

            def _maybe(idx: int, prefix: str, mode: str = mode) -> dict[str, Any]:
                has_signal = rng.random() > 0.25
                if mode == "clip_diverse":
                    return _entry(
                        f"{prefix}{idx:02d}",
                        None,
                        _rand_vec(rng) if has_signal else None,
                    )
                return _entry(
                    f"{prefix}{idx:02d}",
                    _rand_phash(rng) if has_signal else None,
                    None,
                )

            candidates = [_maybe(i, "c") for i in range(n_cand)]
            rng.shuffle(candidates)
            window = [_maybe(i, "w") for i in range(n_win)]
            got = _greedy_select_next(list(candidates), list(window), mode)
            want = _oracle_review_greedy_select_next(
                list(candidates), list(window), mode
            )
            assert got == want, (
                f"diverged: mode={mode} candidates={candidates!r} window={window!r}"
            )

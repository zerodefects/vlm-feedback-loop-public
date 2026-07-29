# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared greedy max-min (farthest-point) selection core.

The Blueprint runs the same greedy loop in three places — Test-Pool
rebalancing (``pool_service``), the review selector
(``review_selector_service``), and the pHash-diverse embedding order
(``clip_embedding_service``). Each round it picks, from the
unselected candidates, the one whose **minimum distance to the
selected/reference set is largest**, breaking ties by ascending key.
This module is the single implementation of that loop; each call site
keeps its own distance definition, signal filtering, seeding rule, and
fallback semantics.

DETERMINISM-SENSITIVE. The output order feeds persisted pool
assignments and the review order; the tie-break contract
(``score > best`` OR ``score == best AND key < best_key``) and the
incremental min-distance update must stay bit-identical to the
per-site loops they replaced. ``tests/unit/test_greedy_max_min.py``
holds verbatim transcriptions of the pre-consolidation implementations
as oracles and asserts identical output on tie-heavy inputs.

Per-site semantics that intentionally stay OUTSIDE this helper:

- ``review_selector_service``: the empty-recent-window fallback picks
  the first candidate in caller order (Tier-1 priority), NOT the lowest
  key — the caller handles it before invoking this helper (which it
  only uses with a non-empty reference and ``n_select=1``).
- ``pool_service``: the no-signal remainder fill
  and the mode-dependent signal filtering happen at the call site.
- Similarity-based sites pass negated similarity as the distance
  (negation is exact in IEEE-754, so argmax(-sim) ≡ argmin(sim) with
  identical tie behavior).

NaN distances are unsupported (none of the call sites can produce one:
cosine helpers zero-norm-guard, pHash hamming is integral).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def greedy_max_min_indices(
    keys: Sequence[str],
    *,
    pair_dist: Callable[[int, int], float] | None = None,
    init_dist: Callable[[int], float] | None = None,
    seed_index: int | None = None,
    n_select: int | None = None,
) -> list[int]:
    """Greedy max-min selection over candidates identified by ``keys``.

    Args:
        keys: Unique, stable tie-break keys — one per candidate. The
            returned indices refer to positions in this sequence.
        pair_dist: ``pair_dist(i, j)`` = distance from candidate ``i``
            to the (already selected) candidate ``j``. Consulted to
            update min-distances after each pick; may be ``None`` only
            when it can never be consulted (``n_select == 1`` or fewer
            than two candidates).
        init_dist: ``init_dist(i)`` = minimum distance from candidate
            ``i`` to a pre-seeded reference set (e.g. current pool
            members, the recent review window). ``None`` with no
            ``seed_index`` means an empty reference: every candidate
            starts at ``+inf`` so the first pick falls to the lowest
            key.
        seed_index: Candidate pre-placed as the first selection (its
            index is the first element of the result). Mutually
            exclusive with ``init_dist``.
        n_select: Number of picks to make (excluding the seed).
            ``None`` selects every candidate.

    Returns:
        Selected candidate indices in pick order (seed first when
        given).
    """
    if seed_index is not None and init_dist is not None:
        raise ValueError("seed_index and init_dist are mutually exclusive")

    count = len(keys)
    if count == 0:
        return []

    selected = [False] * count
    order: list[int] = []
    min_dist: list[float]

    if seed_index is not None:
        selected[seed_index] = True
        order.append(seed_index)
        if pair_dist is None:
            raise ValueError("pair_dist is required when seeding")
        min_dist = [pair_dist(i, seed_index) for i in range(count)]
    elif init_dist is not None:
        min_dist = [init_dist(i) for i in range(count)]
    else:
        min_dist = [float("inf")] * count

    unselected = count - len(order)
    picks_budget = unselected if n_select is None else min(n_select, unselected)
    if pair_dist is None and picks_budget > 1:
        raise ValueError("pair_dist is required when selecting more than one item")

    for pick_no in range(picks_budget):
        best_i = -1
        best_score = float("-inf")
        best_key: str | None = None
        for i in range(count):
            if selected[i]:
                continue
            score = min_dist[i]
            if score > best_score or (
                score == best_score and (best_key is None or keys[i] < best_key)
            ):
                best_i, best_score, best_key = i, score, keys[i]
        if best_i < 0:  # pragma: no cover — budget is bounded by unselected count
            break
        selected[best_i] = True
        order.append(best_i)
        # Extend the reference set with the pick; skip the update after
        # the final pick (no further reads).
        if pick_no + 1 < picks_budget:
            assert pair_dist is not None  # narrowed by the budget guard above
            for i in range(count):
                if not selected[i]:
                    d = pair_dist(i, best_i)
                    if d < min_dist[i]:
                        min_dist[i] = d

    return order

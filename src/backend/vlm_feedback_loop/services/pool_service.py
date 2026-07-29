# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test Pool management service.

Implements:
  - Pool assignment routing: assigns newly Verified examples to the Test Pool
    or non-pool based on a configurable target fraction.
  - Automatic rebalancing: promotes non-pool Verified examples to the Test Pool
    using greedy max-min diversity selection (CLIP or pHash).
  - Evaluation snapshots: creates frozen Pool records for reproducible evaluation.
  - Log point 8: pool_routing operational logging.
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vlm_feedback_loop._defaults import DEFAULTS
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.clip_embedding_service import (
    embedding_cache,
    ensure_embedding_cache_loaded,
)
from vlm_feedback_loop.services.greedy_max_min import greedy_max_min_indices
from vlm_feedback_loop.services.review_selector_service import sim_clip, sim_phash

logger = logging.getLogger("vlm_feedback_loop.pool_routing")

# Default switchover threshold (overridable via project feature flags).
# Single source of truth is ``_defaults.py:DEFAULTS["CLIP_SWITCHOVER_MIN_COUNT"]``.
_DEFAULT_CLIP_SWITCHOVER_MIN_COUNT = DEFAULTS["CLIP_SWITCHOVER_MIN_COUNT"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_verified(session: Session, project_id: str) -> int:
    """Count all Verified labels in the project."""
    result = session.execute(
        select(func.count())
        .select_from(Label)
        .where(
            Label.project_id == project_id,
            Label.label_status == "verified",
        )
    )
    return result.scalar() or 0


def _count_pool_members(session: Session, project_id: str) -> int:
    """Count labels assigned to the Test Pool."""
    result = session.execute(
        select(func.count())
        .select_from(Label)
        .where(
            Label.project_id == project_id,
            Label.label_status == "verified",
            Label.pool_assignment == "test_pool",
        )
    )
    return result.scalar() or 0


def _compute_pool_target(total_verified: int, test_pool_fraction: float) -> int:
    """Return ``floor(total_verified * test_pool_fraction)``."""
    return math.floor(total_verified * test_pool_fraction)


def _determine_pool_diversity_mode(
    eligible_clip_count: int,
    clip_switchover_min_count: int = _DEFAULT_CLIP_SWITCHOVER_MIN_COUNT,
) -> str:
    """Determine whether to use CLIP or pHash for diversity selection.

    Uses the same switchover policy as the review selector:
    CLIP when >= threshold eligible examples have CLIP embeddings, else pHash.
    """
    if eligible_clip_count >= clip_switchover_min_count:
        return "clip_diverse"
    return "phash_diverse"


# ---------------------------------------------------------------------------
# Greedy max-min diversity selection for pool membership
# ---------------------------------------------------------------------------


def _greedy_max_min_select(
    candidates: list[dict[str, Any]],
    pool_ref: list[dict[str, Any]],
    mode: str,
    n_needed: int,
) -> list[str]:
    """Greedy max-min diversity selection.

    Each candidate/pool_ref dict has:
      - ``key``: example_key
      - ``phash``: hex string or None
      - ``clip``: float list or None

    Returns an ordered list of selected ``example_key`` values (up to *n_needed*).
    """
    if n_needed <= 0 or not candidates:
        return []

    # Step 1: separate candidates by signal availability
    if mode == "clip_diverse":
        c_with = [c for c in candidates if c["clip"] is not None]
        c_without = [c for c in candidates if c["clip"] is None]
        p_ref = [p for p in pool_ref if p["clip"] is not None]

        def _sim(a: dict[str, Any], b: dict[str, Any]) -> float:
            return sim_clip(a["clip"], b["clip"])
    else:
        c_with = [c for c in candidates if c["phash"] is not None]
        c_without = [c for c in candidates if c["phash"] is None]
        p_ref = [p for p in pool_ref if p["phash"] is not None]

        def _sim(a: dict[str, Any], b: dict[str, Any]) -> float:
            return sim_phash(a["phash"], b["phash"])

    selected: list[str] = []

    # Greedy max-min via the shared core. "Most dissimilar to the
    # reference set" = argmax over candidates of min(-similarity); with an
    # empty reference every start score is +inf, so the first pick falls
    # to the lowest key. Picks extend the reference.
    if c_with:
        keys = [c["key"] for c in c_with]

        def _init_dist(i: int) -> float:
            return min(-_sim(c_with[i], r) for r in p_ref)

        order = greedy_max_min_indices(
            keys,
            pair_dist=lambda i, j: -_sim(c_with[i], c_with[j]),
            init_dist=_init_dist if p_ref else None,
            n_select=n_needed,
        )
        selected = [keys[i] for i in order]

    # Fill remaining from candidates without the active signal
    remaining = n_needed - len(selected)
    if remaining > 0 and c_without:
        c_without.sort(key=lambda c: c["key"])
        for c in c_without[:remaining]:
            selected.append(c["key"])

    return selected


# ---------------------------------------------------------------------------
# Pool assignment routing
# ---------------------------------------------------------------------------


def route_pool(
    project_id: str,
    session: Session,
    engine: Any,
    project: Project,
    label: Label,
    verified_outcome: str,
) -> str | None:
    """Route a newly Verified example to ``test_pool`` or ``None``.

    Called within *save_label*'s open session/transaction.  Implements
    and persists the initial assignment, then triggers rebalancing when the
    pool is below its target size.

    Returns the ``pool_assignment`` value for the new label.
    """
    example_key = label.example_key

    # Flush so the new label is visible to aggregate queries
    session.flush()

    total_verified = _count_verified(session, project_id)
    pool_count = _count_pool_members(session, project_id)
    target = _compute_pool_target(total_verified, project.test_pool_fraction)

    # --- Initial routing decision ---
    # For a single candidate (one save at a time), the spec says:
    # "assign the next Verified example directly" — no diversity needed.
    if pool_count < target:
        assignment: str | None = "test_pool"
    else:
        assignment = None

    # Rebalancing must score against every current Test member, including
    # the label assigned by this save. Keep the assignment in the caller's
    # transaction so a later save failure rolls back it and all promotions.
    label.pool_assignment = assignment
    session.flush()

    # --- Rebalancing ---
    new_pool_count = pool_count + (1 if assignment == "test_pool" else 0)
    rebalanced_keys: list[str] = []

    if new_pool_count < target:
        n_needed = target - new_pool_count
        rebalanced_keys = _rebalance_pool(
            session,
            project_id,
            project,
            engine,
            n_needed,
            exclude_key=example_key,
        )

    # --- Log point 8 ---
    final_pool_count = new_pool_count + len(rebalanced_keys)
    _log_pool_routing(
        project_id=project_id,
        destination=assignment,
        verified_outcome=verified_outcome,
        pool_count=final_pool_count,
        target_count=target,
        rebalancing_triggered=len(rebalanced_keys) > 0,
        rebalanced_keys=rebalanced_keys,
    )

    return assignment


# ---------------------------------------------------------------------------
# Automatic rebalancing
# ---------------------------------------------------------------------------


def _rebalance_pool(
    session: Session,
    project_id: str,
    project: Project,
    engine: Any,
    n_needed: int,
    exclude_key: str | None = None,
) -> list[str]:
    """Promote non-pool Verified examples to the Test Pool using the
    greedy max-min diversity selection algorithm.

    Returns a list of promoted ``example_key`` values.
    """
    if n_needed <= 0:
        return []

    # 1. Non-pool Verified labels (candidates for promotion)
    candidate_stmt = select(Label).where(
        Label.project_id == project_id,
        Label.label_status == "verified",
        Label.pool_assignment.is_(None),
    )
    candidate_labels = list(session.execute(candidate_stmt).scalars().all())

    # Exclude the example just routed (already handled by initial routing)
    if exclude_key:
        candidate_labels = [
            lbl for lbl in candidate_labels if lbl.example_key != exclude_key
        ]

    if not candidate_labels:
        return []

    # 2. Current pool members (reference set for diversity)
    pool_labels = list(
        session.execute(
            select(Label).where(
                Label.project_id == project_id,
                Label.label_status == "verified",
                Label.pool_assignment == "test_pool",
            )
        )
        .scalars()
        .all()
    )

    # 3. Load Example records for pHash and CLIP presence
    all_keys = [lbl.example_key for lbl in candidate_labels] + [
        lbl.example_key for lbl in pool_labels
    ]
    examples = list(
        session.execute(
            select(Example).where(
                Example.project_id == project_id,
                Example.example_key.in_(all_keys),
            )
        )
        .scalars()
        .all()
    )
    example_map: dict[str, Example] = {ex.example_key: ex for ex in examples}

    # 4. Determine diversity mode via switchover
    clip_count = sum(
        1
        for lbl in candidate_labels
        if example_map.get(lbl.example_key)
        and example_map[lbl.example_key].clip_embedding_present
    )

    ff = project.feature_flags or {}
    switchover = ff.get("CLIP_SWITCHOVER_MIN_COUNT", _DEFAULT_CLIP_SWITCHOVER_MIN_COUNT)
    mode = _determine_pool_diversity_mode(clip_count, switchover)

    # 5. Load CLIP vectors if needed
    clip_vectors: dict[str, list[float]] = {}
    if mode == "clip_diverse":
        ensure_embedding_cache_loaded(project_id, engine, project.embedding_dim)
        clip_vectors = embedding_cache.get_all(project_id)

    # 6. Build entry dicts
    def _make_entry(example_key: str) -> dict[str, Any]:
        ex = example_map.get(example_key)
        return {
            "key": example_key,
            "phash": ex.phash if ex else None,
            "clip": (clip_vectors.get(example_key) if mode == "clip_diverse" else None),
        }

    candidate_entries = [_make_entry(lbl.example_key) for lbl in candidate_labels]
    pool_ref_entries = [_make_entry(lbl.example_key) for lbl in pool_labels]

    # 7. Run the greedy max-min selection
    selected_keys = _greedy_max_min_select(
        candidate_entries,
        pool_ref_entries,
        mode,
        n_needed,
    )

    # 8. Promote selected labels
    label_by_key = {lbl.example_key: lbl for lbl in candidate_labels}
    for key in selected_keys:
        lbl = label_by_key.get(key)
        if lbl is not None:
            lbl.pool_assignment = "test_pool"

    return selected_keys


# ---------------------------------------------------------------------------
# Evaluation snapshots
# ---------------------------------------------------------------------------


def create_pool_snapshot(
    session: Session,
    project_id: str,
    guidance_id: str,
) -> Pool | None:
    """Create a frozen Pool version snapshot for an evaluation run.

    Returns ``None`` if the Test Pool has no members (evaluation cannot
    proceed).
    """
    # Gather current pool member keys (sorted for determinism)
    member_keys = list(
        session.execute(
            select(Label.example_key)
            .where(
                Label.project_id == project_id,
                Label.label_status == "verified",
                Label.pool_assignment == "test_pool",
            )
            .order_by(Label.example_key)
        )
        .scalars()
        .all()
    )

    if not member_keys:
        return None

    # Monotonically increasing pool_version within the project
    max_version = session.execute(
        select(func.max(Pool.pool_version)).where(
            Pool.project_id == project_id,
        )
    ).scalar()
    next_version = (max_version or 0) + 1

    pool = Pool(
        pool_id=generate_uuid4(),
        project_id=project_id,
        pool_type="test_pool",
        pool_version=next_version,
        member_example_keys=member_keys,
        member_count=len(member_keys),
        guidance_id=guidance_id,
    )
    session.add(pool)

    return pool


# ---------------------------------------------------------------------------
# Diagnostic: Test/Train pool class coverage (non-invasive guard)
# ---------------------------------------------------------------------------


def _find_class_field(fields: list[dict[str, Any]]) -> str | None:
    """Return the first Core categorical (enum) or boolean field by display order.

    This is the de-facto "class" for a project — the field a relevance-ICL
    evaluation must be able to retrieve same-class exemplars for. Mirrors the
    diversity-bucket field selection used by the ICL selector
    (``icl_service._find_diversity_bucket_field``) so the diagnostic keys off
    the same notion of "class" the product uses elsewhere.

    Returns ``None`` when the schema has no Core enum/boolean field, in which
    case class-coverage assessment is not applicable.
    """
    core_class_fields = [
        f
        for f in fields
        if f.get("role") == "core" and f.get("type") in ("enum", "boolean")
    ]
    if not core_class_fields:
        return None
    core_class_fields.sort(
        key=lambda f: (f.get("display_order", 0), f.get("field_name", ""))
    )
    return core_class_fields[0]["field_name"]


def assess_pool_class_coverage(engine: Any, project_id: str) -> dict[str, Any]:
    """Assess whether the Test Pool and Train Pool hold overlapping class sets.

    Diagnostic only — does NOT change pool routing.
    It detects the latent vulnerability where a *class-clustered*
    labeling order (e.g. manifest-order autorun, or a class-sorted batch
    import) drives the order-sensitive Test/Train split into holding **disjoint
    class sets**. Because relevance-ICL evaluation draws its ICL exemplars only
    from the Train Pool (``Label.pool_assignment IS NULL AND
    verified_outcome='Edit'``), any Test-Pool class with no Train-Pool
    representation can only ever retrieve wrong-class exemplars, so relevance-ICL
    metrics structurally understate true accuracy for those classes. The
    CLIP-diverse review selector (the product default) keeps the
    split class-representative and avoids this; this guard surfaces the
    degenerate case when some other order produced it.

    The "class" is the project's first Core enum/boolean field (see
    :func:`_find_class_field`). When the schema has no such field, the result
    is benign (``applicable=False``) and callers should emit no warning.

    Cheap: one pass over verified labels for the project's active guidance.

    Returns a dict::

        {
            "applicable": bool,        # False => no Core enum/boolean class field
            "class_field": str | None, # the de-facto class field name
            "test_classes": list[str],     # classes present in the Test Pool
            "train_classes": list[str],    # classes present in the Train Pool
            "test_only_classes": list[str],# in Test Pool but absent from Train Pool
            "overlap_count": int,          # |test_classes ∩ train_classes|
            "degenerate": bool,            # True => some Test class has no Train rep
        }
    """
    not_applicable: dict[str, Any] = {
        "applicable": False,
        "class_field": None,
        "test_classes": [],
        "train_classes": [],
        "test_only_classes": [],
        "overlap_count": 0,
        "degenerate": False,
    }

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None or not project.active_guidance_id:
            return not_applicable

        guidance = session.execute(
            select(Guidance).where(
                Guidance.project_id == project_id,
                Guidance.guidance_id == project.active_guidance_id,
            )
        ).scalar_one_or_none()
        if guidance is None or not guidance.schema:
            return not_applicable

        fields_obj: Any = (guidance.schema or {}).get("fields")
        fields: list[dict[str, Any]] = (
            cast("list[dict[str, Any]]", fields_obj)
            if isinstance(fields_obj, list)
            else []
        )
        class_field = _find_class_field(fields)
        if class_field is None:
            return not_applicable

        # One pass over verified labels for the active guidance, projecting the
        # class value and the pool assignment.
        rows = session.execute(
            select(Label.label_json, Label.pool_assignment).where(
                Label.project_id == project_id,
                Label.label_status == "verified",
                Label.guidance_id == project.active_guidance_id,
            )
        ).all()

    test_classes: set[str] = set()
    train_classes: set[str] = set()
    for label_json_obj, pool_assignment in rows:
        if not isinstance(label_json_obj, dict):
            continue
        label_json = cast("dict[str, Any]", label_json_obj)
        if class_field not in label_json:
            continue
        value = label_json.get(class_field)
        if value is None:
            continue
        cls = str(value)
        if pool_assignment == "test_pool":
            test_classes.add(cls)
        elif pool_assignment is None:
            # Train Pool = non-pool verified labels (the relevance-ICL candidate
            # set). Any other assignment value is ignored here.
            train_classes.add(cls)

    test_only = test_classes - train_classes
    return {
        "applicable": True,
        "class_field": class_field,
        "test_classes": sorted(test_classes),
        "train_classes": sorted(train_classes),
        "test_only_classes": sorted(test_only),
        "overlap_count": len(test_classes & train_classes),
        "degenerate": bool(test_only),
    }


# ---------------------------------------------------------------------------
# Log point 8: pool_routing
# ---------------------------------------------------------------------------


def _log_pool_routing(
    project_id: str,
    destination: str | None,
    verified_outcome: str,
    pool_count: int,
    target_count: int,
    rebalancing_triggered: bool,
    rebalanced_keys: list[str],
) -> None:
    """Log point 8 -- pool_routing.

    Default level: DEBUG.
    Escalates to INFO when rebalancing actually changes pool membership.
    """
    details: dict[str, Any] = {
        "destination": destination,
        "verified_outcome": verified_outcome,
        "pool_count": pool_count,
        "target_count": target_count,
        "rebalancing_triggered": rebalancing_triggered,
    }
    if rebalanced_keys:
        details["rebalanced_keys"] = rebalanced_keys

    level = logging.INFO if rebalancing_triggered else logging.DEBUG
    suffix = f"; rebalanced {len(rebalanced_keys)}" if rebalancing_triggered else ""

    logger.log(
        level,
        "Pool routing: %s -> %s (pool %d/%d%s)",
        "assigned" if destination else "not assigned",
        destination or "non-pool",
        pool_count,
        target_count,
        suffix,
        extra={
            "component": "pool_routing",
            "project_id": project_id,
            "details": details,
        },
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pool_service (Test Pool management).

Sections:
  A. Pool Assignment Routing
  B. Rebalancing
  C. Durability
  D. Evaluation Snapshots
  E. Greedy Max-Min Algorithm
  F. Log Point 8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from conftest import add_example_row, add_project_row, open_project_workspace
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.pool_service import (
    _compute_pool_target,
    _determine_pool_diversity_mode,
    _greedy_max_min_select,
    _log_pool_routing,
    _rebalance_pool,
    create_pool_snapshot,
    route_pool,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

PID = "test-proj"
GID = "guid-001"


def _setup_project_db(tmp_path: Path, project_id: str = PID):
    engine, project_dir, _ = open_project_workspace(tmp_path, project_id)
    return engine, str(project_dir)


def _add_project(session: Session, project_id: str, project_dir: str, **overrides: Any):
    add_project_row(
        session,
        project_id,
        project_dir,
        **{"active_guidance_id": GID, **overrides},
    )
    return session.get(Project, project_id)


def _add_example(
    session: Session,
    project_id: str,
    key: str,
    *,
    state: str = "Verified",
    phash: str | None = "a" * 16,
    clip_present: bool = False,
):
    add_example_row(
        session,
        project_id,
        key,
        state=state,
        phash=phash,
        clip_embedding_present=clip_present,
    )


def _add_label(
    session: Session,
    project_id: str,
    key: str,
    guidance_id: str = GID,
    *,
    outcome: str = "Accept",
    pool_assignment: str | None = None,
) -> Label:
    label = Label(
        label_id=generate_uuid4(),
        project_id=project_id,
        example_key=key,
        label_status="verified",
        guidance_id=guidance_id,
        inference_invocation_id=generate_uuid4(),
        label_json={"severity": "high", "damaged": True, "rationale_note": "test"},
        labeled_at=utc_now(),
        verified_outcome=outcome,
        verified_at=utc_now(),
        edited_core_fields=[],
        edited_aux_fields=[],
        rationale_source="teacher_proposal" if outcome == "Accept" else "sme_edited",
        pool_assignment=pool_assignment,
    )
    session.add(label)
    return label


# ══════════════════════════════════════════════════════════════════════════════
# Section A: Pool Assignment Routing (AC #1, #2)
# ══════════════════════════════════════════════════════════════════════════════


class TestPoolAssignmentRouting:
    """AC #1: outcome-agnostic; AC #2: below-target routing."""

    def test_accept_assigned_to_pool_when_below_target(self, tmp_path):
        """Accept labels go to test_pool when pool is below target."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.50)
            # Need total_verified >= 2 for target >= 1 (floor(2*0.5) = 1)
            _add_example(s, PID, "img_001", state="Verified")
            _add_label(s, PID, "img_001", outcome="Accept", pool_assignment=None)
            # The new example being verified
            _add_example(s, PID, "img_002", state="Verified")
            current_label = _add_label(
                s,
                PID,
                "img_002",
                outcome="Accept",
                pool_assignment=None,
            )
            s.flush()

            result = route_pool(PID, s, engine, project, current_label, "Accept")
            assert result == "test_pool"

    def test_edit_assigned_to_pool_when_below_target(self, tmp_path):
        """Edit labels go to test_pool when pool is below target (outcome-agnostic)."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.50)
            _add_example(s, PID, "img_001", state="Verified")
            _add_label(s, PID, "img_001", outcome="Edit", pool_assignment=None)
            _add_example(s, PID, "img_002", state="Verified")
            current_label = _add_label(
                s,
                PID,
                "img_002",
                outcome="Edit",
                pool_assignment=None,
            )
            s.flush()

            result = route_pool(PID, s, engine, project, current_label, "Edit")
            assert result == "test_pool"

    def test_at_target_assigns_none(self, tmp_path):
        """When pool is already at target, assign None."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.50)
            # 2 verified, target = 1, pool_count = 1 => at target
            _add_example(s, PID, "img_001", state="Verified")
            _add_label(s, PID, "img_001", outcome="Accept", pool_assignment="test_pool")
            _add_example(s, PID, "img_002", state="Verified")
            current_label = _add_label(
                s,
                PID,
                "img_002",
                outcome="Accept",
                pool_assignment=None,
            )
            s.flush()

            result = route_pool(PID, s, engine, project, current_label, "Accept")
            assert result is None

    def test_above_target_assigns_none(self, tmp_path):
        """When pool is above target, assign None."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.30)
            # 3 verified, target = floor(0.9) = 0, but all 3 in pool
            for i in range(3):
                k = f"img_{i:03d}"
                _add_example(s, PID, k, state="Verified")
                _add_label(s, PID, k, outcome="Accept", pool_assignment="test_pool")
            # New example
            _add_example(s, PID, "img_new", state="Verified")
            current_label = _add_label(
                s,
                PID,
                "img_new",
                outcome="Accept",
                pool_assignment=None,
            )
            s.flush()

            # total = 4, target = floor(4*0.30) = 1, pool = 3 => above target
            result = route_pool(PID, s, engine, project, current_label, "Accept")
            assert result is None

    def test_single_example_target_zero(self, tmp_path):
        """With only 1 Verified, floor(1*0.40) = 0 => target 0, assign None."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.40)
            _add_example(s, PID, "img_001", state="Verified")
            current_label = _add_label(
                s,
                PID,
                "img_001",
                outcome="Accept",
                pool_assignment=None,
            )
            s.flush()

            # total = 1, target = floor(0.4) = 0 => at target
            result = route_pool(PID, s, engine, project, current_label, "Accept")
            assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Section B: Rebalancing (AC #3, #6)
# ══════════════════════════════════════════════════════════════════════════════


class TestRebalancing:
    """AC #3: promotes non-pool using diversity; AC #6: null-assignment eligible."""

    def test_rebalancing_promotes_non_pool_when_below_target(self, tmp_path):
        """Pre-populate 10 verified with null, new save triggers rebalancing."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.40)
            # 10 existing verified with null assignment (saved before pool routing existed)
            for i in range(10):
                k = f"img_{i:03d}"
                _add_example(s, PID, k, state="Verified", phash=f"{i:016x}")
                _add_label(s, PID, k, outcome="Accept", pool_assignment=None)
            # New example being verified
            _add_example(s, PID, "img_new", state="Verified", phash="f" * 16)
            current_label = _add_label(
                s,
                PID,
                "img_new",
                outcome="Edit",
                pool_assignment=None,
            )
            s.flush()

            # total = 11, target = floor(11*0.40) = 4, pool = 0
            result = route_pool(PID, s, engine, project, current_label, "Edit")

            # img_new should be assigned to test_pool (initial routing)
            assert result == "test_pool"

            # Count how many are now in the pool (including rebalanced)
            pool_count = (
                s.query(Label)
                .filter_by(
                    project_id=PID,
                    pool_assignment="test_pool",
                )
                .count()
            )
            # Target is 4, so 4 should be in pool: img_new + 3 rebalanced
            assert pool_count == 4

    def test_rebalancing_uses_phash_diversity(self, tmp_path):
        """The newly assigned Test member participates in diversity scoring."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.75)
            _add_example(
                s,
                PID,
                "pool_existing",
                state="Verified",
                phash="0000000000000000",
            )
            _add_label(
                s,
                PID,
                "pool_existing",
                outcome="Accept",
                pool_assignment="test_pool",
            )
            _add_example(
                s,
                PID,
                "cand_duplicate_current",
                state="Verified",
                phash="ffffffffffffffff",
            )
            _add_label(
                s,
                PID,
                "cand_duplicate_current",
                outcome="Accept",
                pool_assignment=None,
            )
            _add_example(
                s,
                PID,
                "cand_balanced",
                state="Verified",
                phash="00000000ffffffff",
            )
            _add_label(
                s,
                PID,
                "cand_balanced",
                outcome="Accept",
                pool_assignment=None,
            )
            s.commit()

            _add_example(
                s,
                PID,
                "new_current",
                state="Verified",
                phash="ffffffffffffffff",
            )
            current_label = _add_label(
                s,
                PID,
                "new_current",
                outcome="Accept",
                pool_assignment=None,
            )
            s.flush()

            # target=floor(4*0.75)=3: current is assigned directly and one
            # historical candidate must be promoted.
            assignment = route_pool(
                PID,
                s,
                engine,
                project,
                current_label,
                "Accept",
            )

            test_keys = {
                lbl.example_key
                for lbl in s.query(Label)
                .filter_by(project_id=PID, pool_assignment="test_pool")
                .all()
            }
            train_keys = {
                lbl.example_key
                for lbl in s.query(Label)
                .filter(Label.project_id == PID, Label.pool_assignment.is_(None))
                .all()
            }
            assert assignment == "test_pool"
            assert test_keys == {
                "pool_existing",
                "new_current",
                "cand_balanced",
            }
            assert train_keys == {"cand_duplicate_current"}

            # Routing is part of the caller's transaction; no current or
            # promoted assignment may survive a failed save.
            s.rollback()

        with Session(engine) as s:
            assert (
                s.query(Label).filter_by(example_key="new_current").one_or_none()
                is None
            )
            test_keys = {
                lbl.example_key
                for lbl in s.query(Label)
                .filter_by(project_id=PID, pool_assignment="test_pool")
                .all()
            }
            train_keys = {
                lbl.example_key
                for lbl in s.query(Label)
                .filter(Label.project_id == PID, Label.pool_assignment.is_(None))
                .all()
            }
            assert test_keys == {"pool_existing"}
            assert train_keys == {"cand_duplicate_current", "cand_balanced"}

    def test_null_assignment_eligible_for_promotion(self, tmp_path):
        """Labels with pool_assignment=None are eligible for rebalancing (AC #6)."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.50)
            # 4 verified with null assignment
            for i in range(4):
                k = f"img_{i:03d}"
                _add_example(s, PID, k, state="Verified", phash=f"{i:016x}")
                _add_label(s, PID, k, outcome="Accept", pool_assignment=None)
            s.flush()

            # total = 4, target = 2, pool = 0
            # Rebalancing should promote 2 of the null-assignment labels
            promoted = _rebalance_pool(s, PID, project, engine, 2)
            assert len(promoted) == 2

            # Verify they now have pool_assignment = "test_pool"
            for key in promoted:
                lbl = (
                    s.query(Label)
                    .filter_by(
                        project_id=PID,
                        example_key=key,
                    )
                    .first()
                )
                assert lbl is not None
                assert lbl.pool_assignment == "test_pool"

    def test_stub_transition_first_save_promotes(self, tmp_path):
        """Pre-populate verified labels with null assignment; first new save fills pool."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.40)
            # 10 verified with null assignment (saved before pool routing existed)
            for i in range(10):
                k = f"img_{i:03d}"
                _add_example(s, PID, k, state="Verified", phash=f"{i:016x}")
                _add_label(
                    s,
                    PID,
                    k,
                    outcome="Accept" if i % 2 == 0 else "Edit",
                    pool_assignment=None,
                )

            # New save
            _add_example(s, PID, "img_new", state="Verified", phash="ff" * 8)
            current_label = _add_label(
                s,
                PID,
                "img_new",
                outcome="Accept",
                pool_assignment=None,
            )
            s.flush()

            # total = 11, target = floor(4.4) = 4, pool = 0
            result = route_pool(PID, s, engine, project, current_label, "Accept")
            assert result == "test_pool"

            pool_count = (
                s.query(Label)
                .filter_by(
                    project_id=PID,
                    pool_assignment="test_pool",
                )
                .count()
            )
            assert pool_count == 4  # img_new + 3 rebalanced from null-pool labels


# ══════════════════════════════════════════════════════════════════════════════
# Section C: Durability (AC #4, #5)
# ══════════════════════════════════════════════════════════════════════════════


class TestPoolDurability:
    """AC #4: never demoted; AC #5: persisted on Label."""

    def test_pool_assignment_never_demoted(self, tmp_path):
        """Once assigned to test_pool, an example stays there across further saves."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.50)
            # 2 verified: img_001 in pool, img_002 not
            _add_example(s, PID, "img_001", state="Verified")
            _add_label(s, PID, "img_001", outcome="Accept", pool_assignment="test_pool")
            _add_example(s, PID, "img_002", state="Verified")
            _add_label(s, PID, "img_002", outcome="Accept", pool_assignment=None)

            # Add more examples to push the pool above target
            for i in range(3, 20):
                k = f"img_{i:03d}"
                _add_example(s, PID, k, state="Verified")
                _add_label(s, PID, k, outcome="Accept", pool_assignment="test_pool")
            s.flush()

            # Now pool has 18 members (way above target).  New save.
            _add_example(s, PID, "img_new", state="Verified")
            current_label = _add_label(
                s,
                PID,
                "img_new",
                outcome="Accept",
                pool_assignment=None,
            )
            s.flush()

            route_pool(PID, s, engine, project, current_label, "Accept")

            # img_001 still in pool (never demoted)
            lbl = (
                s.query(Label)
                .filter_by(
                    project_id=PID,
                    example_key="img_001",
                )
                .first()
            )
            assert lbl is not None
            assert lbl.pool_assignment == "test_pool"

    def test_pool_assignment_persisted_on_label_record(self, tmp_path):
        """pool_assignment is persisted on the Label record (AC #5)."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            project = _add_project(s, PID, pdir, test_pool_fraction=0.50)
            _add_example(s, PID, "img_001", state="Verified")
            _add_label(s, PID, "img_001", outcome="Accept", pool_assignment=None)
            _add_example(s, PID, "img_002", state="Verified")
            current_label = _add_label(
                s,
                PID,
                "img_002",
                outcome="Accept",
                pool_assignment=None,
            )
            s.flush()

            # total = 2, target = 1, pool = 0 => assign to test_pool
            result = route_pool(PID, s, engine, project, current_label, "Accept")
            assert result == "test_pool"

            s.commit()

        # Re-open session to verify persistence
        with Session(engine) as s2:
            lbl = (
                s2.query(Label)
                .filter_by(
                    project_id=PID,
                    example_key="img_002",
                )
                .first()
            )
            assert lbl is not None
            assert lbl.pool_assignment == "test_pool"


# ══════════════════════════════════════════════════════════════════════════════
# Section D: Evaluation Snapshots (AC #7, #8, #9)
# ══════════════════════════════════════════════════════════════════════════════


class TestPoolSnapshot:
    """AC #7: auto-created; AC #8: immutable; AC #9: no members -> None."""

    def test_snapshot_created_with_correct_members(self, tmp_path):
        """Snapshot contains exactly the current pool member keys."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            _add_project(s, PID, pdir)
            # 3 in pool, 2 not
            for i in range(3):
                k = f"pool_{i:03d}"
                _add_example(s, PID, k, state="Verified")
                _add_label(s, PID, k, outcome="Accept", pool_assignment="test_pool")
            for i in range(2):
                k = f"nonpool_{i:03d}"
                _add_example(s, PID, k, state="Verified")
                _add_label(s, PID, k, outcome="Accept", pool_assignment=None)
            s.flush()

            pool = create_pool_snapshot(s, PID, GID)
            assert pool is not None
            assert pool.pool_type == "test_pool"
            assert pool.member_count == 3
            assert sorted(pool.member_example_keys) == [
                "pool_000",
                "pool_001",
                "pool_002",
            ]
            assert pool.guidance_id == GID
            assert pool.pool_version == 1

    def test_snapshot_version_monotonically_increases(self, tmp_path):
        """3 consecutive snapshots get versions 1, 2, 3."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            _add_project(s, PID, pdir)
            _add_example(s, PID, "img_001", state="Verified")
            _add_label(s, PID, "img_001", outcome="Accept", pool_assignment="test_pool")
            s.flush()

            p1 = create_pool_snapshot(s, PID, GID)
            s.flush()
            p2 = create_pool_snapshot(s, PID, GID)
            s.flush()
            p3 = create_pool_snapshot(s, PID, GID)

            assert p1 is not None and p1.pool_version == 1
            assert p2 is not None and p2.pool_version == 2
            assert p3 is not None and p3.pool_version == 3

    def test_snapshot_immutable_after_creation(self, tmp_path):
        """New pool members added after snapshot do not affect the snapshot."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            _add_project(s, PID, pdir)
            _add_example(s, PID, "img_001", state="Verified")
            _add_label(s, PID, "img_001", outcome="Accept", pool_assignment="test_pool")
            s.flush()

            pool = create_pool_snapshot(s, PID, GID)
            assert pool is not None
            original_keys = list(pool.member_example_keys)
            original_count = pool.member_count

            # Add another pool member after snapshot
            _add_example(s, PID, "img_002", state="Verified")
            _add_label(s, PID, "img_002", outcome="Accept", pool_assignment="test_pool")
            s.flush()

            # Re-read the snapshot from the DB
            loaded = s.query(Pool).filter_by(pool_id=pool.pool_id).first()
            assert loaded is not None
            assert loaded.member_example_keys == original_keys
            assert loaded.member_count == original_count

    def test_no_members_returns_none(self, tmp_path):
        """Empty pool returns None (evaluation cannot proceed)."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            _add_project(s, PID, pdir)
            # No labels with pool_assignment="test_pool"
            _add_example(s, PID, "img_001", state="Verified")
            _add_label(s, PID, "img_001", outcome="Accept", pool_assignment=None)
            s.flush()

            pool = create_pool_snapshot(s, PID, GID)
            assert pool is None

    def test_cleared_on_semantic_core_change(self, tmp_path):
        """After schema evolution deletes labels, snapshot reflects empty pool."""
        engine, pdir = _setup_project_db(tmp_path)
        with Session(engine) as s:
            _add_project(s, PID, pdir)
            _add_example(s, PID, "img_001", state="Verified")
            _add_label(s, PID, "img_001", outcome="Accept", pool_assignment="test_pool")
            s.flush()

            # Pre-evolution snapshot
            p1 = create_pool_snapshot(s, PID, GID)
            assert p1 is not None
            assert p1.member_count == 1
            s.flush()

            # Simulate semantic Core change: delete all labels
            s.query(Label).filter_by(project_id=PID).delete()
            s.flush()

            # Post-evolution snapshot
            p2 = create_pool_snapshot(s, PID, GID)
            assert p2 is None

            # Old snapshot still intact (immutable)
            old = s.query(Pool).filter_by(pool_id=p1.pool_id).first()
            assert old is not None
            assert old.member_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# Section E: Greedy Max-Min Algorithm
# ══════════════════════════════════════════════════════════════════════════════


class TestGreedyMaxMinSelect:
    """Unit tests for _greedy_max_min_select."""

    def test_empty_candidates_returns_empty(self):
        assert _greedy_max_min_select([], [], "phash_diverse", 5) == []

    def test_zero_needed_returns_empty(self):
        candidates = [{"key": "a", "phash": "0" * 16, "clip": None}]
        assert _greedy_max_min_select(candidates, [], "phash_diverse", 0) == []

    def test_empty_pool_ref_selects_by_key_ascending(self):
        """When pool_ref is empty, pick the first candidate by key ascending."""
        candidates = [
            {"key": "c", "phash": "cccccccccccccccc", "clip": None},
            {"key": "a", "phash": "aaaaaaaaaaaaaaaa", "clip": None},
            {"key": "b", "phash": "bbbbbbbbbbbbbbbb", "clip": None},
        ]
        result = _greedy_max_min_select(candidates, [], "phash_diverse", 1)
        assert result == ["a"]

    def test_phash_diversity_selects_most_dissimilar(self):
        """With known pHash distances, argmin max-similarity picks most diverse."""
        # pool_ref has pHash 0x0000...
        pool_ref = [{"key": "pool", "phash": "0000000000000000", "clip": None}]
        candidates = [
            {"key": "near", "phash": "0000000000000001", "clip": None},  # hamming=1
            {"key": "mid", "phash": "00000000ffffffff", "clip": None},  # hamming=32
            {"key": "far", "phash": "ffffffffffffffff", "clip": None},  # hamming=64
        ]
        result = _greedy_max_min_select(candidates, pool_ref, "phash_diverse", 1)
        # far has lowest similarity (highest hamming distance) => most diverse
        assert result == ["far"]

    def test_clip_diversity_selects_most_dissimilar(self):
        """With known CLIP vectors, argmin cosine-similarity picks most diverse."""
        # Pool has [1, 0, 0]
        pool_ref = [{"key": "pool", "phash": None, "clip": [1.0, 0.0, 0.0]}]
        candidates = [
            {"key": "similar", "phash": None, "clip": [0.9, 0.1, 0.0]},  # high cosine
            {"key": "orthogonal", "phash": None, "clip": [0.0, 1.0, 0.0]},  # cosine ~0
            {"key": "opposite", "phash": None, "clip": [-1.0, 0.0, 0.0]},  # cosine = -1
        ]
        result = _greedy_max_min_select(candidates, pool_ref, "clip_diverse", 1)
        # opposite has lowest similarity (-1) => selected
        assert result == ["opposite"]

    def test_tiebreak_by_ascending_key(self):
        """When scores are equal, lower example_key wins."""
        pool_ref = [{"key": "pool", "phash": "0000000000000000", "clip": None}]
        # Two candidates with identical pHash => identical similarity
        candidates = [
            {"key": "b_candidate", "phash": "ffffffffffffffff", "clip": None},
            {"key": "a_candidate", "phash": "ffffffffffffffff", "clip": None},
        ]
        result = _greedy_max_min_select(candidates, pool_ref, "phash_diverse", 1)
        assert result == ["a_candidate"]

    def test_overflow_fills_from_no_signal_candidates(self):
        """When n_needed > candidates with signal, fill from no-signal by key."""
        candidates = [
            {"key": "has_phash", "phash": "0000000000000000", "clip": None},
            {"key": "no_phash_b", "phash": None, "clip": None},
            {"key": "no_phash_a", "phash": None, "clip": None},
        ]
        result = _greedy_max_min_select(candidates, [], "phash_diverse", 3)
        assert result == ["has_phash", "no_phash_a", "no_phash_b"]

    def test_multi_slot_selection_extends_reference(self):
        """Each selected candidate is added to reference for subsequent slots."""
        pool_ref = [{"key": "p", "phash": "0000000000000000", "clip": None}]
        candidates = [
            {"key": "a", "phash": "4000000000000000", "clip": None},  # hamming ~1-2
            {"key": "b", "phash": "8000000000000000", "clip": None},  # hamming ~1-2
            {"key": "c", "phash": "ffffffffffffffff", "clip": None},  # hamming 64
        ]
        result = _greedy_max_min_select(candidates, pool_ref, "phash_diverse", 3)
        # First pick: c (most dissimilar to p). Then ref = [p, c].
        # Second pick: a or b (most dissimilar from both p and c).
        # Third: the remaining one.
        assert result[0] == "c"
        assert len(result) == 3
        assert set(result) == {"a", "b", "c"}


# ══════════════════════════════════════════════════════════════════════════════
# Section F: Log Point 8 (AC #10)
# ══════════════════════════════════════════════════════════════════════════════


class TestLogPoint8:
    """Log point 8: pool_routing (section 11)."""

    def test_log_emits_debug_on_normal_routing(self, caplog):
        """Default level is DEBUG when no rebalancing occurs."""
        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.pool_routing"):
            _log_pool_routing(
                project_id=PID,
                destination="test_pool",
                verified_outcome="Accept",
                pool_count=5,
                target_count=5,
                rebalancing_triggered=False,
                rebalanced_keys=[],
            )
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.DEBUG
        assert "pool_routing" in caplog.records[0].__dict__.get("component", "")

    def test_log_escalates_to_info_on_rebalancing(self, caplog):
        """Escalates to INFO when rebalancing changes pool membership."""
        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.pool_routing"):
            _log_pool_routing(
                project_id=PID,
                destination="test_pool",
                verified_outcome="Edit",
                pool_count=8,
                target_count=8,
                rebalancing_triggered=True,
                rebalanced_keys=["img_001", "img_002"],
            )
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.INFO

    def test_log_details_contain_required_fields(self, caplog):
        """Details dict has destination, verified_outcome, pool_count, target_count,
        rebalancing_triggered per section 11 log point 8 requirements."""
        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.pool_routing"):
            _log_pool_routing(
                project_id=PID,
                destination=None,
                verified_outcome="Accept",
                pool_count=4,
                target_count=4,
                rebalancing_triggered=False,
                rebalanced_keys=[],
            )
        record = caplog.records[0]
        details = record.__dict__.get("details", {})
        assert details["destination"] is None
        assert details["verified_outcome"] == "Accept"
        assert details["pool_count"] == 4
        assert details["target_count"] == 4
        assert details["rebalancing_triggered"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Section G: Helper functions
# ══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    """Unit tests for pure helper functions."""

    def test_compute_pool_target(self):
        assert _compute_pool_target(10, 0.40) == 4
        assert _compute_pool_target(1, 0.40) == 0
        assert _compute_pool_target(0, 0.40) == 0
        assert _compute_pool_target(100, 0.40) == 40
        assert _compute_pool_target(3, 0.40) == 1
        assert _compute_pool_target(7, 0.50) == 3

    def test_determine_pool_diversity_mode(self):
        assert _determine_pool_diversity_mode(50, 50) == "clip_diverse"
        assert _determine_pool_diversity_mode(51, 50) == "clip_diverse"
        assert _determine_pool_diversity_mode(49, 50) == "phash_diverse"
        assert _determine_pool_diversity_mode(0, 50) == "phash_diverse"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the review selector.

Covers the review selector endpoint, the diversity-driven selection
algorithm, the example state machine (cold start, Omitted excluded),
and per-log-point operational logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_example_row,
    add_project_row,
    create_project_via_api,
    make_api_client,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now

# ── Helpers ─────────────────────────────────────────────────────────────────


def _setup_project_db(tmp_path: Path, project_id: str = "test-proj") -> tuple:
    """Create an isolated project database and return (engine, project_dir)."""
    from vlm_feedback_loop.db.engine import open_project_db

    project_dir = tmp_path / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    engine = open_project_db(project_dir)
    return engine, project_dir


def _add_project(
    session: Session,
    project_id: str,
    project_dir: str,
    **overrides,
) -> None:
    """Insert a Project record with sensible defaults."""
    overrides = {
        "name": "Test Project",
        "embedding_provider": "none",
        **overrides,
    }
    add_project_row(session, project_id, project_dir, **overrides)


def _add_example(
    session: Session,
    project_id: str,
    key: str,
    *,
    state: str = "Unlabeled",
    phash: str | None = None,
    clip_embedding_present: bool = False,
    prior_verified_label_ref: str | None = None,
    prior_verified_outcome: str | None = None,
) -> None:
    """Insert an Example record."""
    add_example_row(
        session,
        project_id,
        key,
        state=state,
        phash=phash,
        clip_embedding_present=clip_embedding_present,
        prior_verified_label_ref=prior_verified_label_ref,
        prior_verified_outcome=prior_verified_outcome,
    )


def _add_label(
    session: Session,
    project_id: str,
    key: str,
    *,
    label_status: str = "auto_labeled",
    guidance_id: str = "g1",
) -> None:
    """Insert a Label record."""
    from vlm_feedback_loop.db.models.label import Label

    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=project_id,
            example_key=key,
            label_status=label_status,
            guidance_id=guidance_id,
            inference_invocation_id=generate_uuid4(),
            label_json={"test": True},
            labeled_at=utc_now(),
        )
    )


def _cleanup_selector():
    """Reset module-level selector state (mode tracker + embedding cache).

    Called mid-test by the determinism tests to restore initial state
    between two selector runs; the autouse fixture below handles the
    between-tests reset.
    """
    from vlm_feedback_loop.services.clip_embedding_service import embedding_cache
    from vlm_feedback_loop.services.review_selector_service import (
        _reset_mode_tracker,
    )

    _reset_mode_tracker()
    embedding_cache.invalidate("test-proj")


@pytest.fixture(autouse=True)
def _reset_selector_state():
    """Reset selector module state after every test, even on failure."""
    yield
    _cleanup_selector()


# ═══════════════════════════════════════════════════════════════════════════
# Section A: Similarity functions
# ═══════════════════════════════════════════════════════════════════════════


class TestSimPhash:
    """Unit tests for sim_phash()."""

    def test_identical_hashes_returns_1(self):
        from vlm_feedback_loop.services.review_selector_service import sim_phash

        assert sim_phash("0000000000000000", "0000000000000000") == 1.0
        assert sim_phash("ffffffffffffffff", "ffffffffffffffff") == 1.0

    def test_completely_different_returns_0(self):
        from vlm_feedback_loop.services.review_selector_service import sim_phash

        # All 64 bits differ
        assert sim_phash("0000000000000000", "ffffffffffffffff") == 0.0

    def test_none_input_returns_0(self):
        from vlm_feedback_loop.services.review_selector_service import sim_phash

        assert sim_phash(None, "0000000000000000") == 0.0
        assert sim_phash("0000000000000000", None) == 0.0
        assert sim_phash(None, None) == 0.0

    def test_partial_difference(self):
        from vlm_feedback_loop.services.review_selector_service import sim_phash

        # 1 bit different out of 64 → similarity = 63/64
        result = sim_phash("0000000000000000", "0000000000000001")
        assert abs(result - 63.0 / 64.0) < 1e-9


class TestSimClip:
    """Unit tests for sim_clip()."""

    def test_identical_vectors_returns_1(self):
        from vlm_feedback_loop.services.review_selector_service import sim_clip

        v = [1.0, 2.0, 3.0]
        assert abs(sim_clip(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_returns_0(self):
        from vlm_feedback_loop.services.review_selector_service import sim_clip

        assert abs(sim_clip([1.0, 0.0], [0.0, 1.0])) < 1e-9

    def test_zero_vector_returns_0(self):
        from vlm_feedback_loop.services.review_selector_service import sim_clip

        assert sim_clip([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert sim_clip([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_known_cosine_value(self):
        import math

        from vlm_feedback_loop.services.review_selector_service import sim_clip

        # cos(45 degrees) = sqrt(2)/2 ≈ 0.7071
        v1 = [1.0, 0.0]
        v2 = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2)
        assert abs(sim_clip(v1, v2) - expected) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# Section B: Mode selection logic
# ═══════════════════════════════════════════════════════════════════════════


class TestDetermineSelectionMode:
    """Unit tests for _determine_selection_mode()."""

    def test_auto_below_threshold_returns_phash(self):
        from vlm_feedback_loop.services.review_selector_service import (
            _determine_selection_mode,
        )

        assert _determine_selection_mode(49, "auto", 50) == "phash_diverse"

    def test_auto_at_threshold_returns_clip(self):
        from vlm_feedback_loop.services.review_selector_service import (
            _determine_selection_mode,
        )

        assert _determine_selection_mode(50, "auto", 50) == "clip_diverse"

    def test_force_phash_ignores_clip_count(self):
        from vlm_feedback_loop.services.review_selector_service import (
            _determine_selection_mode,
        )

        assert _determine_selection_mode(1000, "phash_diverse", 50) == "phash_diverse"

    def test_force_clip_above_threshold(self):
        from vlm_feedback_loop.services.review_selector_service import (
            _determine_selection_mode,
        )

        assert _determine_selection_mode(50, "clip_diverse", 50) == "clip_diverse"

    def test_force_clip_below_threshold_falls_back(self):
        from vlm_feedback_loop.services.review_selector_service import (
            _determine_selection_mode,
        )

        assert _determine_selection_mode(10, "clip_diverse", 50) == "phash_diverse"


# ═══════════════════════════════════════════════════════════════════════════
# Section C: Review selector endpoint
# ═══════════════════════════════════════════════════════════════════════════


class TestEndpointResponseShape:
    """AC1: GET .../review_selector/next returns all required fields."""

    def test_response_has_required_fields(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        resp = client.get(f"/v1/projects/{pid}/review_selector/next")
        assert resp.status_code == 200
        data = resp.json()
        assert "example_key" in data
        assert "example_state" in data
        assert "has_existing_label" in data
        assert "selection_mode" in data
        assert "queue_empty" in data
        assert "storage_ref" in data
        assert "prior_verified_label_ref" in data

    def test_404_for_unknown_project(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        resp = client.get("/v1/projects/nonexistent/review_selector/next")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"


class TestEndpointQueueEmpty:
    """AC2: queue_empty=true when no eligible examples."""

    def test_empty_queue_returns_null_key(self, tmp_path: Path):
        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        resp = client.get(f"/v1/projects/{pid}/review_selector/next")
        data = resp.json()
        assert data["queue_empty"] is True
        assert data["example_key"] is None
        assert data["example_state"] is None
        assert data["storage_ref"] is None
        assert data["prior_verified_label_ref"] is None


class TestEndpointExampleFields:
    """The next response carries the selected Example's
    ``storage_ref`` and ``prior_verified_label_ref`` so the labeling screen
    renders the missing-image path and prior-label hints without re-querying
    the examples list."""

    def test_selected_example_carries_storage_ref_and_prior_ref(self, tmp_path: Path):
        from vlm_feedback_loop.services.project_service import get_project_engine

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        prior_snapshot = (
            '{"label_json": {"damage_type": "scratch"}, "verified_outcome": "Edit"}'
        )
        engine = get_project_engine(pid, str(tmp_path / "workspace"))
        with Session(engine) as session:
            add_example_row(
                session,
                pid,
                "img1",
                storage_ref="/data/images/img1.jpg",
                prior_verified_label_ref=prior_snapshot,
                prior_verified_outcome="Edit",
            )
            session.commit()

        resp = client.get(f"/v1/projects/{pid}/review_selector/next")
        assert resp.status_code == 200
        data = resp.json()
        assert data["example_key"] == "img1"
        assert data["storage_ref"] == "/data/images/img1.jpg"
        assert data["prior_verified_label_ref"] == prior_snapshot

    def test_example_without_prior_returns_null_prior_ref(self, tmp_path: Path):
        from vlm_feedback_loop.services.project_service import get_project_engine

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        engine = get_project_engine(pid, str(tmp_path / "workspace"))
        with Session(engine) as session:
            add_example_row(session, pid, "img1")
            session.commit()

        resp = client.get(f"/v1/projects/{pid}/review_selector/next")
        data = resp.json()
        assert data["example_key"] == "img1"
        assert data["storage_ref"] == "/fake/img1.jpg"
        assert data["prior_verified_label_ref"] is None


class TestEndpointHasExistingLabel:
    """AC3: has_existing_label=true for Auto-Labeled with Label record."""

    def test_auto_labeled_with_label_returns_true(self, tmp_path: Path):
        from vlm_feedback_loop.services.project_service import (
            get_project_engine,
        )

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        workspace = tmp_path / "workspace"
        engine = get_project_engine(pid, str(workspace))
        with Session(engine) as session:
            _add_example(session, pid, "img1", state="Auto-Labeled", phash="a" * 16)
            _add_label(session, pid, "img1", label_status="auto_labeled")
            session.commit()

        resp = client.get(f"/v1/projects/{pid}/review_selector/next")
        data = resp.json()
        assert data["has_existing_label"] is True
        assert data["example_key"] == "img1"

    def test_unlabeled_returns_false(self, tmp_path: Path):
        from vlm_feedback_loop.services.project_service import (
            get_project_engine,
        )

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        engine = get_project_engine(pid, str(tmp_path / "workspace"))
        with Session(engine) as session:
            _add_example(session, pid, "img1", state="Unlabeled", phash="a" * 16)
            session.commit()

        resp = client.get(f"/v1/projects/{pid}/review_selector/next")
        data = resp.json()
        assert data["has_existing_label"] is False


class TestEndpointStatePersisted:
    """AC4: selector state persisted on Project record."""

    def test_scheduler_state_written_to_project(self, tmp_path: Path):
        from vlm_feedback_loop.services.project_service import (
            get_project,
            get_project_engine,
        )

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        engine = get_project_engine(pid, str(tmp_path / "workspace"))
        with Session(engine) as session:
            _add_example(session, pid, "img1", state="Unlabeled", phash="a" * 16)
            session.commit()

        client.get(f"/v1/projects/{pid}/review_selector/next")

        proj = get_project(pid, str(tmp_path / "workspace"))
        assert proj is not None
        state = proj.review_selector_scheduler_state
        assert state is not None
        assert "recent_window" in state
        assert "img1" in state["recent_window"]

    def test_repeated_calls_build_recent_window(self, tmp_path: Path):
        from vlm_feedback_loop.services.project_service import (
            get_project,
            get_project_engine,
        )

        client = make_api_client(tmp_path)
        project = create_project_via_api(client)
        pid = project["project_id"]

        engine = get_project_engine(pid, str(tmp_path / "workspace"))
        with Session(engine) as session:
            _add_example(session, pid, "img1", state="Unlabeled", phash="0" * 16)
            _add_example(session, pid, "img2", state="Unlabeled", phash="f" * 16)
            session.commit()

        client.get(f"/v1/projects/{pid}/review_selector/next")
        client.get(f"/v1/projects/{pid}/review_selector/next")

        proj = get_project(pid, str(tmp_path / "workspace"))
        state = proj.review_selector_scheduler_state
        assert len(state["recent_window"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Section D: Diversity-driven review selector
# ═══════════════════════════════════════════════════════════════════════════


class TestClipCosinePhashHamming:
    """AC5: CLIP-diverse uses cosine; pHash-diverse uses hamming."""

    def test_phash_mode_uses_hamming(self, tmp_path: Path):
        """In pHash mode, the item most dissimilar (by hamming) to the
        recent window is selected."""
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            # Recent window will contain "recent" with phash 0000...
            _add_example(session, pid, "recent", state="Verified", phash="0" * 16)
            # Candidate near recent (1 bit diff)
            _add_example(session, pid, "near", phash="0000000000000001")
            # Candidate far from recent (all bits diff)
            _add_example(session, pid, "far", phash="f" * 16)
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = {"recent_window": ["recent"]}
            session.commit()

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            result = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert result.example_key == "far"  # most dissimilar by hamming

    def test_clip_mode_uses_cosine(self, tmp_path: Path):
        """In CLIP mode, the item with lowest cosine similarity to recent
        window is selected."""
        from vlm_feedback_loop.services.clip_embedding_service import embedding_cache
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir), embedding_dim=3)
            _add_example(
                session,
                pid,
                "recent",
                state="Verified",
                phash="0" * 16,
                clip_embedding_present=True,
            )
            _add_example(
                session, pid, "near", phash="a" * 16, clip_embedding_present=True
            )
            _add_example(
                session, pid, "far", phash="b" * 16, clip_embedding_present=True
            )
            session.commit()

        # Populate CLIP cache: recent=[1,0,0], near=[0.9,0.1,0], far=[0,1,0]
        embedding_cache.put(pid, "recent", [1.0, 0.0, 0.0])
        embedding_cache.put(pid, "near", [0.9, 0.1, 0.0])  # high cosine with recent
        embedding_cache.put(pid, "far", [0.0, 1.0, 0.0])  # orthogonal to recent

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = {"recent_window": ["recent"]}
            session.commit()

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            result = select_next(engine, proj, clip_switchover_min_count=1)

        assert result.example_key == "far"  # lowest cosine to recent
        assert result.selection_mode == "clip_diverse"


class TestClipMatrixEquivalence:
    """The cache-matrix CLIP fast path picks the same image as the exact
    per-pair sim_clip fallback — the vectorized optimization must never
    change which example the SME is shown."""

    def test_matrix_path_matches_pure_python(self):
        from vlm_feedback_loop.services.clip_embedding_service import _EmbeddingCache
        from vlm_feedback_loop.services.review_selector_service import (
            _greedy_select_next,
        )

        candidates = [
            {"key": "a", "phash": None, "clip": [1.0, 0.0, 0.0]},
            {"key": "b", "phash": None, "clip": [0.9, 0.1, 0.0]},
            {"key": "c", "phash": None, "clip": [0.0, 1.0, 0.0]},
            {"key": "d", "phash": None, "clip": [0.2, 0.2, 0.9]},
        ]
        window = [
            {"key": "w1", "phash": None, "clip": [1.0, 0.0, 0.0]},
            {"key": "w2", "phash": None, "clip": [0.0, 0.0, 1.0]},
        ]
        cache = _EmbeddingCache()
        for entry in candidates + window:
            cache.put("p", entry["key"], entry["clip"])
        clip_matrix = cache.get_normalized_matrix("p")

        fallback = _greedy_select_next(candidates, window, "clip_diverse")
        fast = _greedy_select_next(candidates, window, "clip_diverse", clip_matrix)
        assert fast[0] == fallback[0]


class TestMinMaxDiversity:
    """AC6: selects most dissimilar + stable ascending-key tiebreak."""

    def test_selects_most_dissimilar(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "recent", state="Verified", phash="0" * 16)
            # Three candidates at different hamming distances
            _add_example(session, pid, "c1", phash="0000000000000001")  # 1 bit diff
            _add_example(session, pid, "c2", phash="00000000000000ff")  # 8 bits diff
            _add_example(session, pid, "c3", phash="f" * 16)  # 64 bits diff
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = {"recent_window": ["recent"]}
            session.commit()

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            result = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert result.example_key == "c3"  # most dissimilar

    def test_tiebreak_ascending_key(self, tmp_path: Path):
        """When two candidates have equal diversity score, the one with
        the lexicographically smaller key wins."""
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        # All candidates have the same phash → same score → tie-break by key
        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "recent", state="Verified", phash="0" * 16)
            _add_example(session, pid, "bbb", phash="f" * 16)
            _add_example(session, pid, "aaa", phash="f" * 16)
            _add_example(session, pid, "ccc", phash="f" * 16)
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = {"recent_window": ["recent"]}
            session.commit()

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            result = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert result.example_key == "aaa"  # ascending key wins tie


class TestAutoModeSelection:
    """AC7: REVIEW_SELECTION_MODE=auto uses CLIP when available, else pHash."""

    def test_auto_with_enough_clip_uses_clip(self, tmp_path: Path):
        from vlm_feedback_loop.services.clip_embedding_service import embedding_cache
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir), embedding_dim=2)
            _add_example(
                session, pid, "e1", phash="0" * 16, clip_embedding_present=True
            )
            _add_example(
                session, pid, "e2", phash="f" * 16, clip_embedding_present=True
            )
            session.commit()

        embedding_cache.put(pid, "e1", [1.0, 0.0])
        embedding_cache.put(pid, "e2", [0.0, 1.0])

        with Session(engine) as session:
            from vlm_feedback_loop.db.models.project import Project as PModel

            proj = session.get(PModel, pid)
            result = select_next(
                engine, proj, review_selection_mode="auto", clip_switchover_min_count=2
            )

        assert result.selection_mode == "clip_diverse"

    def test_auto_without_clip_uses_phash(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "e1", phash="0" * 16)
            session.commit()

        with Session(engine) as session:
            from vlm_feedback_loop.db.models.project import Project as PModel

            proj = session.get(PModel, pid)
            result = select_next(
                engine, proj, review_selection_mode="auto", clip_switchover_min_count=50
            )

        assert result.selection_mode == "phash_diverse"


class TestNoRandomMode:
    """AC8: no random selection mode; pHash is baseline."""

    def test_phash_diverse_is_default_mode(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "e1", phash="0" * 16)
            session.commit()

        with Session(engine) as session:
            from vlm_feedback_loop.db.models.project import Project as PModel

            proj = session.get(PModel, pid)
            result = select_next(engine, proj)

        # Default settings: auto mode with clip_switchover=50, no CLIP → pHash
        assert result.selection_mode == "phash_diverse"

    def test_same_inputs_same_output(self, tmp_path: Path):
        """Determinism: same state → same selection."""
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "e1", phash="0" * 16)
            _add_example(session, pid, "e2", phash="f" * 16)
            session.commit()

        # First call
        with Session(engine) as session:
            from vlm_feedback_loop.db.models.project import Project as PModel

            proj = session.get(PModel, pid)
            r1 = select_next(engine, proj, review_selection_mode="phash_diverse")

        # Reset state to initial
        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = None
            session.commit()
        _cleanup_selector()

        # Second call — same initial state
        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r2 = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert r1.example_key == r2.example_key


class TestReproducible:
    """AC9: reproducible given persisted state."""

    def test_same_state_same_selection(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "a", phash="0" * 16)
            _add_example(session, pid, "b", phash="8" * 16)
            _add_example(session, pid, "c", phash="f" * 16)
            session.commit()

        # Set a known state
        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = {"recent_window": ["a"]}
            session.commit()

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r1 = select_next(engine, proj, review_selection_mode="phash_diverse")

        # Restore the same state
        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = {"recent_window": ["a"]}
            session.commit()
        _cleanup_selector()

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r2 = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert r1.example_key == r2.example_key

    def test_different_recent_window_different_selection(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            # "a" is close to "0000" and "c" is close to "ffff"
            _add_example(session, pid, "a", phash="0000000000000001")
            _add_example(session, pid, "c", phash="fffffffffffffffe")
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        # Window has "0000" → should pick "c" (most distant)
        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = None
            session.commit()

        # First call picks one (no window → ascending key → "a")
        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r1 = select_next(engine, proj, review_selection_mode="phash_diverse")

        # Second call has "a" in window → picks most dissimilar
        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r2 = select_next(engine, proj, review_selection_mode="phash_diverse")

        # First selects "a" (ascending, empty window), second selects "c" (diversity)
        assert r1.example_key == "a"
        assert r2.example_key == "c"


# ═══════════════════════════════════════════════════════════════════════════
# Section E: Data model
# ═══════════════════════════════════════════════════════════════════════════


class TestStateTransitions:
    """AC10: only Unlabeled + Auto-Labeled eligible for selector."""

    def test_only_unlabeled_and_autolabeled_eligible(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "unlabeled1", state="Unlabeled", phash="a" * 16)
            _add_example(session, pid, "auto1", state="Auto-Labeled", phash="b" * 16)
            session.commit()

        with Session(engine) as session:
            from vlm_feedback_loop.db.models.project import Project as PModel

            proj = session.get(PModel, pid)
            result = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert result.queue_empty is False
        assert result.example_key in ("unlabeled1", "auto1")

    def test_verified_examples_not_in_candidates(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "verified1", state="Verified", phash="a" * 16)
            _add_example(session, pid, "unlabeled1", state="Unlabeled", phash="b" * 16)
            session.commit()

        with Session(engine) as session:
            from vlm_feedback_loop.db.models.project import Project as PModel

            proj = session.get(PModel, pid)
            result = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert result.example_key == "unlabeled1"


class TestColdStart:
    """AC11: system operates with Verified=0."""

    def test_zero_verified_still_functions(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "e1", phash="a" * 16)
            session.commit()

        with Session(engine) as session:
            from vlm_feedback_loop.db.models.project import Project as PModel

            proj = session.get(PModel, pid)
            result = select_next(engine, proj)

        assert result.queue_empty is False
        assert result.example_key == "e1"


class TestOmittedExcluded:
    """AC12: Omitted examples excluded from review selector."""

    def test_omitted_not_in_candidates(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "omitted1", state="Omitted", phash="a" * 16)
            _add_example(session, pid, "unlabeled1", state="Unlabeled", phash="b" * 16)
            session.commit()

        with Session(engine) as session:
            from vlm_feedback_loop.db.models.project import Project as PModel

            proj = session.get(PModel, pid)
            result = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert result.example_key == "unlabeled1"

    def test_only_omitted_gives_empty_queue(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "omitted1", state="Omitted", phash="a" * 16)
            session.commit()

        with Session(engine) as session:
            from vlm_feedback_loop.db.models.project import Project as PModel

            proj = session.get(PModel, pid)
            result = select_next(engine, proj)

        assert result.queue_empty is True
        assert result.example_key is None


# ═══════════════════════════════════════════════════════════════════════════
# Section F: Operational logging — log point 7 (1 acceptance item)
# ═══════════════════════════════════════════════════════════════════════════


class TestLogPoint7:
    """AC13: log point 7 emits on selection with required fields."""

    def test_emits_debug_on_selection(self, tmp_path: Path, caplog):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "e1", phash="a" * 16)
            session.commit()

        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.review_selector"):
            with Session(engine) as session:
                from vlm_feedback_loop.db.models.project import Project as PModel

                proj = session.get(PModel, pid)
                select_next(engine, proj, review_selection_mode="phash_diverse")

        assert any("Review selector" in r.message for r in caplog.records)
        selector_records = [r for r in caplog.records if "Review selector" in r.message]
        assert len(selector_records) >= 1
        rec = selector_records[0]
        assert rec.levelno == logging.DEBUG

    def test_emits_info_on_mode_switch(self, tmp_path: Path, caplog):
        """Mode switch from pHash to CLIP should escalate to INFO."""
        from vlm_feedback_loop.services.clip_embedding_service import embedding_cache
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir), embedding_dim=2)
            _add_example(
                session, pid, "e1", phash="0" * 16, clip_embedding_present=True
            )
            _add_example(
                session, pid, "e2", phash="f" * 16, clip_embedding_present=True
            )
            session.commit()

        embedding_cache.put(pid, "e1", [1.0, 0.0])
        embedding_cache.put(pid, "e2", [0.0, 1.0])

        from vlm_feedback_loop.db.models.project import Project as PModel

        # First call: force pHash mode
        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.review_selector"):
            with Session(engine) as session:
                proj = session.get(PModel, pid)
                select_next(engine, proj, review_selection_mode="phash_diverse")

        # Reset state for second call
        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = None
            session.commit()

        caplog.clear()

        # Second call: auto mode with CLIP threshold=1 → switches to clip_diverse
        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.review_selector"):
            with Session(engine) as session:
                proj = session.get(PModel, pid)
                select_next(
                    engine,
                    proj,
                    review_selection_mode="auto",
                    clip_switchover_min_count=1,
                )

        selector_records = [r for r in caplog.records if "Review selector" in r.message]
        # Mode switched: should be INFO
        assert any(r.levelno == logging.INFO for r in selector_records)
        assert any("mode switched" in r.message for r in selector_records)

    def test_log_contains_required_fields(self, tmp_path: Path, caplog):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "e1", phash="0" * 16)
            _add_example(session, pid, "e2", phash="f" * 16)
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        # Set a window so diversity scoring occurs
        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = {"recent_window": ["e1"]}
            session.commit()

        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.review_selector"):
            with Session(engine) as session:
                proj = session.get(PModel, pid)
                select_next(engine, proj, review_selection_mode="phash_diverse")

        selector_records = [r for r in caplog.records if "Review selector" in r.message]
        assert len(selector_records) >= 1
        rec = selector_records[0]
        details = getattr(rec, "details", None)
        assert details is not None
        assert "selection_mode" in details
        assert "candidate_set_size" in details
        assert "selected_example_key" in details
        assert "diversity_score" in details


# ═══════════════════════════════════════════════════════════════════════════
# Section G: Prior-label priority
# ═══════════════════════════════════════════════════════════════════════════


class TestPriorLabelPriority:
    """Prior-label examples (Tier 1) are exhausted before standard (Tier 2)."""

    def test_tier1_exhausted_before_tier2(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            # Tier 2: standard Unlabeled
            _add_example(session, pid, "standard1", phash="0" * 16)
            # Tier 1: has prior label
            _add_example(
                session,
                pid,
                "prior1",
                phash="f" * 16,
                prior_verified_label_ref='{"label": "old"}',
                prior_verified_outcome="Edit",
            )
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r = select_next(engine, proj, review_selection_mode="phash_diverse")

        # Tier 1 (prior label) should be selected first
        assert r.example_key == "prior1"

    def test_context_key_presented_first(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(
                session,
                pid,
                str(project_dir),
                schema_change_context_example_key="ctx_key",
            )
            _add_example(
                session,
                pid,
                "edit1",
                phash="a" * 16,
                prior_verified_label_ref='{"l": 1}',
                prior_verified_outcome="Edit",
            )
            _add_example(
                session,
                pid,
                "ctx_key",
                phash="b" * 16,
                prior_verified_label_ref='{"l": 2}',
                prior_verified_outcome="Accept",
            )
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r = select_next(engine, proj, review_selection_mode="phash_diverse")

        # Context key takes priority even over Edits
        assert r.example_key == "ctx_key"

    def test_edits_before_accepts_in_tier1(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(
                session,
                pid,
                "accept1",
                phash="a" * 16,
                prior_verified_label_ref='{"l": 1}',
                prior_verified_outcome="Accept",
            )
            _add_example(
                session,
                pid,
                "edit1",
                phash="b" * 16,
                prior_verified_label_ref='{"l": 2}',
                prior_verified_outcome="Edit",
            )
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r = select_next(engine, proj, review_selection_mode="phash_diverse")

        # Edits first within Tier 1
        assert r.example_key == "edit1"

    def test_context_key_cleared_after_presentation(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(
                session,
                pid,
                str(project_dir),
                schema_change_context_example_key="ctx_key",
            )
            _add_example(
                session,
                pid,
                "ctx_key",
                phash="a" * 16,
                prior_verified_label_ref='{"l": 1}',
                prior_verified_outcome="Edit",
            )
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            select_next(engine, proj, review_selection_mode="phash_diverse")

        # Context key should be cleared on the Project record
        with Session(engine) as session:
            proj = session.get(PModel, pid)
            assert proj.schema_change_context_example_key is None


# ═══════════════════════════════════════════════════════════════════════════
# Section H: Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case behaviors."""

    def test_single_example_returned(self, tmp_path: Path):
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "only_one", phash="a" * 16)
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r = select_next(engine, proj)

        assert r.example_key == "only_one"
        assert r.queue_empty is False

    def test_identical_phash_tiebreak_by_key(self, tmp_path: Path):
        """All candidates have the same phash → all equally similar
        to the recent window → tie-break by ascending key."""
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        same_hash = "abcdef0123456789"
        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            _add_example(session, pid, "recent", state="Verified", phash=same_hash)
            _add_example(session, pid, "zzz", phash=same_hash)
            _add_example(session, pid, "aaa", phash=same_hash)
            _add_example(session, pid, "mmm", phash=same_hash)
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            proj.review_selector_scheduler_state = {"recent_window": ["recent"]}
            session.commit()

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert r.example_key == "aaa"

    def test_no_signal_fallback_to_ascending_key(self, tmp_path: Path):
        """Candidates without pHash or CLIP → fallback to ascending key."""
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            # No phash on any candidate
            _add_example(session, pid, "zzz", phash=None)
            _add_example(session, pid, "aaa", phash=None)
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            r = select_next(engine, proj, review_selection_mode="phash_diverse")

        assert r.example_key == "aaa"  # ascending key fallback

    def test_recent_window_trimmed_to_k(self, tmp_path: Path):
        """Window is trimmed to REVIEW_RECENT_WINDOW_K."""
        from vlm_feedback_loop.services.review_selector_service import select_next

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, str(project_dir))
            for i in range(5):
                _add_example(session, pid, f"e{i}", phash=f"{i:016x}")
            session.commit()

        from vlm_feedback_loop.db.models.project import Project as PModel

        # Make 5 calls with window size k=3
        for _ in range(5):
            with Session(engine) as session:
                proj = session.get(PModel, pid)
                select_next(engine, proj, review_recent_window_k=3)

        with Session(engine) as session:
            proj = session.get(PModel, pid)
            state = proj.review_selector_scheduler_state
            assert len(state["recent_window"]) <= 3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for guidance_service Schema Refinement Reminders.

Reminders nudge the SME to review their schema after enough Verified
labels accumulate: backend-owned thresholds, higher-of-two selection,
dismissal counting, and suppression once Guidance was edited post-save.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from conftest import (
    FIXTURE_SCHEMA,
    add_guidance_row,
    add_project_row,
    make_settings,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.services.guidance_service import (
    dismiss_reminder,
    get_reminder_status,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _setup_project_db(tmp_path, project_id: str = "test-proj"):
    engine, project_dir, _ = open_project_workspace(tmp_path, project_id)
    return engine, str(project_dir)


def _add_guidance(session, project_id, guidance_id, version_number=1):
    add_guidance_row(
        session,
        project_id,
        guidance_id,
        FIXTURE_SCHEMA,
        version_number=version_number,
        description="Classify damage.",
        rules="Focus on visible defects.",
    )


class TestReminderStatus:
    """AC: Schema refinement reminders fire at correct thresholds."""

    def _setup_with_verified(
        self, tmp_path, verified_count, dismissed=0, guidance_versions=1
    ):
        pid = "test-proj"
        gid = generate_uuid4()
        engine, project_dir = _setup_project_db(tmp_path, pid)
        workspace = str(tmp_path / "workspace")

        with Session(engine) as session:
            add_project_row(
                session,
                pid,
                project_dir,
                active_guidance_id=gid,
                schema_refinement_reminders_dismissed=dismissed,
            )
            _add_guidance(session, pid, gid, version_number=1)
            # Add extra guidance versions if needed (to test "edited post-save")
            for v in range(2, guidance_versions + 1):
                _add_guidance(session, pid, generate_uuid4(), version_number=v)
            # Add Verified Labels
            for i in range(verified_count):
                session.add(
                    Example(
                        example_key=f"img_{i}",
                        project_id=pid,
                        storage_ref=f"/fake/{i}.jpg",
                        ingested_at=utc_now(),
                        source_metadata={},
                        state="Verified",
                        phash="a" * 16,
                    )
                )
                session.add(
                    Label(
                        label_id=generate_uuid4(),
                        project_id=pid,
                        example_key=f"img_{i}",
                        label_status="verified",
                        guidance_id=gid,
                        inference_invocation_id=generate_uuid4(),
                        label_json={"severity": "high"},
                        labeled_at=utc_now(),
                        verified_outcome="Accept",
                        verified_at=utc_now(),
                    )
                )
            session.commit()

        from vlm_feedback_loop.services.project_service import set_project_engine

        set_project_engine(pid, engine)

        settings = make_settings(tmp_path / "workspace")
        return pid, workspace, settings

    def test_first_reminder_at_threshold_10(self, tmp_path):
        pid, ws, settings = self._setup_with_verified(tmp_path, verified_count=10)
        result = get_reminder_status(pid, ws, settings)
        assert result is not None
        assert result["active_reminder"] == 1

    def test_second_reminder_at_threshold_35(self, tmp_path):
        pid, ws, settings = self._setup_with_verified(
            tmp_path,
            verified_count=35,
            dismissed=1,
        )
        result = get_reminder_status(pid, ws, settings)
        assert result is not None
        assert result["active_reminder"] == 2

    def test_no_reminder_below_threshold(self, tmp_path):
        pid, ws, settings = self._setup_with_verified(tmp_path, verified_count=5)
        result = get_reminder_status(pid, ws, settings)
        assert result is not None
        assert result["active_reminder"] is None

    def test_suppressed_if_guidance_edited_post_save(self, tmp_path):
        pid, ws, settings = self._setup_with_verified(
            tmp_path,
            verified_count=15,
            guidance_versions=2,
        )
        result = get_reminder_status(pid, ws, settings)
        assert result is not None
        assert result["active_reminder"] is None  # suppressed

    def test_only_higher_fires_when_both_crossed(self, tmp_path):
        pid, ws, settings = self._setup_with_verified(
            tmp_path,
            verified_count=40,
            dismissed=0,
        )
        result = get_reminder_status(pid, ws, settings)
        assert result is not None
        assert result["active_reminder"] == 2  # higher fires, not both

    def test_threshold_0_disables(self, tmp_path):
        pid, ws, _ = self._setup_with_verified(tmp_path, verified_count=15)
        settings = make_settings(
            tmp_path / "workspace",
            SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1=0,
            SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2=0,
        )
        result = get_reminder_status(pid, ws, settings)
        assert result is not None
        assert result["active_reminder"] is None


class TestReminderDismiss:
    """AC: Dismissal increments counter."""

    def test_increments_dismissed_count(self, tmp_path):
        pid = "test-proj"
        gid = generate_uuid4()
        engine, project_dir = _setup_project_db(tmp_path, pid)
        workspace = str(tmp_path / "workspace")

        with Session(engine) as session:
            add_project_row(
                session,
                pid,
                project_dir,
                active_guidance_id=gid,
                schema_refinement_reminders_dismissed=0,
            )
            session.commit()

        from vlm_feedback_loop.services.project_service import set_project_engine

        set_project_engine(pid, engine)

        result = dismiss_reminder(pid, workspace)
        assert not isinstance(result, str)
        assert result["dismissed_count"] == 1

        # Dismiss again
        result2 = dismiss_reminder(pid, workspace)
        assert result2["dismissed_count"] == 2

    def test_second_reminder_fires_after_first_dismissed(self, tmp_path):
        pid = "test-proj"
        gid = generate_uuid4()
        engine, project_dir = _setup_project_db(tmp_path, pid)
        workspace = str(tmp_path / "workspace")
        settings = make_settings(tmp_path / "workspace")

        with Session(engine) as session:
            add_project_row(
                session,
                pid,
                project_dir,
                active_guidance_id=gid,
                schema_refinement_reminders_dismissed=0,
            )
            _add_guidance(session, pid, gid)
            # Add 35 Verified labels
            for i in range(35):
                session.add(
                    Example(
                        example_key=f"img_{i}",
                        project_id=pid,
                        storage_ref=f"/fake/{i}.jpg",
                        ingested_at=utc_now(),
                        source_metadata={},
                        state="Verified",
                        phash="a" * 16,
                    )
                )
                session.add(
                    Label(
                        label_id=generate_uuid4(),
                        project_id=pid,
                        example_key=f"img_{i}",
                        label_status="verified",
                        guidance_id=gid,
                        inference_invocation_id=generate_uuid4(),
                        label_json={"severity": "high"},
                        labeled_at=utc_now(),
                        verified_outcome="Accept",
                        verified_at=utc_now(),
                    )
                )
            session.commit()

        from vlm_feedback_loop.services.project_service import set_project_engine

        set_project_engine(pid, engine)

        # Before dismissal, reminder 2 fires (both crossed, higher wins)
        status = get_reminder_status(pid, workspace, settings)
        assert status["active_reminder"] == 2

        # Dismiss once (counter → 1)
        dismiss_reminder(pid, workspace)

        # Still fires as reminder 2 (dismissed=1, need dismissed<2)
        status2 = get_reminder_status(pid, workspace, settings)
        assert status2["active_reminder"] == 2

        # Dismiss again (counter → 2)
        dismiss_reminder(pid, workspace)

        # Now no reminder (both dismissed)
        status3 = get_reminder_status(pid, workspace, settings)
        assert status3["active_reminder"] is None

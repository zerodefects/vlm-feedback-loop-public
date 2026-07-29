# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pool_service.assess_pool_class_coverage — the non-invasive
diagnostic guard for the order-sensitive Test/Train pool split (section 4.3).

A class-clustered labeling order (manifest-order autorun, class-sorted batch
import) can drive the Test Pool and Train Pool into holding *disjoint* class
sets. Because relevance-ICL evaluation draws exemplars only from the Train Pool,
any Test-Pool class with no Train-Pool representation is structurally unscorable
under relevance ICL. This guard detects that degenerate split. It does NOT
change pool routing.

Covered:
  - class-disjoint split  => degenerate=True with the right test_only_classes
  - balanced split        => degenerate=False
  - no Core enum/boolean  => applicable=False (benign, no warning)
  - boolean class field is recognized as a "class"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from conftest import add_guidance_row, add_project_row, open_project_workspace
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.services.pool_service import assess_pool_class_coverage

PID = "test-proj"
GID = "guid-001"


# ── Fixtures / helpers ───────────────────────────────────────────────────────


def _setup_project_db(tmp_path: Path, project_id: str = PID):
    engine, project_dir, _ = open_project_workspace(tmp_path, project_id)
    return engine, str(project_dir)


def _add_project(session: Session, project_dir: str, **overrides: Any) -> None:
    add_project_row(
        session, PID, project_dir, **{"active_guidance_id": GID, **overrides}
    )


def _add_guidance(
    session: Session,
    fields: list[dict[str, Any]],
    guidance_id: str = GID,
) -> None:
    add_guidance_row(
        session,
        PID,
        guidance_id,
        {"fields": fields},
        description="d",
        created_at=utc_now(),
    )


def _add_label(
    session: Session,
    key: str,
    class_value: Any,
    class_field: str = "defect_type",
    *,
    pool_assignment: str | None = None,
    guidance_id: str = GID,
) -> None:
    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=PID,
            example_key=key,
            label_status="verified",
            guidance_id=guidance_id,
            inference_invocation_id=generate_uuid4(),
            label_json={class_field: class_value, "rationale_note": "t"},
            labeled_at=utc_now(),
            verified_outcome="Edit",
            verified_at=utc_now(),
            edited_core_fields=[class_field],
            edited_aux_fields=[],
            rationale_source="sme_edited",
            pool_assignment=pool_assignment,
        )
    )


_ENUM_FIELD = [
    {"field_name": "defect_type", "type": "enum", "role": "core", "display_order": 0},
]


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_class_disjoint_split_is_degenerate(tmp_path):
    """Test Pool and Train Pool hold disjoint classes => degenerate=True with
    the exact set of test-only classes (the structurally unscorable ones)."""
    engine, pdir = _setup_project_db(tmp_path)
    with Session(engine) as s:
        _add_project(s, pdir)
        _add_guidance(s, _ENUM_FIELD)
        # Test Pool: classes A, B (the first-labeled contiguous classes)
        _add_label(s, "img_a1", "A", pool_assignment="test_pool")
        _add_label(s, "img_b1", "B", pool_assignment="test_pool")
        # Train Pool (non-pool): classes C, D — no overlap with Test
        _add_label(s, "img_c1", "C", pool_assignment=None)
        _add_label(s, "img_d1", "D", pool_assignment=None)
        s.commit()

    result = assess_pool_class_coverage(engine, PID)

    assert result["applicable"] is True
    assert result["class_field"] == "defect_type"
    assert result["test_classes"] == ["A", "B"]
    assert result["train_classes"] == ["C", "D"]
    assert result["test_only_classes"] == ["A", "B"]
    assert result["overlap_count"] == 0
    assert result["degenerate"] is True


def test_balanced_split_is_not_degenerate(tmp_path):
    """Every Test class also appears in the Train Pool => degenerate=False."""
    engine, pdir = _setup_project_db(tmp_path)
    with Session(engine) as s:
        _add_project(s, pdir)
        _add_guidance(s, _ENUM_FIELD)
        # Classes A and B each appear in both pools (CLIP-diverse-like order).
        _add_label(s, "img_a1", "A", pool_assignment="test_pool")
        _add_label(s, "img_b1", "B", pool_assignment="test_pool")
        _add_label(s, "img_a2", "A", pool_assignment=None)
        _add_label(s, "img_b2", "B", pool_assignment=None)
        # An extra train-only class C is fine — only Test classes must be covered.
        _add_label(s, "img_c1", "C", pool_assignment=None)
        s.commit()

    result = assess_pool_class_coverage(engine, PID)

    assert result["applicable"] is True
    assert result["test_classes"] == ["A", "B"]
    assert result["train_classes"] == ["A", "B", "C"]
    assert result["test_only_classes"] == []
    assert result["overlap_count"] == 2
    assert result["degenerate"] is False


def test_no_core_categorical_field_is_not_applicable(tmp_path):
    """A schema with no Core enum/boolean field => applicable=False, no warning
    signal (degenerate=False) even if the split would otherwise be disjoint."""
    engine, pdir = _setup_project_db(tmp_path)
    with Session(engine) as s:
        _add_project(s, pdir)
        # Only a free-text Core field and a numeric Aux field — no class.
        _add_guidance(
            s,
            [
                {
                    "field_name": "caption",
                    "type": "string",
                    "role": "core",
                    "display_order": 0,
                },
                {
                    "field_name": "count",
                    "type": "integer",
                    "role": "aux",
                    "display_order": 1,
                },
            ],
        )
        _add_label(s, "img_a1", "ignored", "caption", pool_assignment="test_pool")
        _add_label(s, "img_b1", "ignored", "caption", pool_assignment=None)
        s.commit()

    result = assess_pool_class_coverage(engine, PID)

    assert result["applicable"] is False
    assert result["class_field"] is None
    assert result["degenerate"] is False


def test_boolean_core_field_is_a_class(tmp_path):
    """A Core boolean field is recognized as the class; a Test-only True value
    with only False in Train is degenerate."""
    engine, pdir = _setup_project_db(tmp_path)
    with Session(engine) as s:
        _add_project(s, pdir)
        _add_guidance(
            s,
            [
                {
                    "field_name": "damaged",
                    "type": "boolean",
                    "role": "core",
                    "display_order": 0,
                }
            ],
        )
        _add_label(s, "img_t1", True, "damaged", pool_assignment="test_pool")
        _add_label(s, "img_f1", False, "damaged", pool_assignment=None)
        s.commit()

    result = assess_pool_class_coverage(engine, PID)

    assert result["applicable"] is True
    assert result["class_field"] == "damaged"
    # bool stringified -> "True" / "False"
    assert result["test_only_classes"] == ["True"]
    assert result["degenerate"] is True

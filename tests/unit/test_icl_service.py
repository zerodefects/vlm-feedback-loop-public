# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ICL selection, token budget, and Inference Contract.

Covers ICL selection and prompt assembly, token budget derivation,
per-log-point operational logging, and the Inference Contract for
Students.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from conftest import add_example_row, add_project_row, open_project_workspace
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.services.schema_core import RESERVED_FIELD_NAME

# ── Helpers ─────────────────────────────────────────────────────────────────


def _setup_project_db(tmp_path: Path, project_id: str = "test-proj"):
    engine, project_dir, _ = open_project_workspace(tmp_path, project_id, subdirs=())
    return engine, project_dir


def _add_project(session, project_id, project_dir, **overrides):
    add_project_row(
        session,
        project_id,
        str(project_dir),
        **{"embedding_provider": "none", **overrides},
    )


def _add_example(
    session,
    project_id,
    key,
    *,
    state="Verified",
    phash=None,
    clip_present=False,
):
    add_example_row(
        session,
        project_id,
        key,
        state=state,
        phash=phash,
        clip_embedding_present=clip_present,
    )


def _add_verified_edit(
    session,
    project_id,
    key,
    *,
    guidance_id="g1",
    label_json=None,
    labeled_at=None,
    pool_assignment=None,
):
    from vlm_feedback_loop.db.models.label import Label

    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=project_id,
            example_key=key,
            label_status="verified",
            guidance_id=guidance_id,
            inference_invocation_id=generate_uuid4(),
            label_json=label_json or {"rationale_note": "test", "damage_type": "crack"},
            labeled_at=labeled_at or utc_now(),
            verified_outcome="Edit",
            verified_at=utc_now(),
            pool_assignment=pool_assignment,
        )
    )


def _make_field(name, ftype, role="core", **kwargs):
    d = {
        "field_id": generate_uuid4(),
        "field_name": name,
        "type": ftype,
        "role": role,
        "display_order": kwargs.pop("display_order", 0),
    }
    d.update(kwargs)
    return d


# ═══════════════════════════════════════════════════════════════════════════
# Section A: Inference Contract
# ═══════════════════════════════════════════════════════════════════════════


class TestInferenceContract:
    """AC: Teacher contract and Student contract validation.

    The TEACHER_CONTRACT constant snapshot is pinned in
    ``test_evaluation_service.py::test_teacher_contract_constant``.
    """

    def test_extra_field_rejected(self):
        from pydantic import ValidationError

        from vlm_feedback_loop.schemas.inference_contract import (
            InferenceContract,
        )

        with pytest.raises(ValidationError):
            InferenceContract(
                output_field_mode="all",
                icl_field_mode="all",
                bogus_field="nope",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Section B: Per-field token estimation
# ═══════════════════════════════════════════════════════════════════════════


class TestFieldTokenEstimation:
    """AC: Per-field worst-case estimates match spec formulas."""

    def test_rationale_note_returns_160(self):
        from vlm_feedback_loop.services.token_budget_service import (
            estimate_field_tokens,
        )

        f = _make_field(RESERVED_FIELD_NAME, "string", role="aux")
        assert estimate_field_tokens(f) == 160

    def test_boolean_returns_6(self):
        from vlm_feedback_loop.services.token_budget_service import (
            estimate_field_tokens,
        )

        assert estimate_field_tokens(_make_field("is_ok", "boolean")) == 6

    def test_integer_returns_8(self):
        from vlm_feedback_loop.services.token_budget_service import (
            estimate_field_tokens,
        )

        assert estimate_field_tokens(_make_field("severity", "integer")) == 8

    def test_enum_formula(self):
        from vlm_feedback_loop.services.token_budget_service import (
            count_tokens,
            estimate_field_tokens,
        )

        f = _make_field(
            "damage_type", "enum", allowed_values=["crack", "dent", "scratch"]
        )
        result = estimate_field_tokens(f)
        key_overhead = count_tokens('"damage_type": ')
        max_val = max(count_tokens(v) for v in ["crack", "dent", "scratch"])
        assert result == key_overhead + max_val + 6

    def test_enum_set_formula(self):
        from vlm_feedback_loop.services.token_budget_service import (
            count_tokens,
            estimate_field_tokens,
        )

        vals = ["crush", "rip"]
        f = _make_field("types", "enum_set", allowed_values=vals)
        result = estimate_field_tokens(f)
        key_overhead = count_tokens('"types": ')
        sum_val = sum(count_tokens(v) for v in vals)
        array_overhead = 4 + max(0, len(vals) - 1) * 2
        assert result == key_overhead + sum_val + array_overhead

    def test_string_with_max_length(self):
        from vlm_feedback_loop.services.token_budget_service import (
            count_tokens,
            estimate_field_tokens,
        )

        f = _make_field("notes", "string", max_length=200)
        result = estimate_field_tokens(f)
        key_overhead = count_tokens('"notes": ')
        assert result == key_overhead + math.ceil(200 / 4)

    def test_string_without_max_length(self):
        from vlm_feedback_loop.services.token_budget_service import (
            estimate_field_tokens,
        )

        f = _make_field("notes", "string")
        assert estimate_field_tokens(f) == 200


# ═══════════════════════════════════════════════════════════════════════════
# Section C: Schema output budget
# ═══════════════════════════════════════════════════════════════════════════


class TestSchemaOutputEstimate:
    """AC: JSON_STRUCTURAL_OVERHEAD_TOKENS added, field_mode filtering works.

    The overhead constant (48) is asserted only through the field-mode
    sums below — a bare ``== 48`` on an empty schema just re-encoded the
    constant's value.
    """

    def test_field_mode_all_includes_everything(self):
        from vlm_feedback_loop.services.token_budget_service import (
            estimate_schema_output_tokens,
        )

        fields = [
            _make_field(RESERVED_FIELD_NAME, "string", role="aux"),
            _make_field("aux1", "boolean", role="aux"),
            _make_field("core1", "boolean", role="core"),
        ]
        total = estimate_schema_output_tokens(fields, "all")
        assert total == 48 + 160 + 6 + 6  # overhead + rationale + 2 booleans

    def test_field_mode_aux_and_core_excludes_rationale(self):
        from vlm_feedback_loop.services.token_budget_service import (
            estimate_schema_output_tokens,
        )

        fields = [
            _make_field(RESERVED_FIELD_NAME, "string", role="aux"),
            _make_field("aux1", "boolean", role="aux"),
            _make_field("core1", "boolean", role="core"),
        ]
        total = estimate_schema_output_tokens(fields, "aux_and_core")
        assert total == 48 + 6 + 6  # no rationale

    def test_field_mode_core_only(self):
        from vlm_feedback_loop.services.token_budget_service import (
            estimate_schema_output_tokens,
        )

        fields = [
            _make_field(RESERVED_FIELD_NAME, "string", role="aux"),
            _make_field("aux1", "boolean", role="aux"),
            _make_field("core1", "boolean", role="core"),
        ]
        total = estimate_schema_output_tokens(fields, "core_only")
        assert total == 48 + 6  # only core1


# ═══════════════════════════════════════════════════════════════════════════
# Section D: Token budget derivation — 8 acceptance items
# ═══════════════════════════════════════════════════════════════════════════


class TestTokenBudgetDerivation:
    """The token budget derivation formulas."""

    def _fields(self):
        return [
            _make_field(RESERVED_FIELD_NAME, "string", role="aux"),
            _make_field(
                "damage_type", "enum", role="core", allowed_values=["crack", "dent"]
            ),
        ]

    def test_max_output_derived_from_schema(self):
        from vlm_feedback_loop.services.token_budget_service import (
            derive_token_budget,
        )

        budget = derive_token_budget(self._fields(), "all", 256000, False)
        assert budget.schema_output_estimate > 48
        assert budget.base_output_tokens == max(256, 2 * budget.schema_output_estimate)
        assert budget.max_output_tokens > 0

    def test_thinking_on_adds_headroom(self):
        from vlm_feedback_loop.services.token_budget_service import (
            derive_token_budget,
        )

        budget = derive_token_budget(self._fields(), "all", 256000, True)
        assert budget.reasoning_headroom == 16384

    def test_thinking_off_zero_headroom(self):
        from vlm_feedback_loop.services.token_budget_service import (
            derive_token_budget,
        )

        budget = derive_token_budget(self._fields(), "all", 256000, False)
        assert budget.reasoning_headroom == 0

    def test_max_output_capped_by_fraction(self):
        from vlm_feedback_loop.services.token_budget_service import (
            derive_token_budget,
        )

        # With thinking ON, base+headroom is large. Verify cap.
        budget = derive_token_budget(self._fields(), "all", 1000, True)
        assert budget.max_output_tokens <= math.floor(1000 * 0.25)

    def test_effective_max_input_formula(self):
        from vlm_feedback_loop.services.token_budget_service import (
            derive_token_budget,
        )

        budget = derive_token_budget(self._fields(), "all", 256000, False)
        expected = math.floor((256000 - budget.max_output_tokens) * 0.85)
        assert budget.effective_max_input_tokens == expected

    def test_override_replaces_derived(self):
        from vlm_feedback_loop.services.token_budget_service import (
            derive_token_budget,
        )

        budget = derive_token_budget(
            self._fields(),
            "all",
            256000,
            False,
            runtime_prompt_output_max_tokens_override=999,
        )
        assert budget.max_output_tokens == 999

    def test_thinking_on_smaller_input_budget(self):
        from vlm_feedback_loop.services.token_budget_service import (
            derive_token_budget,
        )

        b_off = derive_token_budget(self._fields(), "all", 256000, False)
        b_on = derive_token_budget(self._fields(), "all", 256000, True)
        assert b_on.effective_max_input_tokens < b_off.effective_max_input_tokens


# ═══════════════════════════════════════════════════════════════════════════
# Section E: ICL eligibility — 2 acceptance items
# ═══════════════════════════════════════════════════════════════════════════


class TestICLEligibility:
    """AC: Only verified Edits eligible; cold start returns empty."""

    def test_only_verified_edits_eligible(self, tmp_path: Path):
        from vlm_feedback_loop.services.icl_service import query_icl_candidates

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"
        gid = "g1"

        with Session(engine) as session:
            _add_project(session, pid, project_dir, active_guidance_id=gid)

            # Eligible: verified Edit, no pool, matching guidance
            _add_example(session, pid, "edit1", phash="a" * 16)
            _add_verified_edit(session, pid, "edit1", guidance_id=gid)

            # Not eligible: verified Accept
            _add_example(session, pid, "accept1", phash="b" * 16)
            from vlm_feedback_loop.db.models.label import Label

            session.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=pid,
                    example_key="accept1",
                    label_status="verified",
                    guidance_id=gid,
                    inference_invocation_id=generate_uuid4(),
                    label_json={"test": True},
                    labeled_at=utc_now(),
                    verified_outcome="Accept",
                    verified_at=utc_now(),
                )
            )

            # Not eligible: in test pool
            _add_example(session, pid, "pool1", phash="c" * 16)
            _add_verified_edit(
                session, pid, "pool1", guidance_id=gid, pool_assignment="test_pool"
            )

            # Not eligible: wrong guidance
            _add_example(session, pid, "wrong_g", phash="d" * 16)
            _add_verified_edit(session, pid, "wrong_g", guidance_id="other_guidance")

            # Not eligible: auto_labeled
            _add_example(session, pid, "auto1", state="Auto-Labeled", phash="e" * 16)
            session.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=pid,
                    example_key="auto1",
                    label_status="auto_labeled",
                    guidance_id=gid,
                    inference_invocation_id=generate_uuid4(),
                    label_json={"test": True},
                    labeled_at=utc_now(),
                )
            )

            session.commit()

        with Session(engine) as session:
            candidates = query_icl_candidates(session, pid, gid)

        assert len(candidates) == 1
        assert candidates[0].example_key == "edit1"

    def test_cold_start_returns_empty(self, tmp_path: Path):
        from vlm_feedback_loop.services.icl_service import query_icl_candidates

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"

        with Session(engine) as session:
            _add_project(session, pid, project_dir, active_guidance_id="g1")
            session.commit()

        with Session(engine) as session:
            candidates = query_icl_candidates(session, pid, "g1")

        assert candidates == []

    def test_equal_timestamps_use_example_key_tie_break(self, tmp_path: Path):
        from vlm_feedback_loop.services.icl_service import query_icl_candidates

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"
        gid = "g1"
        timestamp = "2026-06-29T12:00:00Z"

        with Session(engine) as session:
            _add_project(session, pid, project_dir, active_guidance_id=gid)
            for key in ("z-edit", "a-edit", "m-edit"):
                _add_example(session, pid, key, phash="a" * 16)
                _add_verified_edit(
                    session, pid, key, guidance_id=gid, labeled_at=timestamp
                )
            session.commit()

        with Session(engine) as session:
            candidates = query_icl_candidates(session, pid, gid)

        assert [candidate.example_key for candidate in candidates] == [
            "a-edit",
            "m-edit",
            "z-edit",
        ]

    def test_pool_members_excluded(self, tmp_path: Path):
        from vlm_feedback_loop.services.icl_service import query_icl_candidates

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"
        gid = "g1"

        with Session(engine) as session:
            _add_project(session, pid, project_dir, active_guidance_id=gid)
            _add_example(session, pid, "pooled", phash="a" * 16)
            _add_verified_edit(
                session, pid, "pooled", guidance_id=gid, pool_assignment="test_pool"
            )
            session.commit()

        with Session(engine) as session:
            candidates = query_icl_candidates(session, pid, gid)

        assert len(candidates) == 0

    def test_storage_ref_attached_for_inline_icl_image_injection(self, tmp_path: Path):
        """``query_icl_candidates`` must populate ``ICLExample.storage_ref``.

        Inline ICL image injection requires invoke_teacher to
        batch-prepare each ICL example's image alongside the query image.
        If the SELECT joins only ``Example.phash``, the storage_ref is
        lost between Label and the prompt pipeline — ICL examples reach
        the model as text-JSON labels with no associated image. The test
        guards the join.
        """
        from vlm_feedback_loop.services.icl_service import query_icl_candidates

        engine, project_dir = _setup_project_db(tmp_path)
        pid = "test-proj"
        gid = "g1"

        with Session(engine) as session:
            _add_project(session, pid, project_dir, active_guidance_id=gid)
            _add_example(session, pid, "edit_a", phash="a" * 16)
            _add_verified_edit(session, pid, "edit_a", guidance_id=gid)
            _add_example(session, pid, "edit_b", phash="b" * 16)
            _add_verified_edit(session, pid, "edit_b", guidance_id=gid)
            session.commit()

        with Session(engine) as session:
            candidates = query_icl_candidates(session, pid, gid)

        assert {c.example_key for c in candidates} == {"edit_a", "edit_b"}
        for c in candidates:
            # _add_example sets storage_ref=f"/fake/{key}.jpg"
            assert c.storage_ref == f"/fake/{c.example_key}.jpg", (
                f"ICL candidate {c.example_key} missing storage_ref — "
                f"the join with Example.storage_ref regressed."
            )


# ═══════════════════════════════════════════════════════════════════════════
# Section F: ICL selection algorithm
# ═══════════════════════════════════════════════════════════════════════════


class TestICLSelection:
    """Relevance-only selection: rank by query similarity, cap at depth.

    Adaptive-K (the similarity-gap trim between ranking and capping) is
    pinned separately in ``test_adaptive_k.py``.
    """

    def _ex(self, key, *, phash=None, clip=None, label=None):
        from vlm_feedback_loop.services.icl_service import ICLExample

        return ICLExample(
            example_key=key,
            label_json=label or {"damage_type": "crack"},
            labeled_at="2026-01-01T00:00:00Z",
            phash=phash,
            clip_embedding=clip,
        )

    def test_relevance_orders_by_query_similarity(self):
        from vlm_feedback_loop.services.icl_service import select_icl_examples

        # One-hot embeddings so cosine(query, e_i) is proportional to query_i.
        cands = [
            self._ex("e0", clip=[1.0, 0.0, 0.0, 0.0]),
            self._ex("e1", clip=[0.0, 1.0, 0.0, 0.0]),
            self._ex("e2", clip=[0.0, 0.0, 1.0, 0.0]),
            self._ex("e3", clip=[0.0, 0.0, 0.0, 1.0]),
        ]
        # Query most similar to e2 (0.4), then e3 (0.3), then e1, e0.
        q = [0.1, 0.2, 0.4, 0.3]
        rel = select_icl_examples(
            cands,
            icl_max_examples=2,
            query_clip_embedding=q,
        )
        assert rel.selected_keys == ["e2", "e3"]

    def test_relevance_falls_back_without_query_embedding(self):
        from vlm_feedback_loop.services.icl_service import select_icl_examples

        cands = [self._ex(f"e{i}", clip=[float(i), 1.0]) for i in range(4)]
        # No query embedding -> incoming (newest-first) order, no crash.
        r = select_icl_examples(
            cands,
            icl_max_examples=2,
            query_clip_embedding=None,
        )
        assert r.selected_keys == ["e0", "e1"]

    def test_icl_max_examples_caps_total(self):
        from vlm_feedback_loop.services.icl_service import select_icl_examples

        cands = [self._ex(f"e{i}", phash=f"{i:016x}") for i in range(10)]
        result = select_icl_examples(cands, icl_max_examples=3)
        assert result.total_count == 3
        assert result.selected_keys == ["e0", "e1", "e2"]
        assert result.candidate_pool_size == 10

    def test_uncapped_keeps_all(self):
        from vlm_feedback_loop.services.icl_service import select_icl_examples

        cands = [self._ex(f"e{i}", phash=f"{i:016x}") for i in range(5)]
        result = select_icl_examples(cands, icl_max_examples=None)
        assert result.total_count == 5


class TestBookendOrdering:
    """Final prompt layout keeps the two strongest examples at the edges."""

    def _examples(self, count):
        from vlm_feedback_loop.services.icl_service import ICLExample

        return [
            ICLExample(
                example_key=f"e{i}",
                label_json={"category": str(i)},
                labeled_at=f"t{i}",
            )
            for i in range(count)
        ]

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, []),
            (1, ["e0"]),
            (2, ["e0", "e1"]),
            (5, ["e0", "e2", "e3", "e4", "e1"]),
        ],
    )
    def test_bookends_relevance_ranked_examples(self, count, expected):
        from vlm_feedback_loop.services.icl_service import bookend_icl_examples

        original = self._examples(count)
        ordered = bookend_icl_examples(original)
        assert [example.example_key for example in ordered] == expected
        assert [example.example_key for example in original] == [
            f"e{i}" for i in range(count)
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Section G: Budget pruning (relevance-tail dropping)
# ═══════════════════════════════════════════════════════════════════════════


class TestBudgetPruning:
    """Token/image budgets drop from the tail of the selection order.

    The selection order is relevance-descending (newest-first in the
    embedding-less degrade), so the tail is always the least query-relevant
    exemplar — the correct victim under Selective-K.
    """

    def _examples(self, n):
        from vlm_feedback_loop.services.icl_service import ICLExample

        return [
            ICLExample(
                f"e{i}",
                {"rationale_note": "x", "d": True},
                f"t{i}",
                phash=f"{i:016x}",
            )
            for i in range(1, n + 1)
        ]

    _FIELDS = [
        _make_field(RESERVED_FIELD_NAME, "string", role="aux"),
        _make_field("d", "boolean", role="core"),
    ]
    _GEN_ORDER = [RESERVED_FIELD_NAME, "d"]

    def test_token_pruner_drops_tail_until_fit(self):
        from vlm_feedback_loop.services.icl_service import prune_icl_by_budget
        from vlm_feedback_loop.services.token_budget_service import (
            estimate_icl_example_tokens,
        )

        examples = self._examples(3)
        one_example_cost = estimate_icl_example_tokens(
            examples[0].label_json, self._FIELDS, self._GEN_ORDER, "all"
        )

        # Budget fits exactly one example → the head survives, the tail
        # drops in tail-first order.
        retained, dropped = prune_icl_by_budget(
            examples,
            max_icl_tokens=one_example_cost,
            fields=self._FIELDS,
            generation_order=self._GEN_ORDER,
            icl_field_mode="all",
        )
        assert [ex.example_key for ex in retained] == ["e1"]
        assert dropped == ["e3", "e2"]

    def test_token_pruner_can_empty_the_list(self):
        from vlm_feedback_loop.services.icl_service import prune_icl_by_budget

        examples = self._examples(3)
        retained, dropped = prune_icl_by_budget(
            examples,
            max_icl_tokens=0,
            fields=self._FIELDS,
            generation_order=self._GEN_ORDER,
            icl_field_mode="all",
        )
        # Nothing fits — the caller's cold-start render path takes over.
        assert retained == []
        assert dropped == ["e3", "e2", "e1"]

    def test_stops_when_within_budget(self):
        from vlm_feedback_loop.services.icl_service import prune_icl_by_budget

        examples = self._examples(2)
        retained, dropped = prune_icl_by_budget(
            examples,
            max_icl_tokens=999999,
            fields=self._FIELDS,
            generation_order=self._GEN_ORDER,
            icl_field_mode="all",
        )
        assert len(retained) == 2
        assert dropped == []

    def test_image_budget_keeps_selection_head(self):
        from vlm_feedback_loop.services.icl_service import prune_icl_by_image_budget

        examples = self._examples(4)
        retained, dropped = prune_icl_by_image_budget(examples, 2)
        assert [ex.example_key for ex in retained] == ["e1", "e2"]
        assert dropped == ["e3", "e4"]

    def test_image_budget_zero_drops_everything(self):
        from vlm_feedback_loop.services.icl_service import prune_icl_by_image_budget

        examples = self._examples(3)
        retained, dropped = prune_icl_by_image_budget(examples, 0)
        assert retained == []
        assert dropped == ["e1", "e2", "e3"]


# ═══════════════════════════════════════════════════════════════════════════
# Section H: ICL field rendering — Inference Contract consumption
# ═══════════════════════════════════════════════════════════════════════════


class TestICLFieldRendering:
    """AC: core_only renders Core only; field ordering matches generation_order."""

    def _fields(self):
        return [
            _make_field(RESERVED_FIELD_NAME, "string", role="aux", display_order=-1),
            _make_field("fragile", "boolean", role="aux", display_order=0),
            _make_field(
                "damage_type",
                "enum",
                role="core",
                display_order=1,
                allowed_values=["crack", "dent"],
            ),
            _make_field("severity", "integer", role="core", display_order=2),
        ]

    def _gen_order(self):
        return [RESERVED_FIELD_NAME, "fragile", "damage_type", "severity"]

    def _label(self):
        return {
            RESERVED_FIELD_NAME: "visible crack",
            "fragile": True,
            "damage_type": "crack",
            "severity": 3,
        }

    def test_core_only_renders_core_fields(self):
        from vlm_feedback_loop.services.token_budget_service import (
            render_icl_fields,
        )

        result = render_icl_fields(
            self._label(),
            self._fields(),
            self._gen_order(),
            "core_only",
        )
        assert RESERVED_FIELD_NAME not in result
        assert "fragile" not in result
        assert "damage_type" in result
        assert "severity" in result

    def test_all_renders_everything(self):
        from vlm_feedback_loop.services.token_budget_service import (
            render_icl_fields,
        )

        result = render_icl_fields(
            self._label(),
            self._fields(),
            self._gen_order(),
            "all",
        )
        assert set(result.keys()) == {
            RESERVED_FIELD_NAME,
            "fragile",
            "damage_type",
            "severity",
        }

    def test_aux_and_core_excludes_rationale(self):
        from vlm_feedback_loop.services.token_budget_service import (
            render_icl_fields,
        )

        result = render_icl_fields(
            self._label(),
            self._fields(),
            self._gen_order(),
            "aux_and_core",
        )
        assert RESERVED_FIELD_NAME not in result
        assert "fragile" in result
        assert "damage_type" in result

    def test_field_ordering_matches_generation_order(self):
        from vlm_feedback_loop.services.token_budget_service import (
            render_icl_fields,
        )

        result = render_icl_fields(
            self._label(),
            self._fields(),
            self._gen_order(),
            "all",
        )
        assert list(result.keys()) == [
            RESERVED_FIELD_NAME,
            "fragile",
            "damage_type",
            "severity",
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Section I: Operational logging — log points 4 & 5
# ═══════════════════════════════════════════════════════════════════════════


class TestICLLogging:
    """AC: Log point 4 (ICL selection) emits with required fields."""

    def test_log_point_4_emits_fields(self, caplog):
        from vlm_feedback_loop.services.icl_service import (
            ICLSelectionResult,
            log_icl_selection,
        )

        result = ICLSelectionResult(
            candidate_pool_size=10,
            selected_keys=["e1", "e2", "e3"],
            total_count=3,
            pruned_count=0,
            pruned_keys=[],
        )

        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.icl"):
            log_icl_selection("test-proj", result)

        records = [r for r in caplog.records if "ICL selection" in r.message]
        assert len(records) >= 1
        rec = records[0]
        assert rec.levelno == logging.DEBUG
        details = getattr(rec, "details", None)
        assert details is not None
        assert "candidate_pool_size" in details
        assert "selected_keys" in details

    def test_log_point_4_escalates_on_pruning(self, caplog):
        from vlm_feedback_loop.services.icl_service import (
            ICLSelectionResult,
            log_icl_selection,
        )

        result = ICLSelectionResult(
            candidate_pool_size=10,
            selected_keys=["e1", "e2"],
            total_count=2,
            pruned_count=1,
            pruned_keys=["e3"],
        )

        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.icl"):
            log_icl_selection("test-proj", result)

        records = [r for r in caplog.records if "ICL selection" in r.message]
        assert any(r.levelno == logging.INFO for r in records)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference Contract derivation for Student models.

The contract surfaces in three places:

  1. ``student_model_service.register_from_tao_terminal`` snapshots it onto
     ``StudentModel.training_inference_contract`` at registration so the
     deployment-handoff gate has a stable comparison key.
  2. ``student_nim_lifecycle._student_inference_contract`` takes a
     ``StudentSnapshot`` + ``workspace_root`` and delegates to this helper.
  3. ``deployment_handoff_generator`` can recompute an unexpectedly absent
     snapshot from the authoritative DatasetExport lineage.

This helper is the single canonical implementation of the derivation
rule: "Student's ``output_field_mode`` and ``icl_field_mode`` both default to
the training DatasetExport's ``export_field_mode``" with a defensive fallback
to the first export when no ``training``-intent row exists. Always returns a
dict shape — callers may freely call ``InferenceContract.model_validate`` on
the result for type checking.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from vlm_feedback_loop.db.models.dataset_export import DatasetExport


def resolve_training_inference_contract(
    session: Session,
    dataset_export_ids: list[str],
) -> dict[str, Any]:
    """Return the Inference Contract dict derived from training DatasetExports.

    Lookup priority:
      1. The first DatasetExport row in ``dataset_export_ids`` with
         ``dataset_intent="training"``.
      2. Failing that, the first row in the list (defensive fallback).
      3. Failing that (empty list / all rows missing), the Teacher contract
         shape with ``"all"`` field modes.

    The output dict mirrors :class:`schemas.inference_contract.InferenceContract`
    field names and includes ``icl_max_examples`` set to ``None`` (no
    per-Student override; ICL sizing remains driven by project settings).
    """
    field_mode = "all"
    if dataset_export_ids:
        for export_id in dataset_export_ids:
            row = session.get(DatasetExport, export_id)
            if row is None:
                continue
            if row.dataset_intent == "training":
                field_mode = row.export_field_mode
                break
        else:
            first = session.get(DatasetExport, dataset_export_ids[0])
            if first is not None:
                field_mode = first.export_field_mode

    return {
        "output_field_mode": field_mode,
        "icl_field_mode": field_mode,
        "icl_max_examples": None,
    }

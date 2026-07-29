# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive proposal router.

Generates a Teacher proposal for a given example by orchestrating the
full pipeline: ICL selection → prompt rendering → Teacher invocation →
schema validation → Operation Record persistence.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.proposal import ProposalRequest, ProposalResponse
from vlm_feedback_loop.services.errors import map_service_error
from vlm_feedback_loop.services.priority import priority_dispatch
from vlm_feedback_loop.services.proposal_service import create_proposal

proposals_router = APIRouter(
    prefix="/projects/{project_id}/proposals",
    tags=["proposals"],
)


@proposals_router.post("", response_model=ProposalResponse)
async def create_proposal_endpoint(
    project_id: str,
    body: ProposalRequest,
    settings: Settings = Depends(get_current_settings),
) -> ProposalResponse:
    """Request a Teacher proposal for a specific example.

    Used for initial proposals and Retry.  For Auto-Labeled examples,
    set ``use_existing_label=True`` to return the stored label without
    a fresh Teacher call.
    """
    # Foreground-priority dispatch: hold new background HTTP dispatches
    # (evaluation / batch labeling) while this interactive proposal — or Retry —
    # is in flight, so the SME's request isn't slowed by competing NIM calls.
    async with priority_dispatch.foreground():
        result = await create_proposal(
            project_id,
            example_key=body.example_key,
            teacher_model_config_id_override=body.teacher_model_config_id_override,
            guidance_id_override=body.guidance_id_override,
            generation_preset_key_override=body.generation_preset_key_override,
            thinking_mode_override=body.thinking_mode_override,
            visual_budget_preset_key_override=body.visual_budget_preset_key_override,
            retry_of_inference_invocation_id=body.retry_of_inference_invocation_id,
            use_existing_label=body.use_existing_label,
            settings=settings,
        )

    if isinstance(result, str):
        raise map_service_error(result)

    return ProposalResponse(
        inference_invocation_id=result.inference_invocation_id,
        example_key=result.example_key,
        proposal_json=result.proposal_json,
        schema_valid_core=result.schema_valid_core,
        validation_errors_core=result.validation_errors_core,
        validation_errors_aux=result.validation_errors_aux,
        invocation_status=result.invocation_status,
        latency_ms_end_to_end=result.latency_ms_end_to_end,
        icl_images_attached_count=result.icl_images_attached_count,
        icl_example_keys_used=result.icl_example_keys_used,
        used_existing_label=result.used_existing_label,
    )

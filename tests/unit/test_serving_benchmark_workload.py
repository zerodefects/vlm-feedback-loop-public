# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import subprocess
from unittest.mock import MagicMock

import pytest
from PIL import Image
from sqlalchemy.orm import Session

from conftest import make_settings
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services import project_service
from vlm_feedback_loop.services.serving_benchmark_workload import (
    MAX_WORKLOAD_IMAGES,
    _resolve_code_revision,
    _select_keys,
    build_serving_benchmark_workload,
)
from vlm_feedback_loop.services.serving_request_service import (
    build_uncapped_student_request,
)


def test_stable_selection_is_without_replacement_and_capped():
    keys = [f"example-{index:03d}" for index in range(250)]
    first = _select_keys(keys, "basis")
    second = _select_keys(list(reversed(keys)), "basis")
    assert first == second
    assert len(first) == MAX_WORKLOAD_IMAGES
    assert len(set(first)) == MAX_WORKLOAD_IMAGES
    assert _select_keys(keys[:17], "basis") == _select_keys(keys[:17], "basis")


def test_code_revision_prefers_packaged_build_override(monkeypatch):
    monkeypatch.setenv("VLM_FEEDBACK_LOOP_BUILD_SHA", "release-sha")
    run = MagicMock(side_effect=AssertionError("Git fallback must not run"))
    monkeypatch.setattr(subprocess, "run", run)

    assert _resolve_code_revision() == "release-sha"
    run.assert_not_called()


def test_code_revision_falls_back_to_source_checkout(monkeypatch):
    revision = "a" * 40
    monkeypatch.delenv("VLM_FEEDBACK_LOOP_BUILD_SHA", raising=False)
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(return_value=subprocess.CompletedProcess([], 0, revision + "\n", "")),
    )

    assert _resolve_code_revision() == revision


def test_workload_uses_real_images_guidance_contract_and_no_output_cap(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLM_FEEDBACK_LOOP_BUILD_SHA", "benchmark-build-sha")
    settings = make_settings(tmp_path)
    project = project_service.create_project("real-serving-workload", None, settings)
    engine = project_service.get_project_engine(
        project.project_id, settings.WORKSPACE_ROOT
    )
    assert engine is not None

    image_paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (16, 16), color=color).save(path)
        image_paths.append(path)

    guidance_id = "guidance-benchmark"
    pool_id = "pool-benchmark"
    run_id = "run-benchmark"
    fields = [
        {
            "field_name": "rationale_note",
            "role": "aux",
            "type": "string",
            "display_order": 0,
        },
        {
            "field_name": "gesture",
            "role": "core",
            "type": "enum",
            "allowed_values": ["rock", "paper"],
            "display_order": 1,
        },
    ]
    schema = {
        "fields": fields,
        "generation_order": ["rationale_note", "gesture"],
        "derived_json_schema": {
            "type": "object",
            "properties": {
                "rationale_note": {"type": "string"},
                "gesture": {"type": "string", "enum": ["rock", "paper"]},
            },
            "required": ["gesture"],
            "additionalProperties": False,
            "x-generation-order": ["rationale_note", "gesture"],
        },
        "schema_hash": "schema-hash",
    }
    inference_contract = {
        "output_field_mode": "core_only",
        "icl_field_mode": "core_only",
        "icl_max_examples": None,
    }
    sampling_params = {"temperature": 0.0, "top_p": 1.0}
    thinking_fields = {"enable_thinking": False}
    visual_params = {"min_pixels": 1024}
    _, evaluated_prompt_hash = build_uncapped_student_request(
        served_model="student-served",
        guidance_description="Classify the hand gesture from the image.",
        guidance_rules="Use only the permitted gesture values.",
        guidance_fields=fields,
        generation_order=["rationale_note", "gesture"],
        derived_json_schema=schema["derived_json_schema"],
        inference_contract=inference_contract,
        structured_generation_attempted=True,
        sampling_params=sampling_params,
        thinking_request_fields=thinking_fields,
        visual_budget_params=visual_params,
        image_content_part={
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,unused"},
        },
    )
    with Session(engine) as session:
        session.add(
            Guidance(
                guidance_id=guidance_id,
                project_id=project.project_id,
                version_number=1,
                description="Classify the hand gesture from the image.",
                rules="Use only the permitted gesture values.",
                schema=schema,
            )
        )
        keys = ["key-red", "key-green"]
        session.add(
            Pool(
                pool_id=pool_id,
                project_id=project.project_id,
                pool_type="test_pool",
                pool_version=1,
                member_example_keys=keys,
                member_count=2,
                guidance_id=guidance_id,
            )
        )
        for key, path in zip(keys, image_paths, strict=True):
            session.add(
                Example(
                    example_key=key,
                    project_id=project.project_id,
                    storage_ref=str(path),
                    ingested_at="2026-08-04T00:00:00Z",
                    source_metadata={},
                )
            )
        session.add(
            RunRecord(
                run_id=run_id,
                project_id=project.project_id,
                run_type="evaluation_run",
                status="completed",
                pool_version_id=pool_id,
                guidance_id=guidance_id,
                model_config_id="student-mc",
                inference_contract=inference_contract,
                created_at="2026-08-04T00:00:00Z",
            )
        )
        session.add(
            OperationRecord(
                inference_invocation_id="invocation-benchmark",
                project_id=project.project_id,
                purpose="evaluation",
                example_key=keys[0],
                evaluation_run_id=run_id,
                invocation_status="success",
                sampling_params_effective=sampling_params,
                thinking_request_fields_effective=thinking_fields,
                visual_budget_params_effective=visual_params,
                structured_generation_attempted=True,
                prompt_hash=evaluated_prompt_hash,
            )
        )
        session.commit()

    workload = asyncio.run(
        build_serving_benchmark_workload(
            project_id=project.project_id,
            serving_run_id=run_id,
            student_model_id="student-1",
            served_model="student-served",
            workspace_root=settings.WORKSPACE_ROOT,
            settings=settings,
        )
    )
    payloads = [
        json.loads(line) for line in workload.input_file.read_text().splitlines()
    ]
    assert len(payloads) == 2
    assert workload.manifest["selected_count"] == 2
    assert workload.manifest["output_limit_mode"] == "uncapped"
    assert workload.manifest["kv_cache_reuse"] == "disabled"
    assert workload.manifest["prompt_hash"] == evaluated_prompt_hash
    assert workload.manifest["evaluated_prompt_hash"] == evaluated_prompt_hash
    assert workload.manifest["code_revision"] == "benchmark-build-sha"
    for payload in payloads:
        assert "max_tokens" not in payload
        assert "max_completion_tokens" not in payload
        assert payload["model"] == "student-served"
        rendered_text = json.dumps(payload["messages"])
        # The production prompt deliberately uses Guidance's compact schema
        # contract rather than its verbose description/rules prose.
        assert "gesture" in rendered_text
        assert "rock" in rendered_text
        image_parts = [
            part
            for part in payload["messages"][-1]["content"]
            if part.get("type") == "image_url"
        ]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/")
        response_schema = payload["response_format"]["json_schema"]["schema"]
        assert list(response_schema["properties"]) == ["gesture"]

    with Session(engine) as session:
        operation = session.get(OperationRecord, "invocation-benchmark")
        assert operation is not None
        operation.prompt_hash = "different-evaluated-prompt"
        session.commit()
    with pytest.raises(RuntimeError, match="prompt drifted"):
        asyncio.run(
            build_serving_benchmark_workload(
                project_id=project.project_id,
                serving_run_id=run_id,
                student_model_id="student-1",
                served_model="student-served",
                workspace_root=settings.WORKSPACE_ROOT,
                settings=settings,
            )
        )

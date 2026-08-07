# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelConfig CRUD, capability probes, and re-probe.

Covers: list with role filtering, create validation, PATCH, reprobe with
409 on active use, three capability probe functions, probe image generation,
and log point 6.
"""

from __future__ import annotations

import base64
import logging
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from conftest import create_project_via_api
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS3_SUPER_REASONER,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
    MISTRAL_MEDIUM_3_5,
    NEMOTRON_3_NANO_OMNI_REASONING,
    NEMOTRON_NANO_12B_VL,
)
from vlm_feedback_loop.services.http_client import HttpResult
from vlm_feedback_loop.services.model_config_service import (
    generate_probe_image_data_url,
    probe_structured_generation,
    probe_thinking_toggle,
    probe_visual_budget,
)

_RETIRED_MISTRAL_LARGE_3 = "mistralai/mistral-large-3-675b-instruct-2512"

# ── Helpers ─────────────────────────────────────────────────────────────────


def _create_project(client: TestClient) -> str:
    return create_project_via_api(client, name="Test Project")["project_id"]


def _get_seeded_endpoint_id(client: TestClient, project_id: str) -> str:
    """Get the seeded hosted NimEndpoint ID for a project."""
    resp = client.get(f"/v1/projects/{project_id}/nim_endpoints")
    return resp.json()["items"][0]["endpoint_id"]


# ── Probe image ─────────────────────────────────────────────────────────────


class TestProbeImage:
    def test_valid_data_url(self):
        url = generate_probe_image_data_url()
        assert url.startswith("data:image/png;base64,")

    def test_deterministic(self):
        a = generate_probe_image_data_url()
        b = generate_probe_image_data_url()
        assert a == b

    def test_decodable_png(self):
        url = generate_probe_image_data_url()
        b64_part = url.split(",", 1)[1]
        png_bytes = base64.b64decode(b64_part)
        # PNG signature: \x89PNG\r\n\x1a\n
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        # Has IHDR chunk type
        assert b"IHDR" in png_bytes[:30]


# ── Capability probe functions (unit tests) ─────────────────────────────────


class TestStructuredGenerationProbe:
    @pytest.mark.asyncio
    async def test_supported_on_valid_json(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={
                "choices": [
                    {"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}
                ]
            },
            error_class=None,
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=mock_result),
        )
        result = await probe_structured_generation(
            "http://host:8000/v1", {}, "test-model", 10.0
        )
        assert result == "supported"

    @pytest.mark.asyncio
    async def test_qwen_payload_disables_thinking(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={
                "choices": [
                    {"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}
                ]
            },
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            mock_fn,
        )

        result = await probe_structured_generation(
            "http://host:8000/v1",
            {},
            "test-model",
            10.0,
            thinking_toggle_mode="qwen_enable_thinking",
        )

        assert result == "supported"
        sent_body = mock_fn.call_args.kwargs["json_body"]
        assert sent_body["chat_template_kwargs"] == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_always_on_reasoner_gets_probe_headroom(self, monkeypatch):
        """Structured probing must leave room for an unavoidable trace."""
        mock_result = HttpResult(
            status_code=200,
            body={
                "choices": [
                    {"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}
                ]
            },
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            mock_fn,
        )

        result = await probe_structured_generation(
            "http://host:8000/v1",
            {},
            "always-on-model",
            180.0,
            thinking_toggle_mode="always_on_reasoning",
        )

        assert result == "supported"
        assert mock_fn.call_args.kwargs["json_body"]["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_truncated_reasoning_is_unknown_not_unsupported(self, monkeypatch):
        """A length-limited trace does not prove response_format rejection."""
        mock_result = HttpResult(
            status_code=200,
            body={
                "choices": [
                    {
                        "message": {"content": "unfinished reasoning"},
                        "finish_reason": "length",
                    }
                ]
            },
            error_class=None,
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=mock_result),
        )

        result = await probe_structured_generation(
            "http://host:8000/v1",
            {},
            "always-on-model",
            180.0,
            thinking_toggle_mode="always_on_reasoning",
        )

        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_unsupported_on_4xx(self, monkeypatch):
        mock_result = HttpResult(
            status_code=400,
            body={"error": "response_format not supported"},
            error_class="endpoint_error",
            error_detail="HTTP 400",
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=mock_result),
        )
        result = await probe_structured_generation(
            "http://host:8000/v1", {}, "test-model", 10.0
        )
        assert result == "unsupported"

    @pytest.mark.asyncio
    async def test_unknown_on_timeout(self, monkeypatch):
        mock_result = HttpResult(
            error_class="timeout",
            error_detail="Request timed out",
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=mock_result),
        )
        result = await probe_structured_generation(
            "http://host:8000/v1", {}, "test-model", 10.0
        )
        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_unsupported_when_json_invalid(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={
                "choices": [
                    {"message": {"content": "not json"}, "finish_reason": "stop"}
                ]
            },
            error_class=None,
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=mock_result),
        )
        result = await probe_structured_generation(
            "http://host:8000/v1", {}, "test-model", 10.0
        )
        assert result == "unsupported"


class TestThinkingToggleProbe:
    @pytest.mark.asyncio
    async def test_none_mode_returns_unsupported(self):
        result = await probe_thinking_toggle(
            "http://host:8000/v1", {}, "test-model", "none", 10.0
        )
        assert result == "unsupported"

    @pytest.mark.asyncio
    async def test_qwen_supported(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            mock_fn,
        )
        result = await probe_thinking_toggle(
            "http://host:8000/v1", {}, "test-model", "qwen_enable_thinking", 10.0
        )
        assert result == "supported"
        # Verify chat_template_kwargs was passed
        sent_body = mock_fn.call_args.kwargs["json_body"]
        assert sent_body["chat_template_kwargs"] == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_kimi_unsupported_4xx(self, monkeypatch):
        mock_result = HttpResult(
            status_code=422,
            body={"error": "chat_template_kwargs not supported"},
            error_class="endpoint_error",
            error_detail="HTTP 422",
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=mock_result),
        )
        result = await probe_thinking_toggle(
            "http://host:8000/v1", {}, "test-model", "kimi_thinking", 10.0
        )
        assert result == "unsupported"


class TestVisualBudgetProbe:
    @pytest.mark.asyncio
    async def test_none_mode_returns_unsupported(self):
        result = await probe_visual_budget(
            "http://host:8000/v1", {}, "test-model", "none", 10.0
        )
        assert result == "unsupported"

    @pytest.mark.asyncio
    async def test_supported_both_stages_ok(self, monkeypatch):
        success_result = HttpResult(
            status_code=200,
            body={"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=success_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            mock_fn,
        )
        result = await probe_visual_budget(
            "http://host:8000/v1", {}, "test-model", "mm_processor_size", 10.0
        )
        assert result == "supported"
        # Two calls: baseline then capability
        assert mock_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_baseline_fail_returns_unknown(self, monkeypatch):
        fail_result = HttpResult(
            error_class="timeout",
            error_detail="Request timed out",
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=fail_result),
        )
        result = await probe_visual_budget(
            "http://host:8000/v1", {}, "test-model", "mm_processor_size", 10.0
        )
        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_capability_fail_returns_unsupported(self, monkeypatch):
        call_count = 0

        async def _mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Stage 1 baseline: success
                return HttpResult(
                    status_code=200,
                    body={
                        "choices": [
                            {"message": {"content": "OK"}, "finish_reason": "stop"}
                        ]
                    },
                    error_class=None,
                    attempts=1,
                )
            else:
                # Stage 2 capability: fail
                return HttpResult(
                    status_code=500,
                    body={"error": "mm_processor_kwargs rejected"},
                    error_class="endpoint_error",
                    error_detail="HTTP 500",
                    attempts=1,
                )

        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            _mock_request,
        )
        result = await probe_visual_budget(
            "http://host:8000/v1", {}, "test-model", "mm_processor_tiles", 10.0
        )
        assert result == "unsupported"

    @pytest.mark.asyncio
    async def test_probe_includes_image_in_messages(self, monkeypatch):
        success_result = HttpResult(
            status_code=200,
            body={"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=success_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            mock_fn,
        )
        await probe_visual_budget(
            "http://host:8000/v1", {}, "test-model", "mm_processor_size", 10.0
        )
        # Check first call (baseline) has image in messages
        first_call_body = mock_fn.call_args_list[0].kwargs["json_body"]
        content = first_call_body["messages"][0]["content"]
        assert any(p.get("type") == "image_url" for p in content)


class TestImageCapSupportProbe:
    """Image-cap probe — verifies seeded ``max_images_per_request`` is correct.

    Pins a real regression class: hosted Qwen 3.5 rejects requests with
    >8 images even though the catalog claimed 10.
    Without a runtime probe, every project on a wrongly-seeded model
    silently breaks once it hits the cap.
    """

    @pytest.mark.asyncio
    async def test_supported_when_at_cap_passes_above_cap_fails(self, monkeypatch):
        """N=cap → 200, N=cap+1 → 400. The seeded cap is exactly right."""
        from vlm_feedback_loop.services.model_config_service import (
            _probe_image_cap_support,
        )

        call_count = 0

        async def _mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            content = kwargs["json_body"]["messages"][0]["content"]
            n_images = sum(1 for p in content if p.get("type") == "image_url")
            if n_images <= 8:
                return HttpResult(
                    status_code=200,
                    body={
                        "choices": [
                            {"message": {"content": "ok"}, "finish_reason": "stop"}
                        ]
                    },
                    error_class=None,
                    attempts=1,
                )
            return HttpResult(
                status_code=400,
                body={"error": "At most 8 image(s) may be provided"},
                error_class="endpoint_error",
                error_detail="HTTP 400",
                attempts=1,
            )

        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            _mock_request,
        )
        result = await _probe_image_cap_support(
            "http://host:8000/v1", {}, NEMOTRON_NANO_12B_VL, 8, 10.0
        )
        assert result == "supported"
        assert call_count == 2  # cap then cap+1

    @pytest.mark.asyncio
    async def test_unsupported_when_at_cap_fails(self, monkeypatch):
        """N=cap → 400. The seeded cap is too high — loop will break."""
        from vlm_feedback_loop.services.model_config_service import (
            _probe_image_cap_support,
        )

        fail_result = HttpResult(
            status_code=400,
            body={"error": "At most 8 image(s) may be provided"},
            error_class="endpoint_error",
            error_detail="HTTP 400",
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=fail_result),
        )
        # Seeded value is 10 but real cap is 8 — probe MUST flag.
        result = await _probe_image_cap_support(
            "http://host:8000/v1", {}, NEMOTRON_NANO_12B_VL, 10, 10.0
        )
        assert result == "unsupported"

    @pytest.mark.asyncio
    async def test_supported_with_hint_when_above_cap_also_passes(self, monkeypatch):
        """N=cap → 200, N=cap+1 → 200. Seeded value conservative, real cap is bigger.

        Returns "supported" (the seeded value works) but logs a hint that
        ``scripts/probe_hosted_image_caps.py`` should be re-run to find
        the real cap.
        """
        from vlm_feedback_loop.services.model_config_service import (
            _probe_image_cap_support,
        )

        success_result = HttpResult(
            status_code=200,
            body={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            error_class=None,
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=success_result),
        )
        result = await _probe_image_cap_support(
            "http://host:8000/v1",
            {},
            NEMOTRON_3_NANO_OMNI_REASONING,
            5,
            10.0,
        )
        assert result == "supported"

    @pytest.mark.asyncio
    async def test_unknown_on_timeout(self, monkeypatch):
        from vlm_feedback_loop.services.model_config_service import (
            _probe_image_cap_support,
        )

        timeout_result = HttpResult(
            error_class="timeout",
            error_detail="Request timed out",
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.model_config_service.nim_client.resilient_request",
            AsyncMock(return_value=timeout_result),
        )
        result = await _probe_image_cap_support(
            "http://host:8000/v1", {}, "test-model", 8, 10.0
        )
        assert result == "unknown"

    @pytest.mark.asyncio
    async def test_unknown_on_zero_cap(self):
        """Defensive — cap < 1 is meaningless."""
        from vlm_feedback_loop.services.model_config_service import (
            _probe_image_cap_support,
        )

        result = await _probe_image_cap_support(
            "http://host:8000/v1", {}, "test-model", 0, 10.0
        )
        assert result == "unknown"


# ── ModelConfig CRUD endpoint tests ─────────────────────────────────────────


class TestModelConfigCRUD:
    """CRUD endpoint tests. Autouse mock prevents real network calls."""

    @pytest.fixture(autouse=True)
    def _mock_probes(self):
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={"data": [{"id": "model-a"}]},
                error_class=None,
                attempts=1,
            )
            yield mock_req

    def test_list_seeded_entries(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.get(f"/v1/projects/{project_id}/model_configs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 8
        assert _RETIRED_MISTRAL_LARGE_3 not in {
            item["model_name"] for item in data["items"]
        }

    def test_list_filter_teacher(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.get(
            f"/v1/projects/{project_id}/model_configs?eligible_role=teacher"
        )
        items = resp.json()["items"]
        for item in items:
            assert "teacher" in item["eligible_roles"]
            assert item["supports_image_input"] is True

    def test_list_filter_student_base(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.get(
            f"/v1/projects/{project_id}/model_configs?eligible_role=student_base"
        )
        items = resp.json()["items"]
        assert len(items) == 4
        names = {i["model_name"] for i in items}
        assert names == {
            COSMOS_REASON2_8B,
            COSMOS_REASON2_2B,
            COSMOS3_NANO_REASONER,
            COSMOS3_SUPER_REASONER,
        }

    def test_create_model_config(self, test_app_client):
        project_id = _create_project(test_app_client)
        endpoint_id = _get_seeded_endpoint_id(test_app_client, project_id)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs",
            json={
                "endpoint_id": endpoint_id,
                "model_name": "custom-model-1",
                "context_window_tokens": 4096,
                "eligible_roles": ["teacher"],
                "supports_image_input": True,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["model_name"] == "custom-model-1"
        assert data["model_config_id"]  # UUID4
        assert data["project_id"] == project_id
        assert data["structured_generation_support"] == "unknown"
        assert data["thinking_toggle_support"] == "unknown"
        assert data["visual_budget_support"] == "unknown"
        assert data["max_images_per_request"] == 5

    def test_create_model_config_persists_explicit_image_cap(self, test_app_client):
        project_id = _create_project(test_app_client)
        endpoint_id = _get_seeded_endpoint_id(test_app_client, project_id)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs",
            json={
                "endpoint_id": endpoint_id,
                "model_name": "custom-model-with-cap",
                "context_window_tokens": 4096,
                "eligible_roles": ["teacher"],
                "supports_image_input": True,
                "max_images_per_request": 17,
            },
        )
        assert resp.status_code == 201
        assert resp.json()["max_images_per_request"] == 17

    def test_create_rejects_empty_roles(self, test_app_client):
        project_id = _create_project(test_app_client)
        endpoint_id = _get_seeded_endpoint_id(test_app_client, project_id)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs",
            json={
                "endpoint_id": endpoint_id,
                "model_name": "bad-model",
                "context_window_tokens": 4096,
                "eligible_roles": [],
                "supports_image_input": True,
            },
        )
        assert resp.status_code == 400
        assert "eligible_roles must be non-empty" in resp.json()["detail"]

    def test_create_rejects_student_base_on_non_cosmos(self, test_app_client):
        project_id = _create_project(test_app_client)
        endpoint_id = _get_seeded_endpoint_id(test_app_client, project_id)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs",
            json={
                "endpoint_id": endpoint_id,
                "model_name": "llama-70b",
                "context_window_tokens": 8192,
                "eligible_roles": ["student_base"],
                "supports_image_input": True,
            },
        )
        assert resp.status_code == 400
        assert "Cosmos Reason2" in resp.json()["detail"]

    def test_create_accepts_student_base_on_cosmos3(self, test_app_client):
        # Cosmos 3 (2026-06) ships as unified ``nvidia/Cosmos3-Nano`` /
        # ``-Super`` checkpoints (catalog-namespaced ``nvidia/cosmos3-*``).
        # The student_base allowlist admits that family alongside Cosmos
        # Reason2 so a CR3 fine-tune can be configured through real
        # Blueprint code. (Whether cosmos-rl trains it is gated TAO-side;
        # this test only asserts the app-side role guard accepts it.)
        project_id = _create_project(test_app_client)
        endpoint_id = _get_seeded_endpoint_id(test_app_client, project_id)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs",
            json={
                "endpoint_id": endpoint_id,
                "model_name": "nvidia/cosmos3-nano",
                "context_window_tokens": 256000,
                "eligible_roles": ["student_base"],
                "supports_image_input": True,
            },
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["model_name"] == "nvidia/cosmos3-nano"
        assert "student_base" in data["eligible_roles"]

    def test_create_validates_endpoint_belongs_to_project(self, test_app_client):
        project_id = _create_project(test_app_client)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs",
            json={
                "endpoint_id": "nonexistent-endpoint-id",
                "model_name": "model-x",
                "context_window_tokens": 4096,
                "eligible_roles": ["teacher"],
                "supports_image_input": True,
            },
        )
        assert resp.status_code == 400
        assert "endpoint_id" in resp.json()["detail"]

    def test_get_by_id(self, test_app_client):
        project_id = _create_project(test_app_client)
        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]
        mc_id = items[0]["model_config_id"]

        resp = test_app_client.get(f"/v1/projects/{project_id}/model_configs/{mc_id}")
        assert resp.status_code == 200
        assert resp.json()["model_config_id"] == mc_id

    def test_get_nonexistent_404(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.get(
            f"/v1/projects/{project_id}/model_configs/nonexistent"
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Model config not found"

    def test_patch_updates_fields(self, test_app_client):
        project_id = _create_project(test_app_client)
        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]
        mc_id = items[0]["model_config_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{project_id}/model_configs/{mc_id}",
            json={"context_window_tokens": 999999},
        )
        assert resp.status_code == 200
        assert resp.json()["context_window_tokens"] == 999999

    def test_patch_endpoint_id_rebind(self, test_app_client):
        project_id = _create_project(test_app_client)

        # Create a second endpoint in the same project
        ep_resp = test_app_client.post(
            f"/v1/projects/{project_id}/nim_endpoints",
            json={
                "display_name": "Self-Hosted",
                "endpoint_mode": "self_hosted",
                "base_url": "http://my-nim:8000/v1",
            },
        )
        new_ep_id = ep_resp.json()["endpoint_id"]

        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]
        mc_id = items[0]["model_config_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{project_id}/model_configs/{mc_id}",
            json={"endpoint_id": new_ep_id},
        )
        assert resp.status_code == 200
        assert resp.json()["endpoint_id"] == new_ep_id

    def test_patch_nonexistent_404(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.patch(
            f"/v1/projects/{project_id}/model_configs/nonexistent",
            json={"context_window_tokens": 100},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Model config not found"

    def test_model_name_immutable(self, test_app_client):
        """model_name is not in ModelConfigUpdate schema → extra=forbid rejects it."""
        project_id = _create_project(test_app_client)
        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]
        mc_id = items[0]["model_config_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{project_id}/model_configs/{mc_id}",
            json={"model_name": "new-name"},
        )
        assert resp.status_code == 422
        assert "model_name" in resp.text

    def test_capability_fields_preseeded_never_unknown(self, test_app_client):
        """Seeded models ship with resolved capability support, so
        every support field on every seeded row is `"supported"` or
        `"unsupported"` — never `"unknown"`. Values come from live probes,
        vendor docs, or mode=none ⇒ unsupported (see SEEDED_MODEL_CATALOG
        comments in project_service.py)."""
        project_id = _create_project(test_app_client)
        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]
        for item in items:
            assert item["structured_generation_support"] in (
                "supported",
                "unsupported",
            ), f"{item['model_name']} sg={item['structured_generation_support']}"
            assert item["thinking_toggle_support"] in (
                "supported",
                "unsupported",
            ), f"{item['model_name']} tt={item['thinking_toggle_support']}"
            assert item["visual_budget_support"] in (
                "supported",
                "unsupported",
            ), f"{item['model_name']} vb={item['visual_budget_support']}"

    def test_seeded_icl_depth_defaults_match_measured_values(self, test_app_client):
        """Every seeded Teacher carries the ICL depth default the July 2026
        cross-model depth studies established for it (Nemotron Nano VL 2 ·
        Omni 4 · CR3 8 · CR2-2B 8 · CR2-8B 16 · Mistral 2)."""
        project_id = _create_project(test_app_client)
        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]
        by_name = {i["model_name"]: i["default_icl_max_examples"] for i in items}
        assert by_name[NEMOTRON_NANO_12B_VL] == 2
        assert by_name[NEMOTRON_3_NANO_OMNI_REASONING] == 4
        assert by_name[COSMOS3_NANO_REASONER] == 8
        assert by_name[COSMOS3_SUPER_REASONER] == 8
        assert by_name[COSMOS_REASON2_2B] == 8
        assert by_name[COSMOS_REASON2_8B] == 16
        assert by_name[MISTRAL_MEDIUM_3_5] == 2
        assert "minimaxai/minimax-m3" not in by_name

    def test_patch_default_icl_max_examples(self, test_app_client):
        """Operators can re-tune a model's ICL depth default per project
        via PATCH (§10.2.19)."""
        project_id = _create_project(test_app_client)
        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]
        mc_id = items[0]["model_config_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{project_id}/model_configs/{mc_id}",
            json={"default_icl_max_examples": 4},
        )
        assert resp.status_code == 200
        assert resp.json()["default_icl_max_examples"] == 4

    def test_local_deploy_metadata_present_on_cosmos(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.get(
            f"/v1/projects/{project_id}/model_configs?eligible_role=student_base"
        )
        for item in resp.json()["items"]:
            assert item["local_deploy_metadata"] is not None
            assert "nim_container_image" in item["local_deploy_metadata"]
            assert "nim_gpu_memory_minimum_gb" in item["local_deploy_metadata"]

    def test_local_deploy_metadata_absent_on_hosted_only_models(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.get(f"/v1/projects/{project_id}/model_configs")
        omni = None
        for item in resp.json()["items"]:
            if item["model_name"] == NEMOTRON_3_NANO_OMNI_REASONING:
                omni = item
            elif "student_base" not in item["eligible_roles"]:
                assert item["local_deploy_metadata"] is None
        assert omni is not None
        assert omni["local_deploy_metadata"]["nim_compute_capability_minimum"] == 9.0

    def test_cursor_pagination(self, test_app_client):
        project_id = _create_project(test_app_client)
        # 8 seeded entries — paginate with limit=3 across 3 pages.
        items: list[dict] = []
        cursor: str | None = None
        for _ in range(5):  # safety bound > ceil(8/3)
            url = f"/v1/projects/{project_id}/model_configs?limit=3"
            if cursor:
                url += f"&cursor={cursor}"
            data = test_app_client.get(url).json()
            page = data["items"]
            assert len(page) <= 3
            items.extend(page)
            cursor = data["next_cursor"]
            if cursor is None:
                break
        assert cursor is None  # fully drained
        # All unique
        all_ids = {i["model_config_id"] for i in items}
        assert len(all_ids) == 8


# ── Re-probe endpoint tests ────────────────────────────────────────────────


class TestReprobe:
    @pytest.fixture(autouse=True)
    def _mock_probes(self):
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={
                    "choices": [
                        {
                            "message": {"content": '{"ok": true}'},
                            "finish_reason": "stop",
                        }
                    ]
                },
                error_class=None,
                attempts=1,
            )
            yield mock_req

    def test_reprobe_returns_updated_capabilities(self, test_app_client):
        project_id = _create_project(test_app_client)
        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]

        # Find nvidia/cosmos-reason2-8b (has all three modes active)
        cosmos_8b = next(i for i in items if i["model_name"] == COSMOS_REASON2_8B)
        mc_id = cosmos_8b["model_config_id"]

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs/{mc_id}:reprobe"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["structured_generation_support"] == "supported"
        assert data["thinking_toggle_support"] == "supported"
        assert data["visual_budget_support"] == "supported"

    def test_reprobe_409_active_run(self, test_app_client):
        project_id = _create_project(test_app_client)
        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]
        mc_id = items[0]["model_config_id"]

        # Insert an active RunRecord referencing this model_config_id
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.base import generate_uuid4
        from vlm_feedback_loop.db.models.run import RunRecord
        from vlm_feedback_loop.routers.projects import get_current_settings
        from vlm_feedback_loop.services.project_service import get_project_engine

        settings = test_app_client.app.dependency_overrides[get_current_settings]()
        engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)

        with Session(engine) as session:
            run = RunRecord(
                run_id=generate_uuid4(),
                project_id=project_id,
                run_type="evaluation_run",
                status="running",
                model_config_id=mc_id,
            )
            session.add(run)
            session.commit()

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs/{mc_id}:reprobe"
        )
        assert resp.status_code == 409
        assert "Cannot re-probe" in resp.json()["detail"]

    def test_reprobe_409_active_tao_job(self, test_app_client):
        project_id = _create_project(test_app_client)
        # Get a cosmos model (has student_base role)
        items = test_app_client.get(
            f"/v1/projects/{project_id}/model_configs?eligible_role=student_base"
        ).json()["items"]
        mc_id = items[0]["model_config_id"]

        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.base import generate_uuid4
        from vlm_feedback_loop.db.models.tao_job import TAOJob
        from vlm_feedback_loop.routers.projects import get_current_settings
        from vlm_feedback_loop.services.project_service import get_project_engine

        settings = test_app_client.app.dependency_overrides[get_current_settings]()
        engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)

        with Session(engine) as session:
            job = TAOJob(
                tao_job_id=generate_uuid4(),
                project_id=project_id,
                student_base_model_config_id=mc_id,
                dataset_export_ids=[],
                action="train",
                status="running",
                training_backend="cosmos_rl_tao_vlm",
                job_config={},
                tao_create_job_request={},
            )
            session.add(job)
            session.commit()

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs/{mc_id}:reprobe"
        )
        assert resp.status_code == 409
        assert "Cannot re-probe" in resp.json()["detail"]

    def test_reprobe_404_nonexistent(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs/nonexistent:reprobe"
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Model config not found"

    def test_probes_run_independently(self, test_app_client):
        """One probe failing doesn't block the others."""
        project_id = _create_project(test_app_client)

        # Find the current Mistral seed (thinking/visual modes are both none).
        items = test_app_client.get(f"/v1/projects/{project_id}/model_configs").json()[
            "items"
        ]
        mistral = next(i for i in items if i["model_name"] == MISTRAL_MEDIUM_3_5)
        mc_id = mistral["model_config_id"]

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/model_configs/{mc_id}:reprobe"
        )
        assert resp.status_code == 200
        data = resp.json()
        # Structured gen should probe and succeed (mock returns valid JSON)
        assert data["structured_generation_support"] == "supported"
        # Thinking toggle mode=none → auto unsupported (no network call)
        assert data["thinking_toggle_support"] == "unsupported"
        # Visual budget mode=none → auto unsupported (no network call)
        assert data["visual_budget_support"] == "unsupported"


# ── Log point 6 ─────────────────────────────────────────────────────────────


class TestProbeLogging:
    def test_reprobe_emits_structured_log(self, test_app_client, caplog):
        """Log point 6: capability probes emit structured JSON."""
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={
                    "choices": [
                        {
                            "message": {"content": '{"ok": true}'},
                            "finish_reason": "stop",
                        }
                    ]
                },
                error_class=None,
                attempts=1,
            )

            project_id = _create_project(test_app_client)
            items = test_app_client.get(
                f"/v1/projects/{project_id}/model_configs"
            ).json()["items"]
            mc_id = items[0]["model_config_id"]

            with caplog.at_level(logging.INFO):
                test_app_client.post(
                    f"/v1/projects/{project_id}/model_configs/{mc_id}:reprobe"
                )

            # Check that capability_probes log records were emitted
            probe_records = [r for r in caplog.records if "capability_probes" in r.name]
            assert len(probe_records) >= 2, (
                f"Expected at least 2 capability_probes log records (start + complete), "
                f"got {len(probe_records)}"
            )

            # Check that the completion record contains probe results
            completion_records = [
                r for r in probe_records if "complete" in r.getMessage().lower()
            ]
            assert len(completion_records) >= 1, "Expected a completion log record"

            # The structured formatter reads details from the record.
            # With caplog, details may be in the record's __dict__
            rec = completion_records[0]
            msg = rec.getMessage()
            assert "complete" in msg.lower()

            # Verify component is set correctly on the logger name
            assert "capability_probes" in rec.name


# ── tao_base_experiment_* visibility ──────────────────────────────────


class TestModelConfigResponseExposesTaoBaseExperiment:
    """GET /model_configs[/{mc_id}] MUST expose tao_base_experiment_id
    and tao_base_experiment_pull_status so operators can see provisioning
    state without direct sqlite3 access. PATCH continues to reject these
    fields as extra inputs (they are read-only via the bootstrap CLI).
    """

    def test_get_includes_tao_base_experiment_fields(self, test_app_client):
        """The GET response payload exposes both fields, defaulting to null
        when never provisioned. The Pydantic schema declares them as
        Optional[str], so absence of provisioning yields ``null`` JSON
        (not a dropped key).
        """
        from vlm_feedback_loop.schemas.model_config import ModelConfigResponse

        # Schema-level assertion: both fields appear on the response model.
        fields = ModelConfigResponse.model_fields
        assert "tao_base_experiment_id" in fields, (
            "ModelConfigResponse must expose tao_base_experiment_id "
            "so GET /model_configs surfaces provisioning state"
        )
        assert "tao_base_experiment_pull_status" in fields, (
            "ModelConfigResponse must expose tao_base_experiment_pull_status"
        )

        # End-to-end assertion: a freshly seeded ModelConfig serializes
        # both fields (as null) over the HTTP boundary.
        pid = _create_project(test_app_client)
        resp = test_app_client.get(f"/v1/projects/{pid}/model_configs")
        assert resp.status_code == 200
        items = resp.json()["items"]
        student_bases = [
            m for m in items if "student_base" in (m.get("eligible_roles") or [])
        ]
        assert len(student_bases) >= 1, "Seed should include Cosmos student_base"
        for m in student_bases:
            assert "tao_base_experiment_id" in m, (
                f"response missing tao_base_experiment_id on {m['model_name']}"
            )
            assert "tao_base_experiment_pull_status" in m, (
                f"response missing tao_base_experiment_pull_status on {m['model_name']}"
            )

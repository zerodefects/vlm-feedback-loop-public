# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the pinned AIPerf subprocess against an OpenAI-compatible server."""

from __future__ import annotations

import json

import pytest

from mock_nim_server import MockNIMServer
from vlm_feedback_loop.services.benchmark_adapter import AIPerfAdapter


@pytest.mark.asyncio
async def test_aiperf_replays_raw_multimodal_payload_without_output_cap(tmp_path):
    """The real driver must preserve the production request body verbatim."""
    input_file = tmp_path / "requests.jsonl"
    payloads = [
        {
            "model": "student-under-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
        },
        {
            "model": "student-under-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify the second image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
        },
    ]
    input_file.write_text("".join(json.dumps(item) + "\n" for item in payloads))

    with MockNIMServer() as server:
        result = await AIPerfAdapter().run(
            base_url=server.base_url,
            model="student-under-test",
            concurrency=2,
            input_file=input_file,
            artifact_dir=tmp_path / "artifacts",
            request_count=2,
            auth_headers={"Authorization": "Bearer benchmark-secret"},
            deadline_s=30.0,
        )

        assert result.failed is False, result.failure_reason
        assert result.successful_request_count == 2
        assert result.failed_request_count == 0
        assert sorted(
            request["body"]["messages"][0]["content"][0]["text"]
            for request in server.chat_requests
        ) == sorted(item["messages"][0]["content"][0]["text"] for item in payloads)
        assert all(request["body"] in payloads for request in server.chat_requests)
        assert all(
            request["headers"].get("source") == "vlm-feedback-loop"
            for request in server.chat_requests
        )
        assert all(
            request["headers"].get("Authorization") == "Bearer benchmark-secret"
            for request in server.chat_requests
        )
        assert all(
            "max_tokens" not in request["body"] for request in server.chat_requests
        )
        assert all(
            "max_completion_tokens" not in request["body"]
            for request in server.chat_requests
        )
        config_text = (tmp_path / "aiperf-c2.yaml").read_text()
        assert "${VLM_BENCHMARK_API_KEY}" in config_text
        assert "benchmark-secret" not in config_text
        for artifact in (tmp_path / "artifacts").rglob("*"):
            if artifact.is_file():
                assert b"benchmark-secret" not in artifact.read_bytes()

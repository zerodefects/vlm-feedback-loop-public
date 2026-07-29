# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process test helpers shared by the unit and integration suites.

Everything here is xdist-safe and spawns no servers: Settings factories,
NIM-transport fakes for tests that stub the ``prompt_service`` seam, an
in-memory S3 client fake for the TAO upload suites, a project seeder for
TestClient / ASGITransport pipelines. Live-server machinery stays in
``tests/integration/conftest.py`` and optional-feature scaffolding stays in
its feature-specific support module.

Import as ``from support import ...`` — ``tests/`` is on ``sys.path``
because pytest inserts the rootdir conftest directory.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from vlm_feedback_loop._defaults import DEFAULTS, SECRET_KEYS
from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services import project_service
from vlm_feedback_loop.services.image_transport import (
    BatchPrepareResult,
    PreparedImage,
)
from vlm_feedback_loop.services.nim_client import NimChatCompletionsResult

#: Teacher model used by the in-process proposal-pipeline tests.
TEST_TEACHER_MODEL = "qwen/qwen3.5-397b-a17b"

#: 1×1 transparent PNG — the byte payload for materialized image fixtures.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

#: Minimal two-field SchemaCore envelope (aux rationale + core enum) used
#: by the ICL-loop and structured-generation-fallback e2e tests.
CATEGORY_ENUM_SCHEMA_ENVELOPE: dict[str, Any] = {
    "fields": [
        {
            "field_id": "rn",
            "field_name": "rationale_note",
            "type": "string",
            "role": "aux",
            "display_order": 0,
        },
        {
            "field_id": "cat",
            "field_name": "category",
            "type": "enum",
            "role": "core",
            "allowed_values": ["alpha", "beta", "gamma"],
            "display_order": 1,
        },
    ],
    "generation_order": ["rationale_note", "category"],
    "derived_json_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["category"],
        "properties": {
            "rationale_note": {"type": "string"},
            "category": {
                "type": "string",
                "enum": ["alpha", "beta", "gamma"],
            },
        },
        "x-generation-order": ["rationale_note", "category"],
    },
}


def build_test_settings(workspace: Path | str, **overrides: Any) -> Settings:
    """Settings from the non-secret defaults + a temp workspace.

    Bypasses ``load_settings``/config.yaml entirely. Secrets are absent
    unless passed explicitly (e.g. ``NVIDIA_API_KEY="nvapi-test-key"``);
    overrides win over any key already present in ``DEFAULTS``.
    """
    values = {k: v for k, v in DEFAULTS.items() if k not in SECRET_KEYS}
    values.update(overrides)
    return Settings(WORKSPACE_ROOT=str(workspace), **values)


def fake_nim_success(content: str) -> NimChatCompletionsResult:
    """Successful chat-completions transport result carrying ``content``."""
    return NimChatCompletionsResult(
        success=True,
        content=content,
        finish_reason="stop",
        usage={"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550},
        status_code=200,
    )


def fake_nim_failure(
    error: str, status_code: int | None = 500
) -> NimChatCompletionsResult:
    """Failed chat-completions transport result carrying ``error``."""
    return NimChatCompletionsResult(
        success=False,
        content=None,
        finish_reason=None,
        usage=None,
        error=error,
        status_code=status_code,
    )


def fake_nim_timeout() -> NimChatCompletionsResult:
    """Timed-out transport result: no status code, ``Request timed out``."""
    return fake_nim_failure("Request timed out", status_code=None)


def fake_prepare_result(n: int, media_type: str = "image/jpeg") -> BatchPrepareResult:
    """N-image ``BatchPrepareResult`` with deterministic ``IMG{i}`` data URLs."""
    return BatchPrepareResult(
        images=[
            PreparedImage(
                content_part={
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,IMG{i}"},
                },
                transport_mode="base64_inline",
                format_transmitted=media_type,
            )
            for i in range(n)
        ],
        success=True,
    )


class FakeS3Client:
    """Minimal in-memory fake implementing S3ClientProtocol.

    Records every call in ``calls`` as ``(method, kwargs)`` pairs and
    stores object bodies + metadata in ``objects``. The ``raise_on_*``
    knobs let failure-path tests make a specific S3 call blow up.
    """

    def __init__(
        self, *, raise_on_put: bool = False, raise_on_head: bool = False
    ) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # Populated during create_multipart_upload; consumed by complete.
        self._multipart_state: dict[str, list[bytes]] = {}
        self._multipart_meta: dict[str, dict[str, str]] = {}
        self._upload_id_counter = 0
        self.raise_on_put = raise_on_put
        self.raise_on_head = raise_on_head

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, kwargs))

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self._record("head_object", {"Bucket": Bucket, "Key": Key})
        if self.raise_on_head:
            raise RuntimeError("head_object not allowed in this test")
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise KeyError(f"NoSuchKey {Bucket}/{Key}")
        return {"Metadata": obj.get("Metadata", {}), "ContentLength": len(obj["Body"])}

    def put_object(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,
        Body: bytes,
        Metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._record(
            "put_object",
            {"Bucket": Bucket, "Key": Key, "size": len(Body), "Metadata": Metadata},
        )
        if self.raise_on_put:
            raise RuntimeError("put_object failed")
        self.objects[(Bucket, Key)] = {
            "Body": bytes(Body),
            "Metadata": dict(Metadata or {}),
        }
        return {"ETag": '"stub-etag"'}

    def create_multipart_upload(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,
        Metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._record(
            "create_multipart_upload",
            {"Bucket": Bucket, "Key": Key, "Metadata": Metadata},
        )
        self._upload_id_counter += 1
        upload_id = f"upload-{self._upload_id_counter}"
        self._multipart_state[upload_id] = []
        self._multipart_meta[upload_id] = dict(Metadata or {})
        return {"UploadId": upload_id}

    def upload_part(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,
        PartNumber: int,
        UploadId: str,
        Body: bytes,
    ) -> dict[str, Any]:
        self._record(
            "upload_part",
            {
                "Bucket": Bucket,
                "Key": Key,
                "PartNumber": PartNumber,
                "UploadId": UploadId,
                "size": len(Body),
            },
        )
        self._multipart_state[UploadId].append(bytes(Body))
        return {"ETag": f'"part-{PartNumber}-etag"'}

    def complete_multipart_upload(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,
        UploadId: str,
        MultipartUpload: dict[str, Any],
    ) -> dict[str, Any]:
        self._record(
            "complete_multipart_upload",
            {
                "Bucket": Bucket,
                "Key": Key,
                "UploadId": UploadId,
                "Parts": MultipartUpload.get("Parts"),
            },
        )
        body = b"".join(self._multipart_state.pop(UploadId, []))
        meta = self._multipart_meta.pop(UploadId, {})
        self.objects[(Bucket, Key)] = {"Body": body, "Metadata": meta}
        return {"ETag": '"multipart-etag"'}

    def abort_multipart_upload(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,
        UploadId: str,
    ) -> dict[str, Any]:
        self._record(
            "abort_multipart_upload",
            {"Bucket": Bucket, "Key": Key, "UploadId": UploadId},
        )
        self._multipart_state.pop(UploadId, None)
        self._multipart_meta.pop(UploadId, None)
        return {}


def seed_hosted_teacher_project(
    workspace: Path,
    *,
    project_id: str,
    project_name: str,
    guidance_id: str,
    endpoint_id: str,
    model_config_id: str,
    guidance_description: str,
    guidance_rules: str = "",
    schema_envelope: dict[str, Any] | None = None,
    model_name: str = TEST_TEACHER_MODEL,
    max_images_per_request: int = 10,
    example_keys: Sequence[str] = (),
) -> Engine:
    """Seed a minimal hosted-Teacher project ready for /proposals calls.

    Creates the project DB under ``workspace/projects/{project_id}`` with
    a Project, hosted NimEndpoint, teacher ModelConfig, active Guidance,
    and one Unlabeled Example per key in ``example_keys``, then registers
    the engine in the project-service cache and returns it.

    ``test_pool_fraction`` is forced to 0.0 so every Verified save routes
    to non-pool (floor(N * 0.0) = 0 ⇒ pool_assignment=None) and stays
    ICL-eligible — the in-process proposal tests depend on that.
    """
    project_dir = workspace / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "artifacts").mkdir(exist_ok=True)
    engine = open_project_db(project_dir)

    with Session(engine) as s:
        s.add(
            Project(
                project_id=project_id,
                name=project_name,
                project_dir=str(project_dir),
                active_guidance_id=guidance_id,
                teacher_model_config_id=model_config_id,
                labeling_generation_preset_key="precise",
                thinking_default_on=False,
                visual_budget_preset_key="balanced",
                structured_generation_mode_default="auto",
                rationale_anti_anchoring=True,
                auto_evaluate_enabled=False,
                icl_recommendation_dismissed_at_count=0,
                export_field_mode="all",
                phash_algorithm="dct_phash_64",
                test_pool_fraction=0.0,
                scaleup_exact_match_threshold=0.80,
                scaleup_per_field_match_threshold=0.80,
                scaleup_min_per_value_f1_threshold=0.80,
                scaleup_accept_rate_threshold=0.80,
                scaleup_accept_rate_window=50,
                scaleup_min_test_pool_size=20,
                schema_refinement_reminders_dismissed=0,
            )
        )
        s.add(
            NimEndpoint(
                endpoint_id=endpoint_id,
                project_id=project_id,
                display_name="Hosted NIM",
                base_url="https://test.nvidia.com/v1",
                endpoint_mode="hosted",
                api_format="openai_compatible",
                auth_mode="bearer",
                source_kind="seeded_hosted",
            )
        )
        s.add(
            ModelConfig(
                model_config_id=model_config_id,
                project_id=project_id,
                endpoint_id=endpoint_id,
                model_name=model_name,
                context_window_tokens=262144,
                eligible_roles=json.dumps(["teacher"]),
                supports_image_input=True,
                structured_generation_support="supported",
                thinking_toggle_mode="qwen_enable_thinking",
                thinking_toggle_support="supported",
                visual_budget_mode="none",
                visual_budget_support="unsupported",
                max_images_per_request=max_images_per_request,
            )
        )
        s.add(
            Guidance(
                guidance_id=guidance_id,
                project_id=project_id,
                version_number=1,
                description=guidance_description,
                schema=schema_envelope or CATEGORY_ENUM_SCHEMA_ENVELOPE,
                rules=guidance_rules,
            )
        )
        for i, key in enumerate(example_keys):
            s.add(
                Example(
                    example_key=key,
                    project_id=project_id,
                    storage_ref=f"/fake/{key}.jpg",
                    ingested_at=utc_now(),
                    source_metadata={},
                    state="Unlabeled",
                    # Distinct hex hashes so diversity scoring can rank.
                    phash=format(i, "x").rjust(16, "0"),
                )
            )
        s.commit()

    project_service.set_project_engine(project_id, engine)
    return engine

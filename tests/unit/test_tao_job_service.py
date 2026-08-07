# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tao_job_service.

Coverage:
  - TAO job endpoints
  - TAO job state machine detail
  - TAO submission protocol and restart recovery
  - dataset binding

``_submit_to_tao`` and ``poll_tao_job`` are monkeypatched in every test —
no real HTTP calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_endpoint_row,
    add_model_config_row,
    add_project_row,
    make_tao_settings,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.schemas.tao_job import (
    JobConfig,
    LoRAConfig,
    TAOCreateJobRequest,
    TAOJobCreateRequest,
    TAOJobResponse,
)
from vlm_feedback_loop.services import tao_job_service
from vlm_feedback_loop.services.tao_job_service import (
    ALLOWED_TRANSITIONS,
    CANONICAL_STATUSES,
    TERMINAL_STATUSES,
    apply_dataset_binding,
    can_transition,
    cancel_tao_job,
    classify_tao_failure,
    compute_request_checksum,
    create_tao_job,
    extract_actionable_failure_from_logs,
    get_tao_job,
    list_tao_jobs,
    map_tao_raw_status,
    recover_interrupted_tao_jobs,
    submit_chain_job,
)

# ── Test constants ──────────────────────────────────────────────────────────

PID = "test-proj"
MCID = "mc-cosmos-8b"
EID = "ep-001"
GID = "guid-001"
EXPORT_A = "de-train-a"
EXPORT_B = "de-train-b"
EXPORT_EVAL = "de-eval"


# ── Fixtures ────────────────────────────────────────────────────────────────


def _setup_project_db(tmp_path: Path, project_id: str = PID):
    return open_project_workspace(tmp_path, project_id, register_engine=True)


def _add_project(session, project_id=PID, project_dir="/tmp/test"):
    add_project_row(session, project_id, str(project_dir))


def _add_endpoint(session, project_id=PID, endpoint_id=EID):
    add_endpoint_row(session, project_id, endpoint_id)


def _add_student_base_model(session, project_id=PID, mc_id=MCID, ep_id=EID):
    add_model_config_row(
        session,
        project_id,
        mc_id,
        ep_id,
        model_name="nvidia/cosmos-reason2-8b",
        eligible_roles=json.dumps(["teacher", "student_base"]),
        thinking_toggle_mode="qwen_enable_thinking",
        thinking_toggle_support="supported",
        visual_budget_mode="mm_processor_size",
        visual_budget_support="supported",
    )


def _add_non_student_model(session, project_id=PID, mc_id="mc-mistral", ep_id=EID):
    add_model_config_row(
        session,
        project_id,
        mc_id,
        ep_id,
        model_name="mistral-large-3",
        context_window_tokens=262144,
        eligible_roles=json.dumps(["teacher"]),
    )


def _add_dataset_export(
    session,
    export_id,
    project_id=PID,
    *,
    dataset_intent="training",
    export_field_mode="all",
    guidance_id=GID,
):
    session.add(
        DatasetExport(
            dataset_export_id=export_id,
            project_id=project_id,
            dataset_intent=dataset_intent,
            export_field_mode=export_field_mode,
            guidance_id=guidance_id,
            label_tier_filter="verified_only",
            selection_definition_snapshot={},
            artifact_refs={"archive_path": f"/tmp/{export_id}.tar.gz"},
            manifest_ref=f"/tmp/{export_id}.manifest.json",
            example_count=10,
        )
    )


def _make_request_body(
    *,
    action="train",
    dataset_export_ids=None,
    student_base_model_config_id=MCID,
    hyperparameters=None,
) -> TAOJobCreateRequest:
    if dataset_export_ids is None:
        dataset_export_ids = [EXPORT_A]
    return TAOJobCreateRequest(
        student_base_model_config_id=student_base_model_config_id,
        dataset_export_ids=dataset_export_ids,
        job_config=JobConfig(
            training_preset="standard",
            lora_config=LoRAConfig(
                enable_lora=True,
                lora_rank=16,
                lora_alpha=32,
                lora_dropout=0.05,
                lora_target_modules=["q_proj", "v_proj"],
            ),
            hyperparameters=hyperparameters or {"train": {"epoch": 3}},
            num_nodes=1,
            num_gpus_per_node=8,
            tao_release_version="6.26.3",
            cosmos_rl_container_tag="6.26.3-cosmos-rl",
        ),
        tao_create_job_request=TAOCreateJobRequest(
            kind="experiment",
            action=action,
            specs={"train": {"epoch": 3}},
        ),
    )


@pytest.fixture()
def seeded(tmp_path):
    """Set up a project DB with all seed data for the happy path."""
    engine, project_dir, workspace = _setup_project_db(tmp_path)
    with Session(engine) as s:
        _add_project(s, project_dir=project_dir)
        _add_endpoint(s)
        _add_student_base_model(s)
        _add_non_student_model(s)
        _add_dataset_export(s, EXPORT_A, export_field_mode="all")
        _add_dataset_export(s, EXPORT_B, export_field_mode="all")
        _add_dataset_export(
            s, EXPORT_EVAL, dataset_intent="evaluation", export_field_mode="all"
        )
        s.commit()
    settings = make_tao_settings(workspace)
    yield engine, settings


# ═══════════════════════════════════════════════════════════════════════════
# State machine
# ═══════════════════════════════════════════════════════════════════════════


class TestStateMachine:
    def test_all_10_canonical_statuses_present(self):
        expected = {
            "not_started",
            "submitting",
            "submitted",
            "queued",
            "running",
            "paused",
            "succeeded",
            "failed",
            "canceled",
            "deleted",
        }
        assert expected == CANONICAL_STATUSES

    def test_terminal_statuses(self):
        assert {"succeeded", "failed", "canceled", "deleted"} == TERMINAL_STATUSES

    @pytest.mark.parametrize(
        "frm,to,ok",
        [
            ("not_started", "submitting", True),
            ("not_started", "failed", True),
            ("not_started", "canceled", True),
            ("not_started", "running", False),  # no direct jump
            ("submitting", "submitted", True),
            ("submitting", "failed", True),
            ("submitting", "canceled", False),  # only → submitted|failed
            ("submitted", "queued", True),
            ("submitted", "running", True),
            ("submitted", "failed", True),
            ("submitted", "canceled", True),
            ("queued", "running", True),
            ("queued", "paused", False),
            ("running", "succeeded", True),
            ("running", "paused", True),
            ("running", "canceled", True),
            # submitted/queued/paused → succeeded allowed because TAO
            # can complete a fast 3-epoch SFT within a single poll window.
            # Observed reality wins.
            ("submitted", "succeeded", True),
            ("queued", "succeeded", True),
            ("paused", "succeeded", True),
            # But not_started/submitting CANNOT reach succeeded
            # directly — those require a TAO external_id round-trip first.
            ("not_started", "succeeded", False),
            ("submitting", "succeeded", False),
            ("paused", "running", True),
            ("paused", "canceled", True),
            ("paused", "failed", False),
            # terminal states never transition out (except deleted)
            ("succeeded", "running", False),
            ("failed", "running", False),
            ("canceled", "running", False),
            # any non-deleted → deleted allowed
            ("running", "deleted", True),
            ("failed", "deleted", True),
            ("deleted", "running", False),
            ("deleted", "deleted", False),
        ],
    )
    def test_can_transition(self, frm, to, ok):
        assert can_transition(frm, to) is ok

    def test_allowed_transitions_table_shape(self):
        # Every status in the table is a canonical status.
        assert set(ALLOWED_TRANSITIONS.keys()) <= CANONICAL_STATUSES
        # Terminal statuses have empty transition sets (deleted handled separately).
        assert ALLOWED_TRANSITIONS["succeeded"] == frozenset()
        assert ALLOWED_TRANSITIONS["failed"] == frozenset()
        assert ALLOWED_TRANSITIONS["canceled"] == frozenset()

    def test_map_tao_raw_status_case_insensitive(self):
        pairs = [
            ("Done", "submitting", "succeeded"),
            ("DONE", "running", "succeeded"),
            ("Failed", "queued", "failed"),
            ("FAILED", "running", "failed"),
            # FTMS 6.26.3 surfaces container-level errors as "Error" rather
            # than "Failed" — without this mapping the polling loop never
            # detects the terminal state and keeps polling until the
            # smoke deadline.
            ("Error", "running", "failed"),
            ("ERROR", "submitted", "failed"),
            ("errored", "running", "failed"),
            ("Running", "submitted", "running"),
            ("running", "submitted", "running"),
            ("Queued", "submitted", "queued"),
            ("PENDING", "submitted", "queued"),
            ("pending", "submitted", "queued"),
            ("Paused", "running", "paused"),
            ("Canceled", "running", "canceled"),
            ("Cancelled", "running", "canceled"),
            ("CANCELLED", "running", "canceled"),
        ]
        for raw, current, expected in pairs:
            assert map_tao_raw_status(raw, current=current) == expected

    def test_map_tao_raw_status_unknown_preserves_terminal(self):
        assert map_tao_raw_status("weird", current="succeeded") == "succeeded"
        assert map_tao_raw_status("mystery", current="failed") == "failed"

    def test_map_tao_raw_status_unknown_nonterminal_falls_back(self):
        # not_started → queued on unknown raw status (we treat as in-progress).
        assert map_tao_raw_status("mystery", current="not_started") == "queued"
        # other non-terminal → running as conservative default.
        assert map_tao_raw_status("mystery", current="submitted") == "running"

    def test_map_tao_raw_status_none_preserves_current(self):
        assert map_tao_raw_status(None, current="running") == "running"
        assert map_tao_raw_status(None, current="paused") == "paused"

    def test_extracts_quantize_exception_hidden_by_generic_ftms_status(self):
        logs = (
            "\x1b[31;20mERROR - Quantization failed: offset overflow while "
            "concatenating arrays, consider casting input to large_list first. "
            "(logging.py:191)\x1b[0m\n"
            "INFO - quantize action failed for cosmos-rl\n"
        )
        result = extract_actionable_failure_from_logs(logs)
        assert result is not None
        assert "offset overflow while concatenating arrays" in result
        assert "2 GiB" in result
        assert "num_calibration_samples" in result

    def test_classifies_arrow_offset_overflow(self):
        friendly, category = classify_tao_failure(
            "pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays"
        )
        assert category == "quantization_arrow_offset_overflow"
        assert "128" in friendly

    def test_job_response_repairs_historical_generic_failure_from_logs(self):
        job = TAOJob(
            tao_job_id="job-old",
            project_id=PID,
            student_base_model_config_id=MCID,
            dataset_export_ids=[EXPORT_A],
            action="quantize",
            status="failed",
            training_backend="cosmos_rl_tao_vlm",
            job_config={},
            tao_create_job_request={},
            error_ref="quantize action failed for cosmos-rl",
            outputs={
                "tao_logs_text": (
                    "Quantization failed: offset overflow while concatenating "
                    "arrays, consider casting to large_list first."
                )
            },
            created_at=utc_now(),
        )

        response = tao_job_service._job_to_dict(job)
        assert "offset overflow while concatenating arrays" in response["error_ref"]
        assert "num_calibration_samples" in response["error_ref"]
        # The read repair does not rewrite the immutable historical row.
        assert job.error_ref == "quantize action failed for cosmos-rl"

    def test_specific_job_error_takes_precedence_over_captured_logs(self):
        """A concrete stored failure must not be replaced by stale log text."""
        job = TAOJob(
            tao_job_id="job-specific",
            project_id=PID,
            student_base_model_config_id=MCID,
            dataset_export_ids=[EXPORT_A],
            action="quantize",
            status="failed",
            training_backend="cosmos_rl_tao_vlm",
            job_config={},
            tao_create_job_request={},
            error_ref="scheduler rejected the requested GPU shape",
            outputs={
                "tao_logs_text": (
                    "Quantization failed: offset overflow while concatenating arrays"
                )
            },
            created_at=utc_now(),
        )

        assert (
            tao_job_service.effective_tao_job_error_ref(job)
            == "scheduler rejected the requested GPU shape"
        )

    def test_job_response_exposes_post_success_artifact_processing(self):
        """A TAO-complete job remains visibly in progress until artifacts finish."""
        job = TAOJob(
            tao_job_id="job-finalizing",
            project_id=PID,
            student_base_model_config_id=MCID,
            dataset_export_ids=[EXPORT_A],
            action="train",
            status="succeeded",
            training_backend="cosmos_rl_tao_vlm",
            job_config={},
            tao_create_job_request={},
            outputs_fetch_status="in_progress",
            outputs_fetch_error_ref=None,
            created_at=utc_now(),
        )

        response = TAOJobResponse(**tao_job_service._job_to_dict(job)).model_dump()

        assert response["status"] == "succeeded"
        assert response["outputs_fetch_status"] == "in_progress"
        assert response["outputs_fetch_error_ref"] is None


# ═══════════════════════════════════════════════════════════════════════════
# HF_TOKEN passthrough via docker_env_vars
# ═══════════════════════════════════════════════════════════════════════════


class TestHfTokenPassthrough:
    """``_submit_to_tao`` MUST inject ``settings.HF_TOKEN`` into
    ``docker_env_vars.HF_TOKEN`` on every TAO POST when the setting is
    populated. Cosmos Reason2 family is gated on HF; without this
    passthrough the cosmos-rl worker hits HTTP 401 gated-repo on every
    transformers.from_pretrained() call and the job fails before
    training starts.

    These tests pin the key name (``HF_TOKEN``) — TAO FTMS 6.26.3
    rejects ``HF_HOME`` / ``HUGGING_FACE_HUB_TOKEN`` /
    ``HUGGINGFACE_TOKEN`` / ``HF_HUB_TOKEN`` / ``HF_ACCESS_TOKEN`` as
    ``Invalid enum member``; only ``HF_TOKEN`` is whitelisted.
    """

    @pytest.mark.asyncio
    async def test_hf_token_injected_into_docker_env_vars_when_settings_set(
        self, tmp_path, monkeypatch
    ):
        from vlm_feedback_loop.services import tao_job_service

        captured: dict[str, Any] = {}

        async def fake_request(*args: Any, **kwargs: Any) -> Any:
            captured["json_body"] = kwargs.get("json_body")

            class _R:
                status_code = 200
                error_class = None
                body = {"id": "remote-1"}

            return _R()

        async def fake_preflight(
            _settings: Any,
        ) -> tuple[dict[str, str], None]:
            return {"Authorization": "Bearer jwt-test"}, None

        monkeypatch.setattr(tao_job_service, "resilient_request", fake_request)
        monkeypatch.setattr(tao_job_service, "tao_preflight", fake_preflight)

        settings = make_tao_settings(
            tmp_path, TAO_API_KEY="jwt-test", HF_TOKEN="hf_secret_xyz"
        )

        result = await tao_job_service._submit_to_tao(
            {"kind": "experiment", "action": "train", "specs": {}},
            settings=settings,
        )

        assert result["success"] is True
        body = captured["json_body"]
        assert body["docker_env_vars"]["HF_TOKEN"] == "hf_secret_xyz"

    @pytest.mark.asyncio
    async def test_hf_token_not_injected_when_settings_unset(
        self, tmp_path, monkeypatch
    ):
        from vlm_feedback_loop.services import tao_job_service

        captured: dict[str, Any] = {}

        async def fake_request(*args: Any, **kwargs: Any) -> Any:
            captured["json_body"] = kwargs.get("json_body")

            class _R:
                status_code = 200
                error_class = None
                body = {"id": "remote-1"}

            return _R()

        async def fake_preflight(
            _settings: Any,
        ) -> tuple[dict[str, str], None]:
            return {"Authorization": "Bearer jwt-test"}, None

        monkeypatch.setattr(tao_job_service, "resilient_request", fake_request)
        monkeypatch.setattr(tao_job_service, "tao_preflight", fake_preflight)

        # HF_TOKEN intentionally None
        settings = make_tao_settings(tmp_path, TAO_API_KEY="jwt-test")

        await tao_job_service._submit_to_tao(
            {"kind": "experiment", "action": "train", "specs": {}},
            settings=settings,
        )

        body = captured["json_body"]
        # Either docker_env_vars is absent OR present but without HF_TOKEN.
        env = body.get("docker_env_vars") or {}
        assert "HF_TOKEN" not in env

    @pytest.mark.asyncio
    async def test_hf_token_does_not_overwrite_explicit_caller_value(
        self, tmp_path, monkeypatch
    ):
        """If a caller already populated ``docker_env_vars.HF_TOKEN``
        (e.g. an admin override), ``_submit_to_tao`` must NOT clobber it.
        Uses dict.setdefault — additive only."""
        from vlm_feedback_loop.services import tao_job_service

        captured: dict[str, Any] = {}

        async def fake_request(*args: Any, **kwargs: Any) -> Any:
            captured["json_body"] = kwargs.get("json_body")

            class _R:
                status_code = 200
                error_class = None
                body = {"id": "remote-1"}

            return _R()

        async def fake_preflight(
            _settings: Any,
        ) -> tuple[dict[str, str], None]:
            return {"Authorization": "Bearer jwt-test"}, None

        monkeypatch.setattr(tao_job_service, "resilient_request", fake_request)
        monkeypatch.setattr(tao_job_service, "tao_preflight", fake_preflight)

        settings = make_tao_settings(
            tmp_path, TAO_API_KEY="jwt-test", HF_TOKEN="hf_settings_xyz"
        )

        await tao_job_service._submit_to_tao(
            {
                "kind": "experiment",
                "action": "train",
                "specs": {},
                "docker_env_vars": {"HF_TOKEN": "hf_caller_explicit"},
            },
            settings=settings,
        )

        body = captured["json_body"]
        # Caller value preserved (setdefault semantics).
        assert body["docker_env_vars"]["HF_TOKEN"] == "hf_caller_explicit"


class TestTaoJwt401Retry:
    """The TAO JWT is cached process-wide with no TTL. A submit/poll/cancel
    that meets an expired token must invalidate the cache and re-exchange,
    or the long-running polling loop wedges permanently on 401."""

    def _settings(self, tmp_path):
        return make_tao_settings(tmp_path, TAO_API_KEY="jwt-test")

    @pytest.mark.asyncio
    async def test_poll_refreshes_jwt_and_retries_once_on_401(
        self, tmp_path, monkeypatch
    ):
        from vlm_feedback_loop.services import tao_auth, tao_job_service

        calls: list[str] = []
        invalidated: list[bool] = []

        async def fake_request(method: Any, url: Any, **kwargs: Any) -> Any:
            bearer = kwargs["headers"]["Authorization"]
            calls.append(bearer)

            class _R:
                # First attempt (stale bearer) → 401; second (fresh) → ok.
                status_code = 401 if bearer == "Bearer stale" else 200
                error_class = "endpoint_error" if bearer == "Bearer stale" else None
                error_detail = "HTTP 401" if bearer == "Bearer stale" else None
                body = None if bearer == "Bearer stale" else {"status": "Running"}

            return _R()

        bearers = iter(["stale", "fresh"])

        async def fake_bearer(_settings: Any) -> str:
            return next(bearers)

        def fake_invalidate(_settings: Any) -> None:
            invalidated.append(True)

        # The initial request goes out through tao_job_service's
        # ``resilient_request`` binding; the auth headers (initial call AND
        # the single-401-retry's invalidate → re-exchange → re-send) resolve
        # the bearer via ``tao_auth.tao_auth_headers``, so the bearer
        # collaborators are patched on ``tao_auth``.
        monkeypatch.setattr(tao_job_service, "resilient_request", fake_request)
        monkeypatch.setattr(tao_auth, "resilient_request", fake_request)
        monkeypatch.setattr(tao_auth, "get_tao_bearer", fake_bearer)
        monkeypatch.setattr(tao_auth, "invalidate_tao_bearer", fake_invalidate)

        result = await tao_job_service.poll_tao_job(
            "job-1", settings=self._settings(tmp_path)
        )

        assert invalidated == [True], "expired JWT must be invalidated"
        assert calls == ["Bearer stale", "Bearer fresh"], "must retry with fresh JWT"
        assert result["success"] is True
        assert result["tao_status_raw"] == "Running"


# ═══════════════════════════════════════════════════════════════════════════
# Progress derivation from job_details (FTMS 6.26.3 poll bodies)
# ═══════════════════════════════════════════════════════════════════════════


_LIVE_EXTERNAL_ID = "6647f4c1-8f0e-4d83-8027-86911682b545"


def _load_tao_fixture(name: str) -> dict:
    import json
    from pathlib import Path

    fixture_path = Path(__file__).parent.parent / "fixtures" / "tao" / name
    return json.loads(fixture_path.read_text())


class TestProgressDerivation:
    """FTMS 6.26.3 returns no top-level ``progress``; the poll parser must
    derive the documented progress shape from ``job_details.{external_id}``
    JobResult fields, and must NOT fabricate progress during pre-training
    phases. Pinned against live-captured poll bodies (2026-07-14)."""

    def test_download_phase_body_yields_no_progress(self):
        """The submitted-state body (base-model download in flight) carries
        an all-null JobResult — deriving progress from it would light the
        panel with an empty shell."""
        from vlm_feedback_loop.services.tao_job_service import (
            _derive_progress_from_job_details,
            _job_details_entry,
        )

        body = _load_tao_fixture("job_status_submitted.json")
        entry = _job_details_entry(body, _LIVE_EXTERNAL_ID)
        assert entry is not None
        assert _derive_progress_from_job_details(entry) is None

    def test_early_running_body_maps_epoch_total_and_drops_cosmos_key_metric(self):
        """A cosmos-rl poll body's ``key_metric: 0.0`` is the trainer's
        never-populated placeholder, not a measurement — deriving progress
        with the body's network_arch drops it so the JobCard never renders
        a fabricated "key_metric 0.0000"."""
        from vlm_feedback_loop.services.tao_job_service import (
            _derive_progress_from_job_details,
            _job_details_entry,
        )

        body = _load_tao_fixture("job_status_running_early.json")
        entry = _job_details_entry(body, _LIVE_EXTERNAL_ID)
        assert entry is not None
        progress = _derive_progress_from_job_details(
            entry, network_arch=body.get("network_arch")
        )
        assert progress == {
            "epoch_current": None,
            "epoch_total": 1,
            "eta_seconds": None,
            "metrics_latest": None,
            "metrics_history_ref": None,
        }

    def test_non_cosmos_zero_key_metric_is_kept(self):
        """The zero-key_metric drop is scoped to cosmos-rl's known-never-populated
        placeholder: a genuine 0.0 from a backend that does report the
        metric stays visible."""
        from vlm_feedback_loop.services.tao_job_service import (
            _derive_progress_from_job_details,
        )

        progress = _derive_progress_from_job_details(
            {"key_metric": 0.0, "max_epoch": 1},
            network_arch="classification_pyt",
        )
        assert progress is not None
        assert progress["metrics_latest"] == {"key_metric": 0.0}

    def test_cosmos_nonzero_key_metric_is_kept(self):
        """A cosmos-rl job that ever reports a real (nonzero) key_metric
        keeps it — the drop targets the placeholder value only."""
        from vlm_feedback_loop.services.tao_job_service import (
            _derive_progress_from_job_details,
        )

        progress = _derive_progress_from_job_details(
            {"key_metric": 0.42, "max_epoch": 1},
            network_arch="cosmos-rl",
        )
        assert progress is not None
        assert progress["metrics_latest"] == {"key_metric": 0.42}

    def test_midtrain_entry_maps_all_fields(self):
        from vlm_feedback_loop.services.tao_job_service import (
            _derive_progress_from_job_details,
        )

        progress = _derive_progress_from_job_details(
            {
                "epoch": 1,
                "max_epoch": 3,
                "cur_iter": 12,
                "eta": "0:05:30",
                "time_per_iter": "0:00:11",
                "key_metric": 0.42,
                "kpi": [],
            }
        )
        assert progress == {
            "epoch_current": 1,
            "epoch_total": 3,
            "eta_seconds": 330.0,
            "metrics_latest": {
                "key_metric": 0.42,
                "cur_iter": 12,
                "time_per_iter": "0:00:11",
            },
            "metrics_history_ref": None,
        }

    def test_quantize_drops_generic_epoch_and_static_eta_telemetry(self):
        """FTMS reuses JobResult for quantization and populates its epoch
        slots with generic work-unit values.  They must not be presented as
        training epoch metrics."""
        from vlm_feedback_loop.services.tao_job_service import (
            _derive_progress_from_job_details,
        )

        progress = _derive_progress_from_job_details(
            {
                "epoch": None,
                "max_epoch": 100,
                "eta": "0:16:40",
                "time_per_epoch": "0:00:01",
                "time_per_iter": None,
                "key_metric": 0.0,
            },
            network_arch="cosmos-rl",
            action="quantize",
        )
        assert progress is None

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (42, 42.0),
            (42.5, 42.5),
            ("42.5", 42.5),
            ("1:02:03.5", 3723.5),
            ("05:30", 330.0),
            ("soon", None),
            ("", None),
            (None, None),
            (True, None),
            ("1:2:3:4", None),
        ],
    )
    def test_eta_parser(self, raw, expected):
        from vlm_feedback_loop.services.tao_job_service import _parse_eta_seconds

        assert _parse_eta_seconds(raw) == expected

    def test_job_details_entry_sole_entry_fallback(self):
        from vlm_feedback_loop.services.tao_job_service import _job_details_entry

        body = {"job_details": {"some-other-key": {"epoch": 2}}}
        entry = _job_details_entry(body, "requested-id")
        assert entry == {"epoch": 2}
        assert _job_details_entry({"job_details": {}}, "x") is None
        assert _job_details_entry({}, "x") is None

    @pytest.mark.asyncio
    async def test_poll_tao_job_derives_progress_and_nested_status_msg(
        self, tmp_path, monkeypatch
    ):
        """End-to-end through poll_tao_job against the live running body:
        progress derives from job_details and status_msg surfaces from the
        6.26.3 nested detailed_status.message location."""
        from vlm_feedback_loop.services import tao_auth, tao_job_service

        body = _load_tao_fixture("job_status_running_early.json")

        async def fake_request(method: Any, url: Any, **kwargs: Any) -> Any:
            class _R:
                status_code = 200
                error_class = None
                error_detail = None

            _R.body = body
            return _R()

        async def fake_bearer(_settings: Any) -> str:
            return "fresh"

        monkeypatch.setattr(tao_job_service, "resilient_request", fake_request)
        monkeypatch.setattr(tao_auth, "resilient_request", fake_request)
        monkeypatch.setattr(tao_auth, "get_tao_bearer", fake_bearer)

        result = await tao_job_service.poll_tao_job(
            _LIVE_EXTERNAL_ID,
            settings=make_tao_settings(tmp_path, TAO_API_KEY="jwt-test"),
        )

        assert result["success"] is True
        assert result["tao_status_raw"] == "Running"
        assert result["progress"] is not None
        assert result["progress"]["epoch_total"] == 1

    @pytest.mark.asyncio
    async def test_poll_uses_authoritative_local_action_when_tao_omits_it(
        self, tmp_path, monkeypatch
    ):
        """Some live quantize responses omit the top-level action.  The
        caller-provided local action must still suppress generic epoch/ETA
        placeholders."""
        from vlm_feedback_loop.services import tao_auth, tao_job_service

        body = _load_tao_fixture("job_status_running_early.json")
        body.pop("action", None)
        entry = next(iter(body["job_details"].values()))
        entry.update(
            {
                "max_epoch": 100,
                "eta": "0:16:40",
                "time_per_epoch": "0:00:01",
            }
        )

        async def fake_request(method: Any, url: Any, **kwargs: Any) -> Any:
            class _R:
                status_code = 200
                error_class = None
                error_detail = None

            _R.body = body
            return _R()

        async def fake_bearer(_settings: Any) -> str:
            return "fresh"

        monkeypatch.setattr(tao_job_service, "resilient_request", fake_request)
        monkeypatch.setattr(tao_auth, "resilient_request", fake_request)
        monkeypatch.setattr(tao_auth, "get_tao_bearer", fake_bearer)

        result = await tao_job_service.poll_tao_job(
            _LIVE_EXTERNAL_ID,
            settings=make_tao_settings(tmp_path, TAO_API_KEY="jwt-test"),
            action="quantize",
        )

        assert result["success"] is True
        assert result["progress"] is None
        assert result["status_msg"] == "Starting Cosmos-RL SFT training"


# ═══════════════════════════════════════════════════════════════════════════
# Dataset binding
# ═══════════════════════════════════════════════════════════════════════════


class TestApplyDatasetBinding:
    def test_train_uses_media_path(self):
        payload = {"kind": "experiment", "action": "train", "specs": {}}
        out = apply_dataset_binding(
            payload,
            action="train",
            annotation_path="/a.json",
            media_root="/images",
        )
        assert out["specs"]["custom"]["train_dataset"]["media_path"] == "/images"
        assert out["specs"]["custom"]["train_dataset"]["annotation_path"] == "/a.json"
        # Cosmos-RL CustomConfig requires the `train_dataset` key by name;
        # a `custom.dataset` shape fails in the container
        # with `ValidationError: train_dataset Field required`.
        assert "dataset" not in out["specs"].get("custom", {})
        # no media_dir for train
        assert "dataset" not in out["specs"] or "media_dir" not in out["specs"].get(
            "dataset", {}
        )

    def test_evaluate_uses_dataset_media_dir(self):
        # Cosmos-RL ``ITSEvaluator`` reads top-level ``config["dataset"]``
        # (see ``cosmos_rl/evaluation/its_evaluator.py``); using the
        # train-side ``custom.val_dataset`` shape crashes evaluate with
        # ``KeyError: 'dataset'``.
        out = apply_dataset_binding(
            {"kind": "experiment", "action": "evaluate", "specs": {}},
            action="evaluate",
            annotation_path="/a.json",
            media_root="/images",
        )
        assert out["specs"]["dataset"]["media_dir"] == "/images"
        assert out["specs"]["dataset"]["annotation_path"] == "/a.json"
        # Train-only key MUST NOT leak onto evaluate specs.
        assert "custom" not in out["specs"]

    def test_quantize_uses_top_level_dataset_media_dir(self):
        """cosmos-rl-quantize's argparse CLI accepts ``--media_dir``,
        NOT ``--media_path`` — the train-style
        ``custom.train_dataset.media_path`` shape is rejected by the
        cosmos-rl-quantize entrypoint with
        ``unrecognized arguments: --media_path /...``. Quantize uses
        the same top-level ``dataset.{media_dir, annotation_path}``
        binding as evaluate — both go through the cosmos-rl-quantize
        entrypoint mapper that reads the ``--media_dir`` flag."""
        out = apply_dataset_binding(
            {"kind": "experiment", "action": "quantize", "specs": {}},
            action="quantize",
            annotation_path="/a.json",
            media_root="/images",
        )
        assert out["specs"]["dataset"]["media_dir"] == "/images"
        assert out["specs"]["dataset"]["annotation_path"] == "/a.json"
        # The legacy train-style binding MUST NOT leak — cosmos-rl-quantize
        # rejects --media_path.
        assert "custom" not in out["specs"]

    def test_preserves_existing_specs(self):
        payload = {
            "kind": "experiment",
            "action": "train",
            "specs": {"train": {"epoch": 3, "other": "preserved"}},
        }
        out = apply_dataset_binding(
            payload,
            action="train",
            annotation_path="/a.json",
            media_root="/m",
        )
        assert out["specs"]["train"]["epoch"] == 3
        assert out["specs"]["train"]["other"] == "preserved"
        assert out["specs"]["custom"]["train_dataset"]["media_path"] == "/m"
        # input untouched
        assert payload["specs"] == {"train": {"epoch": 3, "other": "preserved"}}

    def test_inference_leaves_specs_alone(self):
        payload = {"kind": "experiment", "action": "inference", "specs": {"x": 1}}
        out = apply_dataset_binding(
            payload,
            action="inference",
            annotation_path="/a.json",
            media_root="/m",
        )
        assert out["specs"] == {"x": 1}


# ═══════════════════════════════════════════════════════════════════════════
# Checksum
# ═══════════════════════════════════════════════════════════════════════════


class TestChecksum:
    def test_is_deterministic(self):
        payload = {"b": 2, "a": 1, "nested": {"y": [1, 2], "x": "z"}}
        assert compute_request_checksum(payload) == compute_request_checksum(payload)

    def test_key_order_does_not_matter(self):
        # The function canonicalizes via sort_keys.
        a = {"a": 1, "b": 2}
        b = {"b": 2, "a": 1}
        assert compute_request_checksum(a) == compute_request_checksum(b)

    def test_changes_on_payload_change(self):
        a = {"a": 1}
        b = {"a": 2}
        assert compute_request_checksum(a) != compute_request_checksum(b)


# ═══════════════════════════════════════════════════════════════════════════
# Create — happy path + submission protocol
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateHappyPath:
    @pytest.mark.asyncio
    async def test_create_returns_submitted_with_external_id(self, seeded, monkeypatch):
        engine, settings = seeded
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_external_job_id": "ext-job-42",
                    "error": None,
                }
            ),
        )

        body = _make_request_body(action="train", dataset_export_ids=[EXPORT_A])
        result = await create_tao_job(PID, body=body, settings=settings)

        assert not isinstance(result, str), result
        assert result["status"] == "submitted"
        assert result["tao_external_job_id"] == "ext-job-42"
        assert result["action"] == "train"
        assert result["training_backend"] == "cosmos_rl_tao_vlm"
        assert result["training_policy_type"] == "sft"
        assert result["started_at"] is not None
        assert result["error_ref"] is None

    @pytest.mark.asyncio
    async def test_checksum_persisted_on_job_config(self, seeded, monkeypatch):
        _, settings = seeded
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_external_job_id": "ext",
                    "error": None,
                }
            ),
        )

        body = _make_request_body()
        result = await create_tao_job(PID, body=body, settings=settings)
        assert not isinstance(result, str)
        assert "tao_create_job_request_checksum" in result["job_config"]
        expected = compute_request_checksum(result["tao_create_job_request"])
        assert result["job_config"]["tao_create_job_request_checksum"] == expected

    @pytest.mark.asyncio
    async def test_tao_create_job_request_persisted_verbatim(self, seeded, monkeypatch):
        _, settings = seeded
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_external_job_id": "ext",
                    "error": None,
                }
            ),
        )

        body = TAOJobCreateRequest(
            student_base_model_config_id=MCID,
            dataset_export_ids=[EXPORT_A],
            job_config=JobConfig(
                training_preset="standard",
                lora_config=LoRAConfig(
                    enable_lora=True,
                    lora_rank=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    lora_target_modules=["q_proj"],
                ),
                tao_release_version="6.26.3",
                cosmos_rl_container_tag="6.26.3-cosmos-rl",
            ),
            tao_create_job_request=TAOCreateJobRequest(
                kind="experiment",
                action="train",
                name="finetune-2026-04-16",  # TAO-native extra
                workspace="workspace_123",
                base_experiment_ids=["base_1"],
                specs={"train": {"epoch": 3, "num_gpus": 8}},
            ),
        )
        result = await create_tao_job(PID, body=body, settings=settings)
        assert not isinstance(result, str)
        stored = result["tao_create_job_request"]
        # TAO-native keys preserved
        assert stored["name"] == "finetune-2026-04-16"
        assert stored["workspace"] == "workspace_123"
        assert stored["base_experiment_ids"] == ["base_1"]
        assert stored["specs"]["train"]["num_gpus"] == 8

    @pytest.mark.asyncio
    async def test_submission_protocol_persists_before_http_call(
        self, seeded, monkeypatch
    ):
        """TAOJob persisted with status=submitting BEFORE the POST.

        We verify this by making ``_submit_to_tao`` raise — the TAOJob row
        must already exist afterward.
        """
        engine, settings = seeded

        async def boom(*args, **kwargs):
            raise RuntimeError("simulated crash mid-POST")

        monkeypatch.setattr(tao_job_service, "_submit_to_tao", boom)

        body = _make_request_body()
        result = await create_tao_job(PID, body=body, settings=settings)
        assert not isinstance(result, str)
        # Status landed on "failed" due to the RuntimeError caught by create_tao_job.
        assert result["status"] == "failed"
        assert result["error_ref"] is not None
        assert "RuntimeError" in result["error_ref"]

        # Double-check persistence: the row is actually in the DB.
        with Session(engine) as s:
            rows = s.query(TAOJob).all()
        assert len(rows) == 1
        assert rows[0].status == "failed"

    @pytest.mark.asyncio
    async def test_submission_http_failure_transitions_to_failed(
        self, seeded, monkeypatch
    ):
        engine, settings = seeded
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": False,
                    "tao_external_job_id": None,
                    "error": "TAO submission failed: HTTP 401",
                }
            ),
        )

        body = _make_request_body()
        result = await create_tao_job(PID, body=body, settings=settings)
        assert not isinstance(result, str)
        assert result["status"] == "failed"
        assert "HTTP 401" in result["error_ref"]
        assert result["tao_external_job_id"] is None
        assert result["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_error_ref_sanitized(self, seeded, monkeypatch):
        _, settings = seeded
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": False,
                    "tao_external_job_id": None,
                    "error": "TAO submission failed: Bearer nvapi-LEAKEDTOKEN rejected",
                }
            ),
        )

        body = _make_request_body()
        result = await create_tao_job(PID, body=body, settings=settings)
        assert not isinstance(result, str)
        assert "nvapi-LEAKEDTOKEN" not in result["error_ref"]
        assert "Bearer nvapi-LEAKEDTOKEN" not in result["error_ref"]
        assert "[REDACTED]" in result["error_ref"]


# ═══════════════════════════════════════════════════════════════════════════
# Create — validation errors
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateValidation:
    @pytest.mark.asyncio
    async def test_unknown_project_returns_not_found(self, tmp_path, monkeypatch):
        engine, project_dir, workspace = _setup_project_db(tmp_path)
        settings = make_tao_settings(workspace)
        # No project row — get_project returns None.
        body = _make_request_body()
        result = await create_tao_job(PID, body=body, settings=settings)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_dataset_export_nonexistent_returns_not_found(
        self, seeded, monkeypatch
    ):
        _, settings = seeded
        body = _make_request_body(dataset_export_ids=["does-not-exist"])
        result = await create_tao_job(PID, body=body, settings=settings)
        assert isinstance(result, str)
        assert "not found" in result.lower()
        assert "does-not-exist" in result

    @pytest.mark.asyncio
    async def test_dataset_export_cross_project_rejected(self, seeded):
        engine, settings = seeded
        # Add a DatasetExport that belongs to a DIFFERENT project.
        with Session(engine) as s:
            _add_dataset_export(s, "de-other", project_id="other-proj")
            s.commit()

        body = _make_request_body(dataset_export_ids=["de-other"])
        result = await create_tao_job(PID, body=body, settings=settings)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_student_base_role_rejected(self, seeded):
        _, settings = seeded
        body = _make_request_body(student_base_model_config_id="mc-mistral")
        result = await create_tao_job(PID, body=body, settings=settings)
        assert isinstance(result, str)
        assert "validation" in result.lower()
        assert "student_base" in result

    @pytest.mark.asyncio
    async def test_train_mixed_export_field_mode_rejected(self, seeded):
        engine, settings = seeded
        with Session(engine) as s:
            _add_dataset_export(s, "de-diff-mode", export_field_mode="core_only")
            s.commit()

        body = _make_request_body(
            action="train", dataset_export_ids=[EXPORT_A, "de-diff-mode"]
        )
        result = await create_tao_job(PID, body=body, settings=settings)
        assert isinstance(result, str)
        assert "MIXED_EXPORT_FIELD_MODE" in result

    @pytest.mark.asyncio
    async def test_train_mixed_dataset_intent_rejected(self, seeded):
        _, settings = seeded
        body = _make_request_body(
            action="train", dataset_export_ids=[EXPORT_A, EXPORT_EVAL]
        )
        result = await create_tao_job(PID, body=body, settings=settings)
        assert isinstance(result, str)
        assert "MIXED_EXPORT_FIELD_MODE" in result

    @pytest.mark.asyncio
    async def test_evaluate_action_skips_field_mode_consistency(
        self, seeded, monkeypatch
    ):
        """Evaluate/quantize accept a single evaluation export."""
        _, settings = seeded
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_external_job_id": "ext",
                    "error": None,
                }
            ),
        )
        body = _make_request_body(action="evaluate", dataset_export_ids=[EXPORT_EVAL])
        result = await create_tao_job(PID, body=body, settings=settings)
        assert not isinstance(result, str)
        assert result["status"] == "submitted"
        # training_policy_type is None for non-train actions.
        assert result["training_policy_type"] is None
        assert result["action"] == "evaluate"

    @pytest.mark.asyncio
    async def test_empty_dataset_export_ids_rejected(self, seeded):
        _, settings = seeded
        body = _make_request_body(dataset_export_ids=[])
        result = await create_tao_job(PID, body=body, settings=settings)
        assert isinstance(result, str)
        assert "validation" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Restart recovery
# ═══════════════════════════════════════════════════════════════════════════


class TestRestartRecovery:
    def test_submitting_with_null_external_id_to_failed(self, seeded):
        engine, settings = seeded
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="orphan-1",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="submitting",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                    tao_external_job_id=None,
                    created_at=utc_now(),
                )
            )
            s.commit()

        recover_interrupted_tao_jobs(settings)

        with Session(engine) as s:
            row = s.query(TAOJob).filter_by(tao_job_id="orphan-1").first()
        assert row.status == "failed"
        assert row.error_ref == "submission_interrupted"
        assert row.completed_at is not None

    def test_submitting_with_external_id_is_untouched(self, seeded):
        engine, settings = seeded
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="confirmed-1",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="submitting",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                    tao_external_job_id="ext-42",
                    created_at=utc_now(),
                )
            )
            s.commit()

        recover_interrupted_tao_jobs(settings)

        with Session(engine) as s:
            row = s.query(TAOJob).filter_by(tao_job_id="confirmed-1").first()
        # Untouched — status still submitting, no error_ref set.
        assert row.status == "submitting"
        assert row.error_ref is None

    def test_terminal_and_submitted_rows_untouched(self, seeded):
        engine, settings = seeded
        with Session(engine) as s:
            for job_id, status in [
                ("a", "submitted"),
                ("b", "running"),
                ("c", "succeeded"),
                ("d", "failed"),
            ]:
                s.add(
                    TAOJob(
                        tao_job_id=job_id,
                        project_id=PID,
                        student_base_model_config_id=MCID,
                        dataset_export_ids=[EXPORT_A],
                        action="train",
                        status=status,
                        training_backend="cosmos_rl_tao_vlm",
                        training_policy_type="sft",
                        job_config={},
                        tao_create_job_request={},
                        tao_external_job_id=None if status == "submitted" else "ext",
                        created_at=utc_now(),
                    )
                )
            s.commit()

        recover_interrupted_tao_jobs(settings)

        with Session(engine) as s:
            statuses = {r.tao_job_id: r.status for r in s.query(TAOJob).all()}
        assert statuses == {
            "a": "submitted",
            "b": "running",
            "c": "succeeded",
            "d": "failed",
        }

    def test_recovery_skips_missing_workspace(self, tmp_path):
        settings = make_tao_settings(tmp_path / "nonexistent_workspace")
        # Must not crash.
        recover_interrupted_tao_jobs(settings)


# ═══════════════════════════════════════════════════════════════════════════
# Get — with and without refresh
# ═══════════════════════════════════════════════════════════════════════════


class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_full_record(self, seeded, monkeypatch):
        _, settings = seeded
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_external_job_id": "ext-1",
                    "error": None,
                }
            ),
        )
        body = _make_request_body()
        created = await create_tao_job(PID, body=body, settings=settings)
        assert not isinstance(created, str)

        fetched = await get_tao_job(
            PID, created["tao_job_id"], refresh=False, settings=settings
        )
        assert not isinstance(fetched, str)
        assert fetched["tao_job_id"] == created["tao_job_id"]
        assert fetched["status"] == "submitted"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_not_found(self, seeded):
        _, settings = seeded
        result = await get_tao_job(
            PID, "does-not-exist", refresh=False, settings=settings
        )
        assert isinstance(result, str)
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_get_refresh_true_updates_status_and_persists_raw(
        self, seeded, monkeypatch
    ):
        _, settings = seeded
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_external_job_id": "ext-42",
                    "error": None,
                }
            ),
        )
        body = _make_request_body()
        created = await create_tao_job(PID, body=body, settings=settings)
        assert not isinstance(created, str)

        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Running",
                    "progress": {"epoch_current": 1, "epoch_total": 3},
                    "outputs": None,
                    "error": None,
                }
            ),
        )
        refreshed = await get_tao_job(
            PID, created["tao_job_id"], refresh=True, settings=settings
        )
        assert not isinstance(refreshed, str)
        assert refreshed["status"] == "running"
        assert refreshed["tao_status_raw"] == "Running"
        assert refreshed["progress"] == {"epoch_current": 1, "epoch_total": 3}
        assert refreshed["last_polled_at"] is not None
        assert refreshed["started_at"] is not None  # set on first → running

    @pytest.mark.asyncio
    async def test_get_refresh_terminal_skips_poll(self, seeded, monkeypatch):
        engine, settings = seeded
        # Seed a succeeded job directly.
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="terminal-1",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="succeeded",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                    tao_external_job_id="ext",
                    created_at=utc_now(),
                    completed_at=utc_now(),
                )
            )
            s.commit()

        poll_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "poll_tao_job", poll_mock)

        result = await get_tao_job(PID, "terminal-1", refresh=True, settings=settings)
        assert not isinstance(result, str)
        poll_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_refresh_rate_limited(self, seeded, monkeypatch):
        engine, settings = seeded
        recent_ts = (datetime.now(UTC) - timedelta(seconds=5)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="rl-1",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="running",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                    tao_external_job_id="ext",
                    created_at=utc_now(),
                    last_polled_at=recent_ts,
                )
            )
            s.commit()

        poll_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "poll_tao_job", poll_mock)

        await get_tao_job(PID, "rl-1", refresh=True, settings=settings)
        # Polled less than TAO_POLL_MIN_INTERVAL_S (30s default) ago → no call.
        poll_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_refresh_after_interval_polls(self, seeded, monkeypatch):
        engine, settings = (
            make_tao_settings(
                Path(seeded[1].WORKSPACE_ROOT), TAO_POLL_MIN_INTERVAL_S=1
            ),
            None,
        )
        # Rebuild a fresh settings with a tight interval.
        settings = make_tao_settings(
            Path(seeded[1].WORKSPACE_ROOT), TAO_POLL_MIN_INTERVAL_S=1
        )
        engine = seeded[0]
        stale_ts = (datetime.now(UTC) - timedelta(seconds=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="stale-1",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="running",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                    tao_external_job_id="ext-y",
                    created_at=utc_now(),
                    last_polled_at=stale_ts,
                )
            )
            s.commit()

        poll_mock = AsyncMock(
            return_value={
                "success": True,
                "tao_status_raw": "Done",
                "progress": None,
                "outputs": None,
                "error": None,
            }
        )
        monkeypatch.setattr(tao_job_service, "poll_tao_job", poll_mock)

        result = await get_tao_job(PID, "stale-1", refresh=True, settings=settings)
        assert not isinstance(result, str)
        poll_mock.assert_awaited_once()
        assert result["status"] == "succeeded"


# ═══════════════════════════════════════════════════════════════════════════
# List
# ═══════════════════════════════════════════════════════════════════════════


class TestList:
    def _seed_n(self, engine, n):
        """Seed n rows with distinct created_at timestamps.

        The ``_before_insert`` hook forces ``created_at=utc_now()`` on
        insert, so we insert then bulk-update via a query-level
        UPDATE to override.
        """
        base = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
        with Session(engine) as s:
            for i in range(n):
                s.add(
                    TAOJob(
                        tao_job_id=f"job-{i:02d}",
                        project_id=PID,
                        student_base_model_config_id=MCID,
                        dataset_export_ids=[EXPORT_A],
                        action="train",
                        status="queued" if i % 2 == 0 else "running",
                        training_backend="cosmos_rl_tao_vlm",
                        training_policy_type="sft",
                        job_config={},
                        tao_create_job_request={},
                        tao_external_job_id=f"ext-{i}",
                    )
                )
            s.commit()
        # Override created_at post-insert (query-level UPDATE bypasses the hook).
        with Session(engine) as s:
            for i in range(n):
                ts = (base + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
                s.query(TAOJob).filter_by(tao_job_id=f"job-{i:02d}").update(
                    {"created_at": ts}
                )
            s.commit()

    def test_list_newest_first_paginates(self, seeded):
        engine, settings = seeded
        self._seed_n(engine, 5)

        result = list_tao_jobs(PID, cursor=None, limit=2, settings=settings)
        assert not isinstance(result, str)
        page1, cursor = result
        assert len(page1) == 2
        assert cursor is not None
        # Newest-first: page1 = job-04, job-03
        assert [r["tao_job_id"] for r in page1] == ["job-04", "job-03"]

        result2 = list_tao_jobs(PID, cursor=cursor, limit=2, settings=settings)
        assert not isinstance(result2, str)
        page2, cursor2 = result2
        assert [r["tao_job_id"] for r in page2] == ["job-02", "job-01"]
        assert cursor2 is not None

        result3 = list_tao_jobs(PID, cursor=cursor2, limit=2, settings=settings)
        assert not isinstance(result3, str)
        page3, cursor3 = result3
        assert [r["tao_job_id"] for r in page3] == ["job-00"]
        assert cursor3 is None

    def test_list_filters_by_status(self, seeded):
        engine, settings = seeded
        self._seed_n(engine, 5)

        result = list_tao_jobs(
            PID, status_filter="queued", cursor=None, limit=10, settings=settings
        )
        assert not isinstance(result, str)
        items, cursor = result
        statuses = {r["status"] for r in items}
        assert statuses == {"queued"}
        assert len(items) == 3  # indices 0, 2, 4

    def test_list_unknown_project_returns_not_found(self, tmp_path):
        settings = make_tao_settings(tmp_path / "empty")
        result = list_tao_jobs(PID, cursor=None, limit=5, settings=settings)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_list_invalid_cursor_returns_validation(self, seeded):
        _, settings = seeded
        result = list_tao_jobs(
            PID, cursor="not-a-valid-cursor", limit=5, settings=settings
        )
        assert isinstance(result, str)
        assert "validation" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Cancel
# ═══════════════════════════════════════════════════════════════════════════


def _seed_chain_for_cancel(engine, *, running_status: str = "running"):
    """Seed a 4-job chain: [running train, not_started evaluate, not_started quantize, not_started evaluate].

    Returns the four tao_job_ids in chain order.
    """
    ids = ["chain-train", "chain-eval-base", "chain-quant", "chain-eval-q"]
    actions = ["train", "evaluate", "quantize", "evaluate"]
    statuses = [running_status, "not_started", "not_started", "not_started"]
    with Session(engine) as s:
        for i, (tid, action, status) in enumerate(
            zip(ids, actions, statuses, strict=True)
        ):
            s.add(
                TAOJob(
                    tao_job_id=tid,
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action=action,
                    status=status,
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={
                        "kind": "experiment",
                        "action": action,
                        "specs": {},
                    },
                    tao_external_job_id=f"ext-{tid}"
                    if status != "not_started"
                    else None,
                    chain_id="chain-abc",
                    chain_sequence=i + 1,
                )
            )
        s.commit()
    return ids


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_from_running_transitions_to_canceled(
        self, seeded, monkeypatch
    ):
        engine, settings = seeded
        ids = _seed_chain_for_cancel(engine, running_status="running")

        monkeypatch.setattr(
            tao_job_service,
            "_cancel_tao_external",
            AsyncMock(return_value={"success": True, "error": None}),
        )

        result = await cancel_tao_job(PID, ids[0], settings=settings)
        assert not isinstance(result, str), result
        assert result["status"] == "canceled"
        assert result["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_cancel_from_paused_transitions_to_canceled(
        self, seeded, monkeypatch
    ):
        engine, settings = seeded
        ids = _seed_chain_for_cancel(engine, running_status="paused")
        monkeypatch.setattr(
            tao_job_service,
            "_cancel_tao_external",
            AsyncMock(return_value={"success": True, "error": None}),
        )

        result = await cancel_tao_job(PID, ids[0], settings=settings)
        assert not isinstance(result, str), result
        assert result["status"] == "canceled"

    @pytest.mark.asyncio
    async def test_cancel_halts_downstream_not_started_siblings(
        self, seeded, monkeypatch
    ):
        engine, settings = seeded
        ids = _seed_chain_for_cancel(engine, running_status="running")
        monkeypatch.setattr(
            tao_job_service,
            "_cancel_tao_external",
            AsyncMock(return_value={"success": True, "error": None}),
        )

        await cancel_tao_job(PID, ids[0], settings=settings)

        with Session(engine) as s:
            rows = {
                r.tao_job_id: r
                for r in s.query(TAOJob).filter_by(chain_id="chain-abc").all()
            }
        assert rows[ids[0]].status == "canceled"
        for sib in ids[1:]:
            assert rows[sib].status == "failed", rows[sib].status
            assert rows[sib].chain_halted_reason is not None
            assert "canceled by SME" in rows[sib].chain_halted_reason

    @pytest.mark.asyncio
    async def test_cancel_emits_run_failed_for_canceled_and_each_sibling(
        self, seeded, monkeypatch
    ):
        engine, settings = seeded
        ids = _seed_chain_for_cancel(engine, running_status="running")
        monkeypatch.setattr(
            tao_job_service,
            "_cancel_tao_external",
            AsyncMock(return_value={"success": True, "error": None}),
        )

        emitted: list[tuple[str, str, dict]] = []

        async def fake_emit(project_id, event_type, payload):
            emitted.append((project_id, event_type, payload))

        monkeypatch.setattr(tao_job_service.sse_manager, "emit", fake_emit)

        await cancel_tao_job(PID, ids[0], settings=settings)

        # One event for the canceled job + 3 for halted siblings = 4 total.
        assert len(emitted) == 4
        event_types = [e[1] for e in emitted]
        assert event_types == ["run_failed"] * 4
        # First event is the canceled job itself.
        canceled_payload = emitted[0][2]
        assert canceled_payload["tao_job_id"] == ids[0]
        assert canceled_payload["run_type"] == "tao_job"
        assert canceled_payload["error_summary"] == "tao_job_canceled"
        # Remaining events reference the halted siblings.
        halted_ids = {e[2]["tao_job_id"] for e in emitted[1:]}
        assert halted_ids == set(ids[1:])

    @pytest.mark.asyncio
    async def test_cancel_terminal_status_rejected_with_conflict(
        self, seeded, monkeypatch
    ):
        engine, settings = seeded
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="done-1",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="succeeded",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                    tao_external_job_id="ext",
                    completed_at=utc_now(),
                )
            )
            s.commit()

        result = await cancel_tao_job(PID, "done-1", settings=settings)
        assert isinstance(result, str)
        assert "conflict" in result.lower()
        assert "succeeded" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_submitting_status_rejected(self, seeded, monkeypatch):
        engine, settings = seeded
        # submitting only transitions to submitted | failed, so cancel is refused.
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="submitting-1",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="submitting",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                )
            )
            s.commit()

        result = await cancel_tao_job(PID, "submitting-1", settings=settings)
        assert isinstance(result, str)
        assert "conflict" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_not_found(self, seeded):
        _, settings = seeded
        result = await cancel_tao_job(PID, "does-not-exist", settings=settings)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_tao_failure_leaves_state_unchanged(self, seeded, monkeypatch):
        engine, settings = seeded
        ids = _seed_chain_for_cancel(engine, running_status="running")

        monkeypatch.setattr(
            tao_job_service,
            "_cancel_tao_external",
            AsyncMock(
                return_value={
                    "success": False,
                    "error": "TAO cancel failed: 503 service unavailable",
                }
            ),
        )

        result = await cancel_tao_job(PID, ids[0], settings=settings)
        assert isinstance(result, str)
        assert "tao_error" in result.lower()

        # State must not have flipped — the SME can retry.
        with Session(engine) as s:
            row = s.query(TAOJob).filter_by(tao_job_id=ids[0]).first()
            assert row.status == "running"
            # poll_error_ref is populated with the sanitized message.
            assert row.poll_error_ref is not None
            assert "503" in row.poll_error_ref

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error_class", "expected_prefix"),
        [
            ("timeout", "tao_timeout:"),
            ("endpoint_error", "tao_unreachable:"),
        ],
    )
    async def test_cancel_transport_failure_carries_error_class_prefix(
        self, seeded, monkeypatch, error_class, expected_prefix
    ):
        """Transport failures surface the documented 503/504 contract:
        the returned string carries a machine prefix that
        errors.map_service_error turns into 503 (unreachable) / 504
        (timeout), instead of collapsing everything into 502 tao_error.
        """
        engine, settings = seeded
        ids = _seed_chain_for_cancel(engine, running_status="running")

        monkeypatch.setattr(
            tao_job_service,
            "_cancel_tao_external",
            AsyncMock(
                return_value={
                    "success": False,
                    "error": "TAO cancel failed: transport",
                    "error_class": error_class,
                }
            ),
        )

        result = await cancel_tao_job(PID, ids[0], settings=settings)
        assert isinstance(result, str)
        assert result.startswith(expected_prefix)

    @pytest.mark.asyncio
    async def test_cancel_force_local_skips_tao_call_and_transitions(
        self, seeded, monkeypatch
    ):
        """``force_local=True`` skips the TAO POST entirely and
        transitions the local row to ``canceled`` even when the external
        TAO would have failed. The audit trail records the bypass via
        ``poll_error_ref``.
        """
        engine, settings = seeded
        ids = _seed_chain_for_cancel(engine, running_status="running")

        # Even if TAO would fail, force_local must skip the call.
        cancel_mock = AsyncMock(
            return_value={"success": False, "error": "should not be called"}
        )
        monkeypatch.setattr(tao_job_service, "_cancel_tao_external", cancel_mock)

        result = await cancel_tao_job(PID, ids[0], settings=settings, force_local=True)
        assert not isinstance(result, str), result
        assert result["status"] == "canceled"
        assert result["completed_at"] is not None
        cancel_mock.assert_not_called()

        # poll_error_ref records the force-local provenance for audit.
        with Session(engine) as s:
            row = s.query(TAOJob).filter_by(tao_job_id=ids[0]).first()
            assert row.poll_error_ref is not None
            assert "forced_local_cancel" in row.poll_error_ref

    @pytest.mark.asyncio
    async def test_cancel_without_external_id_skips_tao_call(self, seeded, monkeypatch):
        """A ``queued`` job with no ``tao_external_job_id`` (e.g., chain
        pre-created but not yet kicked off) cancels locally without any
        HTTP call to TAO.
        """
        engine, settings = seeded
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="no-ext-1",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="queued",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                    tao_external_job_id=None,
                )
            )
            s.commit()

        cancel_mock = AsyncMock()
        monkeypatch.setattr(tao_job_service, "_cancel_tao_external", cancel_mock)

        result = await cancel_tao_job(PID, "no-ext-1", settings=settings)
        assert not isinstance(result, str)
        assert result["status"] == "canceled"
        cancel_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_late_submission_cannot_resurrect_suite_canceled_job(
        self, seeded, monkeypatch
    ):
        """A suite cancel can race an in-flight TAO create-job request.
        Once the local row is canceled, the late external id is retained
        for audit and immediately canceled instead of restoring
        ``status=submitted``.
        """
        engine, settings = seeded
        with Session(engine) as session:
            session.add(
                TAOJob(
                    tao_job_id="suite-canceled-submission",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="canceled",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={},
                    completed_at=utc_now(),
                )
            )
            session.commit()

        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_external_job_id": "ext-late",
                    "error": None,
                }
            ),
        )
        remote_cancel = AsyncMock(return_value={"success": True, "error": None})
        monkeypatch.setattr(
            tao_job_service,
            "request_tao_job_cancel",
            remote_cancel,
        )

        result = await tao_job_service._submit_and_persist_outcome(
            engine,
            project_id=PID,
            tao_job_id="suite-canceled-submission",
            tao_create_job_request={},
            settings=settings,
            log_label="TAO chain job",
        )

        assert not isinstance(result, str)
        assert result["status"] == "canceled"
        assert result["tao_external_job_id"] == "ext-late"
        remote_cancel.assert_awaited_once_with("ext-late", settings=settings)


# ═══════════════════════════════════════════════════════════════════════════
# Chain-advancement: top-level ``parent_job_id`` injection
# ═══════════════════════════════════════════════════════════════════════════


class TestSubmitChainParentJobId:
    """``submit_chain_job`` MUST inject the parent TAOJob's
    ``tao_external_job_id`` as a top-level ``parent_job_id`` field on
    the POST body. Without this, FTMS's ``infer_parent_model_folder``
    helper resolves ``parent_id`` to ``None`` and the cosmos-rl worker
    crashes in ``cosmos_rl/evaluation/base.py`` with
    ``TypeError: expected str, bytes or os.PathLike object, not NoneType``.

    ``parent_job_id`` is the *external* (FTMS-assigned) id of the
    predecessor job, not the Blueprint-internal ``parent_tao_job_id``.
    """

    def _seed_parent_and_child(
        self,
        engine: Any,
        *,
        parent_status: str = "succeeded",
        parent_external_id: str | None = "ext-train-001",
    ) -> tuple[str, str]:
        parent_id = "chain-train-001"
        child_id = "chain-eval-001"
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id=parent_id,
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status=parent_status,
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={
                        "kind": "experiment",
                        "action": "train",
                        "specs": {},
                    },
                    tao_external_job_id=parent_external_id,
                    chain_id="chain-001",
                    chain_sequence=1,
                )
            )
            s.add(
                TAOJob(
                    tao_job_id=child_id,
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_EVAL],
                    action="evaluate",
                    status="not_started",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type=None,
                    job_config={},
                    tao_create_job_request={
                        "kind": "experiment",
                        "action": "evaluate",
                        "specs": {"custom": {"val_dataset": {}}},
                    },
                    tao_external_job_id=None,
                    parent_tao_job_id=parent_id,
                    chain_id="chain-001",
                    chain_sequence=2,
                )
            )
            s.commit()
        return parent_id, child_id

    @pytest.mark.asyncio
    async def test_parent_job_id_injected_into_top_level_payload(
        self, seeded, monkeypatch
    ):
        engine, settings = seeded
        _parent_id, child_id = self._seed_parent_and_child(engine)

        captured: dict[str, Any] = {}

        async def fake_submit(payload: dict[str, Any], *, settings: Any) -> Any:
            captured["payload"] = payload
            return {
                "success": True,
                "tao_external_job_id": "ext-eval-001",
                "error": None,
            }

        monkeypatch.setattr(tao_job_service, "_submit_to_tao", fake_submit)
        monkeypatch.setattr(tao_job_service.sse_manager, "emit", AsyncMock())

        result = await submit_chain_job(PID, child_id, settings=settings)
        assert result == "submitted"
        assert "payload" in captured, "POST payload was not captured"
        # Top-level parent_job_id (NOT nested under specs).
        assert captured["payload"].get("parent_job_id") == "ext-train-001"
        assert "parent_job_id" not in (captured["payload"].get("specs") or {}), (
            "parent_job_id must be top-level, not under specs"
        )

    @pytest.mark.asyncio
    async def test_no_parent_no_parent_job_id_field(self, seeded, monkeypatch):
        """Root chain jobs (no parent_tao_job_id) MUST NOT add a
        ``parent_job_id`` key to the payload."""
        engine, settings = seeded
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="root-train",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="not_started",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={
                        "kind": "experiment",
                        "action": "train",
                        "specs": {},
                    },
                    tao_external_job_id=None,
                    chain_id="chain-002",
                    chain_sequence=1,
                )
            )
            s.commit()

        captured: dict[str, Any] = {}

        async def fake_submit(payload: dict[str, Any], *, settings: Any) -> Any:
            captured["payload"] = payload
            return {
                "success": True,
                "tao_external_job_id": "ext-train-root",
                "error": None,
            }

        monkeypatch.setattr(tao_job_service, "_submit_to_tao", fake_submit)
        monkeypatch.setattr(tao_job_service.sse_manager, "emit", AsyncMock())

        result = await submit_chain_job(PID, "root-train", settings=settings)
        assert result == "submitted"
        assert "parent_job_id" not in captured["payload"]

    @pytest.mark.asyncio
    async def test_parent_without_external_id_skips_injection(
        self, seeded, monkeypatch
    ):
        """If the parent TAOJob has no external id (e.g. parent never
        reached ``submitted``), ``submit_chain_job`` MUST NOT inject a
        bogus ``parent_job_id``. The chain advancement should not have
        triggered in that case, but defensively the field must be
        absent so the caller's safeguards detect the missing parent.
        """
        engine, settings = seeded
        _parent_id, child_id = self._seed_parent_and_child(
            engine, parent_status="not_started", parent_external_id=None
        )

        captured: dict[str, Any] = {}

        async def fake_submit(payload: dict[str, Any], *, settings: Any) -> Any:
            captured["payload"] = payload
            return {
                "success": True,
                "tao_external_job_id": "ext-eval-002",
                "error": None,
            }

        monkeypatch.setattr(tao_job_service, "_submit_to_tao", fake_submit)
        monkeypatch.setattr(tao_job_service.sse_manager, "emit", AsyncMock())

        result = await submit_chain_job(PID, child_id, settings=settings)
        assert result == "submitted"
        assert "parent_job_id" not in captured["payload"]


# ═══════════════════════════════════════════════════════════════════════════
# Cancel: suite bookkeeping parity with the poller's terminal-failure flow
# ═══════════════════════════════════════════════════════════════════════════


def _seed_suite_for_cancel(engine, chain_ids: list[str]) -> None:
    from vlm_feedback_loop.db.models.training_suite import TrainingSuite

    with Session(engine) as s:
        s.add(
            TrainingSuite(
                training_suite_id="ts-cancel",
                project_id=PID,
                idempotency_key="cancel-suite-1",
                guidance_id="g-1",
                training_preset="quick",
                export_field_mode="all",
                include_auto_labeled=False,
                training_dataset_export_id="de-t",
                evaluation_dataset_export_id="de-e",
                selected_student_base_model_config_ids=[MCID],
                quantization_schemes=["FP8_DYNAMIC"],
                chain_ids_ordered=chain_ids,
                status="running",
            )
        )
        s.commit()


class TestCancelSuiteBookkeeping:
    @pytest.mark.asyncio
    async def test_cancel_rolls_up_single_chain_suite_to_failed(
        self, seeded, monkeypatch
    ):
        """A user cancel must recompute the owning suite's status. Before
        the fix the suite stayed 'running' forever: the cancel path halted
        siblings but never rolled up, and with no pollable rows left the
        poller never revisited (observed live on two suites, 2026-07-14)."""
        engine, settings = seeded
        ids = _seed_chain_for_cancel(engine, running_status="running")
        _seed_suite_for_cancel(engine, ["chain-abc"])
        monkeypatch.setattr(
            tao_job_service,
            "_cancel_tao_external",
            AsyncMock(return_value={"success": True, "error": None}),
        )

        await cancel_tao_job(PID, ids[0], settings=settings)

        from vlm_feedback_loop.db.models.training_suite import TrainingSuite

        with Session(engine) as s:
            suite = (
                s.query(TrainingSuite).filter_by(training_suite_id="ts-cancel").one()
            )
            assert suite.status == "failed"
            assert suite.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_cross_advances_to_next_chain(self, seeded, monkeypatch):
        """Canceling chain 1's train must start chain 2 (chains are
        independent per-model pipelines), exactly as the poller's
        handle_terminal_failure does on a failure."""
        engine, settings = seeded
        ids = _seed_chain_for_cancel(engine, running_status="running")
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id="chain2-train",
                    project_id=PID,
                    student_base_model_config_id=MCID,
                    dataset_export_ids=[EXPORT_A],
                    action="train",
                    status="not_started",
                    training_backend="cosmos_rl_tao_vlm",
                    training_policy_type="sft",
                    job_config={},
                    tao_create_job_request={
                        "kind": "experiment",
                        "action": "train",
                        "specs": {},
                    },
                    chain_id="chain-def",
                    chain_sequence=1,
                )
            )
            s.commit()
        _seed_suite_for_cancel(engine, ["chain-abc", "chain-def"])

        monkeypatch.setattr(
            tao_job_service,
            "_cancel_tao_external",
            AsyncMock(return_value={"success": True, "error": None}),
        )
        submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await cancel_tao_job(PID, ids[0], settings=settings)

        submit_mock.assert_awaited_once()
        args, kwargs = submit_mock.await_args
        assert args == (PID, "chain2-train")

        from vlm_feedback_loop.db.models.training_suite import TrainingSuite

        with Session(engine) as s:
            suite = (
                s.query(TrainingSuite).filter_by(training_suite_id="ts-cancel").one()
            )
            # Chain 2 hasn't run yet (submit mocked) → suite must NOT be
            # prematurely failed while work remains.
            assert suite.status == "running"

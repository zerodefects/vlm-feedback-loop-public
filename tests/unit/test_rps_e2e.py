# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``rps_e2e.py`` quality/quantization driver.

The driver exposes flags that let it drive
both Cosmos Reason 2 base models, an arbitrary subset of
``VALID_QUANTIZATION_SCHEMES``, the inference-contract export-mode
selector, and an existing labeled project. These tests cover:

  * ``_parse_quantization_arg`` — the pure validator/normalizer.
  * ``_parse_args`` — flag defaults + acceptance of the new flags.
  * ``tao_validation.submit_training_suite`` — the kwargs threaded into
    ``training_suite_service.create_training_suite`` are correct for
    each (base_model × quantization × export_field_mode) combination.
  * ``--base-model 8b`` without ``--auto-provision-base-experiments``
    short-circuits with a clear ``SystemExit``.

The script depends on ``training_suite_service`` and a few SQLAlchemy
models, but every test here either runs in pure parser/normalizer mode
or monkeypatches the service-layer call site, so no live backend, TAO
endpoint, or workspace DB is required.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from vlm_feedback_loop.services import training_suite_service

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rps_e2e.py"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

tao_validation = importlib.import_module("tao_validation")


@pytest.fixture(scope="module")
def rps_e2e():
    """Load ``scripts/rps_e2e.py`` as a module.

    ``scripts/`` is not a Python package, so the directory is added to
    ``sys.path`` above before loading the executable as a module.
    """
    spec = importlib.util.spec_from_file_location("rps_e2e", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── _parse_quantization_arg ────────────────────────────────────────────


class TestParseQuantizationArg:
    def test_empty_returns_baseline(self, rps_e2e):
        assert rps_e2e._parse_quantization_arg("") == []
        assert rps_e2e._parse_quantization_arg("   ") == []

    def test_single_lowercase_uppercased(self, rps_e2e):
        assert rps_e2e._parse_quantization_arg("fp8_dynamic") == ["FP8_DYNAMIC"]

    def test_multiple_with_whitespace_normalized(self, rps_e2e):
        out = rps_e2e._parse_quantization_arg(" fp8_dynamic , w8a16 ")
        assert out == ["FP8_DYNAMIC", "W8A16"]

    def test_dedupes_preserving_order(self, rps_e2e):
        # First seen wins; case folds before dedup.
        out = rps_e2e._parse_quantization_arg("FP8_DYNAMIC,fp8_dynamic,W8A16")
        assert out == ["FP8_DYNAMIC", "W8A16"]

    def test_all_four_schemes_canonical_uppercase(self, rps_e2e):
        # Reflects the canonical VALID_QUANTIZATION_SCHEMES set in
        # services/training_suite_service.py.
        out = rps_e2e._parse_quantization_arg("fp8_dynamic,w8a8,w8a16,w4a16")
        assert sorted(out) == sorted(["FP8_DYNAMIC", "W8A8", "W8A16", "W4A16"])

    def test_rejects_unknown_scheme(self, rps_e2e):
        with pytest.raises(SystemExit) as exc:
            rps_e2e._parse_quantization_arg("int4")
        assert "INT4" in str(exc.value)
        assert "FP8_DYNAMIC" in str(exc.value)  # mentions valid set

    def test_rejects_partial_unknown_in_mixed_list(self, rps_e2e):
        with pytest.raises(SystemExit):
            rps_e2e._parse_quantization_arg("fp8_dynamic,bogus,w4a16")


def test_bundled_sample_is_deliberately_too_small_for_quality_gate(rps_e2e):
    """The 15-image first-run sample must not masquerade as the 372-image gate."""
    bundled_sample = REPO_ROOT / "deploy" / "example-images"

    with pytest.raises(SystemExit) as excinfo:
        rps_e2e._load_rps(bundled_sample)

    message = str(excinfo.value)
    assert "has only 5 images" in message
    assert "need at least 124" in message


# ── _parse_args ────────────────────────────────────────────────────────


class TestParseArgs:
    def test_8b_with_quantization(self, rps_e2e):
        ns = rps_e2e._parse_args(
            [
                "--rps-root",
                "/tmp/x",
                "--base-model",
                "8b",
                "--quantization",
                "w8a8,w4a16",
            ]
        )
        assert ns.base_model == "8b"
        assert ns.quantization == "w8a8,w4a16"

    def test_export_field_mode_core_only(self, rps_e2e):
        ns = rps_e2e._parse_args(
            ["--rps-root", "/tmp/x", "--export-field-mode", "core_only"]
        )
        assert ns.export_field_mode == "core_only"

    def test_project_id_passes_through(self, rps_e2e):
        pid = "deadbeef-feed-face-cafe-c0ffeedecade"
        ns = rps_e2e._parse_args(["--rps-root", "/tmp/x", "--project-id", pid])
        assert ns.project_id == pid

    def test_rejects_unknown_base_model(self, rps_e2e):
        with pytest.raises(SystemExit):
            rps_e2e._parse_args(["--rps-root", "/tmp/x", "--base-model", "12b"])

    def test_rejects_unknown_export_field_mode(self, rps_e2e):
        with pytest.raises(SystemExit):
            rps_e2e._parse_args(
                ["--rps-root", "/tmp/x", "--export-field-mode", "bogus"]
            )


# ── shared suite submission (monkeypatched service) ────────────────────


class _RecordingCreateTrainingSuite:
    """Records the kwargs each call site passes into the service."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, project_id, **kwargs):
        self.calls.append({"project_id": project_id, **kwargs})
        return {
            "training_suite_id": f"ts-{len(self.calls):03d}",
            "chains": [{"jobs": [{"action": "train"}, {"action": "evaluate"}]}],
        }


class TestSubmitTrainingSuite:
    def _assembly(self) -> dict:
        return {
            "project_id": "proj-001",
            "project_dir": Path("/tmp/proj-001"),
            "guidance_id": "guidance-001",
            "mc_2b_id": "mc-2b-id",
            "mc_8b_id": "mc-8b-id",
        }

    async def test_2b_baseline_only_kwargs(self, monkeypatch):
        rec = _RecordingCreateTrainingSuite()
        monkeypatch.setattr(training_suite_service, "create_training_suite", rec)
        await tao_validation.submit_training_suite(
            settings=object(),
            assembly=self._assembly(),
            base_model="2b",
            training_preset="high_quality",
            quantization_schemes=[],
            export_field_mode="all",
            idempotency_prefix="rps-e2e",
        )
        assert len(rec.calls) == 1
        call = rec.calls[0]
        assert call["project_id"] == "proj-001"
        assert call["student_base_model_config_ids"] == ["mc-2b-id"]
        assert call["quantization_schemes"] == []
        assert call["export_field_mode"] == "all"
        assert call["training_preset"] == "high_quality"
        # Idempotency key is still generated; just verify shape, not value.
        assert call["idempotency_key"].startswith("rps-e2e-")

    async def test_2b_with_quantization_kwargs(self, monkeypatch):
        rec = _RecordingCreateTrainingSuite()
        monkeypatch.setattr(training_suite_service, "create_training_suite", rec)
        await tao_validation.submit_training_suite(
            settings=object(),
            assembly=self._assembly(),
            base_model="2b",
            training_preset="high_quality",
            quantization_schemes=["FP8_DYNAMIC", "W8A16"],
            export_field_mode="all",
            idempotency_prefix="rps-e2e",
        )
        call = rec.calls[0]
        assert call["student_base_model_config_ids"] == ["mc-2b-id"]
        assert call["quantization_schemes"] == ["FP8_DYNAMIC", "W8A16"]

    async def test_8b_uses_mc_8b_id(self, monkeypatch):
        rec = _RecordingCreateTrainingSuite()
        monkeypatch.setattr(training_suite_service, "create_training_suite", rec)
        await tao_validation.submit_training_suite(
            settings=object(),
            assembly=self._assembly(),
            base_model="8b",
            training_preset="high_quality",
            quantization_schemes=["W8A8", "W4A16"],
            export_field_mode="all",
            idempotency_prefix="rps-e2e",
        )
        call = rec.calls[0]
        assert call["student_base_model_config_ids"] == ["mc-8b-id"]
        assert call["quantization_schemes"] == ["W8A8", "W4A16"]

    async def test_core_only_export_field_mode_threads_through(self, monkeypatch):
        # The Student's training_inference_contract
        # is derived from this field by inference_contract_resolver.py.
        rec = _RecordingCreateTrainingSuite()
        monkeypatch.setattr(training_suite_service, "create_training_suite", rec)
        await tao_validation.submit_training_suite(
            settings=object(),
            assembly=self._assembly(),
            base_model="2b",
            training_preset="high_quality",
            quantization_schemes=[],
            export_field_mode="core_only",
            idempotency_prefix="rps-e2e",
        )
        call = rec.calls[0]
        assert call["export_field_mode"] == "core_only"

    async def test_service_returning_string_raises(self, monkeypatch):
        async def fake_create(*args, **kwargs):
            return "tao_dataset_upload_failed"  # mimic service error string

        monkeypatch.setattr(
            training_suite_service, "create_training_suite", fake_create
        )
        with pytest.raises(SystemExit) as exc:
            await tao_validation.submit_training_suite(
                settings=object(),
                assembly=self._assembly(),
                base_model="2b",
                training_preset="high_quality",
                quantization_schemes=[],
                export_field_mode="all",
                idempotency_prefix="rps-e2e",
            )
        assert "tao_dataset_upload_failed" in str(exc.value)


# ── _amain validation gates (without running the full pipeline) ───────────


class TestAmainValidationGates:
    """Tests narrowly target the up-front argv validation in ``_amain``.

    Mocks all I/O-bound dependencies so we never touch TAO, the workspace,
    or the network. Each test exercises one validation branch.
    """

    async def test_8b_without_provisioning_exits(self, rps_e2e, monkeypatch):
        # Avoid touching the workspace / settings loader / TAO calls.
        monkeypatch.setattr(rps_e2e, "load_settings", lambda: object())

        # Patch sys.argv so _parse_args() picks up the deliberate combo:
        # --base-model 8b without --auto-provision-base-experiments.
        # (rps_root must exist or the script raises before reaching the
        # 8b-validation branch — we use a real tmp-but-irrelevant path.)
        argv = [
            "rps_e2e.py",
            "--rps-root",
            "/tmp",
            "--base-model",
            "8b",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as exc:
            await rps_e2e._amain()
        msg = str(exc.value)
        assert "--base-model 8b" in msg
        assert "--auto-provision-base-experiments" in msg


# ── HTTP-mode helpers (lookup mode against a running backend) ─────────────


import httpx  # noqa: E402 — helpers below this line; keep import scoped to the helper section


def _mock_http_client(handlers: dict):
    """Build an httpx.AsyncClient backed by a MockTransport whose
    behavior is described by ``handlers``: a mapping of
    ``(method, exact_path)`` → callable returning ``httpx.Response``.

    Routes are matched by exact path equality (after stripping the
    base URL host), so e.g. ``/v1/projects/p1`` and
    ``/v1/projects/p1/model_configs`` are distinct routes.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        method = request.method.upper()
        path = request.url.path
        for (m, exact_path), fn in handlers.items():
            if m.upper() == method and path == exact_path:
                return fn(request)
        return httpx.Response(
            404,
            json={"error": f"no handler for {method} {path} (q={request.url.query!r})"},
        )

    transport = httpx.MockTransport(handle)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


class TestLookupExistingProjectHttp:
    async def test_happy_path(self, rps_e2e):
        handlers = {
            ("GET", "/v1/projects/p1"): lambda r: httpx.Response(
                200,
                json={
                    "project_id": "p1",
                    "active_guidance_id": "g1",
                    "name": "test",
                },
            ),
            ("GET", "/v1/projects/p1/model_configs"): lambda r: httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "model_config_id": "mc-2b",
                            "model_name": rps_e2e.MODEL_NAME_2B,
                        },
                        {
                            "model_config_id": "mc-8b",
                            "model_name": rps_e2e.MODEL_NAME_8B,
                        },
                    ]
                },
            ),
        }
        async with _mock_http_client(handlers) as client:
            assembly = await rps_e2e._lookup_existing_project_http(client, "p1")
        assert assembly == {
            "project_id": "p1",
            "project_dir": None,
            "guidance_id": "g1",
            "mc_2b_id": "mc-2b",
            "mc_8b_id": "mc-8b",
        }

    async def test_no_active_guidance_rejects(self, rps_e2e):
        handlers = {
            ("GET", "/v1/projects/p1"): lambda r: httpx.Response(
                200, json={"project_id": "p1", "active_guidance_id": None}
            ),
        }
        async with _mock_http_client(handlers) as client:
            with pytest.raises(SystemExit) as exc:
                await rps_e2e._lookup_existing_project_http(client, "p1")
        assert "no active guidance" in str(exc.value)

    async def test_missing_cosmos_seed_rejects(self, rps_e2e):
        handlers = {
            ("GET", "/v1/projects/p1"): lambda r: httpx.Response(
                200, json={"project_id": "p1", "active_guidance_id": "g1"}
            ),
            ("GET", "/v1/projects/p1/model_configs"): lambda r: httpx.Response(
                200,
                json={
                    "items": [
                        # Only 2B, no 8B → must reject (driver needs both seeded)
                        {
                            "model_config_id": "mc-2b",
                            "model_name": rps_e2e.MODEL_NAME_2B,
                        }
                    ]
                },
            ),
        }
        async with _mock_http_client(handlers) as client:
            with pytest.raises(SystemExit) as exc:
                await rps_e2e._lookup_existing_project_http(client, "p1")
        assert "missing seeded Cosmos ModelConfigs" in str(exc.value)


class TestSubmitSuiteHttp:
    async def test_happy_path(self, rps_e2e):
        captured: dict = {}

        def post_handler(r: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(r.content.decode())
            return httpx.Response(
                201,
                json={
                    "training_suite_id": "ts-001",
                    "chains": [{"jobs": [{"action": "train"}, {"action": "evaluate"}]}],
                },
            )

        handlers = {("POST", "/v1/projects/p1/training_suites"): post_handler}
        async with _mock_http_client(handlers) as client:
            suite = await rps_e2e._submit_suite_http(
                client,
                project_id="p1",
                base_model="2b",
                mc_id="mc-2b",
                training_preset="high_quality",
                quantization_schemes=["FP8_DYNAMIC", "W8A16"],
                export_field_mode="all",
                idempotency_key="key-001",
            )
        assert suite["training_suite_id"] == "ts-001"
        # Verify the canonical body — every field threaded through correctly.
        body = captured["body"]
        assert body["student_base_model_config_ids"] == ["mc-2b"]
        assert body["training_preset"] == "high_quality"
        assert body["include_auto_labeled"] is False
        assert body["export_field_mode"] == "all"
        assert body["quantization_schemes"] == ["FP8_DYNAMIC", "W8A16"]
        assert body["idempotency_key"] == "key-001"

    async def test_non_201_raises(self, rps_e2e):
        handlers = {
            ("POST", "/v1/projects/p1/training_suites"): lambda r: httpx.Response(
                409, text='{"detail":"tao_dataset_upload_failed"}'
            ),
        }
        async with _mock_http_client(handlers) as client:
            with pytest.raises(SystemExit) as exc:
                await rps_e2e._submit_suite_http(
                    client,
                    project_id="p1",
                    base_model="2b",
                    mc_id="mc-2b",
                    training_preset="high_quality",
                    quantization_schemes=[],
                    export_field_mode="all",
                    idempotency_key="key-001",
                )
        assert "409" in str(exc.value)


# ── Idempotency key derivation ─────────────────────────────────────────────


class TestDeriveIdempotencyKey:
    def test_same_inputs_same_key(self, rps_e2e):
        a = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=["FP8_DYNAMIC", "W8A16"],
            export_field_mode="all",
            training_preset="standard",
        )
        b = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=["FP8_DYNAMIC", "W8A16"],
            export_field_mode="all",
            training_preset="standard",
        )
        assert a == b
        assert a.startswith("rps-e2e-2b-")

    def test_different_base_model_different_key(self, rps_e2e):
        a = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=["FP8_DYNAMIC"],
            export_field_mode="all",
            training_preset="standard",
        )
        b = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="8b",
            quantization_schemes=["FP8_DYNAMIC"],
            export_field_mode="all",
            training_preset="standard",
        )
        assert a != b

    def test_different_quantization_different_key(self, rps_e2e):
        a = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=["FP8_DYNAMIC"],
            export_field_mode="all",
            training_preset="standard",
        )
        b = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=["W4A16"],
            export_field_mode="all",
            training_preset="standard",
        )
        assert a != b

    def test_different_export_mode_different_key(self, rps_e2e):
        # The core_only export mode must collide with neither the production
        # 2B nor 2B-with-quantization keys.
        a = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=[],
            export_field_mode="all",
            training_preset="standard",
        )
        b = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=[],
            export_field_mode="core_only",
            training_preset="standard",
        )
        assert a != b

    def test_different_project_different_key(self, rps_e2e):
        a = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=[],
            export_field_mode="all",
            training_preset="standard",
        )
        b = rps_e2e._derive_idempotency_key(
            project_id="p2",
            base_model="2b",
            quantization_schemes=[],
            export_field_mode="all",
            training_preset="standard",
        )
        assert a != b

    def test_different_training_preset_different_key(self, rps_e2e):
        """Distinct preset values must produce distinct keys (the
        preset materially changes the chain). Evidence: if the preset is
        excluded from the key, canceling a ``standard``-preset chain and
        re-submitting with ``high_quality`` re-POSTs into the canceled
        standard suite, blocking the recovery path. Lock the contract by
        asserting two presets on otherwise-identical args yield
        different keys.
        """
        a = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=["FP8_DYNAMIC", "W8A16"],
            export_field_mode="all",
            training_preset="standard",
        )
        b = rps_e2e._derive_idempotency_key(
            project_id="p1",
            base_model="2b",
            quantization_schemes=["FP8_DYNAMIC", "W8A16"],
            export_field_mode="all",
            training_preset="high_quality",
        )
        assert a != b, "distinct training_preset must produce distinct idempotency keys"


# ── _amain_lookup_http argv validations (no I/O, no live calls) ────────────


class TestAmainLookupHttpValidations:
    async def test_provision_flag_in_lookup_mode_rejected(self, rps_e2e, monkeypatch):
        # --auto-provision-base-experiments is a deployment-level operation
        # that conflicts with the running backend's project locks.
        # The script must reject the combination.
        argv = [
            "rps_e2e.py",
            "--rps-root",
            "/tmp",
            "--project-id",
            "p1",
            "--auto-provision-base-experiments",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as exc:
            await rps_e2e._amain()
        assert "lookup mode" in str(exc.value)
        assert "tao-pull-base-experiments" in str(exc.value)

    async def test_training_suite_id_without_project_id_rejected(
        self, rps_e2e, monkeypatch
    ):
        argv = [
            "rps_e2e.py",
            "--rps-root",
            "/tmp",
            "--training-suite-id",
            "ts-x",
        ]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as exc:
            await rps_e2e._amain()
        assert "--project-id" in str(exc.value)


# ── BASE_MODEL_NAMES public mapping (sanity for downstream consumers) ──


def test_base_model_names_mapping_is_canonical(rps_e2e):
    assert rps_e2e.BASE_MODEL_NAMES == {
        "2b": "nvidia/cosmos-reason2-2b",
        "8b": "nvidia/cosmos-reason2-8b",
    }


# ── shared training outcome predicate ──────────────────────────────────


def _job(action: str, status: str, *, chain_sequence: int = 1) -> dict:
    return {
        "action": action,
        "status": status,
        "chain_sequence": chain_sequence,
        "tao_job_id": f"j-{action}-{chain_sequence}-{status}",
    }


class TestComputeOutcomeStrict:
    """``accept_eval_failure=False`` keeps the strict last-evaluate-wins predicate."""

    def test_baseline_evaluate_succeeded_returns_succeeded(self):
        jobs = [
            _job("train", "succeeded", chain_sequence=1),
            _job("evaluate", "succeeded", chain_sequence=2),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=False)
            == "succeeded"
        )

    def test_evaluate_failed_returns_failed(self):
        jobs = [
            _job("train", "succeeded", chain_sequence=1),
            _job("evaluate", "failed", chain_sequence=2),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=False)
            == "failed"
        )

    def test_train_failed_returns_failed(self):
        jobs = [_job("train", "failed", chain_sequence=1)]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=False)
            == "failed"
        )

    def test_still_running_returns_pending(self):
        jobs = [
            _job("train", "succeeded", chain_sequence=1),
            _job("evaluate", "running", chain_sequence=2),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=False)
            == "pending"
        )


class TestComputeOutcomeChainIsolation:
    """``accept_eval_failure=True`` is the closing-smoke contract."""

    def test_chain_isolated_succeeded_treats_eval_failure_as_pass(self):
        """The exact spec'd closing-smoke shape: train ok,
        evaluate-baseline fails (Qwen3-VL gap), quantize succeeds,
        evaluate-quantized fails. Under chain isolation this MUST be
        treated as successful — quality_status flips downstream via
        NIM-eval-as-quality-fallback."""
        jobs = [
            _job("train", "succeeded", chain_sequence=1),
            _job("evaluate", "failed", chain_sequence=2),
            _job("quantize", "succeeded", chain_sequence=3),
            _job("evaluate", "failed", chain_sequence=4),
            _job("quantize", "succeeded", chain_sequence=5),
            _job("evaluate", "failed", chain_sequence=6),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=True)
            == "succeeded"
        )

    def test_baseline_only_chain_passes_when_train_succeeded(self):
        jobs = [
            _job("train", "succeeded", chain_sequence=1),
            _job("evaluate", "failed", chain_sequence=2),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=True)
            == "succeeded"
        )

    def test_train_failed_still_fails(self):
        """Chain isolation does NOT rescue a failed train."""
        jobs = [
            _job("train", "failed", chain_sequence=1),
            _job("evaluate", "failed", chain_sequence=2),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=True)
            == "failed"
        )

    def test_all_quantize_failed_fails(self):
        """If quantization was requested and every quantize job failed,
        no quantized Student is registered → run cannot satisfy the
        full-quant closing-smoke goal."""
        jobs = [
            _job("train", "succeeded", chain_sequence=1),
            _job("evaluate", "failed", chain_sequence=2),
            _job("quantize", "failed", chain_sequence=3),
            _job("evaluate", "failed", chain_sequence=4),
            _job("quantize", "failed", chain_sequence=5),
            _job("evaluate", "failed", chain_sequence=6),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=True)
            == "failed"
        )

    def test_one_of_two_quantize_succeeded_passes(self):
        """Partial quantization success (≥1 quantize ok) is enough."""
        jobs = [
            _job("train", "succeeded", chain_sequence=1),
            _job("evaluate", "failed", chain_sequence=2),
            _job("quantize", "succeeded", chain_sequence=3),
            _job("evaluate", "failed", chain_sequence=4),
            _job("quantize", "failed", chain_sequence=5),
            _job("evaluate", "failed", chain_sequence=6),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=True)
            == "succeeded"
        )

    def test_jobs_still_running_returns_pending(self):
        """Polling MUST keep going past evaluate failures while quantize
        is still running — an "any failed → fatal" predicate would
        wrongly abort the chain here."""
        jobs = [
            _job("train", "succeeded", chain_sequence=1),
            _job("evaluate", "failed", chain_sequence=2),
            _job("quantize", "running", chain_sequence=3),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=True)
            == "pending"
        )

    def test_canceled_treated_as_terminal(self):
        """``canceled`` and ``deleted`` are terminal job states; a
        chain with a canceled non-train job that still has train+quantize
        succeeded is a pass."""
        jobs = [
            _job("train", "succeeded", chain_sequence=1),
            _job("evaluate", "canceled", chain_sequence=2),
            _job("quantize", "succeeded", chain_sequence=3),
            _job("evaluate", "canceled", chain_sequence=4),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=True)
            == "succeeded"
        )

    def test_supports_orm_objects_and_dicts(self):
        """Service-mode poll passes TAOJob ORM rows; HTTP-mode passes
        dicts. The helper MUST accept both."""

        class _OrmShape:
            def __init__(self, action: str, status: str) -> None:
                self.action = action
                self.status = status

        jobs = [
            _OrmShape("train", "succeeded"),
            _OrmShape("evaluate", "failed"),
            _OrmShape("quantize", "succeeded"),
            _OrmShape("evaluate", "failed"),
        ]
        assert (
            tao_validation.compute_training_outcome(jobs, accept_eval_failure=True)
            == "succeeded"
        )


# ── --accept-eval-failure CLI surface ───────────────────────────────────


class TestAcceptEvalFailureFlag:
    def test_flag_sets_true(self, rps_e2e):
        ns = rps_e2e._parse_args(["--rps-root", "/tmp/x", "--accept-eval-failure"])
        assert ns.accept_eval_failure is True

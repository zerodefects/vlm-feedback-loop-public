# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``services.tao_failure_classifier``.

The conservative gate must:

  - match the live cosmos-rl 6.26.3 + Qwen3-VL-dense gap signature,
  - NOT match generic Python error tokens (``AssertionError``,
    ``RuntimeError``, etc.) on their own — those would fire on legitimate
    dataset / OOM / config failures and silently paper them over,
  - tolerate empty/None evidence (return False — cannot classify),
  - read evidence from any of: ``error_ref``, ``status_reason``,
    ``outputs.tao_logs_text``, or ``outputs.tao_logs_ref`` file contents.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services import tao_failure_classifier as tfc

# ── first_matching_pattern ───────────────────────────────────────────────────


class TestFirstMatchingPattern:
    def test_matches_qwen3vl_for_conditional_generation(self):
        text = "Stack:\n  ... ValueError: Unsupported model type: qwen3_vl\n"
        assert tfc.first_matching_pattern(text)

    def test_matches_qwen2_5_vl_fallback_path(self):
        text = "vllm/model_executor/models/qwen2_5_vl.py:1561 in load_weights"
        assert tfc.first_matching_pattern(text)

    def test_matches_vocab_parallel_embedding_assertion(self):
        text = (
            "AssertionError\n"
            "  File '/opt/.../vllm/.../vocab_parallel_embedding.py', line 457"
        )
        assert tfc.first_matching_pattern(text)

    def test_matches_qwen3_vl_nested_embed_key(self):
        text = "Loading weight model.language_model.embed_tokens.weight ..."
        assert tfc.first_matching_pattern(text)

    def test_matches_lora_adapter_base_model_prefix(self):
        """LoRA adapter-only checkpoint fed to the flat vLLM loader.

        Live signature from the GT-nano LoRA chain's in-chain evaluate
        (2026-07-15): the adapter safetensors carry ``base_model.model.*``
        keys the loader cannot resolve. The merged adapter served and
        scored 0.767 (120/120 parse) through the §9.5 NIM path — an
        upstream loader gap, so a passing NIM eval may satisfy the
        quality gate.
        """
        text = (
            "ValueError: There is no module or parameter named "
            "'base_model' in Qwen2_5_VLForConditionalGeneration"
        )
        assert tfc.first_matching_pattern(text)

    def test_matches_weights_not_initialized_signature(self):
        """Cosmos-Reason2-8B failure signature on TAO cosmos-rl-evaluate.

        Live signature from an 8B chain's evaluate job. The same
        checkpoint loaded cleanly via NIM 1.6.0 in 1.86 s and served
        83 / 84 examples at 83.1 % exact-match — the TAO-side failure
        is genuinely an upstream loader gap, not a real model issue,
        so this pattern belongs in the narrow NIM-eval fallback
        gate.
        """
        text = (
            "ValueError: Following weights were not initialized from "
            "checkpoint: {'visual.blocks.21.attn.qkv.weight', "
            "'language_model.model.layers.3.mlp.gate_up_proj.weight', "
            "'visual.deepstack_merger_list.2.linear_fc1.bias', "
            "'visual.blocks.25.norm1.weight', ...}"
        )
        assert tfc.first_matching_pattern(text)

    def test_does_not_match_plain_assertion_error(self):
        text = (
            "Traceback (most recent call last):\n"
            "  AssertionError: dataset shape mismatch (got 0 examples)"
        )
        assert not tfc.first_matching_pattern(text)

    def test_does_not_match_oom_error(self):
        text = (
            "torch.cuda.OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 24.00 GiB."
        )
        assert not tfc.first_matching_pattern(text)

    def test_does_not_match_dataset_error(self):
        text = (
            "RuntimeError: cosmos_rl: train_dataset annotation_path is invalid: "
            "expected JSON array, got dict."
        )
        assert not tfc.first_matching_pattern(text)

    def test_does_not_match_empty_text(self):
        assert not tfc.first_matching_pattern("")
        assert not tfc.first_matching_pattern(None)


# ── collect_failure_evidence ─────────────────────────────────────────────────


class TestCollectFailureEvidence:
    def _job(self, **kwargs) -> TAOJob:
        defaults = {
            "tao_job_id": generate_uuid4(),
            "project_id": generate_uuid4(),
            "student_base_model_config_id": generate_uuid4(),
            "action": "evaluate",
            "training_backend": "cosmos_rl_tao_vlm",
            "dataset_export_ids": [],
            "job_config": {},
            "tao_create_job_request": {},
            "status": "failed",
        }
        defaults.update(kwargs)
        return TAOJob(**defaults)

    def test_truncation_keeps_log_tail_where_tracebacks_live(self):
        """A signature past the max_chars boundary must survive truncation.

        Container logs end with the terminal traceback; the old head-keeping
        cut silently dropped any signature past 32 KB (observed live: the
        GT-nano evaluate's ValueError at byte 45 618 of a 59 KB log).
        """
        filler = "x" * 50_000
        signature = (
            "ValueError: There is no module or parameter named "
            "'base_model' in Qwen2_5_VLForConditionalGeneration"
        )
        job = self._job(outputs={"tao_logs_text": filler + "\n" + signature})
        evidence = tfc.collect_failure_evidence(job)
        assert len(evidence) <= 32_768
        assert tfc.first_matching_pattern(evidence)

    def test_concatenates_error_refs_and_chain_halted_reason(self):
        job = self._job(
            error_ref="/tmp/some-error-file.json",
            chain_halted_reason="halted_after_train_failure",
        )
        evidence = tfc.collect_failure_evidence(job)
        assert "/tmp/some-error-file.json" in evidence
        assert "halted_after_train_failure" in evidence

    def test_includes_inline_outputs_log_text(self):
        job = self._job(
            outputs={
                "tao_logs_text": "qwen2_5_vl.py:1561 → AssertionError",
            },
        )
        evidence = tfc.collect_failure_evidence(job)
        assert "qwen2_5_vl.py" in evidence

    def test_reads_log_file_when_outputs_carries_a_path(self, tmp_path: Path):
        log_file = tmp_path / "logs.txt"
        log_file.write_text("vocab_parallel_embedding.py:457\nAssertionError")
        job = self._job(outputs={"tao_logs_ref": str(log_file)})
        evidence = tfc.collect_failure_evidence(job)
        assert "vocab_parallel_embedding.py" in evidence

    def test_handles_missing_outputs(self):
        job = self._job(
            outputs=None,
            error_ref=None,
            poll_error_ref=None,
            chain_halted_reason=None,
        )
        evidence = tfc.collect_failure_evidence(job)
        assert evidence == ""

    def test_truncates_to_max_chars(self):
        big = "x" * 100_000
        job = self._job(outputs={"tao_logs_text": big})
        evidence = tfc.collect_failure_evidence(job, max_chars=4096)
        assert len(evidence) <= 4096


# ── matches_known_loader_gap (DB-backed) ─────────────────────────────────────


def _seed_eval_failure(
    session: Session,
    *,
    project_id: str,
    train_tao_job_id: str,
    log_text: str,
    status: str = "failed",
) -> TAOJob:
    job = TAOJob(
        tao_job_id=generate_uuid4(),
        project_id=project_id,
        student_base_model_config_id=generate_uuid4(),
        parent_tao_job_id=train_tao_job_id,
        action="evaluate",
        chain_id="chain-1",
        chain_sequence=2,
        training_backend="cosmos_rl_tao_vlm",
        dataset_export_ids=[],
        job_config={},
        tao_create_job_request={},
        status=status,
        tao_status_raw="Error",
        tao_external_job_id=generate_uuid4(),
        completed_at=utc_now(),
        outputs={"tao_logs_text": log_text},
    )
    session.add(job)
    session.commit()
    return job


class TestMatchesKnownLoaderGap:
    def test_returns_false_when_no_failed_eval_exists(self, test_app_client):
        from vlm_feedback_loop.services.project_service import get_project_engine

        resp = test_app_client.post(
            "/v1/projects",
            json={"name": "tfc-no-eval", "description": ""},
        )
        pid = resp.json()["project_id"]
        from vlm_feedback_loop.routers.projects import get_current_settings

        settings = test_app_client.app.dependency_overrides[get_current_settings]()
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as session:
            matched, sig = tfc.matches_known_loader_gap(
                session,
                project_id=pid,
                student_train_tao_job_id="nonexistent-train-id",
            )
        assert matched is False
        assert sig is None

    def test_returns_true_on_qwen3_vl_signature(self, test_app_client):
        from vlm_feedback_loop.services.project_service import get_project_engine

        resp = test_app_client.post(
            "/v1/projects",
            json={"name": "tfc-loader-gap", "description": ""},
        )
        pid = resp.json()["project_id"]
        from vlm_feedback_loop.routers.projects import get_current_settings

        settings = test_app_client.app.dependency_overrides[get_current_settings]()
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        train_id = "train-job-1"
        with Session(engine) as session:
            _seed_eval_failure(
                session,
                project_id=pid,
                train_tao_job_id=train_id,
                log_text=(
                    "vllm/model_executor/models/qwen2_5_vl.py:1561\n"
                    "AssertionError: vocab_parallel_embedding shape mismatch"
                ),
            )
            matched, sig = tfc.matches_known_loader_gap(
                session,
                project_id=pid,
                student_train_tao_job_id=train_id,
            )
        assert matched is True
        # The classifier returns the FIRST matching pattern in declaration
        # order; multiple patterns may legitimately fire.
        assert sig in tfc.MODEL_LOADER_FAILURE_PATTERNS

    def test_returns_false_on_oom_failure(self, test_app_client):
        from vlm_feedback_loop.services.project_service import get_project_engine

        resp = test_app_client.post(
            "/v1/projects",
            json={"name": "tfc-oom", "description": ""},
        )
        pid = resp.json()["project_id"]
        from vlm_feedback_loop.routers.projects import get_current_settings

        settings = test_app_client.app.dependency_overrides[get_current_settings]()
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        train_id = "train-job-2"
        with Session(engine) as session:
            _seed_eval_failure(
                session,
                project_id=pid,
                train_tao_job_id=train_id,
                log_text="torch.cuda.OutOfMemoryError: CUDA out of memory.",
            )
            matched, sig = tfc.matches_known_loader_gap(
                session,
                project_id=pid,
                student_train_tao_job_id=train_id,
            )
        assert matched is False
        assert sig is None

    def test_canceled_evaluate_also_eligible_for_classification(self, test_app_client):
        """``canceled`` is treated as a terminal-non-success state for
        classification (chain halted upstream)."""
        from vlm_feedback_loop.services.project_service import get_project_engine

        resp = test_app_client.post(
            "/v1/projects",
            json={"name": "tfc-canceled", "description": ""},
        )
        pid = resp.json()["project_id"]
        from vlm_feedback_loop.routers.projects import get_current_settings

        settings = test_app_client.app.dependency_overrides[get_current_settings]()
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        train_id = "train-job-3"
        with Session(engine) as session:
            _seed_eval_failure(
                session,
                project_id=pid,
                train_tao_job_id=train_id,
                log_text="qwen2_5_vl.py:1561 AssertionError",
                status="canceled",
            )
            matched, sig = tfc.matches_known_loader_gap(
                session,
                project_id=pid,
                student_train_tao_job_id=train_id,
            )
        assert matched is True
        assert sig is not None

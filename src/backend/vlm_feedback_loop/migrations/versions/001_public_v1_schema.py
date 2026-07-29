# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public v1 project database baseline.

Creates the complete project-scoped schema for a fresh public installation.
The pre-release development migration chain was intentionally squashed before
the first public release; databases from that private lineage are unsupported.

Revision ID: v1_0001
Revises:
Create Date: 2026-07-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import sqlite

revision = "v1_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column("audit_event_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("event_type", sa.VARCHAR(), nullable=False),
        sa.Column("example_key", sa.VARCHAR(), nullable=True),
        sa.Column("event_data", sqlite.JSON(), nullable=False),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.PrimaryKeyConstraint("audit_event_id"),
    )
    op.create_table(
        "dataset_exports",
        sa.Column("dataset_export_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("dataset_intent", sa.VARCHAR(), nullable=False),
        sa.Column("export_field_mode", sa.VARCHAR(), nullable=False),
        sa.Column("guidance_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("label_tier_filter", sa.VARCHAR(), nullable=False),
        sa.Column("selection_definition_snapshot", sqlite.JSON(), nullable=False),
        sa.Column("artifact_refs", sqlite.JSON(), nullable=True),
        sa.Column("manifest_ref", sa.VARCHAR(), nullable=True),
        sa.Column("example_count", sa.INTEGER(), nullable=False),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("dataset_upload_ref", sa.VARCHAR(), nullable=True),
        sa.Column("dataset_upload_uri", sa.VARCHAR(), nullable=True),
        sa.Column(
            "status",
            sa.VARCHAR(),
            server_default=sa.text("'completed'"),
            nullable=False,
        ),
        sa.Column("status_reason", sa.VARCHAR(), nullable=True),
        sa.Column("progress", sqlite.JSON(), nullable=True),
        sa.Column("started_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("completed_at", sa.VARCHAR(length=24), nullable=True),
        sa.PrimaryKeyConstraint("dataset_export_id"),
    )
    op.create_table(
        "examples",
        sa.Column("example_key", sa.VARCHAR(), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("storage_ref", sa.VARCHAR(), nullable=False),
        sa.Column("ingested_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("source_metadata", sqlite.JSON(), nullable=False),
        sa.Column(
            "state", sa.VARCHAR(), server_default=sa.text("'Unlabeled'"), nullable=False
        ),
        sa.Column("phash", sa.VARCHAR(), nullable=True),
        sa.Column("omitted_source", sa.VARCHAR(), nullable=True),
        sa.Column("omitted_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column(
            "clip_embedding_present",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("clip_embedding_dim", sa.INTEGER(), nullable=True),
        sa.Column("clip_embedding_model_id", sa.VARCHAR(), nullable=True),
        sa.Column("embedding_provider", sa.VARCHAR(), nullable=True),
        sa.Column("prior_verified_label_ref", sa.VARCHAR(), nullable=True),
        sa.Column("prior_verified_outcome", sa.VARCHAR(), nullable=True),
        sa.PrimaryKeyConstraint("example_key"),
    )
    op.create_index("ix_examples_state", "examples", ["state"], unique=False)
    op.create_index(
        "ux_examples_project_key", "examples", ["project_id", "example_key"], unique=1
    )
    op.create_table(
        "guidances",
        sa.Column("guidance_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("version_number", sa.INTEGER(), nullable=False),
        sa.Column("description", sa.VARCHAR(), nullable=False),
        sa.Column("schema", sqlite.JSON(), nullable=False),
        sa.Column(
            "rules", sa.VARCHAR(), server_default=sa.text("('')"), nullable=False
        ),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column(
            "semantic_core_change_from_guidance_id",
            sa.VARCHAR(length=36),
            nullable=True,
        ),
        sa.Column("schema_change_summary", sqlite.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("guidance_id"),
        sa.UniqueConstraint(
            "project_id",
            "version_number",
            name="ux_guidances_project_version",
        ),
    )
    op.create_table(
        "labels",
        sa.Column("label_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("example_key", sa.VARCHAR(), nullable=False),
        sa.Column("label_status", sa.VARCHAR(), nullable=False),
        sa.Column("guidance_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("inference_invocation_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("label_json", sqlite.JSON(), nullable=False),
        sa.Column("labeled_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("verified_outcome", sa.VARCHAR(), nullable=True),
        sa.Column("verified_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("edited_core_fields", sqlite.JSON(), nullable=True),
        sa.Column("edited_aux_fields", sqlite.JSON(), nullable=True),
        sa.Column("rationale_source", sa.VARCHAR(), nullable=True),
        sa.Column(
            "rationale_regeneration_invocation_id", sa.VARCHAR(length=36), nullable=True
        ),
        sa.Column("batch_label_run_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("pool_assignment", sa.VARCHAR(), nullable=True),
        sa.PrimaryKeyConstraint("label_id"),
    )
    op.create_index(
        "ix_labels_example_key_status",
        "labels",
        ["example_key", "label_status"],
        unique=False,
    )
    op.create_index(
        "ux_labels_auto_labeled_example",
        "labels",
        ["project_id", "example_key"],
        unique=1,
        sqlite_where=sa.text("label_status = 'auto_labeled'"),
    )
    op.create_table(
        "local_nim_deployments",
        sa.Column("local_nim_deployment_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("model_config_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("role", sa.VARCHAR(), nullable=False),
        sa.Column("nim_container_image", sa.VARCHAR(), nullable=False),
        sa.Column("container_name", sa.VARCHAR(), nullable=False),
        sa.Column("container_id", sa.VARCHAR(), nullable=True),
        sa.Column("host_port", sa.INTEGER(), nullable=False),
        sa.Column("endpoint_url", sa.VARCHAR(), nullable=False),
        sa.Column("gpu_assignment", sa.VARCHAR(), nullable=False),
        sa.Column(
            "status", sa.VARCHAR(), server_default=sa.text("'starting'"), nullable=False
        ),
        sa.Column("status_reason", sa.VARCHAR(), nullable=True),
        sa.Column("deployed_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("stopped_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("student_model_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("checkpoint_mount_path", sa.VARCHAR(), nullable=True),
        sa.Column("nim_served_model_name", sa.VARCHAR(), nullable=True),
        sa.Column("nim_model_name_path", sa.VARCHAR(), nullable=True),
        sa.Column("precision_method", sa.VARCHAR(), nullable=True),
        sa.Column("displaced_by_deployment_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("displaced_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column(
            "activate_on_success",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("local_nim_deployment_id"),
    )
    op.create_table(
        "model_configs",
        sa.Column("model_config_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("endpoint_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("model_name", sa.VARCHAR(), nullable=False),
        sa.Column("context_window_tokens", sa.INTEGER(), nullable=False),
        sa.Column("eligible_roles", sqlite.JSON(), nullable=False),
        sa.Column("supports_image_input", sa.BOOLEAN(), nullable=False),
        sa.Column(
            "structured_generation_support",
            sa.VARCHAR(),
            server_default=sa.text("'unknown'"),
            nullable=False,
        ),
        sa.Column(
            "thinking_toggle_mode",
            sa.VARCHAR(),
            server_default=sa.text("'none'"),
            nullable=False,
        ),
        sa.Column(
            "thinking_toggle_support",
            sa.VARCHAR(),
            server_default=sa.text("'unknown'"),
            nullable=False,
        ),
        sa.Column(
            "visual_budget_mode",
            sa.VARCHAR(),
            server_default=sa.text("'none'"),
            nullable=False,
        ),
        sa.Column(
            "visual_budget_support",
            sa.VARCHAR(),
            server_default=sa.text("'unknown'"),
            nullable=False,
        ),
        sa.Column("model_quantization", sa.VARCHAR(), nullable=True),
        sa.Column("nim_model_profile", sa.VARCHAR(), nullable=True),
        sa.Column("nim_profile_metadata", sqlite.JSON(), nullable=True),
        sa.Column("local_deploy_metadata", sqlite.JSON(), nullable=True),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("tao_base_experiment_id", sa.VARCHAR(), nullable=True),
        sa.Column("tao_base_experiment_pull_status", sa.VARCHAR(), nullable=True),
        sa.Column(
            "max_images_per_request",
            sa.INTEGER(),
            server_default=sa.text("5"),
            nullable=False,
        ),
        sa.Column(
            "image_cap_support",
            sa.VARCHAR(),
            server_default=sa.text("'unknown'"),
            nullable=False,
        ),
        sa.Column(
            "hosted_compatible",
            sa.BOOLEAN(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column("default_icl_max_examples", sa.INTEGER(), nullable=True),
        sa.PrimaryKeyConstraint("model_config_id"),
    )
    op.create_table(
        "nim_endpoints",
        sa.Column("endpoint_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("display_name", sa.VARCHAR(), nullable=False),
        sa.Column("endpoint_mode", sa.VARCHAR(), nullable=False),
        sa.Column("base_url", sa.VARCHAR(), nullable=False),
        sa.Column(
            "api_format",
            sa.VARCHAR(),
            server_default=sa.text("'openai_compatible'"),
            nullable=False,
        ),
        sa.Column(
            "auth_mode", sa.VARCHAR(), server_default=sa.text("'none'"), nullable=False
        ),
        sa.Column(
            "models_path",
            sa.VARCHAR(),
            server_default=sa.text("'/models'"),
            nullable=False,
        ),
        sa.Column(
            "health_ready_path",
            sa.VARCHAR(),
            server_default=sa.text("'/health/ready'"),
            nullable=False,
        ),
        sa.Column("health_live_path", sa.VARCHAR(), nullable=True),
        sa.Column("metrics_path", sa.VARCHAR(), nullable=True),
        sa.Column(
            "is_enabled", sa.BOOLEAN(), server_default=sa.text("1"), nullable=False
        ),
        sa.Column("last_probe_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column(
            "last_probe_status",
            sa.VARCHAR(),
            server_default=sa.text("'unknown'"),
            nullable=False,
        ),
        sa.Column("last_probe_error_ref", sa.VARCHAR(), nullable=True),
        sa.Column("source_kind", sa.VARCHAR(), nullable=False),
        sa.Column("local_nim_deployment_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("updated_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("max_images_per_request", sa.INTEGER(), nullable=True),
        sa.Column("image_cap_support", sa.VARCHAR(), nullable=True),
        sa.PrimaryKeyConstraint("endpoint_id"),
    )
    op.create_table(
        "operation_records",
        sa.Column("inference_invocation_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("purpose", sa.VARCHAR(), nullable=False),
        sa.Column("example_key", sa.VARCHAR(), nullable=True),
        sa.Column("guidance_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("model_config_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("endpoint_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("model_name", sa.VARCHAR(), nullable=True),
        sa.Column("icl_example_keys_used", sqlite.JSON(), nullable=True),
        sa.Column(
            "invocation_status",
            sa.VARCHAR(),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("latency_ms_end_to_end", sa.INTEGER(), nullable=True),
        sa.Column("generation_preset_key", sa.VARCHAR(), nullable=True),
        sa.Column("sampling_params_effective", sqlite.JSON(), nullable=True),
        sa.Column("thinking_mode_effective", sa.VARCHAR(), nullable=True),
        sa.Column("thinking_request_fields_effective", sqlite.JSON(), nullable=True),
        sa.Column("max_tokens_effective", sa.INTEGER(), nullable=True),
        sa.Column("reasoning_headroom_tokens_effective", sa.INTEGER(), nullable=True),
        sa.Column("visual_budget_preset_key", sa.VARCHAR(), nullable=True),
        sa.Column("visual_budget_params_effective", sqlite.JSON(), nullable=True),
        sa.Column("image_transport_mode", sa.VARCHAR(), nullable=True),
        sa.Column("image_format_transmitted", sa.VARCHAR(), nullable=True),
        sa.Column("label_tier", sa.VARCHAR(), nullable=True),
        sa.Column("raw_model_response_ref", sa.VARCHAR(), nullable=True),
        sa.Column("normalized_json_ref", sa.VARCHAR(), nullable=True),
        sa.Column("validation_report_ref", sa.VARCHAR(), nullable=True),
        sa.Column("schema_valid_core", sa.BOOLEAN(), nullable=True),
        sa.Column("validation_errors_core", sqlite.JSON(), nullable=True),
        sa.Column("validation_errors_aux", sqlite.JSON(), nullable=True),
        sa.Column(
            "structured_generation_fallback_used",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("structured_generation_mode_effective", sa.VARCHAR(), nullable=True),
        sa.Column(
            "structured_generation_attempted",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("finish_reason", sa.VARCHAR(), nullable=True),
        sa.Column("completion_tokens", sa.INTEGER(), nullable=True),
        sa.Column(
            "truncation_attributed_schema_invalid",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("provider_error_ref", sa.VARCHAR(), nullable=True),
        sa.Column(
            "retry_of_inference_invocation_id", sa.VARCHAR(length=36), nullable=True
        ),
        sa.Column("evaluation_run_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("batch_label_run_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column(
            "ignored_due_to_run_cancellation",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("exact_match_pass", sa.BOOLEAN(), nullable=True),
        sa.Column(
            "icl_images_attached_count",
            sa.INTEGER(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("prompt_hash", sa.VARCHAR(length=64), nullable=True),
        sa.Column("seed_effective", sa.INTEGER(), nullable=True),
        sa.Column("t_image_prep_ms", sa.INTEGER(), nullable=True),
        sa.Column("t_prompt_render_ms", sa.INTEGER(), nullable=True),
        sa.Column("t_nim_call_ms", sa.INTEGER(), nullable=True),
        sa.Column("t_validation_ms", sa.INTEGER(), nullable=True),
        sa.Column(
            "thinking_toggle_attempted",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "thinking_fallback_used",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "visual_budget_attempted",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "visual_budget_fallback_used",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("prompt_tokens", sa.INTEGER(), nullable=True),
        sa.Column("total_tokens", sa.INTEGER(), nullable=True),
        sa.PrimaryKeyConstraint("inference_invocation_id"),
    )
    op.create_index(
        "ix_operation_records_batch_label_run_id",
        "operation_records",
        ["batch_label_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_operation_records_evaluation_run_id",
        "operation_records",
        ["evaluation_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_operation_records_example_key",
        "operation_records",
        ["example_key"],
        unique=False,
    )
    op.create_table(
        "pools",
        sa.Column("pool_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column(
            "pool_type",
            sa.VARCHAR(),
            server_default=sa.text("'test_pool'"),
            nullable=False,
        ),
        sa.Column("pool_version", sa.INTEGER(), nullable=False),
        sa.Column("member_example_keys", sqlite.JSON(), nullable=False),
        sa.Column("member_count", sa.INTEGER(), nullable=False),
        sa.Column("guidance_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.PrimaryKeyConstraint("pool_id"),
    )
    op.create_table(
        "projects",
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("name", sa.VARCHAR(), nullable=False),
        sa.Column("description", sa.VARCHAR(), nullable=True),
        sa.Column("project_dir", sa.VARCHAR(), nullable=False),
        sa.Column("teacher_model_config_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("active_guidance_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column(
            "active_student_model_config_id", sa.VARCHAR(length=36), nullable=True
        ),
        sa.Column(
            "labeling_generation_preset_key",
            sa.VARCHAR(),
            server_default=sa.text("'precise'"),
            nullable=False,
        ),
        sa.Column(
            "thinking_default_on",
            sa.BOOLEAN(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "visual_budget_preset_key",
            sa.VARCHAR(),
            server_default=sa.text("'high_detail'"),
            nullable=False,
        ),
        sa.Column(
            "structured_generation_mode_default",
            sa.VARCHAR(),
            server_default=sa.text("'auto'"),
            nullable=False,
        ),
        sa.Column(
            "rationale_anti_anchoring",
            sa.BOOLEAN(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "auto_evaluate_enabled",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "icl_recommendation_dismissed_at_count",
            sa.INTEGER(),
            server_default=sa.text("'0'"),
            nullable=False,
        ),
        sa.Column(
            "export_field_mode",
            sa.VARCHAR(),
            server_default=sa.text("'all'"),
            nullable=False,
        ),
        sa.Column(
            "embedding_provider",
            sa.VARCHAR(),
            server_default=sa.text("'none'"),
            nullable=False,
        ),
        sa.Column("embedding_model_id", sa.VARCHAR(), nullable=True),
        sa.Column("embedding_dim", sa.INTEGER(), nullable=True),
        sa.Column("embedding_endpoint_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column(
            "phash_algorithm",
            sa.VARCHAR(),
            server_default=sa.text("'dct_phash_64'"),
            nullable=False,
        ),
        sa.Column("feature_flags", sqlite.JSON(), nullable=True),
        sa.Column(
            "schema_refinement_reminders_dismissed",
            sa.INTEGER(),
            server_default=sa.text("'0'"),
            nullable=False,
        ),
        sa.Column("schema_change_context_example_key", sa.VARCHAR(), nullable=True),
        sa.Column(
            "test_pool_fraction",
            sa.FLOAT(),
            server_default=sa.text("'0.4'"),
            nullable=False,
        ),
        sa.Column(
            "scaleup_exact_match_threshold",
            sa.FLOAT(),
            server_default=sa.text("'0.8'"),
            nullable=False,
        ),
        sa.Column(
            "scaleup_per_field_match_threshold",
            sa.FLOAT(),
            server_default=sa.text("'0.8'"),
            nullable=False,
        ),
        sa.Column(
            "scaleup_min_per_value_f1_threshold",
            sa.FLOAT(),
            server_default=sa.text("'0.8'"),
            nullable=False,
        ),
        sa.Column(
            "scaleup_accept_rate_threshold",
            sa.FLOAT(),
            server_default=sa.text("'0.8'"),
            nullable=False,
        ),
        sa.Column(
            "scaleup_accept_rate_window",
            sa.INTEGER(),
            server_default=sa.text("'50'"),
            nullable=False,
        ),
        sa.Column(
            "scaleup_min_test_pool_size",
            sa.INTEGER(),
            server_default=sa.text("(60)"),
            nullable=False,
        ),
        sa.Column("review_selector_scheduler_state", sqlite.JSON(), nullable=True),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("updated_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("archived_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("setup_completed_at", sa.VARCHAR(length=24), nullable=True),
        sa.PrimaryKeyConstraint("project_id"),
    )
    op.create_table(
        "run_records",
        sa.Column("run_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("run_type", sa.VARCHAR(), nullable=False),
        sa.Column("status", sa.VARCHAR(), nullable=False),
        sa.Column("status_reason", sa.VARCHAR(), nullable=True),
        sa.Column("cancel_requested_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column(
            "recovered_from_restart",
            sa.BOOLEAN(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("started_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("completed_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("pool_version_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("guidance_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("model_config_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("icl_mode", sa.VARCHAR(), nullable=True),
        sa.Column("evaluation_source", sa.VARCHAR(), nullable=True),
        sa.Column("generation_preset_key", sa.VARCHAR(), nullable=True),
        sa.Column("thinking_mode_effective", sa.VARCHAR(), nullable=True),
        sa.Column("visual_budget_preset_key", sa.VARCHAR(), nullable=True),
        sa.Column("structured_generation_mode_effective", sa.VARCHAR(), nullable=True),
        sa.Column("inference_contract", sqlite.JSON(), nullable=True),
        sa.Column("icl_eligible_count_at_start", sa.INTEGER(), nullable=True),
        sa.Column("icl_eligible_count_at_completion", sa.INTEGER(), nullable=True),
        sa.Column("tao_job_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("tao_native_metrics", sqlite.JSON(), nullable=True),
        sa.Column("rescored_metrics", sqlite.JSON(), nullable=True),
        sa.Column("previous_pool_version", sa.INTEGER(), nullable=True),
        sa.Column("returning_example_keys", sqlite.JSON(), nullable=True),
        sa.Column("new_example_keys", sqlite.JSON(), nullable=True),
        sa.Column("previous_overall_exact_match", sa.FLOAT(), nullable=True),
        sa.Column("coverage_gaps", sqlite.JSON(), nullable=True),
        sa.Column("paused_reason", sa.VARCHAR(), nullable=True),
        sa.Column(
            "examples_succeeded",
            sa.INTEGER(),
            server_default=sa.text("'0'"),
            nullable=False,
        ),
        sa.Column(
            "examples_schema_invalid",
            sa.INTEGER(),
            server_default=sa.text("'0'"),
            nullable=False,
        ),
        sa.Column(
            "examples_timeout",
            sa.INTEGER(),
            server_default=sa.text("'0'"),
            nullable=False,
        ),
        sa.Column(
            "examples_endpoint_error",
            sa.INTEGER(),
            server_default=sa.text("'0'"),
            nullable=False,
        ),
        sa.Column(
            "examples_total",
            sa.INTEGER(),
            server_default=sa.text("'0'"),
            nullable=False,
        ),
        sa.Column("metrics", sqlite.JSON(), nullable=True),
        sa.Column("student_model_config_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("nim_model_profile_requested", sa.VARCHAR(), nullable=True),
        sa.Column("nim_model_profile_selected", sa.VARCHAR(), nullable=True),
        sa.Column("nim_profile_metadata", sqlite.JSON(), nullable=True),
        sa.Column("quantization_method", sa.VARCHAR(), nullable=True),
        sa.Column("gpu_type", sa.VARCHAR(), nullable=True),
        sa.Column("gpu_count", sa.INTEGER(), nullable=True),
        sa.Column("dataset_manifest_sha256", sa.VARCHAR(length=64), nullable=True),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_table(
        "student_models",
        sa.Column("student_model_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column(
            "student_base_model_config_id", sa.VARCHAR(length=36), nullable=False
        ),
        sa.Column("tao_job_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("guidance_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("dataset_export_ids", sqlite.JSON(), nullable=False),
        sa.Column("training_preset", sa.VARCHAR(), nullable=False),
        sa.Column("lora_config", sqlite.JSON(), nullable=False),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column(
            "checkpoint_packaging_status",
            sa.VARCHAR(),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("nim_checkpoint_ref", sa.VARCHAR(), nullable=True),
        sa.Column(
            "quality_status",
            sa.VARCHAR(),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("quality_evaluation_run_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column(
            "serving_status",
            sa.VARCHAR(),
            server_default=sa.text("'not_attempted'"),
            nullable=False,
        ),
        sa.Column("serving_evaluation_run_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("nim_preflight_status", sa.VARCHAR(), nullable=True),
        sa.Column("nim_preflight_details", sqlite.JSON(), nullable=True),
        sa.Column("nim_preflight_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("nim_deployment_mode", sa.VARCHAR(), nullable=True),
        sa.Column("nim_container_id", sa.VARCHAR(), nullable=True),
        sa.Column("nim_endpoint_url", sa.VARCHAR(), nullable=True),
        sa.Column("nim_vlm_release_version", sa.VARCHAR(), nullable=True),
        sa.Column("nim_model_profile_requested", sa.VARCHAR(), nullable=True),
        sa.Column("nim_model_profile_selected", sa.VARCHAR(), nullable=True),
        sa.Column("nim_profile_metadata", sqlite.JSON(), nullable=True),
        sa.Column("gpu_type", sa.VARCHAR(), nullable=True),
        sa.Column("gpu_count", sa.INTEGER(), nullable=True),
        sa.Column("quantization_method", sa.VARCHAR(), nullable=True),
        sa.Column("quantize_tao_job_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("training_inference_contract", sqlite.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("student_model_id"),
    )
    op.create_table(
        "tao_jobs",
        sa.Column("tao_job_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column(
            "student_base_model_config_id", sa.VARCHAR(length=36), nullable=False
        ),
        sa.Column("dataset_export_ids", sqlite.JSON(), nullable=False),
        sa.Column("action", sa.VARCHAR(), nullable=False),
        sa.Column(
            "status",
            sa.VARCHAR(),
            server_default=sa.text("'not_started'"),
            nullable=False,
        ),
        sa.Column("tao_status_raw", sa.VARCHAR(), nullable=True),
        sa.Column(
            "training_backend",
            sa.VARCHAR(),
            server_default=sa.text("'cosmos_rl_tao_vlm'"),
            nullable=False,
        ),
        sa.Column("training_policy_type", sa.VARCHAR(), nullable=True),
        sa.Column("job_config", sqlite.JSON(), nullable=False),
        sa.Column("tao_create_job_request", sqlite.JSON(), nullable=False),
        sa.Column("tao_external_job_id", sa.VARCHAR(), nullable=True),
        sa.Column("progress", sqlite.JSON(), nullable=True),
        sa.Column("outputs", sqlite.JSON(), nullable=True),
        sa.Column("parent_tao_job_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("preflight_result", sqlite.JSON(), nullable=True),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("started_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("completed_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("last_polled_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("error_ref", sa.VARCHAR(), nullable=True),
        sa.Column("poll_error_ref", sa.VARCHAR(), nullable=True),
        sa.Column("chain_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("chain_sequence", sa.INTEGER(), nullable=True),
        sa.Column("chain_halted_reason", sa.VARCHAR(), nullable=True),
        sa.Column(
            "outputs_fetch_status",
            sa.VARCHAR(length=32),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("outputs_fetch_error_ref", sa.VARCHAR(), nullable=True),
        sa.PrimaryKeyConstraint("tao_job_id"),
    )
    op.create_table(
        "training_suites",
        sa.Column("training_suite_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("idempotency_key", sa.VARCHAR(), nullable=False),
        sa.Column("guidance_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("training_preset", sa.VARCHAR(), nullable=False),
        sa.Column("export_field_mode", sa.VARCHAR(), nullable=False),
        sa.Column("include_auto_labeled", sa.BOOLEAN(), nullable=False),
        sa.Column("training_dataset_export_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("evaluation_dataset_export_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column(
            "selected_student_base_model_config_ids", sqlite.JSON(), nullable=False
        ),
        sa.Column("quantization_schemes", sqlite.JSON(), nullable=False),
        sa.Column("chain_ids_ordered", sqlite.JSON(), nullable=False),
        sa.Column(
            "status",
            sa.VARCHAR(),
            server_default=sa.text("'initialized'"),
            nullable=False,
        ),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("started_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("completed_at", sa.VARCHAR(length=24), nullable=True),
        sa.Column("provisioning_run_id", sa.VARCHAR(length=36), nullable=True),
        sa.Column("provisioning_model_names", sqlite.JSON(), nullable=True),
        sa.Column("setup_error_ref", sa.VARCHAR(), nullable=True),
        sa.PrimaryKeyConstraint("training_suite_id"),
        sa.UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_training_suites_project_idempotency",
        ),
    )
    op.create_table(
        "clip_embeddings",
        sa.Column("project_id", sa.VARCHAR(length=36), nullable=False),
        sa.Column("example_key", sa.VARCHAR(), nullable=False),
        sa.Column("embedding_provider", sa.VARCHAR(), nullable=False),
        sa.Column("clip_embedding_model_id", sa.VARCHAR(), nullable=False),
        sa.Column("clip_embedding_dim", sa.INTEGER(), nullable=False),
        sa.Column("vector_blob_f32", sa.BLOB(), nullable=False),
        sa.Column("created_at", sa.VARCHAR(length=24), nullable=False),
        sa.Column("updated_at", sa.VARCHAR(length=24), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id", "example_key"],
            ["examples.project_id", "examples.example_key"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("project_id", "example_key"),
    )


def downgrade() -> None:
    op.drop_table("clip_embeddings")
    op.drop_table("training_suites")
    op.drop_table("tao_jobs")
    op.drop_table("student_models")
    op.drop_table("run_records")
    op.drop_table("projects")
    op.drop_table("pools")
    op.drop_index("ix_operation_records_example_key", table_name="operation_records")
    op.drop_index(
        "ix_operation_records_evaluation_run_id", table_name="operation_records"
    )
    op.drop_index(
        "ix_operation_records_batch_label_run_id", table_name="operation_records"
    )
    op.drop_table("operation_records")
    op.drop_table("nim_endpoints")
    op.drop_table("model_configs")
    op.drop_table("local_nim_deployments")
    op.drop_index(
        "ux_labels_auto_labeled_example",
        table_name="labels",
        sqlite_where=sa.text("label_status = 'auto_labeled'"),
    )
    op.drop_index("ix_labels_example_key_status", table_name="labels")
    op.drop_table("labels")
    op.drop_table("guidances")
    op.drop_index("ux_examples_project_key", table_name="examples")
    op.drop_index("ix_examples_state", table_name="examples")
    op.drop_table("examples")
    op.drop_table("dataset_exports")
    op.drop_table("audit_events")

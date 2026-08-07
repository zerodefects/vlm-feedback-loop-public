# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist immutable model and endpoint inputs on each new run.

Revision ID: v1_0003
Revises: v1_0002
Create Date: 2026-08-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import sqlite

revision = "v1_0003"
down_revision = "v1_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "run_records",
        sa.Column("runtime_config_snapshot", sqlite.JSON(), nullable=True),
    )
    # Freeze every resumable NIM run at the upgrade boundary. These rows were
    # created before the snapshot existed, so the current catalog is the only
    # recoverable approximation of their start-time values. Doing this in the
    # migration closes the window in which a paused run could observe a later
    # catalog edit before its first Resume.
    op.execute(
        sa.text(
            """
            UPDATE run_records
            SET runtime_config_snapshot = (
                SELECT json_object(
                    'version', 1,
                    'project_id', run_records.project_id,
                    'model_config_id', model_configs.model_config_id,
                    'model_name', model_configs.model_name,
                    'endpoint_id', nim_endpoints.endpoint_id,
                    'endpoint_base_url', nim_endpoints.base_url,
                    'endpoint_mode', nim_endpoints.endpoint_mode,
                    'endpoint_auth_mode', nim_endpoints.auth_mode,
                    'context_window_tokens', model_configs.context_window_tokens,
                    'thinking_toggle_mode', model_configs.thinking_toggle_mode,
                    'thinking_toggle_support',
                        model_configs.thinking_toggle_support,
                    'visual_budget_mode', model_configs.visual_budget_mode,
                    'visual_budget_support', model_configs.visual_budget_support,
                    'structured_generation_support',
                        model_configs.structured_generation_support,
                    'max_images_per_request', COALESCE(
                        nim_endpoints.max_images_per_request,
                        model_configs.max_images_per_request
                    ),
                    'default_icl_max_examples',
                        model_configs.default_icl_max_examples
                )
                FROM model_configs
                JOIN nim_endpoints
                  ON nim_endpoints.endpoint_id = model_configs.endpoint_id
                 AND nim_endpoints.project_id = model_configs.project_id
                WHERE model_configs.model_config_id = run_records.model_config_id
                  AND model_configs.project_id = run_records.project_id
            )
            WHERE runtime_config_snapshot IS NULL
              AND model_config_id IS NOT NULL
              AND (
                    (
                        run_type = 'batch_label_run'
                        AND status IN ('queued', 'running', 'canceling', 'paused')
                    )
                    OR (
                        run_type = 'evaluation_run'
                        AND COALESCE(evaluation_source, 'nim') = 'nim'
                        AND status IN ('queued', 'running', 'canceling')
                    )
              )
              AND EXISTS (
                  SELECT 1
                  FROM model_configs
                  JOIN nim_endpoints
                    ON nim_endpoints.endpoint_id = model_configs.endpoint_id
                   AND nim_endpoints.project_id = model_configs.project_id
                  WHERE model_configs.model_config_id = run_records.model_config_id
                    AND model_configs.project_id = run_records.project_id
              )
            """
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("run_records") as batch_op:
        batch_op.drop_column("runtime_config_snapshot")

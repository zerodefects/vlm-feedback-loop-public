# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist the TrainingSuite training mode for exact idempotency replay.

Revision ID: v1_0002
Revises: v1_0001
Create Date: 2026-08-05
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "v1_0002"
down_revision = "v1_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_suites",
        sa.Column(
            "enable_lora",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    # Pre-v1_0002 suites retain the effective mode already frozen on their
    # train job. Suites that never reached chain creation used the historical
    # LoRA-first default.
    op.execute(
        sa.text(
            """
            UPDATE training_suites
            SET enable_lora = COALESCE(
                (
                    SELECT CAST(
                        json_extract(
                            tao_jobs.job_config,
                            '$.lora_config.enable_lora'
                        ) AS INTEGER
                    )
                    FROM tao_jobs
                    WHERE tao_jobs.project_id = training_suites.project_id
                      AND tao_jobs.action = 'train'
                      AND EXISTS (
                          SELECT 1
                          FROM json_each(training_suites.chain_ids_ordered)
                          WHERE json_each.value = tao_jobs.chain_id
                      )
                    ORDER BY tao_jobs.chain_sequence ASC
                    LIMIT 1
                ),
                1
            )
            """
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("training_suites") as batch_op:
        batch_op.drop_column("enable_lora")

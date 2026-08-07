# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reseat the retired MiniMax default on projects with no labels.

Revision ID: v1_0004
Revises: v1_0003
Create Date: 2026-08-06
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "v1_0004"
down_revision = "v1_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # MiniMax M3 was the seeded hosted default before its published
    # non-commercial restriction made it unsuitable for the Blueprint's
    # commercial-first onboarding path. Move only projects with no Label rows:
    # once an SME or batch has produced labels, the selected Teacher is part of
    # that project's provenance and must remain untouched. The catalog rows are
    # retained for historical inspection and an operator may still select one
    # deliberately after this one-time upgrade.
    op.execute(
        sa.text(
            """
            UPDATE projects
            SET teacher_model_config_id = (
                SELECT step.model_config_id
                FROM model_configs AS step
                WHERE step.project_id = projects.project_id
                  AND step.model_name = 'stepfun-ai/step-3.7-flash'
                LIMIT 1
            )
            WHERE EXISTS (
                SELECT 1
                FROM model_configs AS selected
                WHERE selected.project_id = projects.project_id
                  AND selected.model_config_id = projects.teacher_model_config_id
                  AND selected.model_name = 'minimaxai/minimax-m3'
            )
              AND EXISTS (
                SELECT 1
                FROM model_configs AS step
                WHERE step.project_id = projects.project_id
                  AND step.model_name = 'stepfun-ai/step-3.7-flash'
            )
              AND NOT EXISTS (
                SELECT 1
                FROM labels
                WHERE labels.project_id = projects.project_id
            )
            """
        )
    )


def downgrade() -> None:
    # Restoring MiniMax would silently reintroduce a non-commercial default,
    # so downgrade intentionally keeps the safer data selection.
    pass

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export all project-scoped SQLAlchemy models.

Importing this module registers every model with ``ProjectBase.metadata``,
which is required for Alembic autogenerate and ``create_all`` to work.
"""

from vlm_feedback_loop.db.models.audit_event import AuditEvent
from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.local_nim_deployment import LocalNimDeployment
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite

__all__ = [
    "AuditEvent",
    "ClipEmbedding",
    "DatasetExport",
    "Example",
    "Guidance",
    "Label",
    "LocalNimDeployment",
    "ModelConfig",
    "NimEndpoint",
    "OperationRecord",
    "Pool",
    "Project",
    "RunRecord",
    "StudentModel",
    "TAOJob",
    "TrainingSuite",
]

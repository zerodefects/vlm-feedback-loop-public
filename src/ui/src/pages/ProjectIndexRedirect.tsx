// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Index redirect for /projects/:projectId.
 *
 * Looks at the project's setup and active-work state. It routes the SME to
 * the next missing onboarding step, resumes an in-flight Training Suite, or
 * opens Labeling for an established project.
 *
 * The setup-chain screens and Confirm Defaults each auto-skip forward when
 * their own conditions are met, so sending the SME to "setup" is safe: the
 * cascade will resolve to the correct destination.
 *
 * The setup gate is ``setup_completed_at``.
 * A seeded ``teacher_model_config_id`` default exists from the moment
 * the project row is written, so a
 * gate of "any selection set?" would let a brand-new no-keys project
 * bypass NIM setup entirely. Gating on ``setup_completed_at`` fires
 * whenever the SME has not actively walked through the NIM setup and
 * confirm-defaults screens —
 * pre-existing projects are backfilled to ``created_at`` by the migration
 * so they stay past setup.
 */

import { useQuery } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import { Navigate } from "react-router-dom";

import { listTrainingSuites } from "@/api/training";
import { trainingKeys } from "@/api/query-keys";
import { TERMINAL_TRAINING_SUITE_STATUSES } from "@/types/training";
import { useSetupContext } from "./ProjectSetupLayout";

export function ProjectIndexRedirect() {
  const { projectId, project } = useSetupContext();
  const c = project.counts;
  const totalExamples = c.unlabeled + c.auto_labeled + c.verified + c.omitted;
  const shouldCheckTraining =
    project.setup_completed_at !== null &&
    totalExamples > 0 &&
    project.active_guidance_id !== null;

  const {
    data: trainingSuites,
    isLoading: trainingSuitesLoading,
    isError: trainingSuitesError,
    refetch: refetchTrainingSuites,
  } = useQuery({
    queryKey: trainingKeys.suites(projectId),
    queryFn: () => listTrainingSuites(projectId),
    enabled: shouldCheckTraining,
  });

  if (!project.setup_completed_at) {
    return <Navigate to="setup" replace />;
  }

  if (totalExamples === 0) {
    return <Navigate to="ready" replace />;
  }

  if (!project.active_guidance_id) {
    return <Navigate to="create-guidance" replace />;
  }

  if (trainingSuitesLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner size="large" aria-label="Checking active training jobs" />
      </div>
    );
  }

  if (trainingSuitesError) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
        <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
          Failed to check Training Jobs
        </Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          Retry to determine where this project should resume.
        </Text>
        <Button kind="secondary" onClick={() => void refetchTrainingSuites()}>
          Retry
        </Button>
      </div>
    );
  }

  const activeSuite = trainingSuites?.items.find(
    (suite) => !TERMINAL_TRAINING_SUITE_STATUSES.has(suite.status),
  );
  if (activeSuite) {
    return <Navigate to={`training/${activeSuite.training_suite_id}`} replace />;
  }

  return <Navigate to="labeling" replace />;
}

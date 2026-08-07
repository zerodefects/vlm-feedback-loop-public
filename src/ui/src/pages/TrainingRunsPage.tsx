// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Read-only history of project Training Suites, newest first. */

import { useQuery } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import { ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { trainingKeys } from "@/api/query-keys";
import { listTrainingSuites } from "@/api/training";
import { PageContainer } from "@/components/common/PageContainer";
import { SectionCard } from "@/components/common/SectionCard";
import { StatusPill } from "@/components/common/StatusPill";
import { formatTimestamp } from "@/lib/format-date";
import { titleCasePreset } from "@/lib/formatPreset";
import { formatModelDisplayName } from "@/lib/model-display";
import { useSetupContext } from "@/pages/setup-context";
import {
  TERMINAL_TRAINING_SUITE_STATUSES,
  type TrainingSuite,
  type TrainingSuiteStatus,
} from "@/types/training";

function suiteStatusDisplay(status: TrainingSuiteStatus) {
  switch (status) {
    case "completed":
      return { label: "Completed", tone: "success" as const, spinner: false };
    case "failed":
      return { label: "Failed", tone: "error" as const, spinner: false };
    case "canceled":
      return { label: "Canceled", tone: "neutral" as const, spinner: false };
    case "provisioning":
      return { label: "Provisioning", tone: "info" as const, spinner: true };
    case "preparing":
      return { label: "Preparing", tone: "info" as const, spinner: true };
    case "initialized":
      return { label: "Starting", tone: "info" as const, spinner: true };
    case "running":
      return { label: "Running", tone: "success" as const, spinner: true };
  }
}

function suiteModels(suite: TrainingSuite): string {
  const names = suite.chains.map((chain) => chain.base_model_name);
  const effectiveNames = names.length > 0 ? names : suite.provisioning_model_names;
  if (effectiveNames.length === 0) {
    const count = suite.selected_student_base_model_config_ids.length;
    return `${count} selected base${count === 1 ? "" : "s"}`;
  }
  return effectiveNames.map((name) => formatModelDisplayName(name)).join(" · ");
}

function completedJobCount(suite: TrainingSuite): { completed: number; total: number } {
  const jobs = suite.chains.flatMap((chain) => chain.jobs);
  return {
    completed: jobs.filter((job) => job.status === "succeeded").length,
    total: jobs.length,
  };
}

export function TrainingRunsPage() {
  const navigate = useNavigate();
  const { projectId } = useSetupContext();
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: trainingKeys.suites(projectId),
    queryFn: () => listTrainingSuites(projectId),
  });

  if (isLoading) {
    return (
      <PageContainer data-testid="training-runs-loading">
        <div className="flex items-center justify-center py-20">
          <Spinner aria-label="Loading Training Runs" size="large" />
        </div>
      </PageContainer>
    );
  }

  if (isError || !data) {
    return (
      <PageContainer data-testid="training-runs-error">
        <div className="flex flex-col items-center gap-4 py-20 text-center">
          <Text kind="title/sm">Could not load Training Runs</Text>
          <Button kind="secondary" onClick={() => void refetch()}>
            Retry
          </Button>
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer data-testid="training-runs-page">
      <div className="flex items-end justify-between gap-4">
        <div className="flex flex-col gap-2">
          <Text kind="title/md">Training Runs</Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Review training status, metrics, lineage, and logs. Portable deployment
            downloads are available only from Models &amp; Results.
          </Text>
        </div>
        <Button kind="secondary" onClick={() => navigate(`../training`)}>
          Train another Student
        </Button>
      </div>

      {data.items.length === 0 ? (
        <SectionCard data-testid="training-runs-empty">
          <Text kind="title/sm">No Training Runs yet</Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            Configure Student Training to create the first run.
          </Text>
        </SectionCard>
      ) : (
        <div className="flex flex-col gap-3">
          {data.items.map((suite) => {
            const display = suiteStatusDisplay(suite.status);
            const jobs = completedJobCount(suite);
            const active = !TERMINAL_TRAINING_SUITE_STATUSES.has(suite.status);
            const started =
              formatTimestamp(suite.started_at ?? suite.created_at) ?? "Unknown";
            return (
              <SectionCard
                key={suite.training_suite_id}
                density="dense"
                data-testid={`training-run-${suite.training_suite_id}`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex min-w-0 flex-col gap-1">
                    <Text kind="label/bold/sm">{suiteModels(suite)}</Text>
                    <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
                      {titleCasePreset(suite.training_preset)} preset · {jobs.completed}{" "}
                      of {jobs.total} jobs completed
                    </Text>
                    <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
                      Started {started}
                    </Text>
                  </div>
                  <div className="flex shrink-0 items-center gap-3">
                    <StatusPill
                      tone={display.tone}
                      label={display.label}
                      spinner={display.spinner}
                      data-status={suite.status}
                    />
                    <Button
                      kind={active ? "primary" : "secondary"}
                      className={active ? "nvidia-green-button" : undefined}
                      onClick={() => navigate(`../training/${suite.training_suite_id}`)}
                      data-testid={`view-training-run-${suite.training_suite_id}`}
                    >
                      {active ? "Resume" : "View details"} <ArrowRight size={14} />
                    </Button>
                  </div>
                </div>
              </SectionCard>
            );
          })}
        </div>
      )}

      <div className="flex justify-end">
        <Button kind="secondary" onClick={() => navigate(`../overview`)}>
          Back to Project Overview
        </Button>
      </div>
    </PageContainer>
  );
}

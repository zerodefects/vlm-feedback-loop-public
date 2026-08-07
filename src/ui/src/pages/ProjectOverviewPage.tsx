// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * State-aware landing page for a mature project.
 *
 * ProjectIndexRedirect sends projects here after any Training Suite history or
 * Student registration exists. The page separates the three durable intents:
 * continue improving the labeling loop, inspect model results/deployment, and
 * review training history. Active work is resumed before this page is reached.
 */

import { useQuery } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import { ArrowRight, History, ImagePlus, Tags } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { trainingKeys, studentModelKeys } from "@/api/query-keys";
import { listStudentModels } from "@/api/students";
import { listTrainingSuites } from "@/api/training";
import { PageContainer } from "@/components/common/PageContainer";
import { SectionCard } from "@/components/common/SectionCard";
import { SectionHeading } from "@/components/common/SectionHeading";
import { useSetupContext } from "@/pages/setup-context";

export function ProjectOverviewPage() {
  const navigate = useNavigate();
  const { projectId, project } = useSetupContext();

  const {
    data: suites,
    isLoading: suitesLoading,
    isError: suitesError,
    refetch: refetchSuites,
  } = useQuery({
    queryKey: trainingKeys.suites(projectId),
    queryFn: () => listTrainingSuites(projectId),
  });
  const {
    data: students,
    isLoading: studentsLoading,
    isError: studentsError,
    refetch: refetchStudents,
  } = useQuery({
    queryKey: studentModelKeys.list(projectId),
    queryFn: () => listStudentModels(projectId, { limit: 200 }),
  });

  if (suitesLoading || studentsLoading) {
    return (
      <PageContainer data-testid="project-overview-loading">
        <div className="flex items-center justify-center py-20">
          <Spinner aria-label="Loading project overview" size="large" />
        </div>
      </PageContainer>
    );
  }

  if (suitesError || studentsError || !suites || !students) {
    return (
      <PageContainer data-testid="project-overview-error">
        <div className="flex flex-col items-center gap-4 py-20 text-center">
          <Text kind="title/sm">Could not load the project overview</Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            Retry to check Training Runs and Models &amp; Results.
          </Text>
          <Button
            kind="secondary"
            onClick={() => {
              void refetchSuites();
              void refetchStudents();
            }}
          >
            Retry
          </Button>
        </div>
      </PageContainer>
    );
  }

  const suiteCount = suites.items.length;
  const studentCount = students.items.length;
  const qualityValidatedCount = students.items.filter(
    (student) => student.quality_status === "validated",
  ).length;
  const servingValidatedCount = students.items.filter(
    (student) => student.serving_status === "validated",
  ).length;
  const { counts } = project;

  return (
    <PageContainer maxWidthClass="max-w-6xl" data-testid="project-overview-page">
      <div className="flex flex-col gap-2">
        <Text kind="label/regular/xs" className="section-eyebrow">
          PROJECT OVERVIEW
        </Text>
        <Text kind="title/lg">{project.name}</Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
          Continue improving the labeling loop, inspect trained Students, or revisit
          prior training work.
        </Text>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        <SectionCard data-testid="overview-labeling-card">
          <SectionHeading>INTERACTIVE LOOP</SectionHeading>
          <Text kind="title/sm">Continue labeling</Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            {counts.unlabeled} Unlabeled · {counts.verified} Verified ·{" "}
            {counts.auto_labeled} Auto-Labeled
          </Text>
          <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
            Review proposals with ICL or add images to expand the training pool.
          </Text>
          <div className="mt-auto flex flex-col items-start gap-2 pt-2">
            <Button
              kind="tertiary"
              onClick={() => navigate(`../ready`)}
              data-testid="overview-add-images"
            >
              <ImagePlus size={14} /> Add images
            </Button>
            <Button
              kind="primary"
              className="nvidia-green-button"
              onClick={() => navigate(`../labeling`)}
              data-testid="overview-continue-labeling"
            >
              <Tags size={14} /> Continue labeling
            </Button>
          </div>
        </SectionCard>

        <SectionCard data-testid="overview-model-results-card">
          <SectionHeading>MODELS &amp; RESULTS</SectionHeading>
          <Text kind="title/sm">
            {studentCount} Student variant{studentCount === 1 ? "" : "s"}
          </Text>
          {studentCount > 0 ? (
            <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
              {qualityValidatedCount} quality validated · {servingValidatedCount}{" "}
              serving validated
            </Text>
          ) : (
            <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
              No Student model is available from the recorded Training Runs.
            </Text>
          )}
          <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
            Compare quality, run serving benchmarks, and download the portable NIM
            deployment bundle when validation passes.
          </Text>
          <div className="mt-auto pt-2">
            <Button
              kind="primary"
              className="nvidia-green-button"
              disabled={studentCount === 0}
              onClick={() => navigate(`../compare`)}
              data-testid="overview-model-results"
            >
              Models &amp; Results <ArrowRight size={14} />
            </Button>
          </div>
        </SectionCard>

        <SectionCard data-testid="overview-training-runs-card">
          <SectionHeading>TRAINING RUNS</SectionHeading>
          <Text kind="title/sm">
            {suiteCount} run{suiteCount === 1 ? "" : "s"}
          </Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Review job status, metrics, lineage, and logs from prior Student training.
          </Text>
          <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
            Production delivery is consolidated under Models &amp; Results rather than
            individual training outputs.
          </Text>
          <div className="mt-auto pt-2">
            <Button
              kind="primary"
              className="nvidia-green-button"
              disabled={suiteCount === 0}
              onClick={() => navigate(`../training-runs`)}
              data-testid="overview-training-runs"
            >
              <History size={14} /> View training runs
            </Button>
          </div>
        </SectionCard>
      </div>
    </PageContainer>
  );
}

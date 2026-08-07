// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { lazy, Suspense } from "react";
import { Spinner } from "@kui/react";
import { useQuery } from "@tanstack/react-query";
import { Routes, Route } from "react-router-dom";
import { fetchEnvironment } from "@/api/nim";
import { environmentKeys } from "@/api/query-keys";
import { AppShell } from "./components/AppShell";

const ProjectListPage = lazy(async () => ({
  default: (await import("./pages/ProjectListPage")).ProjectListPage,
}));
const ProjectSetupLayout = lazy(async () => ({
  default: (await import("./pages/ProjectSetupLayout")).ProjectSetupLayout,
}));
const ProjectIndexRedirect = lazy(async () => ({
  default: (await import("./pages/ProjectIndexRedirect")).ProjectIndexRedirect,
}));
const ProjectOverviewPage = lazy(async () => ({
  default: (await import("./pages/ProjectOverviewPage")).ProjectOverviewPage,
}));
const NIMConnectionPage = lazy(async () => ({
  default: (await import("./pages/NIMConnectionPage")).NIMConnectionPage,
}));
const NIMNvidiaKeyPage = lazy(async () => ({
  default: (await import("./pages/NIMNvidiaKeyPage")).NIMNvidiaKeyPage,
}));
const NIMNgcKeyPage = lazy(async () => ({
  default: (await import("./pages/NIMNgcKeyPage")).NIMNgcKeyPage,
}));
const NIMSetupGatePage = lazy(async () => ({
  default: (await import("./pages/NIMSetupGatePage")).NIMSetupGatePage,
}));
const ConfirmDefaultsPage = lazy(async () => ({
  default: (await import("./pages/ConfirmDefaultsPage")).ConfirmDefaultsPage,
}));
const CreateGuidancePage = lazy(async () => ({
  default: (await import("./pages/CreateGuidancePage")).CreateGuidancePage,
}));
const EditGuidancePage = lazy(async () => ({
  default: (await import("./pages/EditGuidancePage")).EditGuidancePage,
}));
const ImageIngestPage = lazy(async () => ({
  default: (await import("./pages/ImageIngestPage")).ImageIngestPage,
}));
const LabelingPage = lazy(async () => ({
  default: (await import("./pages/LabelingPage")).LabelingPage,
}));
const ScaleUpHubPage = lazy(async () => ({
  default: (await import("./pages/ScaleUpHubPage")).ScaleUpHubPage,
}));
const BatchPreRunPage = lazy(async () => ({
  default: (await import("./pages/BatchPreRunPage")).BatchPreRunPage,
}));
const BatchRunStatusPage = lazy(async () => ({
  default: (await import("./pages/BatchRunStatusPage")).BatchRunStatusPage,
}));
const StudentTrainingPage = lazy(async () => ({
  default: (await import("./pages/StudentTrainingPage")).StudentTrainingPage,
}));
const TrainingJobMonitorPage = lazy(async () => ({
  default: (await import("./pages/TrainingJobMonitorPage")).TrainingJobMonitorPage,
}));
const TrainingRunsPage = lazy(async () => ({
  default: (await import("./pages/TrainingRunsPage")).TrainingRunsPage,
}));
const CompareBenchmarkPage = lazy(async () => ({
  default: (await import("./pages/CompareBenchmarkPage")).CompareBenchmarkPage,
}));

function RouteFallback() {
  return (
    <div className="flex min-h-[240px] items-center justify-center">
      <Spinner size="large" aria-label="Loading page" />
    </div>
  );
}

/**
 * Start the deployment-scoped assessment once for the browser session without
 * making the Project List wait for it. Routes that truly need the result join
 * this same React Query request; ordinary project pages never gate on it.
 */
function EnvironmentWarmup() {
  useQuery({
    queryKey: environmentKeys.assessment(),
    queryFn: () => fetchEnvironment(),
    staleTime: Number.POSITIVE_INFINITY,
  });
  return null;
}

export function App() {
  return (
    <AppShell>
      <EnvironmentWarmup />
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          <Route path="/" element={<ProjectListPage />} />
          <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
            <Route index element={<ProjectIndexRedirect />} />
            <Route path="overview" element={<ProjectOverviewPage />} />
            {/* FTU three-screen NIM setup.
              Each screen auto-skips when it has nothing to ask, chaining
              forward via location.state.cameFromAutoSkip. A
              fully-configured machine reaches confirm-defaults in zero
              clicks. */}
            <Route path="setup" element={<NIMNvidiaKeyPage />} />
            <Route path="setup/ngc" element={<NIMNgcKeyPage />} />
            <Route path="setup/done" element={<NIMSetupGatePage />} />
            <Route path="confirm-defaults" element={<ConfirmDefaultsPage />} />
            {/* Post-onboarding NIM Configuration entry (NIMConnectionPage):
              power-user surface for overrides and self-hosted /
              local-deploy controls, deliberately kept out of the FTUE
              setup screens. [Save] navigates back to the project
              (which routes to labeling). */}
            <Route path="settings/nim" element={<NIMConnectionPage />} />
            <Route path="create-guidance" element={<CreateGuidancePage />} />
            <Route path="edit-guidance" element={<EditGuidancePage />} />
            {/* "ready" = the project's get-ready / add-images step; the name
              is kept stable because five call sites navigate to
              `/projects/:id/ready`. */}
            <Route path="ready" element={<ImageIngestPage />} />
            <Route path="labeling" element={<LabelingPage />} />
            <Route path="scale-up" element={<ScaleUpHubPage />} />
            <Route path="batch-prerun" element={<BatchPreRunPage />} />
            <Route path="batch-status/:runId" element={<BatchRunStatusPage />} />
            <Route path="training" element={<StudentTrainingPage />} />
            <Route path="training-runs" element={<TrainingRunsPage />} />
            <Route
              path="training/:trainingSuiteId"
              element={<TrainingJobMonitorPage />}
            />
            <Route path="compare" element={<CompareBenchmarkPage />} />
          </Route>
        </Routes>
      </Suspense>
    </AppShell>
  );
}

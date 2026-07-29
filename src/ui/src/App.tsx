// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Routes, Route } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { ProjectListPage } from "./pages/ProjectListPage";
import { ProjectSetupLayout } from "./pages/ProjectSetupLayout";
import { ProjectIndexRedirect } from "./pages/ProjectIndexRedirect";
import { NIMConnectionPage } from "./pages/NIMConnectionPage";
import { NIMNvidiaKeyPage } from "./pages/NIMNvidiaKeyPage";
import { NIMNgcKeyPage } from "./pages/NIMNgcKeyPage";
import { NIMSetupGatePage } from "./pages/NIMSetupGatePage";
import { ConfirmDefaultsPage } from "./pages/ConfirmDefaultsPage";
import { CreateGuidancePage } from "./pages/CreateGuidancePage";
import { EditGuidancePage } from "./pages/EditGuidancePage";
import { ImageIngestPage } from "./pages/ImageIngestPage";
import { LabelingPage } from "./pages/LabelingPage";
import { ScaleUpHubPage } from "./pages/ScaleUpHubPage";
import { BatchPreRunPage } from "./pages/BatchPreRunPage";
import { BatchRunStatusPage } from "./pages/BatchRunStatusPage";
import { StudentTrainingPage } from "./pages/StudentTrainingPage";
import { TrainingJobMonitorPage } from "./pages/TrainingJobMonitorPage";
import { CompareBenchmarkPage } from "./pages/CompareBenchmarkPage";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<ProjectListPage />} />
        <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
          <Route index element={<ProjectIndexRedirect />} />
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
          <Route
            path="training/:trainingSuiteId"
            element={<TrainingJobMonitorPage />}
          />
          <Route path="compare" element={<CompareBenchmarkPage />} />
        </Route>
      </Routes>
    </AppShell>
  );
}

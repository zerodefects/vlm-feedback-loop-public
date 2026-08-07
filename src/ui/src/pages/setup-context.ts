// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useOutletContext } from "react-router-dom";

import type { EnvironmentResponse } from "@/types/nim";
import type { ProjectResponse } from "@/types/project";

export interface SetupContext {
  projectId: string;
  project: ProjectResponse;
  /** Present on routes whose behavior depends on deployment capabilities. */
  environment?: EnvironmentResponse;
}

export interface EnvironmentSetupContext extends SetupContext {
  environment: EnvironmentResponse;
}

export function useSetupContext(): SetupContext {
  return useOutletContext<SetupContext>();
}

export function useEnvironmentSetupContext(): EnvironmentSetupContext {
  const context = useOutletContext<SetupContext>();
  if (!context.environment) {
    throw new Error("This route requires the deployment environment assessment.");
  }
  return context as EnvironmentSetupContext;
}

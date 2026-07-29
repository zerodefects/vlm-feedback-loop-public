// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Layout wrapper for project-scoped screens.
 *
 * Fetches project + environment once and provides them to child routes
 * via React Router's outlet context.
 *
 * Once ``project.setup_completed_at`` is non-null, this layout
 * additionally portals a "NIM Configuration" link into the AppShell
 * header, which reopens the NIM Connection screen in edit mode.
 * (Archiving a project is done from the Project List screen.)
 */

import { Link, Outlet, useOutletContext, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";

import { fetchProject } from "@/api/projects";
import { fetchEnvironment } from "@/api/nim";
import { ApiError, parseApiErrorDetail } from "@/api/client";
import { projectKeys, environmentKeys } from "@/api/query-keys";
import { HeaderRightPortal } from "@/components/HeaderRightPortal";
import type { ProjectResponse } from "@/types/project";
import type { EnvironmentResponse } from "@/types/nim";

export interface SetupContext {
  projectId: string;
  project: ProjectResponse;
  environment: EnvironmentResponse;
}

export function useSetupContext() {
  return useOutletContext<SetupContext>();
}

export function ProjectSetupLayout() {
  const { projectId } = useParams<{ projectId: string }>();

  const {
    data: project,
    isLoading: projectLoading,
    isError: projectError,
    error: projectQueryError,
    refetch: refetchProject,
  } = useQuery({
    queryKey: projectKeys.detail(projectId!),
    queryFn: () => fetchProject(projectId!),
    enabled: !!projectId,
  });

  const {
    data: environment,
    isLoading: envLoading,
    isError: envError,
    error: envQueryError,
    refetch: refetchEnvironment,
  } = useQuery({
    queryKey: environmentKeys.assessment(),
    queryFn: fetchEnvironment,
  });

  if (!projectId) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          No project selected.
        </Text>
      </div>
    );
  }

  if (projectLoading || envLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner size="large" aria-label="Loading project" />
      </div>
    );
  }

  if (projectError || envError || !project || !environment) {
    // apiFetch throws a TypeError (not an ApiError) when the backend is
    // unreachable — the dominant cause of landing here, and one the SME
    // can act on (start the backend, then retry) if the screen says so
    // instead of dead-ending on a generic failure line.
    const failure = projectQueryError ?? envQueryError;
    const unreachable = failure !== null && !(failure instanceof ApiError);
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
        <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
          {unreachable
            ? "Cannot reach the backend"
            : "Failed to load project or environment data"}
        </Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          {unreachable
            ? "Check that the backend is running, then retry."
            : (parseApiErrorDetail(failure) ??
              "The request failed. Retry, or check the backend logs.")}
        </Text>
        <Button
          kind="secondary"
          onClick={() => {
            void refetchProject();
            void refetchEnvironment();
          }}
          data-testid="setup-layout-retry"
        >
          Retry
        </Button>
      </div>
    );
  }

  const showChrome = project.setup_completed_at !== null;

  return (
    <>
      {showChrome && <ProjectChrome projectId={projectId} />}
      <Outlet context={{ projectId, project, environment } satisfies SetupContext} />
    </>
  );
}

// ── Project chrome ────────────────────────────────────────────────────────────

/**
 * Persistent project-scoped "NIM Configuration" link. Portals into the
 * AppShell header's right slot (beside Docs) and re-opens the NIM
 * Connection screen in edit mode. Styled as a header nav-link to match
 * the "Projects" and "Docs" affordances. Archiving lives on the Project
 * List screen, so no kebab menu is needed here.
 */
function ProjectChrome({ projectId }: { projectId: string }): JSX.Element {
  return (
    <HeaderRightPortal>
      <Link
        to={`/projects/${projectId}/settings/nim`}
        className="nav-link"
        data-testid="project-nim-config-link"
      >
        NIM Configuration
      </Link>
    </HeaderRightPortal>
  );
}

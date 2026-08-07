// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Layout wrapper for project-scoped screens.
 *
 * Fetches the project and provides it to child routes via React Router's
 * outlet context. The deployment-scoped environment assessment is warmed at
 * app scope and only gates routes whose behavior actually depends on it.
 *
 * Once ``project.setup_completed_at`` is non-null, this layout
 * additionally portals project navigation into the AppShell header. Overview
 * keeps the mature-project workflow reachable from every project screen; NIM
 * Configuration reopens connection settings.
 * (Archiving a project is done from the Project List screen.)
 */

import { Link, Navigate, Outlet, useLocation, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";

import { fetchProject } from "@/api/projects";
import { fetchEnvironment } from "@/api/nim";
import { ApiError, parseApiErrorDetail } from "@/api/client";
import { projectKeys, environmentKeys } from "@/api/query-keys";
import { HeaderRightPortal } from "@/components/HeaderRightPortal";
import type { SetupContext } from "@/pages/setup-context";
import { isSetupChainState } from "@/types/setupChain";

export function ProjectSetupLayout() {
  const { projectId } = useParams<{ projectId: string }>();
  const location = useLocation();
  const environmentRequired = projectRouteRequiresEnvironment(
    location.pathname,
    projectId,
  );

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
    queryFn: () => fetchEnvironment(),
    enabled: environmentRequired,
    staleTime: Number.POSITIVE_INFINITY,
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

  if (projectLoading || (environmentRequired && envLoading)) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner size="large" aria-label="Loading project" />
      </div>
    );
  }

  if (projectError || !project || (environmentRequired && (envError || !environment))) {
    // apiFetch throws a TypeError (not an ApiError) when the backend is
    // unreachable — the dominant cause of landing here, and one the SME
    // can act on (start the backend, then retry) if the screen says so
    // instead of dead-ending on a generic failure line.
    const failure = projectQueryError ?? (environmentRequired ? envQueryError : null);
    const unreachable = failure !== null && !(failure instanceof ApiError);
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
        <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
          {unreachable
            ? "Cannot reach the backend"
            : environmentRequired
              ? "Failed to load project or environment data"
              : "Failed to load project data"}
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
            if (environmentRequired) void refetchEnvironment();
          }}
          data-testid="setup-layout-retry"
        >
          Retry
        </Button>
      </div>
    );
  }

  const showChrome = project.setup_completed_at !== null;

  // Setup-chain state is deliberately ephemeral. Once onboarding has
  // completed, a copied /setup URL has no truthful path/model selection to
  // summarize: rebuilding it from safe defaults can claim a hosted Teacher
  // while the project is actually attached to a local NIM. Send such re-entry
  // to the authoritative project overview; intentional in-progress
  // transitions carry a complete SetupChainState and remain allowed.
  const onboardingRoutes = new Set([
    `/projects/${projectId}/setup`,
    `/projects/${projectId}/setup/ngc`,
    `/projects/${projectId}/setup/done`,
    `/projects/${projectId}/confirm-defaults`,
  ]);
  const completedSetupDeepLink =
    showChrome &&
    onboardingRoutes.has(location.pathname) &&
    !isSetupChainState(location.state);
  if (completedSetupDeepLink) {
    return <Navigate to={`/projects/${projectId}/overview`} replace />;
  }

  return (
    <>
      {showChrome && <ProjectChrome projectId={projectId} projectName={project.name} />}
      <Outlet context={{ projectId, project, environment } satisfies SetupContext} />
    </>
  );
}

function projectRouteRequiresEnvironment(
  pathname: string,
  projectId: string | undefined,
): boolean {
  if (!projectId) return false;
  const prefix = `/projects/${projectId}/`;
  if (!pathname.startsWith(prefix)) return false;
  const route = pathname.slice(prefix.length);
  return (
    route === "setup" ||
    route.startsWith("setup/") ||
    route === "confirm-defaults" ||
    route === "settings/nim" ||
    route === "ready" ||
    route === "compare"
  );
}

// ── Project chrome ────────────────────────────────────────────────────────────

/**
 * Persistent project-scoped destinations. Portals into the AppShell header's
 * right slot so a user can reach the project overview or NIM configuration
 * without exiting to the global Projects screen.
 */
function ProjectChrome({
  projectId,
  projectName,
}: {
  projectId: string;
  projectName: string;
}): JSX.Element {
  const location = useLocation();
  const links = [
    {
      to: `/projects/${projectId}/overview`,
      label: "Overview",
      testId: "project-overview-link",
    },
    {
      to: `/projects/${projectId}/settings/nim`,
      label: "NIM Configuration",
      testId: "project-nim-config-link",
    },
  ];
  return (
    <HeaderRightPortal>
      <Text
        kind="label/regular/xs"
        title={`Project: ${projectName}`}
        data-testid="project-context-name"
        style={{
          color: "var(--text-secondary)",
          maxWidth: "360px",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        Project: {projectName}
      </Text>
      <div
        aria-hidden="true"
        className="h-5 w-px"
        style={{ backgroundColor: "var(--glass-border)" }}
      />
      {links.map((link) => (
        <Link
          key={link.to}
          to={link.to}
          className={`nav-link ${location.pathname === link.to ? "nav-link-active" : ""}`}
          data-testid={link.testId}
        >
          <Text kind="label/regular/sm" style={{ color: "inherit" }}>
            {link.label}
          </Text>
        </Link>
      ))}
    </HeaderRightPortal>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Project List — home screen.
 *
 * Four states:
 *   - Empty state — no projects, single CTA
 *   - Projects exist — card grid with counts and dates
 *   - Create Project dialog (modal overlay)
 *   - Project Lock Error dialog
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button, Card, CardContent, Dropdown, Spinner, Text } from "@kui/react";
import { MoreVertical } from "lucide-react";

import { ApiError } from "@/api/client";
import { fetchProject, fetchProjectList } from "@/api/projects";
import { projectKeys } from "@/api/query-keys";
import { formatDate, formatRelativeTime } from "@/lib/format-date";
import { CreateProjectDialog } from "@/components/CreateProjectDialog";
import { HeaderRightPortal } from "@/components/HeaderRightPortal";
import { ProjectArchiveDialog } from "@/components/ProjectArchiveDialog";
import { ProjectLockDialog } from "@/components/ProjectLockDialog";
import type { ProjectListItem } from "@/types/project";

// ---------------------------------------------------------------------------
// localStorage helper for "last active screen"
// ---------------------------------------------------------------------------

function saveLastActiveProject(projectId: string): void {
  try {
    localStorage.setItem(`lastActiveScreen:${projectId}`, `/projects/${projectId}`);
  } catch {
    // localStorage unavailable — ignore silently
  }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/**
 * Inline loop diagram for the FTUE empty state. Shows the three-stage
 * cycle (Propose -> Review -> Refine, looping back) and the exit path
 * (Optimize -> Deploy). Styling rules borrowed
 * from the Retail Blueprints: uppercase tracked caps via `.glass-caption`,
 * NVIDIA green concentrated in the primary forward arrows, muted gray
 * everywhere else. Decorative (`aria-hidden`) -- the helper paragraph
 * above carries the semantic description.
 */
function LoopDiagram() {
  return (
    <div
      aria-hidden="true"
      style={{
        position: "relative",
        width: "400px",
        height: "100px",
        marginTop: "20px",
      }}
    >
      <svg
        viewBox="0 0 400 100"
        width="400"
        height="100"
        aria-hidden="true"
        style={{ position: "absolute", inset: 0, overflow: "visible" }}
      >
        <defs>
          <marker
            id="ftue-arrow-green"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent-green)" />
          </marker>
        </defs>

        {/* Forward arrows between cycle stages (NVIDIA green). */}
        <line
          x1="64"
          y1="50"
          x2="82"
          y2="50"
          stroke="var(--accent-green)"
          strokeWidth="1.25"
          markerEnd="url(#ftue-arrow-green)"
        />
        <line
          x1="138"
          y1="50"
          x2="156"
          y2="50"
          stroke="var(--accent-green)"
          strokeWidth="1.25"
          markerEnd="url(#ftue-arrow-green)"
        />

        {/* Exit from Refine -> Optimize. */}
        <line
          x1="212"
          y1="50"
          x2="230"
          y2="50"
          stroke="var(--accent-green)"
          strokeWidth="1.25"
          markerEnd="url(#ftue-arrow-green)"
        />

        {/* Optimize -> Deploy connector. Same SVG line + marker as every
            other forward arrow in the diagram so all green arrowheads
            share one shape. */}
        <line
          x1="300"
          y1="50"
          x2="318"
          y2="50"
          stroke="var(--accent-green)"
          strokeWidth="1.25"
          markerEnd="url(#ftue-arrow-green)"
        />

        {/* Loop-back curve: dashed muted arc from Refine over the row
            back to Propose. Conveys that the three stages cycle until
            the SME exits to the deploy path. */}
        <path
          d="M 200 40 C 170 -10 50 -10 18 40"
          fill="none"
          stroke="var(--accent-green)"
          strokeWidth="1.25"
          strokeDasharray="3 3"
          markerEnd="url(#ftue-arrow-green)"
          opacity="0.85"
        />
      </svg>

      {/* Cycle stage labels — uppercase tracked via .glass-caption. */}
      <span
        className="glass-caption"
        style={{ position: "absolute", top: "44px", left: "6px" }}
      >
        Propose
      </span>
      <span
        className="glass-caption"
        style={{ position: "absolute", top: "44px", left: "86px" }}
      >
        Review
      </span>
      <span
        className="glass-caption"
        style={{ position: "absolute", top: "44px", left: "162px" }}
      >
        Refine
      </span>

      {/* Exit labels -- same .glass-caption typography so the whole
          diagram reads as one typographic register. */}
      <span
        className="glass-caption"
        style={{ position: "absolute", top: "44px", left: "236px" }}
      >
        Optimize
      </span>
      <span
        className="glass-caption"
        style={{ position: "absolute", top: "44px", left: "324px" }}
      >
        Deploy
      </span>
    </div>
  );
}

function EmptyState({
  onCreateClick,
  hasArchived,
  onShowArchived,
}: {
  onCreateClick: () => void;
  hasArchived: boolean;
  onShowArchived: () => void;
}) {
  // FTUE orientation surface. The card lays out a single
  // typographic ladder: eyebrow (positioning), hero (promise), helper
  // (loop in plain prose), loop diagram (the same loop in pictures, plus
  // the exit path), then CTA. Explicit per-slot spacing keeps the
  // rhythm even if any single line changes length. No em-dashes in copy.
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div
        className="glass-card glass-card--elevated flex flex-col items-center text-center"
        style={{
          padding: "48px 56px",
          maxWidth: "600px",
        }}
      >
        <span className="glass-caption">
          NVIDIA Blueprint &middot; Interactive VLM Loop
        </span>

        <Text
          kind="title/lg"
          style={{
            color: "var(--text-primary)",
            marginTop: "16px",
            maxWidth: "440px",
            lineHeight: 1.25,
          }}
        >
          Iterate on a vision model in minutes, not weeks.
        </Text>

        <Text
          kind="body/regular/sm"
          style={{
            color: "var(--text-secondary)",
            marginTop: "14px",
            maxWidth: "460px",
            lineHeight: 1.55,
          }}
        >
          A Teacher VLM proposes labels. Your edits sharpen the next proposal. Then
          train and deploy a Student NIM.
        </Text>

        <LoopDiagram />

        <Button
          kind="primary"
          className="nvidia-green-button"
          onClick={onCreateClick}
          style={{ marginTop: "24px" }}
        >
          + Create Project
        </Button>

        {/* Archived projects are reachable even with zero ACTIVE projects, so
            an SME can restore one without first having to create a throwaway.
            Shown ONLY when something is actually archived. */}
        {hasArchived && (
          <Button
            kind="tertiary"
            onClick={onShowArchived}
            data-testid="show-archived-empty"
            style={{ marginTop: "12px", color: "var(--text-secondary)" }}
          >
            Show archived
          </Button>
        )}
      </div>
    </div>
  );
}

/** Numeric count one step above the surrounding secondary-text label. */
function CountValue({ value }: { value: number }) {
  return <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{value}</span>;
}

function ProjectCard({
  project,
  onClick,
  onArchive,
  onUnarchive,
}: {
  project: ProjectListItem;
  onClick: () => void;
  onArchive: () => void;
  onUnarchive: () => void;
}) {
  const { counts } = project;
  // Loose null check: treat both null and undefined as "not archived". This
  // matters because pre-existing test fixtures (and any older API client
  // that hasn't bumped its types) may send undefined instead of an explicit
  // null when the project is active.
  const isArchived = project.archived_at !== null && project.archived_at !== undefined;

  // Archived cards are non-clickable. Drop role/tabIndex/
  // onClick/keyboard handler so the SME can't accidentally enter a paused
  // project; the only affordance is the Restore item in the kebab menu.
  // We deliberately do NOT set aria-disabled on the card: the kebab inside
  // is interactive, and aria-disabled on the parent would propagate
  // through the accessibility tree (and Playwright actionability checks)
  // marking the kebab as disabled too. Visual opacity + the "Archived"
  // pill + the absence of role="button" are sufficient to communicate
  // non-interactive state.
  const cardInteractiveProps = isArchived
    ? {
        role: undefined,
        tabIndex: undefined,
        onClick: undefined,
        onKeyDown: undefined,
      }
    : {
        role: "button" as const,
        tabIndex: 0,
        onClick,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onClick();
          }
        },
      };

  // Kebab overflow menu items — Archive on active cards, Restore on
  // archived cards. Modeled on the RAG Blueprint's ChatActionsMenu.tsx
  // (Dropdown items array shape) so the pattern matches an existing
  // NVIDIA Blueprint convention.
  const menuItems = isArchived
    ? [{ children: "Restore", onSelect: onUnarchive }]
    : [{ children: "Archive", onSelect: onArchive }];

  return (
    <Card
      className={`glass-card glass-card--elevated h-full ${
        isArchived ? "" : "glass-card--interactive cursor-pointer"
      }`}
      style={isArchived ? { opacity: 0.6 } : undefined}
      aria-label={
        isArchived
          ? `Project ${project.name} (archived)`
          : `Open project ${project.name}`
      }
      {...cardInteractiveProps}
    >
      <CardContent>
        <div className="flex flex-col gap-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2">
              <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
                {project.name}
              </Text>
              {isArchived && (
                <span
                  className="glass-pill"
                  style={{
                    fontSize: "11px",
                    padding: "2px 8px",
                    color: "var(--text-muted)",
                  }}
                  data-testid="archived-badge"
                >
                  Archived
                </span>
              )}
            </div>
            {/* Kebab Dropdown trigger click bubbles to the card's
                role="button" parent — without this stopPropagation
                wrapper, opening the menu would also open the project. */}
            <div
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => e.stopPropagation()}
            >
              <Dropdown
                items={menuItems}
                size="small"
                side="bottom"
                align="end"
                showChevron={false}
                aria-label={`More actions for project ${project.name}`}
              >
                <MoreVertical
                  size={16}
                  aria-hidden="true"
                  data-testid="card-menu-trigger"
                  style={{ color: "var(--text-muted)" }}
                />
              </Dropdown>
            </div>
          </div>

          {project.description && (
            <Text
              kind="body/regular/sm"
              className="line-clamp-2"
              style={{ color: "var(--text-secondary)" }}
            >
              {project.description}
            </Text>
          )}

          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Verified: <CountValue value={counts.verified} /> &middot; Unlabeled:{" "}
            <CountValue value={counts.unlabeled} />
            {counts.auto_labeled > 0 && (
              <>
                {" "}
                &middot; Auto-Labeled: <CountValue value={counts.auto_labeled} />
              </>
            )}
          </Text>
          {counts.omitted > 0 && (
            <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
              Omitted: <CountValue value={counts.omitted} />
            </Text>
          )}

          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            Created {formatDate(project.created_at)} &middot; Updated{" "}
            {formatRelativeTime(project.updated_at)}
          </Text>
        </div>
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function ProjectListPage() {
  const navigate = useNavigate();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [lockDialogOpen, setLockDialogOpen] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const [archiveTarget, setArchiveTarget] = useState<ProjectListItem | null>(null);
  const [unarchiveTarget, setUnarchiveTarget] = useState<ProjectListItem | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: projectKeys.list(showArchived),
    queryFn: () => fetchProjectList(showArchived),
  });

  // Whether ANY archived project exists — even when there are zero ACTIVE
  // projects and the FTUE empty state is showing (the main query returns
  // active-only then). Drives the "Show archived" affordance, which appears
  // only when something is archived. Workspace-global on every list
  // response (cheap marker-file scan server-side), so no second
  // archived-inclusive fetch is needed.
  const hasArchived = data?.has_archived ?? false;

  const projects = data?.items ?? [];
  const isEmpty = !isLoading && projects.length === 0;

  function showToast(message: string) {
    setToast(message);
    window.setTimeout(() => setToast(null), 4000);
  }

  async function handleCardClick(project: ProjectListItem) {
    // Single-process lock: probe the project before navigating. The
    // backend returns 409 (global handler in main.py) when another process
    // holds the lock; render the project-lock dialog instead of navigating
    // into a doomed detail route.
    try {
      await fetchProject(project.project_id);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setLockDialogOpen(true);
        return;
      }
      throw err;
    }
    saveLastActiveProject(project.project_id);
    navigate(`/projects/${project.project_id}`);
  }

  // -- Loading state --
  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Spinner size="large" aria-label="Loading projects" />
      </div>
    );
  }

  // -- Error state --
  if (isError) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8">
        <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
          Failed to load projects
        </Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          Check that the backend is running and try again.
        </Text>
        <Button
          kind="secondary"
          onClick={() => void refetch()}
          data-testid="project-list-retry"
        >
          Retry
        </Button>
      </div>
    );
  }

  // -- Empty state --
  if (isEmpty) {
    return (
      <>
        <EmptyState
          onCreateClick={() => setDialogOpen(true)}
          hasArchived={hasArchived}
          onShowArchived={() => setShowArchived(true)}
        />
        <CreateProjectDialog open={dialogOpen} onOpenChange={setDialogOpen} />
      </>
    );
  }

  // -- Projects exist --
  return (
    <>
      {/* [+ Create Project] lives in the app bar (top-right).
          Portals into AppShell's #header-right-slot; falls back to inline
          rendering when AppShell isn't mounted (unit tests). */}
      <HeaderRightPortal>
        <Button
          kind="primary"
          className="nvidia-green-button"
          onClick={() => setDialogOpen(true)}
        >
          + Create Project
        </Button>
      </HeaderRightPortal>

      <div className="flex flex-1 flex-col p-6">
        <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6">
          <div className="flex items-center justify-between gap-3">
            <Text kind="title/xl" style={{ color: "var(--text-primary)" }}>
              Projects
            </Text>
            {/* Show-archived toggle. Inline next to the title
                rather than in HeaderRightPortal to avoid contention with
                the [+ Create Project] CTA. Pill chrome at rest + green
                tint when active matches the SegmentedControl pattern so
                the ON/OFF state is readable without relying on the label
                flip alone. */}
            {/* Toggle appears only when something is archived.
                Hidden entirely on a workspace with no archived projects. */}
            {hasArchived && (
              <Button
                kind="tertiary"
                onClick={() => setShowArchived((v) => !v)}
                aria-pressed={showArchived}
                data-testid="show-archived-toggle"
                style={{
                  borderRadius: 999,
                  background: showArchived
                    ? "var(--accent-green-bg)"
                    : "var(--block-bg)",
                  border: showArchived
                    ? "1px solid var(--accent-green-border)"
                    : "1px solid var(--glass-border)",
                  color: showArchived ? "var(--accent-green)" : "var(--text-secondary)",
                }}
              >
                {showArchived ? "Hide archived" : "Show archived"}
              </Button>
            )}
          </div>

          {/* Project cards */}
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
            {projects.map((project) => (
              <ProjectCard
                key={project.project_id}
                project={project}
                onClick={() => handleCardClick(project)}
                onArchive={() => setArchiveTarget(project)}
                onUnarchive={() => setUnarchiveTarget(project)}
              />
            ))}
          </div>
        </div>
      </div>

      <CreateProjectDialog open={dialogOpen} onOpenChange={setDialogOpen} />
      <ProjectLockDialog
        open={lockDialogOpen}
        onClose={() => setLockDialogOpen(false)}
      />
      {archiveTarget && (
        <ProjectArchiveDialog
          variant="archive"
          open={true}
          projectId={archiveTarget.project_id}
          projectName={archiveTarget.name}
          onClose={() => setArchiveTarget(null)}
          onDone={() => showToast(`Archived “${archiveTarget.name}”.`)}
        />
      )}
      {unarchiveTarget && (
        <ProjectArchiveDialog
          variant="unarchive"
          open={true}
          projectId={unarchiveTarget.project_id}
          projectName={unarchiveTarget.name}
          onClose={() => setUnarchiveTarget(null)}
          onDone={() => showToast(`Restored “${unarchiveTarget.name}”.`)}
        />
      )}
      {toast && (
        <div
          className="fixed top-4 right-4 z-50 px-4 py-3 toast-success"
          role="status"
          data-testid="archive-toast"
        >
          <Text kind="body/regular/sm">{toast}</Text>
        </div>
      )}
    </>
  );
}

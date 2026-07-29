// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the Project List page — empty state, project cards,
 * create dialog, lock detection, and the archive feature.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import type { ReactNode } from "react";
import { ProjectListPage } from "@/pages/ProjectListPage";

// ---------------------------------------------------------------------------
// Mock API
// ---------------------------------------------------------------------------

const mockFetchProjectList = vi.fn();
const mockFetchProject = vi.fn();
const mockCreateProject = vi.fn();
const mockArchiveProject = vi.fn();
const mockUnarchiveProject = vi.fn();
const mockNavigate = vi.fn();

vi.mock("@/api/projects", () => ({
  fetchProjectList: (...args: unknown[]) => mockFetchProjectList(...args),
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
  createProject: (...args: unknown[]) => mockCreateProject(...args),
  archiveProject: (...args: unknown[]) => mockArchiveProject(...args),
  unarchiveProject: (...args: unknown[]) => mockUnarchiveProject(...args),
}));

// Spy on navigate so lock-detection tests can assert non-navigation.
vi.mock("react-router-dom", async () => {
  const actual: Record<string, unknown> = await vi.importActual("react-router-dom");
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// Mock EventSource used by SSE store (not relevant to this page)
installEventSourceMock();

// ---------------------------------------------------------------------------
// Wrapper
// ---------------------------------------------------------------------------

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
    },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={["/"]}>{children}</MemoryRouter>
      </QueryClientProvider>
    );
  }
  return { Wrapper, queryClient };
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Project-list item (a different wire shape than ProjectResponse — the
 *  shared fixtures don't cover it). Zeroed counts by default; pass only
 *  the deltas a test asserts on. */
function makeListItem(
  overrides: Record<string, unknown> & {
    counts?: Partial<Record<string, number>>;
  } = {},
) {
  const { counts, ...rest } = overrides;
  return {
    project_id: "p1",
    name: "Test Project",
    description: null,
    created_at: "2026-04-11T12:00:00Z",
    updated_at: "2026-04-11T12:00:00Z",
    archived_at: null,
    counts: {
      verified: 0,
      unlabeled: 0,
      auto_labeled: 0,
      omitted: 0,
      pending_relabel: 0,
      prior_relabeled: 0,
      ...counts,
    },
    ...rest,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ProjectListPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // --- Empty state ---

  it("renders empty state with correct copy when no projects exist", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();

    render(<ProjectListPage />, { wrapper: Wrapper });

    // The hero promise is the load-bearing heading on the FTUE empty
    // state — wait on it instead of a status line.
    await waitFor(() => {
      expect(
        screen.getByText(/Iterate on a vision model in minutes/),
      ).toBeInTheDocument();
    });

    // Eyebrow positions the app as a Blueprint reference.
    expect(
      screen.getByText(/NVIDIA Blueprint . Interactive VLM Loop/),
    ).toBeInTheDocument();
    // Helper describes the loop concretely.
    expect(screen.getByText(/A Teacher VLM proposes labels/)).toBeInTheDocument();
    // Loop diagram labels: three-stage cycle + the single exit path
    // "Optimize -> Deploy", rendered as two separate spans connected
    // by the same SVG arrow shape used elsewhere in the diagram.
    expect(screen.getByText("Propose")).toBeInTheDocument();
    expect(screen.getByText("Review")).toBeInTheDocument();
    expect(screen.getByText("Refine")).toBeInTheDocument();
    expect(screen.getByText("Optimize")).toBeInTheDocument();
    expect(screen.getByText("Deploy")).toBeInTheDocument();
    // getByRole throws on multiple matches, so this also pins that the
    // empty state has exactly one Create button (no header duplicate).
    expect(screen.getByRole("button", { name: /Create Project/i })).toBeInTheDocument();
  });

  // --- Projects exist ---

  it("renders project cards with name, description, counts, and dates", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [
        makeListItem({
          name: "Damage Inspection",
          description: "Surface damage classification",
          created_at: "2026-03-15T14:22:07Z",
          updated_at: "2026-04-11T10:00:00Z",
          counts: { verified: 142, unlabeled: 358, auto_labeled: 580, omitted: 12 },
        }),
      ],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();

    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Damage Inspection")).toBeInTheDocument();
    });

    expect(screen.getByText("Surface damage classification")).toBeInTheDocument();
    // Counts render with the numeric value emphasized in a nested span, so
    // assert on the whole card's text rather than a single text node.
    const card = screen.getByText("Damage Inspection").closest('[role="button"]');
    expect(card?.textContent).toMatch(/Verified: 142/);
    expect(card?.textContent).toMatch(/Unlabeled: 358/);
    expect(card?.textContent).toMatch(/Auto-Labeled: 580/);
    expect(card?.textContent).toMatch(/Omitted: 12/);
  });

  // --- Null description ---

  it("does not render a description line when description is null", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [
        makeListItem({
          project_id: "p2",
          name: "No Description Project",
          created_at: "2026-03-30T12:00:00Z",
        }),
      ],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();

    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("No Description Project")).toBeInTheDocument();
    });

    // The description line is ABSENT entirely when the project has no
    // description — no em-dash, no placeholder, no "null" string.
    // Guard against all three.
    const card = screen.getByText("No Description Project").closest('[role="button"]');
    expect(card).not.toBeNull();
    expect(card?.textContent).not.toContain("null");
    expect(card?.textContent).not.toContain("—");
    // No description-level Text (body/regular/sm) between name and counts
    expect(card?.textContent).toMatch(/No Description Project\s*Verified:/);
  });

  // --- Zero counts ---

  it("renders zero counts correctly for a new project", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [makeListItem({ project_id: "p3", name: "Fresh Project" })],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();

    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Fresh Project")).toBeInTheDocument();
    });

    const card = screen.getByText("Fresh Project").closest('[role="button"]');
    expect(card?.textContent).toMatch(/Verified: 0/);
    expect(card?.textContent).toMatch(/Unlabeled: 0/);
    // Auto-Labeled and Omitted are hidden when zero. Only the
    // Verified + Unlabeled baseline counts should render on a fresh
    // project.
    expect(screen.queryByText(/Auto-Labeled/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Omitted/i)).not.toBeInTheDocument();
  });

  // --- Auto-Labeled conditional rendering ---
  // (zero → hidden is pinned by the zero-counts test above)

  it("renders the Auto-Labeled segment when count is positive", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [
        makeListItem({
          project_id: "p-auto-nonzero",
          name: "With AutoLabels",
          counts: { verified: 5, unlabeled: 10, auto_labeled: 42 },
        }),
      ],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();

    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("With AutoLabels")).toBeInTheDocument();
    });

    const card = screen.getByText("With AutoLabels").closest('[role="button"]');
    expect(card?.textContent).toMatch(/Auto-Labeled: 42/);
  });

  // --- pending_relabel NOT displayed ---

  it("does not display pending_relabel in the card", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [
        makeListItem({
          project_id: "p4",
          name: "Count Test",
          counts: { verified: 10, unlabeled: 5, pending_relabel: 3 },
        }),
      ],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();

    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Count Test")).toBeInTheDocument();
    });

    // pending_relabel should NOT appear anywhere in the rendered output
    expect(screen.queryByText(/pending_relabel/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/pending relabel/i)).not.toBeInTheDocument();
  });

  // --- Create dialog opens ---

  it("opens create dialog when CTA is clicked in projects-exist state", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [makeListItem({ project_id: "p5", name: "Existing" })],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();
    const user = userEvent.setup();

    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Existing")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /Create Project/i }));

    // The modal heading should appear (KUI renders it as an h2)
    await waitFor(() => {
      expect(screen.getByTestId("nv-modal-heading")).toBeInTheDocument();
    });

    // Cancel is a no-op — the dialog closes, nothing is created.
    await user.click(screen.getByRole("button", { name: /^Cancel$/i }));
    await waitFor(() => {
      expect(screen.queryByTestId("nv-modal-heading")).not.toBeInTheDocument();
    });
    expect(mockCreateProject).not.toHaveBeenCalled();
  });

  it.each([
    ["Name", "Keyboard Project", null],
    ["Description", "Keyboard Project", "Created from the keyboard"],
  ])(
    "creates a project when Enter is pressed in the %s field",
    async (fieldName, projectName, description) => {
      mockFetchProjectList.mockResolvedValue({
        items: [makeListItem({ project_id: "p5", name: "Existing" })],
        next_cursor: null,
      });
      mockCreateProject.mockResolvedValue({
        project_id: "keyboard-project",
        name: projectName,
        description,
      });
      const { Wrapper } = createWrapper();
      const user = userEvent.setup();

      render(<ProjectListPage />, { wrapper: Wrapper });

      await screen.findByText("Existing");
      await user.click(screen.getByRole("button", { name: /Create Project/i }));

      const nameInput = await screen.findByRole("textbox", { name: "Name" });
      await user.type(nameInput, projectName);

      const field = screen.getByRole("textbox", { name: fieldName });
      if (description) await user.type(field, description);
      await user.type(field, "{Enter}");

      await waitFor(() => {
        expect(mockCreateProject.mock.calls[0]?.[0]).toEqual({
          name: projectName,
          description,
        });
      });
    },
  );

  // --- Lock detection ---

  describe("lock detection", () => {
    const lockedProject = makeListItem({
      project_id: "locked-1",
      name: "Locked Project",
      created_at: "2026-04-20T12:00:00Z",
      updated_at: "2026-04-20T12:00:00Z",
    });

    it("opens the Project In Use dialog on 409 and does not navigate", async () => {
      mockFetchProjectList.mockResolvedValue({
        items: [lockedProject],
        next_cursor: null,
      });
      const { ApiError } = await import("@/api/client");
      mockFetchProject.mockRejectedValueOnce(
        new ApiError(409, "This project is already open in another process."),
      );
      const { Wrapper } = createWrapper();
      const user = userEvent.setup();

      render(<ProjectListPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByText("Locked Project")).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Open project Locked Project/i }),
      );

      // Probe fired
      expect(mockFetchProject).toHaveBeenCalledWith("locked-1");
      // Lock dialog body visible
      await waitFor(() => {
        expect(
          screen.getByText(/already open in another process/i),
        ).toBeInTheDocument();
      });
      // Must NOT have navigated
      expect(mockNavigate).not.toHaveBeenCalled();
    });

    it("navigates normally on a successful probe (no 409)", async () => {
      mockFetchProjectList.mockResolvedValue({
        items: [lockedProject],
        next_cursor: null,
      });
      mockFetchProject.mockResolvedValueOnce({ project_id: "locked-1" });
      const { Wrapper } = createWrapper();
      const user = userEvent.setup();

      render(<ProjectListPage />, { wrapper: Wrapper });

      await waitFor(() => {
        expect(screen.getByText("Locked Project")).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Open project Locked Project/i }),
      );

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith("/projects/locked-1");
      });
      // Lock dialog must NOT appear
      expect(
        screen.queryByText(/already open in another process/i),
      ).not.toBeInTheDocument();
    });

    it("persists lastActiveScreen in localStorage on successful open", async () => {
      mockFetchProjectList.mockResolvedValue({
        items: [lockedProject],
        next_cursor: null,
      });
      mockFetchProject.mockResolvedValueOnce({ project_id: "locked-1" });
      localStorage.removeItem("lastActiveScreen:locked-1");
      const { Wrapper } = createWrapper();
      const user = userEvent.setup();

      render(<ProjectListPage />, { wrapper: Wrapper });
      await waitFor(() => {
        expect(screen.getByText("Locked Project")).toBeInTheDocument();
      });
      await user.click(
        screen.getByRole("button", { name: /Open project Locked Project/i }),
      );

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith("/projects/locked-1");
      });
      expect(localStorage.getItem("lastActiveScreen:locked-1")).toBe(
        "/projects/locked-1",
      );
    });
  });

  // --- Header has Create button when projects exist ---

  it("shows header-level Create button when projects exist", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [makeListItem({ project_id: "p6", name: "Test" })],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();

    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => {
      expect(screen.getByText("Projects")).toBeInTheDocument();
    });

    expect(screen.getByRole("button", { name: /Create Project/i })).toBeInTheDocument();
  });
});

// ─── Archive feature ─────────────────────────────────────────────────────────

const ACTIVE_FIXTURE = makeListItem({
  project_id: "p-active",
  name: "Active Project",
  description: "An active project",
  counts: { verified: 5, unlabeled: 10 },
});

const ARCHIVED_FIXTURE = makeListItem({
  project_id: "p-archived",
  name: "Archived Project",
  description: "Was archived earlier today",
  created_at: "2026-04-10T12:00:00Z",
  updated_at: "2026-04-12T08:00:00Z",
  counts: { verified: 100 },
  archived_at: "2026-04-12T08:30:00Z",
});

describe("ProjectListPage archive feature", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("kebab menu Archive item on an active card opens the confirm dialog", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [ACTIVE_FIXTURE],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();
    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => screen.getByText("Active Project"));
    const user = userEvent.setup();
    // Open the card's kebab overflow menu, then choose Archive.
    await user.click(screen.getByTestId("card-menu-trigger"));
    await user.click(screen.getByRole("menuitem", { name: "Archive" }));

    await waitFor(() => {
      expect(screen.getByText("Archive project?")).toBeInTheDocument();
    });
    expect(screen.getByText(/Archive ["“]Active Project["”]/)).toBeInTheDocument();
  });

  it("Archive confirm calls archiveProject mutation and shows success toast", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [ACTIVE_FIXTURE],
      next_cursor: null,
    });
    mockArchiveProject.mockResolvedValueOnce({
      ...ACTIVE_FIXTURE,
      archived_at: "2026-04-13T10:00:00Z",
    });
    const { Wrapper } = createWrapper();
    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => screen.getByText("Active Project"));
    const user = userEvent.setup();
    // Open kebab → Archive menu item → confirm-dialog Archive button.
    await user.click(screen.getByTestId("card-menu-trigger"));
    await user.click(screen.getByRole("menuitem", { name: "Archive" }));
    await user.click(screen.getByRole("button", { name: /^Archive$/ }));

    await waitFor(() => {
      expect(mockArchiveProject).toHaveBeenCalledWith("p-active");
    });
    await waitFor(() => {
      expect(screen.getByTestId("archive-toast")).toBeInTheDocument();
      expect(screen.getByText(/Archived ["“]Active Project["”]/)).toBeInTheDocument();
    });
  });

  it("Show-archived toggle re-fetches with includeArchived=true", async () => {
    // Arg-aware: the active-only list omits the archived project but still
    // reports has_archived (workspace-global flag on every response); the
    // archived-inclusive list includes the archived card.
    mockFetchProjectList.mockImplementation((includeArchived?: boolean) =>
      Promise.resolve(
        includeArchived
          ? {
              items: [ACTIVE_FIXTURE, ARCHIVED_FIXTURE],
              next_cursor: null,
              has_archived: true,
            }
          : { items: [ACTIVE_FIXTURE], next_cursor: null, has_archived: true },
      ),
    );
    const { Wrapper } = createWrapper();
    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => screen.getByText("Active Project"));
    // The active view does not render the archived project — has_archived
    // drives the toggle, but only the main query feeds the card grid.
    expect(screen.queryByText("Archived Project")).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByTestId("show-archived-toggle"));

    await waitFor(() => {
      expect(screen.getByText("Archived Project")).toBeInTheDocument();
    });
    // Last call to fetchProjectList must have includeArchived=true
    expect(mockFetchProjectList).toHaveBeenLastCalledWith(true);
  });

  it("hides the show-archived toggle when no projects are archived", async () => {
    // Only active projects exist — has_archived comes back false.
    mockFetchProjectList.mockResolvedValue({
      items: [ACTIVE_FIXTURE],
      next_cursor: null,
      has_archived: false,
    });
    const { Wrapper } = createWrapper();
    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => screen.getByText("Active Project"));
    // Toggle must NOT render when nothing is archived.
    expect(screen.queryByTestId("show-archived-toggle")).not.toBeInTheDocument();
  });

  it("empty active list with archived projects shows a Show-archived affordance that reveals them", async () => {
    // Zero active projects, but an archived one exists: the FTUE empty state
    // must still expose a way to reach archived projects so the SME can
    // restore one without creating a throwaway project first.
    mockFetchProjectList.mockImplementation((includeArchived?: boolean) =>
      Promise.resolve(
        includeArchived
          ? { items: [ARCHIVED_FIXTURE], next_cursor: null, has_archived: true }
          : { items: [], next_cursor: null, has_archived: true },
      ),
    );
    const { Wrapper } = createWrapper();
    render(<ProjectListPage />, { wrapper: Wrapper });

    // FTUE empty state renders...
    await waitFor(() =>
      expect(
        screen.getByText(/Iterate on a vision model in minutes/),
      ).toBeInTheDocument(),
    );
    // ...with the Show-archived affordance (because something IS archived).
    const showArchivedBtn = await screen.findByTestId("show-archived-empty");
    expect(showArchivedBtn).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(showArchivedBtn);

    // Revealing archived shows the archived project card with its Restore menu.
    await waitFor(() =>
      expect(screen.getByText("Archived Project")).toBeInTheDocument(),
    );
  });

  it("empty active list with NO archived projects shows only the Create CTA", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [],
      next_cursor: null,
      has_archived: false,
    });
    const { Wrapper } = createWrapper();
    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() =>
      expect(
        screen.getByText(/Iterate on a vision model in minutes/),
      ).toBeInTheDocument(),
    );
    // No archived → no Show-archived affordance on the empty state.
    expect(screen.queryByTestId("show-archived-empty")).not.toBeInTheDocument();
  });

  it("archived cards are non-interactive and the kebab menu offers Restore (not Archive)", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [ARCHIVED_FIXTURE],
      next_cursor: null,
    });
    const { Wrapper } = createWrapper();
    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => screen.getByText("Archived Project"));

    // The card has no role=button affordance — and crucially, no
    // aria-disabled either, so the kebab menu inside the card remains
    // actionable for assistive tech and Playwright.
    const card = screen.getByLabelText(/Archived Project \(archived\)/);
    expect(card).not.toHaveAttribute("role", "button");
    expect(card).not.toHaveAttribute("aria-disabled");

    // The "Archived" badge is rendered next to the name
    expect(screen.getByTestId("archived-badge")).toHaveTextContent("Archived");

    // The kebab overflow menu trigger is present on archived cards too.
    expect(screen.getByTestId("card-menu-trigger")).toBeInTheDocument();

    // Open the kebab — menu offers Restore only, no Archive.
    const user = userEvent.setup();
    await user.click(screen.getByTestId("card-menu-trigger"));
    expect(
      await screen.findByRole("menuitem", { name: "Restore" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Archive" })).not.toBeInTheDocument();
    // Close the menu before the navigation assertion below.
    await user.keyboard("{Escape}");

    // Clicking the card body does NOT trigger navigation (fetchProject
    // is the lock-probe entry point — must not be called).
    await user.click(card);
    expect(mockFetchProject).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it("Unarchive confirm calls unarchiveProject mutation", async () => {
    mockFetchProjectList.mockResolvedValue({
      items: [ARCHIVED_FIXTURE],
      next_cursor: null,
    });
    mockUnarchiveProject.mockResolvedValueOnce({
      ...ARCHIVED_FIXTURE,
      archived_at: null,
    });
    const { Wrapper } = createWrapper();
    render(<ProjectListPage />, { wrapper: Wrapper });

    await waitFor(() => screen.getByText("Archived Project"));
    const user = userEvent.setup();
    // Open kebab → Restore menu item → confirm-dialog Restore button.
    await user.click(screen.getByTestId("card-menu-trigger"));
    await user.click(screen.getByRole("menuitem", { name: "Restore" }));

    await waitFor(() => {
      expect(screen.getByText("Restore project?")).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /^Restore$/ }));

    await waitFor(() => {
      expect(mockUnarchiveProject).toHaveBeenCalledWith("p-archived");
    });
  });
});

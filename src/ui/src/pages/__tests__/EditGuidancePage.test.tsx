// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the Edit Guidance page.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import type { ReactNode } from "react";
import { EditGuidancePage } from "@/pages/EditGuidancePage";
import { ProjectSetupLayout } from "@/pages/ProjectSetupLayout";
import { fakeValidateDraft } from "@/test/fake-validate-draft";
import type { DraftValidationRequest } from "@/types/guidance";

// ── Mocks ───────────────────────────────────────────────────────────────────

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();
const mockFetchGuidance = vi.fn();
const mockValidateDraft = vi.fn();
const mockEditGuidancePreview = vi.fn();
const mockEditGuidanceExecute = vi.fn();
const mockUpdateProject = vi.fn();

vi.mock("@/api/projects", () => ({
  fetchProject: (...args: unknown[]) => mockFetchProject(...args),
  fetchProjectList: vi.fn().mockResolvedValue({ items: [], next_cursor: null }),
  createProject: vi.fn(),
}));

vi.mock("@/api/nim", () => ({
  fetchEnvironment: (...args: unknown[]) => mockFetchEnvironment(...args),
}));

vi.mock("@/api/guidance", () => ({
  fetchGuidance: (...args: unknown[]) => mockFetchGuidance(...args),
  validateDraft: (...args: unknown[]) => mockValidateDraft(...args),
  editGuidancePreview: (...args: unknown[]) => mockEditGuidancePreview(...args),
  editGuidanceExecute: (...args: unknown[]) => mockEditGuidanceExecute(...args),
  createGuidance: vi.fn(),
  listGuidances: vi.fn(),
}));

vi.mock("@/api/model-configs", () => ({
  updateProject: (...args: unknown[]) => mockUpdateProject(...args),
  fetchModelConfigs: vi.fn().mockResolvedValue({ items: [] }),
}));

installEventSourceMock();

// ── Fixtures ────────────────────────────────────────────────────────────────

const PROJECT = makeProjectResponse({
  project_id: "test-pid",
  name: "Damage Inspection",
  description: null,
  project_dir: "/tmp/workspace/projects/test-pid",
  created_at: "2026-04-13T10:00:00Z",
  updated_at: "2026-04-13T10:00:00Z",
  // ProjectResponse.counts mirrors backend ProjectCounts.
  // EditGuidancePage reads counts.verified for the marker tooltip {N}.
  counts: {
    verified: 42,
    unlabeled: 0,
    auto_labeled: 0,
    omitted: 0,
    pending_relabel: 0,
    prior_relabeled: 0,
  },
  teacher_model_config_id: "mc-cosmos-8b",
  active_guidance_id: "guid-001",
  active_student_model_config_id: null,
});

const ENVIRONMENT = makeEnvironmentResponse({
  hosted_nim_available: true,
  nvidia_api_key_configured: true,
  recommended_teacher_mode: "hosted",
  recommended_embedding_mode: "hosted",
});

const GUIDANCE = {
  guidance_id: "guid-001",
  project_id: "test-pid",
  version_number: 3,
  description: "Classify visible damage types.",
  schema_fields: [
    {
      field_id: "fid-1",
      field_name: "damage_type",
      type: "enum",
      role: "core",
      allowed_values: ["crush", "dent", "scratch"],
      display_order: 0,
    },
    {
      field_id: "fid-2",
      field_name: "severity",
      type: "integer",
      role: "core",
      minimum: 0,
      maximum: 4,
      display_order: 1,
    },
    {
      field_id: "fid-rn",
      field_name: "rationale_note",
      type: "string",
      role: "aux",
      display_order: 0,
    },
  ],
  rules: "If damage is partially obscured, classify based on visible portion.",
  derived_json_schema: { type: "object" },
  generation_order: ["rationale_note", "damage_type", "severity"],
  schema_hash: "abc123",
  created_at: "2026-04-13T12:00:00Z",
};

// ── Wrapper ─────────────────────────────────────────────────────────────────

function createWrapper(initialPath = "/projects/test-pid/edit-guidance") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[initialPath]}>
          <Routes>
            <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
              <Route path="edit-guidance" element={<EditGuidancePage />} />
              <Route path="ready" element={<div data-testid="ready-page">Ready</div>} />
            </Route>
          </Routes>
          {children}
        </MemoryRouter>
      </QueryClientProvider>
    );
  }
  return { Wrapper, queryClient };
}

async function renderPage() {
  const { Wrapper } = createWrapper();
  render(<div />, { wrapper: Wrapper });
  await waitFor(() => {
    expect(screen.getByTestId("edit-guidance-page")).toBeInTheDocument();
  });
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("EditGuidancePage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(ENVIRONMENT);
    mockFetchGuidance.mockResolvedValue(GUIDANCE);
    // Wire-mock the draft-validation endpoint — the page's ONLY validator.
    mockValidateDraft.mockImplementation((_projectId: unknown, body: unknown) =>
      Promise.resolve(fakeValidateDraft(body as DraftValidationRequest)),
    );
    mockUpdateProject.mockResolvedValue(PROJECT);
  });

  // ── Header and structure ──────────────────────────────────────────────

  it("renders with 'Edit Guidance' title and version number", async () => {
    await renderPage();
    expect(screen.getByText("Edit Guidance")).toBeInTheDocument();
    expect(screen.getByText(/v3/)).toBeInTheDocument();
  });

  it("has no template selector", async () => {
    await renderPage();
    expect(screen.queryByLabelText("Start from:")).not.toBeInTheDocument();
  });

  it("shows post-save banner with correct text", async () => {
    await renderPage();
    const banner = screen.getByTestId("post-save-banner");
    expect(banner).toHaveTextContent(/Renames apply directly/);
    expect(banner).toHaveTextContent(/invalidates labels/);
  });

  // ── Pre-populated fields ──────────────────────────────────────────────

  it("loads existing guidance data into form", async () => {
    await renderPage();
    const textarea = screen.getByTestId("description-textarea") as HTMLTextAreaElement;
    expect(textarea.value).toBe("Classify visible damage types.");
    expect(screen.getByTestId("field-row-damage_type")).toBeInTheDocument();
    expect(screen.getByTestId("field-row-severity")).toBeInTheDocument();
    expect(screen.queryByTestId("field-row-rationale_note")).not.toBeInTheDocument();
    expect(
      screen.getByRole("switch", { name: "Enable rationale notes" }),
    ).toBeChecked();
  });

  // ── Label-invalidation markers (~) ────────────────────────────────────

  it("shows invalidation markers on label-invalidating controls", async () => {
    await renderPage();
    // Markers should be present somewhere on the page (type selects, delete, move-section)
    const markers = screen.getAllByTestId("invalidation-marker");
    expect(markers.length).toBeGreaterThan(0);
  });

  it("field name input does NOT have a marker (in-place edit)", async () => {
    await renderPage();
    const row = screen.getByTestId("field-row-damage_type");
    const nameInput = within(row).getByDisplayValue("damage_type");
    // Markers render as a corner overlay inside a relative wrapper span
    // around the control they annotate (see the type select / move-section
    // / delete controls). Renaming is an in-place edit, so the name input
    // must have neither a marker sibling nor a marker in its wrapper.
    expect(nameInput.previousElementSibling?.getAttribute("data-testid")).not.toBe(
      "invalidation-marker",
    );
    expect(
      nameInput.parentElement?.querySelector(
        ':scope > [data-testid="invalidation-marker"]',
      ),
    ).toBeNull();
  });

  // ── Description and Rules editable ────────────────────────────────────

  // Timeout note: types two full sentences character-by-character; under
  // full-suite parallel load this runs ~5s and flakes against the 5s
  // default, while passing in <2s isolated.
  it("Description and Rules are freely editable", { timeout: 15_000 }, async () => {
    await renderPage();
    const user = userEvent.setup();
    const desc = screen.getByTestId("description-textarea") as HTMLTextAreaElement;
    await user.type(desc, " Updated.");
    expect(desc.value).toContain("Updated.");

    const rules = screen.getByTestId("rules-textarea") as HTMLTextAreaElement;
    await user.type(rules, " New rule.");
    expect(rules.value).toContain("New rule.");
  });

  // ── Add Core Field has marker; Add Aux Field does not ──────────────────

  it("[+ Add Core Field] button has an invalidation marker", async () => {
    await renderPage();
    const btn = screen.getByTestId("add-core-field-btn");
    expect(within(btn).getByTestId("invalidation-marker")).toBeInTheDocument();
  });

  it("[+ Add Aux Field] does not have a marker", async () => {
    await renderPage();
    const auxBtn = screen.getByTestId("add-aux-field-btn");
    // Add Aux Field should NOT contain an invalidation marker
    expect(within(auxBtn).queryByTestId("invalidation-marker")).not.toBeInTheDocument();
  });

  it("enum value chips do not repeat the invalidation marker", async () => {
    // One editor-level marker (on the [+ add] affordance) flags allowed-value
    // changes; repeating it inside every chip's remove button reads as noise.
    await renderPage();
    const row = screen.getByTestId("field-row-damage_type");
    for (const chip of within(row).getAllByTestId(/^value-chip-/)) {
      expect(within(chip).queryByTestId("invalidation-marker")).not.toBeInTheDocument();
    }
    const addBtn = within(row).getByTestId(/^add-value-btn-/);
    expect(within(addBtn).getByTestId("invalidation-marker")).toBeInTheDocument();
  });

  // ── Validation surfacing after a failed save ──────────────────────────

  it("failed save escalates the status badge from grey 'required' to red 'errors' language", async () => {
    // Edit must match Create: pre-save the badge uses neutral "N required"
    // wording; once a save attempt fails it switches to live error language.
    // Trigger a real error by stripping the damage_type enum down to one
    // value (ENUM_TOO_FEW_VALUES).
    await renderPage();
    const user = userEvent.setup();
    const row = screen.getByTestId("field-row-damage_type");
    await user.click(within(row).getByLabelText("Remove value crush"));
    await user.click(within(row).getByLabelText("Remove value dent"));
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent(
        "Schema: 1 required",
      );
    });
    await user.click(screen.getByTestId("save-guidance-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Schema: 1 error");
    });
  });

  // ── Save flow: in-place edit executes directly ────────────────────────

  it("in-place edit executes without confirmation dialog", async () => {
    mockEditGuidancePreview.mockResolvedValue({
      edit_type: "in_place",
      changes: [
        {
          field_id: "fid-1",
          change_type: "field_rename",
          classification: "in_place",
          detail: {},
        },
      ],
      verified_count: 10,
      auto_labeled_count: 0,
      change_summary: {},
    });
    mockEditGuidanceExecute.mockResolvedValue({
      guidance: { ...GUIDANCE, guidance_id: "guid-002", version_number: 4 },
      edit_type: "in_place",
      verified_reverted_count: 0,
      auto_labeled_reverted_count: 0,
      changes: [],
    });

    await renderPage();
    const user = userEvent.setup();

    // Make an edit to enable save
    const desc = screen.getByTestId("description-textarea") as HTMLTextAreaElement;
    await user.type(desc, " Edit.");

    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));

    await waitFor(() => {
      expect(mockEditGuidancePreview).toHaveBeenCalled();
    });
    // In-place → executes directly, no dialog
    await waitFor(() => {
      expect(mockEditGuidanceExecute).toHaveBeenCalled();
    });
  });

  it("preview failure surfaces an error and does NOT execute the edit", async () => {
    // The preview is the ONLY gate that raises the semantic-change
    // confirmation dialog. If it throws, the save must STOP — falling
    // through to execute would silently return every Verified label to
    // Unlabeled with no warning.
    mockEditGuidancePreview.mockRejectedValue(new Error("backend offline"));

    await renderPage();
    const user = userEvent.setup();

    const desc = screen.getByTestId("description-textarea") as HTMLTextAreaElement;
    await user.type(desc, " Edit.");

    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("preview-error")).toBeInTheDocument();
    });
    expect(screen.getByTestId("preview-error")).toHaveTextContent(
      /labels were not changed/i,
    );
    expect(mockEditGuidanceExecute).not.toHaveBeenCalled();
  });

  // ── Save flow: semantic edit shows confirmation dialog ─────────────────

  it("semantic edit shows confirmation dialog with specific change description and Verified count", async () => {
    mockEditGuidancePreview.mockResolvedValue({
      edit_type: "semantic",
      changes: [
        {
          field_id: "fid-1",
          change_type: "type_change",
          classification: "semantic",
          detail: { field_name: "damage_type", old_type: "enum", new_type: "integer" },
        },
      ],
      verified_count: 42,
      auto_labeled_count: 0,
      change_summary: {},
    });

    await renderPage();
    const user = userEvent.setup();

    // Make an edit
    const desc = screen.getByTestId("description-textarea") as HTMLTextAreaElement;
    await user.type(desc, " Changed.");

    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));

    // Wait for confirmation dialog
    await waitFor(() => {
      expect(screen.getByTestId("confirm-dialog-body")).toBeInTheDocument();
    });
    // The dialog references the specific change
    expect(screen.getByTestId("confirm-change-description")).toHaveTextContent(
      'Changing the type of "damage_type" changes what a correct answer looks like.',
    );
    expect(screen.getByTestId("confirm-dialog-body")).toHaveTextContent(
      "42 labeled images",
    );
  });

  it("confirmation dialog describes allowed_value_change correctly", async () => {
    mockEditGuidancePreview.mockResolvedValue({
      edit_type: "semantic",
      changes: [
        {
          field_id: "fid-1",
          change_type: "allowed_value_change",
          classification: "semantic",
          detail: {
            field_name: "primary_damage",
            removed: ["old_val"],
            added: ["new_val"],
          },
        },
      ],
      verified_count: 142,
      auto_labeled_count: 0,
      change_summary: {},
    });

    await renderPage();
    const user = userEvent.setup();
    await user.type(screen.getByTestId("description-textarea"), " X.");

    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("confirm-dialog-body")).toBeInTheDocument();
    });
    expect(screen.getByTestId("confirm-change-description")).toHaveTextContent(
      'Changing the allowed values for "primary_damage" changes what a correct answer looks like.',
    );
  });

  it("confirmation dialog shows Auto-Labeled paragraph when applicable", async () => {
    mockEditGuidancePreview.mockResolvedValue({
      edit_type: "semantic",
      changes: [],
      verified_count: 42,
      auto_labeled_count: 100,
      change_summary: {},
    });

    await renderPage();
    const user = userEvent.setup();
    await user.type(screen.getByTestId("description-textarea"), " X.");

    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("confirm-dialog-body")).toBeInTheDocument();
    });
    expect(screen.getByTestId("confirm-auto-labeled-text")).toHaveTextContent(
      "100 Auto-Labeled",
    );
  });

  it("[Update and Re-label] calls editGuidanceExecute", async () => {
    mockEditGuidancePreview.mockResolvedValue({
      edit_type: "semantic",
      changes: [],
      verified_count: 10,
      auto_labeled_count: 0,
      change_summary: {},
    });
    mockEditGuidanceExecute.mockResolvedValue({
      guidance: { ...GUIDANCE, guidance_id: "guid-003", version_number: 4 },
      edit_type: "semantic",
      verified_reverted_count: 10,
      auto_labeled_reverted_count: 0,
      changes: [],
    });

    await renderPage();
    const user = userEvent.setup();
    await user.type(screen.getByTestId("description-textarea"), " Y.");

    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));

    await waitFor(() => {
      expect(screen.getByTestId("confirm-update-btn")).toBeInTheDocument();
    });
    await user.click(screen.getByTestId("confirm-update-btn"));

    await waitFor(() => {
      expect(mockEditGuidanceExecute).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
        active_guidance_id: "guid-003",
      });
    });
  });

  // ── Optional rationale_note ───────────────────────────────────────────

  it("can disable rationale_note without exposing it as a field row", async () => {
    await renderPage();
    const user = userEvent.setup();
    const toggle = screen.getByRole("switch", { name: "Enable rationale notes" });
    expect(toggle).toBeChecked();
    await user.click(toggle);
    expect(toggle).not.toBeChecked();
  });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the Create Guidance page.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { installEventSourceMock } from "@/test/event-source-mock";
import { makeEnvironmentResponse, makeProjectResponse } from "@/test/fixtures";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import type { ReactNode } from "react";
import { CreateGuidancePage } from "@/pages/CreateGuidancePage";
import { ProjectSetupLayout } from "@/pages/ProjectSetupLayout";
import { GUIDANCE_TEMPLATES } from "@/lib/guidance-templates";
import { FIELD_TYPE_OPTIONS } from "@/components/guidance/field-utils";
import { fakeValidateDraft } from "@/test/fake-validate-draft";
import type { DraftValidationRequest } from "@/types/guidance";

// ── Mocks ───────────────────────────────────────────────────────────────────

const mockFetchProject = vi.fn();
const mockFetchEnvironment = vi.fn();
const mockCreateGuidance = vi.fn();
const mockValidateDraft = vi.fn();
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
  createGuidance: (...args: unknown[]) => mockCreateGuidance(...args),
  validateDraft: (...args: unknown[]) => mockValidateDraft(...args),
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
  description: "Surface damage classification",
  project_dir: "/tmp/workspace/projects/test-pid",
  created_at: "2026-04-13T10:00:00Z",
  updated_at: "2026-04-13T10:00:00Z",
  teacher_model_config_id: "mc-cosmos-8b",
  active_guidance_id: null,
  active_student_model_config_id: null,
});

const ENVIRONMENT = makeEnvironmentResponse({
  hosted_nim_available: true,
  nvidia_api_key_configured: true,
  recommended_teacher_mode: "hosted",
  recommended_embedding_mode: "hosted",
});

/** Wire-mock the draft-validation endpoint — the page's ONLY validator. */
function installFakeValidateDraft() {
  mockValidateDraft.mockImplementation((_projectId: unknown, body: unknown) =>
    Promise.resolve(fakeValidateDraft(body as DraftValidationRequest)),
  );
}

const GUIDANCE_RESPONSE = {
  guidance_id: "guid-001",
  project_id: "test-pid",
  version_number: 1,
  description: "Classify each image into one category.",
  schema_fields: [],
  rules: "",
  derived_json_schema: {},
  generation_order: [],
  schema_hash: "abc123",
  created_at: "2026-04-13T12:00:00Z",
};

// ── KUI Select test helper ──────────────────────────────────────────────────

/**
 * Click a KUI Select trigger, then click the option by label text.
 * KUI Select hoists data-testid to the trigger button itself (role="combobox").
 */
async function selectKuiOption(
  user: ReturnType<typeof userEvent.setup>,
  testId: string,
  optionLabel: string,
) {
  const trigger = screen.getByTestId(testId);
  await user.click(trigger);
  const option = await screen.findByRole("option", { name: optionLabel });
  await user.click(option);
}

// ── Template/type label lookups (value → display label for KUI Select) ────
// Derived from the product modules that feed the dropdowns: these tables
// exist to *pick options*, not to assert label text (the label contract is
// covered by lib/__tests__/guidance-templates.test.ts).
const TEMPLATE_LABELS: Record<string, string> = Object.fromEntries(
  GUIDANCE_TEMPLATES.map((t) => [t.name, t.label]),
);

const TYPE_LABELS: Record<string, string> = Object.fromEntries(
  FIELD_TYPE_OPTIONS.map((o) => [o.value, o.label]),
);

// ── Wrapper ─────────────────────────────────────────────────────────────────

function createWrapper(
  initialEntries: string[] = ["/projects/test-pid/create-guidance"],
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={initialEntries}>
          <Routes>
            <Route path="/projects/:projectId" element={<ProjectSetupLayout />}>
              <Route path="create-guidance" element={<CreateGuidancePage />} />
              <Route path="ready" element={<div data-testid="ready-page">Ready</div>} />
              <Route
                path="labeling"
                element={<div data-testid="labeling-page">Labeling</div>}
              />
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
    expect(screen.getByTestId("create-guidance-page")).toBeInTheDocument();
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// Page structure, cards, template selector
// ═══════════════════════════════════════════════════════════════════════════

describe("CreateGuidancePage — Page Structure", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(ENVIRONMENT);
    installFakeValidateDraft();
  });

  it("renders page chrome: header, edit-policy banner, section headings", async () => {
    await renderPage();
    expect(screen.getByText("Create Guidance")).toBeInTheDocument();
    expect(
      within(screen.getByTestId("create-guidance-page")).getByText(
        "Project: Damage Inspection",
      ),
    ).toBeInTheDocument();
    expect(screen.getByTestId("edit-policy-banner")).toHaveTextContent(
      /Core field changes/,
    );
    expect(screen.getByText("Core Fields")).toBeInTheDocument();
    expect(screen.getByText("Aux Fields")).toBeInTheDocument();
    expect(screen.getByText("Rules & Edge Cases")).toBeInTheDocument();
  });

  it("names the free-form Guidance controls for assistive technology", async () => {
    await renderPage();
    expect(screen.getByRole("combobox", { name: "Guidance starter" })).toBe(
      screen.getByTestId("template-selector"),
    );
    expect(screen.getByRole("textbox", { name: "Task description" })).toBe(
      screen.getByTestId("description-textarea"),
    );
    expect(screen.getByRole("textbox", { name: "Rules and edge cases" })).toBe(
      screen.getByTestId("rules-textarea"),
    );
  });

  it("shows neutral required-count in status badge when blank template (pre-save)", async () => {
    await renderPage();
    // The badge appears once the (debounced) backend validation responds.
    // Blank template's only structural requirement is NO_CORE_FIELDS.
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent(
        "Schema: 1 required",
      );
    });
  });

  it("flips badge from neutral 'N required' to red 'N errors' after a save attempt", async () => {
    await renderPage();
    // Cold start: neutral required-count, not error language.
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent(
        "Schema: 1 required",
      );
    });
    expect(screen.getByTestId("status-badge")).not.toHaveTextContent("error");
    // Click Save with errors present — badge flips to live error count
    // (singular at exactly one error).
    const user = userEvent.setup();
    await user.click(screen.getByTestId("save-guidance-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Schema: 1 error");
    });
  });

  it("updates status badge to 'Valid' after selecting Classification template", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Schema: Valid");
    });
  });

  it("disables Save Guidance after a save attempt when errors still exist", async () => {
    await renderPage();
    const btn = screen.getByTestId("save-guidance-btn");
    expect(btn).toHaveTextContent("Save Guidance");
    // Pre-save: button is enabled so the SME can click to reveal inline errors.
    expect(btn).not.toBeDisabled();
    const user = userEvent.setup();
    await user.click(btn);
    // After a failed save attempt (backend re-validates on save), errors are
    // surfaced and the button disables.
    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).toBeDisabled();
    });
  });

  it("enables Save Guidance when core fields are present", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
  });

  it("defaults template selector to Blank", async () => {
    await renderPage();
    const trigger = screen.getByTestId("template-selector");
    expect(trigger).toHaveTextContent("Blank");
  });

  it("renders one option per defined template", async () => {
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("template-selector"));
    const options = await screen.findAllByRole("option");
    expect(options).toHaveLength(GUIDANCE_TEMPLATES.length);
  });

  it("does not offer the retired extraction, damage, recycling, or grocery templates", async () => {
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("template-selector"));
    expect(
      screen.queryByRole("option", { name: "Attribute extraction" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: "Damage severity assessment" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: "Recycling classification" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: "Coarse grocery classification" }),
    ).not.toBeInTheDocument();
  });

  it("pre-fills description and shows category field when Classification selected", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    expect(
      (screen.getByTestId("description-textarea") as HTMLTextAreaElement).value,
    ).toBe("Classify each image into one category.");
    expect(screen.getByTestId("field-row-category")).toBeInTheDocument();
  });

  it("Presence and count demonstrates boolean, bounded integer, and consistency rules", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["presence_count"]);
    expect(screen.getByTestId("field-row-target_present")).toBeInTheDocument();
    expect(
      within(screen.getByTestId("field-row-target_count")).getByDisplayValue("0"),
    ).toBeInTheDocument();
    expect((screen.getByTestId("rules-textarea") as HTMLTextAreaElement).value).toMatch(
      /target_count to 0/i,
    );
  });

  it("Packaging information audit shows its tested fields and dataset context", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(
      user,
      "template-selector",
      TEMPLATE_LABELS["packaging_audit"],
    );
    expect(screen.getByTestId("field-row-language_on_packaging")).toBeInTheDocument();
    expect(
      screen.getByTestId("field-row-contains_nutrition_table"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("template-dataset-metadata")).toHaveTextContent(
      /Open Food Facts images.*CC BY-SA \(images\)/,
    );
    expect(screen.getByRole("link", { name: /View dataset source/i })).toHaveAttribute(
      "href",
      "https://world.openfoodfacts.org/data",
    );
  });

  it("opens confirmation modal when template changed after user has edited", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    const textarea = screen.getByTestId("description-textarea") as HTMLTextAreaElement;
    await user.type(textarea, " Extra.");
    const editedValue = textarea.value;
    await selectKuiOption(
      user,
      "template-selector",
      TEMPLATE_LABELS["industrial_anomaly"],
    );
    expect(
      await screen.findByTestId("confirm-template-replace-body"),
    ).toHaveTextContent(/Industrial anomaly inspection/);
    expect(textarea.value).toBe(editedValue);
  });

  it("Cancel in the replace modal preserves edits and reverts the dropdown", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    const textarea = screen.getByTestId("description-textarea") as HTMLTextAreaElement;
    await user.type(textarea, " Extra.");
    const editedValue = textarea.value;
    await selectKuiOption(
      user,
      "template-selector",
      TEMPLATE_LABELS["industrial_anomaly"],
    );
    await user.click(await screen.findByTestId("cancel-template-replace-btn"));
    expect(textarea.value).toBe(editedValue);
    expect(screen.getByTestId("template-selector")).toHaveTextContent(
      TEMPLATE_LABELS["classification"],
    );
  });

  // Timeout note: the two template-switch tests below chain several KUI
  // Select interactions plus typing; under full-suite parallel load they run
  // 5-6s and flake against the 5s default, while passing in ~2-3s isolated.
  it(
    "Replace in the modal applies the new template and clears user-edited state",
    { timeout: 15_000 },
    async () => {
      await renderPage();
      const user = userEvent.setup();
      await selectKuiOption(
        user,
        "template-selector",
        TEMPLATE_LABELS["classification"],
      );
      const textarea = screen.getByTestId(
        "description-textarea",
      ) as HTMLTextAreaElement;
      await user.type(textarea, " Extra.");
      await selectKuiOption(
        user,
        "template-selector",
        TEMPLATE_LABELS["industrial_anomaly"],
      );
      await user.click(await screen.findByTestId("confirm-template-replace-btn"));
      expect(screen.getByTestId("field-row-object_category")).toBeInTheDocument();
      expect(screen.getByTestId("field-row-has_anomaly")).toBeInTheDocument();
      // hasUserEdited cleared: picking another template now works without a second modal.
      await selectKuiOption(
        user,
        "template-selector",
        TEMPLATE_LABELS["classification"],
      );
      expect(screen.getByTestId("field-row-category")).toBeInTheDocument();
      expect(
        screen.queryByTestId("confirm-template-replace-body"),
      ).not.toBeInTheDocument();
    },
  );

  it(
    "switching templates without editing applies the new template directly (no modal)",
    { timeout: 15_000 },
    async () => {
      await renderPage();
      const user = userEvent.setup();
      await selectKuiOption(
        user,
        "template-selector",
        TEMPLATE_LABELS["classification"],
      );
      expect(screen.getByTestId("field-row-category")).toBeInTheDocument();
      await selectKuiOption(
        user,
        "template-selector",
        TEMPLATE_LABELS["industrial_anomaly"],
      );
      expect(
        screen.queryByTestId("confirm-template-replace-body"),
      ).not.toBeInTheDocument();
      expect(screen.getByTestId("field-row-object_category")).toBeInTheDocument();
    },
  );

  it("Industrial anomaly template shows its evidence-backed Core fields and context", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(
      user,
      "template-selector",
      TEMPLATE_LABELS["industrial_anomaly"],
    );
    expect(screen.getByTestId("field-row-object_category")).toBeInTheDocument();
    expect(screen.getByTestId("field-row-has_anomaly")).toBeInTheDocument();
    expect(screen.queryByTestId("field-row-observed_defect")).not.toBeInTheDocument();
    expect((screen.getByTestId("rules-textarea") as HTMLTextAreaElement).value).toMatch(
      /visible surface or structural flaw/i,
    );
    expect(screen.getByTestId("template-dataset-metadata")).toHaveTextContent(
      /Visual Anomaly \(VisA\).*CC BY 4.0/,
    );
    expect(screen.getByRole("link", { name: /View dataset source/i })).toHaveAttribute(
      "href",
      "https://registry.opendata.aws/visa/",
    );
    expect(screen.queryByTestId("field-row-rationale_note")).not.toBeInTheDocument();
    expect(
      screen.getByRole("switch", { name: "Enable rationale notes" }),
    ).not.toBeChecked();
  });

  it("focuses Description textarea on first load", async () => {
    await renderPage();
    expect(document.activeElement).toBe(screen.getByTestId("description-textarea"));
  });

  it("Cancel navigates back to the previous route", async () => {
    // Seed a two-entry history so navigate(-1) has somewhere to go.
    const { Wrapper } = createWrapper([
      "/projects/test-pid/ready",
      "/projects/test-pid/create-guidance",
    ]);
    render(<div />, { wrapper: Wrapper });
    await waitFor(() => {
      expect(screen.getByTestId("create-guidance-page")).toBeInTheDocument();
    });
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /Cancel/i }));
    expect(await screen.findByTestId("ready-page")).toBeInTheDocument();
  });

  it("shows allowed values as chips for enum fields", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    const row = screen.getByTestId("field-row-category");
    expect(within(row).getByText("replace_me_a")).toBeInTheDocument();
  });

  it("shows min/max inputs for integer fields", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    const row = screen.getByTestId("field-row-category");
    await user.click(within(row).getByRole("combobox"));
    await user.click(
      await screen.findByRole("option", { name: TYPE_LABELS["integer"] }),
    );
    expect(within(row).getByLabelText("Min:")).toBeInTheDocument();
    expect(within(row).getByLabelText("Max:")).toBeInTheDocument();
  });

  it("keeps string length constraints collapsed and boolean fields constraint-free", async () => {
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    const row = screen.getByTestId("field-row-");

    // New fields start as strings. Their uncommon length constraints stay
    // behind an explicit disclosure instead of adding permanent form noise.
    expect(within(row).getByText("Advanced constraints")).toBeInTheDocument();
    expect(within(row).queryByLabelText("minLength:")).not.toBeInTheDocument();
    expect(within(row).queryByLabelText("maxLength:")).not.toBeInTheDocument();
    await user.click(within(row).getByText("Advanced constraints"));
    expect(within(row).getByLabelText("minLength:")).toBeInTheDocument();
    expect(within(row).getByLabelText("maxLength:")).toBeInTheDocument();

    // Boolean has no meaningful constraint inputs. Changing type must also
    // remove the string disclosure rather than leaving stale controls behind.
    await user.click(within(row).getByRole("combobox"));
    await user.click(
      await screen.findByRole("option", { name: TYPE_LABELS["boolean"] }),
    );
    expect(within(row).queryByText("Advanced constraints")).not.toBeInTheDocument();
    expect(within(row).queryByLabelText("minLength:")).not.toBeInTheDocument();
    expect(within(row).queryByLabelText("maxLength:")).not.toBeInTheDocument();
    expect(within(row).queryByLabelText("Min:")).not.toBeInTheDocument();
    expect(within(row).queryByLabelText("Max:")).not.toBeInTheDocument();
  });

  it("shows 'No core fields defined' for blank template", async () => {
    await renderPage();
    expect(screen.getByText("No core fields defined.")).toBeInTheDocument();
  });

  it("stays Valid with Save enabled when description is empty but core fields exist", async () => {
    // Task Description is optional: an empty description is not a validation
    // issue, so a draft with core fields and no description saves cleanly.
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await user.clear(screen.getByTestId("description-textarea"));
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Schema: Valid");
    });
    expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Interactive field builder
// ═══════════════════════════════════════════════════════════════════════════

describe("CreateGuidancePage — Field Builder", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(ENVIRONMENT);
    installFakeValidateDraft();
  });

  it("Add Core Field creates row with empty name and type string", async () => {
    await renderPage();
    expect(screen.getByTestId("add-core-field-btn")).toHaveTextContent(
      "Add Core Field",
    );
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    const row = screen.getByTestId("field-row-");
    expect(within(row).getByPlaceholderText("field_name")).toBeInTheDocument();
  });

  it("Add Core Field appends the new row after existing core fields", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(
      user,
      "template-selector",
      TEMPLATE_LABELS["industrial_anomaly"],
    );
    await user.click(screen.getByTestId("add-core-field-btn"));
    // The new (empty-name) row must land last in the core section — not be
    // slotted between existing rows by the display-order recalc.
    const rowIds = screen
      .getAllByTestId(/^field-row-/)
      .map((el) => el.getAttribute("data-testid"));
    expect(rowIds.indexOf("field-row-")).toBeGreaterThan(
      rowIds.indexOf("field-row-has_anomaly"),
    );
  });

  it("Add Aux Field creates row in aux section", async () => {
    await renderPage();
    expect(screen.getByTestId("add-aux-field-btn")).toHaveTextContent("Add Aux Field");
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await user.click(screen.getByTestId("add-aux-field-btn"));
    const row = screen.getByTestId("field-row-");
    expect(within(row).getByText("aux")).toBeInTheDocument();
  });

  it("adding a core field removes 'No core fields defined'", async () => {
    await renderPage();
    expect(screen.getByText("No core fields defined.")).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    expect(screen.queryByText("No core fields defined.")).not.toBeInTheDocument();
  });

  it("adding a field sets hasUserEdited (template no longer overwrites)", async () => {
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    expect(
      (screen.getByTestId("description-textarea") as HTMLTextAreaElement).value,
    ).toBe("");
  });

  it("delete removes a field row", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    const row = screen.getByTestId("field-row-category");
    await user.click(within(row).getByLabelText("Delete field"));
    expect(screen.queryByTestId("field-row-category")).not.toBeInTheDocument();
  });

  it("typing in name input updates the field name", async () => {
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    await user.type(
      within(screen.getByTestId("field-row-")).getByPlaceholderText("field_name"),
      "my_field",
    );
    expect(screen.getByTestId("field-row-my_field")).toBeInTheDocument();
  });

  it("changing type clears previous constraints", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    const row = screen.getByTestId("field-row-category");
    expect(within(row).getByText("replace_me_a")).toBeInTheDocument();
    const typeTrigger = within(row).getByRole("combobox");
    await user.click(typeTrigger);
    await user.click(
      await screen.findByRole("option", { name: TYPE_LABELS["integer"] }),
    );
    expect(within(row).queryByText("replace_me_a")).not.toBeInTheDocument();
    expect(within(row).getByText("Min:")).toBeInTheDocument();
  });

  it("[+] → type → Enter adds a chip", async () => {
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    const row = screen.getByTestId("field-row-");
    const typeTrigger2 = within(row).getByRole("combobox");
    await user.click(typeTrigger2);
    await user.click(await screen.findByRole("option", { name: TYPE_LABELS["enum"] }));
    await user.click(within(row).getByText("add"));
    await user.type(
      within(row).getByPlaceholderText("Type a value and press Enter"),
      "my_value{enter}",
    );
    expect(within(row).getByText("my_value")).toBeInTheDocument();
  });

  it("[x] on chip removes it", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    const row = screen.getByTestId("field-row-category");
    await user.click(within(row).getByLabelText("Remove value replace_me_a"));
    expect(within(row).queryByText("replace_me_a")).not.toBeInTheDocument();
  });

  it("Move to Aux moves core field to aux section", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await user.click(
      within(screen.getByTestId("field-row-category")).getByLabelText("Move to Aux"),
    );
    expect(
      within(screen.getByTestId("field-row-category")).getByText("aux"),
    ).toBeInTheDocument();
  });

  it("Move-up reorders within section", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(
      user,
      "template-selector",
      TEMPLATE_LABELS["industrial_anomaly"],
    );
    const first = screen.getByTestId("field-row-object_category");
    expect(within(first).getByLabelText("Move field up")).toBeDisabled();
    const second = screen.getByTestId("field-row-has_anomaly");
    await user.click(within(second).getByLabelText("Move field up"));
    expect(
      within(screen.getByTestId("field-row-has_anomaly")).getByLabelText(
        "Move field up",
      ),
    ).toBeDisabled();
  });

  it("rationale notes are opt-in and can be toggled freely", async () => {
    await renderPage();
    const user = userEvent.setup();
    const toggle = screen.getByRole("switch", { name: "Enable rationale notes" });
    expect(toggle).not.toBeChecked();
    await user.click(toggle);
    expect(toggle).toBeChecked();
    await user.click(toggle);
    expect(toggle).not.toBeChecked();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Validation, previews, save flow
// ═══════════════════════════════════════════════════════════════════════════

describe("CreateGuidancePage — Validation, Previews, Save", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchProject.mockResolvedValue(PROJECT);
    mockFetchEnvironment.mockResolvedValue(ENVIRONMENT);
    installFakeValidateDraft();
    mockCreateGuidance.mockResolvedValue(GUIDANCE_RESPONSE);
    mockUpdateProject.mockResolvedValue(PROJECT);
  });

  // ── Local validation error display ────────────────────────────────────
  // Inline field-level errors do NOT appear on load — they are revealed
  // only after the SME clicks Save with errors present. The header status
  // badge stays live from the first render.

  it("hides NO_CORE_FIELDS error until Save is attempted, then shows it inline", async () => {
    await renderPage();
    expect(screen.queryByTestId("error-NO_CORE_FIELDS")).not.toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("save-guidance-btn"));
    expect(await screen.findByTestId("error-NO_CORE_FIELDS")).toBeInTheDocument();
  });

  it("shows ENUM_TOO_FEW_VALUES inline under the field after Save", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    // Remove one of the two values from category enum
    const row = screen.getByTestId("field-row-category");
    await user.click(within(row).getByLabelText("Remove value replace_me_a"));
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).not.toHaveTextContent("Schema: Valid");
    });
    // Click Save — error should appear inline.
    await user.click(screen.getByTestId("save-guidance-btn"));
    expect(await screen.findByTestId("error-ENUM_TOO_FEW_VALUES")).toBeInTheDocument();
  });

  it("shows INVALID_FIELD_NAME inline under the name input after Save", async () => {
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    const row = screen.getByTestId("field-row-");
    await user.type(within(row).getByPlaceholderText("field_name"), "1bad");
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).not.toHaveTextContent("Schema: Valid");
    });
    await user.click(screen.getByTestId("save-guidance-btn"));
    expect(await screen.findByTestId("error-INVALID_FIELD_NAME")).toBeInTheDocument();
  });

  it("shows DUPLICATE_FIELD_NAME inline under the name input after Save", async () => {
    await renderPage();
    const user = userEvent.setup();
    // Add two core fields and give them the same name
    await user.click(screen.getByTestId("add-core-field-btn"));
    await user.click(screen.getByTestId("add-core-field-btn"));
    const emptyRows = screen.getAllByTestId("field-row-");
    await user.type(within(emptyRows[0]).getByPlaceholderText("field_name"), "dup");
    await user.type(within(emptyRows[1]).getByPlaceholderText("field_name"), "dup");
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).not.toHaveTextContent("Schema: Valid");
    });
    await user.click(screen.getByTestId("save-guidance-btn"));
    expect(await screen.findByTestId("error-DUPLICATE_FIELD_NAME")).toBeInTheDocument();
  });

  it("hides the field-name helper when INVALID_FIELD_NAME is showing (but not on MISSING_FIELD_NAME)", async () => {
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    // Pre-save: helper text is visible alongside the empty name input.
    const helperBefore = screen.getByText(/Letters, numbers, and underscores only/);
    expect(helperBefore).toBeInTheDocument();
    // Empty-name save: MISSING_FIELD_NAME fires, but helper stays (complementary info).
    await user.click(screen.getByTestId("save-guidance-btn"));
    expect(await screen.findByTestId("error-MISSING_FIELD_NAME")).toBeInTheDocument();
    expect(
      screen.getByText(/Letters, numbers, and underscores only/),
    ).toBeInTheDocument();
    // Now fill an invalid-format name — INVALID_FIELD_NAME replaces MISSING, helper hides.
    const row = screen.getByTestId("field-row-");
    await user.type(within(row).getByPlaceholderText("field_name"), "1bad");
    expect(await screen.findByTestId("error-INVALID_FIELD_NAME")).toBeInTheDocument();
    expect(
      screen.queryByText(/Letters, numbers, and underscores only/),
    ).not.toBeInTheDocument();
  });

  it("clicking the status badge reveals inline errors and scrolls to the first one", async () => {
    const scrollIntoViewMock = vi.fn();
    // jsdom lacks scrollIntoView; stub it so the click handler does not throw.
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: scrollIntoViewMock,
    });
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    const row = screen.getByTestId("field-row-");
    await user.type(within(row).getByPlaceholderText("field_name"), "bad name");
    // No Save click — errors should still be hidden at this point.
    expect(screen.queryByTestId("error-INVALID_FIELD_NAME")).not.toBeInTheDocument();
    // Click the badge button (it renders once backend validation responds).
    await user.click(await screen.findByTestId("status-badge-button"));
    expect(await screen.findByTestId("error-INVALID_FIELD_NAME")).toBeInTheDocument();
    await waitFor(() => {
      expect(scrollIntoViewMock).toHaveBeenCalled();
    });
  });

  // ── Fix-it buttons ────────────────────────────────────────────────────

  it("MIN_EXCEEDS_MAX fix-it swaps min and max values", async () => {
    await renderPage();
    const user = userEvent.setup();
    await user.click(screen.getByTestId("add-core-field-btn"));
    const row = screen.getByTestId("field-row-");
    await user.type(within(row).getByPlaceholderText("field_name"), "score");
    const namedRow = screen.getByTestId("field-row-score");
    const typeTrigger = within(namedRow).getByRole("combobox");
    await user.click(typeTrigger);
    await user.click(
      await screen.findByRole("option", { name: TYPE_LABELS["integer"] }),
    );
    // Set min > max to trigger the error
    const minInput = within(namedRow).getByLabelText("Min:") as HTMLInputElement;
    const maxInput = within(namedRow).getByLabelText("Max:") as HTMLInputElement;
    await user.type(minInput, "10");
    await user.type(maxInput, "5");
    // Inline errors (and their fix-its) are revealed by a save attempt.
    await user.click(screen.getByTestId("save-guidance-btn"));
    await user.click(await screen.findByTestId("fix-MIN_EXCEEDS_MAX"));
    expect(minInput.value).toBe("5");
    expect(maxInput.value).toBe("10");
  });

  // ── Previews ──────────────────────────────────────────────────────────

  it("JSON Schema preview is collapsed by default and toggles open", async () => {
    await renderPage();
    const user = userEvent.setup();
    expect(screen.queryByTestId("json-schema-preview")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("json-schema-toggle"));
    expect(screen.getByTestId("json-schema-preview")).toBeInTheDocument();
  });

  it("Example output preview is collapsed by default and toggles open", async () => {
    await renderPage();
    const user = userEvent.setup();
    expect(screen.queryByTestId("example-output-preview")).not.toBeInTheDocument();
    await user.click(screen.getByTestId("example-output-toggle"));
    expect(screen.getByTestId("example-output-preview")).toBeInTheDocument();
  });

  it("JSON Schema preview shows backend-derived schema when valid", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    // Wait for backend validation of the template draft to complete (the
    // badge flips to Valid on its response).
    await waitFor(() => {
      expect(screen.getByTestId("status-badge")).toHaveTextContent("Schema: Valid");
    });
    await user.click(screen.getByTestId("json-schema-toggle"));
    const preview = screen.getByTestId("json-schema-preview");
    expect(preview.textContent).toContain('"type"');
  });

  it("Example output preview shows the applied template's Core fields", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(
      user,
      "template-selector",
      TEMPLATE_LABELS["industrial_anomaly"],
    );
    await user.click(screen.getByTestId("example-output-toggle"));
    const preview = screen.getByTestId("example-output-preview");
    expect(preview.textContent).toContain('"object_category"');
    expect(preview.textContent).toContain('"has_anomaly"');
  });

  // ── Debounced backend validation ──────────────────────────────────────

  it("calls validateDraft (debounced) after edits", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    await renderPage();
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    // Wait for debounced validation
    vi.advanceTimersByTime(600);
    await waitFor(() => {
      expect(mockValidateDraft).toHaveBeenCalledWith(
        "test-pid",
        expect.objectContaining({
          description: "Classify each image into one category.",
        }),
      );
    });
    vi.useRealTimers();
  });

  it("calls validateDraft even when the draft has errors — the backend is the only validator", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    await renderPage();
    // Blank template (NO_CORE_FIELDS): the backend is still
    // consulted so its issues drive the badge and inline errors.
    vi.advanceTimersByTime(600);
    await waitFor(() => {
      expect(mockValidateDraft).toHaveBeenCalledWith(
        "test-pid",
        expect.objectContaining({ description: "" }),
      );
    });
    vi.useRealTimers();
  });

  // ── Save flow ─────────────────────────────────────────────────────────

  it("Save calls createGuidance then updateProject on success", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));
    await waitFor(() => {
      expect(mockCreateGuidance).toHaveBeenCalledWith(
        "test-pid",
        expect.objectContaining({
          description: "Classify each image into one category.",
          rules: expect.stringMatching(/Replace the starter category values/),
          schema: expect.not.arrayContaining([
            expect.objectContaining({ field_name: "rationale_note" }),
          ]),
        }),
      );
    });
    await waitFor(() => {
      expect(mockUpdateProject).toHaveBeenCalledWith("test-pid", {
        active_guidance_id: "guid-001",
      });
    });
  });

  it("includes rationale_note in the saved schema only after opt-in", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await user.click(screen.getByRole("switch", { name: "Enable rationale notes" }));
    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));
    await waitFor(() => {
      expect(mockCreateGuidance).toHaveBeenCalledWith(
        "test-pid",
        expect.objectContaining({
          schema: expect.arrayContaining([
            expect.objectContaining({
              field_name: "rationale_note",
              role: "aux",
              type: "string",
            }),
          ]),
        }),
      );
    });
  });

  it("shows success toast after save", async () => {
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("save-toast")).toHaveTextContent("Guidance v1 saved.");
    });
  });

  it("shows error when save fails", async () => {
    mockCreateGuidance.mockRejectedValueOnce(new Error("Server error"));
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("save-error")).toBeInTheDocument();
    });
  });

  it("shows 'Saving...' text while save is in progress", async () => {
    // Make createGuidance hang to test loading state
    mockCreateGuidance.mockImplementation(() => new Promise(() => {}));
    await renderPage();
    const user = userEvent.setup();
    await selectKuiOption(user, "template-selector", TEMPLATE_LABELS["classification"]);
    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).not.toBeDisabled();
    });
    await user.click(screen.getByTestId("save-guidance-btn"));
    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).toHaveTextContent("Saving...");
    });
  });

  // ── Accessibility ─────────────────────────────────────────────────────

  it("has aria-live region announcing validation status", async () => {
    await renderPage();
    const liveRegion = screen.getByTestId("aria-live-region");
    // Announced once the (debounced) backend validation responds; a single
    // issue is announced in the singular ("1 validation error").
    await waitFor(() => {
      expect(liveRegion).toHaveTextContent("1 validation error");
    });
    expect(liveRegion).not.toHaveTextContent("errors");
  });

  it("Save button has aria-disabled after a save attempt when errors exist", async () => {
    await renderPage();
    const btn = screen.getByTestId("save-guidance-btn");
    // Pre-save: clickable + aria-disabled=false so the SME can reveal inline errors.
    expect(btn).toHaveAttribute("aria-disabled", "false");
    const user = userEvent.setup();
    await user.click(btn);
    await waitFor(() => {
      expect(screen.getByTestId("save-guidance-btn")).toHaveAttribute(
        "aria-disabled",
        "true",
      );
    });
  });

  it("card headings are focusable", async () => {
    await renderPage();
    const heading = screen.getByText("Task Description");
    expect(heading.getAttribute("tabindex")).toBe("-1");
  });

  // ── SCHEMA_COMPILE_FAILURE from backend ───────────────────────────────

  it("shows SCHEMA_COMPILE_FAILURE banner when backend reports it", async () => {
    mockValidateDraft.mockResolvedValue({
      issues: [
        {
          severity: "error",
          code: "SCHEMA_COMPILE_FAILURE",
          message: "Internal error",
        },
      ],
      derived_json_schema: null,
      schema_hash: null,
      save_allowed: false,
    });
    await renderPage();
    // The banner is not gated on a save attempt — it appears as soon as the
    // debounced validation reports the failure.
    expect(
      await screen.findByTestId("error-SCHEMA_COMPILE_FAILURE"),
    ).toBeInTheDocument();
  });
});

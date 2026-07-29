// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for guidance template definitions.
 */

import { describe, it, expect } from "vitest";
import {
  GUIDANCE_TEMPLATES,
  DEFAULT_TEMPLATE_NAME,
  RATIONALE_NOTE_FIELD_NAME,
  getTemplateByName,
} from "@/lib/guidance-templates";

/**
 * What is deliberately NOT tested here: template count, name list, and
 * per-template field literals — those tests were byte-for-byte re-encodings
 * of the GUIDANCE_TEMPLATES literal and failed only when someone edited the
 * catalog on purpose. The behavioral half (selecting a template populates
 * the builder form, all templates appear in the selector, Blank is the
 * default) lives in pages/__tests__/CreateGuidancePage.test.tsx. This file
 * keeps only the cross-template invariants and the lookup-function
 * behavior.
 */
describe("guidance-templates", () => {
  it("templates leave rationale notes disabled by default", () => {
    for (const t of GUIDANCE_TEMPLATES) {
      const rn = t.fields.find((f) => f.field_name === RATIONALE_NOTE_FIELD_NAME);
      expect(rn, `${t.name} unexpectedly enables rationale_note`).toBeUndefined();
    }
  });

  it("the default template is a usable blank slate with no predefined fields", () => {
    const blank = getTemplateByName(DEFAULT_TEMPLATE_NAME);
    expect(blank.description).toBe("");
    expect(blank.fields).toHaveLength(0);
  });

  it("getTemplateByName returns correct template", () => {
    const t = getTemplateByName("classification");
    expect(t.name).toBe("classification");
    expect(t.label).toBe("Classification");
  });

  it("the bundled-image template produces the RPS classification contract", () => {
    const template = getTemplateByName("rock_paper_scissors");
    expect(template.description).toMatch(/rock, paper, or scissors/i);
    expect(template.fields).toEqual([
      expect.objectContaining({
        field_name: "category",
        type: "enum",
        role: "core",
        allowed_values: ["rock", "paper", "scissors"],
      }),
    ]);
  });

  it("getTemplateByName throws for unknown name", () => {
    expect(() => getTemplateByName("nonexistent" as never)).toThrow("Unknown template");
  });
});

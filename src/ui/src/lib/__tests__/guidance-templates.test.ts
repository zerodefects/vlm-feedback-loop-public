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
 * What is deliberately NOT tested here: the exact template count, complete
 * name list, and every field literal — those tests were byte-for-byte
 * re-encodings of GUIDANCE_TEMPLATES and failed only when someone edited the
 * catalog on purpose. Selector behavior and the absence of retired choices
 * live in pages/__tests__/CreateGuidancePage.test.tsx. This file keeps only
 * cross-template invariants, the dataset-backed contracts, and
 * lookup-function behavior.
 */
describe("guidance-templates", () => {
  it("templates leave rationale notes disabled by default", () => {
    for (const t of GUIDANCE_TEMPLATES) {
      const rn = t.fields.find((f) => f.field_name === RATIONALE_NOTE_FIELD_NAME);
      expect(rn, `${t.name} unexpectedly enables rationale_note`).toBeUndefined();
    }
  });

  it("non-blank templates model edge-case handling with starter rules", () => {
    for (const template of GUIDANCE_TEMPLATES) {
      if (template.name === "blank") continue;
      expect(template.rules.trim(), `${template.name} has no starter rules`).not.toBe(
        "",
      );
    }
  });

  it("the default template is a usable blank slate with no predefined fields", () => {
    const blank = getTemplateByName(DEFAULT_TEMPLATE_NAME);
    expect(blank.description).toBe("");
    expect(blank.fields).toHaveLength(0);
    expect(blank.rules).toBe("");
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
    expect(template.rules).toMatch(/closed fist/i);
    expect(template.dataset).toEqual(expect.objectContaining({ license: "CC BY 2.0" }));
  });

  it("the presence-and-count template models a consistent boolean and integer pair", () => {
    const template = getTemplateByName("presence_count");
    expect(template.fields).toEqual([
      expect.objectContaining({
        field_name: "target_present",
        type: "boolean",
        role: "core",
      }),
      expect.objectContaining({
        field_name: "target_count",
        type: "integer",
        role: "core",
        minimum: 0,
      }),
    ]);
    expect(template.rules).toMatch(/target_count to 0/i);
  });

  it("the packaging audit uses the tested controlled-language and panel contract", () => {
    const template = getTemplateByName("packaging_audit");
    expect(template.fields).toEqual([
      expect.objectContaining({
        field_name: "language_on_packaging",
        type: "enum",
        role: "core",
        allowed_values: ["fr", "en", "es", "de", "it", "nl", "other"],
      }),
      expect.objectContaining({
        field_name: "contains_nutrition_table",
        type: "boolean",
        role: "core",
      }),
    ]);
    expect(template.dataset).toEqual(
      expect.objectContaining({
        name: "Open Food Facts images",
        license: "CC BY-SA (images)",
      }),
    );
  });

  it("the VisA template keeps only the two reliable evaluated fields", () => {
    const template = getTemplateByName("industrial_anomaly");
    expect(template.fields).toEqual([
      expect.objectContaining({
        field_name: "object_category",
        type: "enum",
        role: "core",
      }),
      expect.objectContaining({
        field_name: "has_anomaly",
        type: "enum",
        role: "core",
        allowed_values: ["no", "yes"],
      }),
    ]);
    expect(template.fields.every((field) => field.role === "core")).toBe(true);
    expect(template.rules).toMatch(/Do not treat pose, lighting/i);
    expect(template.dataset).toEqual(
      expect.objectContaining({
        name: "Visual Anomaly (VisA)",
        license: "CC BY 4.0",
        sourceUrl: "https://registry.opendata.aws/visa/",
      }),
    );
  });

  it("getTemplateByName throws for unknown name", () => {
    expect(() => getTemplateByName("nonexistent" as never)).toThrow("Unknown template");
  });
});

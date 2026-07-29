// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Guidance template definitions for the "Start from:" selector.
 *
 * Pure data — no React dependencies.
 */

import type { SchemaFieldInput } from "@/types/guidance";

// ── Constants ───────────────────────────────────────────────────────────────

export const RATIONALE_NOTE_FIELD_NAME = "rationale_note";

export type TemplateName =
  | "blank"
  | "classification"
  | "rock_paper_scissors"
  | "multi_label"
  | "attribute_extraction"
  | "damage_severity"
  | "recycling"
  | "grocery";

export const DEFAULT_TEMPLATE_NAME: TemplateName = "blank";

export interface GuidanceTemplate {
  name: TemplateName;
  /** Display label in the dropdown. */
  label: string;
  /** Pre-fills the Description textarea. */
  description: string;
  /** Pre-fills the SchemaCore fields. */
  fields: SchemaFieldInput[];
}

// ── Templates ───────────────────────────────────────────────────────────────

export const GUIDANCE_TEMPLATES: readonly GuidanceTemplate[] = [
  {
    name: "blank",
    label: "Blank",
    description: "",
    fields: [],
  },
  {
    name: "classification",
    label: "Classification",
    description: "Classify each image into one category.",
    fields: [
      {
        field_name: "category",
        type: "enum",
        role: "core",
        allowed_values: ["category_a", "category_b"],
        display_order: 0,
      },
    ],
  },
  {
    name: "rock_paper_scissors",
    label: "Rock, paper, scissors",
    description: "Classify the hand gesture in each image as rock, paper, or scissors.",
    fields: [
      {
        field_name: "category",
        type: "enum",
        role: "core",
        allowed_values: ["rock", "paper", "scissors"],
        display_order: 0,
      },
    ],
  },
  {
    name: "multi_label",
    label: "Multi-label classification",
    description: "Select all labels that apply to each image.",
    fields: [
      {
        field_name: "labels",
        type: "enum_set",
        role: "core",
        allowed_values: ["label_a", "label_b"],
        display_order: 0,
      },
    ],
  },
  {
    name: "attribute_extraction",
    label: "Attribute extraction",
    description: "Extract structured attributes from each image.",
    fields: [
      {
        field_name: "attribute_1",
        type: "string",
        role: "core",
        display_order: 0,
      },
      {
        field_name: "attribute_2",
        type: "boolean",
        role: "core",
        display_order: 1,
      },
    ],
  },
  {
    name: "damage_severity",
    label: "Damage severity assessment",
    description:
      "Classify visible damage types, identify the primary damage, and rate overall severity.",
    fields: [
      {
        field_name: "primary_damage_type",
        type: "enum",
        role: "core",
        allowed_values: ["crush", "rip", "tear", "leak", "dent", "scratch"],
        display_order: 0,
      },
      {
        field_name: "damage_types_present",
        type: "enum_set",
        role: "core",
        allowed_values: ["crush", "rip", "tear", "leak", "dent", "scratch"],
        display_order: 1,
      },
      {
        field_name: "severity",
        type: "integer",
        role: "core",
        minimum: 0,
        maximum: 4,
        display_order: 2,
      },
      {
        field_name: "fragile_content",
        type: "boolean",
        role: "aux",
        display_order: 4,
      },
      {
        field_name: "hazmat_indicators",
        type: "boolean",
        role: "aux",
        display_order: 5,
      },
    ],
  },
  {
    // Matches the TrashNet dataset (~/trashnet): six recyclable-material classes.
    name: "recycling",
    label: "Recycling classification",
    description:
      "Classify each image of a discarded item into its recyclable material category.",
    fields: [
      {
        field_name: "material",
        type: "enum",
        role: "core",
        allowed_values: ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
        display_order: 0,
      },
    ],
  },
  {
    // Matches the Freiburg Groceries dataset (~/freiburg_groceries_dataset): 25
    // coarse product classes (class names verbatim from the dataset manifest).
    name: "grocery",
    label: "Coarse grocery classification",
    description: "Classify each grocery product image into one product category.",
    fields: [
      {
        field_name: "product_category",
        type: "enum",
        role: "core",
        allowed_values: [
          "BEANS",
          "CAKE",
          "CANDY",
          "CEREAL",
          "CHIPS",
          "CHOCOLATE",
          "COFFEE",
          "CORN",
          "FISH",
          "FLOUR",
          "HONEY",
          "JAM",
          "JUICE",
          "MILK",
          "NUTS",
          "OIL",
          "PASTA",
          "RICE",
          "SODA",
          "SPICES",
          "SUGAR",
          "TEA",
          "TOMATO_SAUCE",
          "VINEGAR",
          "WATER",
        ],
        display_order: 0,
      },
    ],
  },
] as const;

// ── Lookup ──────────────────────────────────────────────────────────────────

const templateMap = new Map<string, GuidanceTemplate>(
  GUIDANCE_TEMPLATES.map((t) => [t.name, t]),
);

export function getTemplateByName(name: TemplateName): GuidanceTemplate {
  const t = templateMap.get(name);
  if (!t) throw new Error(`Unknown template: ${name}`);
  return t;
}

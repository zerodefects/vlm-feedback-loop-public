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
  | "presence_count"
  | "packaging_audit"
  | "industrial_anomaly";

export const DEFAULT_TEMPLATE_NAME: TemplateName = "blank";

export interface GuidanceTemplate {
  name: TemplateName;
  /** Display label in the dropdown. */
  label: string;
  /** One-line explanation shown below the selector after this template is applied. */
  summary: string;
  /** Pre-fills the Description textarea. */
  description: string;
  /** Pre-fills the SchemaCore fields. */
  fields: SchemaFieldInput[];
  /** Pre-fills Rules & Edge Cases so templates demonstrate a complete Guidance. */
  rules: string;
  /** Optional provenance for a template designed around a public or bundled dataset. */
  dataset?: {
    name: string;
    detail: string;
    license: string;
    sourceUrl?: string;
  };
}

// ── Templates ───────────────────────────────────────────────────────────────

export const GUIDANCE_TEMPLATES: readonly GuidanceTemplate[] = [
  {
    name: "blank",
    label: "Blank",
    summary: "Build a task from an empty schema.",
    description: "",
    fields: [],
    rules: "",
  },
  {
    name: "classification",
    label: "Classification",
    summary: "Assign exactly one controlled category to each image.",
    description: "Classify each image into one category.",
    fields: [
      {
        field_name: "category",
        type: "enum",
        role: "core",
        allowed_values: ["replace_me_a", "replace_me_b"],
        display_order: 0,
      },
    ],
    rules:
      "Replace the starter category values with the task's real categories before saving. Assign exactly one category per image. If the image cannot be classified confidently, Skip it rather than inventing a category.",
  },
  {
    name: "rock_paper_scissors",
    label: "Rock, paper, scissors",
    summary: "Classify the hand gestures in the bundled first-run walkthrough.",
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
    rules:
      "Use rock for a closed fist, paper for an open hand with the fingers extended, and scissors for two extended separated fingers. Judge the primary foreground hand. Skip images where the gesture is too occluded or ambiguous to identify.",
    dataset: {
      name: "Bundled rock-paper-scissors sample",
      detail: "15 images for the first-run walkthrough",
      license: "CC BY 2.0",
    },
  },
  {
    name: "multi_label",
    label: "Multi-label classification",
    summary: "Apply every controlled label that is visibly supported.",
    description: "Select all labels that apply to each image.",
    fields: [
      {
        field_name: "labels",
        type: "enum_set",
        role: "core",
        allowed_values: ["replace_me_a", "replace_me_b"],
        display_order: 0,
      },
    ],
    rules:
      "Replace the starter label values with the task's real labels before saving. Select every visibly supported label, use an empty set when none apply, and do not add near-synonym labels that mean the same thing.",
  },
  {
    name: "presence_count",
    label: "Presence and count",
    summary: "Record whether a chosen target is visible and how many instances appear.",
    description:
      "Determine whether the target object is visible in each image and count the visible instances.",
    fields: [
      {
        field_name: "target_present",
        type: "boolean",
        role: "core",
        display_order: 0,
      },
      {
        field_name: "target_count",
        type: "integer",
        role: "core",
        minimum: 0,
        display_order: 1,
      },
    ],
    rules:
      "Replace 'target object' in the Description with the exact object to count before saving. Set target_present to true if and only if at least one identifiable instance is visible, and set target_count to 0 whenever target_present is false. Count a partially visible instance only when enough remains to identify it. Skip images when overlap or framing prevents a reliable count.",
  },
  {
    name: "packaging_audit",
    label: "Packaging information audit",
    summary:
      "Read visible package text and check whether a nutrition panel is present.",
    description:
      "Read each food-packaging photo, identify the dominant language of the visible text, and determine whether a nutrition-information panel is visible.",
    fields: [
      {
        field_name: "language_on_packaging",
        type: "enum",
        role: "core",
        allowed_values: ["fr", "en", "es", "de", "it", "nl", "other"],
        display_order: 0,
      },
      {
        field_name: "contains_nutrition_table",
        type: "boolean",
        role: "core",
        display_order: 1,
      },
    ],
    rules:
      "Choose language_on_packaging from the language used by most of the legible printed words; use other when it is not one of the listed languages. Set contains_nutrition_table to true only when a structured nutrition panel with energy and macronutrient rows is visibly identifiable. Skip the image when too little text is legible to determine its language.",
    dataset: {
      name: "Open Food Facts images",
      detail: "open product-packaging photos with extracted text",
      license: "CC BY-SA (images)",
      sourceUrl:
        "https://openfoodfacts.github.io/openfoodfacts-server/api/aws-images-dataset/",
    },
  },
  {
    name: "industrial_anomaly",
    label: "Industrial anomaly inspection",
    summary: "Identify the VisA object and decide whether it has a visible anomaly.",
    description:
      "Inspect each product image from the VisA dataset, identify the object category, and determine whether a visible manufacturing anomaly is present.",
    fields: [
      {
        field_name: "object_category",
        type: "enum",
        role: "core",
        allowed_values: [
          "candle",
          "capsules",
          "cashew",
          "chewinggum",
          "fryum",
          "macaroni1",
          "macaroni2",
          "pcb1",
          "pcb2",
          "pcb3",
          "pcb4",
          "pipe_fryum",
        ],
        display_order: 0,
      },
      {
        field_name: "has_anomaly",
        type: "enum",
        role: "core",
        allowed_values: ["no", "yes"],
        display_order: 1,
      },
    ],
    rules:
      "Set has_anomaly to yes only when a visible surface or structural flaw is present, such as a scratch, dent, color spot, crack, breakage, or missing or misplaced part. Do not treat pose, lighting, or ordinary appearance differences as defects. Use no when the item has no visible anomaly, and Skip when the image does not provide enough evidence to decide.",
    dataset: {
      name: "Visual Anomaly (VisA)",
      detail: "10,821 images across 12 object categories",
      license: "CC BY 4.0",
      sourceUrl: "https://github.com/amazon-science/spot-diff",
    },
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

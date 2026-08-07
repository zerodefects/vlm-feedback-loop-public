// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ProposalForm } from "@/components/labeling/ProposalForm";
import type { SchemaFieldResponse } from "@/types/guidance";

const fields: SchemaFieldResponse[] = [
  {
    field_id: "material-id",
    field_name: "material",
    type: "enum",
    role: "core",
    allowed_values: ["glass", "paper"],
    display_order: 0,
  },
  {
    field_id: "damage-id",
    field_name: "damage_types",
    type: "enum_set",
    role: "core",
    allowed_values: ["crush", "dent"],
    display_order: 1,
  },
  {
    field_id: "recyclable-id",
    field_name: "is_recyclable",
    type: "boolean",
    role: "core",
    display_order: 2,
  },
  {
    field_id: "severity-id",
    field_name: "severity",
    type: "integer",
    role: "core",
    minimum: 0,
    maximum: 3,
    display_order: 3,
  },
  {
    field_id: "note-id",
    field_name: "review_note",
    type: "string",
    role: "aux",
    display_order: 4,
  },
];

describe("ProposalForm accessible field names", () => {
  it("associates every editor and enum-set group with its schema field name", () => {
    render(
      <ProposalForm
        schemaFields={fields}
        proposalJson={{
          material: "glass",
          damage_types: ["crush"],
          is_recyclable: true,
          severity: 2,
          review_note: "intact",
        }}
        onValuesChange={vi.fn()}
      />,
    );

    expect(screen.getByRole("combobox", { name: "material" })).toHaveValue("glass");
    const damageGroup = screen.getByRole("group", { name: "damage_types" });
    expect(within(damageGroup).getByRole("checkbox", { name: "crush" })).toBeChecked();
    expect(
      within(damageGroup).getByRole("checkbox", { name: "dent" }),
    ).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "is_recyclable" })).toBeChecked();
    expect(screen.getByRole("spinbutton", { name: "severity" })).toHaveValue(2);
    expect(screen.getByRole("textbox", { name: "review_note" })).toHaveValue("intact");
  });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  formatModelDisplayName,
  localTeacherDisplayName,
  shortBaseLabel,
  thinkingToggleVisible,
  visualBudgetVisible,
} from "../model-display";

describe("formatModelDisplayName (title casing, default)", () => {
  it("renders the canonical Cosmos Reason2 names", () => {
    expect(formatModelDisplayName("nvidia/cosmos-reason2-8b")).toBe(
      "Cosmos Reason2 8B",
    );
    expect(formatModelDisplayName("nvidia/cosmos-reason2-2b")).toBe(
      "Cosmos Reason2 2B",
    );
  });

  it("strips any provider prefix, not just nvidia/", () => {
    expect(formatModelDisplayName("meta/llama3-70b")).toBe("Llama3 70B");
  });

  it("leaves an unprefixed name intact except for casing", () => {
    expect(formatModelDisplayName("qwen2-vl-7b")).toBe("Qwen2 Vl 7B");
  });

  it("returns em-dash for null / undefined / empty", () => {
    expect(formatModelDisplayName(null)).toBe("—");
    expect(formatModelDisplayName(undefined)).toBe("—");
    expect(formatModelDisplayName("")).toBe("—");
  });
});

describe("formatModelDisplayName (upper casing)", () => {
  it("strips nvidia/ prefix and uppercases", () => {
    expect(formatModelDisplayName("nvidia/cosmos-reason2-8b", "upper")).toBe(
      "COSMOS REASON2 8B",
    );
  });

  it("strips mistralai/ prefix and uppercases", () => {
    expect(formatModelDisplayName("mistralai/mistral-large-3", "upper")).toBe(
      "MISTRAL LARGE 3",
    );
  });

  it("handles no org prefix", () => {
    expect(formatModelDisplayName("custom-model-2b-vision", "upper")).toBe(
      "CUSTOM MODEL 2B VISION",
    );
  });

  it("returns em-dash for null", () => {
    expect(formatModelDisplayName(null, "upper")).toBe("—");
  });

  it("returns em-dash for undefined", () => {
    expect(formatModelDisplayName(undefined, "upper")).toBe("—");
  });

  it("returns em-dash for empty string", () => {
    expect(formatModelDisplayName("", "upper")).toBe("—");
  });

  it("preserves already-upper model ids without org", () => {
    expect(formatModelDisplayName("MY-MODEL-v1", "upper")).toBe("MY MODEL V1");
  });

  it("keeps multi-slash tails intact (takes everything after first slash)", () => {
    expect(formatModelDisplayName("ns/sub/my-model-7b", "upper")).toBe(
      "SUB/MY MODEL 7B",
    );
  });
});

describe("localTeacherDisplayName", () => {
  it("names the quality-default Omni Teacher", () => {
    expect(
      localTeacherDisplayName("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
    ).toBe("Nemotron 3 Nano Omni");
  });

  it("names CR3 Nano for the cosmos3-nano-reasoner recommendation", () => {
    expect(localTeacherDisplayName("nvidia/cosmos3-nano-reasoner")).toBe(
      "Cosmos 3 Nano (Reasoner)",
    );
  });

  it("names CR3 Super for the cosmos3-super-reasoner recommendation", () => {
    expect(localTeacherDisplayName("nvidia/cosmos3-super-reasoner")).toBe(
      "Cosmos 3 Super (Reasoner)",
    );
  });

  it("names Cosmos Reason2 variants by size suffix", () => {
    expect(localTeacherDisplayName("nvidia/cosmos-reason2-8b")).toBe(
      "Cosmos Reason2 8B",
    );
    expect(localTeacherDisplayName("nvidia/cosmos-reason2-2b")).toBe(
      "Cosmos Reason2 2B",
    );
  });

  it("falls back without misnaming an unknown model as Cosmos", () => {
    expect(localTeacherDisplayName("vendor/custom-vision-7b")).toBe("Custom Vision 7B");
    expect(localTeacherDisplayName(null)).toBe("Local Teacher");
    expect(localTeacherDisplayName(undefined)).toBe("Local Teacher");
  });
});

describe("thinkingToggleVisible", () => {
  const mc = (thinking_toggle_mode: string, thinking_toggle_support = "supported") => ({
    thinking_toggle_mode,
    thinking_toggle_support,
    visual_budget_support: "supported",
  });

  it("shows the toggle only when the model has one confirmed to flip", () => {
    expect(thinkingToggleVisible(mc("optional"))).toBe(true);
    expect(thinkingToggleVisible(mc("none"))).toBe(false);
    expect(thinkingToggleVisible(mc("always_on_reasoning"))).toBe(false);
    expect(thinkingToggleVisible(mc("optional", "unknown"))).toBe(false);
    expect(thinkingToggleVisible(mc("optional", "unsupported"))).toBe(false);
  });

  it("hides the toggle on a config lookup miss", () => {
    expect(thinkingToggleVisible(undefined)).toBe(false);
  });
});

describe("visualBudgetVisible", () => {
  const mc = (visual_budget_support: string) => ({
    thinking_toggle_mode: "optional",
    thinking_toggle_support: "supported",
    visual_budget_support,
  });

  it("hides the control when the model ignores the setting", () => {
    expect(visualBudgetVisible(mc("supported"))).toBe(true);
    expect(visualBudgetVisible(mc("unsupported"))).toBe(false);
  });

  it("hides the control on a config lookup miss", () => {
    expect(visualBudgetVisible(undefined)).toBe(false);
  });
});

describe("shortBaseLabel", () => {
  it("extracts size suffix for 8b", () => {
    expect(shortBaseLabel("nvidia/cosmos-reason2-8b")).toBe("8B");
  });

  it("extracts size suffix for 2b", () => {
    expect(shortBaseLabel("nvidia/cosmos-reason2-2b")).toBe("2B");
  });

  it("falls back to full name when no size", () => {
    expect(shortBaseLabel("mistralai/mistral-large-3")).toBe(
      "mistralai/mistral-large-3",
    );
  });

  it("returns em-dash for null", () => {
    expect(shortBaseLabel(null)).toBe("—");
  });
});

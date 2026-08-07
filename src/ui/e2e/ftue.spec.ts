// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Page, type Route } from "@playwright/test";

const projectId = "ftue-rps";
const samplePath = "/data/images";
const now = "2026-07-29T12:00:00Z";
const actualLabels: Record<string, string> = {
  "rock-01": "rock",
  "paper-01": "paper",
};

function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function projectResponse() {
  return {
    project_id: projectId,
    name: "RPS walkthrough",
    description: "First-time Blueprint exercise",
    project_dir: `/tmp/projects/${projectId}`,
    created_at: now,
    updated_at: now,
    counts: {
      verified: state.labelSaved ? 1 : 0,
      unlabeled: Math.max(0, state.ingested - (state.labelSaved ? 1 : 0)),
      auto_labeled: 0,
      omitted: 0,
      pending_relabel: 0,
      prior_relabeled: 0,
    },
    teacher_model_config_id: "teacher-1",
    active_guidance_id: state.guidanceActive ? "guidance-rps" : null,
    active_student_model_config_id: null,
    labeling_generation_preset_key: "precise",
    thinking_default_on: false,
    visual_budget_preset_key: "balanced",
    structured_generation_mode_default: "auto",
    rationale_anti_anchoring: true,
    auto_evaluate_enabled: false,
    icl_recommendation_dismissed_at_count: 0,
    export_field_mode: "all",
    embedding_provider: "hosted_nvclip",
    embedding_model_id: "nvidia/llama-nemotron-embed-vl-1b-v2",
    embedding_dim: 2048,
    embedding_endpoint_id: "hosted-embedding",
    phash_algorithm: "dct_phash_64",
    schema_refinement_reminders_dismissed: 0,
    schema_change_context_example_key: null,
    test_pool_fraction: 0.4,
    scaleup_exact_match_threshold: 0.8,
    scaleup_per_field_match_threshold: 0.8,
    scaleup_min_per_value_f1_threshold: 0.6,
    scaleup_accept_rate_threshold: 0.8,
    scaleup_accept_rate_window: 50,
    scaleup_min_test_pool_size: 60,
    archived_at: null,
    // This test starts after an operator-provided hosted environment is ready.
    // The separate NIM setup tests cover credential collection.
    setup_completed_at: now,
  };
}

function guidanceResponse() {
  return {
    guidance_id: "guidance-rps",
    project_id: projectId,
    version_number: 1,
    description: "Classify the hand gesture in each image as rock, paper, or scissors.",
    schema_fields: [
      {
        field_id: "field-category",
        field_name: "category",
        type: "enum",
        role: "core",
        allowed_values: ["rock", "paper", "scissors"],
        minimum: null,
        maximum: null,
        min_length: null,
        max_length: null,
        display_order: 0,
      },
    ],
    rules: "",
    derived_json_schema: {
      type: "object",
      properties: { category: { type: "string", enum: ["rock", "paper", "scissors"] } },
      required: ["category"],
    },
    generation_order: ["category"],
    schema_hash: "rps-schema",
    created_at: now,
  };
}

const state = {
  ingested: 0,
  guidanceActive: false,
  labelSaved: false,
  savedLabels: [] as Array<Record<string, unknown>>,
};

async function installFtueApi(page: Page) {
  state.ingested = 0;
  state.guidanceActive = false;
  state.labelSaved = false;
  state.savedLabels = [];

  await page.route("**/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    if (path === "/v1/projects" && method === "POST") {
      return json(route, projectResponse(), 201);
    }
    if (path === "/v1/projects" && method === "GET") {
      const project = projectResponse();
      return json(route, {
        items: state.ingested > 0 || state.guidanceActive ? [project] : [],
        next_cursor: null,
        has_archived: false,
      });
    }
    if (path === `/v1/projects/${projectId}` && method === "GET") {
      return json(route, projectResponse());
    }
    if (path === `/v1/projects/${projectId}` && method === "PATCH") {
      const body = request.postDataJSON() as Record<string, unknown>;
      if (body.active_guidance_id === "guidance-rps") state.guidanceActive = true;
      return json(route, projectResponse());
    }
    if (path === "/v1/environment") {
      return json(route, {
        hosted_nim_available: true,
        local_deploy_available: false,
        docker_available: true,
        nvidia_toolkit_available: false,
        nvidia_api_key_configured: true,
        ngc_api_key_configured: false,
        gpus: [],
        local_deployable_models: [],
        embedding_deployment: {
          model_name: "nvidia/llama-nemotron-embed-vl-1b-v2",
          nim_container_image: "",
          gpu_memory_minimum_gb: 24,
          fits: false,
          provider: "hosted_nvclip",
        },
        missing_prerequisites: [],
        recommended_teacher_mode: "hosted",
        recommended_embedding_mode: "hosted",
        nim_startup_timeout_s: 1200,
        student_latency_test_concurrencies: [1, 8, 24],
        default_teacher_model_name: "nvidia/nemotron-nano-vl",
        allow_secret_persist: false,
      });
    }
    if (path === "/v1/filesystem/browse") {
      const requestedPath = url.searchParams.get("path");
      if (!requestedPath) {
        return json(route, {
          path: "/data",
          parent: null,
          bundled_sample_path: samplePath,
          entries: [
            { name: "images", type: "directory", path: samplePath, size_bytes: null },
          ],
        });
      }
      return json(route, {
        path: samplePath,
        parent: "/data",
        bundled_sample_path: samplePath,
        entries: ["rock", "paper", "scissors"].map((label) => ({
          name: label,
          type: "directory",
          path: `${samplePath}/${label}`,
          size_bytes: null,
        })),
      });
    }
    if (path === "/v1/filesystem/scan" && method === "POST") {
      const body = request.postDataJSON() as { path: string };
      const label = body.path.split("/").at(-1) ?? "rock";
      const images = Array.from({ length: 5 }, (_, index) => ({
        storage_ref: `${body.path}/${label}-${index + 1}.png`,
        suggested_example_key: `${label}-${String(index + 1).padStart(2, "0")}`,
        size_bytes: 1024,
        key_status: "available",
        existing_storage_ref: null,
      }));
      return json(route, {
        path: body.path,
        images,
        skipped: [],
        total_images: images.length,
        total_skipped: 0,
        total_collisions: 0,
      });
    }
    if (path === `/v1/projects/${projectId}/examples:ingest` && method === "POST") {
      const body = request.postDataJSON() as {
        examples: Array<{ example_key: string; storage_ref: string }>;
      };
      state.ingested += body.examples.length;
      return json(
        route,
        {
          results: body.examples.map((example) => ({
            example_key: example.example_key,
            status: "created",
            error: null,
            error_code: null,
            warnings: [],
            example: { example_key: example.example_key, state: "Unlabeled" },
          })),
        },
        202,
      );
    }
    if (
      path === `/v1/projects/${projectId}/guidance:validate_draft` &&
      method === "POST"
    ) {
      return json(route, {
        issues: [],
        derived_json_schema: guidanceResponse().derived_json_schema,
        schema_hash: "rps-schema",
        save_allowed: true,
      });
    }
    if (path === `/v1/projects/${projectId}/guidance` && method === "POST") {
      return json(route, guidanceResponse(), 201);
    }
    if (path === `/v1/projects/${projectId}/guidance` && method === "GET") {
      return json(route, { items: [guidanceResponse()], next_cursor: null });
    }
    if (path === `/v1/projects/${projectId}/guidance/guidance-rps`) {
      return json(route, guidanceResponse());
    }
    if (path === `/v1/projects/${projectId}/guidance:reminder_status`) {
      return json(route, {
        active_reminder: null,
        verified_count: state.labelSaved ? 1 : 0,
        threshold_1: 20,
        threshold_2: 50,
        dismissed_count: 0,
      });
    }
    if (path === `/v1/projects/${projectId}/model_configs`) {
      return json(route, {
        items: [
          {
            model_config_id: "teacher-1",
            project_id: projectId,
            endpoint_id: "hosted-teacher",
            model_name: "nvidia/nemotron-nano-vl",
            context_window_tokens: 32768,
            eligible_roles: ["teacher"],
            supports_image_input: true,
            structured_generation_support: "supported",
            thinking_toggle_mode: "none",
            thinking_toggle_support: "unsupported",
            visual_budget_mode: "none",
            visual_budget_support: "unsupported",
            model_quantization: null,
            nim_model_profile: null,
            nim_profile_metadata: null,
            local_deploy_metadata: null,
            hosted_compatible: true,
            availability: { available: true, reason: null },
            created_at: now,
          },
        ],
        next_cursor: null,
      });
    }
    if (path === `/v1/projects/${projectId}/scaleup_gate`) {
      return json(route, {
        gate_status: "not_ready",
        criteria: [
          {
            criterion_name: "min_test_pool_size",
            passed: false,
            current_value: 0,
            threshold: 60,
            message: "Need 60 Test Pool examples.",
            details: { pool_target: 0 },
          },
        ],
        evaluated_at: now,
      });
    }
    if (path === `/v1/projects/${projectId}/evaluation_trigger_status`) {
      const inactive = {
        is_active: false,
        dismissed: false,
        message: "",
        context: null,
      };
      return json(route, {
        auto_evaluate_enabled: false,
        first_pool_threshold: inactive,
        configuration_change: inactive,
        icl_growth: inactive,
        updated_at: now,
      });
    }
    if (path === `/v1/projects/${projectId}/evaluation_runs`) {
      return json(route, { items: [], next_cursor: null });
    }
    if (path === `/v1/projects/${projectId}/review_selector/next`) {
      const exampleKey = state.labelSaved ? "paper-01" : "rock-01";
      return json(route, {
        example_key: exampleKey,
        example_state: "Unlabeled",
        has_existing_label: false,
        selection_mode: "phash_diversity",
        queue_empty: false,
        storage_ref: `${samplePath}/${actualLabels[exampleKey]}/${exampleKey}.png`,
        prior_verified_label_ref: null,
      });
    }
    if (path === `/v1/projects/${projectId}/proposals` && method === "POST") {
      const body = request.postDataJSON() as { example_key: string };
      const afterEdit = body.example_key === "paper-01";
      return json(route, {
        inference_invocation_id: afterEdit ? "invocation-2" : "invocation-1",
        example_key: body.example_key,
        // The first proposal is deliberately wrong: rock-01 is ground-truth rock.
        proposal_json: { category: afterEdit ? "paper" : "paper" },
        schema_valid_core: true,
        validation_errors_core: [],
        validation_errors_aux: [],
        invocation_status: "success",
        latency_ms_end_to_end: 120,
        icl_images_attached_count: afterEdit ? 1 : 0,
        icl_example_keys_used: afterEdit ? ["rock-01"] : [],
        used_existing_label: false,
      });
    }
    if (path === `/v1/projects/${projectId}/labels` && method === "POST") {
      const body = request.postDataJSON() as Record<string, unknown>;
      state.savedLabels.push(body);
      state.labelSaved = true;
      return json(route, {
        example_key: body.example_key,
        label_status: "Verified",
        verified_outcome: "Edit",
        verified_at: now,
        edited_core_fields: ["category"],
        edited_aux_fields: [],
        pool_assignment: null,
      });
    }
    if (
      path.startsWith(`/v1/projects/${projectId}/examples/`) &&
      path.endsWith("/image")
    ) {
      return route.fulfill({
        status: 200,
        contentType: "image/png",
        body: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlWvWQAAAAASUVORK5CYII=",
          "base64",
        ),
      });
    }

    return json(route, { detail: `Unhandled FTUE mock: ${method} ${path}` }, 501);
  });
}

test("SME creates a project, ingests the bundled sample, activates RPS Guidance, and sees ICL after a genuine edit", async ({
  page,
}) => {
  await installFtueApi(page);
  await page.goto("/");

  await page.getByRole("button", { name: "+ Create Project" }).click();
  await page.getByPlaceholder("e.g., Damage Inspection").fill("RPS walkthrough");
  await page
    .getByPlaceholder("e.g., Surface damage classification for manufacturing QA")
    .fill("First-time Blueprint exercise");
  await page.getByRole("button", { name: "Create Project", exact: true }).click();

  await expect(page).toHaveURL(new RegExp(`/projects/${projectId}/ready$`));
  // Local-source mode starts beside the sample so its root remains a normal,
  // selectable directory; Compose opens directly inside /data/images.
  await page.getByText("images/").click();
  await expect(page.getByText("rock/")).toBeVisible();

  for (const checkbox of await page.getByRole("checkbox").all()) {
    await checkbox.check();
  }
  await page.getByRole("button", { name: "Ingest Selected" }).click();

  await expect(page.getByTestId("ingestion-summary")).toBeVisible();
  await expect(
    page.getByText(/bundled sample is a walkthrough, not a Scale-Up-ready dataset/),
  ).toBeVisible();
  await page.getByRole("button", { name: /Start labeling/ }).click();

  await expect(page).toHaveURL(new RegExp(`/projects/${projectId}/create-guidance$`));
  await page.getByTestId("template-selector").click();
  await page.getByText("Rock, paper, scissors", { exact: true }).last().click();
  await expect(page.getByText("rock", { exact: true }).first()).toBeVisible();
  await page.getByTestId("save-guidance-btn").click();

  await expect(page).toHaveURL(new RegExp(`/projects/${projectId}/labeling$`));
  await expect(page.getByTestId("field-input-category")).toHaveValue("paper");
  await expect(page.getByTestId("icl-chip-coldstart")).toContainText("no edits yet");

  // rock-01 is known ground-truth rock, while the mocked Teacher proposed paper.
  await page.getByTestId("field-input-category").selectOption(actualLabels["rock-01"]);
  await page.getByTestId("save-btn").click();

  await expect.poll(() => state.savedLabels.length).toBe(1);
  expect(state.savedLabels[0]).toMatchObject({
    example_key: "rock-01",
    label_json: { category: "rock" },
  });
  await expect(page.getByTestId("icl-chip-active")).toContainText(
    "ICL: 1 edit in context",
  );
  await expect(page.getByTestId("field-input-category")).toHaveValue("paper");
});

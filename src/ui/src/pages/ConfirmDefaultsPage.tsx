// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Confirm Model Defaults screen.
 *
 * 2 states:
 *   - Skip (fast path) — default valid, auto-advance
 *   - Confirm required — a Teacher dropdown
 */

import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button, Spinner, Text, Select } from "@kui/react";
import { ArrowLeft, ArrowRight } from "lucide-react";

import { useSetupContext } from "@/pages/ProjectSetupLayout";
import { fetchModelConfigs } from "@/api/model-configs";
import { updateProject } from "@/api/model-configs";
import { markSetupCompleted } from "@/api/projects";
import { modelConfigKeys, projectKeys } from "@/api/query-keys";
import type { ModelConfigResponse } from "@/types/nim";
import type { SetupAutoSkipState } from "@/types/setupChain";

// ── Helpers ─────────────────────────────────────────────────────────────────

function getCapabilityBadges(mc: ModelConfigResponse): string[] {
  const badges: string[] = [];
  if (mc.supports_image_input) badges.push("vision");
  if (mc.thinking_toggle_mode !== "none") badges.push("thinking");
  if (mc.visual_budget_mode !== "none") badges.push("visual budget");
  return badges;
}

function preferredDefaultId(
  options: ModelConfigResponse[],
  preferredName: string,
): string | null {
  if (options.length === 0) return null;
  const named = options.find((mc) => mc.model_name === preferredName);
  return (named ?? options[0]).model_config_id;
}

// ── Component ───────────────────────────────────────────────────────────────

export function ConfirmDefaultsPage() {
  const { projectId, project, environment } = useSetupContext();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const didAutoSkip = useRef(false);

  // Selection
  const [selectedTeacher, setSelectedTeacher] = useState(
    project.teacher_model_config_id,
  );

  const { data: configsData, isLoading } = useQuery({
    queryKey: modelConfigKeys.list(projectId),
    queryFn: () => fetchModelConfigs(projectId),
  });

  // Memoized: allConfigs feeds completeSetup's dependency array — a
  // fresh [] every render would re-create the callback each tick.
  const allConfigs = useMemo(() => configsData?.items ?? [], [configsData]);

  // Filter for the dropdown: role + capability ONLY. Do NOT filter by
  // ``availability.available`` — that's a runtime "can I call this
  // model right now?" check used by the labeling-screen Teacher picker.
  // The Confirm Defaults screen is a SETUP screen where the SME picks
  // their default; a model that is currently still deploying (local
  // Cosmos), or temporarily unreachable (hosted Mistral while the SME
  // hasn't pasted a key yet), is a perfectly valid default. Filtering
  // by availability here means a no-NVIDIA-key/Run-locally user lands
  // on this screen with an EMPTY dropdown because every catalog entry
  // currently reports ``available=false`` (cause: ``no_nvidia_api_key``).
  const teacherOptions = allConfigs.filter(
    (mc) => mc.eligible_roles.includes("teacher") && mc.supports_image_input,
  );

  // ── Shared setup-completion tail ─────────────────────────────────────
  // Stamp ``setup_completed_at`` server-side before navigating forward.
  // Idempotent — if the setup gate (NIMSetupGatePage) already stamped
  // on [Start labeling], the second call is a no-op (per the
  // service-layer guard). Best-effort: never block navigation on the
  // stamp. Used by both the auto-skip path (replace-navigation) and
  // the manual Save mutation (push-navigation).
  const completeSetup = useCallback(
    (autoSkip: boolean) => {
      const nextPath = project.active_guidance_id ? "../labeling" : "../ready";
      // A fully-auto-skipped chain never showed the SME a setup screen,
      // so forward the auto-selected configuration through router state
      // for the landing screen's SetupAutoSkipBanner. Router state is
      // deliberate: it doesn't persist, keeping the banner one-shot.
      const state = autoSkip
        ? {
            setupAutoSkip: {
              teacherMode: environment.recommended_teacher_mode,
              embeddingMode: environment.recommended_embedding_mode,
              teacherName:
                allConfigs.find(
                  (mc) => mc.model_config_id === project.teacher_model_config_id,
                )?.model_name ?? null,
            } satisfies SetupAutoSkipState,
          }
        : undefined;
      void markSetupCompleted(projectId, {
        auto_skip: autoSkip,
        teacher_mode: environment.recommended_teacher_mode,
        embedding_mode: environment.recommended_embedding_mode,
        embedding_provider: environment.embedding_deployment.provider,
      })
        .catch((err: unknown) => {
          console.warn(
            `mark_setup_completed (ConfirmDefaults ${autoSkip ? "auto-skip" : "manual"}) failed:`,
            err,
          );
        })
        .finally(() => {
          // The NIM Configuration header link gates on the cached
          // project's setup_completed_at — refetch after the stamp
          // lands so it appears immediately on the landing screen.
          void queryClient.invalidateQueries({
            queryKey: projectKeys.detail(projectId),
          });
          navigate(nextPath, { replace: autoSkip, state });
        });
    },
    [
      project.active_guidance_id,
      project.teacher_model_config_id,
      allConfigs,
      projectId,
      environment,
      navigate,
      queryClient,
    ],
  );

  // ── Auto-skip ────────────────────────────────────────────────────────
  useEffect(() => {
    if (didAutoSkip.current || isLoading || !configsData) return;

    const teacherValid = teacherOptions.some(
      (mc) => mc.model_config_id === project.teacher_model_config_id,
    );

    if (teacherValid) {
      didAutoSkip.current = true;
      completeSetup(true);
    }
  }, [isLoading, configsData, project, teacherOptions, completeSetup]);

  // ── Preselect the seeded default when stored ID is null OR orphan ────
  // Two cases trigger a preselect:
  //   1. The project has no stored model ID (rare — only when project
  //      state was explicitly cleared).
  //   2. The stored ID is "orphan": it is a valid UUID string but no
  //      matching entry exists in the role-filtered options (e.g. the
  //      previously-selected model was archived, or the user connected a
  //      custom endpoint where the seeded model is not present). Without
  //      this branch, KUI Select would render the raw UUID as the
  //      dropdown's display text and the capability-pill row would be
  //      hidden — both regressions from the intent of giving SMEs
  //      a name-based, capability-summarized choice.
  useEffect(() => {
    if (isLoading) return;
    const teacherIsOrphan =
      !!selectedTeacher &&
      !teacherOptions.some((mc) => mc.model_config_id === selectedTeacher);
    if (!selectedTeacher || teacherIsOrphan) {
      const id = preferredDefaultId(
        teacherOptions,
        environment.default_teacher_model_name,
      );
      if (id) setSelectedTeacher(id);
    }
  }, [
    isLoading,
    selectedTeacher,
    teacherOptions,
    environment.default_teacher_model_name,
  ]);

  const saveMutation = useMutation({
    mutationFn: () =>
      updateProject(projectId, {
        teacher_model_config_id: selectedTeacher,
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({
        queryKey: projectKeys.detail(projectId),
      });
      // Manual Save also stamps acknowledgment. Idempotent with any
      // earlier auto-skip path that may have fired.
      completeSetup(false);
    },
  });

  // ── Render ───────────────────────────────────────────────────────────

  // When the SME has just clicked [Start labeling] on the setup gate, this
  // page is almost always going to auto-skip — but for the ~0.5–1s
  // window while ``model_configs`` fetches, we'd otherwise render a
  // bare spinner in the void, which feels jarring after the confident
  // "You're set up" card the SME just confirmed. Render a contextual
  // glass card that visually continues the FTU flow.
  if (isLoading) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center p-6">
        <div
          className="glass-card--elevated flex w-full max-w-[640px] flex-col items-center gap-4 p-8"
          data-testid="confirm-defaults-loading-card"
        >
          <Spinner size="large" aria-label="Finishing setup" />
          <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
            Finishing setup…
          </Text>
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-muted)", textAlign: "center" }}
          >
            Verifying your model defaults. You'll land on the next step in a moment.
          </Text>
        </div>
      </div>
    );
  }

  const selectedTeacherConfig = allConfigs.find(
    (c) => c.model_config_id === selectedTeacher,
  );

  return (
    <div className="p-8 max-w-4xl mx-auto w-full">
      {/* Header (outside the card — matches Retail Catalog header + card pattern) */}
      <div className="mb-6">
        <Text kind="title/xl" style={{ color: "var(--text-primary)" }}>
          Confirm Model Defaults
        </Text>
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-muted)", display: "block", marginTop: 4 }}
        >
          Choose which model proposes labels during labeling. You can change this later.
        </Text>
      </div>

      {/* Glass card wrapping the field block. Matches the Project List
          and NIM Connection elevated card treatment and the Retail
          Catalog Fields panel. */}
      <div className="glass-card glass-card--elevated p-6">
        {/* Teacher field */}
        <div>
          <Text
            kind="label/bold/sm"
            style={{ color: "var(--text-primary)", display: "block" }}
          >
            Teacher
          </Text>
          <Text
            kind="body/regular/sm"
            style={{
              color: "var(--text-muted)",
              display: "block",
              marginTop: 2,
              marginBottom: 10,
            }}
          >
            Proposes labels for each image during interactive labeling.
          </Text>
          <Select
            items={teacherOptions.map((mc) => ({
              value: mc.model_config_id,
              children: mc.model_name,
            }))}
            value={selectedTeacher}
            onValueChange={(val) => setSelectedTeacher(val)}
            data-testid="teacher-select"
          />
          {selectedTeacherConfig && (
            <div className="mt-2 flex flex-wrap gap-2">
              {getCapabilityBadges(selectedTeacherConfig).map((badge) => (
                <span key={badge} className="glass-pill">
                  {badge}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Error */}
      {saveMutation.isError && (
        <Text kind="body/regular/sm" className="mt-4 text-error">
          Failed to save model selections.
        </Text>
      )}

      {/* Footer navigation — sits directly below the card, not sticky-bottom */}
      <div className="flex items-center justify-between mt-6">
        <Button kind="secondary" onClick={() => navigate("../setup")}>
          <ArrowLeft size={14} /> Back
        </Button>
        <Button
          kind="primary"
          className="nvidia-green-button"
          onClick={() => saveMutation.mutate()}
          disabled={!selectedTeacher || saveMutation.isPending}
        >
          {saveMutation.isPending ? (
            "Saving..."
          ) : (
            <>
              Continue <ArrowRight size={14} />
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Labeling screen — the primary interactive loop.
 *
 * Orchestrates the cycle: fetch next image → fetch proposal →
 * display → save/skip/retry → fetch next.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import { ChevronRight, Plus, RotateCcw } from "lucide-react";

import { useSetupContext } from "@/pages/setup-context";
import { SetupAutoSkipBanner } from "@/components/common/SetupAutoSkipBanner";
import {
  ImagePanel,
  ProposalForm,
  ProposalFailure,
  LabelingStatusBar,
  RationalePanel,
  TopBarControls,
  TeacherModelPicker,
  IclChip,
  RetryPanel,
  InlineNotices,
  QueueEmpty,
  EvaluationStrip,
  ResultsPanel,
} from "@/components/labeling";
import type { RationalePanelState } from "@/components/labeling";
import type { RetryOverrides } from "@/components/labeling";
import {
  fetchNextReviewItem,
  createProposal,
  saveLabel,
  skipExample,
  regenerateRationale,
  restoreOmitted,
} from "@/api/labeling";
import {
  fetchGuidance,
  listGuidances,
  fetchReminderStatus,
  dismissReminder,
} from "@/api/guidance";
import { fetchModelConfigs, updateProject } from "@/api/model-configs";
import { fetchProjectList } from "@/api/projects";
import { fetchScaleUpGate } from "@/api/evaluation";
import { listStudentModels } from "@/api/students";
import {
  evaluationKeys,
  guidanceKeys,
  modelConfigKeys,
  projectKeys,
  studentModelKeys,
} from "@/api/query-keys";
import { RATIONALE_NOTE_FIELD_NAME } from "@/lib/guidance-templates";
import { thinkingToggleVisible, visualBudgetVisible } from "@/lib/model-display";
import type { EvaluationRunResponse } from "@/types/evaluation";
import type {
  ProposalResponse,
  RationaleSource,
  PriorLabelSnapshot,
} from "@/types/labeling";
import type { SchemaFieldResponse } from "@/types/guidance";

// Stable empty-proposal fallback — used when a failed invocation leaves
// `proposal_json` null so the SME can label from scratch. A
// module-scope constant keeps the identity stable across re-renders so
// ProposalForm's "proposal changed" reset check does not fire every tick
// and wipe the user's in-progress manual label.
const EMPTY_PROPOSAL: Record<string, unknown> = Object.freeze({});

// ── State Machine ────────────────────────────────────────────────────────────

type LabelingPhase =
  | "fetching_next"
  | "fetching_proposal"
  | "reviewing"
  | "saving"
  | "skipping"
  | "queue_empty"
  | "error";

interface LabelingState {
  phase: LabelingPhase;
  currentExampleKey: string | null;
  hasExistingLabel: boolean;
  proposal: ProposalResponse | null;
  error: string | null;
  imageMissing: boolean;
}

type LabelingAction =
  | { type: "FETCH_NEXT_START" }
  | { type: "FETCH_NEXT_OK"; exampleKey: string; hasExistingLabel: boolean }
  | { type: "FETCH_NEXT_EMPTY" }
  | { type: "FETCH_NEXT_FAIL"; error: string }
  | { type: "FETCH_PROPOSAL_OK"; proposal: ProposalResponse }
  | { type: "FETCH_PROPOSAL_FAIL"; error: string }
  | { type: "SAVE_START" }
  | { type: "ACTION_OK" }
  | { type: "ACTION_FAIL"; error: string }
  | { type: "SKIP_START" }
  | { type: "RETRY" }
  | { type: "IMAGE_MISSING" }
  | { type: "IMAGE_LOADED" };

const INITIAL_STATE: LabelingState = {
  phase: "fetching_next",
  currentExampleKey: null,
  hasExistingLabel: false,
  proposal: null,
  error: null,
  imageMissing: false,
};

function reducer(state: LabelingState, action: LabelingAction): LabelingState {
  switch (action.type) {
    case "FETCH_NEXT_START":
      return {
        ...state,
        phase: "fetching_next",
        proposal: null,
        error: null,
        imageMissing: false,
      };
    case "FETCH_NEXT_OK":
      return {
        ...state,
        phase: "fetching_proposal",
        currentExampleKey: action.exampleKey,
        hasExistingLabel: action.hasExistingLabel,
      };
    case "FETCH_NEXT_EMPTY":
      return { ...state, phase: "queue_empty", currentExampleKey: null };
    case "FETCH_NEXT_FAIL":
      return { ...state, phase: "error", error: action.error };
    case "FETCH_PROPOSAL_OK":
      return { ...state, phase: "reviewing", proposal: action.proposal };
    case "FETCH_PROPOSAL_FAIL":
      return { ...state, phase: "error", error: action.error };
    case "SAVE_START":
      return { ...state, phase: "saving" };
    case "SKIP_START":
      return { ...state, phase: "skipping" };
    case "ACTION_OK":
      // Auto-advance
      return {
        ...state,
        phase: "fetching_next",
        proposal: null,
        imageMissing: false,
      };
    case "ACTION_FAIL":
      return { ...state, phase: "reviewing", error: action.error };
    case "RETRY":
      return { ...state, phase: "fetching_proposal", proposal: null };
    case "IMAGE_MISSING":
      return { ...state, imageMissing: true };
    case "IMAGE_LOADED":
      return { ...state, imageMissing: false };
    default:
      return state;
  }
}

// ── Component ────────────────────────────────────────────────────────────────

export function LabelingPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { projectId, project } = useSetupContext();

  const { data: guidance } = useQuery({
    queryKey: guidanceKeys.detail(projectId, project.active_guidance_id ?? ""),
    queryFn: () => fetchGuidance(projectId, project.active_guidance_id!),
    enabled: !!project.active_guidance_id,
  });
  const schemaFields: SchemaFieldResponse[] = guidance?.schema_fields ?? [];
  const rationaleEnabled = schemaFields.some(
    (field) => field.field_name === RATIONALE_NOTE_FIELD_NAME,
  );

  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);
  const retryInvocationRef = useRef<string | null>(null);

  // Track form values and dirty state
  const [formValues, setFormValues] = useState<Record<string, unknown>>({});
  const [dirtyFields, setDirtyFields] = useState<Set<string>>(new Set());
  const [resetKey, setResetKey] = useState(0);

  // ── Rationale state ─────────────────────────────────────────────────────
  const [rationaleState, setRationaleState] = useState<RationalePanelState>("hidden");
  const [rationaleText, setRationaleText] = useState("");
  const [regenerationInvocationId, setRegenerationInvocationId] = useState<
    string | null
  >(null);
  const [regenerationError, setRegenerationError] = useState<string | null>(null);
  const originalRationaleRef = useRef("");

  // ── Retry panel state ─────────────────────────────────────────────────
  const [showRetryPanel, setShowRetryPanel] = useState(false);

  // True once any of the four top-bar controls (Teacher, Stability,
  // Thinking, Visual Budget) has actually PATCHed since the current
  // proposal landed. ``handleRetry`` consults this to skip the retry
  // panel when the SME has already expressed their intent via the top
  // bar — the panel would just ask them to re-confirm the same change.
  // Reset to false whenever a fresh proposal lands so the next click
  // starts from a clean baseline.
  const [topBarDirty, setTopBarDirty] = useState(false);
  // Header settings are project defaults for the *next* proposal. Keep
  // label actions gated until the PATCH commits so Save/Skip/Retry cannot
  // start that proposal against stale defaults.
  const [topBarSaving, setTopBarSaving] = useState(false);
  const [topBarError, setTopBarError] = useState<string | null>(null);
  const retryOverridesRef = useRef<RetryOverrides | null>(null);

  // ── Inline notices & prior-label state ──────────────────────────────
  const [coldStartDismissed, setColdStartDismissed] = useState(() => {
    try {
      return localStorage.getItem(`vlm_cold_start_dismissed_${projectId}`) === "true";
    } catch {
      return false;
    }
  });

  // ── Evaluation strip state ──────────────────────────────────────────
  const [resultsRun, setResultsRun] = useState<EvaluationRunResponse | null>(null);

  const { data: gateData } = useQuery({
    queryKey: evaluationKeys.gate(projectId),
    queryFn: () => fetchScaleUpGate(projectId),
    refetchInterval: 30_000,
  });
  const gatePassCount = gateData?.criteria?.filter((c) => c.passed).length ?? 0;
  const { data: studentModels } = useQuery({
    queryKey: studentModelKeys.list(projectId),
    // Keep the shared list key's query contract identical to AppShell,
    // Project Overview, and Compare so their cached pages cannot collide.
    queryFn: () => listStudentModels(projectId, { limit: 200 }),
  });
  const hasStudentModels = (studentModels?.items.length ?? 0) > 0;
  const poolCriterion = gateData?.criteria?.find(
    (c) => c.criterion_name === "min_test_pool_size",
  );
  const poolCount = poolCriterion?.current_value ?? 0;
  // The pool's growth target from the gate criterion details —
  // shown on the strip chip when verifying will grow the holdout.
  const poolTargetRaw = poolCriterion?.details?.["pool_target"];
  const poolTarget = typeof poolTargetRaw === "number" ? poolTargetRaw : null;
  const [priorLabelSnapshot, setPriorLabelSnapshot] =
    useState<PriorLabelSnapshot | null>(null);
  // Filesystem path on the backend host for the current example. Surfaced on
  // the missing-image state so the SME's [Report Missing Files]
  // Action Request carries the disk path the admin needs, not the API URL.
  const [currentStorageRef, setCurrentStorageRef] = useState<string | null>(null);
  const [restoringOmitted, setRestoringOmitted] = useState(false);
  const [restoreOmittedError, setRestoreOmittedError] = useState<string | null>(null);

  const handleValuesChange = useCallback(
    (values: Record<string, unknown>, dirty: Set<string>) => {
      setFormValues(values);
      setDirtyFields(dirty);

      // Expand rationale review only when Guidance opted into rationale_note.
      if (rationaleEnabled && dirty.size > 0) {
        setRationaleState((prev) => (prev === "hidden" ? "needs_review" : prev));
      }
    },
    [rationaleEnabled],
  );

  // ── Route Guards ─────────────────────────────────────────────────────────

  useEffect(() => {
    if (!project.active_guidance_id) {
      navigate(`/projects/${projectId}/create-guidance`, { replace: true });
    }
  }, [project.active_guidance_id, projectId, navigate]);

  // ── Data Queries ─────────────────────────────────────────────────────────

  const { data: modelConfigs } = useQuery({
    queryKey: modelConfigKeys.list(projectId, "teacher"),
    queryFn: () => fetchModelConfigs(projectId, "teacher"),
  });

  const { data: projectList, isSuccess: countsLoaded } = useQuery({
    queryKey: projectKeys.list(),
    queryFn: () => fetchProjectList(),
    refetchInterval: 10_000,
  });

  const counts = useMemo(() => {
    const item = projectList?.items.find((p) => p.project_id === projectId);
    return (
      item?.counts ?? {
        verified: 0,
        unlabeled: 0,
        omitted: 0,
        auto_labeled: 0,
        pending_relabel: 0,
        prior_relabeled: 0,
      }
    );
  }, [projectList, projectId]);

  // Guidance list for Retry panel
  const { data: guidanceList } = useQuery({
    queryKey: guidanceKeys.list(projectId),
    queryFn: () => listGuidances(projectId),
  });

  // Schema refinement reminder state — the backend owns eligibility
  // (thresholds, higher-of-two, dismissals, guidance-edited suppression).
  // Invalidated after each label save and after a dismiss.
  const { data: reminderStatus } = useQuery({
    queryKey: guidanceKeys.reminderStatus(projectId),
    queryFn: () => fetchReminderStatus(projectId),
  });

  // Active teacher config — for control visibility
  const activeTeacher = useMemo(
    () =>
      modelConfigs?.items?.find(
        (mc) => mc.model_config_id === project.teacher_model_config_id,
      ),
    [modelConfigs, project.teacher_model_config_id],
  );

  const thinkingVisible = thinkingToggleVisible(activeTeacher);
  const visualBudgetShown = visualBudgetVisible(activeTeacher);

  // ── Labeling Cycle: Side Effects ─────────────────────────────────────────

  // Fetch next example when phase is "fetching_next". The response also
  // carries the selected example's storage_ref (missing-image "Expected:"
  // path) and prior_verified_label_ref (per-field re-labeling
  // hints) — the backend attaches them so no examples-list scan is
  // needed here.
  useEffect(() => {
    if (state.phase !== "fetching_next") return;
    if (!project.active_guidance_id) return; // Guard: wait for redirect
    let cancelled = false;

    setPriorLabelSnapshot(null);
    setCurrentStorageRef(null);

    void fetchNextReviewItem(projectId).then(
      (res) => {
        if (cancelled) return;
        if (res.queue_empty) {
          dispatch({ type: "FETCH_NEXT_EMPTY" });
        } else {
          setCurrentStorageRef(res.storage_ref ?? null);
          if (res.prior_verified_label_ref) {
            try {
              setPriorLabelSnapshot(
                JSON.parse(res.prior_verified_label_ref) as PriorLabelSnapshot,
              );
            } catch {
              /* ignore parse errors — hints are optional */
            }
          }
          dispatch({
            type: "FETCH_NEXT_OK",
            exampleKey: res.example_key!,
            hasExistingLabel: res.has_existing_label,
          });
        }
      },
      (err) => {
        if (!cancelled) {
          dispatch({
            type: "FETCH_NEXT_FAIL",
            error: err instanceof Error ? err.message : "Failed to fetch next example",
          });
        }
      },
    );

    return () => {
      cancelled = true;
    };
  }, [state.phase, projectId, navigate, project.active_guidance_id]);

  // ── Proposal fetch ─────────────────────────────────────────────────────
  useEffect(() => {
    if (state.phase !== "fetching_proposal" || !state.currentExampleKey) return;
    let cancelled = false;

    const overrides = retryOverridesRef.current;
    const proposalPromise = createProposal(projectId, {
      example_key: state.currentExampleKey,
      use_existing_label: state.hasExistingLabel,
      retry_of_inference_invocation_id: retryInvocationRef.current ?? undefined,
      ...(overrides ?? {}),
    });

    void proposalPromise.then(
      (res) => {
        if (cancelled) return;
        retryInvocationRef.current = null;
        retryOverridesRef.current = null;
        // Fresh proposal is now in sync with current project state —
        // start the dirty-tracking baseline fresh for the next click.
        setTopBarDirty(false);
        dispatch({ type: "FETCH_PROPOSAL_OK", proposal: res });
      },
      (err) => {
        if (!cancelled) {
          dispatch({
            type: "FETCH_PROPOSAL_FAIL",
            error: err instanceof Error ? err.message : "Failed to fetch proposal",
          });
        }
      },
    );

    return () => {
      cancelled = true;
    };
  }, [state.phase, state.currentExampleKey, state.hasExistingLabel, projectId]);

  // ── Initialize rationale when proposal arrives ──────────────────────────

  // Must run for FAILED proposals too (null proposal_json): the previous
  // example's rationale text/state/regeneration id would otherwise
  // survive into the manual-label flow and be saved onto a different
  // image's Verified Edit — polluting ground truth and ICL.
  useEffect(() => {
    if (!state.proposal) return;
    const origRationale = rationaleEnabled
      ? String(state.proposal.proposal_json?.[RATIONALE_NOTE_FIELD_NAME] ?? "")
      : "";
    originalRationaleRef.current = origRationale;
    setRationaleText(origRationale);
    setRationaleState("hidden");
    setRegenerationInvocationId(null);
    setRegenerationError(null);
  }, [state.proposal, rationaleEnabled]);

  // ── Action Handlers ──────────────────────────────────────────────────────

  async function handleSave() {
    if (!state.proposal || !state.currentExampleKey) return;
    dispatch({ type: "SAVE_START" });

    const isDirty = dirtyFields.size > 0;

    let rationaleSource: RationaleSource | undefined;
    if (rationaleEnabled) {
      if (!isDirty) {
        rationaleSource = "teacher_proposal";
      } else if (rationaleState === "approved") {
        rationaleSource = "teacher_regenerated_approved";
      } else {
        rationaleSource = "sme_edited";
      }
    }

    // When rationale is enabled for an Edit, inject the reviewed note.
    const labelJson =
      rationaleEnabled && isDirty
        ? { ...formValues, [RATIONALE_NOTE_FIELD_NAME]: rationaleText }
        : formValues;
    const rationaleMetadata = rationaleEnabled
      ? {
          rationale_source: rationaleSource,
          rationale_regeneration_invocation_id:
            rationaleSource === "teacher_regenerated_approved"
              ? regenerationInvocationId
              : undefined,
        }
      : {};

    try {
      await saveLabel(projectId, {
        example_key: state.currentExampleKey,
        inference_invocation_id: state.proposal.inference_invocation_id,
        label_json: labelJson,
        ...rationaleMetadata,
      });
      await queryClient.invalidateQueries({ queryKey: projectKeys.list() });
      // Automatic pool routing happens in the same save transaction. Refresh
      // the gate immediately so the visible Test Pool count reconciles with
      // the saved label instead of waiting for the 30-second poll.
      await queryClient.invalidateQueries({
        queryKey: evaluationKeys.gate(projectId),
      });
      // A save can change the Verified count past a reminder threshold —
      // refetch the backend's reminder decision.
      await queryClient.invalidateQueries({
        queryKey: guidanceKeys.reminderStatus(projectId),
      });
      dispatch({ type: "ACTION_OK" });
    } catch (err) {
      dispatch({
        type: "ACTION_FAIL",
        error: err instanceof Error ? err.message : "Save failed",
      });
    }
  }

  async function handleSkip() {
    if (!state.currentExampleKey) return;
    dispatch({ type: "SKIP_START" });

    try {
      await skipExample(projectId, state.currentExampleKey);
      await queryClient.invalidateQueries({ queryKey: projectKeys.list() });
      dispatch({ type: "ACTION_OK" });
    } catch (err) {
      dispatch({
        type: "ACTION_FAIL",
        error: err instanceof Error ? err.message : "Skip failed",
      });
    }
  }

  function handleRetry() {
    // If the SME already changed a top-bar control since the current
    // proposal arrived, treat the click as "retry with those settings"
    // and skip the panel — the panel would only re-confirm what's
    // already in project state. handleRetryConfirm({}) fires the same
    // retry pathway with no per-attempt overrides, so the backend
    // invokes with the now-PATCHed project state.
    if (topBarDirty) {
      handleRetryConfirm({});
      return;
    }
    setShowRetryPanel(true);
  }

  function handleRetryConfirm(overrides: RetryOverrides) {
    if (state.proposal) {
      retryInvocationRef.current = state.proposal.inference_invocation_id;
    }
    retryOverridesRef.current = Object.keys(overrides).length > 0 ? overrides : null;
    setShowRetryPanel(false);
    dispatch({ type: "RETRY" });
  }

  function handleRetryCancel() {
    setShowRetryPanel(false);
  }

  function handleReset() {
    setResetKey((k) => k + 1);
    setRationaleState("hidden");
    setRationaleText(originalRationaleRef.current);
    setRegenerationInvocationId(null);
    setRegenerationError(null);
    setDirtyFields(new Set());
  }

  function handleImageMissing() {
    dispatch({ type: "IMAGE_MISSING" });
  }

  function handleImageLoaded() {
    dispatch({ type: "IMAGE_LOADED" });
  }

  function handleRationaleTextChange(text: string) {
    setRationaleText(text);
    setRegenerationError(null);

    // Check for meaningful change (not whitespace-only)
    const meaningful = originalRationaleRef.current.trim() !== text.trim();
    if (meaningful) {
      setRationaleState((prev) =>
        prev === "needs_review" || prev === "ai_review_required" || prev === "approved"
          ? "edited"
          : prev === "edited"
            ? "edited"
            : prev,
      );
    } else {
      // Whitespace-only: revert to the prior non-edited state
      setRationaleState((prev) => (prev === "edited" ? "needs_review" : prev));
    }
  }

  async function handleGenerateAIRationale() {
    if (!state.currentExampleKey) return;
    setRationaleState("regenerating");
    setRegenerationError(null);

    try {
      // The rationale writer receives the image and task context, but no
      // proposed or corrected answer, so it must inspect independently.
      const res = await regenerateRationale(projectId, state.currentExampleKey, {});
      if (res.invocation_status === "success") {
        setRationaleText(res.rationale_note);
        setRegenerationInvocationId(res.inference_invocation_id);
        setRationaleState("ai_review_required");
      } else {
        setRegenerationError(
          `Rationale generation ${res.invocation_status}. Edit the rationale directly.`,
        );
        setRationaleState("needs_review");
      }
    } catch (err) {
      setRegenerationError(
        err instanceof Error ? err.message : "Failed to generate rationale",
      );
      setRationaleState("needs_review");
    }
  }

  function handleApproveAIRationale() {
    setRationaleState("approved");
  }

  // ── Top bar project-setting handlers ────────────────────────────────────

  /**
   * Build a top-bar setting handler: no-op guard against the current
   * value (segmented controls / pickers can re-fire on the same value —
   * only an actual change counts for the retry-skip dirty flag), PATCH
   * the single project field, and put the authoritative response directly
   * into the project-detail cache. While the PATCH is pending, every label
   * action is disabled: otherwise Save could auto-advance before the
   * selected Teacher / generation controls had committed.
   */
  function makeSettingHandler<T extends string | boolean>(
    currentValue: T | null,
    field: string,
    extraInvalidations: readonly (readonly unknown[])[] = [],
  ) {
    return async (value: T) => {
      if (value === currentValue) return;
      setTopBarSaving(true);
      setTopBarError(null);
      try {
        const updatedProject = await updateProject(projectId, { [field]: value });
        queryClient.setQueryData(projectKeys.detail(projectId), updatedProject);
        setTopBarDirty(true);
        for (const queryKey of extraInvalidations) {
          void queryClient.invalidateQueries({ queryKey });
        }
      } catch (err) {
        setTopBarError(
          err instanceof Error
            ? `Could not save proposal settings: ${err.message}`
            : "Could not save proposal settings. The previous settings remain active.",
        );
      } finally {
        setTopBarSaving(false);
      }
    };
  }

  // Teacher change additionally invalidates the project list and trigger
  // status so `activeTeacher` recomputes, the capability-gated controls
  // (Thinking, Visual Budget) flip to match the new model's declared
  // capabilities, and the
  // eval strip's config-change nudge — driven by backend-tracked
  // evaluation_trigger_status — surfaces on the next poll.
  const handleTeacherChange = makeSettingHandler(
    project.teacher_model_config_id,
    "teacher_model_config_id",
    [projectKeys.list(), evaluationKeys.triggerStatus(projectId)],
  );
  const handleGenerationPresetChange = makeSettingHandler(
    project.labeling_generation_preset_key,
    "labeling_generation_preset_key",
  );
  const handleThinkingChange = makeSettingHandler(
    project.thinking_default_on,
    "thinking_default_on",
  );
  const handleVisualBudgetChange = makeSettingHandler(
    project.visual_budget_preset_key,
    "visual_budget_preset_key",
  );

  // ── Notice handlers ──────────────────────────────────────────────────

  function handleDismissColdStart() {
    setColdStartDismissed(true);
    try {
      localStorage.setItem(`vlm_cold_start_dismissed_${projectId}`, "true");
    } catch {
      /* ignore */
    }
  }

  async function handleDismissReminder() {
    // The backend owns the dismissal counter — POST the dedicated
    // endpoint, then refetch its reminder decision.
    try {
      await dismissReminder(projectId);
      await queryClient.invalidateQueries({
        queryKey: guidanceKeys.reminderStatus(projectId),
      });
    } catch {
      /* swallow */
    }
  }

  function handleReviewSchema() {
    navigate(`/projects/${projectId}/edit-guidance`);
  }

  async function handleRestoreOmitted() {
    setRestoringOmitted(true);
    setRestoreOmittedError(null);
    try {
      await restoreOmitted(projectId);
      await queryClient.invalidateQueries({ queryKey: projectKeys.list() });
      dispatch({ type: "FETCH_NEXT_START" });
    } catch (err) {
      // Do not swallow a user-initiated action: a failed Restore Omitted
      // would otherwise leave the SME clicking a button that appears to do
      // nothing. Surface it so they can retry.
      setRestoreOmittedError(
        err instanceof Error
          ? err.message
          : "Could not restore omitted images. Try again.",
      );
    }
    setRestoringOmitted(false);
  }

  function handleAdoptPrior(fieldName: string, value: unknown) {
    setFormValues((prev) => ({ ...prev, [fieldName]: value }));
    // The dirty tracking in ProposalForm will detect the change via onValuesChange
  }

  // ── Guard: queue empty with no examples → redirect to ingestion ──────

  useEffect(() => {
    if (state.phase !== "queue_empty") return;
    if (!countsLoaded) return;
    const totalExamples =
      counts.verified + counts.unlabeled + counts.omitted + counts.auto_labeled;
    if (totalExamples === 0) {
      navigate(`/projects/${projectId}/ready`, { replace: true });
    }
  }, [state.phase, countsLoaded, counts, projectId, navigate]);

  // ── Guard: no guidance ───────────────────────────────────────────────────

  if (!project.active_guidance_id) {
    return null; // redirect handled by useEffect
  }

  // ── Render ───────────────────────────────────────────────────────────────

  const isLoading =
    state.phase === "fetching_next" || state.phase === "fetching_proposal";
  const isActionInFlight = state.phase === "saving" || state.phase === "skipping";
  const topBarDisabled =
    topBarSaving || isActionInFlight || state.phase !== "reviewing" || showRetryPanel;

  const proposalFailed =
    state.proposal && state.proposal.invocation_status !== "success";

  const isDirty = dirtyFields.size > 0;
  const saveDisabled =
    topBarSaving ||
    isActionInFlight ||
    // Manual-labeling path: when the Teacher
    // call fails, the form still renders with empty defaults so the SME
    // can label from scratch. Save stays gated by the standard dirty +
    // rationale rules — a failure state on its own does not block it.
    (!isDirty && Boolean(proposalFailed)) ||
    (rationaleEnabled &&
      isDirty &&
      rationaleState !== "edited" &&
      rationaleState !== "approved");

  // Show rationale as read-only when anti-anchoring is OFF and no fields are dirty
  const showRationaleReadOnly =
    rationaleEnabled &&
    !project.rationale_anti_anchoring &&
    !isDirty &&
    rationaleState === "hidden";

  return (
    <div className="flex flex-1 flex-col" data-testid="labeling-page">
      {/* Acknowledgment banner when the FTUE setup chain auto-skipped
          entirely — reads location.state.setupAutoSkip forwarded by
          ConfirmDefaultsPage; renders nothing otherwise. */}
      <SetupAutoSkipBanner />

      {/* ── Top Bar ─────────────────────────────────────────────────── */}
      {/* z-30: .glass-info's backdrop-filter makes this bar (and each
          notice strip below) its own stacking context, so the context
          budget popover anchored inside the bar can only paint above the
          strips if the bar itself outranks them. */}
      <div
        className="glass-info relative z-30 flex items-center justify-between px-4 py-2"
        data-testid="labeling-top-bar"
      >
        <TeacherModelPicker
          teacherConfigs={modelConfigs?.items ?? []}
          currentTeacherId={project.teacher_model_config_id}
          onChange={handleTeacherChange}
          disabled={topBarDisabled}
          projectId={projectId}
        />
        <div className="flex items-center gap-4" data-testid="top-bar-controls">
          <TopBarControls
            generationPreset={project.labeling_generation_preset_key}
            thinkingOn={project.thinking_default_on}
            visualBudgetPreset={project.visual_budget_preset_key}
            thinkingVisible={thinkingVisible}
            visualBudgetVisible={visualBudgetShown}
            disabled={topBarDisabled}
            onGenerationPresetChange={handleGenerationPresetChange}
            onThinkingChange={handleThinkingChange}
            onVisualBudgetChange={handleVisualBudgetChange}
          />
          <IclChip
            idle={state.phase === "queue_empty"}
            count={
              state.proposal
                ? (state.proposal.icl_example_keys_used?.length ?? 0)
                : countsLoaded && counts.verified === 0
                  ? 0
                  : null
            }
          />
        </div>
      </div>
      {topBarError && (
        <div
          className="toast-error mx-4 mt-2 flex items-center px-4 py-3"
          data-testid="top-bar-settings-error"
          role="alert"
        >
          <Text kind="body/regular/sm">{topBarError}</Text>
        </div>
      )}

      {/* ── Inline Notices ──────────────────────────────────────────── */}
      <InlineNotices
        verifiedCount={counts.verified}
        reminderStatus={reminderStatus}
        coldStartDismissed={coldStartDismissed}
        pendingRelabel={counts.pending_relabel}
        priorRelabeled={counts.prior_relabeled}
        embeddingProvider={project.embedding_provider}
        onDismissColdStart={handleDismissColdStart}
        onDismissReminder={handleDismissReminder}
        onReviewSchema={handleReviewSchema}
      />

      {/* ── Evaluation Strip ──────────────────────────────────────────── */}
      {/* When a schema refinement reminder (InlineNotices) is active,
          suppress the first-pool evaluation banner so the SME sees one
          nudge at a time. Schema review wins because semantic schema
          changes invalidate existing labels — higher stakes than deferring
          the first evaluation by a few labels. Eligibility is the backend's
          reminder_status decision. */}
      <EvaluationStrip
        projectId={projectId}
        poolCount={poolCount}
        poolTarget={poolTarget}
        onShowResults={(run) => setResultsRun(run)}
        suppressFirstPoolTrigger={reminderStatus?.active_reminder != null}
      />

      {/* ── Main Content: Two-Panel Layout ────────────────────────────── */}
      {/* Queue-empty is a terminal state with no current example; collapsing
          to a single full-width column avoids showing
          ImagePanel's loading spinner against a null exampleKey. */}
      <div
        className="grid flex-1 gap-4 p-4 min-h-0"
        style={{
          gridTemplateColumns: state.phase === "queue_empty" ? "1fr" : "45fr 55fr",
        }}
      >
        {/* Left: Image Panel (hidden when the queue is empty) */}
        {/* `isLoading` scopes to `fetching_next` only — once we know the
            example key we start fetching the image in parallel with the
            Teacher proposal so the SME sees what they are labeling as
            soon as the image is ready, not when the proposal lands. */}
        {state.phase !== "queue_empty" && (
          <ImagePanel
            projectId={projectId}
            exampleKey={state.currentExampleKey}
            storageRef={currentStorageRef}
            isLoading={state.phase === "fetching_next"}
            onImageMissing={handleImageMissing}
            onImageLoaded={handleImageLoaded}
          />
        )}

        {/* Right: Proposal Form or Failure or Loading */}
        <div
          className="glass-card flex flex-col overflow-y-auto p-6"
          data-testid="proposal-panel"
        >
          {isLoading && (
            <div
              className="flex flex-1 flex-col items-center justify-center gap-3"
              data-testid="proposal-loading"
            >
              <Spinner size="large" aria-label="Loading proposal" />
              <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                Requesting Teacher proposal…
              </Text>
            </div>
          )}

          {/* Failure banner — rendered above the editable form
              so the SME can see what the model got wrong while labeling
              manually from scratch. */}
          {state.phase === "reviewing" &&
            state.proposal &&
            proposalFailed &&
            !showRetryPanel && (
              <div className="mb-4">
                <ProposalFailure
                  status={
                    state.proposal.invocation_status as Exclude<
                      typeof state.proposal.invocation_status,
                      "success"
                    >
                  }
                  validationErrorsCore={state.proposal.validation_errors_core}
                  validationErrorsAux={state.proposal.validation_errors_aux}
                  projectId={projectId}
                />
              </div>
            )}

          {state.phase === "reviewing" &&
            state.proposal &&
            !state.imageMissing &&
            !showRetryPanel && (
              <>
                <ProposalForm
                  schemaFields={schemaFields}
                  proposalJson={state.proposal.proposal_json ?? EMPTY_PROPOSAL}
                  onValuesChange={handleValuesChange}
                  disabled={isActionInFlight}
                  resetKey={resetKey}
                  priorLabel={priorLabelSnapshot}
                  onAdoptPrior={handleAdoptPrior}
                />

                {/* Rationale: read-only display when anti-anchoring OFF and no
                  edits. Suppressed when the proposal failed — there is no
                  Teacher rationale to display. */}
                {showRationaleReadOnly && !proposalFailed && (
                  <div
                    className="mt-3 border-accent-aux pl-4"
                    data-testid="rationale-note-display"
                  >
                    <Text
                      kind="label/regular/sm"
                      style={{ color: "var(--text-secondary)" }}
                      className="mb-1 block"
                    >
                      rationale_note
                    </Text>
                    <Text
                      kind="body/regular/sm"
                      className="glass-info px-3 py-2 text-sm"
                      style={{ color: "var(--text-secondary)", display: "block" }}
                    >
                      {String(
                        state.proposal.proposal_json?.[RATIONALE_NOTE_FIELD_NAME] ?? "",
                      )}
                    </Text>
                  </div>
                )}

                {/* Rationale panel: appears when fields are dirty */}
                {rationaleEnabled && (
                  <RationalePanel
                    state={rationaleState}
                    rationaleText={rationaleText}
                    onRationaleTextChange={handleRationaleTextChange}
                    onGenerateAI={handleGenerateAIRationale}
                    onApproveAI={handleApproveAIRationale}
                    regenerationError={regenerationError}
                    disabled={isActionInFlight}
                  />
                )}

                {/* Aux warnings — only for successful proposals. In failure
                  states the ProposalFailure banner above already lists any
                  Core errors, and Aux warnings attached to a null proposal
                  aren't actionable. */}
                {!proposalFailed && state.proposal.validation_errors_aux.length > 0 && (
                  <div className="mt-3 px-4">
                    <Text
                      kind="label/regular/xs"
                      style={{ color: "var(--warning-amber)" }}
                    >
                      Aux warnings: {state.proposal.validation_errors_aux.join(", ")}
                    </Text>
                  </div>
                )}
              </>
            )}

          {state.phase === "queue_empty" && (
            <>
              {restoreOmittedError && (
                <div
                  className="toast-error flex items-center gap-2 px-4 py-3"
                  data-testid="restore-omitted-error"
                >
                  <Text kind="body/regular/sm">{restoreOmittedError}</Text>
                </div>
              )}
              <QueueEmpty
                verified={counts.verified}
                unlabeled={counts.unlabeled}
                omitted={counts.omitted}
                onAddImages={() => navigate(`/projects/${projectId}/ready`)}
                onRestoreOmitted={handleRestoreOmitted}
                restoringOmitted={restoringOmitted}
                onGoToScaleUp={() => navigate(`/projects/${projectId}/scale-up`)}
              />
            </>
          )}

          {state.phase === "error" && (
            <div
              className="flex flex-1 flex-col items-center justify-center gap-3"
              data-testid="labeling-error"
            >
              <div className="toast-error flex items-center gap-2 px-4 py-3">
                <Text kind="body/regular/sm">
                  {state.error ?? "An error occurred."}
                </Text>
              </div>
              <Button
                kind="secondary"
                onClick={() => dispatch({ type: "FETCH_NEXT_START" })}
              >
                Retry
              </Button>
            </div>
          )}

          {/* ── Action Buttons / Retry Panel ─────────────────────────── */}
          {/* Push actions to the bottom of the panel; the form + rationale
              panel fill the column above. Applies uniformly for success and
              failure states — the form renders in both. */}
          {state.phase === "reviewing" &&
            state.proposal &&
            !state.imageMissing &&
            !showRetryPanel && (
              <div
                className="flex items-center justify-end gap-3 border-t pt-4 mt-auto"
                style={{ borderColor: "var(--glass-border)" }}
                data-testid="labeling-actions"
              >
                {/* Reset — shown when any field is dirty */}
                {isDirty && (
                  <Button
                    kind="secondary"
                    onClick={handleReset}
                    disabled={isActionInFlight}
                    data-testid="reset-btn"
                  >
                    <RotateCcw size={14} /> Reset
                  </Button>
                )}
                <Button
                  kind="secondary"
                  onClick={handleSkip}
                  disabled={isActionInFlight || topBarSaving}
                  data-testid="skip-btn"
                >
                  Skip
                </Button>
                {/* When the proposal failed and Save is still gated (empty
                  form), Retry is the recommended recovery and takes the
                  primary treatment; once the SME types and Save enables,
                  Retry drops back to secondary so Save stays the screen's
                  sole primary. */}
                <Button
                  kind={proposalFailed && saveDisabled ? "primary" : "secondary"}
                  className={
                    proposalFailed && saveDisabled ? "nvidia-green-button" : undefined
                  }
                  onClick={handleRetry}
                  disabled={isActionInFlight || topBarSaving}
                  data-testid="retry-btn"
                >
                  Retry
                </Button>
                {/* Save is always rendered, even in proposal-failed states, so
                  the SME can label manually from scratch — the SME fills in
                  field values by hand. The `saveDisabled` derivation above
                  keeps the button gated by the standard dirty + rationale
                  rules; in a failure state with an empty form, Save stays
                  disabled until the SME types something. */}
                <Button
                  kind="primary"
                  className="nvidia-green-button"
                  onClick={handleSave}
                  disabled={saveDisabled}
                  title={
                    rationaleEnabled && isDirty && saveDisabled
                      ? rationaleState === "ai_review_required"
                        ? "Approve the AI-generated rationale before saving"
                        : "Update the rationale before saving"
                      : undefined
                  }
                  data-testid="save-btn"
                >
                  {isActionInFlight ? (
                    <>
                      <Spinner size="small" aria-label="Saving" /> Saving...
                    </>
                  ) : (
                    "Save"
                  )}
                </Button>
              </div>
            )}

          {/* Retry Panel — replaces the proposal form + action strip when
              open. Anchors at the top of the right column so
              the "Retry with different settings" heading lands where the
              SME's eye starts scanning; the panel's own action row
              bottom-anchors via mt-auto, so the flex chain must reach the
              card (flex-1 wrapper). */}
          {state.phase === "reviewing" && state.proposal && showRetryPanel && (
            <div className="flex flex-1 flex-col">
              <RetryPanel
                teacherConfigs={modelConfigs?.items ?? []}
                guidanceVersions={guidanceList?.items ?? []}
                currentTeacherId={project.teacher_model_config_id}
                currentGuidanceId={project.active_guidance_id ?? ""}
                currentPreset={project.labeling_generation_preset_key}
                currentThinking={project.thinking_default_on}
                currentVisualBudget={project.visual_budget_preset_key}
                onRetry={handleRetryConfirm}
                onCancel={handleRetryCancel}
              />
            </div>
          )}

          {/* Missing image: the left panel (ImagePanel) already
              renders the full diagnostic — "Image not found at original
              location", expected path, and [Report Missing Files]. The right
              panel holds a short explainer so the column doesn't sit blank
              above the [Skip] action. */}
          {state.phase === "reviewing" && state.imageMissing && (
            <div className="flex flex-1 flex-col" data-testid="missing-image-panel">
              <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
                <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
                  Image not found
                </Text>
                <Text
                  kind="body/regular/sm"
                  style={{ color: "var(--text-muted)", maxWidth: 320 }}
                >
                  The image file at the recorded path could not be loaded. See the left
                  panel to report missing files, or skip this image to continue.
                </Text>
              </div>
              <div
                className="mt-auto flex items-center justify-end gap-3 border-t pt-4"
                style={{ borderColor: "var(--glass-border)" }}
                data-testid="labeling-actions-missing"
              >
                <Button
                  kind="secondary"
                  onClick={handleSkip}
                  disabled={isActionInFlight}
                  data-testid="skip-btn"
                >
                  Skip
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Status Bar (Verified / Unlabeled / Omitted counters) ─────── */}
      {/* The horizontal counter strip sits
       *below* the main two-pane content, above the footer. */}
      <LabelingStatusBar
        verified={counts.verified}
        unlabeled={counts.unlabeled}
        omitted={counts.omitted}
      />

      {/* ── Footer ────────────────────────────────────────────────────── */}
      <div
        className="flex items-center justify-between px-4 py-3 border-t"
        style={{ borderColor: "var(--glass-border)" }}
        data-testid="labeling-footer"
      >
        <Button
          kind="secondary"
          onClick={() => navigate(`/projects/${projectId}/ready`)}
          data-testid="add-images-btn"
        >
          <Plus size={14} /> Add Images
        </Button>
        <div className="flex items-center gap-3" data-testid="footer-right">
          {hasStudentModels && (
            <Button
              kind="tertiary"
              className="glass-pill"
              onClick={() => navigate(`/projects/${projectId}/compare`)}
              data-testid="models-results-indicator"
            >
              Models &amp; Results
              <ChevronRight size={14} style={{ marginLeft: 4 }} />
            </Button>
          )}
          <Button
            kind="tertiary"
            className="glass-pill"
            onClick={() => navigate(`/projects/${projectId}/scale-up`)}
            data-testid="scaleup-indicator"
          >
            Scale-Up: {gatePassCount}/5
            <ChevronRight size={14} style={{ marginLeft: 4 }} />
          </Button>
        </div>
      </div>

      {/* ── Results Panel overlay ────────────────────────────────────── */}
      {resultsRun && (
        <ResultsPanel
          run={resultsRun}
          onClose={() => setResultsRun(null)}
          teacherConfigs={modelConfigs?.items ?? []}
          guidanceVersions={guidanceList?.items ?? []}
          minPerValueF1Threshold={project.scaleup_min_per_value_f1_threshold}
        />
      )}
    </div>
  );
}

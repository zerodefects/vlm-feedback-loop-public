// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Student Training configuration screen.
 *
 * The default "Validate training setup" path is deliberately small: one
 * preferred 2B Student base, Quick preset, and one FP8 comparison beside the
 * full-precision baseline. The backend owns TAO/data preflight and exact
 * export-eligible counts. A confirmation expands the selection into the exact
 * remote job count before any TAO work starts.
 */

import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Button, Modal, Spinner, Text } from "@kui/react";
import { AlertTriangle, CheckCircle2, CircleX } from "lucide-react";

import {
  createTrainingSuite,
  listStudentBaseModelConfigs,
  resolveTrainingPresets,
  runTrainingPreflight,
} from "@/api/training";
import { trainingKeys } from "@/api/query-keys";
import { ApiError, parseApiErrorDetail } from "@/api/client";
import { ActionRequestPanel } from "@/components/ActionRequestPanel";
import { PageContainer } from "@/components/common/PageContainer";
import { SectionCard } from "@/components/common/SectionCard";
import { SectionHeading } from "@/components/common/SectionHeading";
import { SegmentedControl } from "@/components/SegmentedControl";
import { useSetupContext } from "@/pages/ProjectSetupLayout";
import { BaseModelSelector } from "@/components/training/BaseModelSelector";
import { QuantizationCheckboxes } from "@/components/training/QuantizationCheckboxes";
import { TrainingAdvancedExpander } from "@/components/training/TrainingAdvancedExpander";
import { formatModelDisplayName } from "@/lib/model-display";
import { hasReadyTaoBaseExperiment } from "@/lib/training/baseModelReadiness";
import { PRESET_HELP, PRESET_LABEL } from "@/lib/training/presetCopy";
import type { QuantizationScheme, TrainingPreset } from "@/types/training";

const PRESETS: TrainingPreset[] = ["quick", "standard", "high_quality", "max_quality"];
const VALIDATION_QUANTIZATION: QuantizationScheme[] = ["FP8_DYNAMIC"];
const COMPARISON_QUANTIZATION: QuantizationScheme[] = ["FP8_DYNAMIC", "W4A16"];
const RECOMMENDED_VERIFIED_IMAGES = 150;

type TrainingIntent = "validate" | "compare";

interface BaseModelOption {
  modelConfigId: string;
  modelName: string;
  provisioned: boolean;
}

function genIdempotencyKey(): string {
  if (
    typeof globalThis.crypto !== "undefined" &&
    typeof globalThis.crypto.randomUUID === "function"
  ) {
    return globalThis.crypto.randomUUID();
  }
  return `ts-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function pickValidationBaseId(options: BaseModelOption[]): string | null {
  const preference = [
    "cosmos-reason2-2b",
    "cosmos-reason-2-2b",
    "cosmos3-nano",
    "cosmos-3-nano",
  ];
  for (const token of preference) {
    const match = options.find((option) =>
      option.modelName.toLowerCase().includes(token),
    );
    if (match) return match.modelConfigId;
  }
  return options[0]?.modelConfigId ?? null;
}

function friendlySubmitError(error: unknown): string {
  const detail = parseApiErrorDetail(error);
  if (detail) {
    return detail.replace(/^validation:\s*/i, "").replace(/^conflict:\s*/i, "");
  }
  if (error instanceof ApiError) {
    console.error("[training] suite creation failed:", error.status, error.body);
  } else {
    console.error("[training] suite creation failed:", error);
  }
  return "Training could not start. Re-run the readiness check and try again.";
}

export function StudentTrainingPage() {
  const { projectId } = useSetupContext();
  const navigate = useNavigate();

  const [intent, setIntent] = useState<TrainingIntent>("validate");
  const [selectedBaseIds, setSelectedBaseIds] = useState<string[] | null>(null);
  const [preset, setPreset] = useState<TrainingPreset>("quick");
  const [includeAutoLabeled, setIncludeAutoLabeled] = useState(true);
  const [quantSchemes, setQuantSchemes] = useState<QuantizationScheme[]>(
    VALIDATION_QUANTIZATION,
  );
  const [idempotencyKey] = useState(genIdempotencyKey);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [confirmationOpen, setConfirmationOpen] = useState(false);
  const [showTaoRequest, setShowTaoRequest] = useState(false);

  const { data: studentBases, isLoading: basesLoading } = useQuery({
    queryKey: trainingKeys.studentBases(projectId),
    queryFn: () => listStudentBaseModelConfigs(projectId),
  });

  const baseModelOptions = useMemo<BaseModelOption[]>(
    () =>
      (studentBases?.items ?? []).map((model) => ({
        modelConfigId: model.model_config_id,
        modelName: model.model_name,
        provisioned: hasReadyTaoBaseExperiment(model),
      })),
    [studentBases],
  );

  useEffect(() => {
    if (selectedBaseIds !== null || baseModelOptions.length === 0) return;
    const preferred = pickValidationBaseId(baseModelOptions);
    setSelectedBaseIds(preferred ? [preferred] : []);
  }, [baseModelOptions, selectedBaseIds]);

  const effectiveSelectedIds = useMemo(() => selectedBaseIds ?? [], [selectedBaseIds]);

  const { data: resolvedPresetData } = useQuery({
    queryKey: trainingKeys.presets(projectId, effectiveSelectedIds),
    queryFn: () => resolveTrainingPresets(projectId, effectiveSelectedIds),
    enabled: effectiveSelectedIds.length > 0,
    staleTime: Number.POSITIVE_INFINITY,
  });

  const {
    data: preflight,
    isFetching: preflightLoading,
    isError: preflightError,
    refetch: refetchPreflight,
  } = useQuery({
    queryKey: trainingKeys.preflight(
      projectId,
      effectiveSelectedIds,
      includeAutoLabeled,
    ),
    queryFn: () =>
      runTrainingPreflight(projectId, effectiveSelectedIds, includeAutoLabeled),
    enabled: effectiveSelectedIds.length > 0,
    retry: false,
  });

  const submitMutation = useMutation({
    mutationFn: () =>
      createTrainingSuite(projectId, {
        student_base_model_config_ids: effectiveSelectedIds,
        training_preset: preset,
        include_auto_labeled: includeAutoLabeled,
        export_field_mode: "all",
        quantization_schemes: quantSchemes,
        idempotency_key: idempotencyKey,
      }),
    onSuccess: (suite) => {
      navigate(`../training/${suite.training_suite_id}`);
    },
    onError: (error) => {
      setSubmitError(friendlySubmitError(error));
    },
  });

  const selectedModelDescriptors = useMemo(
    () =>
      baseModelOptions.filter((model) =>
        effectiveSelectedIds.includes(model.modelConfigId),
      ),
    [baseModelOptions, effectiveSelectedIds],
  );

  const dataSummary = preflight?.data_summary;
  const totalVerified =
    (dataSummary?.verified_training_count ?? 0) + (dataSummary?.test_pool_count ?? 0);
  const excludedCount =
    (dataSummary?.excluded_test_pool_count ?? 0) +
    (dataSummary?.excluded_auto_labeled_count ?? 0);
  const taoFailures =
    preflight?.checks.filter(
      (check) => !check.passed && check.check_name !== "verified_train_examples",
    ) ?? [];
  const dataFailure = preflight?.checks.find(
    (check) => check.check_name === "verified_train_examples" && !check.passed,
  );
  const workflowBusy = submitMutation.isPending;
  const canStart =
    effectiveSelectedIds.length > 0 &&
    !workflowBusy &&
    !preflightLoading &&
    !preflightError &&
    preflight?.status === "passed" &&
    (dataSummary?.usable_training_count ?? 0) > 0;

  const jobsPerModel = 2 + quantSchemes.length * 2;
  const exactJobCount = effectiveSelectedIds.length * jobsPerModel;
  const variantCount = effectiveSelectedIds.length * (1 + quantSchemes.length);
  const trainJobCount = effectiveSelectedIds.length;
  const baselineEvaluationJobCount = effectiveSelectedIds.length;
  const quantizeJobCount = effectiveSelectedIds.length * quantSchemes.length;
  const quantizedEvaluationJobCount = quantizeJobCount;

  function selectIntent(nextIntent: TrainingIntent) {
    setIntent(nextIntent);
    setSubmitError(null);
    setShowTaoRequest(false);
    if (nextIntent === "validate") {
      const preferred = pickValidationBaseId(baseModelOptions);
      setSelectedBaseIds(preferred ? [preferred] : []);
      setPreset("quick");
      setQuantSchemes(VALIDATION_QUANTIZATION);
    } else {
      setSelectedBaseIds(baseModelOptions.map((model) => model.modelConfigId));
      setPreset("standard");
      setQuantSchemes(COMPARISON_QUANTIZATION);
    }
  }

  if (basesLoading) {
    return (
      <PageContainer data-testid="student-training-page">
        <div className="flex items-center justify-center py-20">
          <Spinner aria-label="Loading student base models" size="large" />
        </div>
      </PageContainer>
    );
  }

  return (
    <PageContainer data-testid="student-training-page">
      <Text kind="title/md">Student Training</Text>
      <Text
        kind="body/regular/sm"
        className="block"
        style={{ color: "var(--text-secondary)" }}
      >
        Fine-tune and evaluate a Student in TAO. Production deployment remains an
        infrastructure handoff after quality and serving validation.
      </Text>

      <SectionCard data-testid="training-intent-card">
        <SectionHeading>RUN TYPE</SectionHeading>
        <SegmentedControl
          testId="training-intent"
          options={[
            { key: "validate", label: "Validate training setup" },
            { key: "compare", label: "Compare candidate variants" },
          ]}
          value={intent}
          disabled={workflowBusy}
          onChange={(key) => selectIntent(key as TrainingIntent)}
        />
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          {intent === "validate"
            ? "Recommended first run: one 2B base, Quick training, baseline + FP8."
            : "Advanced campaign: review every selected base and compression variant before starting."}
        </Text>
      </SectionCard>

      <SectionCard>
        <SectionHeading>BASE MODELS</SectionHeading>
        <BaseModelSelector
          options={baseModelOptions}
          selected={effectiveSelectedIds}
          onChange={setSelectedBaseIds}
          disabled={workflowBusy}
        />
        {effectiveSelectedIds.length === 0 && (
          <Text
            kind="body/regular/sm"
            className="block"
            style={{ color: "var(--warning-amber, #f59e0b)" }}
            data-testid="base-model-required-hint"
          >
            Select at least one base model.
          </Text>
        )}
      </SectionCard>

      <SectionCard>
        <SectionHeading>TRAINING INTENSITY</SectionHeading>
        <select
          className={`glass-input w-full max-w-md ${
            workflowBusy ? "cursor-not-allowed opacity-70" : ""
          }`}
          value={preset}
          disabled={workflowBusy}
          onChange={(event) => setPreset(event.target.value as TrainingPreset)}
          data-testid="training-preset-select"
        >
          {PRESETS.map((option) => (
            <option key={option} value={option}>
              {PRESET_LABEL[option]}
            </option>
          ))}
        </select>
        <Text
          kind="body/regular/sm"
          className="block"
          style={{ color: "var(--text-muted)" }}
        >
          {PRESET_HELP[preset]}
        </Text>
        <TrainingAdvancedExpander
          preset={preset}
          baseModels={selectedModelDescriptors}
          resolvedPresets={resolvedPresetData?.resolved_presets}
        />
      </SectionCard>

      <SectionCard data-testid="training-data-card">
        <SectionHeading>TRAINING DATA</SectionHeading>
        {preflightLoading && !dataSummary ? (
          <div className="flex items-center gap-2 py-2">
            <Spinner aria-label="Calculating training data" />
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              Calculating export-eligible examples…
            </Text>
          </div>
        ) : (
          <>
            <DataCountRow
              label="Verified Training Pool"
              value={dataSummary?.verified_training_count ?? 0}
              detail="Included · active Guidance · Test Pool excluded"
              testId="training-data-verified-count"
            />
            <DataCountRow
              label="Test Pool"
              value={dataSummary?.test_pool_count ?? 0}
              detail="Excluded from training · used only for evaluation"
              testId="training-data-test-pool-count"
            />
            <div
              className={`flex items-center justify-between gap-4 ${
                workflowBusy ? "opacity-70" : ""
              }`}
            >
              <label
                className={`flex items-center gap-3 ${
                  workflowBusy ? "cursor-not-allowed" : "cursor-pointer"
                }`}
              >
                <input
                  type="checkbox"
                  className="glass-input"
                  checked={includeAutoLabeled}
                  disabled={workflowBusy}
                  onChange={(event) => setIncludeAutoLabeled(event.target.checked)}
                  data-testid="training-data-auto-labeled-checkbox"
                />
                <Text kind="body/regular/sm">Auto-Labeled eligible</Text>
              </label>
              <Text
                kind="body/regular/sm"
                style={{ color: "var(--text-secondary)" }}
                data-testid="training-data-auto-labeled-count"
              >
                {dataSummary?.auto_labeled_eligible_count ?? 0} examples ·{" "}
                {includeAutoLabeled ? "included" : "excluded by your selection"}
              </Text>
            </div>
            <DataCountRow
              label="Excluded"
              value={excludedCount}
              detail={`${dataSummary?.excluded_test_pool_count ?? 0} Test Pool · ${
                dataSummary?.excluded_auto_labeled_count ?? 0
              } unselected Auto-Labeled`}
              testId="training-data-excluded-count"
            />
            <div
              className="flex items-center justify-between pt-3 border-t"
              style={{
                borderColor: "var(--glass-border, rgba(255,255,255,0.08))",
              }}
            >
              <Text kind="label/bold/sm">Usable training total</Text>
              <Text kind="label/bold/sm" data-testid="training-data-total">
                {dataSummary?.usable_training_count ?? 0} examples
              </Text>
            </div>
          </>
        )}
        {dataFailure && (
          <Text
            kind="body/regular/sm"
            className="block"
            style={{ color: "var(--warning-amber, #f59e0b)" }}
            data-testid="no-verified-warning"
          >
            {dataFailure.message}
          </Text>
        )}
        {dataSummary && totalVerified < RECOMMENDED_VERIFIED_IMAGES && (
          <div
            className="glass-inner-panel rounded-[14px] p-4 flex items-start gap-3"
            style={{ boxShadow: "inset 4px 0 0 var(--warning-amber, #f59e0b)" }}
            data-testid="small-training-data-warning"
          >
            <AlertTriangle
              size={16}
              className="mt-0.5 flex-shrink-0"
              style={{ color: "var(--warning-amber, #f59e0b)" }}
            />
            <Text kind="body/regular/sm">
              Fewer than 150 Verified images. A successful run can validate the TAO
              wiring, but it is not evidence of a production-quality model.
            </Text>
          </div>
        )}
      </SectionCard>

      <SectionCard>
        <SectionHeading>QUANTIZATION</SectionHeading>
        <Text
          kind="body/regular/sm"
          className="block"
          style={{ color: "var(--text-muted)" }}
        >
          Each selected scheme adds one quantize job and one evaluation job beside the
          full-precision baseline.
        </Text>
        <QuantizationCheckboxes
          schemes={quantSchemes}
          onChange={setQuantSchemes}
          disabled={workflowBusy}
        />
      </SectionCard>

      <SectionCard data-testid="training-preflight-card">
        <SectionHeading>READINESS CHECK</SectionHeading>
        {effectiveSelectedIds.length === 0 ? (
          <Text kind="body/regular/sm">Select a Student base to run preflight.</Text>
        ) : preflightLoading && !preflight ? (
          <div className="flex items-center gap-2">
            <Spinner aria-label="Running training preflight" />
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              Checking TAO, workspace, Student base, and data…
            </Text>
          </div>
        ) : preflightError ? (
          <div className="space-y-3">
            <Text kind="body/regular/sm">
              TAO readiness could not be checked. No training work has started.
            </Text>
            <Button
              kind="secondary"
              onClick={() => void refetchPreflight()}
              data-testid="training-preflight-retry"
            >
              Retry readiness check
            </Button>
          </div>
        ) : preflight ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              {preflight.status === "passed" ? (
                <CheckCircle2 size={16} style={{ color: "var(--accent-green)" }} />
              ) : (
                <CircleX size={16} className="text-error" />
              )}
              <Text kind="label/bold/sm">
                {preflight.status === "passed"
                  ? "Ready to create the training jobs"
                  : "Training setup is incomplete"}
              </Text>
            </div>
            {preflight.checks.map((check, index) => (
              <div
                key={`${check.check_name}-${check.model_config_id ?? "global"}-${index}`}
                className="flex items-start gap-2"
              >
                {check.passed ? (
                  <CheckCircle2
                    size={14}
                    className="mt-0.5 flex-shrink-0"
                    style={{ color: "var(--accent-green)" }}
                  />
                ) : (
                  <CircleX size={14} className="mt-0.5 flex-shrink-0 text-error" />
                )}
                <Text kind="body/regular/sm">{check.message}</Text>
              </div>
            ))}
            <div className="flex flex-wrap gap-2">
              <Button
                kind="secondary"
                onClick={() => void refetchPreflight()}
                disabled={preflightLoading}
                data-testid="training-preflight-retry"
              >
                Retry readiness check
              </Button>
              {taoFailures.length > 0 && (
                <Button
                  kind="tertiary"
                  onClick={() => setShowTaoRequest((value) => !value)}
                  data-testid="training-request-tao-setup"
                >
                  {showTaoRequest ? "Hide TAO setup request" : "Request TAO setup"}
                </Button>
              )}
            </div>
          </div>
        ) : null}
        {showTaoRequest && taoFailures.length > 0 && (
          <div className="pt-3" data-testid="training-tao-action-request">
            <ActionRequestPanel
              projectId={projectId}
              requestType="tao_setup"
              onClose={() => setShowTaoRequest(false)}
            />
          </div>
        )}
      </SectionCard>

      {submitError && (
        <div
          className="glass-card p-6"
          style={{ borderLeft: "3px solid var(--error-red)" }}
          data-testid="training-submit-error"
        >
          <Text kind="label/bold/sm" className="block text-error">
            Training did not start
          </Text>
          <Text kind="body/regular/sm" className="block text-error mt-2">
            {submitError}
          </Text>
        </div>
      )}

      <div className="flex items-center justify-end gap-3">
        <Button
          kind="secondary"
          onClick={() => navigate("../scale-up")}
          data-testid="training-cancel"
        >
          Cancel
        </Button>
        <Button
          kind="primary"
          className="nvidia-green-button"
          onClick={() => {
            setSubmitError(null);
            setConfirmationOpen(true);
          }}
          disabled={!canStart}
          data-testid="training-start"
        >
          Review {exactJobCount || ""} jobs
        </Button>
      </div>

      <Modal
        open={confirmationOpen}
        onOpenChange={(open) => setConfirmationOpen(open)}
        dismissible={!workflowBusy}
        slotHeading={`Start ${exactJobCount} TAO jobs?`}
        slotFooter={
          <div className="flex justify-end gap-3">
            <Button
              kind="secondary"
              onClick={() => setConfirmationOpen(false)}
              disabled={workflowBusy}
              data-testid="training-confirm-cancel"
            >
              Go back
            </Button>
            <Button
              kind="primary"
              className="nvidia-green-button"
              onClick={() => {
                setConfirmationOpen(false);
                submitMutation.mutate();
              }}
              disabled={workflowBusy}
              data-testid="training-confirm-start"
            >
              {workflowBusy ? "Starting…" : "Start training"}
            </Button>
          </div>
        }
      >
        <div className="space-y-4" data-testid="training-confirmation-summary">
          <Text kind="body/regular/sm" className="block">
            TAO will run {exactJobCount} sequential jobs and produce {variantCount}{" "}
            Student variant{variantCount === 1 ? "" : "s"}.
          </Text>
          <ConfirmationRow
            label="Models"
            value={selectedModelDescriptors
              .map((model) => formatModelDisplayName(model.modelName))
              .join(", ")}
          />
          <ConfirmationRow
            label="Preset"
            value={`${PRESET_LABEL[preset]} (resolved per selected model)`}
          />
          <ConfirmationRow
            label="Variants"
            value={`Full precision${
              quantSchemes.length > 0 ? ` + ${quantSchemes.join(" + ")}` : ""
            }`}
          />
          <ConfirmationRow
            label="Jobs"
            value={`${trainJobCount} train · ${baselineEvaluationJobCount} baseline evaluate · ${quantizeJobCount} quantize · ${quantizedEvaluationJobCount} quantized evaluate`}
          />
          <ConfirmationRow
            label="Training data"
            value={`${dataSummary?.usable_training_count ?? 0} usable · ${
              dataSummary?.test_pool_count ?? 0
            } held out for evaluation`}
          />
          <ConfirmationRow
            label="Duration"
            value="Long-running; duration depends on the TAO cluster and selected model."
          />
          {totalVerified < RECOMMENDED_VERIFIED_IMAGES && (
            <div
              className="glass-inner-panel rounded-[14px] p-4"
              style={{ boxShadow: "inset 4px 0 0 var(--warning-amber, #f59e0b)" }}
            >
              <Text kind="body/regular/sm">
                This project has fewer than 150 Verified images. Confirm only if this is
                a wiring-validation run, not a production-quality claim.
              </Text>
            </div>
          )}
          <Text
            kind="body/regular/xs"
            className="block"
            style={{ color: "var(--text-muted)" }}
          >
            Training consumes remote TAO infrastructure. You can monitor the suite after
            leaving this page.
          </Text>
        </div>
      </Modal>
    </PageContainer>
  );
}

function DataCountRow({
  label,
  value,
  detail,
  testId,
}: {
  label: string;
  value: number;
  detail: string;
  testId: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex flex-col">
        <Text kind="body/regular/sm">{label}</Text>
        <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
          {detail}
        </Text>
      </div>
      <Text
        kind="body/regular/sm"
        style={{ color: "var(--text-secondary)" }}
        data-testid={testId}
      >
        {value} examples
      </Text>
    </div>
  );
}

function ConfirmationRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-6">
      <Text kind="label/bold/sm">{label}</Text>
      <Text
        kind="body/regular/sm"
        style={{ color: "var(--text-secondary)", textAlign: "right" }}
      >
        {value}
      </Text>
    </div>
  );
}

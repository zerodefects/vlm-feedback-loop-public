// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Compatible local-Teacher chooser for post-onboarding NIM Configuration.
 *
 * The environment API owns hardware compatibility; the project catalog owns
 * ModelConfig identities. This component joins the two without reimplementing
 * GPU policy in TypeScript, then drives the backend's exact reuse/replace
 * lifecycle. The first deploy call is always non-destructive. Only a
 * structured 409 naming a replaceable resident opens the confirmation modal.
 */

import { useEffect, useMemo, useState } from "react";
import { Badge, Button, Modal, Select, Spinner, Text } from "@kui/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Server, TriangleAlert } from "lucide-react";

import { parseApiErrorDetail } from "@/api/client";
import {
  deployLocalNim,
  listLocalNimDeployments,
  parseLocalNimGpuConflict,
} from "@/api/nim";
import {
  environmentKeys,
  localNimKeys,
  modelConfigKeys,
  projectKeys,
} from "@/api/query-keys";
import { LOCAL_NIM_POLL_INTERVAL_MS } from "@/lib/local-nim";
import { localTeacherDisplayName } from "@/lib/model-display";
import type {
  EnvironmentResponse,
  LocalNimGpuConflict,
  ModelConfigResponse,
} from "@/types/nim";

interface LocalTeacherDeploymentPanelProps {
  projectId: string;
  activeTeacherConfigId: string | null;
  environment: EnvironmentResponse;
  modelConfigs: ModelConfigResponse[];
  onSwitchToHosted: () => void;
}

interface CompatibleModel {
  config: ModelConfigResponse;
  modelName: string;
  gpuMinimumGb: number;
  computeMinimum: number | null;
  recommended: boolean;
}

function modelQualityNote(modelName: string): string {
  if (modelName.includes("nemotron-3-nano-omni")) {
    return "Best measured long-horizon multiclass quality; Teacher-only.";
  }
  if (modelName.includes("cosmos3-nano")) {
    return "Quality-ranked Cosmos fallback with the strongest schema stability.";
  }
  if (modelName.includes("cosmos3-super")) {
    return "Selectable larger Cosmos model; strongest result on the VisA anomaly task.";
  }
  if (modelName.endsWith("reason2-8b")) {
    return "Trainable Cosmos baseline with the deepest measured useful ICL range.";
  }
  if (modelName.endsWith("reason2-2b")) {
    return "Smallest supported local Teacher for 36–55 GB GPUs.";
  }
  return "Blueprint-supported local Teacher.";
}

function residentName(conflict: LocalNimGpuConflict): string {
  return localTeacherDisplayName(conflict.resident?.model_name);
}

export function LocalTeacherDeploymentPanel({
  projectId,
  activeTeacherConfigId,
  environment,
  modelConfigs,
  onSwitchToHosted,
}: LocalTeacherDeploymentPanelProps) {
  const queryClient = useQueryClient();
  const [selectedConfigId, setSelectedConfigId] = useState("");
  const [replacementConflict, setReplacementConflict] =
    useState<LocalNimGpuConflict | null>(null);
  const [trackedDeploymentId, setTrackedDeploymentId] = useState<string | null>(null);
  const [reusedModelName, setReusedModelName] = useState<string | null>(null);
  const [deployError, setDeployError] = useState<string | null>(null);

  const compatibleModels = useMemo<CompatibleModel[]>(() => {
    const configsByName = new Map(
      modelConfigs.map((config) => [config.model_name, config]),
    );
    const compatible = environment.local_deployable_models.flatMap((entry) => {
      const config = configsByName.get(entry.model_name);
      if (
        !entry.fits ||
        !config ||
        !config.local_deploy_metadata ||
        !config.eligible_roles.includes("teacher")
      ) {
        return [];
      }
      return [
        {
          config,
          modelName: entry.model_name,
          gpuMinimumGb: entry.gpu_memory_minimum_gb,
          computeMinimum: entry.compute_capability_minimum ?? null,
          recommended:
            entry.model_name === environment.recommended_local_teacher_model_name,
        },
      ];
    });
    return compatible.sort(
      (left, right) => Number(right.recommended) - Number(left.recommended),
    );
  }, [environment, modelConfigs]);

  useEffect(() => {
    if (
      selectedConfigId &&
      compatibleModels.some(
        (model) => model.config.model_config_id === selectedConfigId,
      )
    ) {
      return;
    }
    const activeLocal = compatibleModels.find(
      (model) =>
        model.config.model_config_id === activeTeacherConfigId &&
        (environment.active_local_nim_residents ?? []).some(
          (resident) =>
            resident.model_name === model.modelName && resident.status === "running",
        ),
    );
    const recommended = compatibleModels.find((model) => model.recommended);
    setSelectedConfigId(
      activeLocal?.config.model_config_id ??
        recommended?.config.model_config_id ??
        compatibleModels[0]?.config.model_config_id ??
        "",
    );
  }, [
    activeTeacherConfigId,
    compatibleModels,
    environment.active_local_nim_residents,
    selectedConfigId,
  ]);

  const selected = compatibleModels.find(
    (model) => model.config.model_config_id === selectedConfigId,
  );
  const exactResident = (environment.active_local_nim_residents ?? []).find(
    (resident) =>
      resident.role === "teacher" &&
      resident.model_name === selected?.modelName &&
      resident.status === "running",
  );
  const activeAndRunning =
    selected?.config.model_config_id === activeTeacherConfigId &&
    exactResident !== undefined;

  const deploymentsQuery = useQuery({
    queryKey: localNimKeys.deployments(projectId),
    queryFn: () => listLocalNimDeployments(projectId),
    enabled: trackedDeploymentId !== null,
    refetchInterval: LOCAL_NIM_POLL_INTERVAL_MS,
  });
  const trackedDeployment = deploymentsQuery.data?.items.find(
    (deployment) => deployment.local_nim_deployment_id === trackedDeploymentId,
  );

  useEffect(() => {
    if (
      trackedDeployment?.status !== "running" &&
      trackedDeployment?.status !== "failed"
    ) {
      return;
    }
    void queryClient.invalidateQueries({ queryKey: projectKeys.detail(projectId) });
    void queryClient.invalidateQueries({ queryKey: modelConfigKeys.list(projectId) });
    void queryClient.invalidateQueries({ queryKey: environmentKeys.assessment() });
  }, [projectId, queryClient, trackedDeployment?.status]);

  const deployMutation = useMutation({
    mutationFn: ({
      replaceResident,
      gpuAssignment,
    }: {
      replaceResident: boolean;
      gpuAssignment?: string;
    }) => {
      if (!selected) throw new Error("Select a compatible local Teacher.");
      return deployLocalNim(projectId, {
        role: "teacher",
        model_config_id: selected.config.model_config_id,
        gpu_assignment: gpuAssignment,
        replace_resident: replaceResident,
        activate_on_success: true,
      });
    },
    onSuccess: (result) => {
      setReplacementConflict(null);
      setDeployError(null);
      if (result.disposition === "reused") {
        setTrackedDeploymentId(null);
        setReusedModelName(selected?.modelName ?? result.resident?.model_name ?? null);
        void queryClient.invalidateQueries({
          queryKey: projectKeys.detail(projectId),
        });
        void queryClient.invalidateQueries({
          queryKey: modelConfigKeys.list(projectId),
        });
        void queryClient.invalidateQueries({
          queryKey: environmentKeys.assessment(),
        });
        return;
      }
      setReusedModelName(null);
      setTrackedDeploymentId(result.deployment?.local_nim_deployment_id ?? null);
      void queryClient.invalidateQueries({
        queryKey: localNimKeys.deployments(projectId),
      });
    },
    onError: (error: unknown) => {
      const conflict = parseLocalNimGpuConflict(error);
      if (conflict) {
        setReplacementConflict(conflict);
        setDeployError(null);
        return;
      }
      setReplacementConflict(null);
      setDeployError(
        parseApiErrorDetail(error) ??
          (error instanceof Error ? error.message : "Local NIM deployment failed."),
      );
    },
  });

  function handleSelectionChange(configId: string) {
    setSelectedConfigId(configId);
    setReplacementConflict(null);
    setTrackedDeploymentId(null);
    setReusedModelName(null);
    setDeployError(null);
    deployMutation.reset();
  }

  if (compatibleModels.length === 0) {
    return (
      <div className="flex flex-col gap-3">
        <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
          No Blueprint-supported Teacher model is compatible with the detected GPU.
        </Text>
        <Button kind="secondary" onClick={onSwitchToHosted}>
          Switch to Hosted
        </Button>
      </div>
    );
  }

  const modalOpen =
    replacementConflict?.can_replace === true && replacementConflict.resident !== null;
  // The durable POST response can arrive before the first deployment-list
  // refetch contains its row. Keep the action disabled throughout that
  // reconciliation interval so rapid clicks cannot enqueue the same Teacher
  // deployment twice.
  const isStarting =
    trackedDeployment?.status === "starting" ||
    (trackedDeploymentId !== null && trackedDeployment === undefined);
  const isRunning = trackedDeployment?.status === "running";
  const isFailed = trackedDeployment?.status === "failed";
  const buttonLabel = activeAndRunning
    ? "Active Teacher"
    : exactResident
      ? "Use running NIM"
      : "Deploy selected model";

  return (
    <div
      className="glass-inner-panel flex flex-col gap-4 rounded-[14px] p-4"
      data-testid="local-teacher-deployment-panel"
    >
      <div className="flex flex-col gap-1">
        <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
          Local Teacher model
        </Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          {compatibleModels.length} Blueprint-supported model
          {compatibleModels.length === 1 ? " is" : "s are"} compatible with this
          machine. Only the selected model is deployed.
        </Text>
      </div>

      <Select
        aria-label="Local Teacher model"
        items={compatibleModels.map((model) => {
          const running = (environment.active_local_nim_residents ?? []).some(
            (resident) =>
              resident.model_name === model.modelName && resident.status === "running",
          );
          const suffix = [
            model.recommended ? "Recommended" : null,
            running ? "Running" : null,
          ]
            .filter(Boolean)
            .join(", ");
          return {
            value: model.config.model_config_id,
            children: `${localTeacherDisplayName(model.modelName)}${
              suffix ? ` — ${suffix}` : ""
            }`,
          };
        })}
        value={selectedConfigId}
        onValueChange={handleSelectionChange}
        data-testid="local-teacher-model-select"
      />

      {selected && (
        <div className="flex flex-col gap-3" data-testid="selected-local-model-details">
          <div className="flex flex-wrap items-center gap-2">
            {selected.recommended && (
              <Badge color="green" kind="outline">
                Recommended
              </Badge>
            )}
            {exactResident && (
              <Badge color="green" kind="solid">
                Running · {exactResident.gpu_assignment}
              </Badge>
            )}
            <Badge color="gray" kind="outline">
              {selected.gpuMinimumGb}+ GB
            </Badge>
            {selected.computeMinimum !== null && (
              <Badge color="gray" kind="outline">
                Compute capability {selected.computeMinimum}+
              </Badge>
            )}
          </div>
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            {modelQualityNote(selected.modelName)}
          </Text>
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            The backend runs the exact model/profile preflight before container startup.
            If a confirmed replacement fails, it requeues the displaced NIM.
          </Text>
        </div>
      )}

      {(environment.active_local_nim_residents ?? []).length > 0 && !exactResident && (
        <div className="glass-info flex items-start gap-3 rounded-lg px-3 py-3">
          <Server
            size={16}
            style={{ color: "var(--text-muted)", marginTop: 2, flexShrink: 0 }}
          />
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            A different NIM currently occupies host GPU capacity. The Blueprint will use
            a free compatible GPU when possible; if replacement is required, it will
            name the affected model and project before anything is stopped.
          </Text>
        </div>
      )}

      {replacementConflict && !modalOpen && (
        <div className="flex items-start gap-2" role="status">
          <TriangleAlert
            size={16}
            style={{ color: "var(--text-warning, #eab308)", marginTop: 2 }}
          />
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            {replacementConflict.code === "resident_starting"
              ? `${residentName(replacementConflict)} is already starting. Wait for it to become ready, then try again.`
              : replacementConflict.message}
          </Text>
        </div>
      )}

      {(isStarting || deployMutation.isPending) && (
        <div className="flex items-center gap-2" role="status">
          <Spinner aria-label="Deploying selected local Teacher" />
          <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
            {deployMutation.isPending
              ? "Checking GPU placement..."
              : `Deploying ${localTeacherDisplayName(selected?.modelName)}. It becomes the active Teacher only after verification passes.`}
          </Text>
        </div>
      )}

      {(isRunning || reusedModelName) && (
        <div className="flex items-center gap-2" role="status">
          <Check size={16} style={{ color: "var(--accent-green)" }} />
          <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
            {localTeacherDisplayName(reusedModelName ?? selected?.modelName)} is running
            and selected as this project's Teacher.
          </Text>
        </div>
      )}

      {isFailed && (
        <div className="flex flex-col gap-1" role="alert">
          <Text kind="body/regular/sm" className="text-error">
            The selected NIM failed to start.
          </Text>
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            {trackedDeployment?.status_reason ?? "Unknown deployment error"} The
            Blueprint has attempted to restore any NIM displaced by this replacement.
          </Text>
        </div>
      )}

      {deployError && (
        <Text kind="body/regular/sm" className="text-error" role="alert">
          {deployError}
        </Text>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button
          kind="primary"
          className="nvidia-green-button"
          onClick={() =>
            deployMutation.mutate({
              replaceResident: false,
            })
          }
          disabled={
            !selected ||
            activeAndRunning ||
            isStarting ||
            deployMutation.isPending ||
            !environment.ngc_api_key_configured
          }
        >
          {buttonLabel}
        </Button>
        <Button kind="secondary" onClick={onSwitchToHosted}>
          Switch to Hosted
        </Button>
      </div>

      {!environment.ngc_api_key_configured && (
        <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
          Add and save an NGC API key below before deploying a local model.
        </Text>
      )}

      <Modal
        open={modalOpen}
        onOpenChange={(open) => {
          if (!open) setReplacementConflict(null);
        }}
        slotHeading={`Replace ${replacementConflict ? residentName(replacementConflict) : "current NIM"}?`}
        slotFooter={
          <>
            <Button
              kind="secondary"
              onClick={() => setReplacementConflict(null)}
              disabled={deployMutation.isPending}
            >
              Keep current
            </Button>
            <Button
              kind="primary"
              className="nvidia-green-button"
              onClick={() =>
                deployMutation.mutate({
                  replaceResident: true,
                  gpuAssignment:
                    replacementConflict?.resident?.gpu_assignment ?? undefined,
                })
              }
              disabled={deployMutation.isPending}
            >
              {deployMutation.isPending ? "Starting replacement..." : "Replace NIM"}
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-3">
          <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
            {replacementConflict?.resident
              ? `${residentName(replacementConflict)} is running on ${replacementConflict.resident.gpu_assignment} for project “${replacementConflict.resident.project_name}”.`
              : "A Blueprint-managed NIM currently occupies the required GPU."}
          </Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Starting {localTeacherDisplayName(selected?.modelName)} will stop that NIM
            first. The new model becomes active only after verification succeeds. If
            startup fails, the Blueprint will best-effort restore the displaced NIM.
          </Text>
        </div>
      </Modal>
    </div>
  );
}

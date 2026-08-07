// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Post-onboarding local embedding-NIM lifecycle.
 *
 * The first deploy request is always non-destructive. If the backend reports
 * that a GPU resident must be replaced, this panel names the affected model
 * and project before offering the explicit stop-and-start retry. A deployment
 * owned by this project can also be stopped here so the embedding provider
 * falls back through the backend-authoritative provider cascade.
 */

import { useEffect, useState } from "react";
import { Badge, Button, Modal, Spinner, Text } from "@kui/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Server, TriangleAlert } from "lucide-react";

import { parseApiErrorDetail } from "@/api/client";
import {
  deployLocalNim,
  listLocalNimDeployments,
  parseLocalNimGpuConflict,
  stopLocalNim,
} from "@/api/nim";
import { environmentKeys, localNimKeys, projectKeys } from "@/api/query-keys";
import { LOCAL_NIM_POLL_INTERVAL_MS } from "@/lib/local-nim";
import { localTeacherDisplayName } from "@/lib/model-display";
import type {
  ActiveLocalNimResident,
  EnvironmentResponse,
  LocalNimGpuConflict,
} from "@/types/nim";

interface LocalEmbeddingDeploymentPanelProps {
  projectId: string;
  environment: EnvironmentResponse;
  onSwitchToHosted: () => void;
}

function residentName(resident: ActiveLocalNimResident | null | undefined): string {
  if (!resident) return "current NIM";
  if (resident.role === "embedding") return "NeMo Retriever VL embedding NIM";
  return localTeacherDisplayName(resident.model_name);
}

export function LocalEmbeddingDeploymentPanel({
  projectId,
  environment,
  onSwitchToHosted,
}: LocalEmbeddingDeploymentPanelProps) {
  const queryClient = useQueryClient();
  const [replacementConflict, setReplacementConflict] =
    useState<LocalNimGpuConflict | null>(null);
  const [trackedDeploymentId, setTrackedDeploymentId] = useState<string | null>(null);
  const [deployError, setDeployError] = useState<string | null>(null);
  const [stopError, setStopError] = useState<string | null>(null);

  const activeEmbeddingResident = (environment.active_local_nim_residents ?? []).find(
    (resident) =>
      resident.role === "embedding" &&
      (resident.status === "starting" || resident.status === "running"),
  );
  const ownedEmbeddingResident =
    activeEmbeddingResident?.project_id === projectId
      ? activeEmbeddingResident
      : undefined;

  const deploymentsQuery = useQuery({
    queryKey: localNimKeys.deployments(projectId),
    queryFn: () => listLocalNimDeployments(projectId),
    refetchInterval: LOCAL_NIM_POLL_INTERVAL_MS,
  });
  const trackedDeployment = deploymentsQuery.data?.items.find(
    (deployment) => deployment.local_nim_deployment_id === trackedDeploymentId,
  );
  const deploymentStatus =
    trackedDeployment?.status ?? activeEmbeddingResident?.status ?? null;
  // The POST returns a durable starting id before the first deployment-list
  // refetch necessarily contains that row. Treat the tracked-but-not-yet-seen
  // interval as starting so a rapid second click cannot submit a duplicate.
  const isStarting =
    deploymentStatus === "starting" ||
    (trackedDeploymentId !== null && trackedDeployment === undefined);
  const isRunning = deploymentStatus === "running";
  const isFailed = trackedDeployment?.status === "failed";
  // A durable starting/running deployment is stronger evidence than a stale
  // pre-request environment assessment. The replacement request has already
  // passed backend placement and stopped the named resident, so do not briefly
  // keep showing "GPU required" or "different NIM occupies" while the
  // environment query reconciles.
  const placementAccepted = isStarting || isRunning;

  useEffect(() => {
    if (
      trackedDeployment?.status !== "running" &&
      trackedDeployment?.status !== "failed"
    ) {
      return;
    }
    void queryClient.invalidateQueries({ queryKey: environmentKeys.assessment() });
    void queryClient.invalidateQueries({ queryKey: projectKeys.all });
  }, [queryClient, trackedDeployment?.status]);

  const deployMutation = useMutation({
    mutationFn: ({
      replaceResident,
      gpuAssignment,
    }: {
      replaceResident: boolean;
      gpuAssignment?: string;
    }) =>
      deployLocalNim(projectId, {
        role: "embedding",
        gpu_assignment: gpuAssignment,
        replace_resident: replaceResident,
      }),
    onSuccess: (result) => {
      setReplacementConflict(null);
      setDeployError(null);
      setTrackedDeploymentId(result.deployment?.local_nim_deployment_id ?? null);
      void queryClient.invalidateQueries({
        queryKey: localNimKeys.deployments(projectId),
      });
      void queryClient.invalidateQueries({ queryKey: environmentKeys.assessment() });
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
          (error instanceof Error ? error.message : "Embedding NIM deployment failed."),
      );
    },
  });

  const stopMutation = useMutation({
    mutationFn: (deploymentId: string) => stopLocalNim(projectId, deploymentId),
    onSuccess: () => {
      setStopError(null);
      setTrackedDeploymentId(null);
      void queryClient.invalidateQueries({
        queryKey: localNimKeys.deployments(projectId),
      });
      void queryClient.invalidateQueries({ queryKey: environmentKeys.assessment() });
      void queryClient.invalidateQueries({ queryKey: projectKeys.all });
    },
    onError: (error: unknown) => {
      setStopError(
        parseApiErrorDetail(error) ??
          (error instanceof Error ? error.message : "Embedding NIM stop failed."),
      );
    },
  });

  const modalOpen =
    replacementConflict?.can_replace === true && replacementConflict.resident !== null;
  const canStopOwned =
    isRunning && ownedEmbeddingResident?.local_nim_deployment_id !== undefined;
  const floorGb = environment.embedding_deployment.gpu_memory_minimum_gb;
  const occupiedGpuIndices = new Set(
    (environment.active_local_nim_residents ?? [])
      .map((resident) => /^device=(\d+)$/.exec(resident.gpu_assignment)?.[1])
      .filter((index): index is string => index !== undefined),
  );
  const freeCapableGpuExists = environment.gpus.some(
    (gpu, index) =>
      gpu.memory_total_gb >= floorGb && !occupiedGpuIndices.has(String(index)),
  );
  // ``embedding_deployment.fits`` is placement-aware. It can be false even
  // on capable free hardware because the recommendation reserves that GPU for
  // the local Teacher. Name that policy instead of claiming the GPU is absent.
  const gpuReservedForTeacher =
    !environment.embedding_deployment.fits &&
    !placementAccepted &&
    environment.recommended_teacher_mode === "local" &&
    freeCapableGpuExists;

  return (
    <div
      className="glass-inner-panel flex flex-col gap-4 rounded-[14px] p-4"
      data-testid="local-embedding-deployment-panel"
    >
      <div className="flex flex-col gap-1">
        <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
          NeMo Retriever VL embedding NIM
        </Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          Deploys the Blueprint-supported 2,048-dimensional image embedding model. The
          backend verifies Docker, NVIDIA runtime, GPU memory, NGC access, and image
          compatibility before startup.
        </Text>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Badge
          color={
            environment.embedding_deployment.fits || placementAccepted
              ? "green"
              : "gray"
          }
          kind="outline"
        >
          {floorGb}+ GB GPU{" "}
          {environment.embedding_deployment.fits || placementAccepted
            ? "available"
            : gpuReservedForTeacher
              ? "reserved for Teacher"
              : "required"}
        </Badge>
        <Badge color="gray" kind="outline">
          Local · no rate limits
        </Badge>
        {activeEmbeddingResident && (
          <Badge color="green" kind="solid">
            {activeEmbeddingResident.status === "running" ? "Running" : "Starting"}
            {` · ${activeEmbeddingResident.gpu_assignment}`}
          </Badge>
        )}
      </div>

      {(environment.active_local_nim_residents ?? []).length > 0 &&
        !activeEmbeddingResident &&
        !placementAccepted && (
          <div className="glass-info flex items-start gap-3 rounded-lg px-3 py-3">
            <Server
              size={16}
              style={{ color: "var(--text-muted)", marginTop: 2, flexShrink: 0 }}
            />
            <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
              A different NIM currently occupies host GPU capacity. The first deploy
              request is non-destructive; if replacement is required, the Blueprint
              names the affected model and project before anything is stopped.
            </Text>
          </div>
        )}

      {activeEmbeddingResident?.project_id !== projectId && activeEmbeddingResident && (
        <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
          The embedding NIM is owned by project “{activeEmbeddingResident.project_name}
          ”. It is shared deployment infrastructure; manage its lifecycle from that
          project's NIM Configuration screen.
        </Text>
      )}

      {replacementConflict && !modalOpen && (
        <div className="flex items-start gap-2" role="status">
          <TriangleAlert
            size={16}
            style={{ color: "var(--text-warning, #eab308)", marginTop: 2 }}
          />
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            {replacementConflict.message}
          </Text>
        </div>
      )}

      {(isStarting || deployMutation.isPending) && (
        <div className="flex items-center gap-2" role="status">
          <Spinner aria-label="Deploying local embedding NIM" />
          <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
            {deployMutation.isPending
              ? "Checking GPU placement..."
              : "Deploying the embedding NIM. Image review continues with pHash until verification passes."}
          </Text>
        </div>
      )}

      {isRunning && (
        <div className="flex items-center gap-2" role="status">
          <Check size={16} style={{ color: "var(--accent-green)" }} />
          <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
            The local embedding NIM is running. Projects switch to it only after a live
            2,048-dimensional probe succeeds.
          </Text>
        </div>
      )}

      {isFailed && (
        <div className="flex flex-col gap-1" role="alert">
          <Text kind="body/regular/sm" className="text-error">
            The embedding NIM failed to start.
          </Text>
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            {trackedDeployment?.status_reason ?? "Unknown deployment error"} The
            Blueprint attempted to restore any NIM displaced by this replacement.
          </Text>
        </div>
      )}

      {(deployError || stopError) && (
        <Text kind="body/regular/sm" className="text-error" role="alert">
          {deployError ?? stopError}
        </Text>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {!activeEmbeddingResident && (
          <Button
            kind="primary"
            className="nvidia-green-button"
            onClick={() => deployMutation.mutate({ replaceResident: false })}
            disabled={
              !environment.ngc_api_key_configured ||
              isStarting ||
              deployMutation.isPending ||
              stopMutation.isPending
            }
          >
            {isStarting ? "Embedding NIM starting" : "Deploy embedding NIM"}
          </Button>
        )}
        {canStopOwned && (
          <Button
            kind="secondary"
            onClick={() =>
              stopMutation.mutate(ownedEmbeddingResident.local_nim_deployment_id)
            }
            disabled={stopMutation.isPending}
          >
            {stopMutation.isPending ? "Stopping..." : "Stop embedding NIM"}
          </Button>
        )}
        <Button kind="secondary" onClick={onSwitchToHosted}>
          Switch to Hosted
        </Button>
      </div>

      {!environment.ngc_api_key_configured && (
        <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
          Add and save an NGC API key below before deploying locally.
        </Text>
      )}
      {gpuReservedForTeacher && (
        <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
          A compatible GPU is free but reserved for the recommended local Teacher.
          Deploying embeddings uses that GPU; on a one-GPU host, restore the Teacher
          afterward for local labeling.
        </Text>
      )}
      {!environment.embedding_deployment.fits &&
        !placementAccepted &&
        !gpuReservedForTeacher && (
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            No compatible GPU is currently free. Deploy remains non-destructive and will
            name a replaceable resident or report the hardware floor.
          </Text>
        )}

      <Modal
        open={modalOpen}
        onOpenChange={(open) => {
          if (!open) setReplacementConflict(null);
        }}
        slotHeading={`Replace ${residentName(replacementConflict?.resident)}?`}
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
              {deployMutation.isPending
                ? "Starting replacement..."
                : "Stop and deploy embeddings"}
            </Button>
          </>
        }
      >
        <div className="flex flex-col gap-3">
          <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
            {replacementConflict?.resident
              ? `${residentName(replacementConflict.resident)} is running on ${replacementConflict.resident.gpu_assignment} for project “${replacementConflict.resident.project_name}”.`
              : "A Blueprint-managed NIM currently occupies the required GPU."}
          </Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
            Starting the embedding NIM stops that resident first. On a single-GPU host,
            interactive labeling needs another Teacher endpoint until you restore the
            local Teacher. If startup fails, the Blueprint best-effort restores the
            displaced NIM.
          </Text>
        </div>
      </Modal>
    </div>
  );
}

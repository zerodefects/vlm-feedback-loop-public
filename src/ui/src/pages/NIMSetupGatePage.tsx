// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NIM setup — path-aware confirmation gate (last step of the
 * three-screen FTU setup chain).
 *
 * Three responsibilities:
 *
 *   1. **Safety-net summary.** Glass-card with Teacher + Embedding rows
 *      driven by ``activePath`` (carried forward from the earlier
 *      setup screens via ``location.state``). On the hybrid path a third
 *      row names the preferred local Teacher that will take over after
 *      backend verification.
 *
 *   2. **Background-deploy dispatcher.** When ``localDeployQueued`` is
 *      non-empty, the [Start labeling] click fires one ``POST :deploy``
 *      per queued model name. Deploys are fire-and-forget — the
 *      endpoint returns ``starting`` quickly and the ``LocalDeployBanner``
 *      in AppShell takes over status display from there. An identical
 *      host-wide Teacher resident is reused automatically. A different
 *      resident produces an explicit replace confirmation instead of a
 *      dead-end GPU-busy error.
 *
 *   3. **Authoritative ``mark_setup_completed`` call site.** Always
 *      called (auto-skip path and manual click), with
 *      ``local_deploy_queued`` flowing through into the AuditEvent.
 *
 * Auto-skip rule: ``cameFromAutoSkip === true`` — only a fully
 * unattended chain auto-skips.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import { ArrowRight, Check } from "lucide-react";

import { markSetupCompleted } from "@/api/projects";
import {
  deployLocalNim,
  listLocalNimDeployments,
  parseLocalNimGpuConflict,
} from "@/api/nim";
import type {
  ActiveLocalNimResident,
  LocalNimDeploymentListResponse,
  LocalNimDeploymentResponse,
  LocalNimGpuConflict,
} from "@/types/nim";
import { fetchModelConfigs, updateProject } from "@/api/model-configs";
import { localNimKeys, modelConfigKeys, projectKeys } from "@/api/query-keys";
import { useSetupContext } from "@/pages/ProjectSetupLayout";
import { LOCAL_NIM_POLL_INTERVAL_MS, latestPerRole } from "@/lib/local-nim";
import { formatModelDisplayName, localTeacherDisplayName } from "@/lib/model-display";
import { runStatusRefetchInterval } from "@/lib/run-status-polling";
import { DEFAULT_SETUP_CHAIN_STATE, type SetupChainState } from "@/types/setupChain";

interface ModelRow {
  eyebrow: string;
  name: string;
  meta: string;
}

function isEmbeddingModel(name: string): boolean {
  return name.includes("embed") || name.includes("nvclip");
}

function residentDisplayName(resident: ActiveLocalNimResident): string {
  if (resident.model_name) return localTeacherDisplayName(resident.model_name);
  if (resident.role === "embedding") return "the embedding NIM";
  if (resident.role === "student") return "a Student NIM";
  return "another local NIM";
}

// User-facing meta suffix for a deployment row, built from the
// authoritative ``LocalNimDeployment.status`` field. Falls back to
// "queued" when the row hasn't been created yet (between dispatch
// fire and the backend's first persist).
function statusSuffix(deployment: LocalNimDeploymentResponse | undefined): string {
  if (deployment === undefined) return "queued";
  switch (deployment.status) {
    case "running":
      return "running";
    case "failed": {
      const reason = (deployment.status_reason ?? "").split(/<html|\n/)[0].trim();
      if (reason.length === 0) return "deploy failed";
      const trimmed = reason.length > 80 ? `${reason.slice(0, 80)}…` : reason;
      return `deploy failed — ${trimmed}`;
    }
    case "stopped":
      return "stopped";
    case "starting":
    default:
      return "deploying in background";
  }
}

export function NIMSetupGatePage(): JSX.Element {
  const { projectId, environment } = useSetupContext();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const incoming = {
    ...DEFAULT_SETUP_CHAIN_STATE,
    ...((location.state as Partial<SetupChainState>) ?? {}),
  };
  const didAutoSkip = useRef(false);
  const env = environment;

  const [submitting, setSubmitting] = useState(false);
  // Dispatch-failure UX: when the Teacher deploy is
  // blocked by the one-NIM-per-GPU gate, the page renders an inline
  // error block instead of navigating to a broken labeling screen.
  // The reason string is the parsed 409 code (``gpu_occupied`` /
  // ``gpu_exhausted``) or a fallback detail. Setting this back to
  // null on retry-click clears the error.
  const [teacherBlocked, setTeacherBlocked] = useState<{
    modelName: string;
    reason: string;
    conflict: LocalNimGpuConflict | null;
  } | null>(null);

  const { activePath, cameFromAutoSkip, localDeployQueued } = incoming;

  // Fetch teacher-eligible model_configs only when we need to deploy a
  // local Teacher. The deploy endpoint resolves by ``model_config_id``,
  // not by ``model_name``, so we need to map the env recommendation
  // (a model name string) onto the project's catalog row.
  const needsTeacherConfigs = localDeployQueued.some((name) => !isEmbeddingModel(name));
  const modelConfigsQuery = useQuery({
    queryKey: modelConfigKeys.list(projectId, "teacher"),
    queryFn: () => fetchModelConfigs(projectId, "teacher"),
    enabled: needsTeacherConfigs && !cameFromAutoSkip,
  });

  // model_name → model_config_id lookup for the Teacher role. A missing
  // entry means the seed migration hasn't created a project-scoped
  // catalog row for the recommendation — callers log and skip rather
  // than crash.
  const teacherConfigByName = useMemo(() => {
    const map = new Map<string, string>();
    for (const c of modelConfigsQuery.data?.items ?? []) {
      map.set(c.model_name, c.model_config_id);
    }
    return map;
  }, [modelConfigsQuery.data]);

  // Poll the local-NIM deployments so the summary rows reflect the
  // ACTUAL ``LocalNimDeployment.status`` (not a static "deploying in
  // background" string that goes stale once the container is healthy).
  // Only roles in ``localDeployQueued`` are considered — older orphan
  // rows from a previous setup attempt are filtered out so we don't
  // surface a stale "running" status for a container that no longer
  // exists. Polling stops once every queued role has reached a
  // terminal state (running / failed / stopped).
  const queuedRoles = useMemo(() => {
    const roles = new Set<"teacher" | "embedding">();
    for (const m of localDeployQueued) {
      roles.add(isEmbeddingModel(m) ? "embedding" : "teacher");
    }
    return roles;
  }, [localDeployQueued]);

  const localDeploymentsQuery = useQuery({
    queryKey: localNimKeys.deployments(projectId),
    queryFn: () => listLocalNimDeployments(projectId),
    enabled: !cameFromAutoSkip && queuedRoles.size > 0,
    refetchInterval: runStatusRefetchInterval({
      isSettled: (data: LocalNimDeploymentListResponse | undefined) => {
        const latest = latestPerRole(
          (data?.items ?? []).filter((d) =>
            queuedRoles.has(d.role as "teacher" | "embedding"),
          ),
        );
        // Keep polling until every queued role has surfaced a row AND
        // none of them is still in flight.
        if (latest.length === 0) return false;
        return !latest.some((d) => d.status === "starting");
      },
      activeMs: LOCAL_NIM_POLL_INTERVAL_MS,
    }),
    // Stale-while-revalidate so the cache populates from sibling
    // pollers (e.g. LocalDeployBanner) without flicker.
    staleTime: LOCAL_NIM_POLL_INTERVAL_MS / 2,
  });

  const deploymentByRole = useMemo(() => {
    const items = localDeploymentsQuery.data?.items ?? [];
    const filtered = items.filter((d) =>
      queuedRoles.has(d.role as "teacher" | "embedding"),
    );
    const map = new Map<"teacher" | "embedding", LocalNimDeploymentResponse>();
    for (const d of latestPerRole(filtered)) {
      map.set(d.role as "teacher" | "embedding", d);
    }
    return map;
  }, [localDeploymentsQuery.data, queuedRoles]);

  const compatibleTeacherResident = useMemo(() => {
    const requestedModel = env.recommended_local_teacher_model_name;
    if (!requestedModel) return null;
    return (
      (env.active_local_nim_residents ?? []).find(
        (resident) =>
          resident.role === "teacher" && resident.model_name === requestedModel,
      ) ?? null
    );
  }, [env.active_local_nim_residents, env.recommended_local_teacher_model_name]);

  // ── Auto-skip (→ confirm-defaults) ───────────────────────────────────
  useEffect(() => {
    if (didAutoSkip.current) return;
    if (!cameFromAutoSkip) return;
    didAutoSkip.current = true;
    void markSetupCompleted(projectId, {
      auto_skip: true,
      teacher_mode: env.recommended_teacher_mode,
      embedding_mode: env.recommended_embedding_mode,
      embedding_provider: env.embedding_deployment.provider,
      local_deploy_queued: localDeployQueued,
    })
      .catch((err: unknown) => {
        console.warn("mark_setup_completed (auto-skip 2C) failed:", err);
      })
      .finally(() => {
        // The NIM Configuration header link gates on the cached project's
        // setup_completed_at — refetch after the stamp lands so it
        // appears immediately on the post-setup screens.
        void queryClient.invalidateQueries({
          queryKey: projectKeys.detail(projectId),
        });
        navigate("../confirm-defaults", { replace: true });
      });
  }, [
    cameFromAutoSkip,
    env.recommended_teacher_mode,
    env.recommended_embedding_mode,
    env.embedding_deployment.provider,
    localDeployQueued,
    navigate,
    projectId,
    queryClient,
  ]);

  // ── Summary rows, driven by activePath ──────────────────────────────
  // NOTE: these memos run unconditionally (rules-of-hooks) even on the
  // auto-skip render; they are pure computations, and the early return
  // below still short-circuits before any JSX is rendered.
  const teacherRow: ModelRow = useMemo(() => {
    if (activePath === "local") {
      const dep = deploymentByRole.get("teacher");
      const residentStatus =
        compatibleTeacherResident?.status === "running"
          ? `already running for ${compatibleTeacherResident.project_name}; will be reused`
          : compatibleTeacherResident?.status === "starting"
            ? `already starting for ${compatibleTeacherResident.project_name}`
            : statusSuffix(dep);
      return {
        eyebrow: "TEACHER MODEL",
        name: localTeacherDisplayName(env.recommended_local_teacher_model_name),
        meta: `Local NIM · ${env.gpus[0]?.name ?? "your GPU"} (${residentStatus})`,
      };
    }
    return {
      eyebrow: "TEACHER MODEL",
      name: formatModelDisplayName(env.default_teacher_model_name),
      meta:
        activePath === "hybrid"
          ? "Starting now · Hosted until the local Teacher is ready"
          : "Default · Hosted on build.nvidia.com",
    };
  }, [
    activePath,
    env.default_teacher_model_name,
    env.recommended_local_teacher_model_name,
    env.gpus,
    deploymentByRole,
    compatibleTeacherResident,
  ]);

  // What actually happens with image embeddings on this host:
  //  - "local"   → a queued embedding deploy fires on [Start labeling]
  //                (multi-GPU local path, or the small-GPU host class:
  //                GPU below every Teacher floor but at/above the
  //                embedding floor → hosted Teacher + local NeMo
  //                Retriever VL).
  //  - "hosted"  → routed to build.nvidia.com via the NVIDIA API key.
  //  - "none"    → no embedding NIM will run (hosted unavailable AND
  //                local is blocked by the one-NIM-per-GPU rule —
  //                single-GPU host with a Teacher queued silently
  //                skips the embedding deploy in dispatchLocalDeploys
  //                below). The /done screen MUST NOT claim something
  //                that isn't going to happen.
  const teacherIsLocallyQueued = localDeployQueued.some((m) => !isEmbeddingModel(m));
  const embeddingIsLocallyQueued = localDeployQueued.some(isEmbeddingModel);
  const singleGpuHost = env.gpus.length === 1;
  const embeddingTarget: "hosted" | "local" | "none" = useMemo(() => {
    // A queued local embedding deploy is what will ACTUALLY happen
    // ([Start labeling] dispatches it below) — it must win over the
    // hosted claim, otherwise a multi-GPU local path with an NVIDIA
    // key would promise "Hosted" while deploying a local NIM. The
    // guard mirrors the dispatcher's single-GPU NIM-on-NIM skip; the
    // backend recommendation can't apply it because it doesn't know
    // this chain queued a local Teacher.
    if (embeddingIsLocallyQueued && !(singleGpuHost && teacherIsLocallyQueued)) {
      return "local";
    }
    // Hosted path requires an NVIDIA API key. Without it,
    // build.nvidia.com is not reachable — promising "Hosted" would
    // be a lie. The on-mount probe in NIMNvidiaKeyPage already
    // validates this; we mirror the gate here so the /done summary
    // tells the truth even if the SME skipped that probe.
    if (env.nvidia_api_key_configured) return "hosted";
    // A recommendation alone deploys nothing: every flow that will
    // run local embeddings has the deploy queued by now, so anything
    // else is "none".
    return "none";
  }, [
    env.nvidia_api_key_configured,
    singleGpuHost,
    teacherIsLocallyQueued,
    embeddingIsLocallyQueued,
  ]);

  const embeddingRow: ModelRow | null = useMemo(() => {
    if (embeddingTarget === "hosted") {
      return {
        eyebrow: "EMBEDDING MODEL",
        name: "NeMo Retriever VL 1B v2",
        meta: "Hosted on build.nvidia.com",
      };
    }
    if (embeddingTarget === "local") {
      const dep = deploymentByRole.get("embedding");
      return {
        eyebrow: "EMBEDDING MODEL",
        name: "NeMo Retriever VL 1B v2",
        meta: `Local NIM · ${env.gpus[0]?.name ?? "your GPU"} (${statusSuffix(dep)})`,
      };
    }
    // "none" — no embedding NIM is going to run. Omit the row so
    // the /done summary doesn't promise something the system isn't
    // going to deliver.
    return null;
  }, [embeddingTarget, env.gpus, deploymentByRole]);

  // Hybrid path adds a third row naming the preferred local Teacher
  // that takes over automatically after the backend verifies it.
  const hybridAlternateRow: ModelRow | null = useMemo(() => {
    if (activePath !== "hybrid") return null;
    const dep = deploymentByRole.get("teacher");
    const suffix =
      compatibleTeacherResident?.status === "running"
        ? `already running for ${compatibleTeacherResident.project_name}; will be reused`
        : compatibleTeacherResident?.status === "starting"
          ? `already starting for ${compatibleTeacherResident.project_name}`
          : statusSuffix(dep);
    const activationHint =
      dep?.status === "running" || compatibleTeacherResident?.status === "running"
        ? "becomes active after verification"
        : "will become active automatically once verified";
    return {
      eyebrow: "NEXT · LOCAL TEACHER",
      name: localTeacherDisplayName(env.recommended_local_teacher_model_name),
      meta: `Local NIM · ${env.gpus[0]?.name ?? "your GPU"} (${suffix}, ${activationHint})`,
    };
  }, [
    activePath,
    env.recommended_local_teacher_model_name,
    env.gpus,
    deploymentByRole,
    compatibleTeacherResident,
  ]);

  if (cameFromAutoSkip) return <></>;

  /**
   * Result returned by ``dispatchLocalDeploys``. When the Teacher
   * deploy is blocked by the one-NIM-per-GPU router gate (HTTP 409
   * gpu_occupied / gpu_exhausted), ``teacherDeployBlocked`` is
   * non-null and ``handleStartLabeling`` MUST abort navigation —
   * silently swallowing this error strands the SME on a labeling
   * screen bound to an unreachable hosted endpoint.
   */
  interface DispatchResult {
    teacherDeployBlocked: {
      modelName: string;
      reason: string;
      conflict: LocalNimGpuConflict | null;
    } | null;
  }

  async function dispatchLocalDeploys(
    replaceResident = false,
  ): Promise<DispatchResult> {
    if (localDeployQueued.length === 0) {
      return { teacherDeployBlocked: null };
    }

    // Single-GPU NIM-on-NIM placement decision (empirical result, see
    // README.md "One-NIM-per-GPU policy"). On a
    // single-GPU host where the Teacher is queued, skip the embedding
    // deploy entirely — they cannot coexist regardless of VRAM math.
    // The Cosmos Reason2 NIM's vLLM hardcodes
    // ``gpu_memory_utilization=0.9`` and its profile selector treats
    // any neighbor process (even a 12 GB embedding at 15 % util) as
    // making the GPU "non-free" — ``Detected 0 compatible profile(s)``
    // and Cosmos refuses to start. Symmetric: Teacher-first leaves
    // ~10 GB free, embedding's Triton OOMs. Three-data-point baseline
    // in README.md.
    //
    // Behavioral consequence: single-GPU + local Teacher path means
    // image embedding falls back to pHash diversity (system handles
    // transparently). Operators who need semantic embeddings on a
    // single-GPU host should use the hybrid path (Teacher local,
    // embedding hosted via NVIDIA_API_KEY → build.nvidia.com).
    //
    // The skip requires a queued Teacher: an embedding-only queue on
    // a single GPU passes through — that's the small-GPU host class
    // (GPU below every Teacher floor but at/above the embedding
    // floor) running hosted Teacher + local embeddings.
    const teacherQueued = localDeployQueued.some((m) => !isEmbeddingModel(m));
    const singleGpu = env.gpus.length === 1;
    const skipEmbeddingSingleGpuNimOnNim = teacherQueued && singleGpu;

    let teacherDeployBlocked: DispatchResult["teacherDeployBlocked"] = null;

    for (const modelName of localDeployQueued) {
      const isEmbedding = isEmbeddingModel(modelName);
      try {
        if (isEmbedding) {
          if (skipEmbeddingSingleGpuNimOnNim) {
            console.warn(
              "Skipping local embedding NIM: single-GPU host with " +
                "Teacher NIM queued. NIM's profile selector treats any " +
                "neighbor process as making the GPU non-free, so the " +
                "two cannot coexist on one GPU regardless of VRAM math " +
                "(see README.md 'One-NIM-per-GPU policy'). " +
                "Embedding falls back to pHash diversity; add " +
                "NVIDIA_API_KEY for hosted embeddings (hybrid path).",
            );
            continue;
          }
          // Multi-GPU host: omit gpu_assignment and let the backend's
          // auto-placer pick a different device from the Teacher.
          await deployLocalNim(projectId, { role: "embedding" });
        } else {
          const configId = teacherConfigByName.get(modelName);
          if (!configId) {
            console.warn(
              `No teacher ModelConfig found for ${modelName}; skipping deploy. ` +
                "The model will not appear in the catalog until the SME deploys it via NIM Configuration.",
            );
            continue;
          }
          await deployLocalNim(projectId, {
            role: "teacher",
            model_config_id: configId,
            replace_resident: replaceResident,
            ...(activePath === "hybrid" ? { activate_on_success: true } : {}),
          });
        }
      } catch (err: unknown) {
        console.warn(`Local NIM deploy for ${modelName} failed:`, err);
        // Dispatch-failure UX: when the backend's
        // one-NIM-per-GPU gate rejects a Teacher
        // deploy with HTTP 409 (gpu_occupied or gpu_exhausted), the
        // SME cannot silently land on a labeling screen — the
        // project would end up with teacher_model_config_id bound to
        // a Cosmos catalog row pointed at the hosted endpoint, which
        // doesn't serve Cosmos to most accounts. Capture the 409 and
        // let handleStartLabeling abort navigation.
        //
        // Embedding 409s remain non-fatal (system falls back to
        // pHash diversity). Other Teacher errors (preflight,
        // network) are warn-logged and the loop continues — see the
        // partial-success note below.
        if (!isEmbedding) {
          const apiErr = err as { status?: number; message?: string };
          const detail = apiErr.message ?? "Unknown deploy error";
          if (apiErr.status === 409) {
            const conflict = parseLocalNimGpuConflict(err);
            teacherDeployBlocked = {
              modelName,
              reason:
                conflict?.code ??
                (detail.includes("gpu_occupied")
                  ? "gpu_occupied"
                  : detail.includes("gpu_exhausted")
                    ? "gpu_exhausted"
                    : detail),
              conflict,
            };
            // Stop the loop on a 409 — there's no point dispatching
            // the embedding when the Teacher is blocked.
            break;
          }
        }
        // Non-Teacher or non-409 failures continue — partial success
        // is acceptable, and LocalDeployBanner picks the failures up
        // from the DB rows.
      }
    }

    return { teacherDeployBlocked };
  }

  async function handleStartLabeling(replaceResident = false) {
    setSubmitting(true);
    setTeacherBlocked(null);
    // Gate the unconditional ``finally`` navigation so the
    // one-NIM-per-GPU abort path doesn't navigate away from the inline
    // error block before the SME can read it.
    let blockedByGpuGate = false;
    try {
      // Fire the deploys first so the LocalDeployBanner can light up
      // immediately on the labeling screen. Each call returns quickly
      // (the actual container start is async on the backend).
      const dispatchResult = await dispatchLocalDeploys(replaceResident);

      // Dispatch-failure UX:
      // when the Teacher deploy was rejected by the one-NIM-per-GPU
      // gate, abort navigation + surface the error inline. Do NOT
      // call updateProject() to bind the local Teacher (it would
      // strand the SME at a hosted endpoint that doesn't serve
      // Cosmos). Do NOT call markSetupCompleted() — leaving
      // setup_completed_at null lets the SME return to the setup-choice
      // screen and pick a different path.
      if (dispatchResult.teacherDeployBlocked !== null) {
        setTeacherBlocked(dispatchResult.teacherDeployBlocked);
        setSubmitting(false);
        blockedByGpuGate = true;
        return;
      }

      // For activePath="local" the SME has explicitly chosen the local
      // Teacher (Cosmos) — make the project record match the choice
      // this gate just told them about. Without this, the project keeps
      // its seeded hosted Teacher and the labeling screen will try to
      // call it (which is unreachable when there's no NVIDIA key) on the
      // first proposal. For "hybrid" we keep the hosted default primary
      // during startup; the deploy request carries activate_on_success so
      // the backend switches atomically only after the local NIM is verified.
      if (activePath === "local") {
        const localTeacherName = localDeployQueued.find(
          (name) => !isEmbeddingModel(name),
        );
        const localTeacherConfigId = localTeacherName
          ? teacherConfigByName.get(localTeacherName)
          : undefined;
        if (localTeacherConfigId) {
          try {
            await updateProject(projectId, {
              teacher_model_config_id: localTeacherConfigId,
            });
            // Invalidate the project query so ConfirmDefaultsPage reads
            // the new teacher_model_config_id when it mounts — without
            // this it would read a stale cached value and its auto-skip
            // would still compare against the stale hosted default.
            await queryClient.invalidateQueries({
              queryKey: projectKeys.detail(projectId),
            });
          } catch (err: unknown) {
            console.warn(
              "Failed to set project teacher to local Cosmos; project will keep seeded teacher:",
              err,
            );
          }
        }
      }
      await markSetupCompleted(projectId, {
        auto_skip: false,
        teacher_mode: activePath === "local" ? "local" : env.recommended_teacher_mode,
        embedding_mode: env.recommended_embedding_mode,
        embedding_provider: env.embedding_deployment.provider,
        local_deploy_queued: localDeployQueued,
      });
      // The NIM Configuration header link gates on the cached project's
      // setup_completed_at — refetch after the stamp lands so it
      // appears immediately on the post-setup screens.
      void queryClient.invalidateQueries({
        queryKey: projectKeys.detail(projectId),
      });
    } catch (err: unknown) {
      console.warn("mark_setup_completed (manual 2C) failed:", err);
    } finally {
      if (!blockedByGpuGate) {
        navigate("../confirm-defaults");
      }
    }
  }

  // Tagline that's honest about what's happening. The generic copy
  // ("We picked the recommended models for your project") only fits
  // the hosted-only path; on the local / hybrid path deploys are
  // about to fire in the background.
  const subtitle =
    activePath === "hybrid" && localDeployQueued.some((name) => !isEmbeddingModel(name))
      ? "Start labeling immediately with the hosted Teacher. The faster local Teacher downloads in the background and becomes active automatically once verified."
      : compatibleTeacherResident?.status === "running" &&
          localDeployQueued.some((name) => !isEmbeddingModel(name))
        ? "The selected Teacher NIM is already running on this machine. This project will reuse it; no second container or model reload is needed."
        : localDeployQueued.length > 0
          ? "Local NIMs deploy in the background after you continue. You can start labeling while they come up; change anything later from the project menu."
          : "We picked the recommended models for your project. You can change either of these later from the project menu.";

  const blockedResident = teacherBlocked?.conflict?.resident ?? null;
  const blockedCanReplace = teacherBlocked?.conflict?.can_replace === true;
  const blockedResidentStarting =
    teacherBlocked?.conflict?.code === "resident_starting" &&
    teacherBlocked.conflict.matches_requested_model;

  return (
    <div className="flex flex-1 flex-col items-center justify-center p-6">
      <div className="glass-card--elevated flex w-full max-w-[640px] flex-col gap-6 p-8">
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <span
              className="flex h-7 w-7 items-center justify-center rounded-full"
              style={{
                background: "rgba(118, 185, 0, 0.18)",
                border: "1px solid rgba(118, 185, 0, 0.55)",
                color: "var(--accent-green)",
              }}
              aria-hidden="true"
            >
              <Check size={16} strokeWidth={2.5} />
            </span>
            <Text kind="title/xl" style={{ color: "var(--text-primary)" }}>
              You're set up
            </Text>
          </div>
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            {subtitle}
          </Text>
        </div>

        <div
          aria-hidden="true"
          style={{ height: 1, background: "rgba(255, 255, 255, 0.08)" }}
        />

        <div className="flex flex-col gap-5">
          <ModelSummary row={teacherRow} />
          {embeddingRow !== null && <ModelSummary row={embeddingRow} />}
          {hybridAlternateRow !== null && <ModelSummary row={hybridAlternateRow} />}
        </div>

        {teacherBlocked !== null && (
          <div
            data-testid="teacher-deploy-blocked"
            role="alert"
            className="flex flex-col gap-2 rounded-lg p-4"
            style={{
              background: "rgba(239, 68, 68, 0.10)",
              border: "1px solid rgba(239, 68, 68, 0.32)",
            }}
          >
            <Text
              kind="label/bold/sm"
              style={{ color: "var(--warning-amber, #fbbf24)" }}
            >
              {blockedResidentStarting
                ? `${localTeacherDisplayName(teacherBlocked.modelName)} is already starting`
                : blockedCanReplace
                  ? `Replace the NIM on ${blockedResident?.gpu_assignment ?? "the GPU"}?`
                  : `Can't start ${localTeacherDisplayName(teacherBlocked.modelName)} — your GPU is busy`}
            </Text>
            <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
              {blockedResidentStarting && blockedResident !== null
                ? `${residentDisplayName(blockedResident)} is already being started for project “${blockedResident.project_name}”. Wait for it to become ready, then check again; the Blueprint will reuse it automatically.`
                : blockedCanReplace && blockedResident !== null
                  ? `${residentDisplayName(blockedResident)} is running for project “${blockedResident.project_name}”. Starting ${localTeacherDisplayName(teacherBlocked.modelName)} will stop that NIM first, so the other project cannot use its local model until it is started again.`
                  : "A local NIM is using every compatible GPU, but the Blueprint could not identify a safe replacement from this screen. Keep the current NIM or return to Projects to manage it."}
            </Text>
            <div className="flex flex-wrap justify-end gap-2">
              {/* The project list lives at "/" — there is no /projects route. */}
              <Button kind="tertiary" onClick={() => navigate("/")}>
                Keep current NIM
              </Button>
              {blockedResidentStarting && (
                <Button
                  kind="secondary"
                  disabled={submitting}
                  onClick={() => {
                    void handleStartLabeling();
                  }}
                >
                  Check again
                </Button>
              )}
              {blockedCanReplace && (
                <Button
                  kind="secondary"
                  disabled={submitting}
                  onClick={() => {
                    void handleStartLabeling(true);
                  }}
                >
                  Stop current &amp; start new NIM
                </Button>
              )}
            </div>
          </div>
        )}

        <div
          aria-hidden="true"
          style={{ height: 1, background: "rgba(255, 255, 255, 0.08)" }}
        />
        <div className="flex items-center justify-end">
          <Button
            kind="primary"
            className="nvidia-green-button"
            disabled={submitting || teacherBlocked !== null}
            onClick={() => {
              void handleStartLabeling();
            }}
          >
            {submitting ? (
              <>
                <Spinner aria-label="Starting" /> Starting...
              </>
            ) : (
              <>
                Start labeling <ArrowRight size={14} />
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

function ModelSummary({ row }: { row: ModelRow }): JSX.Element {
  return (
    <div className="flex flex-col gap-1">
      <Text
        kind="label/bold/xs"
        style={{
          color: "var(--text-muted)",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
        }}
      >
        {row.eyebrow}
      </Text>
      <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
        {row.name}
      </Text>
      <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
        {row.meta}
      </Text>
    </div>
  );
}

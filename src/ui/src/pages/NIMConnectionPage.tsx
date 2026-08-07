// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NIM Configuration screen — the post-onboarding edit surface reached
 * via the NIM Configuration header link (``/projects/:id/settings/nim``).
 * The FTUE
 * never shows it; onboarding runs through the three-screen setup chain
 * (NIMNvidiaKeyPage → NIMNgcKeyPage → NIMSetupGatePage).
 *
 * 6 states: recommendation, hosted override, self-hosted override,
 * local deploy override, missing prerequisites, Action Request
 * expanded.
 */

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Select, Spinner, Text } from "@kui/react";
import { ArrowLeft, ArrowRight, Check, Copy, Terminal, X } from "lucide-react";

import { useQuery, useQueryClient } from "@tanstack/react-query";

import { markSetupCompleted } from "@/api/projects";
import {
  configureSelfHostedEmbedding,
  configureSelfHostedTeacher,
  testNgcCredential,
  testNvidiaCredential,
} from "@/api/nim";
import { setSecret, type SecretName } from "@/api/secrets";
import { parseApiErrorDetail } from "@/api/client";
import { fetchModelConfigs } from "@/api/model-configs";
import { environmentKeys, modelConfigKeys, projectKeys } from "@/api/query-keys";
import { useEnvironmentSetupContext } from "@/pages/setup-context";
import { ServiceRow } from "@/components/ServiceRow";
import { SegmentedControl } from "@/components/SegmentedControl";
import { ConnectionTestPanel } from "@/components/ConnectionTestPanel";
import { ActionRequestPanel } from "@/components/ActionRequestPanel";
import { LocalEmbeddingDeploymentPanel } from "@/components/LocalEmbeddingDeploymentPanel";
import { CredentialInput } from "@/components/CredentialInput";
import { ApplyControls, type ApplyControlsValue } from "@/components/ApplyControls";
import { LocalTeacherDeploymentPanel } from "@/components/LocalTeacherDeploymentPanel";
import { useCopyToClipboard } from "@/hooks/useCopyToClipboard";
import { usePersistedKeyProbe } from "@/hooks/usePersistedKeyProbe";
import { formatDetectedLine } from "@/lib/environment";
import { localTeacherDisplayName } from "@/lib/model-display";
import type {
  ConnectionTestResponse,
  EnvironmentResponse,
  MissingPrerequisite,
} from "@/types/nim";

// ── Helpers ─────────────────────────────────────────────────────────────────

const HOSTED_TEACHER_ROSTER = "Mistral Medium, Step, and Nemotron";

function getServiceDescription(
  mode: string,
  env: EnvironmentResponse,
  role: "teacher" | "embeddings",
): string {
  // SME-friendly value props: tell the SME *what they
  // get*, not just the technical mode name.
  if (mode === "hosted") {
    if (role === "teacher") {
      return `${HOSTED_TEACHER_ROSTER} — top quality, no setup beyond a free API key`;
    }
    return "build.nvidia.com hosted embeddings — works on any machine";
  }
  if (mode === "local") {
    if (role === "embeddings") {
      return "NeMo Retriever VL on your GPU — local, no rate limits · one NIM per GPU";
    }
    const recommendedName = env.recommended_local_teacher_model_name;
    const gpu = env.gpus[0];
    if (recommendedName && gpu) {
      return `${localTeacherDisplayName(recommendedName)} · ${gpu.name}`;
    }
    if (recommendedName) return localTeacherDisplayName(recommendedName);
    return gpu ? `On this machine · ${gpu.name}` : "On this machine";
  }
  if (role === "embeddings") {
    return "No embedding endpoint — image review remains diverse through pHash";
  }
  // Defensive fallback for an unavailable Teacher.
  return "No endpoint available";
}

function getMinLocalTeacherGpuGb(
  models: EnvironmentResponse["local_deployable_models"],
): number | null {
  const values = models.map((m) => m.gpu_memory_minimum_gb).filter((n) => n > 0);
  if (values.length === 0) return null;
  return Math.min(...values);
}

// ── Component ───────────────────────────────────────────────────────────────

type OverrideMode = "hosted" | "self_hosted" | "local";

interface PendingSelfHostedTeacher {
  baseUrl: string;
  matchingModelConfigIds: string[];
  selectedModelConfigId: string;
}

export function NIMConnectionPage() {
  const { projectId, project, environment } = useEnvironmentSetupContext();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Project-scoped catalog lookup supplies the exact ModelConfig identities
  // for the local Teacher chooser. The environment API is deployment-scoped
  // and intentionally carries no project config ids.
  const modelConfigsQuery = useQuery({
    queryKey: modelConfigKeys.list(projectId),
    queryFn: () => fetchModelConfigs(projectId),
  });
  const modelConfigs = modelConfigsQuery.data?.items ?? [];

  const [expandedService, setExpandedService] = useState<
    "teacher" | "embeddings" | null
  >(null);
  const [teacherOverride, setTeacherOverride] = useState<OverrideMode | null>(null);
  const [embeddingsOverride, setEmbeddingsOverride] = useState<OverrideMode | null>(
    null,
  );
  const [showActionRequest, setShowActionRequest] = useState(false);
  const [pendingSelfHostedTeacher, setPendingSelfHostedTeacher] =
    useState<PendingSelfHostedTeacher | null>(null);
  const [pendingSelfHostedEmbeddingUrl, setPendingSelfHostedEmbeddingUrl] = useState<
    string | null
  >(null);
  const [selfHostedTeacherError, setSelfHostedTeacherError] = useState<string | null>(
    null,
  );
  const [savePending, setSavePending] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  // Missing-prereqs → recommendation transition: SME can opt to bypass
  // the missing-prerequisites landing and go straight to hosted.
  const [bypassMissingPrereqs, setBypassMissingPrereqs] = useState(false);

  // ── Credential test/apply state ─────────────────────────────────────
  // ``pendingCredentials``: a KEYED map of just-tested credential values
  // held in memory only — each handed to ``POST /v1/secrets:set`` on
  // Save. Keyed (not a single slot) because the NVIDIA input and the
  // optional NGC input can render together; a single slot let the
  // second-tested key overwrite the first, silently dropping one on Save.
  //
  // ``applyValue``: ApplyControls state (persist checkbox).
  const [pendingCredentials, setPendingCredentials] = useState<
    Partial<Record<SecretName, string>>
  >({});

  function stagePendingCredential(name: SecretName, value: string): void {
    setPendingCredentials((prev) => ({ ...prev, [name]: value }));
  }

  function clearPendingCredential(name: SecretName): void {
    setPendingCredentials((prev) => {
      const { [name]: _omit, ...rest } = prev;
      return rest;
    });
  }

  // Shared ConnectionTestPanel / CredentialInput wiring for the NVIDIA
  // key — identical across the Teacher hosted override, the Embeddings
  // hosted override, and the inline CredentialInput, so the panels
  // differ only in ``mode``. A green test stages the just-tested value
  // for [Save]; editing the field clears the stale stage.
  function handleNvidiaTestSuccess(_result: unknown, credential: string): void {
    if (credential) {
      stagePendingCredential("NVIDIA_API_KEY", credential);
    }
  }

  function handleNvidiaCredentialChange(): void {
    clearPendingCredential("NVIDIA_API_KEY");
  }

  function handleSelfHostedTeacherTestSuccess(
    result: ConnectionTestResponse,
    _credential: string,
    baseUrl?: string,
  ): void {
    const servedModels = new Set(result.models ?? []);
    const matchingConfigs = modelConfigs.filter(
      (config) =>
        config.eligible_roles.includes("teacher") &&
        config.supports_image_input &&
        servedModels.has(config.model_name),
    );
    if (!baseUrl || matchingConfigs.length === 0) {
      setPendingSelfHostedTeacher(null);
      setSelfHostedTeacherError(
        "Connected, but the endpoint did not report a vision Teacher in this project's model catalog.",
      );
      return;
    }
    const selected =
      matchingConfigs.find(
        (config) => config.model_config_id === project.teacher_model_config_id,
      ) ?? matchingConfigs[0];
    setSelfHostedTeacherError(null);
    setPendingSelfHostedTeacher({
      baseUrl,
      matchingModelConfigIds: matchingConfigs.map((config) => config.model_config_id),
      selectedModelConfigId: selected.model_config_id,
    });
  }

  function clearSelfHostedTeacherTest(): void {
    setPendingSelfHostedTeacher(null);
    setSelfHostedTeacherError(null);
  }

  function handleSelfHostedEmbeddingTestSuccess(
    _result: ConnectionTestResponse,
    _credential: string,
    baseUrl?: string,
  ): void {
    setPendingSelfHostedEmbeddingUrl(baseUrl ?? null);
  }

  function clearSelfHostedEmbeddingTest(): void {
    setPendingSelfHostedEmbeddingUrl(null);
  }
  const [applyValue, setApplyValue] = useState<ApplyControlsValue>({
    persist: false,
  });
  // Proactive validation of currently-effective keys when the page
  // mounts (parallel of the FTU setup-choice path). If the persisted key is
  // bad, surface it via a red banner at the top of the card so the SME
  // sees the diagnosis without first clicking through the override UI
  // to discover it only after a local deployment request reaches its
  // container-pull check.
  const [persistedNgcRejected, setPersistedNgcRejected] = useState<string | null>(null);
  const [persistedNvidiaRejected, setPersistedNvidiaRejected] = useState<string | null>(
    null,
  );

  const env = environment;

  // Effective mode per service = user override (if expanded) else recommended.
  // Derive this before the persisted-key probes: merely having an unused key
  // configured must not cause an outbound credential check. Besides avoiding
  // needless network traffic, that keeps a deliberately local-only workflow
  // local when an old hosted key remains in the canonical environment.
  const effectiveTeacherMode: string =
    expandedService === "teacher" && teacherOverride
      ? teacherOverride
      : env.recommended_teacher_mode;
  const effectiveEmbeddingsMode: string =
    expandedService === "embeddings" && embeddingsOverride
      ? embeddingsOverride
      : env.recommended_embedding_mode;
  const usesHostedService =
    effectiveTeacherMode === "hosted" || effectiveEmbeddingsMode === "hosted";
  const usesSystemManagedLocalService =
    env.local_deploy_available &&
    (effectiveTeacherMode === "local" || effectiveEmbeddingsMode === "local");

  // ── Proactive persisted-key probes ──────────────────────────────────
  // Same hook as NIMNvidiaKeyPage's on-mount probes, narrowed to credentials
  // required by the currently effective configuration. Surfaces a relevant
  // persisted-but-bad key before deployment without contacting an unused
  // hosted or registry service.
  usePersistedKeyProbe({
    enabled: usesSystemManagedLocalService,
    configured: env.ngc_api_key_configured,
    probe: (signal) => testNgcCredential(undefined, signal),
    fallbackError: "NGC key validation failed.",
    onRejected: setPersistedNgcRejected,
  });

  usePersistedKeyProbe({
    enabled: usesHostedService,
    configured: env.nvidia_api_key_configured,
    probe: (signal) => testNvidiaCredential(undefined, signal),
    fallbackError: "NVIDIA key validation failed.",
    onRejected: setPersistedNvidiaRejected,
  });

  // ── Missing prerequisites landing ────────────────────────────────────
  const wantsLocal =
    env.recommended_teacher_mode === "local" ||
    env.recommended_embedding_mode === "local";
  const showMissingPrereqs =
    !bypassMissingPrereqs && wantsLocal && env.missing_prerequisites.length > 0;

  if (showMissingPrereqs) {
    return (
      <MissingPrerequisitesView
        env={env}
        onSwitchToHosted={() => setBypassMissingPrereqs(true)}
        onBack={() => navigate("/")}
      />
    );
  }

  // ── Handlers ─────────────────────────────────────────────────────────

  function handleTeacherConfigure() {
    if (expandedService === "teacher") {
      setExpandedService(null);
      setTeacherOverride(null);
      clearSelfHostedTeacherTest();
    } else {
      setExpandedService("teacher");
      setTeacherOverride(
        env.recommended_teacher_mode === "none"
          ? "hosted"
          : (env.recommended_teacher_mode as OverrideMode),
      );
    }
  }

  function handleTeacherModeChange(mode: OverrideMode): void {
    setTeacherOverride(mode);
    if (mode !== "self_hosted") clearSelfHostedTeacherTest();
  }

  function handleEmbeddingsConfigure() {
    if (expandedService === "embeddings") {
      setExpandedService(null);
      setEmbeddingsOverride(null);
      clearSelfHostedEmbeddingTest();
    } else {
      setExpandedService("embeddings");
      setEmbeddingsOverride(
        env.recommended_embedding_mode === "none"
          ? "hosted"
          : (env.recommended_embedding_mode as OverrideMode),
      );
    }
  }

  function handleEmbeddingsModeChange(mode: OverrideMode): void {
    setEmbeddingsOverride(mode);
    if (mode !== "self_hosted") clearSelfHostedEmbeddingTest();
  }

  // ── Derived helper lines (recommendation state) ──────────────────────

  const detectedLine = formatDetectedLine(env.gpus, env.docker_available);
  const minLocalTeacherGpuGb = getMinLocalTeacherGpuGb(env.local_deployable_models);
  const anyTeacherModelFits = env.local_deployable_models.some((m) => m.fits);
  const showGpuInsufficientNote =
    env.recommended_teacher_mode === "hosted" &&
    env.gpus.length > 0 &&
    env.local_deploy_available &&
    !anyTeacherModelFits &&
    minLocalTeacherGpuGb !== null;

  // An expanded override that itself renders the credential input suppresses
  // the inline prompt for that credential (the API Key lives inside the
  // hosted override when expanded). Prevents duplicate fields on screen.
  const hostedOverrideRendersApiKey =
    !env.nvidia_api_key_configured &&
    ((expandedService === "teacher" && teacherOverride === "hosted") ||
      (expandedService === "embeddings" && embeddingsOverride === "hosted"));

  const showNvidiaApiKeyInput =
    !env.nvidia_api_key_configured &&
    (effectiveTeacherMode === "hosted" || effectiveEmbeddingsMode === "hosted") &&
    !hostedOverrideRendersApiKey;
  const showNgcApiKeyInput =
    !env.ngc_api_key_configured &&
    (effectiveTeacherMode === "local" || effectiveEmbeddingsMode === "local") &&
    env.local_deploy_available;

  // Confirmation lines above the service rows.
  const showNvidiaApiKeyConfirmed =
    env.nvidia_api_key_configured &&
    (effectiveTeacherMode === "hosted" || effectiveEmbeddingsMode === "hosted");
  const showNgcApiKeyConfirmed =
    env.ngc_api_key_configured &&
    env.local_deploy_available &&
    (effectiveTeacherMode === "local" || effectiveEmbeddingsMode === "local");

  // ── Render ───────────────────────────────────────────────────────────

  return (
    <div className="flex flex-1 flex-col gap-6 p-6">
      <div className="mx-auto w-full max-w-[1600px] flex flex-col gap-6">
        {/* Header */}
        <div>
          <Text
            kind="title/xl"
            role="heading"
            aria-level={1}
            style={{ color: "var(--text-primary)" }}
          >
            NIM Configuration
          </Text>
          <Text
            kind="body/regular/sm"
            style={{
              color: "var(--text-muted)",
              display: "block",
              marginTop: 4,
            }}
          >
            Update how Teacher and embedding models connect.
          </Text>
        </div>

        {/* Persisted-key rejection banner. Fires when the on-mount
            proactive probe of an env-resident key fails. Shown above
            everything else so the SME sees the issue before navigating
            the override UI. The CredentialInput Test button + Save gate
            are the fix path. */}
        {(persistedNvidiaRejected || persistedNgcRejected) && (
          <div
            className="flex flex-col gap-1 rounded-lg px-4 py-3"
            style={{
              background: "rgba(239, 68, 68, 0.10)",
              border: "1px solid rgba(239, 68, 68, 0.30)",
            }}
            role="alert"
            data-testid="edit-mode-persisted-key-rejected-banner"
          >
            {persistedNvidiaRejected && (
              <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
                <strong>Saved NVIDIA API key was rejected.</strong>{" "}
                {persistedNvidiaRejected}
              </Text>
            )}
            {persistedNgcRejected && (
              <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
                <strong>Saved NGC API key was rejected.</strong> {persistedNgcRejected}
              </Text>
            )}
          </div>
        )}

        {/* Detected hardware — eyebrow-styled (.glass-caption). The
            leading ✓ glyph frames the line as "good news / your machine
            is ready" rather than a diagnostic readout — an all-caps
            tracked treatment reads too clinical for the SME-friendly
            intent of the recommendation state. */}
        {detectedLine && (
          <Text kind="label/semibold/xs" className="glass-caption">
            <Check
              size={12}
              style={{
                color: "var(--accent-green)",
                marginRight: 6,
                display: "inline-block",
                verticalAlign: "middle",
              }}
            />
            Detected · {detectedLine}
          </Text>
        )}

        {/* Service rows */}
        <div className="flex flex-col gap-4">
          {/* Teacher */}
          <ServiceRow
            serviceName="Teacher"
            recommendedMode={env.recommended_teacher_mode}
            description={getServiceDescription(
              env.recommended_teacher_mode,
              env,
              "teacher",
            )}
            onConfigureClick={handleTeacherConfigure}
            isExpanded={expandedService === "teacher"}
          >
            {teacherOverride && (
              <div className="flex flex-col gap-4">
                <ModeToggle
                  value={teacherOverride}
                  onChange={handleTeacherModeChange}
                  localAvailable={env.local_deploy_available}
                  idPrefix="teacher"
                />

                {teacherOverride === "hosted" && (
                  <ConnectionTestPanel
                    mode="hosted"
                    apiKeyConfigured={env.nvidia_api_key_configured}
                    onTestSuccess={handleNvidiaTestSuccess}
                    onCredentialChange={handleNvidiaCredentialChange}
                  />
                )}
                {teacherOverride === "self_hosted" && (
                  <>
                    <ConnectionTestPanel
                      mode="self_hosted"
                      apiKeyConfigured={false}
                      onTestSuccess={handleSelfHostedTeacherTestSuccess}
                      onCredentialChange={clearSelfHostedTeacherTest}
                    />
                    {selfHostedTeacherError && (
                      <Text kind="body/regular/sm" role="alert" className="text-error">
                        {selfHostedTeacherError}
                      </Text>
                    )}
                    {pendingSelfHostedTeacher && (
                      <div className="flex max-w-4xl flex-col gap-2">
                        <Text
                          kind="label/bold/sm"
                          style={{ color: "var(--text-primary)" }}
                        >
                          Teacher served by this endpoint
                        </Text>
                        <Select
                          aria-label="Self-hosted Teacher model"
                          items={pendingSelfHostedTeacher.matchingModelConfigIds.map(
                            (modelConfigId) => {
                              const config = modelConfigs.find(
                                (item) => item.model_config_id === modelConfigId,
                              );
                              return {
                                value: modelConfigId,
                                children: config?.model_name ?? modelConfigId,
                              };
                            },
                          )}
                          value={pendingSelfHostedTeacher.selectedModelConfigId}
                          onValueChange={(selectedModelConfigId) =>
                            setPendingSelfHostedTeacher((current) =>
                              current ? { ...current, selectedModelConfigId } : current,
                            )
                          }
                        />
                        <Text
                          kind="body/regular/sm"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Save verifies this exact model again, binds the endpoint, and
                          refreshes its capability controls.
                        </Text>
                      </div>
                    )}
                    <div className="glass-info flex max-w-4xl flex-wrap items-center justify-between gap-4 px-4 py-3">
                      <div className="flex flex-col gap-1">
                        <Text
                          kind="label/bold/sm"
                          style={{ color: "var(--text-primary)" }}
                        >
                          Need a self-hosted endpoint?
                        </Text>
                        <Text
                          kind="body/regular/sm"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Generate a copy-ready request for your IT or AI platform team.
                        </Text>
                      </div>
                      <Button
                        kind="secondary"
                        onClick={() => setShowActionRequest(true)}
                      >
                        Request NIM Setup
                      </Button>
                    </div>
                    {showActionRequest && (
                      <ActionRequestPanel
                        projectId={projectId}
                        requestType="nim_setup"
                        onClose={() => setShowActionRequest(false)}
                      />
                    )}
                  </>
                )}
                {teacherOverride === "local" && (
                  <LocalTeacherDeploymentPanel
                    projectId={projectId}
                    activeTeacherConfigId={project.teacher_model_config_id}
                    environment={env}
                    modelConfigs={modelConfigs}
                    onSwitchToHosted={() => setTeacherOverride("hosted")}
                  />
                )}
              </div>
            )}
          </ServiceRow>

          {/* Embeddings */}
          <ServiceRow
            serviceName="Embeddings"
            recommendedMode={env.recommended_embedding_mode}
            description={getServiceDescription(
              env.recommended_embedding_mode,
              env,
              "embeddings",
            )}
            onConfigureClick={handleEmbeddingsConfigure}
            isExpanded={expandedService === "embeddings"}
          >
            {embeddingsOverride && (
              <div className="flex flex-col gap-4">
                <ModeToggle
                  value={embeddingsOverride}
                  onChange={handleEmbeddingsModeChange}
                  localAvailable={env.local_deploy_available}
                  idPrefix="embeddings"
                />
                {embeddingsOverride === "hosted" && (
                  <ConnectionTestPanel
                    mode="hosted"
                    apiKeyConfigured={env.nvidia_api_key_configured}
                    onTestSuccess={handleNvidiaTestSuccess}
                    onCredentialChange={handleNvidiaCredentialChange}
                  />
                )}
                {embeddingsOverride === "self_hosted" && (
                  <>
                    <ConnectionTestPanel
                      mode="self_hosted"
                      apiKeyConfigured={false}
                      probeKind="embeddings"
                      onTestSuccess={handleSelfHostedEmbeddingTestSuccess}
                      onCredentialChange={clearSelfHostedEmbeddingTest}
                    />
                    {pendingSelfHostedEmbeddingUrl && (
                      <Text
                        kind="body/regular/sm"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Save obtains another finite 2,048-dimensional embedding before
                        switching all projects to this endpoint.
                      </Text>
                    )}
                  </>
                )}
                {embeddingsOverride === "local" && (
                  <LocalEmbeddingDeploymentPanel
                    projectId={projectId}
                    environment={env}
                    onSwitchToHosted={() => setEmbeddingsOverride("hosted")}
                  />
                )}
              </div>
            )}
          </ServiceRow>
        </div>

        {/* Inline credential prompts + confirmations */}
        {(showNvidiaApiKeyConfirmed ||
          showNgcApiKeyConfirmed ||
          showNvidiaApiKeyInput ||
          showNgcApiKeyInput ||
          showGpuInsufficientNote) && (
          <div className="glass-inner-panel flex flex-col gap-4 rounded-[14px] px-4 py-4">
            {showNvidiaApiKeyConfirmed && (
              <span
                className="flex items-center gap-2"
                style={{ color: "var(--accent-green)" }}
              >
                <Check size={14} />
                <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                  NVIDIA API key configured
                </Text>
              </span>
            )}
            {/* The GPU-insufficient note pairs visually with the green-check
                confirmation above ("here's why we recommend hybrid"), so it
                renders BEFORE any input fields — not
                at the bottom of the panel below the NGC input. */}
            {showGpuInsufficientNote && minLocalTeacherGpuGb !== null && (
              <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                Note: GPU insufficient for local Teacher (need at least{" "}
                {minLocalTeacherGpuGb} GB). Teacher will run via hosted NIM.
              </Text>
            )}
            {showNgcApiKeyConfirmed && (
              <span
                className="flex items-center gap-2"
                style={{ color: "var(--accent-green)" }}
              >
                <Check size={14} />
                <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                  NGC API key configured
                </Text>
              </span>
            )}
            {showNvidiaApiKeyInput && (
              <div className="flex flex-col gap-2">
                {/* Value-prop block — explains WHY the SME
                    wants this key before showing the field. Frames the
                    friction (one key entry) in terms of the value
                    (instant access to NVIDIA's top hosted models). */}
                <div
                  className="glass-callout flex flex-col gap-1 rounded-md px-3 py-2"
                  style={{
                    borderLeft: "3px solid var(--accent-green)",
                  }}
                >
                  <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
                    Why a free NVIDIA API key?
                  </Text>
                  <Text
                    kind="body/regular/sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Hosted models on build.nvidia.com give you instant access to{" "}
                    {HOSTED_TEACHER_ROSTER}. No GPU needed for the Teacher itself. Takes
                    30 seconds to grab a key.
                  </Text>
                </div>
                <CredentialInput
                  kind="nvidia_api_key"
                  onTestSuccess={handleNvidiaTestSuccess}
                  onCredentialChange={handleNvidiaCredentialChange}
                />
              </div>
            )}
            {showNgcApiKeyInput && (
              <div className="flex flex-col gap-2">
                {/* NGC value prop — recommends but does not require:
                    without NGC the project falls back to hosted
                    embeddings (or pHash). */}
                <div
                  className="glass-callout flex flex-col gap-1 rounded-md px-3 py-2"
                  style={{
                    borderLeft: "3px solid var(--text-muted)",
                  }}
                >
                  <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
                    Optional — free NGC API key for local embeddings
                  </Text>
                  <Text
                    kind="body/regular/sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    With your GPU, we can run NeMo Retriever VL locally — faster, no
                    rate limits, doesn't compete with the Teacher for hosted quota. Skip
                    this if you'd rather get started quickly; you can add it later from
                    the project's NIM Configuration screen.
                  </Text>
                </div>
                <CredentialInput
                  kind="ngc_api_key"
                  onTestSuccess={(_result, credential) => {
                    stagePendingCredential("NGC_API_KEY", credential);
                  }}
                  onCredentialChange={() => {
                    clearPendingCredential("NGC_API_KEY");
                  }}
                />
              </div>
            )}
          </div>
        )}

        {/* Apply Controls ─────────────────────────────────────────────── */}
        {/* Surfaced only after the SME has a green test AND there's a
            credential to apply. ``allow_secret_persist`` from the env
            response controls whether the persist checkbox is visible. */}
        {Object.keys(pendingCredentials).length > 0 && (
          <ApplyControls
            allowPersist={env.allow_secret_persist}
            value={applyValue}
            onChange={setApplyValue}
          />
        )}

        {/* All-local rationale — each post-onboarding panel owns an explicit
            deploy action. Save only applies newly tested credentials, so this
            copy must not imply that navigation or Save starts containers. */}
        {effectiveTeacherMode === "local" && effectiveEmbeddingsMode === "local" && (
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            Deploy each local service from its configuration panel. First startup may
            take several minutes while NIM builds runtime artifacts.
          </Text>
        )}

        {/* Footer navigation */}
        <div
          className="flex items-center justify-between pt-4"
          style={{ borderTop: "1px solid var(--glass-border-subtle)" }}
        >
          <Button kind="secondary" onClick={() => navigate(`/projects/${projectId}`)}>
            <ArrowLeft size={14} /> Cancel
          </Button>
          {saveError && (
            <Text kind="body/regular/sm" role="alert" className="text-error">
              {saveError}
            </Text>
          )}
          <Button
            kind="primary"
            className="nvidia-green-button"
            // A self-hosted choice is not durable until the URL and exact
            // catalog model have passed the connection test. Other modes may
            // be saved with no new credential when the SME is only reviewing
            // an already-configured deployment.
            disabled={
              savePending ||
              (effectiveTeacherMode === "self_hosted" &&
                pendingSelfHostedTeacher === null) ||
              (effectiveEmbeddingsMode === "self_hosted" &&
                pendingSelfHostedEmbeddingUrl === null)
            }
            onClick={() => {
              void (async () => {
                setSavePending(true);
                setSaveError(null);
                try {
                  // Apply EVERY just-tested key (NVIDIA and/or NGC). The
                  // backend installs each in the runtime-override layer
                  // (always) and optionally writes it to .env (when the SME
                  // checked persist + deployment allows it). Applying the
                  // whole map — not a single slot — is what stops the
                  // second-tested key from dropping the first.
                  for (const [name, value] of Object.entries(pendingCredentials)) {
                    await setSecret({
                      name: name as SecretName,
                      value,
                      persist: applyValue.persist,
                    });
                  }
                  if (
                    effectiveTeacherMode === "self_hosted" &&
                    pendingSelfHostedTeacher
                  ) {
                    await configureSelfHostedTeacher(projectId, {
                      base_url: pendingSelfHostedTeacher.baseUrl,
                      model_config_id: pendingSelfHostedTeacher.selectedModelConfigId,
                    });
                  }
                  if (
                    effectiveEmbeddingsMode === "self_hosted" &&
                    pendingSelfHostedEmbeddingUrl
                  ) {
                    await configureSelfHostedEmbedding({
                      base_url: pendingSelfHostedEmbeddingUrl,
                    });
                  }
                  // Re-stamp setup acknowledgment. Post-onboarding the
                  // field is already non-null, so this is a guaranteed
                  // no-op (audit-clean) — kept for the service-layer
                  // idempotency guard.
                  try {
                    await markSetupCompleted(projectId, {
                      auto_skip: false,
                      teacher_mode: env.recommended_teacher_mode,
                      embedding_mode: env.recommended_embedding_mode,
                      embedding_provider: env.embedding_deployment.provider,
                    });
                  } catch (err: unknown) {
                    console.warn("markSetupCompleted on Save failed:", err);
                  }
                  await Promise.all([
                    queryClient.invalidateQueries({
                      queryKey: projectKeys.detail(projectId),
                    }),
                    queryClient.invalidateQueries({
                      queryKey: modelConfigKeys.list(projectId),
                    }),
                    queryClient.invalidateQueries({
                      queryKey: environmentKeys.assessment(),
                    }),
                  ]);
                  navigate(`/projects/${projectId}`);
                } catch (err: unknown) {
                  setSaveError(
                    parseApiErrorDetail(err) ??
                      (err instanceof Error
                        ? err.message
                        : "Failed to save NIM configuration."),
                  );
                } finally {
                  setSavePending(false);
                }
              })();
            }}
          >
            {savePending ? (
              <>
                <Spinner aria-label="Saving NIM configuration" /> Saving…
              </>
            ) : (
              <>
                Save <ArrowRight size={14} />
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

// ── Mode toggle (segmented pill) ────────────────────────────────────────────

interface ModeToggleProps {
  value: OverrideMode;
  onChange: (mode: OverrideMode) => void;
  localAvailable: boolean;
  idPrefix: string;
}

function ModeToggle({ value, onChange, localAvailable, idPrefix }: ModeToggleProps) {
  const options: { key: string; label: string }[] = [
    { key: "hosted", label: "Hosted NIM" },
    { key: "self_hosted", label: "Self-hosted" },
  ];
  if (localAvailable) {
    options.push({ key: "local", label: "Deploy locally" });
  }
  return (
    <SegmentedControl
      testId={`${idPrefix}-mode`}
      options={options}
      value={value}
      onChange={(key) => onChange(key as OverrideMode)}
    />
  );
}

// ── Missing prerequisites landing ───────────────────────────────────────────

interface MissingPrerequisitesViewProps {
  env: EnvironmentResponse;
  onSwitchToHosted: () => void;
  onBack: () => void;
}

function MissingPrerequisitesView({
  env,
  onSwitchToHosted,
  onBack,
}: MissingPrerequisitesViewProps) {
  const { copied, copy } = useCopyToClipboard();
  const setupCommand = "./scripts/setup-local.sh";

  const detectedLine = formatDetectedLine(env.gpus, env.docker_available);
  const missingCheckNames = new Set(
    env.missing_prerequisites.map((p) => p.check.toLowerCase()),
  );
  // Build a lightweight ✓/✗ prereq list. Positive items come from the
  // detected environment; negative items come from missing_prerequisites[*].
  const presentChecks: { name: string }[] = [];
  if (env.gpus.length > 0) {
    const gpuCount = env.gpus.length;
    presentChecks.push({
      name: `GPU: ${detectedLine.split(" · ")[0] || `${gpuCount} GPU(s) detected`}`,
    });
  }
  if (env.docker_available && !missingCheckNames.has("docker")) {
    presentChecks.push({ name: "Docker" });
  }
  if (
    env.nvidia_toolkit_available &&
    !missingCheckNames.has("nvidia container toolkit")
  ) {
    presentChecks.push({ name: "NVIDIA Container Toolkit" });
  }

  return (
    <div className="flex flex-1 flex-col gap-6 p-6">
      <div className="mx-auto w-full max-w-[1600px] flex flex-col gap-6">
        <div>
          <Text
            kind="title/xl"
            role="heading"
            aria-level={1}
            style={{ color: "var(--text-primary)" }}
          >
            NIM Configuration
          </Text>
          <Text
            kind="body/regular/sm"
            style={{
              color: "var(--text-muted)",
              display: "block",
              marginTop: 4,
            }}
          >
            Choose how Teacher and embedding models connect. We'll recommend the best
            setup for your hardware.
          </Text>
        </div>

        {detectedLine && (
          <Text kind="label/semibold/xs" className="glass-caption">
            Detected · {detectedLine}
          </Text>
        )}

        <div className="glass-inner-panel flex flex-col gap-4 rounded-[14px] px-4 py-4">
          <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
            Local deployment recommended but prerequisites missing:
          </Text>

          <div className="flex flex-col gap-2">
            {presentChecks.map((c) => (
              <span key={c.name} className="flex items-center gap-2">
                <Check size={14} style={{ color: "var(--accent-green)" }} />
                <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
                  {c.name}
                </Text>
              </span>
            ))}
            {env.missing_prerequisites.map((p: MissingPrerequisite) => (
              <div key={p.check} className="flex flex-col gap-1">
                <span className="flex items-center gap-2">
                  <X size={14} className="text-error" />
                  <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
                    {p.check} not found
                  </Text>
                </span>
                <Text
                  kind="body/regular/sm"
                  style={{ color: "var(--text-muted)", marginLeft: 22 }}
                >
                  {p.install_hint}
                </Text>
              </div>
            ))}
          </div>

          <div className="flex flex-col gap-2 pt-2">
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              Run the setup script to install prerequisites:
            </Text>
            <div className="glass-terminal flex items-center justify-between gap-3 rounded-md px-3 py-2">
              <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
                <code style={{ fontSize: 13 }}>{setupCommand}</code>
              </Text>
              <Button kind="secondary" onClick={() => void copy(setupCommand)}>
                {copied ? (
                  <>
                    <Check size={14} /> Copied
                  </>
                ) : (
                  <>
                    <Copy size={14} /> Copy to Clipboard
                  </>
                )}
              </Button>
            </div>
            <span className="flex items-center gap-1">
              <Terminal size={14} style={{ color: "var(--text-muted)" }} />
              <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                Then restart this application and return here.
              </Text>
            </span>
          </div>

          <div className="flex items-center gap-2 pt-2">
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              Or continue with NVIDIA hosted NIM:
            </Text>
            <Button kind="secondary" onClick={onSwitchToHosted}>
              Switch to Hosted
            </Button>
          </div>
        </div>

        <div
          className="flex items-center justify-start pt-4"
          style={{ borderTop: "1px solid var(--glass-border-subtle)" }}
        >
          <Button kind="secondary" onClick={onBack}>
            <ArrowLeft size={14} /> Back
          </Button>
        </div>
      </div>
    </div>
  );
}

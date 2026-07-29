// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NIM setup — path-aware setup choice screen (first step of the
 * three-screen FTU setup chain).
 *
 * The filename is historical: this screen began as a plain
 * NVIDIA-API-key paste form and has since grown into the path-aware
 * choice screen described below; the name is kept to avoid churning
 * its route and import call sites.
 *
 * Renders one of four layouts based on the environment assessment:
 *
 *   - **Case A** — no NVIDIA key + GPU + a teacher-eligible local model
 *     fits: PRIMARY card invites the SME to run the recommended local
 *     Teacher selected by the backend's quality-first hardware policy;
 *     PEER card below offers the configured hosted default via an inline API-key
 *     paste. Both are first-class — the local CTA is not a tiny escape
 *     link, and the hosted (NVIDIA API key) row is checked by default.
 *   - **Case B** — NVIDIA key already configured + GPU + local Teacher
 *     fits: PRIMARY card recommends the faster local Teacher while
 *     promising an immediate hosted bridge during download. PEER card
 *     keeps hosted-only available. Deploy → activePath="hybrid";
 *     Hosted only → activePath="hosted".
 *   - **Case C** — no key + (no GPU OR no fitting local Teacher):
 *     the plain hosted-only flow — paste API key + Go.
 *   - **Resident reuse** — an exact Blueprint-managed Teacher is already
 *     running and selected by project creation: auto-skip setup with the
 *     resident as the active local Teacher. No deploy or NGC key is needed.
 *   - **Case D** — key configured + no GPU: auto-skip to the NGC key
 *     screen.
 *
 * Case B does not auto-skip when the GPU is merely capable of a local
 * Teacher. It does auto-skip when the backend has already attached and
 * selected an exact running resident.
 *
 * State carried forward via ``location.state`` is a ``SetupChainState``
 * (see ``types/setupChain.ts``): ``activePath`` + ``localDeployQueued``
 * are the load-bearing fields for the NGC key screen and the setup
 * gate.
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button, Spinner, Text } from "@kui/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight } from "lucide-react";

import { testNgcCredential, testNvidiaCredential } from "@/api/nim";
import { fetchModelConfigs } from "@/api/model-configs";
import { environmentKeys, modelConfigKeys } from "@/api/query-keys";
import { setSecret } from "@/api/secrets";
import { useSetupContext } from "@/pages/ProjectSetupLayout";
import { DetectedHardwareEyebrow } from "@/components/DetectedHardwareEyebrow";
import { KeyPasteInput } from "@/components/KeyPasteInput";
import { KeyPortalLink } from "@/components/KeyPortalLink";
import { PathChoiceCard } from "@/components/PathChoiceCard";
import { usePersistedKeyProbe } from "@/hooks/usePersistedKeyProbe";
import {
  NGC_API_KEY_PORTAL_URL,
  NVIDIA_API_KEY_PORTAL_URL,
} from "@/lib/key-portal-urls";
import { formatModelDisplayName, localTeacherDisplayName } from "@/lib/model-display";
import { describeSecretPersistError } from "@/lib/secret-errors";
import type { ActivePath, SetupChainState } from "@/types/setupChain";

interface IncomingLocationState {
  cameFromAutoSkip?: boolean;
}

function buildLocalQueue(
  localTeacher: string | null | undefined,
  recommendEmbeddingLocal: boolean,
  embeddingModel: string,
): string[] {
  const queue: string[] = [];
  if (localTeacher) queue.push(localTeacher);
  if (recommendEmbeddingLocal) queue.push(embeddingModel);
  return queue;
}

// ── NgcKeyCluster ────────────────────────────────────────────────────────────
//
// The stacked NGC collection cluster — paste input, inline error line,
// "Get an NGC …" portal link — shared by the Case A local row and the
// Case B peer card. ``stopPropagation`` marks the label-embedded
// variant (Case A), where clicks inside the cluster must not toggle
// the surrounding row checkbox; ``onKeyDown`` lets that variant wire
// Enter-to-continue.

interface NgcKeyClusterProps {
  name: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  error: string | null;
  testId: string;
  portalLabel: string;
  inputRef?: React.RefObject<HTMLInputElement | null>;
  stopPropagation?: boolean;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
}

function NgcKeyCluster({
  name,
  value,
  onChange,
  error,
  testId,
  portalLabel,
  inputRef,
  stopPropagation = false,
  onKeyDown,
}: NgcKeyClusterProps): JSX.Element {
  const swallowClick = stopPropagation
    ? (e: React.MouseEvent<HTMLElement>) => e.stopPropagation()
    : undefined;
  return (
    <>
      <KeyPasteInput
        inputRef={inputRef}
        name={name}
        ariaLabel="NGC API key value"
        placeholder="Paste your NGC API key"
        value={value}
        onChange={onChange}
        onClick={swallowClick}
        onKeyDown={onKeyDown}
        testId={testId}
      />
      {error && (
        <Text kind="body/regular/sm" role="alert" className="text-error">
          {error}
        </Text>
      )}
      <KeyPortalLink
        href={NGC_API_KEY_PORTAL_URL}
        label={portalLabel}
        onClick={swallowClick}
        className="self-start"
      />
    </>
  );
}

export function NIMNvidiaKeyPage(): JSX.Element {
  const { projectId, project, environment } = useSetupContext();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const didAutoSkip = useRef(false);
  const env = environment;

  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  // Case A combined card: two independent checkboxes that
  // map to the two model stacks. Pick one or both; the button below
  // executes whatever combination is checked. Local defaults ON
  // (recommended for the detected GPU); hosted (NVIDIA API key) also
  // defaults ON — it reveals a key input when checked, and the SME can
  // uncheck it to continue local-only.
  const [localChecked, setLocalChecked] = useState(true);
  const [hostedChecked, setHostedChecked] = useState(true);
  // NGC API key lives inside the Local row: the "Run
  // locally" checkbox is the single toggle for "use my GPU." When it's
  // checked AND env.ngc_api_key_configured is false, the NGC input +
  // "Get an NGC API key" link appear inside the Local row, structurally
  // parallel to how the NVIDIA key appears inside the hosted row. No
  // separate ngcChecked state — Local owns NGC's visibility.
  const [ngcValue, setNgcValue] = useState("");
  const [ngcError, setNgcError] = useState<string | null>(null);
  // Set to true after the persisted-NGC-key probe returns
  // ``success: false``. While true, the NGC input + "Get an NGC API key"
  // link reveal even though ``env.ngc_api_key_configured`` is true —
  // the SME needs a place to type the corrected key. Resets never
  // happen here; the only way out is a successful probe of a freshly-
  // pasted key, which navigates away. Driven by both the on-mount
  // proactive probe (typical path) and the click-time backstop.
  const [ngcReplacementMode, setNgcReplacementMode] = useState(false);
  // Parallel of ``ngcReplacementMode`` for the NVIDIA API key. Driven by
  // the on-mount NVIDIA probe (Case B/D when env.nvidia_api_key_configured).
  // When true: Case B's primary hosted-ready card flips to
  // a replacement card with NVIDIA input + autofocus; Case D's auto-skip
  // is blocked and the screen renders a Case-C-style replacement layout.
  const [nvidiaReplacementMode, setNvidiaReplacementMode] = useState(false);
  // Tri-state probe completion signal used by Case D's auto-skip guard.
  // ``"pending"`` until the on-mount NVIDIA probe resolves (or is
  // skipped because env has no key); ``"resolved"`` once we have a
  // verdict (success ⇒ continue with auto-skip; failure ⇒ already
  // flipped nvidiaReplacementMode, which also blocks auto-skip).
  // Without this, Case D might fire its skip before the probe finishes,
  // re-introducing the silent-bad-key gap on no-GPU machines.
  const [nvidiaProbeState, setNvidiaProbeState] = useState<"pending" | "resolved">(
    "pending",
  );
  // True while handleCombinedContinue / handleCaseBYes is in flight.
  // Used in addition to mutation pending states so the button reflects
  // the full continue-flow latency and rejects double-clicks.
  const [submitting, setSubmitting] = useState(false);
  // Refs for autofocus on reveal. When a proactive probe flips a
  // replacement mode to true, the corresponding input becomes visible
  // and we focus it so the SME can start pasting immediately.
  const ngcInputRef = useRef<HTMLInputElement | null>(null);
  const nvidiaInputRef = useRef<HTMLInputElement | null>(null);

  const hasGpu = env.gpus.length > 0;
  const localTeacherName = env.recommended_local_teacher_model_name ?? null;
  // Case A vs Case C hinges on whether ANY teacher-eligible local model
  // fits the GPU. The backend has already done the role + memory
  // filtering in ``_pick_local_teacher_recommendation``; the frontend
  // just reads the result.
  const localTeacherFits = localTeacherName !== null;
  const recommendEmbeddingLocal = env.recommended_embedding_mode === "local";
  const runningRecommendedTeacher = (env.active_local_nim_residents ?? []).find(
    (resident) =>
      resident.role === "teacher" &&
      resident.status === "running" &&
      resident.model_name === localTeacherName &&
      env.recommended_teacher_mode === "local",
  );
  const selectedTeacherQuery = useQuery({
    queryKey: modelConfigKeys.list(projectId, "teacher"),
    queryFn: () => fetchModelConfigs(projectId, "teacher"),
    enabled: runningRecommendedTeacher !== undefined,
  });
  const selectedTeacher = selectedTeacherQuery.data?.items.find(
    (config) => config.model_config_id === project.teacher_model_config_id,
  );
  const reusedTeacherIsSelected =
    runningRecommendedTeacher !== undefined &&
    selectedTeacher?.model_name === runningRecommendedTeacher.model_name;
  const localEmbeddingAlreadyRunning = ["self_hosted_nvclip", "local_nvclip"].includes(
    env.embedding_deployment.provider,
  );
  const needsLocalEmbeddingSetup =
    reusedTeacherIsSelected &&
    env.recommended_embedding_mode === "local" &&
    !localEmbeddingAlreadyRunning;

  const keyConfigured = env.nvidia_api_key_configured;

  // ── Case identification ─────────────────────────────────────────────
  // The four cases drive the render branch. ``activePath`` is the
  // committed-to-FTUE-chain value; it's set at navigation time, not at
  // render time, because Cases A and B let the SME pick.
  let renderCase: "A" | "B" | "C" | "D";
  if (keyConfigured && !hasGpu) {
    renderCase = "D";
  } else if (keyConfigured && localTeacherFits) {
    renderCase = "B";
  } else if (!keyConfigured && localTeacherFits) {
    renderCase = "A";
  } else {
    renderCase = "C";
  }

  // ── Single-GPU handling (one-NIM-per-GPU) ───────────────────────────
  // Case A copy uses this to avoid promising a second local embedding
  // NIM beside the selected Teacher. Case B uses the hosted endpoint as
  // an immediate bridge while the recommended local Teacher downloads.
  const singleGpu = env.gpus.length === 1;

  // ── Proactive validation on mount (see hooks/usePersistedKeyProbe) ──
  // Probe whichever effective keys are configured AND relevant to this
  // case. On failure the input REVEALS with the error + autofocus;
  // success makes no visible change.

  usePersistedKeyProbe({
    enabled:
      runningRecommendedTeacher === undefined &&
      (renderCase === "A" || renderCase === "B"),
    configured: env.ngc_api_key_configured,
    probe: () => testNgcCredential(),
    fallbackError: "NGC key validation failed.",
    onRejected: (message) => {
      setNgcReplacementMode(true);
      setNgcError(message);
    },
  });

  usePersistedKeyProbe({
    enabled:
      runningRecommendedTeacher === undefined &&
      (renderCase === "B" || renderCase === "D"),
    configured: env.nvidia_api_key_configured,
    probe: () => testNvidiaCredential(),
    fallbackError: "NVIDIA API key validation failed.",
    onRejected: (message) => {
      setNvidiaReplacementMode(true);
      setError(message);
    },
    // Settling (even when there's nothing to probe) unblocks Case D's
    // auto-skip gate on ``nvidiaProbeState``.
    onSettled: () => setNvidiaProbeState("resolved"),
  });

  // Autofocus the inputs when their reveal flips. The refs attach to
  // the input elements in the JSX below (which only render when the
  // corresponding replacement mode or "no key configured" case applies).
  useEffect(() => {
    if (ngcReplacementMode) ngcInputRef.current?.focus();
  }, [ngcReplacementMode]);
  useEffect(() => {
    if (nvidiaReplacementMode) nvidiaInputRef.current?.focus();
  }, [nvidiaReplacementMode]);

  // ── Auto-skip an already-selected reusable Teacher ──────────────────
  // Project creation has already attached a project-local endpoint and
  // selected this exact model config. Nothing needs deploying and NGC is
  // irrelevant until the user chooses a different local NIM later.
  useEffect(() => {
    if (didAutoSkip.current || !reusedTeacherIsSelected) return;
    didAutoSkip.current = true;
    const next: SetupChainState = {
      activePath: "local",
      // The Teacher needs no action. A separately recommended embedding NIM
      // still needs the NGC/gate flow so it can collect credentials and
      // actually dispatch that deployment.
      cameFromAutoSkip: !needsLocalEmbeddingSetup,
      localDeployQueued: [],
    };
    navigate(needsLocalEmbeddingSetup ? "../setup/ngc" : "../setup/done", {
      replace: true,
      state: next,
    });
  }, [reusedTeacherIsSelected, needsLocalEmbeddingSetup, navigate]);

  // ── Auto-skip (Case D only) ─────────────────────────────────────────
  // Two gates besides the Case D check itself:
  //   1. Wait for the NVIDIA probe to resolve (``nvidiaProbeState``).
  //   2. If the probe said the key is bad (``nvidiaReplacementMode``),
  //      don't skip — render the replacement screen so the SME can
  //      paste a working key.
  useEffect(() => {
    if (didAutoSkip.current) return;
    if (runningRecommendedTeacher !== undefined) return;
    if (renderCase !== "D") return;
    if (nvidiaProbeState !== "resolved") return;
    if (nvidiaReplacementMode) return;
    didAutoSkip.current = true;
    const next: SetupChainState = {
      activePath: "hosted",
      cameFromAutoSkip: true,
      localDeployQueued: [],
    };
    navigate("../setup/ngc", { replace: true, state: next });
  }, [
    renderCase,
    nvidiaProbeState,
    nvidiaReplacementMode,
    navigate,
    runningRecommendedTeacher,
  ]);

  // ── Test-on-Go mutation (validates + persists a hosted NVIDIA key).
  //
  // The mutation handles only the network steps — the component decides
  // what to navigate to on success based on which card the SME engaged
  // with (peer card honors ``hybridChecked``; Case C goes hosted-only;
  // primary card's disclosure chains into a local-deploy navigation).
  // ──────────────────────────────────────────────────────────────────
  const testAndPersistKey = useMutation({
    mutationFn: async (credential: string) => {
      // Use the real auth-gated probe — testConnection's probe_kind="models"
      // path hits GET /v1/models which build.nvidia.com serves PUBLICLY,
      // so it validates nothing. The NVIDIA credential endpoint hits
      // POST /v1/chat/completions which actually validates the bearer.
      const result = await testNvidiaCredential(credential);
      if (!result.success) {
        throw new Error(result.error ?? "Connection failed.");
      }
      await setSecret({
        name: "NVIDIA_API_KEY",
        value: credential,
        persist: true,
      });
      // The environment assessment drives downstream honesty gates —
      // the setup gate only renders the "Hosted on build.nvidia.com"
      // embedding row when ``nvidia_api_key_configured`` is true.
      // Without this refetch the layout's cached assessment (fetched
      // before the paste) still says false and the gate silently omits
      // the row the key just paid for. Awaited so the gate mounts with
      // fresh env.
      await queryClient.invalidateQueries({
        queryKey: environmentKeys.assessment(),
      });
      return result;
    },
  });

  // Hosted-key-paste flow for Case C ONLY (Case A uses the combined
  // handler below). On success, navigates to the NGC screen as a
  // hosted-only chain — Case C is the no-GPU branch so there's no local
  // queue to build.
  async function handleHostedGo() {
    setError(null);
    if (!value) return;
    try {
      await testAndPersistKey.mutateAsync(value);
      const next: SetupChainState = {
        activePath: "hosted",
        cameFromAutoSkip: false,
        localDeployQueued: [],
      };
      navigate("../setup/ngc", { state: next });
    } catch (err) {
      setError(describeSecretPersistError(err, "NVIDIA_API_KEY"));
    }
  }

  // Continue handler used by the NVIDIA replacement card (Case B or D
  // when ``nvidiaReplacementMode === true``). Validates + persists the
  // freshly-pasted key via the existing ``testAndPersistKey`` mutation,
  // then navigates to /setup/done with the hosted-only path. The SME
  // can add a local Teacher later via NIM Configuration.
  async function handleNvidiaReplacementContinue() {
    setError(null);
    if (!value) return;
    setSubmitting(true);
    try {
      await testAndPersistKey.mutateAsync(value);
      // Clear the replacement-mode banner so a subsequent navigation
      // back to /setup doesn't show it stale.
      setNvidiaReplacementMode(false);
      navigate("../setup/done", {
        state: {
          activePath: "hosted",
          cameFromAutoSkip: false,
          localDeployQueued: [],
        } satisfies SetupChainState,
      });
    } catch (err) {
      setError(describeSecretPersistError(err, "NVIDIA_API_KEY"));
    } finally {
      setSubmitting(false);
    }
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setValue(e.target.value);
    if (error) setError(null);
  }

  function handleNgcChange(e: React.ChangeEvent<HTMLInputElement>) {
    setNgcValue(e.target.value);
    if (ngcError) setNgcError(null);
  }

  // ── NGC validate + persist (shared by Case A and Case B-Yes) ───────
  //
  // Probes whatever NGC key is in play, then persists a fresh paste.
  // Two sub-cases, both gated:
  //
  //   (a) SME pasted a fresh NGC key (``ngcValue`` non-empty — the
  //       input only renders when NGC needs collecting) → probe THAT
  //       key, then persist on success.
  //   (b) ``env.ngc_api_key_configured`` is true (prior session left
  //       a key in .env) → probe the currently-effective key (no
  //       credential body — backend pulls from runtime secrets). No
  //       persist needed; the key is already on disk.
  //
  // Sub-case (b) matters: without it, a bad-but-already-persisted key
  // sails past the gate just because the SME didn't re-paste, then
  // fails later when ``POST :deploy`` hits nvcr.io.
  //
  // Returns false when the caller must stop (probe rejected or persist
  // failed) — the error state is already set.
  async function validateAndPersistNgcKey(): Promise<boolean> {
    const freshlyPasted = ngcValue.trim().length > 0;
    const wantsLocalNgc = freshlyPasted || env.ngc_api_key_configured;
    if (wantsLocalNgc) {
      try {
        const probe = await testNgcCredential(freshlyPasted ? ngcValue : undefined);
        if (!probe.success) {
          // Flip replacement mode so the input stays visible on the
          // next render (covers the persisted-key path where the
          // input was hidden before the click).
          setNgcReplacementMode(true);
          setNgcError(probe.error ?? "NGC key validation failed.");
          return false;
        }
      } catch (err) {
        // Network error / 5xx — fall through so a flaky nvcr.io
        // doesn't strand the SME. LocalDeployBanner still catches a
        // true rejection later if the probe was lying.
        console.warn("NGC pre-flight against nvcr.io threw:", err);
      }
    }
    if (freshlyPasted) {
      try {
        await setSecret({
          name: "NGC_API_KEY",
          value: ngcValue,
          persist: true,
        });
      } catch (err) {
        setNgcError(describeSecretPersistError(err, "NGC_API_KEY"));
        return false;
      }
    }
    return true;
  }

  // ── Case A combined handler ────────────────────────────────────────
  //
  // One button drives whatever combination of checkboxes the SME picked:
  //   * local only           → activePath="local",   queue=[cosmos, embedding-if-local]
  //   * hosted only          → activePath="hosted",  queue=[embedding-if-local]
  //   * both (hybrid)        → activePath="hybrid",  queue=[cosmos, embedding-if-local]
  //   * neither              → button is disabled
  //
  // The handler validates and persists credentials BEFORE navigating so a
  // bad key blocks the entire setup instead of half-configuring the
  // project. With NGC collected inline, there's nothing left
  // for the NGC key screen to ask in this flow, so we navigate directly
  // to the setup gate.
  async function handleCombinedContinue() {
    setError(null);
    setNgcError(null);
    setSubmitting(true);
    try {
      await runCombinedContinue();
    } finally {
      setSubmitting(false);
    }
  }

  async function runCombinedContinue() {
    const wantHosted = hostedChecked && value.trim().length > 0;
    if (wantHosted) {
      try {
        await testAndPersistKey.mutateAsync(value);
      } catch (err) {
        setError(describeSecretPersistError(err, "NVIDIA_API_KEY"));
        return;
      }
    }

    // NGC validation when Local is checked (see validateAndPersistNgcKey
    // for the fresh-paste vs. already-persisted sub-cases).
    if (localChecked) {
      const ok = await validateAndPersistNgcKey();
      if (!ok) return;
    }

    // Local-deploy queue: ONLY when the Local checkbox is checked.
    // Hosted-only path doesn't queue any local NIM containers; the
    // embedding falls back to whatever the backend configures from env
    // (hosted embeddings via the NVIDIA API key, or `none`). This makes
    // the "Run locally" checkbox the single switch for "use my GPU."
    const queue = localChecked
      ? buildLocalQueue(
          localTeacherName,
          recommendEmbeddingLocal,
          env.embedding_deployment.model_name,
        )
      : [];

    // Checking Local IS choosing the local Teacher — the row copy reads
    // "<model> as your Teacher, on your GPU", so the setup gate must bind it
    // (activePath="local" rebinds teacher_model_config_id on [Start
    // labeling]). Also checking Hosted adds the NVIDIA key on top: hosted
    // embeddings for review-queue variety (the single-GPU hybrid mode)
    // plus alternate hosted Teachers in the picker — it does NOT demote
    // the local pick to an alternate. Case B uses activePath="hybrid"
    // differently: hosted is the immediate bridge while the preferred
    // local Teacher downloads and activates after verification.
    const activePath: ActivePath = localChecked ? "local" : "hosted";

    navigate("../setup/done", {
      state: {
        activePath,
        cameFromAutoSkip: false,
        localDeployQueued: queue,
      } satisfies SetupChainState,
    });
  }

  // Case B [Start now & deploy local Teacher]: NVIDIA key is already
  // configured. NGC is collected inline on the local card; we validate the
  // key against nvcr.io BEFORE persisting (same pattern as Case A) so
  // a wrong-scope key is caught here rather than after navigation.
  async function handleCaseBYes() {
    setError(null);
    setNgcError(null);
    setSubmitting(true);
    try {
      await runCaseBYes();
    } finally {
      setSubmitting(false);
    }
  }

  async function runCaseBYes() {
    // Same Case A logic: probe whatever NGC key is in play (fresh paste
    // OR effective from .env). The hybrid path is local-deploy-bound so
    // a bad NGC key would fail the container pull just like in Case A.
    const ok = await validateAndPersistNgcKey();
    if (!ok) return;

    const queue = buildLocalQueue(
      localTeacherName,
      recommendEmbeddingLocal,
      env.embedding_deployment.model_name,
    );
    navigate("../setup/done", {
      state: {
        activePath: "hybrid",
        cameFromAutoSkip: false,
        localDeployQueued: queue,
      } satisfies SetupChainState,
    });
  }

  // Case B [Use hosted only]: NVIDIA key configured, user opted out of
  // local. No NGC needed (no local deploys queued); navigate
  // straight to the setup gate as a hosted-only chain.
  function goHostedOnly() {
    navigate("../setup/done", {
      state: {
        activePath: "hosted",
        cameFromAutoSkip: false,
        localDeployQueued: [],
      } satisfies SetupChainState,
    });
  }

  // Suppress render while resident selection is being confirmed or its
  // auto-skip is firing. A failed lookup falls through to the normal choice
  // screen instead of assuming project creation succeeded.
  if (
    runningRecommendedTeacher !== undefined &&
    (selectedTeacherQuery.isLoading || reusedTeacherIsSelected)
  ) {
    return <></>;
  }

  // Suppress render while Case D auto-skip is firing — UNLESS the
  // NVIDIA probe flagged the persisted key as bad, in which case auto-
  // skip is blocked and we need to render the replacement card so the
  // SME can paste a fresh key.
  if (
    renderCase === "D" &&
    !nvidiaReplacementMode &&
    !(location.state as IncomingLocationState)?.cameFromAutoSkip
  ) {
    return <></>;
  }

  // Case C [Go]: hosted-only single-input flow.
  const disabled = testAndPersistKey.isPending || value.length === 0;
  // Plain-English variant name used in both the Case A combined card
  // and the Case B hybrid offer — always the exact variant the backend
  // picked for the detected GPU.
  const localTeacherDisplay = localTeacherDisplayName(localTeacherName);
  const hostedTeacherDisplay = formatModelDisplayName(env.default_teacher_model_name);
  // NGC inline input lives inside the Local row. Shown when Local is
  // relevant AND either env has no key OR the persisted key has been
  // probed and found bad (``ngcReplacementMode``). For Hosted-only, NGC
  // is irrelevant — no local NIM containers will be pulled.
  const needsNgcInline =
    renderCase === "A" && (!env.ngc_api_key_configured || ngcReplacementMode);
  // Case A [Set up & continue]: enabled when at least one option is
  // picked AND all required credentials are filled. If Local is checked
  // and NGC needs collecting, the NGC value must be non-empty.
  const combinedDisabled =
    submitting ||
    testAndPersistKey.isPending ||
    (!localChecked && !hostedChecked) ||
    (hostedChecked && value.trim().length === 0) ||
    (localChecked && needsNgcInline && ngcValue.trim().length === 0);

  return (
    <div className="flex flex-1 flex-col items-center justify-center p-6">
      <div className="flex w-full max-w-[680px] flex-col gap-6">
        {/* Page-level hardware eyebrow renders for Cases B/C — Case A
            inlines the GPU/Docker/Toolkit chips inside the Local row
            because they're the prerequisites for THAT option. */}
        {renderCase !== "A" && <DetectedHardwareEyebrow env={env} />}

        {renderCase === "A" && (
          <div
            className="glass-card--elevated flex flex-col gap-5 p-6"
            data-testid="combined-card-local-and-hosted"
          >
            <div className="flex flex-col gap-1">
              <Text kind="title/lg" style={{ color: "var(--text-primary)" }}>
                Set up your Teacher
              </Text>
              <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                Pick one or both. You can add or change models later from the project
                menu.
              </Text>
            </div>

            {/* Local row — checked by default (recommended for the
                detected GPU). The whole row is a label so clicking
                anywhere on it toggles the checkbox. When checked AND
                NGC isn't already configured, the NGC API key input + a
                "Get an NGC API key" link reveal inside the row, exactly
                mirroring how the hosted/NVIDIA row reveals its key. */}
            <label
              className="flex items-start gap-3 cursor-pointer rounded-lg p-3"
              style={{
                background: "rgba(118, 185, 0, 0.06)",
                border: "1px solid rgba(118, 185, 0, 0.18)",
              }}
              data-testid="local-row-label"
            >
              <input
                type="checkbox"
                className="glass-input mt-1"
                checked={localChecked}
                onChange={(e) => setLocalChecked(e.target.checked)}
                data-testid="local-row-checkbox"
              />
              <div className="flex flex-1 flex-col gap-2">
                <div className="flex items-center gap-2">
                  <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
                    Run {localTeacherDisplay} locally
                  </Text>
                  <Text
                    kind="label/bold/xs"
                    style={{
                      color: "var(--accent-green)",
                      letterSpacing: "0.06em",
                    }}
                  >
                    · RECOMMENDED
                  </Text>
                </div>
                <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                  {/* Promise exactly what buildLocalQueue will queue:
                      NeMo Retriever VL is only mentioned when the
                      backend's placement-aware assessment recommends
                      the local embedding NIM (a heterogeneous host can
                      be multi-GPU yet have no fitting device left). */}
                  {recommendEmbeddingLocal && !singleGpu
                    ? `${localTeacherDisplay} as your Teacher + NeMo Retriever VL for image embeddings, each on its own GPU. Local responses are typically seconds. Local embeddings improve image variety in review. We deploy both in the background after the next step.`
                    : `${localTeacherDisplay} as your Teacher, on your GPU. Local responses are typically seconds. We deploy it in the background after the next step.`}
                </Text>
                {/* Prereq chips sit next to the option they gate. All
                    three are guaranteed to be ✓ when this card renders
                    (env.recommended_local_teacher_* is non-null only when
                    GPU + Docker + Toolkit all passed). */}
                <DetectedHardwareEyebrow env={env} />
                {localChecked && needsNgcInline && (
                  <NgcKeyCluster
                    inputRef={ngcInputRef}
                    name="ngc_api_key_paste"
                    value={ngcValue}
                    onChange={handleNgcChange}
                    error={ngcError}
                    testId="ngc-key-input"
                    portalLabel="Get an NGC API key"
                    stopPropagation
                    onKeyDown={(e) => {
                      e.stopPropagation();
                      if (e.key === "Enter" && !combinedDisabled) {
                        void handleCombinedContinue();
                      }
                    }}
                  />
                )}
              </div>
            </label>

            {/* Hosted (NVIDIA API key) row — unchecked by default. When
                checked, the inline credential input expands underneath
                with the "Get an NVIDIA API key" link right next to it. */}
            <label
              className="flex items-start gap-3 cursor-pointer rounded-lg p-3"
              style={{
                background: "rgba(255, 255, 255, 0.03)",
                border: "1px solid rgba(255, 255, 255, 0.08)",
              }}
              data-testid="hosted-row-label"
            >
              <input
                type="checkbox"
                className="glass-input mt-1"
                checked={hostedChecked}
                onChange={(e) => setHostedChecked(e.target.checked)}
                data-testid="hosted-row-checkbox"
              />
              <div className="flex flex-1 flex-col gap-2">
                <div className="flex items-center gap-2">
                  <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
                    NVIDIA API key
                  </Text>
                  <Text
                    kind="label/bold/xs"
                    style={{
                      color: "var(--accent-green)",
                      letterSpacing: "0.06em",
                    }}
                  >
                    · RECOMMENDED
                  </Text>
                </div>
                <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                  A free NVIDIA API key unlocks more Teacher models and hosted image
                  embeddings, improving review variety without using your GPU. Takes
                  about 30 seconds to get.
                </Text>
                {hostedChecked && (
                  <>
                    <KeyPasteInput
                      inputRef={nvidiaInputRef}
                      name="nvidia_api_key_paste"
                      ariaLabel="NVIDIA API key value"
                      placeholder="nvapi-..."
                      value={value}
                      onChange={handleChange}
                      onClick={(e) => e.stopPropagation()}
                      onKeyDown={(e) => {
                        e.stopPropagation();
                        if (e.key === "Enter" && !combinedDisabled) {
                          void handleCombinedContinue();
                        }
                      }}
                      testId="hosted-key-input"
                    />
                    <KeyPortalLink
                      href={NVIDIA_API_KEY_PORTAL_URL}
                      label="Get an NVIDIA API key"
                      onClick={(e) => e.stopPropagation()}
                      className="self-start"
                    />
                  </>
                )}
              </div>
            </label>

            {error && (
              <Text kind="body/regular/sm" role="alert" className="text-error">
                {error}
              </Text>
            )}

            <div className="flex items-center justify-end">
              <Button
                kind="primary"
                className="nvidia-green-button"
                disabled={combinedDisabled}
                onClick={() => {
                  void handleCombinedContinue();
                }}
              >
                {submitting || testAndPersistKey.isPending ? (
                  <>
                    <Spinner aria-label="Setting up" /> Setting up...
                  </>
                ) : (
                  <>
                    Set up &amp; continue <ArrowRight size={14} />
                  </>
                )}
              </Button>
            </div>

            {!localChecked && !hostedChecked && (
              <Text
                kind="body/regular/xs"
                style={{ color: "var(--text-muted)" }}
                data-testid="combined-helper"
              >
                Pick at least one option to continue.
              </Text>
            )}
          </div>
        )}

        {/* Case B renders this in place of its primary card; Case D
            normally auto-skips, but when the on-mount NVIDIA probe
            flags the persisted key as bad, auto-skip is blocked
            upstream and the replacement card renders so the SME can
            paste a fresh key without going to NIM Configuration. Only
            one renderCase applies per mount. */}
        {(renderCase === "B" || renderCase === "D") && nvidiaReplacementMode && (
          <NvidiaReplacementCard
            error={error}
            value={value}
            onChange={handleChange}
            inputRef={nvidiaInputRef}
            disabled={
              submitting || testAndPersistKey.isPending || value.trim().length === 0
            }
            submitting={submitting || testAndPersistKey.isPending}
            onContinue={() => {
              void handleNvidiaReplacementContinue();
            }}
          />
        )}

        {renderCase === "B" && !nvidiaReplacementMode && (
          <>
            <div className="flex flex-col gap-2">
              <Text kind="title/xl" style={{ color: "var(--text-primary)" }}>
                Set up your Teacher
              </Text>
              <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                Your GPU can run a faster local Teacher. Use the hosted Teacher now
                while the local NIM downloads.
              </Text>
            </div>

            <PathChoiceCard
              kind="primary"
              eyebrow="RECOMMENDED · FASTEST RESPONSES"
              title={`Run ${localTeacherDisplay} locally`}
              meta={`Local responses are typically seconds. We will start with ${hostedTeacherDisplay} now while the local NIM downloads.`}
              testId="primary-card-local-recommended"
            >
              <Text
                kind="label/bold/xs"
                style={{
                  color: "var(--accent-green)",
                  letterSpacing: "0.04em",
                }}
              >
                HOSTED FIRST · SWITCHES TO LOCAL AFTER VERIFICATION
              </Text>
              {(!env.ngc_api_key_configured || ngcReplacementMode) && (
                <div className="flex flex-col gap-2">
                  <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                    Needs an NGC API key to download the container image from nvcr.io.
                  </Text>
                  <NgcKeyCluster
                    name="ngc_api_key_paste_caseb"
                    value={ngcValue}
                    onChange={handleNgcChange}
                    error={ngcError}
                    testId="caseb-ngc-key-input"
                    portalLabel="Get an NGC API key"
                  />
                </div>
              )}
              <div className="flex items-center justify-end">
                <Button
                  kind="primary"
                  className="nvidia-green-button"
                  disabled={
                    submitting ||
                    ((!env.ngc_api_key_configured || ngcReplacementMode) &&
                      ngcValue.trim().length === 0)
                  }
                  onClick={() => {
                    void handleCaseBYes();
                  }}
                >
                  {submitting ? (
                    <>
                      <Spinner aria-label="Setting up" /> Setting up...
                    </>
                  ) : (
                    <>
                      Start now &amp; deploy local Teacher <ArrowRight size={14} />
                    </>
                  )}
                </Button>
              </div>
            </PathChoiceCard>

            <PathChoiceCard
              kind="peer"
              eyebrow="ALTERNATIVE · HOSTED ONLY"
              title={`Keep ${hostedTeacherDisplay} hosted`}
              meta="Nothing to download and no GPU changes. Hosted response times vary with service demand."
              testId="hosted-only-card"
            >
              <div className="flex items-center justify-end">
                <Button kind="secondary" onClick={goHostedOnly}>
                  Use hosted only <ArrowRight size={14} />
                </Button>
              </div>
            </PathChoiceCard>
          </>
        )}

        {renderCase === "C" && (
          <>
            <div className="flex flex-col gap-2">
              <Text kind="title/xl" style={{ color: "var(--text-primary)" }}>
                Paste your NVIDIA API key to get started
              </Text>
              <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                Grab a key from build.nvidia.com in about 30 seconds. It gives you
                access to NVIDIA's top hosted models with no GPU required.
              </Text>
            </div>

            <div className="flex flex-col gap-2">
              <KeyPasteInput
                name="nvidia_api_key_paste"
                ariaLabel="NVIDIA API key value"
                placeholder="nvapi-..."
                value={value}
                onChange={handleChange}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !disabled) handleHostedGo();
                }}
              />
              {error && (
                <Text kind="body/regular/sm" role="alert" className="text-error">
                  {error}
                </Text>
              )}
            </div>

            <div className="flex items-center justify-between">
              <KeyPortalLink
                href={NVIDIA_API_KEY_PORTAL_URL}
                label="Get a NVIDIA API key"
                className="gap-2"
              />
              <Button
                kind="primary"
                className="nvidia-green-button"
                disabled={disabled}
                onClick={() => {
                  void handleHostedGo();
                }}
              >
                {testAndPersistKey.isPending ? (
                  <>
                    <Spinner aria-label="Testing" /> Testing...
                  </>
                ) : (
                  <>
                    Go <ArrowRight size={14} />
                  </>
                )}
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── NvidiaReplacementCard ────────────────────────────────────────────────────
//
// Shared full-card layout used by Case B (overrides the hosted-ready
// Teacher choices) and Case D (renders in place of the auto-skip
// redirect) when the on-mount NVIDIA probe fails. Single-input flow,
// autofocused via the parent's ref so the SME can start pasting the
// moment the card appears.

interface NvidiaReplacementCardProps {
  error: string | null;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  disabled: boolean;
  submitting: boolean;
  onContinue: () => void;
}

function NvidiaReplacementCard({
  error,
  value,
  onChange,
  inputRef,
  disabled,
  submitting,
  onContinue,
}: NvidiaReplacementCardProps): JSX.Element {
  return (
    <div
      className="glass-card--elevated flex flex-col gap-4 p-6"
      data-testid="nvidia-replacement-card"
    >
      <div className="flex flex-col gap-1">
        <Text kind="title/lg" style={{ color: "var(--text-primary)" }}>
          Your NVIDIA API key was rejected
        </Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          The key build.nvidia.com has on file no longer works. Paste a fresh key and
          we'll re-test it before continuing.
        </Text>
      </div>

      <KeyPasteInput
        inputRef={inputRef}
        name="nvidia_api_key_paste_replacement"
        ariaLabel="NVIDIA API key value"
        placeholder="nvapi-..."
        value={value}
        onChange={onChange}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !disabled) onContinue();
        }}
        testId="nvidia-replacement-input"
      />

      {error && (
        <Text kind="body/regular/sm" role="alert" className="text-error">
          {error}
        </Text>
      )}

      <div className="flex items-center justify-between">
        <KeyPortalLink href={NVIDIA_API_KEY_PORTAL_URL} label="Get an NVIDIA API key" />
        <Button
          kind="primary"
          className="nvidia-green-button"
          disabled={disabled}
          onClick={onContinue}
        >
          {submitting ? (
            <>
              <Spinner aria-label="Testing" /> Testing...
            </>
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

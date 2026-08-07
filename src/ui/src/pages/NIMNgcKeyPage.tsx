// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NIM setup — NGC API key screen (second step of the three-screen FTU
 * setup chain).
 *
 * Only hosted-path flows reach this screen (Case C's key paste, Case
 * B-Skip, and Case D's auto-skip): local and hybrid flows collect NGC
 * inline on the setup-choice screen and navigate straight to the setup
 * gate. Here NGC is therefore always **optional** — an offer of faster
 * local embeddings ("Want faster local embeddings?"), never a
 * requirement.
 *
 * Local embeddings are queued only when the backend recommends that
 * provider. This preserves explicit ``EMBEDDING_PROVIDER=none`` opt-out
 * and keeps GPU support-matrix policy in the backend.
 *
 * Instead of a "Skip" affordance, the screen offers a plain ``[Back]``
 * button that returns the SME to the setup-choice screen. Going back
 * lets them adjust their toggles honestly; the hosted key, if
 * persisted on that screen's continue, is preserved (a fresh
 * setup-choice screen renders as Case B hybrid offer in that
 * situation).
 *
 * Auto-skip rule: skip if ``ngc_api_key_configured`` (nothing to ask)
 * OR the backend does not recommend local embeddings. In those cases the
 * screen's only offer cannot or should not be delivered, so collecting an
 * NGC key would be misleading.
 */

import { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button, Spinner, Text } from "@kui/react";
import { useMutation } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, Check } from "lucide-react";

import { setSecret } from "@/api/secrets";
import { KeyPasteInput } from "@/components/KeyPasteInput";
import { KeyPortalLink } from "@/components/KeyPortalLink";
import { SetupTransitionCard } from "@/components/common/SetupTransitionCard";
import { useEnvironmentSetupContext } from "@/pages/setup-context";
import { NGC_API_KEY_PORTAL_URL } from "@/lib/key-portal-urls";
import { describeSecretPersistError } from "@/lib/secret-errors";
import type { EnvironmentResponse } from "@/types/nim";
import { DEFAULT_SETUP_CHAIN_STATE, type SetupChainState } from "@/types/setupChain";

// The backend owns provider selection, placement, and hardware eligibility.
// The includes-guard keeps deep-linked chains duplicate-free.
function withLocalEmbeddingQueued(env: EnvironmentResponse, queue: string[]): string[] {
  if (env.recommended_embedding_mode !== "local") return queue;
  const model = env.embedding_deployment.model_name;
  return queue.includes(model) ? queue : [...queue, model];
}

export function NIMNgcKeyPage(): JSX.Element {
  const { environment } = useEnvironmentSetupContext();
  const navigate = useNavigate();
  const location = useLocation();
  const incoming = {
    ...DEFAULT_SETUP_CHAIN_STATE,
    ...((location.state as Partial<SetupChainState>) ?? {}),
  };
  const didAutoSkip = useRef(false);
  const env = environment;

  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);

  // Skip when there is nothing to ask or the backend selected a hosted/pHash
  // embedding path. The frontend does not recreate provider policy.
  const shouldAutoSkip =
    env.ngc_api_key_configured || env.recommended_embedding_mode !== "local";

  // Apply the backend-selected local provider on both exits: manual
  // [Use my GPU] and the NGC-already-configured auto-skip.
  const forwardedQueue = withLocalEmbeddingQueued(env, incoming.localDeployQueued);

  // ── Auto-skip (→ setup gate) ─────────────────────────────────────────
  useEffect(() => {
    if (didAutoSkip.current) return;
    if (!shouldAutoSkip) return;
    didAutoSkip.current = true;
    const next: SetupChainState = {
      activePath: incoming.activePath,
      cameFromAutoSkip: incoming.cameFromAutoSkip,
      localDeployQueued: forwardedQueue,
    };
    navigate("../setup/done", { replace: true, state: next });
  }, [
    shouldAutoSkip,
    incoming.activePath,
    incoming.cameFromAutoSkip,
    forwardedQueue,
    navigate,
  ]);

  // ── Apply NGC key ───────────────────────────────────────────────────
  const applyMutation = useMutation({
    mutationFn: async (credential: string) => {
      await setSecret({
        name: "NGC_API_KEY",
        value: credential,
        persist: true,
      });
    },
    onSuccess: () => {
      // [Use my GPU] makes the heading's promise real: the embedding
      // NIM joins the deploy queue (the gate dispatches it on [Start
      // labeling]). This screen only renders when the host fits it.
      const next: SetupChainState = {
        activePath: incoming.activePath,
        cameFromAutoSkip: false,
        localDeployQueued: forwardedQueue,
      };
      navigate("../setup/done", { state: next });
    },
    onError: (err: unknown) => {
      setError(describeSecretPersistError(err, "NGC_API_KEY"));
    },
  });

  function handleApply() {
    setError(null);
    if (!value) return;
    applyMutation.mutate(value);
  }

  // [Back] navigates one step in history (returns to the setup-choice
  // screen). If the SME pasted an NVIDIA key on its combined card, it
  // was persisted, so a fresh setup-choice screen renders as Case B
  // (hybrid offer) — they see their hosted setup is intact and can
  // re-decide about local cleanly. The mental model is honest: "go
  // back and adjust."
  function handleBack() {
    navigate(-1);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setValue(e.target.value);
    if (error) setError(null);
  }

  if (shouldAutoSkip) {
    return (
      <SetupTransitionCard
        title="Applying your setup…"
        description="No additional registry key is needed. You'll continue automatically."
        testId="ngc-auto-skip-transition"
      />
    );
  }

  const disabled = applyMutation.isPending || value.length === 0;
  const gpuLabel = env.gpus[0]?.name ?? "your GPU";

  return (
    <div className="flex flex-1 flex-col items-center justify-center p-6">
      <div
        className="glass-card glass-card--elevated flex w-full max-w-[640px] flex-col gap-6 p-8"
        data-testid="ngc-setup-card"
      >
        {/* Confirmation pill — the hosted path arrives here with the
            NVIDIA key in place (freshly pasted or already configured). */}
        <span
          className="flex items-center gap-2"
          style={{ color: "var(--accent-green)" }}
        >
          <Check size={14} />
          <Text kind="body/regular/sm" style={{ color: "inherit" }}>
            NVIDIA key saved
          </Text>
        </span>

        <div className="flex flex-col gap-2">
          <Text kind="title/xl" style={{ color: "var(--text-primary)" }}>
            Want faster local embeddings?
          </Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            Paste an NGC API key and we'll run NeMo Retriever VL on {gpuLabel}.
          </Text>
        </div>

        <div className="flex flex-col gap-2">
          <KeyPasteInput
            name="ngc_api_key_paste"
            ariaLabel="NGC API Key"
            placeholder="Paste your NGC API key"
            value={value}
            onChange={handleChange}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !disabled) handleApply();
            }}
          />
          {error && (
            <Text kind="body/regular/sm" role="alert" className="text-error">
              {error}
            </Text>
          )}
        </div>

        <div className="flex items-center justify-between">
          <KeyPortalLink href={NGC_API_KEY_PORTAL_URL} label="Get an NGC API key" />
          <div className="flex items-center gap-3">
            <Button kind="secondary" onClick={handleBack} data-testid="ngc-back-button">
              <ArrowLeft size={14} /> Back
            </Button>
            <Button
              kind="primary"
              className="nvidia-green-button"
              disabled={disabled}
              onClick={handleApply}
            >
              {applyMutation.isPending ? (
                <>
                  <Spinner aria-label="Saving" /> Saving...
                </>
              ) : (
                <>
                  Use my GPU <ArrowRight size={14} />
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Connection test panel for hosted and self-hosted NIM overrides.
 *
 * Hosted mode with no key configured delegates to ``CredentialInput``
 * (the single NVIDIA-key entry + Test Connection + persistence-hint
 * surface). This panel itself renders the hosted already-configured
 * summary (green check + effective-key probe) and the self-hosted
 * endpoint form.
 *
 * Self-hosted mode collects only a base URL. Self-hosted NIMs are
 * expected to run on a trusted private network or behind an external
 * gateway, so the app attaches no credential — the probe hits
 * ``GET {base_url}/models`` unauthenticated (``auth_mode="none"``).
 */

import { useId, useState } from "react";
import { Button, Spinner, Text } from "@kui/react";
import { useMutation } from "@tanstack/react-query";
import { Check, ShieldCheck, X } from "lucide-react";

import { testConnection, testNvidiaCredential } from "@/api/nim";
import { CredentialInput } from "@/components/CredentialInput";
import type { ConnectionTestResponse } from "@/types/nim";

interface TestVars {
  mode: "hosted" | "self_hosted";
  baseUrl: string;
}

interface ConnectionTestPanelProps {
  mode: "hosted" | "self_hosted";
  apiKeyConfigured: boolean;
  /**
   * Fired when a connection probe returns ``success=true``. The second
   * argument is the credential VALUE that just probed green — for
   * self-hosted it is always the empty string (no credential). The
   * parent (NIMConnectionPage) uses it to dispatch ``POST
   * /v1/secrets:set`` on Continue / Save for the hosted key path.
   */
  onTestSuccess?: (result: ConnectionTestResponse, credential: string) => void;
  /**
   * Fired whenever the SME edits the endpoint field after a previous
   * green test. Lets the parent reset its ``connectionVerified`` state
   * so [Continue] re-disables — closes the stale-success defect where an
   * edited endpoint would otherwise still unlock navigation.
   */
  onCredentialChange?: () => void;
}

/**
 * Non-blocking heuristic: does the base URL path end with a version
 * segment like ``/v1``, ``/v2``, ``/v10``? Used only to surface a gentle
 * hint below the field — it never blocks [Test Connection].
 */
function hasVersionSuffix(rawUrl: string): boolean {
  const trimmed = rawUrl.trim();
  if (!trimmed) return true; // nothing typed yet → no hint
  const path = trimmed.split(/[?#]/)[0].replace(/\/+$/, "");
  return /\/v\d+$/i.test(path);
}

export function ConnectionTestPanel({
  mode,
  apiKeyConfigured,
  onTestSuccess,
  onCredentialChange,
}: ConnectionTestPanelProps) {
  const [baseUrl, setBaseUrl] = useState(
    mode === "hosted" ? "https://integrate.api.nvidia.com/v1" : "",
  );
  const [testResult, setTestResult] = useState<ConnectionTestResponse | null>(null);
  const baseUrlInputId = useId();
  const baseUrlHelpId = `${baseUrlInputId}-help`;

  const mutation = useMutation({
    mutationFn: (vars: TestVars): Promise<ConnectionTestResponse> => {
      if (vars.mode === "hosted") {
        // Real validation: POST /chat/completions. GET /v1/models (the
        // ``probe_kind: "models"`` path) is fully public and succeeds with
        // ANY key, so a bad key would pass and get saved as verified. An
        // empty credential probes the already-configured .env key.
        return testNvidiaCredential(undefined);
      }
      // Self-hosted endpoints run on a trusted network — probe
      // GET {base_url}/models with no credential.
      return testConnection({
        base_url: vars.baseUrl,
        auth_mode: "none",
        probe_kind: "models",
      });
    },
    onSuccess: (result) => {
      setTestResult(result);
      if (result.success && onTestSuccess) {
        onTestSuccess(result, "");
      }
    },
  });

  // Hosted + key-not-yet-configured is exactly the CredentialInput
  // surface (labeled key input, portal link, Test Connection, result,
  // persistence hint) — reuse it instead of hand-rolling a copy.
  // Rendered after the hooks above so the hook order stays stable if
  // ``apiKeyConfigured`` flips while mounted.
  if (mode === "hosted" && !apiKeyConfigured) {
    return (
      <CredentialInput
        kind="nvidia_api_key"
        onTestSuccess={onTestSuccess}
        onCredentialChange={onCredentialChange}
      />
    );
  }

  function handleTest() {
    setTestResult(null);
    mutation.mutate({ mode, baseUrl });
  }

  const showVersionHint =
    mode === "self_hosted" && baseUrl.trim() !== "" && !hasVersionSuffix(baseUrl);

  return (
    <div className="flex max-w-4xl flex-col gap-4">
      {mode === "hosted" && (
        <div
          className="flex items-center gap-2"
          style={{ color: "var(--accent-green)" }}
        >
          <Check size={14} />{" "}
          <Text kind="body/regular/sm" style={{ color: "inherit" }}>
            NVIDIA API key configured
          </Text>
        </div>
      )}

      {mode === "self_hosted" && (
        <div className="glass-inner-panel flex flex-col gap-4 rounded-[14px] p-4">
          <div className="flex flex-col gap-2">
            <label htmlFor={baseUrlInputId} className="flex items-center gap-1">
              <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
                Base URL
              </Text>
              <span aria-hidden="true" className="text-error">
                *
              </span>
            </label>
            <input
              id={baseUrlInputId}
              aria-describedby={baseUrlHelpId}
              type="text"
              className="glass-input w-full px-3 py-3 text-sm"
              placeholder="http://10.0.1.50:8000/v1"
              value={baseUrl}
              onChange={(e) => {
                setBaseUrl(e.target.value);
                if (testResult) {
                  setTestResult(null);
                }
                onCredentialChange?.();
              }}
            />
            <div id={baseUrlHelpId} className="flex flex-col gap-1">
              <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
                Use the OpenAI-compatible base URL and include /v1. It must be reachable
                from the VLM Feedback Loop backend.
              </Text>
              {showVersionHint && (
                <Text kind="body/regular/sm" style={{ color: "var(--accent-yellow)" }}>
                  This URL has no version suffix — most NIM endpoints expect one, such
                  as /v1.
                </Text>
              )}
            </div>
          </div>

          <div className="glass-info flex items-start gap-3 px-4 py-3">
            <ShieldCheck
              aria-hidden="true"
              className="mt-0.5 shrink-0"
              size={18}
              style={{ color: "var(--accent-green)" }}
            />
            <div className="flex flex-col gap-1">
              <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
                Network security
              </Text>
              <Text kind="body/regular/sm" style={{ color: "var(--text-secondary)" }}>
                Self-hosted NIMs should be available only on a trusted private network
                or secured by an external gateway. VLM Feedback Loop does not send a
                credential to this endpoint.
              </Text>
            </div>
          </div>
        </div>
      )}

      {/* Test Connection button */}
      <div className="flex items-center gap-3">
        <Button
          kind="secondary"
          onClick={handleTest}
          disabled={mutation.isPending || (!baseUrl && mode === "self_hosted")}
        >
          {mutation.isPending ? (
            <>
              <Spinner aria-label="Testing" /> Testing...
            </>
          ) : (
            "Test Connection"
          )}
        </Button>

        {mode === "self_hosted" && !baseUrl && (
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            Enter a Base URL to enable testing.
          </Text>
        )}

        {/* Result display — icon and text share a single flex row so long
            error copy stays inline with the ✗/✓ glyph instead of wrapping
            underneath it. Mirrors the working pattern in CredentialInput. */}
        {testResult &&
          (testResult.success ? (
            <span
              className="flex items-center gap-1"
              style={{ color: "var(--accent-green)" }}
            >
              <Check size={14} />
              <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                Connected to {mode === "hosted" ? "NVIDIA hosted NIM" : baseUrl}.
                {testResult.models && testResult.models.length > 0 && (
                  <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
                    {" "}
                    ({testResult.models.length} models)
                  </Text>
                )}
              </Text>
            </span>
          ) : (
            <span className="flex items-center gap-1 text-error">
              <X size={14} />
              <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                {testResult.error || "Connection failed."}
              </Text>
            </span>
          ))}
      </div>
    </div>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Inline credential input for the NIM Connection screen.
 *
 * Two kinds, both with a Test Connection button:
 *   - `nvidia_api_key`: labeled "NVIDIA API Key *" + [Get NVIDIA API Key ->]
 *     link + [Test Connection] probing build.nvidia.com via
 *     ``testNvidiaCredential`` (POST /v1/chat/completions, the actually-
 *     gated path — GET /v1/models is public on build.nvidia.com).
 *   - `ngc_api_key`:    labeled "NGC API Key" + [Get NGC API Key ->] link
 *     + [Test Connection] probing nvcr.io via ``testNgcCredential``.
 *
 * SECURITY: the entered value lives ONLY in React useState.  It is sent as
 * `credential_transient` on the test probe and is NEVER persisted.  To keep
 * the key across restarts, the operator writes it to ~/.vlm_feedback_loop/.env.
 */

import { useState } from "react";
import { Button, Spinner, Text } from "@kui/react";
import { useMutation } from "@tanstack/react-query";
import { Check, X } from "lucide-react";

import { testNgcCredential, testNvidiaCredential } from "@/api/nim";
import { KeyPasteInput } from "@/components/KeyPasteInput";
import { KeyPortalLink } from "@/components/KeyPortalLink";
import {
  NGC_API_KEY_PORTAL_URL,
  NVIDIA_API_KEY_PORTAL_URL,
} from "@/lib/key-portal-urls";
import type { ConnectionTestResponse } from "@/types/nim";

type CredentialKind = "nvidia_api_key" | "ngc_api_key";

interface CredentialInputProps {
  kind: CredentialKind;
  /**
   * Fired when the test probe returns ``success=true``. The second
   * argument is the credential VALUE that just probed green —
   * NIMConnectionPage uses it to dispatch ``POST /v1/secrets:set`` on
   * Continue. Held only in component state, never persisted
   * client-side.
   */
  onTestSuccess?: (result: ConnectionTestResponse, credential: string) => void;
  /**
   * Fired whenever the SME edits the field after a previous green
   * test. Lets the parent reset its ``connectionVerified`` state so
   * [Continue] re-disables instead of acting on a stale success.
   */
  onCredentialChange?: () => void;
}

const COPY: Record<
  CredentialKind,
  {
    label: string;
    placeholder: string;
    linkText: string;
    linkHref: string;
    helper: string;
  }
> = {
  nvidia_api_key: {
    label: "NVIDIA API Key",
    placeholder: "nvapi-...",
    linkText: "Get NVIDIA API Key",
    linkHref: NVIDIA_API_KEY_PORTAL_URL,
    helper: "for hosted Teacher and embeddings",
  },
  ngc_api_key: {
    // Personal API keys issued from org.ngc.nvidia.com do not share the
    // `nvapi-` prefix used by build.nvidia.com keys, and the format has
    // varied across NGC versions. Use a neutral prompt instead of a
    // specific prefix so the placeholder doesn't suggest the wrong shape
    // (which would otherwise be visually identical to the NVIDIA API Key
    // input directly above it).
    label: "NGC API Key",
    placeholder: "Paste your NGC API key",
    linkText: "Get NGC API Key",
    linkHref: NGC_API_KEY_PORTAL_URL,
    helper: "for local NIM container images",
  },
};

export function CredentialInput({
  kind,
  onTestSuccess,
  onCredentialChange,
}: CredentialInputProps) {
  const [value, setValue] = useState("");
  const [testResult, setTestResult] = useState<ConnectionTestResponse | null>(null);
  const [credentialUnderTest, setCredentialUnderTest] = useState("");
  const copy = COPY[kind];

  const mutation = useMutation({
    mutationFn: async (credential: string) =>
      kind === "nvidia_api_key"
        ? testNvidiaCredential(credential)
        : testNgcCredential(credential),
    onSuccess: (result) => {
      setTestResult(result);
      if (result.success && onTestSuccess) {
        onTestSuccess(result, credentialUnderTest);
      }
    },
  });

  function handleTest() {
    setTestResult(null);
    setCredentialUnderTest(value);
    mutation.mutate(value);
  }

  // NVIDIA API key is hard-required (red ``*`` indicator); NGC API key
  // is optional (no indicator — falls back to hosted embeddings or
  // pHash diversity).
  const isRequired = kind === "nvidia_api_key";

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-col gap-1">
        <label className="flex items-center gap-1">
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            {copy.label}
          </Text>
          {isRequired && (
            <span aria-hidden="true" className="text-error">
              *
            </span>
          )}
        </label>
        <KeyPasteInput
          name={kind}
          ariaLabel={copy.label}
          placeholder={copy.placeholder}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            // Reset stale-success state when the SME edits the key
            // after a previous green test.
            if (testResult?.success) {
              setTestResult(null);
            }
            onCredentialChange?.();
          }}
        />
        <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
          {copy.helper}
        </Text>
      </div>

      <div className="flex items-center gap-3">
        <KeyPortalLink href={copy.linkHref} label={copy.linkText} dense />

        <Button
          kind="secondary"
          onClick={handleTest}
          disabled={mutation.isPending || !value}
        >
          {mutation.isPending ? (
            <>
              <Spinner aria-label="Testing" /> Testing...
            </>
          ) : (
            "Test Connection"
          )}
        </Button>

        {testResult && (
          <span className="flex items-center gap-1">
            {testResult.success ? (
              <span
                className="flex items-center gap-1"
                style={{ color: "var(--accent-green)" }}
              >
                <Check size={14} />
                <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                  {kind === "nvidia_api_key"
                    ? "Connected to NVIDIA hosted NIM."
                    : "Authenticated to nvcr.io."}
                </Text>
              </span>
            ) : (
              <span className="flex items-center gap-1 text-error">
                <X size={14} />
                <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                  {testResult.error || "Connection failed."}
                </Text>
              </span>
            )}
          </span>
        )}
      </div>

      <Text kind="label/regular/xs" style={{ color: "var(--text-faint)" }}>
        To keep this key across restarts, add{" "}
        <code>{kind === "nvidia_api_key" ? "NVIDIA_API_KEY" : "NGC_API_KEY"}</code> to{" "}
        <code>~/.vlm_feedback_loop/.env</code>.
      </Text>
    </div>
  );
}

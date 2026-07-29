// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Preflight check result display for local NIM deployment.
 *
 * The missing-prerequisites landing is owned by
 * ``NIMConnectionPage``'s page-level ``MissingPrerequisitesView`` — this
 * panel only reports live preflight results.
 */

import { useEffect } from "react";
import { Button, Spinner, Text } from "@kui/react";
import { useMutation } from "@tanstack/react-query";
import { Check, X } from "lucide-react";

import { runPreflight } from "@/api/nim";
import { NGC_API_KEY_PORTAL_URL } from "@/lib/key-portal-urls";
import type { PreflightResponse } from "@/types/nim";

interface PreflightResultPanelProps {
  projectId: string;
  role: "teacher" | "embedding";
  modelConfigId?: string;
  onSwitchToHosted?: () => void;
}

// Render a backend-returned check_name (snake_case, all lowercase) as a
// human-facing label. Acronyms like GPU / NGC / NIM / CUDA must capitalize;
// everything else becomes sentence case. Unknown check_names fall back to
// a simple space-substitution + leading uppercase so no label ever renders
// as pure lowercase text that looks like a typo.
const CHECK_NAME_LABELS: Record<string, string> = {
  docker: "Docker",
  nvidia_toolkit: "NVIDIA Container Toolkit",
  gpu_memory: "GPU memory",
  ngc_api_key: "NGC API key",
  image_pullable: "NIM image pullable",
  model_profile: "Model profile",
};

function formatCheckName(checkName: string): string {
  const mapped = CHECK_NAME_LABELS[checkName];
  if (mapped) return mapped;
  const spaced = checkName.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

// Map an error message / code to human-readable copy. Unknown
// errors fall back to a generic primary line with the raw detail as
// secondary context for debugging. Keep this pure so it's unit-testable.
function humanizePreflightError(error: string): {
  primary: string;
  secondary?: string;
} {
  const normalized = error.toUpperCase();
  // Order matters: check nvcr.io / unauthorized BEFORE the generic NGC
  // branch, otherwise a 401 from the registry would resolve as "key not
  // configured" (it IS configured — it just lacks the right scope).
  if (
    normalized.includes("NVCR.IO") ||
    normalized.includes("UNAUTHORIZED") ||
    /\b401\b/.test(normalized)
  ) {
    return {
      primary: "NGC key was rejected by nvcr.io.",
      secondary: `Your key needs the NGC Catalog + Private Registry scopes. Regenerate at ${NGC_API_KEY_PORTAL_URL} with both services enabled, then paste the new key here.`,
    };
  }
  if (normalized.includes("GPU_INSUFFICIENT") || normalized.includes("GPU")) {
    return {
      primary: "This machine cannot run this model locally.",
      secondary: "Use hosted NIM or a self-hosted endpoint.",
    };
  }
  if (normalized.includes("DOCKER")) {
    return {
      primary: "Docker is not available on this machine.",
      secondary: "Install Docker and the NVIDIA Container Toolkit, then restart.",
    };
  }
  if (normalized.includes("NGC")) {
    return {
      primary: "NGC API key is not configured.",
      secondary: "Add NGC_API_KEY to ~/.vlm_feedback_loop/.env and restart the app.",
    };
  }
  // Unknown error shape — don't leak raw API JSON to user-facing UI. The raw
  // string is still exposed via the tooltip `title` attribute at the call
  // site and logged to the browser console below.
  console.error("[preflight] unmatched error:", error);
  return {
    primary: "Preflight check failed to run.",
    secondary: "Try again or switch to hosted.",
  };
}

// Per-check diagnostic strings come straight from the backend and can
// embed Docker daemon output — which for image-pull failures includes
// the raw HTML 401 page from nvcr.io. Rendering that verbatim would be
// ugly and confusing. This function strips HTML and pattern-matches the
// known failure modes; everything else flows through truncated to a
// sane single line.
function cleanCheckDiagnostic(
  checkName: string,
  diagnostic: string,
  passed: boolean,
): string {
  if (passed) return diagnostic;
  if (checkName === "image_pullable") {
    const upper = diagnostic.toUpperCase();
    if (
      upper.includes("UNAUTHORIZED") ||
      /\b401\b/.test(upper) ||
      upper.includes("DENIED")
    ) {
      return "NGC key was rejected by nvcr.io. The key needs the NGC Catalog + Private Registry scopes.";
    }
    if (upper.includes("404") || upper.includes("NOT FOUND")) {
      return "Container image not found at nvcr.io. The release tag may be wrong or unpublished.";
    }
  }
  // Generic clean-up: drop any HTML body, drop trailing newlines, keep
  // the first sane line. Truncate hard at 220 chars so a runaway daemon
  // message can't blow out the layout.
  const head = diagnostic.split(/<html|\n/)[0].trim();
  return head.length > 220 ? `${head.slice(0, 220)}…` : head;
}

export function PreflightResultPanel({
  projectId,
  role,
  modelConfigId,
  onSwitchToHosted,
}: PreflightResultPanelProps) {
  const mutation = useMutation({
    mutationFn: () =>
      runPreflight(projectId, {
        role,
        model_config_id: modelConfigId,
      }),
  });

  // Auto-run preflight on mount
  useEffect(() => {
    if (!mutation.data && !mutation.isPending && !mutation.isError) {
      mutation.mutate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Loading state
  if (mutation.isPending) {
    return (
      <div className="flex items-center gap-2 py-4">
        <Spinner aria-label="Running preflight" />
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          Running preflight checks...
        </Text>
      </div>
    );
  }

  // Error state — use human-readable copy. Raw error is
  // preserved in a tooltip `title` for debugging rather than shown in a
  // monospace block, which would read as confusing developer output.
  if (mutation.isError) {
    const errorMessage = mutation.error instanceof Error ? mutation.error.message : "";
    const humanized = humanizePreflightError(errorMessage);
    return (
      <div className="flex flex-col gap-2 py-2">
        <Text
          kind="body/regular/sm"
          style={{ color: "var(--text-primary)" }}
          title={errorMessage || undefined}
        >
          {humanized.primary}
        </Text>
        {humanized.secondary && (
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            {humanized.secondary}
          </Text>
        )}
        <div className="flex items-center gap-2 pt-1">
          <Button kind="secondary" onClick={() => mutation.mutate()}>
            Retry
          </Button>
          {onSwitchToHosted && (
            <Button kind="secondary" onClick={onSwitchToHosted}>
              Switch to Hosted
            </Button>
          )}
        </div>
      </div>
    );
  }

  const result: PreflightResponse | undefined = mutation.data;
  if (!result) return null;

  return (
    <div className="flex flex-col gap-3">
      {/* Overall status */}
      <div className="flex items-center gap-2">
        <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
          Preflight: {result.all_passed ? "passed" : "failed"}
        </Text>
      </div>

      {/* Individual checks */}
      {result.checks.map((check) => (
        <div
          key={check.check_name}
          className="flex items-start gap-2"
          title={check.passed ? undefined : check.diagnostic}
        >
          {check.passed ? (
            <Check size={14} style={{ color: "var(--accent-green)", marginTop: 2 }} />
          ) : (
            <X size={14} className="text-error" style={{ marginTop: 2 }} />
          )}
          <div className="flex flex-col">
            <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
              {formatCheckName(check.check_name)}
            </Text>
            <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
              {cleanCheckDiagnostic(check.check_name, check.diagnostic, check.passed)}
            </Text>
          </div>
        </div>
      ))}

      {/* Port + GPU summary — passed state */}
      {result.all_passed && (result.resolved_port != null || result.gpu_assignment) && (
        <Text
          kind="label/regular/xs"
          style={{ color: "var(--text-muted)", marginTop: 4 }}
        >
          {result.resolved_port != null && `Port: ${result.resolved_port}`}
          {result.resolved_port != null && result.gpu_assignment && "  ·  "}
          {result.gpu_assignment && `GPU: ${result.gpu_assignment}`}
        </Text>
      )}

      {/* Fallback actions for failures */}
      {!result.all_passed && onSwitchToHosted && (
        <div className="pt-2">
          <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
            This machine cannot run this model locally. Use hosted NIM or a self-hosted
            endpoint.
          </Text>
          <div className="mt-2">
            <Button kind="secondary" onClick={onSwitchToHosted}>
              Switch to Hosted
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Read-only Action Request display with Copy to Clipboard.
 *
 * The default mutation calls the generic
 * ``POST /v1/projects/{id}/action_requests:generate`` endpoint; that path
 * is appropriate for ``tao_setup`` / ``nim_setup`` / ``nim_issue`` /
 * ``missing_files`` / ``tao_issue`` / ``student_nim_deploy`` request types.
 *
 * The ``deployment_handoff`` request type is special: the dual readiness
 * gate (``quality_status`` + ``serving_status``) and the Inference
 * Contract parity gate are enforced **only** by the Student-scoped
 * endpoint (``POST /student_models/{id}:deployment_handoff``). Callers
 * that surface a Deployment Handoff Action Request MUST pass an
 * ``mutationFn`` that hits the gated endpoint so 409s actually fire and
 * the rendered content is fully populated. See ``api/students.ts``
 * ``requestDeploymentHandoff``.
 *
 * When the mutation throws ``ApiError(409, …)``, the panel surfaces the
 * error body inline with a one-sentence explanation rather than the
 * generic "Failed to generate" fallback.
 */

import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { Button, Spinner, Text } from "@kui/react";
import { useMutation } from "@tanstack/react-query";
import { Copy, Check, X } from "lucide-react";

import { ApiError } from "@/api/client";
import { generateActionRequest, logActionRequestCopy } from "@/api/nim";
import { useCopyToClipboard } from "@/hooks/useCopyToClipboard";
import type {
  ActionRequestGenerateResponse,
  ActionRequestGenerateRequest,
} from "@/types/nim";

interface ActionRequestPanelProps {
  projectId: string;
  requestType: string;
  /**
   * Optional context forwarded to the generic ``:generate`` endpoint when
   * the default mutation is used. Ignored when ``mutationFn`` is provided.
   */
  context?: ActionRequestGenerateRequest["context"];
  /**
   * Optional override for the data-fetching call. When provided, the
   * panel calls this instead of the generic ``:generate`` endpoint.
   * Use this for ``deployment_handoff`` to dispatch through the gated
   * Student-scoped endpoint.
   */
  mutationFn?: () => Promise<ActionRequestGenerateResponse>;
  onClose?: () => void;
}

/**
 * Section headings that the ``deployment_handoff`` Action Request always
 * emits, in order. The backend renders these as flush-left lines on their own; we
 * recognise them at render time and bold the matching lines so the
 * five-section structure reads as discrete blocks instead of flat
 * prose. Order in this list matches the backend's render order; we don't
 * rely on the order at parse time, only on exact string match.
 */
const HANDOFF_SECTION_HEADINGS: ReadonlyArray<string> = [
  "Checkpoint",
  "NIM Configuration",
  "Model",
  "Evaluation",
  "Training Lineage",
];

/**
 * Render the body of a ``deployment_handoff`` Action Request line by
 * line, bolding any line whose trimmed content exactly matches one of
 * the five canonical section headings. Other lines render as plain
 * text inside the panel's monospace pre-wrap surface. The clipboard
 * payload (``result.rendered_text``) is unaffected — this is a render-
 * only enhancement that does not modify the string the SME pastes into
 * their infrastructure tooling.
 *
 * For other Action Request types (``tao_setup``, ``nim_setup``,
 * ``student_nim_deploy``, etc.) we return the raw string unchanged so
 * the existing flat-text rendering applies.
 */
function renderBodyWithHeadings(body: string, requestType: string): ReactNode {
  if (requestType !== "deployment_handoff") return body;
  const lines = body.split("\n");
  return lines.map((line, idx) => {
    const trimmed = line.trim();
    const isHeading = HANDOFF_SECTION_HEADINGS.includes(trimmed);
    const trailingNewline = idx < lines.length - 1 ? "\n" : "";
    if (isHeading) {
      return (
        <span key={idx}>
          <span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{line}</span>
          {trailingNewline}
        </span>
      );
    }
    return (
      <span key={idx}>
        {line}
        {trailingNewline}
      </span>
    );
  });
}

/**
 * Plain-language explanations for 409 conflict bodies returned by gated
 * Action Request endpoints. Keys match the substring searched on the
 * error body string (see ``services/errors.py::conflict``).
 */
const CONFLICT_EXPLANATIONS: ReadonlyArray<{
  match: string;
  message: string;
}> = [
  {
    match: "quality_status_partial",
    message:
      "Quality validation is partial — the NIM evaluation produced parseable output on most but not all examples. Re-run NIM evaluation to promote to fully validated before requesting production deployment.",
  },
  {
    match: "quality_status_not_validated",
    message:
      "Quality validation has not completed for this Student yet. Run a TAO evaluation first.",
  },
  {
    match: "serving_status_not_validated",
    message:
      "Serving validation has not completed for this Student yet. Benchmark the Student via NIM first.",
  },
  {
    match: "serving_evaluation_run_missing",
    message:
      "No serving evaluation run is recorded for this Student. Run [Benchmark] before requesting deployment.",
  },
  {
    match: "INFERENCE_CONTRACT_MISMATCH",
    message:
      "The training-time and serving-time Inference Contracts disagree. Re-evaluate this Student under the same field-mode contract it was trained for before requesting deployment.",
  },
];

function explainConflict(body: string): string | null {
  for (const { match, message } of CONFLICT_EXPLANATIONS) {
    if (body.includes(match)) return message;
  }
  return null;
}

export function ActionRequestPanel({
  projectId,
  requestType,
  context,
  mutationFn,
  onClose,
}: ActionRequestPanelProps) {
  const { copied, copy } = useCopyToClipboard();
  const [result, setResult] = useState<ActionRequestGenerateResponse | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // The panel usually opens below existing page content, so the generated
  // request (and its Copy CTA) can land under the fold. Scroll it into view
  // once the result arrives. Optional-call guard: jsdom lacks scrollIntoView.
  useEffect(() => {
    if (result) {
      rootRef.current?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
    }
  }, [result]);

  const generateMutation = useMutation({
    mutationFn:
      mutationFn ??
      (() =>
        generateActionRequest(projectId, {
          request_type: requestType,
          context: context ?? null,
        })),
    onSuccess: (data) => setResult(data),
  });

  const logCopyMutation = useMutation({
    mutationFn: () =>
      logActionRequestCopy(projectId, {
        request_type: requestType,
        rendered_text: result?.rendered_text ?? "",
      }),
  });

  // Auto-generate on mount. Must be in useEffect — calling mutate() during
  // render loses the onSuccess state update and leaves the spinner stuck.
  const { mutate: generate } = generateMutation;
  useEffect(() => {
    generate();
  }, [generate]);

  async function handleCopy() {
    if (!result) return;
    // Log the copy only when the clipboard write actually succeeded.
    if (await copy(result.rendered_text)) logCopyMutation.mutate();
  }

  if (generateMutation.isPending) {
    return (
      <div
        className="flex items-center gap-2 py-4"
        data-testid="action-request-loading"
      >
        <Spinner aria-label="Generating request" />
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          Generating request...
        </Text>
      </div>
    );
  }

  if (generateMutation.isError) {
    const error = generateMutation.error;
    if (error instanceof ApiError && error.status === 409) {
      const explanation = explainConflict(error.body);
      return (
        <div
          className="flex flex-col gap-2 max-w-3xl"
          data-testid="action-request-conflict"
        >
          <div
            className="glass-inner-panel rounded-[14px] p-4 text-sm leading-relaxed"
            style={{
              color: "var(--text-secondary)",
              // The 4-px amber inset stripe marks this panel as a warning.
              // Using `boxShadow` (inset, +x) instead of `borderLeft` so the
              // stripe wins over `.glass-inner-panel`'s `border` shorthand
              // without a specificity skirmish — same pattern as
              // TeacherBaselineCard (`boxShadow: "inset 4px 0 0 ..."`).
              boxShadow: "inset 4px 0 0 var(--warning-amber, #f59e0b)",
            }}
          >
            <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
              Cannot generate this request yet
            </Text>
            <div className="mt-2 flex flex-col gap-2">
              <Text kind="body/regular/sm">
                {explanation ??
                  "The backend rejected this request as not yet ready for handoff."}
              </Text>
              <pre
                className="text-xs leading-snug whitespace-pre-wrap break-all m-0"
                style={{
                  color: "var(--text-muted)",
                  fontFamily:
                    'ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace',
                }}
                data-testid="action-request-conflict-detail"
              >
                {error.body}
              </pre>
            </div>
          </div>
          {onClose && (
            <div className="flex items-center gap-3">
              <Button kind="secondary" onClick={onClose}>
                Close
              </Button>
            </div>
          )}
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2 py-4" data-testid="action-request-error">
        <X size={14} className="text-error" />
        <Text kind="body/regular/sm" className="text-error">
          Failed to generate request.
        </Text>
      </div>
    );
  }

  if (!result) return null;

  // The human-readable generated-at timestamp is rendered once, inside
  // the body header — the footer carries only the two action buttons.
  // The backend-rendered body always starts with the
  // title line (e.g. "NIM Setup Request") followed by a blank line; we lift
  // that title into a flex header row so the timestamp can sit right-aligned
  // alongside it without modifying the clipboard payload.
  const [bodyTitle, bodyRest] = splitTitleAndRest(result.rendered_text);

  return (
    <div
      ref={rootRef}
      className="flex flex-col gap-3 max-w-3xl"
      data-testid="action-request-ready"
    >
      <div
        className="glass-inner-panel overflow-auto rounded-[14px] p-4 text-sm leading-relaxed"
        style={{
          color: "var(--text-secondary)",
          // 800 px lets the deployment_handoff payload (5 sections:
          // Checkpoint / NIM Configuration / Model / Evaluation /
          // Training Lineage + Verification footer) fit without scrolling.
          // At 480 px the Verification line silently clips; at 600 px the
          // Training Lineage Preset/Policy/LoRA lines clip at the seeded
          // multi-section length. The 60vh clamp keeps the body plus the
          // Copy/Close row on screen on shorter viewports; `overflow-auto`
          // provides the inner scroll when clamped.
          maxHeight: "min(800px, 60vh)",
          whiteSpace: "pre-wrap",
          // Monospace family aligns the NIM_MODEL_NAME / NIM_MODEL_PROFILE /
          // Backend / NIM release columns and the docker-run backslash
          // continuations cleanly. The Action Request audience is operators
          // who paste this into terminals — monospace is the right idiom.
          fontFamily:
            'ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace',
        }}
      >
        <div className="flex items-baseline justify-between gap-3 mb-3">
          <Text kind="label/bold/sm" style={{ color: "var(--text-primary)" }}>
            {bodyTitle}
          </Text>
          {result.generated_at && (
            <Text kind="label/regular/xs" style={{ color: "var(--text-muted)" }}>
              {formatGeneratedAt(result.generated_at)}
            </Text>
          )}
        </div>
        {renderBodyWithHeadings(bodyRest, result.request_type)}
      </div>

      <div className="flex items-center gap-3">
        <Button kind="primary" className="nvidia-green-button" onClick={handleCopy}>
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
        {onClose && (
          <Button kind="secondary" onClick={onClose}>
            Close
          </Button>
        )}
      </div>
    </div>
  );
}

// Backend-rendered body always begins with the title on its own line followed
// by a blank line. Split it so the title can be lifted into a header row that
// pairs with the timestamp. If the body doesn't follow that
// shape we fall through to rendering the whole text under an empty header.
function splitTitleAndRest(body: string): [string, string] {
  const idx = body.indexOf("\n\n");
  if (idx === -1) return [body.trimEnd(), ""];
  return [body.slice(0, idx).trim(), body.slice(idx + 2)];
}

// The timestamp renders as "2026-04-03 14:22" — a local date + 24-h
// time, no seconds, no timezone.
function formatGeneratedAt(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const date = d.toLocaleDateString("en-CA");
  const time = d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  return `${date} ${time}`;
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Failure display for proposal states:
 *  - schema_invalid: lists Core validation errors
 *  - timeout: deadline message with helper copy
 *  - rate_limited: hosted 429 exhaustion with wait-and-retry copy
 *  - endpoint_error: NIM issue with Action Request
 */

import { useState } from "react";
import { Button, Text } from "@kui/react";
import { AlertTriangle, Clock, Hourglass, WifiOff } from "lucide-react";

import { ActionRequestPanel } from "@/components/ActionRequestPanel";
import { InfoBanner } from "@/components/common/InfoBanner";
import type { InvocationStatus } from "@/types/labeling";

interface ProposalFailureProps {
  status: Exclude<InvocationStatus, "success">;
  validationErrorsCore: string[];
  validationErrorsAux: string[];
  projectId: string;
}

export function ProposalFailure({
  status,
  validationErrorsCore,
  validationErrorsAux,
  projectId,
}: ProposalFailureProps) {
  const [showActionRequest, setShowActionRequest] = useState(false);

  if (status === "schema_invalid") {
    // Heading + body + Core errors all render
    // inside one "Proposal failed" card. Render the errors as
    // children of InfoBanner so the red surface contains them.
    return (
      <InfoBanner
        tone="error"
        icon={AlertTriangle}
        data-testid="proposal-failure-schema"
      >
        <div className="space-y-1">
          <Text kind="label/bold/sm" className="block">
            Proposal failed
          </Text>
          <Text
            kind="body/regular/sm"
            className="block"
            style={{ color: "var(--text-secondary)" }}
          >
            Schema-invalid: the model returned JSON that does not match your Core
            schema.
          </Text>
        </div>

        {validationErrorsCore.length > 0 && (
          <div className="mt-3">
            <Text
              kind="label/bold/sm"
              style={{ color: "var(--text-muted)" }}
              className="mb-1 block"
            >
              Core errors:
            </Text>
            <ul className="list-disc pl-5 space-y-1">
              {validationErrorsCore.map((err, i) => (
                <li key={i}>
                  <Text
                    kind="body/regular/sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {err}
                  </Text>
                </li>
              ))}
            </ul>
          </div>
        )}

        {validationErrorsAux.length > 0 && (
          <div className="mt-3">
            <Text
              kind="label/bold/sm"
              style={{ color: "var(--text-muted)" }}
              className="mb-1 block"
            >
              Aux warnings:
            </Text>
            <ul className="list-disc pl-5 space-y-1">
              {validationErrorsAux.map((err, i) => (
                <li key={i}>
                  <Text
                    kind="body/regular/sm"
                    style={{ color: "var(--warning-amber)" }}
                  >
                    {err}
                  </Text>
                </li>
              ))}
            </ul>
          </div>
        )}
      </InfoBanner>
    );
  }

  if (status === "timeout") {
    // Error tone like the other two terminal failures — all three share
    // the same "Proposal failed" heading and Skip/Retry recovery; the
    // Clock icon + body copy carry the differentiation.
    return (
      <InfoBanner
        tone="error"
        icon={Clock}
        heading="Proposal failed"
        body="Timeout: the model did not respond within the deadline (180s)."
        extra="If this keeps happening, check with your administrator."
        data-testid="proposal-failure-timeout"
      />
    );
  }

  if (status === "rate_limited") {
    // Hosted-NIM 429 exhaustion is a transient shared-quota condition,
    // not an endpoint fault — no [Report NIM Issue] Action Request here;
    // Skip/Retry in the action strip below remain the recovery.
    return (
      <InfoBanner
        tone="error"
        icon={Hourglass}
        heading="Proposal failed"
        body="Rate limited: the hosted endpoint is throttling requests. Wait a moment, then retry."
        extra="If this keeps happening, the shared request quota may be saturated — try again later."
        data-testid="proposal-failure-rate-limited"
      />
    );
  }

  // endpoint_error
  // The [Report NIM Issue] button sits
  // inside the "Proposal failed" card. When the user expands into the
  // full Action Request panel, that panel renders as its own surface as a
  // sibling — it is a richer composite component, not a button.
  return (
    <div className="flex flex-col gap-3" data-testid="proposal-failure-endpoint">
      <InfoBanner tone="error" icon={WifiOff}>
        <div className="space-y-1">
          <Text kind="label/bold/sm" className="block">
            Proposal failed
          </Text>
          <Text
            kind="body/regular/sm"
            className="block"
            style={{ color: "var(--text-secondary)" }}
          >
            Endpoint error: could not reach the NIM endpoint.
          </Text>
        </div>
        {!showActionRequest && (
          <div className="mt-3">
            <Button
              kind="secondary"
              onClick={() => setShowActionRequest(true)}
              data-testid="report-nim-issue-btn"
            >
              Report NIM Issue
            </Button>
          </div>
        )}
      </InfoBanner>

      {showActionRequest && (
        <ActionRequestPanel
          projectId={projectId}
          requestType="nim_issue"
          onClose={() => setShowActionRequest(false)}
        />
      )}
    </div>
  );
}

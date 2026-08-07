// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the ProposalFailure banner — pins that each backend
 * ``invocation_status`` failure value renders its own diagnosis, in
 * particular that ``rate_limited`` (retry-exhausted hosted-NIM 429s)
 * shows wait-and-retry copy instead of falling through to the
 * endpoint-error banner with its [Report NIM Issue] Action Request.
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { ProposalFailure } from "@/components/labeling/ProposalFailure";
import type { InvocationStatus } from "@/types/labeling";

function renderFailure(status: Exclude<InvocationStatus, "success">) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ProposalFailure
        status={status}
        validationErrorsCore={[]}
        validationErrorsAux={[]}
        projectId="test-pid"
      />
    </QueryClientProvider>,
  );
}

describe("ProposalFailure", () => {
  it.each(["schema_invalid", "timeout", "rate_limited", "endpoint_error"] as const)(
    "%s announces the terminal failure",
    (status) => {
      renderFailure(status);

      expect(screen.getByRole("alert")).toHaveTextContent("Proposal failed");
    },
  );

  it("rate_limited renders wait-and-retry copy without the Report NIM Issue CTA", () => {
    // A 429-exhausted hosted invocation is a transient shared-quota
    // condition — telling the SME the endpoint is unreachable (and
    // nudging a nim_issue Action Request) would be wrong diagnostics.
    renderFailure("rate_limited");

    const banner = screen.getByTestId("proposal-failure-rate-limited");
    expect(banner).toHaveTextContent("Proposal failed");
    expect(banner).toHaveTextContent(/rate limit/i);
    expect(banner).toHaveTextContent(/wait a moment, then retry/i);
    expect(screen.queryByTestId("report-nim-issue-btn")).not.toBeInTheDocument();
    expect(screen.queryByTestId("proposal-failure-endpoint")).not.toBeInTheDocument();
  });

  it("endpoint_error keeps the Report NIM Issue CTA", () => {
    renderFailure("endpoint_error");

    expect(screen.getByTestId("proposal-failure-endpoint")).toHaveTextContent(
      /could not reach the NIM endpoint/i,
    );
    expect(screen.getByTestId("report-nim-issue-btn")).toBeInTheDocument();
  });
});

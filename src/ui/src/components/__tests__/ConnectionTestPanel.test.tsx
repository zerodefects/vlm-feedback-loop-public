// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for ConnectionTestPanel — self-hosted mode.
 *
 * The self-hosted form has no Auth Header field (removed as a
 * simplification): self-hosted NIMs run on a trusted network, so the probe
 * is unauthenticated. These pin the behavior that replaced it — the
 * trusted-network disclaimer + Base URL helper render, no credential field
 * exists, the probe carries ``auth_mode="none"`` with no credential, and
 * the missing-``/v1`` hint is non-blocking.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

import { ConnectionTestPanel } from "@/components/ConnectionTestPanel";

const mockTestConnection = vi.fn();

vi.mock("@/api/nim", () => ({
  testConnection: (...args: unknown[]) => mockTestConnection(...args),
  testNvidiaCredential: vi.fn(),
}));

function Wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const URL_PLACEHOLDER = "http://10.0.1.50:8000/v1";

describe("ConnectionTestPanel — self-hosted", () => {
  beforeEach(() => {
    mockTestConnection.mockReset();
  });

  it("renders a labeled endpoint field, security callout, and no auth-header field", () => {
    render(<ConnectionTestPanel mode="self_hosted" apiKeyConfigured={false} />, {
      wrapper: Wrapper,
    });

    const input = screen.getByRole("textbox", { name: /Base URL/i });
    expect(input).toHaveAccessibleDescription(
      /OpenAI-compatible base URL and include \/v1/i,
    );
    expect(
      screen.getByText(
        /available only on a trusted private network or secured by an external gateway/i,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/Network security/i)).toBeInTheDocument();
    expect(screen.getByText(/Enter a Base URL to enable testing/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Test Connection/i })).toBeDisabled();
    // The removed Auth Header field must not reappear in any form.
    expect(screen.queryByText(/Auth Header/i)).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText(/Bearer/i)).not.toBeInTheDocument();
  });

  it("probes GET /models unauthenticated (auth_mode=none, no credential)", async () => {
    mockTestConnection.mockResolvedValue({ success: true, models: ["m1"] });
    const onTestSuccess = vi.fn();
    const user = userEvent.setup();
    render(
      <ConnectionTestPanel
        mode="self_hosted"
        apiKeyConfigured={false}
        onTestSuccess={onTestSuccess}
      />,
      { wrapper: Wrapper },
    );

    await user.type(screen.getByPlaceholderText(URL_PLACEHOLDER), URL_PLACEHOLDER);
    await user.click(screen.getByRole("button", { name: /Test Connection/i }));

    await waitFor(() => expect(mockTestConnection).toHaveBeenCalledTimes(1));
    expect(mockTestConnection).toHaveBeenCalledWith({
      base_url: URL_PLACEHOLDER,
      auth_mode: "none",
      probe_kind: "models",
    });
    await waitFor(() => expect(screen.getByText(/Connected to/i)).toBeInTheDocument());
    expect(onTestSuccess).toHaveBeenCalledWith(
      { success: true, models: ["m1"] },
      "",
      URL_PLACEHOLDER,
    );
  });

  it("can require a real embedding operation instead of a model-list probe", async () => {
    mockTestConnection.mockResolvedValue({ success: true });
    const user = userEvent.setup();
    render(
      <ConnectionTestPanel
        mode="self_hosted"
        apiKeyConfigured={false}
        probeKind="embeddings"
      />,
      { wrapper: Wrapper },
    );

    await user.type(screen.getByPlaceholderText(URL_PLACEHOLDER), URL_PLACEHOLDER);
    await user.click(screen.getByRole("button", { name: /Test Connection/i }));

    await waitFor(() =>
      expect(mockTestConnection).toHaveBeenCalledWith({
        base_url: URL_PLACEHOLDER,
        auth_mode: "none",
        probe_kind: "embeddings",
      }),
    );
  });

  it("shows a non-blocking hint when the base URL lacks a version suffix", async () => {
    const user = userEvent.setup();
    render(<ConnectionTestPanel mode="self_hosted" apiKeyConfigured={false} />, {
      wrapper: Wrapper,
    });

    const input = screen.getByPlaceholderText(URL_PLACEHOLDER);
    await user.type(input, "http://10.0.1.50:8000");
    expect(screen.getByText(/no version suffix/i)).toBeInTheDocument();
    // The hint never blocks — [Test Connection] stays enabled.
    expect(screen.getByRole("button", { name: /Test Connection/i })).toBeEnabled();

    await user.type(input, "/v1");
    expect(screen.queryByText(/no version suffix/i)).not.toBeInTheDocument();
  });
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { EmbeddingStatusLine } from "@/components/ingest/EmbeddingStatusLine";

describe("EmbeddingStatusLine", () => {
  it("names the NVIDIA hosted NIM while hosted embeddings compute", () => {
    render(<EmbeddingStatusLine provider="hosted_nvclip" />);

    expect(screen.getByTestId("embedding-status-line")).toHaveTextContent(
      "Computing CLIP embeddings via NVIDIA hosted NIM…",
    );
  });

  it.each(["self_hosted_nvclip", "local_nvclip"])(
    "names the local embedding NIM for provider %s",
    (provider) => {
      render(<EmbeddingStatusLine provider={provider} />);

      expect(screen.getByTestId("embedding-status-line")).toHaveTextContent(
        "Computing CLIP embeddings via local embedding NIM…",
      );
    },
  );

  it("does not claim embeddings are computing when no provider is active", () => {
    const { container } = render(<EmbeddingStatusLine provider="none" />);

    expect(container).toBeEmptyDOMElement();
  });

  it("keeps first-ingest provider activation neutral instead of claiming unavailable", () => {
    render(
      <EmbeddingStatusLine provider="none" providerSettling showUnavailableNotice />,
    );

    expect(screen.getByTestId("embedding-status-line")).toHaveTextContent(
      "Confirming the configured embedding provider",
    );
    expect(screen.queryByText(/Embeddings unavailable/)).not.toBeInTheDocument();
  });
});

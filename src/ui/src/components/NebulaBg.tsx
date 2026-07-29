// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Nebula animated background — fixed full-viewport particle canvas
 * with green-cyan gradient overlays. Matches the Retail Agentic Commerce
 * and Retail Catalog Enrichment Blueprint background pattern.
 *
 * Three layers (all pointer-events-none, fixed, z-0):
 *   1. Nebula canvas with ambient particle animation
 *   2. Top gradient overlay (green-to-cyan, masked radial from top)
 *   3. Bottom gradient overlay (same colors, masked radial from bottom)
 */

// Vendored @kui-contrib/nebula dist; types resolve from the adjacent index.d.mts.
import { Nebula } from "./nebula/dist/index.mjs";

const GRADIENT = "linear-gradient(80.22deg, #BFF230 1.49%, #7CD7FE 99.95%)";

export function NebulaBg() {
  return (
    <>
      {/* Layer 1: Nebula particle canvas */}
      <div
        className="pointer-events-none"
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          width: "100vw",
          height: "100vh",
          zIndex: 0,
          overflow: "hidden",
        }}
      >
        <div style={{ width: "100%", height: "100%" }}>
          <Nebula variant="ambient" />
        </div>
      </div>

      {/* Layer 2: Top green-cyan gradient overlay */}
      <div
        className="pointer-events-none"
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: "500px",
          background: GRADIENT,
          opacity: 0.12,
          zIndex: 0,
          maskImage:
            "radial-gradient(ellipse 150% 120% at top, black 0%, black 30%, transparent 70%)",
          WebkitMaskImage:
            "radial-gradient(ellipse 150% 120% at top, black 0%, black 30%, transparent 70%)",
        }}
      />

      {/* Layer 3: Bottom green-cyan gradient overlay */}
      <div
        className="pointer-events-none"
        style={{
          position: "fixed",
          bottom: 0,
          left: 0,
          right: 0,
          height: "300px",
          background: GRADIENT,
          opacity: 0.12,
          zIndex: 0,
          maskImage:
            "radial-gradient(ellipse 120% 130% at bottom, black 0%, black 25%, transparent 60%)",
          WebkitMaskImage:
            "radial-gradient(ellipse 120% 130% at bottom, black 0%, black 25%, transparent 60%)",
        }}
      />
    </>
  );
}

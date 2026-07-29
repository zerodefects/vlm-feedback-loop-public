// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { configDefaults, defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    exclude: [...configDefaults.exclude, "e2e/**"],
    // Pool config: stay on vitest's defaults. `pool: "threads"` causes
    // jsdom-contention failures (testing-library + jsdom share hidden
    // state), and capping forks to 2 is stable but slower than the
    // default. The default fork-per-file model is the right tradeoff.
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      // Exclude pure-type files (no runtime), entry points, and visual
      // decoration. Counting them as 0% poisons the headline metric.
      exclude: [
        "src/types/**",
        "src/**/*.d.ts",
        "src/main.tsx",
        "src/components/NebulaBg.tsx",
        // vitest's own defaults that we want to keep
        "**/node_modules/**",
        "**/__tests__/**",
        "**/*.test.{ts,tsx}",
        "**/*.config.{ts,js}",
        "**/dist/**",
        "src/test-setup.ts",
        // Shared test utilities (mocks, typed fixtures) — test infra, not
        // product code.
        "src/test/**",
        "src/vite-env.d.ts",
      ],
    },
  },
});

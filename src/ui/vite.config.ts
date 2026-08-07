// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BACKEND_TARGET = process.env.VITE_BACKEND_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    host: "127.0.0.1",
    proxy: {
      // Proxy /v1/ API routes (including the SSE events path) to the FastAPI
      // backend. http-proxy streams chunked responses as-is, so long-lived
      // SSE connections pass through without extra configuration.
      "/v1": {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
      "/health": {
        target: BACKEND_TARGET,
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 5173,
    host: "127.0.0.1",
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Secrets API — deployment-scoped.
 *
 * Backs ``POST /v1/secrets:set``, the runtime-applicable secret
 * plumbing. The frontend calls this when the
 * SME applies a pasted key from the NIM Configuration screen — either
 * session-only (runtime override only) or persisted (write to
 * ~/.vlm_feedback_loop/.env, then reload Settings).
 */

import { apiFetch } from "@/api/client";

export type SecretName = "NVIDIA_API_KEY" | "NGC_API_KEY" | "TAO_API_KEY";

export interface SecretSetRequest {
  name: SecretName;
  value: string;
  persist: boolean;
}

export interface SecretSetResponse {
  effective: boolean;
  persisted: boolean;
  env_path: string | null;
  allow_persist: boolean;
}

export function setSecret(body: SecretSetRequest): Promise<SecretSetResponse> {
  return apiFetch<SecretSetResponse>("/secrets:set", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

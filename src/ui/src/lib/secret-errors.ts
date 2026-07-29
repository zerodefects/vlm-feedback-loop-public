// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Shared error→message mapping for ``POST /v1/secrets:set`` failures.
 *
 * When the deployment disables UI key persistence
 * (``ALLOW_UI_SECRET_PERSIST=false``), the backend rejects the write
 * with 403 ``ui_secret_persist_disabled``. Every screen that persists a
 * key surfaces the same fix-it hint, parameterized by the secret's env
 * var name.
 */

import type { SecretName } from "@/api/secrets";

export function describeSecretPersistError(
  err: unknown,
  secretName: SecretName,
): string {
  const message = err instanceof Error ? err.message : String(err);
  if (message.includes("ui_secret_persist_disabled") || message.includes("403")) {
    return `This deployment doesn't allow saving keys from the UI. Add ${secretName}=... to ~/.vlm_feedback_loop/.env and restart.`;
  }
  return message;
}

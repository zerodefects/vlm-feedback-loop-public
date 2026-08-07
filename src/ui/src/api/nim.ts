// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * API functions for NIM endpoints, environment assessment, connection
 * testing, local NIM deployment, and Action Requests.
 */

import { ApiError, apiFetch } from "@/api/client";
import type {
  ActionRequestGenerateRequest,
  ActionRequestGenerateResponse,
  ConnectionTestRequest,
  ConnectionTestResponse,
  EnvironmentResponse,
  LocalNimDeployRequest,
  LocalNimDeployResponse,
  LocalNimDeploymentResponse,
  LocalNimDeploymentListResponse,
  LocalNimGpuConflict,
  LocalNimPreflightRequest,
  NimEndpointListResponse,
  PreflightResponse,
  SelfHostedTeacherConfigureRequest,
  SelfHostedTeacherConfigureResponse,
  SelfHostedEmbeddingConfigureRequest,
  EmbeddingDeploymentConfigResponse,
} from "@/types/nim";

export function fetchNimEndpoints(projectId: string): Promise<NimEndpointListResponse> {
  return apiFetch<NimEndpointListResponse>(`/projects/${projectId}/nim_endpoints`);
}

// ── Deployment-scoped ───────────────────────────────────────────────────────

export function fetchEnvironment(
  refreshHardware = false,
): Promise<EnvironmentResponse> {
  const suffix = refreshHardware ? "?refresh_hardware=true" : "";
  return apiFetch<EnvironmentResponse>(`/environment${suffix}`);
}

export function testConnection(
  body: ConnectionTestRequest,
): Promise<ConnectionTestResponse> {
  return apiFetch<ConnectionTestResponse>("/nim/test_connection", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function configureSelfHostedTeacher(
  projectId: string,
  body: SelfHostedTeacherConfigureRequest,
): Promise<SelfHostedTeacherConfigureResponse> {
  return apiFetch<SelfHostedTeacherConfigureResponse>(
    `/projects/${projectId}/nim_endpoints:configure_self_hosted_teacher`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

export function configureSelfHostedEmbedding(
  body: SelfHostedEmbeddingConfigureRequest,
): Promise<EmbeddingDeploymentConfigResponse> {
  return apiFetch<EmbeddingDeploymentConfigResponse>(
    "/embedding_deployment_config:configure_self_hosted",
    { method: "POST", body: JSON.stringify(body) },
  );
}

/**
 * Validate an NGC API key against nvcr.io (registry pull scope).
 *
 * Two modes, both backed by the same endpoint:
 *
 * - Pass a non-empty ``credential`` to probe THAT key. Held in request
 *   memory only; the backend NEVER writes it to any durable store. Use
 *   this BEFORE persisting a freshly-pasted NGC key during setup.
 * - Pass ``undefined`` / ``""`` to probe the currently-effective NGC
 *   key from the backend's runtime secrets layer (the ``.env`` file
 *   plus any in-process overrides). Use this when the SME hasn't
 *   re-pasted a key but a prior session left one in ``.env`` —
 *   ``env.ngc_api_key_configured === true`` doesn't mean the key
 *   actually works.
 *
 * Roundtrip ≈ 300–500 ms in practice (single nvcr.io token exchange).
 */
export function testNgcCredential(
  credential?: string,
  signal?: AbortSignal,
): Promise<ConnectionTestResponse> {
  const body =
    credential !== undefined && credential.length > 0
      ? { credential_transient: credential }
      : {};
  return apiFetch<ConnectionTestResponse>("/nim/test_ngc_credential", {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
}

/**
 * Validate an NVIDIA API key against build.nvidia.com.
 *
 * Parallel of ``testNgcCredential``. Same two-mode shape. Backed by a
 * minimal POST /v1/chat/completions probe — NOT GET /v1/models, which
 * is fully public and succeeds with any key, making it a validation
 * placebo. Roundtrip ≈ 0.2–2 s.
 */
export function testNvidiaCredential(
  credential?: string,
  signal?: AbortSignal,
): Promise<ConnectionTestResponse> {
  const body =
    credential !== undefined && credential.length > 0
      ? { credential_transient: credential }
      : {};
  return apiFetch<ConnectionTestResponse>("/nim/test_nvidia_credential", {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
}

// ── Local NIM deployment ────────────────────────────────────────────────────

export function runPreflight(
  projectId: string,
  body: LocalNimPreflightRequest,
): Promise<PreflightResponse> {
  return apiFetch<PreflightResponse>(`/projects/${projectId}/local_nim/preflight`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function deployLocalNim(
  projectId: string,
  body: LocalNimDeployRequest,
): Promise<LocalNimDeployResponse> {
  return apiFetch<LocalNimDeployResponse>(`/projects/${projectId}/local_nim/deploy`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/**
 * Parse the local-NIM router's structured occupied-GPU detail. FastAPI wraps
 * it under ``detail``; older tests/proxies may expose the object directly, so
 * both shapes are accepted and malformed bodies safely return null.
 */
export function parseLocalNimGpuConflict(error: unknown): LocalNimGpuConflict | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  try {
    const parsed = JSON.parse(error.body) as { detail?: unknown } | unknown;
    const candidate =
      typeof parsed === "object" && parsed !== null && "detail" in parsed
        ? (parsed as { detail: unknown }).detail
        : parsed;
    if (typeof candidate !== "object" || candidate === null) return null;
    const detail = candidate as Record<string, unknown>;
    if (
      !["gpu_occupied", "gpu_exhausted", "resident_starting"].includes(
        String(detail.code),
      )
    ) {
      return null;
    }
    const resident =
      typeof detail.resident === "object" && detail.resident !== null
        ? (detail.resident as LocalNimGpuConflict["resident"])
        : null;
    return {
      code: detail.code as LocalNimGpuConflict["code"],
      message:
        typeof detail.message === "string" ? detail.message : "The GPU is occupied.",
      can_replace: detail.can_replace === true,
      matches_requested_model: detail.matches_requested_model === true,
      resident,
    };
  } catch {
    return null;
  }
}

export function listLocalNimDeployments(
  projectId: string,
): Promise<LocalNimDeploymentListResponse> {
  return apiFetch<LocalNimDeploymentListResponse>(
    `/projects/${projectId}/local_nim/deployments`,
  );
}

export function stopLocalNim(
  projectId: string,
  deploymentId: string,
): Promise<LocalNimDeploymentResponse> {
  return apiFetch<LocalNimDeploymentResponse>(
    `/projects/${projectId}/local_nim/deployments/${deploymentId}:stop`,
    { method: "POST" },
  );
}

// ── Action Requests ─────────────────────────────────────────────────────────

export function generateActionRequest(
  projectId: string,
  body: ActionRequestGenerateRequest,
): Promise<ActionRequestGenerateResponse> {
  return apiFetch<ActionRequestGenerateResponse>(
    `/projects/${projectId}/action_requests:generate`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

export function logActionRequestCopy(
  projectId: string,
  body: { request_type: string; rendered_text: string },
): Promise<{ audit_event_id: string }> {
  return apiFetch<{ audit_event_id: string }>(
    `/projects/${projectId}/action_requests:log_copy`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

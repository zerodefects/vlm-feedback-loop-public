// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * LocalDeployBanner.
 *
 * Non-blocking inline banner that surfaces local NIM deploy state across
 * the labeling screen, edit-guidance, and other project-scoped routes
 * after the FTUE setup-gate confirm has fired background deploys (Cosmos
 * Reason2 Teacher, NeMo Retriever VL embedding NIM). Polls
 * ``GET /v1/projects/{id}/local_nim/deployments`` every 10 s.
 *
 * Two variants, prioritised in this order:
 *
 *   1. **Failure** (red/amber) — when the latest deploy attempt for any
 *      role is in ``"failed"`` status and there's no newer
 *      starting/running deploy for that role superseding it. Friendly
 *      message for ``401 unauthorized`` (almost always a bad NGC key);
 *      generic "Deploy failed" with the short reason for everything
 *      else. Always carries a [Reconfigure NIM] CTA so the SME has a
 *      one-click path to ``/projects/:id/settings/nim`` where they can
 *      re-paste the key and retry.
 *
 *      Two staleness guards (a failure banner must not outlive
 *      its truth): (a) failures whose model config the project's active
 *      role config no longer references are suppressed (backend-computed
 *      ``matches_active_role_config`` — the SME switched Teachers and
 *      labeling works); (b) an explicit [Dismiss] hides that specific
 *      failed deployment id (persisted in localStorage), while any NEW
 *      failure — a new deployment id — surfaces again.
 *
 *   2. **Starting** (green) — when at least one deploy is in
 *      ``"starting"``.
 *
 * Mount point: AppShell. Uses ``useMatch("/projects/:projectId/*")`` to
 * derive the project ID; non-project routes short-circuit to null.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useMatch, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Button, Spinner, Text } from "@kui/react";
import { AlertTriangle } from "lucide-react";

import { listLocalNimDeployments } from "@/api/nim";
import { localNimKeys } from "@/api/query-keys";
import { LOCAL_NIM_POLL_INTERVAL_MS, latestPerRole } from "@/lib/local-nim";
import type { LocalNimDeploymentResponse } from "@/types/nim";

const DISMISSED_FAILURES_KEY = "vlm.dismissedLocalDeployFailures";

function readDismissedIds(): Set<string> {
  try {
    const raw = window.localStorage.getItem(DISMISSED_FAILURES_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return new Set(
      Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [],
    );
  } catch {
    return new Set();
  }
}

function persistDismissedIds(ids: Set<string>): void {
  try {
    window.localStorage.setItem(DISMISSED_FAILURES_KEY, JSON.stringify([...ids]));
  } catch {
    // Storage unavailable (private mode) — dismissal lasts for the session only.
  }
}

function shortModelName(image: string): string {
  // ``nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0`` → ``cosmos-reason2-8b``
  const last = image.split("/").pop() ?? image;
  return last.split(":")[0];
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function isAuthFailure(reason: string | null): boolean {
  if (reason === null) return false;
  return /\b401\b|unauthorized/i.test(reason);
}

function friendlyReason(deployment: LocalNimDeploymentResponse): string {
  const reason = deployment.status_reason ?? "Unknown error";
  if (isAuthFailure(reason)) {
    return "NGC API key was rejected by nvcr.io. Re-paste your NGC key and redeploy.";
  }
  // Generic path: take the prefix before any HTML body the daemon may
  // have appended, then trim to a single sane line.
  const head = reason.split(/<html|\n/)[0].trim();
  return head.length > 160 ? `${head.slice(0, 160)}…` : head;
}

export function LocalDeployBanner(): JSX.Element | null {
  const match = useMatch("/projects/:projectId/*");
  const projectId = match?.params.projectId;
  const navigate = useNavigate();

  const query = useQuery({
    queryKey: localNimKeys.deployments(projectId ?? ""),
    queryFn: () => listLocalNimDeployments(projectId ?? ""),
    enabled: projectId !== undefined,
    refetchInterval: LOCAL_NIM_POLL_INTERVAL_MS,
    // Stale-while-revalidate so a banner that just disappeared doesn't
    // flash back on tab refocus before the next poll lands.
    staleTime: LOCAL_NIM_POLL_INTERVAL_MS / 2,
  });

  const items = query.data?.items;

  const [dismissedIds, setDismissedIds] = useState<Set<string>>(readDismissedIds);

  const { starting, failed } = useMemo(() => {
    const latest = latestPerRole(items ?? []);
    return {
      starting: latest.filter((d) => d.status === "starting"),
      failed: latest.filter(
        (d) =>
          d.status === "failed" &&
          // Staleness guard (a): the active role config moved on — stale evidence.
          d.matches_active_role_config !== false &&
          // Staleness guard (b): explicitly dismissed by the SME.
          !dismissedIds.has(d.local_nim_deployment_id),
      ),
    };
  }, [items, dismissedIds]);

  const dismissFailed = useCallback(() => {
    setDismissedIds((prev) => {
      const next = new Set(prev);
      for (const d of failed) next.add(d.local_nim_deployment_id);
      persistDismissedIds(next);
      return next;
    });
  }, [failed]);

  // Re-render every second so the elapsed timer on the starting variant
  // ticks even between polls. The polling itself stays at 10 s so we
  // don't hammer the local NIM service on a 5-minute deploy.
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (starting.length === 0) return;
    const id = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [starting.length]);
  void tick;

  if (projectId === undefined) return null;

  // Failure variant takes precedence: if a role's most-recent deploy
  // failed, the SME needs to see it even if some other role is still
  // starting. (In practice they tend to fail together — same key.)
  if (failed.length > 0) {
    const firstFailed = failed[0];
    const label =
      failed.length === 1
        ? `${shortModelName(firstFailed.nim_container_image)} deploy failed`
        : `${failed.length} local NIM deploys failed`;
    return (
      <div
        className="flex items-center justify-center gap-3 border-b px-4 py-2"
        style={{
          background: "rgba(239, 68, 68, 0.10)",
          borderColor: "rgba(239, 68, 68, 0.30)",
        }}
        data-testid="local-deploy-banner"
        data-banner-variant="failed"
        role="alert"
      >
        <AlertTriangle
          size={16}
          strokeWidth={2.25}
          style={{ color: "var(--text-error, #ef4444)" }}
        />
        <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
          {label}
        </Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          · {friendlyReason(firstFailed)}
        </Text>
        <Button
          kind="tertiary"
          onClick={() => navigate(`/projects/${projectId}/settings/nim`)}
          data-testid="local-deploy-banner-fix-cta"
        >
          Reconfigure NIM
        </Button>
        <Button
          kind="tertiary"
          onClick={dismissFailed}
          data-testid="local-deploy-banner-dismiss"
        >
          Dismiss
        </Button>
      </div>
    );
  }

  if (starting.length === 0) return null;

  // Anchor the elapsed timer to the earliest still-starting deployment
  // so the user sees the longest wait, not the most recent submission.
  const earliest = starting.reduce<LocalNimDeploymentResponse>(
    (acc, d) => (d.created_at < acc.created_at ? d : acc),
    starting[0],
  );
  const startedMs = new Date(earliest.created_at).getTime();
  const elapsedSec = Math.max(0, (Date.now() - startedMs) / 1000);

  const label =
    starting.length === 1
      ? `${shortModelName(starting[0].nim_container_image)} deploying`
      : `${starting.length} local NIMs deploying`;

  return (
    <div
      className="flex items-center justify-center gap-3 border-b px-4 py-2"
      style={{
        background: "rgba(118, 185, 0, 0.08)",
        borderColor: "rgba(118, 185, 0, 0.25)",
      }}
      data-testid="local-deploy-banner"
      data-banner-variant="starting"
      role="status"
      aria-live="polite"
    >
      <Spinner aria-label="Deploying" />
      <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
        {label}
      </Text>
      <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
        · {formatElapsed(elapsedSec)} elapsed · switchable from the Teacher picker once
        ready
      </Text>
    </div>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * On-mount proactive validation of a persisted credential.
 *
 * The persisted-but-bad key gap: ``env.{ngc,nvidia}_api_key_configured``
 * only signals "a value exists in .env", not "the value works." If we
 * trust the flag and skip validation, the bad key sails past setup and
 * surfaces later as a deploy 401 or first-proposal 403. Fix: when the
 * key is configured (and the caller's gate allows), probe the
 * currently-effective key once on mount. The page renders
 * optimistically; on failure ``onRejected`` fires so the caller can
 * reveal an input / banner. Success makes no visible change.
 *
 * Network errors during the probe fall through silently — click-time
 * probes and downstream error surfaces are the backstop.
 *
 * Shared by ``NIMNvidiaKeyPage`` (the FTU setup-choice screen) and
 * ``NIMConnectionPage`` (the post-onboarding NIM Configuration screen).
 */

import { useEffect, useRef } from "react";

import type { ConnectionTestResponse } from "@/types/nim";

// Persisted-key validation is a best-effort setup guard, not a reason to hold
// the SME on an automatic transition for the full backend inference deadline.
// Healthy credential probes normally settle in 0.2-2 seconds. If the provider
// is degraded, continue with the configured key and let the ordinary
// click-time / first-use error surfaces provide the backstop.
export const PERSISTED_KEY_PROBE_TIMEOUT_MS = 8_000;

interface UsePersistedKeyProbeOptions {
  /**
   * Caller-side gate (e.g. "only probe NVIDIA on render cases B/D").
   * When false, the probe never fires and ``onSettled`` never fires.
   * Defaults to true.
   */
  enabled?: boolean;
  /** Whether the environment reports a persisted key worth probing. */
  configured: boolean;
  /**
   * Probe of the currently-effective key — called with no credential
   * body so the backend pulls from runtime secrets (e.g.
   * ``testNgcCredential`` / ``testNvidiaCredential``).
   */
  probe: (signal?: AbortSignal) => Promise<ConnectionTestResponse>;
  /** Message used when the probe fails without an error string. */
  fallbackError: string;
  /** Fired when the probe reports the persisted key is bad. */
  onRejected: (message: string) => void;
  /**
   * Fired once a verdict exists: probe success, probe failure, probe
   * network error, or nothing-to-probe (``configured === false``).
   * Lets callers gate auto-skips on probe completion.
   */
  onSettled?: () => void;
}

export function usePersistedKeyProbe({
  enabled = true,
  configured,
  probe,
  fallbackError,
  onRejected,
  onSettled,
}: UsePersistedKeyProbeOptions): void {
  // Latest-ref pattern: the effect re-runs only when the gate flips,
  // never because a caller passed a fresh closure — re-probing on every
  // render would hammer the credential endpoints.
  const latest = useRef({ probe, fallbackError, onRejected, onSettled });
  latest.current = { probe, fallbackError, onRejected, onSettled };

  useEffect(() => {
    if (!enabled) return;
    if (!configured) {
      // No key configured ⇒ nothing to probe ⇒ settle immediately so
      // callers waiting on a verdict (e.g. auto-skip gates) unblock.
      latest.current.onSettled?.();
      return;
    }
    let cancelled = false;
    let settled = false;
    let probeTimeoutId: number | undefined;
    const controller = new AbortController();

    const settle = () => {
      if (cancelled || settled) return;
      settled = true;
      if (probeTimeoutId !== undefined) window.clearTimeout(probeTimeoutId);
      latest.current.onSettled?.();
    };
    // React StrictMode replays mount effects in development. Defer the
    // outbound probe one task so the simulated first mount cancels before
    // dispatch; the retained mount then sends exactly one request. This is
    // the same single-dispatch pattern used by ActionRequestPanel.
    const timerId = window.setTimeout(() => {
      probeTimeoutId = window.setTimeout(() => {
        // A slow provider check must not make an auto-configured setup route
        // look permanently stuck. Aborting also releases the browser-side
        // request; the configured credential is re-checked at first use.
        controller.abort();
        settle();
      }, PERSISTED_KEY_PROBE_TIMEOUT_MS);
      void (async () => {
        try {
          const result = await latest.current.probe(controller.signal);
          if (cancelled) return;
          if (!result.success) {
            latest.current.onRejected(result.error ?? latest.current.fallbackError);
          }
        } catch {
          // Quiet: backstops still apply.
        } finally {
          settle();
        }
      })();
    }, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
      if (probeTimeoutId !== undefined) window.clearTimeout(probeTimeoutId);
      controller.abort();
    };
  }, [enabled, configured]);
}

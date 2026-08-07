// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Image Ingestion screen.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Button, Text, Spinner } from "@kui/react";
import { AlertTriangle, ArrowLeft, FolderUp } from "lucide-react";

import { useEnvironmentSetupContext } from "@/pages/setup-context";
import { browseFilesystem, scanDirectory, ingestExamples } from "@/api/filesystem";
import { ApiError, parseApiErrorDetail } from "@/api/client";
import { projectKeys } from "@/api/query-keys";
import { InlineError } from "@/components/InlineError";
import { SetupAutoSkipBanner } from "@/components/common/SetupAutoSkipBanner";
import { FileBrowser } from "@/components/ingest/FileBrowser";
import {
  IngestionProgress,
  type FailureItem,
} from "@/components/ingest/IngestionProgress";
import { IngestionSummary } from "@/components/ingest/IngestionSummary";
import type { BrowseEntry, BrowseResponse } from "@/types/filesystem";

type ScreenState =
  | "loading"
  | "browse"
  | "scanning"
  | "ingesting"
  | "queued"
  | "browse_disabled";

const RECENT_PATHS_KEY = "vlm-ingest-recent-paths";
// Sequential dispatch is deliberate: concurrent
// dispatch causes SQLite write-lock contention — 4 concurrent POSTs
// + the pHash background sweeper + the CLIP embedding worker = 6
// writers fighting one project DB, some hitting
// ``OperationalError: database is locked`` past the 5s busy_timeout
// (500 returned to the SME, batches lost). Sequential client-side
// dispatch eliminates the fight: each ingest POST has the project's
// write lock to itself, then the sweeper / CLIP worker take their
// turns. Wall-clock cost on a 10k-image ingest is ~50s (vs ~25s
// with 4-way concurrency, ~200s with 50-item batches). Counter
// still feels snappy because the small first batch (~250-400ms
// round-trip) unsticks the meter quickly.
const INGEST_BATCH_SIZE = 200;
// First batch is intentionally small so the SME sees the counter unstick
// from 0 — and can move forward to labeling — after just a handful of
// images rather than waiting on a big first batch. A 10-item round-trip is
// ~100–200ms; the pHash sweep then has its first rows to work on almost
// immediately.
const INGEST_FIRST_BATCH_SIZE = 10;

interface IngestBatchOutcome {
  accepted: number;
  skipped: FailureItem[];
  errors: FailureItem[];
  warnings: string[];
}

function getRecentPaths(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_PATHS_KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

function saveRecentPath(path: string) {
  const recent = getRecentPaths().filter((p) => p !== path);
  recent.unshift(path);
  localStorage.setItem(RECENT_PATHS_KEY, JSON.stringify(recent.slice(0, 5)));
}

function parentDirectory(path: string): string {
  const normalized = path.replace(/\/+$/, "");
  const separator = normalized.lastIndexOf("/");
  return separator <= 0 ? "/" : normalized.slice(0, separator);
}

export function ImageIngestPage() {
  const { projectId, project, environment } = useEnvironmentSetupContext();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const embeddingActivationExpected =
    environment.nvidia_api_key_configured ||
    (environment.embedding_deployment?.provider ?? "none") !== "none" ||
    environment.recommended_embedding_mode === "local";

  // Core state
  const [screenState, setScreenState] = useState<ScreenState>("loading");
  const [browseRoot, setBrowseRoot] = useState("");
  const [bundledSamplePath, setBundledSamplePath] = useState<string | null>(null);
  const [embeddingProviderSettling, setEmbeddingProviderSettling] = useState(false);
  const [currentPath, setCurrentPath] = useState("");
  const [pathInput, setPathInput] = useState("");
  const [pathError, setPathError] = useState<string | null>(null);
  // Scan errors keep the directory tree visible (unlike ``pathError``,
  // which replaces the listing) — the SME needs the tree to adjust the
  // selection and retry.
  const [scanError, setScanError] = useState<string | null>(null);

  // Browse state
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set());
  const [isLoadingBrowse, setIsLoadingBrowse] = useState(false);

  // Ingestion state
  const [ingestionTotal, setIngestionTotal] = useState(0);
  const [ingestionProcessed, setIngestionProcessed] = useState(0);
  const [ingestionAccepted, setIngestionAccepted] = useState(0);
  const [ingestionSkipped, setIngestionSkipped] = useState<FailureItem[]>([]);
  const [ingestionErrors, setIngestionErrors] = useState<FailureItem[]>([]);
  const [ingestionWarnings, setIngestionWarnings] = useState<string[]>([]);

  // ── Browse ───────────────────────────────────────────────────────────────

  const doBrowse = useCallback(async (path?: string) => {
    setPathError(null);
    setScanError(null);
    setIsLoadingBrowse(true);
    try {
      const initialResp: BrowseResponse = await browseFilesystem(path);
      let resp = initialResp;
      if (path === undefined) {
        // Keep the backend-selected root as the browser boundary, but start
        // beside the bundled sample so its directory can be selected as one
        // ordinary folder. This replaces the former sample-specific button
        // and avoids opening inside a directory the picker cannot select.
        setBrowseRoot(initialResp.path);
        if (initialResp.bundled_sample_path) {
          const sampleParent = parentDirectory(initialResp.bundled_sample_path);
          if (
            initialResp.path !== sampleParent &&
            initialResp.path !== initialResp.bundled_sample_path
          ) {
            try {
              resp = await browseFilesystem(sampleParent);
            } catch {
              // A custom IMAGE_ROOT can make the repository sample's parent
              // inaccessible. Fall back to the authoritative initial root.
              resp = initialResp;
            }
          }
        }
      }
      setBundledSamplePath(resp.bundled_sample_path);
      setEntries(resp.entries);
      setParentPath(resp.parent);
      setCurrentPath(resp.path);
      setPathInput(resp.path);
      setScreenState("browse");
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 403) {
          const detail = parseApiErrorDetail(err) ?? err.body;
          if (detail.includes("Filesystem browsing is disabled")) {
            setScreenState("browse_disabled");
            return;
          }
          setPathError(detail);
        } else if (err.status === 404) {
          setPathError(parseApiErrorDetail(err) ?? `Directory not found: ${path}`);
        } else {
          setPathError(`Error: ${err.body}`);
        }
        // The inline error replaces the directory listing.
        // Clear stale entries from the previous path so a mismatched tree
        // doesn't render below the error message.
        setEntries([]);
        setParentPath(null);
        setSelectedPaths(new Set());
        setScreenState("browse");
      } else {
        // A non-ApiError (fetch/network failure — apiFetch throws a
        // TypeError, not an ApiError) must not be swallowed: on the initial
        // mount browse it would otherwise leave screenState="loading"
        // forever with a spinner and no retry. Surface a retryable error.
        setPathError(
          `Could not reach the backend to browse ${path ?? "the image root"}. Check the connection and try again.`,
        );
        setEntries([]);
        setParentPath(null);
        setSelectedPaths(new Set());
        setScreenState("browse");
      }
    } finally {
      setIsLoadingBrowse(false);
    }
  }, []);

  // Initial browse on mount
  const didInit = useRef(false);
  useEffect(() => {
    if (!didInit.current) {
      didInit.current = true;
      void doBrowse();
    }
  }, [doBrowse]);

  const handleNavigate = useCallback(
    (path: string) => {
      setSelectedPaths(new Set());
      void doBrowse(path);
    },
    [doBrowse],
  );

  const handlePathSubmit = useCallback(() => {
    void doBrowse(pathInput.trim() || undefined);
  }, [doBrowse, pathInput]);

  const handleToggleSelect = useCallback((path: string) => {
    setSelectedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  // ── Ingestion ────────────────────────────────────────────────────────────

  const handleIngestSelected = useCallback(async () => {
    if (selectedPaths.size === 0 || screenState !== "browse") return;

    // Flip to the scanning state BEFORE the directory scans: a recursive
    // scan of a large tree takes seconds, and without an immediate state
    // change the click appears to do nothing — SMEs press the button
    // again or assume it's broken.
    setScanError(null);
    setScreenState("scanning");

    // For selected items, scan each path and collect images. The backend
    // is the example-key authority (slug + canonical-path hash scheme with
    // key_status dedupe) — individually selected files reuse the scan
    // endpoint on their parent directory instead of re-deriving keys
    // client-side, which collided on same-named files from different
    // folders.
    const allImages: Array<{ example_key: string; storage_ref: string }> = [];
    let scanFailures = 0;
    const selectedFiles = new Set<string>();

    for (const path of selectedPaths) {
      const entry = entries.find((e) => e.path === path);
      if (entry?.type === "directory") {
        try {
          const result = await scanDirectory(path, true, projectId);
          for (const img of result.images) {
            if (img.key_status === "available") {
              allImages.push({
                example_key: img.suggested_example_key,
                storage_ref: img.storage_ref,
              });
            }
          }
        } catch {
          scanFailures++;
        }
      } else if (entry?.type === "file") {
        selectedFiles.add(path);
      }
    }

    // FileBrowser is a flat single-directory listing, so every selected
    // file is a child of currentPath and one non-recursive scan covers
    // them all. Revisit if selection ever persists across navigation or
    // the browser grows inline tree expansion.
    if (selectedFiles.size > 0) {
      try {
        const result = await scanDirectory(currentPath, false, projectId);
        for (const img of result.images) {
          if (selectedFiles.has(img.storage_ref) && img.key_status === "available") {
            allImages.push({
              example_key: img.suggested_example_key,
              storage_ref: img.storage_ref,
            });
          }
        }
      } catch {
        scanFailures++;
      }
    }

    if (allImages.length === 0) {
      // Return to the browse card with an inline error instead of
      // silently doing nothing — an empty result here means either the
      // scans failed (backend unreachable, permissions) or the selection
      // holds no new ingestible images.
      setScanError(
        scanFailures > 0
          ? `Could not scan ${scanFailures} selected folder${scanFailures > 1 ? "s" : ""}. Check the backend connection and try again.`
          : "No ingestible images were found in the selection.",
      );
      setScreenState("browse");
      return;
    }

    setScreenState("ingesting");
    setIngestionTotal(allImages.length);
    setIngestionProcessed(0);
    setIngestionAccepted(0);
    setIngestionSkipped([]);
    setIngestionErrors([]);
    setIngestionWarnings([]);

    let accepted = 0;
    const skipped: FailureItem[] = [];
    const errors: FailureItem[] = [];
    const warnings: string[] = [];

    async function runBatch(
      items: Array<{ example_key: string; storage_ref: string }>,
    ): Promise<IngestBatchOutcome> {
      const out: IngestBatchOutcome = {
        accepted: 0,
        skipped: [],
        errors: [],
        warnings: [],
      };
      try {
        const resp = await ingestExamples(projectId, { examples: items });
        for (const r of resp.results) {
          if (r.status === "created") {
            out.accepted++;
            if (r.warnings.length > 0) out.warnings.push(...r.warnings);
          } else if (r.status === "exists") {
            out.skipped.push({ name: r.example_key, reason: "already exists" });
          } else {
            out.errors.push({
              name: r.example_key,
              reason: r.error ?? "unknown error",
            });
          }
        }
      } catch {
        for (const item of items) {
          out.errors.push({
            name: item.example_key,
            reason: "batch request failed",
          });
        }
      }
      return out;
    }

    // Build batches on a ramp: a small first batch so the SME can move
    // forward almost immediately, then doubling each batch
    // (10 → 20 → 40 → 80 → 160 → 200 …) up to the full INGEST_BATCH_SIZE.
    // Early batches stay snappy and keep the meter climbing; later batches
    // reach the efficient full size once the fast-feedback window is past.
    const batches: Array<Array<{ example_key: string; storage_ref: string }>> = [];
    let batchSize = INGEST_FIRST_BATCH_SIZE;
    for (let i = 0; i < allImages.length; ) {
      batches.push(allImages.slice(i, i + batchSize));
      i += batchSize;
      batchSize = Math.min(batchSize * 2, INGEST_BATCH_SIZE);
    }

    // Dispatch batches one at a time (see the SQLite write-lock note on
    // INGEST_BATCH_SIZE above); per-batch state updates keep the counter
    // ticking as each batch lands.
    let processed = 0;
    for (const batch of batches) {
      const out = await runBatch(batch);
      accepted += out.accepted;
      skipped.push(...out.skipped);
      errors.push(...out.errors);
      warnings.push(...out.warnings);
      processed += batch.length;

      // The SME may leave for Labeling as soon as the first batch lands. The
      // backend's created responses are authoritative: every item submitted
      // by this screen enters Unlabeled. Apply that committed delta to the
      // shared project snapshot before exposing the exit so Scale Up cannot
      // retain its pre-ingest count. The loop survives this page unmount, so
      // later batches keep the still-active ProjectSetupLayout cache current.
      if (out.accepted > 0) {
        queryClient.setQueryData(
          projectKeys.detail(projectId),
          (current: typeof project | undefined) =>
            current
              ? {
                  ...current,
                  counts: {
                    ...current.counts,
                    unlabeled: current.counts.unlabeled + out.accepted,
                  },
                }
              : current,
        );
      }

      setIngestionProcessed(Math.min(processed, allImages.length));
      setIngestionAccepted(accepted);
      setIngestionSkipped([...skipped]);
      setIngestionErrors([...errors]);
      setIngestionWarnings([...warnings]);
    }

    if (project.embedding_provider === "none" && embeddingActivationExpected) {
      setEmbeddingProviderSettling(true);
    }
    setScreenState("queued");
    saveRecentPath(currentPath);
  }, [
    selectedPaths,
    entries,
    projectId,
    currentPath,
    screenState,
    project.embedding_provider,
    embeddingActivationExpected,
    queryClient,
  ]);

  // ── Reset to browse ──────────────────────────────────────────────────────

  const handleAddMore = useCallback(() => {
    setIngestionProcessed(0);
    setIngestionAccepted(0);
    setIngestionSkipped([]);
    setIngestionErrors([]);
    setIngestionWarnings([]);
    setSelectedPaths(new Set());
    setScreenState("browse");
    void doBrowse(currentPath);
  }, [doBrowse, currentPath]);

  // Both "Start labeling" CTAs (mid-ingest and completion summary) exit
  // here. Only the FTUE first run continues to Create Guidance; re-entry
  // from a mature project ([Add Images], queue-empty) must land back in
  // labeling — mirrors ConfirmDefaultsPage's active_guidance_id branch.
  const handleContinue = useCallback(() => {
    navigate(
      project.active_guidance_id
        ? `/projects/${projectId}/labeling`
        : `/projects/${projectId}/create-guidance`,
    );
  }, [navigate, projectId, project.active_guidance_id]);

  // ── Project query refresh on ingestion complete ──────────────────────────
  // The embedding worker's first-ingest re-probe (clip_embedding_service.py)
  // mutates ``project.embedding_provider`` SERVER-side
  // when the user starts ingesting on a project that was created before a
  // hosted NIM key was configured: "none" -> "hosted_nvclip" (or the
  // self-hosted equivalent). React Query's cached project snapshot from
  // the SetupLayout mount is unchanged, so IngestionSummary reads the
  // stale "none" and shows "Embeddings unavailable" even though the
  // backend is happily computing on the hosted NIM. Invalidating on the
  // ``complete`` transition is a final defense-in-depth refresh for the
  // post-probe value before the summary renders. Per-batch count refreshes
  // above already protect the mid-ingest exit path. Idempotent: if nothing
  // changed, the refetch resolves to the same object.
  useEffect(() => {
    if (screenState !== "queued") return;
    void queryClient.invalidateQueries({
      queryKey: projectKeys.detail(projectId),
    });
  }, [screenState, queryClient, projectId]);

  // A first ingest can activate hosted or local embeddings asynchronously. Keep the
  // completion card in a neutral "confirming" state while the backend
  // promotes ``embedding_provider`` instead of briefly telling the SME that
  // embeddings are unavailable. Stop after a bounded 10-second window so a
  // genuinely failed activation can surface the actionable Configure notice.
  useEffect(() => {
    if (
      screenState !== "queued" ||
      project.embedding_provider !== "none" ||
      !embeddingActivationExpected
    ) {
      setEmbeddingProviderSettling(false);
      return;
    }

    let canceled = false;
    let attempts = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const refresh = async () => {
      await queryClient.refetchQueries({
        queryKey: projectKeys.detail(projectId),
        exact: true,
      });
      if (canceled) return;
      attempts += 1;
      if (attempts >= 20) {
        setEmbeddingProviderSettling(false);
        return;
      }
      timer = setTimeout(() => void refresh(), 500);
    };
    void refresh();
    return () => {
      canceled = true;
      if (timer) clearTimeout(timer);
    };
  }, [
    screenState,
    project.embedding_provider,
    embeddingActivationExpected,
    projectId,
    queryClient,
  ]);

  const recentPaths = getRecentPaths().filter(
    (path) =>
      !browseRoot ||
      browseRoot === "/" ||
      path === browseRoot ||
      path.startsWith(`${browseRoot}/`),
  );

  // Selection summary — "3 selected (1 folder, 2 files)". Rendered beside
  // the [Ingest Selected] CTA so the disabled state self-explains ("0
  // selected"). Build the parts list first so the parenthetical assembly
  // can't fall out of sync (an inline ternary composition emits misplaced
  // commas/parens when both folders and files are selected).
  const selectedCount = selectedPaths.size;
  const selectedDirs = entries.filter(
    (e) => e.type === "directory" && selectedPaths.has(e.path),
  ).length;
  const selectedFiles = selectedCount - selectedDirs;
  const selectionParts: string[] = [];
  if (selectedDirs > 0)
    selectionParts.push(`${selectedDirs} folder${selectedDirs > 1 ? "s" : ""}`);
  if (selectedFiles > 0)
    selectionParts.push(`${selectedFiles} file${selectedFiles > 1 ? "s" : ""}`);
  const selectionSummary =
    `${selectedCount} selected` +
    (selectionParts.length > 0 ? ` (${selectionParts.join(", ")})` : "");

  // ── Render ───────────────────────────────────────────────────────────────

  // Hide the page-level header when an inner stage component provides its own
  // heading ("Ingesting Images" / "Images queued for processing") —
  // progress and summary states show only the stage heading, not stacked
  // with "Ingest Images".
  const showPageHeader = screenState !== "ingesting" && screenState !== "queued";
  // The browse-disabled and scanning states retain the
  // "Ingest Images" title but NOT the "Select images to get started…"
  // subhead — the subhead invites an action the SME cannot take, so it
  // actively contradicts the card's message.
  const showPageSubhead =
    showPageHeader && screenState !== "browse_disabled" && screenState !== "scanning";

  // The Path row + file browser tree are visible whenever we're in the
  // browse state. Pressing Enter on the path input drives ``doBrowse`` to
  // navigate the tree (see ``handlePathSubmit``).
  const showPathRow = screenState === "browse";

  return (
    <div className="flex flex-col flex-1 overflow-auto" data-testid="image-ingest-page">
      {/* max-w-4xl (896 px) gives the file tree and the scan preview's
          Images/Skipped lists enough room to breathe without wrapping
          mid-list. Confirm Defaults also uses max-w-4xl so the onboarding
          flow keeps a consistent column width; the NIM Connection screen
          widens to max-w-[1600px] only because it has a multi-panel
          layout. */}
      <div className="p-8 max-w-4xl mx-auto w-full flex flex-col gap-4">
        {/* Acknowledgment banner when the FTUE setup chain auto-skipped
          entirely — reads location.state.setupAutoSkip forwarded by
          ConfirmDefaultsPage; renders nothing otherwise. */}
        <SetupAutoSkipBanner />

        {/* Header — title/md matches the workflow page-header tier used by
          Student Training, Batch, Scale-Up, and the Training Job Monitor,
          and keeps the onboarding step-down from the setup screens
          (title/xl) smooth instead of dropping two tiers to title/sm. */}
        {showPageHeader && (
          <div>
            <Text
              kind="title/md"
              style={{ color: "var(--text-primary)", display: "block" }}
            >
              Ingest Images
            </Text>
            {showPageSubhead && (
              <Text
                kind="body/regular/sm"
                style={{ color: "var(--text-muted)", display: "block", marginTop: 4 }}
              >
                Select images to get started. You can add more at any time from the
                labeling screen.
              </Text>
            )}
          </div>
        )}

        {/* Browse Disabled */}
        {screenState === "browse_disabled" && (
          <div
            className="glass-card glass-card--elevated p-6 text-center"
            data-testid="browse-disabled"
          >
            <AlertTriangle
              size={24}
              style={{ color: "var(--text-muted)", margin: "0 auto 12px" }}
            />
            <Text
              kind="title/md"
              style={{
                color: "var(--text-primary)",
                display: "block",
                marginBottom: 8,
              }}
            >
              Filesystem browsing is disabled.
            </Text>
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              {/* Three-sentence user-facing copy. The backend 403 detail is
                shorter, so we always render this fuller copy instead of
                echoing the backend string. */}
              The backend is network-accessible but no image root is configured. Image
              ingestion requires filesystem access. Ask your administrator to set
              IMAGE_ROOT in the server configuration.
            </Text>
            <div className="mt-4">
              <Button kind="secondary" onClick={() => navigate(-1)}>
                <ArrowLeft size={14} /> Previous screen
              </Button>
            </div>
          </div>
        )}

        {/* Loading */}
        {screenState === "loading" && (
          <div className="flex flex-1 items-center justify-center">
            <Spinner size="large" aria-label="Loading filesystem" />
          </div>
        )}

        {/* The browse and inline-error states share one elevated glass
          card so the ingest surface matches the Confirm Defaults card and
          the adjacent progress/summary card rhythm. Path row + browse tree
          + footer all live inside this card. */}
        {showPathRow && (
          <div className="glass-card glass-card--elevated p-6 flex flex-col gap-4">
            {/* Path row. Press Enter to navigate the directory tree below
              to the typed path. The file browser below is the only entry
              into ingestion — check directories or images, then click
              [Ingest Selected]. */}
            <div className="flex items-center gap-2">
              <Text
                kind="body/regular/sm"
                style={{ color: "var(--text-secondary)", whiteSpace: "nowrap" }}
              >
                Path
              </Text>
              <div className="flex-1">
                <input
                  type="text"
                  aria-label="Path"
                  value={pathInput}
                  onChange={(e) => setPathInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handlePathSubmit();
                  }}
                  className="glass-input w-full px-3 py-1.5 rounded text-sm"
                  aria-invalid={pathError ? "true" : undefined}
                  data-testid="path-input"
                />
              </div>
              <Button
                kind="secondary"
                onClick={() => {
                  if (parentPath) handleNavigate(parentPath);
                }}
                disabled={!parentPath || isLoadingBrowse}
                title={
                  parentPath
                    ? `Open parent folder: ${parentPath}`
                    : browseRoot === "/"
                      ? "You are at the filesystem root."
                      : "You are at the configured image root."
                }
                data-testid="browse-parent-button"
              >
                <FolderUp size={14} /> Up one folder
              </Button>
            </div>

            {!pathError && !parentPath && currentPath === browseRoot && (
              <Text
                kind="body/regular/xs"
                style={{ color: "var(--text-muted)" }}
                data-testid="browse-root-boundary"
              >
                {browseRoot === "/"
                  ? "You’re at the filesystem root."
                  : "You’re at the configured image root. Files above this folder are intentionally unavailable; change IMAGE_ROOT in the server configuration to browse somewhere else."}
              </Text>
            )}

            {/* File Browser */}
            {screenState === "browse" && (
              <>
                {/* Inline path errors */}
                {pathError && <InlineError message={pathError} testId="path-error" />}

                {/* Scan errors — rendered above the tree, which stays
                  visible so the selection can be adjusted. */}
                {scanError && <InlineError message={scanError} testId="scan-error" />}

                {/* Recently used paths — "·" separators
                  between entries. Rendering the middle-dot as muted text
                  between clickable buttons keeps the pattern aligned with
                  the scan-preview "·" separator list above and with the
                  address-bar rhythm of the Retail Blueprint. */}
                {recentPaths.length > 0 && (
                  <div className="flex items-center gap-2 flex-wrap">
                    <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
                      Recent:
                    </Text>
                    {recentPaths.map((p, i) => (
                      <span key={p} className="flex items-center gap-2">
                        {i > 0 && (
                          <span
                            aria-hidden="true"
                            className="text-xs"
                            style={{ color: "var(--text-muted)" }}
                          >
                            &middot;
                          </span>
                        )}
                        <button
                          onClick={() => {
                            setPathInput(p);
                            void doBrowse(p);
                          }}
                          className="text-xs cursor-pointer hover:underline"
                          style={{ color: "var(--text-secondary)" }}
                          type="button"
                        >
                          {p}
                        </button>
                      </span>
                    ))}
                  </div>
                )}

                {/* Directory tree — hidden when an inline path
                  error is shown so the error replaces the listing. */}
                {isLoadingBrowse ? (
                  <div className="flex justify-center py-8">
                    <Spinner size="medium" aria-label="Loading directory" />
                  </div>
                ) : pathError ? null : (
                  <FileBrowser
                    entries={entries}
                    rootPath={browseRoot}
                    currentPath={currentPath}
                    selectedPaths={selectedPaths}
                    onNavigate={handleNavigate}
                    onToggleSelect={handleToggleSelect}
                  />
                )}

                {/* Footer */}
                <div className="flex items-center justify-between mt-2">
                  <Button kind="secondary" onClick={() => navigate(-1)}>
                    <ArrowLeft size={14} /> Previous screen
                  </Button>
                  <div className="flex items-center gap-3">
                    <Text
                      kind="body/regular/sm"
                      style={{
                        color:
                          selectedCount > 0
                            ? "var(--text-secondary)"
                            : "var(--text-muted)",
                      }}
                    >
                      {selectionSummary}
                    </Text>
                    <Button
                      kind="primary"
                      className="nvidia-green-button"
                      onClick={handleIngestSelected}
                      disabled={selectedCount === 0}
                    >
                      Ingest Selected
                    </Button>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* Scanning — immediate feedback between [Ingest Selected] and the
          first ingest batch. The recursive directory scans that expand the
          selection can take seconds on large trees; without this state the
          click looks like a no-op. */}
        {screenState === "scanning" && (
          <div
            className="glass-card glass-card--elevated p-6 flex flex-col items-center gap-3"
            data-testid="ingest-scanning"
          >
            <Spinner size="medium" aria-label="Scanning selection" />
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              Scanning{" "}
              {selectionParts.length > 0 ? selectionParts.join(", ") : "selection"} for
              images…
            </Text>
          </div>
        )}

        {/* Ingestion Progress */}
        {screenState === "ingesting" && (
          <IngestionProgress
            sourcePath={currentPath}
            total={ingestionTotal}
            processed={ingestionProcessed}
            accepted={ingestionAccepted}
            skippedItems={ingestionSkipped}
            errorItems={ingestionErrors}
            embeddingProvider={project.embedding_provider}
            embeddingProviderSettling={embeddingProviderSettling}
            onStartLabeling={handleContinue}
          />
        )}

        {/* Completion Summary */}
        {screenState === "queued" && (
          <IngestionSummary
            totalProcessed={ingestionProcessed}
            accepted={ingestionAccepted}
            skippedItems={ingestionSkipped}
            errorItems={ingestionErrors}
            warnings={ingestionWarnings}
            // Project-cumulative Unlabeled count (refreshed by the
            // projectKeys.detail invalidation on the `queued` transition
            // above), not this batch's accepted count — ingestion is
            // cumulative, so a second "Add More Images" batch would otherwise
            // show only the latest batch under a "Total Unlabeled" label.
            totalUnlabeled={project.counts.unlabeled}
            embeddingProvider={project.embedding_provider}
            embeddingProviderSettling={embeddingProviderSettling}
            isBundledSample={currentPath === bundledSamplePath}
            onAddMore={handleAddMore}
            onContinue={handleContinue}
          />
        )}
      </div>
    </div>
  );
}

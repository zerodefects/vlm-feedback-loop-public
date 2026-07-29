// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Directory tree browser with checkbox selection for the ingestion
 * screen.
 */

import { Text } from "@kui/react";
import { Folder, File, ChevronRight } from "lucide-react";
import { formatBytes } from "@/lib/format-bytes";
import type { BrowseEntry } from "@/types/filesystem";

interface FileBrowserProps {
  entries: BrowseEntry[];
  rootPath: string;
  currentPath: string;
  parentPath: string | null;
  selectedPaths: Set<string>;
  onNavigate: (path: string) => void;
  onToggleSelect: (path: string) => void;
}

export function FileBrowser({
  entries,
  rootPath,
  currentPath,
  parentPath,
  selectedPaths,
  onNavigate,
  onToggleSelect,
}: FileBrowserProps) {
  const dirs = entries.filter((e) => e.type === "directory");
  const files = entries.filter((e) => e.type === "file");

  const normalizedRoot =
    rootPath && rootPath !== "/" ? rootPath.replace(/\/+$/, "") : "/";
  const relativePath =
    normalizedRoot === "/"
      ? currentPath.replace(/^\/+/, "")
      : currentPath.startsWith(`${normalizedRoot}/`)
        ? currentPath.slice(normalizedRoot.length + 1)
        : "";
  const segments = relativePath.split("/").filter(Boolean);
  const atRoot = currentPath === normalizedRoot;
  const rootLabel =
    normalizedRoot === "/"
      ? "/"
      : (normalizedRoot.split("/").filter(Boolean).at(-1) ?? normalizedRoot);

  return (
    <div data-testid="file-browser">
      {/* Breadcrumb — rooted at the backend-selected IMAGE_ROOT. Ancestor
          segments above that boundary are never rendered as navigation. */}
      {!atRoot && (
        <div
          className="flex items-center gap-1 mb-3 text-sm"
          style={{ color: "var(--text-muted)" }}
          data-testid="breadcrumb"
        >
          <button
            className="hover:underline cursor-pointer"
            style={{ color: "var(--text-secondary)" }}
            onClick={() => onNavigate(normalizedRoot)}
            type="button"
          >
            {rootLabel}
          </button>
          {segments.map((seg, i) => {
            const suffix = segments.slice(0, i + 1).join("/");
            const segPath =
              normalizedRoot === "/" ? `/${suffix}` : `${normalizedRoot}/${suffix}`;
            return (
              <span key={segPath} className="flex items-center gap-1">
                <ChevronRight size={12} />
                <button
                  className="hover:underline cursor-pointer"
                  style={{ color: "var(--text-secondary)" }}
                  onClick={() => onNavigate(segPath)}
                  type="button"
                >
                  {seg}
                </button>
              </span>
            );
          })}
        </div>
      )}

      {/* Directory listing */}
      <div
        className="glass-inner-panel rounded-[14px] overflow-auto"
        style={{ maxHeight: 400 }}
      >
        {/* Parent directory link */}
        {parentPath && (
          <button
            className="flex items-center gap-2 w-full px-3 py-2 text-left hover:bg-white/5 cursor-pointer"
            onClick={() => onNavigate(parentPath)}
            type="button"
          >
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              ..
            </Text>
            <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
              (parent)
            </Text>
          </button>
        )}

        {dirs.map((entry) => (
          <div
            key={entry.path}
            className="flex items-center gap-2 px-3 py-2 hover:bg-white/5"
          >
            <input
              type="checkbox"
              checked={selectedPaths.has(entry.path)}
              onChange={() => onToggleSelect(entry.path)}
              className="glass-input"
            />
            <Folder size={16} style={{ color: "var(--text-muted)" }} />
            <button
              className="flex-1 text-left hover:underline cursor-pointer"
              onClick={() => onNavigate(entry.path)}
              type="button"
            >
              <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
                {entry.name}/
              </Text>
            </button>
          </div>
        ))}

        {files.map((entry) => (
          <div
            key={entry.path}
            className="flex items-center gap-2 px-3 py-2 hover:bg-white/5"
          >
            <input
              type="checkbox"
              checked={selectedPaths.has(entry.path)}
              onChange={() => onToggleSelect(entry.path)}
              className="glass-input"
            />
            <File size={16} style={{ color: "var(--text-muted)" }} />
            <Text
              kind="body/regular/sm"
              className="flex-1"
              style={{ color: "var(--text-primary)" }}
            >
              {entry.name}
            </Text>
            <Text kind="body/regular/xs" style={{ color: "var(--text-muted)" }}>
              {formatBytes(entry.size_bytes)}
            </Text>
          </div>
        ))}

        {entries.length === 0 && (
          <div className="px-3 py-4 text-center">
            <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
              Empty directory
            </Text>
          </div>
        )}
      </div>
    </div>
  );
}

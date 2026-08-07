// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * TypeScript mirrors of backend filesystem and ingestion schemas.
 */

// ── Browse ──────────────────────────────────────────────────────────────────

export interface BrowseEntry {
  name: string;
  type: "directory" | "file";
  path: string;
  size_bytes: number | null;
}

export interface BrowseResponse {
  path: string;
  parent: string | null;
  entries: BrowseEntry[];
  bundled_sample_path: string | null;
}

// ── Scan ────────────────────────────────────────────────────────────────────

interface ScanImageEntry {
  storage_ref: string;
  suggested_example_key: string;
  size_bytes: number;
  key_status: "available" | "already_exists_same_path" | "collision_different_path";
  existing_storage_ref: string | null;
}

interface ScanSkippedEntry {
  path: string;
  reason: string;
}

export interface ScanResponse {
  path: string;
  images: ScanImageEntry[];
  skipped: ScanSkippedEntry[];
  total_images: number;
  total_skipped: number;
  total_collisions: number;
}

// ── Ingest ──────────────────────────────────────────────────────────────────

interface IngestItem {
  example_key: string;
  storage_ref: string;
  source_metadata?: Record<string, unknown>;
}

export interface IngestRequest {
  examples: IngestItem[];
}

interface IngestResultItem {
  example_key: string;
  status: "created" | "exists" | "error";
  error: string | null;
  error_code: string | null;
  warnings: string[];
  example: Record<string, unknown> | null;
}

export interface IngestResponse {
  results: IngestResultItem[];
}

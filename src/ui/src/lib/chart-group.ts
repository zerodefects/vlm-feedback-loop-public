// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Stable series-value lookup key for a grouped chart bucket. */
export function chartGroupKey(group: {
  label: string;
  cluster?: string | null;
}): string {
  return `${group.cluster ?? ""}::${group.label}`;
}

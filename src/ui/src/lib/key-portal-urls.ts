// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Canonical NVIDIA key-portal URLs.
 *
 * Every "get a key" link in the setup chain and credential panels points
 * at one of these two portals — keep the strings here so a portal move
 * is a one-line change.
 */

/** Where to generate an `nvapi-` NVIDIA API key (hosted models). */
export const NVIDIA_API_KEY_PORTAL_URL = "https://build.nvidia.com/settings/api-keys";

/** Where to generate a personal NGC API key (container pulls, TAO). */
export const NGC_API_KEY_PORTAL_URL = "https://org.ngc.nvidia.com/setup/api-key";

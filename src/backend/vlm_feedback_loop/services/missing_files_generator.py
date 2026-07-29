# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""missing_files Action Request content generator.

Registered at import time.  Consumed by the labeling screen's
[Report Missing Files] button.
"""

from __future__ import annotations

from typing import Any, cast

from vlm_feedback_loop.services.action_requests import register_generator

_MAX_DISPLAYED_PATHS = 20


def _generate_missing_files(
    project_name: str,
    project_id: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Generate a missing_files Action Request.

    Pre-fills: unresolved storage_ref paths, project name, suggested fix
    (restore files or run bulk path remapping).
    """
    raw: Any = context.get("missing_paths", [])
    missing_paths: list[str] = (
        [str(p) for p in cast("list[Any]", raw)] if isinstance(raw, list) else []
    )

    # Format paths for display, capped at _MAX_DISPLAYED_PATHS
    if missing_paths:
        displayed = missing_paths[:_MAX_DISPLAYED_PATHS]
        path_lines = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(displayed))
        if len(missing_paths) > _MAX_DISPLAYED_PATHS:
            path_lines += f"\n  ...and {len(missing_paths) - _MAX_DISPLAYED_PATHS} more"
    else:
        path_lines = "  (no paths provided)"

    rendered_text = (
        f"Missing Image Files\n"
        f"\n"
        f"Project: {project_name}\n"
        f"\n"
        f"The following image files could not be found at their stored paths:\n"
        f"{path_lines}\n"
        f"\n"
        f"Suggested fixes:\n"
        f"  1. Restore the missing files to their original locations.\n"
        f"  2. If files were moved, use path remapping to update stored\n"
        f"     references:\n"
        f"\n"
        f"     POST /v1/projects/{project_id}/examples:remap_paths\n"
        f'     {{"old_prefix": "<original base path>",\n'
        f'      "new_prefix": "<new base path>",\n'
        f'      "dry_run": true}}\n'
        f"\n"
        f"     Run with dry_run=true first to preview changes,\n"
        f"     then set dry_run=false to apply.\n"
    )

    technical_requirements = {
        "missing_paths": missing_paths[:50],
        "affected_example_count": len(missing_paths),
        "remap_endpoint": f"POST /v1/projects/{project_id}/examples:remap_paths",
    }

    return {
        "technical_requirements": technical_requirements,
        "current_environment": {},
        "rendered_text": rendered_text,
    }


# Register at import time (side-effect import in main.py)
register_generator("missing_files", _generate_missing_files)

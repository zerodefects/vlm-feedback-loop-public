# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem browse and scan service.

Provides two operations:
  - **browse**: list directory entries filtered to supported image formats.
  - **scan**: recursively discover images and generate deterministic example keys.

Security:
  - ``IMAGE_ROOT`` restricts filesystem access to one directory tree.
  - Paths outside the root → 403.
  - Symlinks escaping the root → rejected.
  - Non-loopback bind + unconfigured root → 403 with an explanatory message.
  - Loopback bind + unconfigured root → unrestricted from ``/``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.models.example import Example

# Supported image extensions — canonical set lives in
# ``image_transport``; re-exported here for the browse/ingest call sites.
from vlm_feedback_loop.services.authorized_file import check_image_access_allowed
from vlm_feedback_loop.services.image_transport import SUPPORTED_IMAGE_EXTENSIONS
from vlm_feedback_loop.services.project_service import get_project_engine

logger = logging.getLogger("vlm_feedback_loop.filesystem")

# ── Security helpers ────────────────────────────────────────────────────────


def check_browse_allowed(settings: Settings) -> str | None:
    """Return an error message if browsing is disabled, else ``None``.

    The unrestricted-browse exemption applies ONLY when the backend binds a
    genuine loopback address (the local single-user dev posture). This relies
    on ``BIND_HOST`` reflecting the real bind: the shipped Docker image sets
    ``ENV BIND_HOST=0.0.0.0`` and binds ``${BIND_HOST}`` so the two agree — a
    networked container is correctly treated as non-loopback and must
    configure ``IMAGE_ROOT``.
    """
    if check_image_access_allowed(settings) is not None:
        return (
            "Filesystem browsing is disabled. Configure IMAGE_ROOT to allow "
            "browsing when the backend is network-accessible."
        )
    return None


def _resolve_and_check_root(path: Path, image_root: str | None) -> str | None:
    """Resolve *path* and verify it falls within the configured image root.

    Returns an error message on violation, ``None`` if the path is allowed.
    An unset root means "unrestricted" after the loopback guard has passed.
    """
    if image_root is None:
        return None

    try:
        resolved = path.resolve(strict=False)
        root = Path(image_root).resolve(strict=False)
    except (OSError, ValueError):
        return f"Cannot resolve path: {path}"

    try:
        resolved.relative_to(root)
        return None
    except ValueError:
        return f"Path is outside IMAGE_ROOT: {path}"


def check_path_allowed(path: Path, settings: Settings) -> str | None:
    """Return an error message if *path* is outside ``IMAGE_ROOT``.

    An unset root is unrestricted only for loopback development. The same
    non-loopback guard used by browse and scan applies here so direct ingest
    cannot bypass a disabled browser. This helper is shared by browse,
    ingest, image serving, and path remapping.
    """
    disabled_msg = check_browse_allowed(settings)
    if disabled_msg is not None:
        return disabled_msg

    err = _resolve_and_check_root(path, settings.IMAGE_ROOT)
    if err is not None:
        return err
    if _check_symlink_escape(path, settings.IMAGE_ROOT):
        return f"Path is a symlink escaping IMAGE_ROOT: {path}"
    return None


def _check_symlink_escape(entry_path: Path, image_root: str | None) -> bool:
    """Return ``True`` when a symlink resolves outside ``IMAGE_ROOT``.

    When the root is unset (loopback), symlinks are unrestricted.
    """
    if image_root is None:
        return False
    if not entry_path.is_symlink():
        return False

    try:
        resolved = entry_path.resolve(strict=True)
    except OSError:
        return True  # broken symlink — treat as escape

    root = Path(image_root).resolve(strict=False)
    try:
        resolved.relative_to(root)
        return False
    except ValueError:
        return True


# ── Browse ──────────────────────────────────────────────────────────────────


def browse_directory(
    path: str | None,
    settings: Settings,
    *,
    show_files: bool = True,
    image_formats_only: bool = True,
) -> dict[str, Any]:
    """List entries in *path*, or the deployment's initial image root.

    Returns ``{"path": ..., "parent": ..., "entries": [...]}``.

    Raises:
        PermissionError: when browse is disabled or path is outside IMAGE_ROOT.
        FileNotFoundError: when path does not exist.
    """
    disabled_msg = check_browse_allowed(settings)
    if disabled_msg:
        raise PermissionError(disabled_msg)

    target = Path(path) if path else Path(settings.IMAGE_ROOT or "/")

    if not target.is_absolute():
        raise FileNotFoundError(f"Path must be absolute: {path}")

    root_err = _resolve_and_check_root(target, settings.IMAGE_ROOT)
    if root_err:
        raise PermissionError(root_err)

    if not target.exists():
        raise FileNotFoundError(f"Directory not found: {target}")

    if not target.is_dir():
        raise FileNotFoundError(f"Not a directory: {target}")

    if not os.access(target, os.R_OK):
        raise PermissionError(f"Permission denied: {target}")

    entries: list[dict[str, Any]] = []

    try:
        children = sorted(target.iterdir(), key=lambda p: p.name)
    except PermissionError as exc:
        raise PermissionError(f"Permission denied: {target}") from exc

    for child in children:
        # Skip hidden files/directories
        if child.name.startswith("."):
            continue

        # Skip symlinks escaping the configured image root.
        if _check_symlink_escape(child, settings.IMAGE_ROOT):
            continue

        if child.is_dir():
            entries.append(
                {
                    "name": child.name,
                    "type": "directory",
                    "path": str(child),
                }
            )
        elif show_files and child.is_file():
            ext = child.suffix.lower()
            if image_formats_only and ext not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            entry: dict[str, Any] = {
                "name": child.name,
                "type": "file",
                "path": str(child),
                "size_bytes": child.stat().st_size,
            }
            entries.append(entry)

    # Sort: directories first (alphabetical), then files (alphabetical)
    entries.sort(key=lambda e: (0 if e["type"] == "directory" else 1, e["name"]))

    configured_root = (
        Path(settings.IMAGE_ROOT).resolve(strict=False)
        if settings.IMAGE_ROOT is not None
        else None
    )
    target_resolved = target.resolve(strict=False)
    at_browse_root = target == target.parent or (
        configured_root is not None and target_resolved == configured_root
    )
    parent_str = None if at_browse_root else str(target.parent)
    return {
        "path": str(target),
        "parent": parent_str,
        "entries": entries,
        "bundled_sample_path": _resolve_bundled_sample_path(settings),
    }


def _resolve_bundled_sample_path(settings: Settings) -> str | None:
    """Return the shipped RPS sample when it is readable and in scope.

    Compose mounts the sample itself as ``IMAGE_ROOT=/data/images``. Source
    mode leaves ``IMAGE_ROOT`` unset, so the repository-relative candidate lets
    the UI start beside it without changing the unrestricted ``/`` boundary. A
    configured custom root remains authoritative: repository data outside it
    is never offered.
    """
    candidates: list[Path] = []
    if settings.IMAGE_ROOT is not None:
        configured = Path(settings.IMAGE_ROOT)
        if (configured / "LICENSE.DATA").is_file():
            candidates.append(configured)
    candidates.append(Path(__file__).resolve().parents[4] / "deploy" / "example-images")

    configured_root = (
        Path(settings.IMAGE_ROOT).resolve(strict=False)
        if settings.IMAGE_ROOT is not None
        else None
    )
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if not resolved.is_dir() or not (resolved / "LICENSE.DATA").is_file():
            continue
        if configured_root is not None:
            try:
                resolved.relative_to(configured_root)
            except ValueError:
                continue
        if os.access(resolved, os.R_OK):
            return str(resolved)
    return None


# ── Scan ────────────────────────────────────────────────────────────────────

# Regex to build a readable slug from a relative path
_SLUG_CLEAN = re.compile(r"[^a-zA-Z0-9_-]+")


def _generate_example_key(canonical_path: str) -> str:
    """Generate a deterministic, collision-resistant example key.

    Rule:
      1. Normalize the canonical path to POSIX form.
      2. Build slug from the canonical path without extension, replacing path
         separators with ``_``.
      3. ``hash12 = first_12_hex(sha256(canonical_path_with_extension))``.
      4. ``suggested_example_key = "{slug}--{hash12}"``.
    """
    posix = PurePosixPath(canonical_path)
    stem_path = str(posix.with_suffix(""))  # relative path without extension
    slug = _SLUG_CLEAN.sub("_", stem_path).strip("_")

    # Hash the full relative path (with extension) normalized to POSIX
    hash_input = str(posix)
    hash12 = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:12]

    return f"{slug}--{hash12}"


def _canonical_key_path(file_path: Path, settings: Settings) -> str:
    """Return the scan-root-independent path identity used for example keys.

    A configured ``IMAGE_ROOT`` is the stable deployment boundary, so keys use
    paths relative to it. Loopback development may intentionally omit that
    boundary; there the normalized absolute path is the only identity that
    remains stable when the same file is scanned from different directories.
    """
    resolved_file = file_path.resolve(strict=False)
    if settings.IMAGE_ROOT is not None:
        resolved_root = Path(settings.IMAGE_ROOT).resolve(strict=False)
        return str(PurePosixPath(resolved_file.relative_to(resolved_root)))
    return str(PurePosixPath(resolved_file))


def scan_directory(
    path: str,
    settings: Settings,
    *,
    recursive: bool = True,
    project_id: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Recursively discover supported images and generate example keys.

    Returns ``{"path": ..., "images": [...], "skipped": [...],
    "total_images": N, "total_skipped": N, "total_collisions": N}``.

    Raises:
        PermissionError: when browse is disabled or path is outside IMAGE_ROOT.
        FileNotFoundError: when path does not exist.
    """
    disabled_msg = check_browse_allowed(settings)
    if disabled_msg:
        raise PermissionError(disabled_msg)

    target = Path(path)

    if not target.is_absolute():
        raise FileNotFoundError(f"Path must be absolute: {path}")

    root_err = _resolve_and_check_root(target, settings.IMAGE_ROOT)
    if root_err:
        raise PermissionError(root_err)

    if not target.exists():
        raise FileNotFoundError(f"Directory not found: {target}")

    if not target.is_dir():
        raise FileNotFoundError(f"Not a directory: {target}")

    if not os.access(target, os.R_OK):
        raise PermissionError(f"Permission denied: {target}")

    # Collect all files
    images: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    walker = target.rglob("*") if recursive else target.iterdir()

    file_paths: list[Path] = []
    for child in walker:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            continue
        if _check_symlink_escape(child, settings.IMAGE_ROOT):
            continue
        if child.is_file():
            file_paths.append(child)

    # Sort for deterministic ordering
    file_paths.sort()

    # Load existing keys for collision checking
    existing_keys: dict[str, str] = {}  # example_key → storage_ref
    existing_paths: dict[str, str] = {}  # storage_ref → example_key
    if project_id and workspace_root:
        engine = get_project_engine(project_id, workspace_root)
        if engine is not None:
            with Session(engine) as session:
                rows = session.execute(
                    select(Example.example_key, Example.storage_ref).where(
                        Example.project_id == project_id
                    )
                ).all()
                existing_keys = {row[0]: row[1] for row in rows}
                existing_paths = {row[1]: row[0] for row in rows}

    total_collisions = 0

    for fp in file_paths:
        ext = fp.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            skipped.append({"path": str(fp), "reason": "unsupported_format"})
            continue

        canonical_key_path = _canonical_key_path(fp, settings)
        suggested_key = _generate_example_key(canonical_key_path)
        storage_ref = str(fp)
        size_bytes = fp.stat().st_size

        # Collision checking
        key_status = "available"
        existing_storage_ref: str | None = None

        if project_id and storage_ref in existing_paths:
            # Existing projects may contain keys generated by an older scan
            # root. Path identity still wins: never offer the same source
            # image for ingestion under a second key.
            suggested_key = existing_paths[storage_ref]
            key_status = "already_exists_same_path"
        elif project_id and suggested_key in existing_keys:
            if existing_keys[suggested_key] == storage_ref:
                key_status = "already_exists_same_path"
            else:
                key_status = "collision_different_path"
                existing_storage_ref = existing_keys[suggested_key]
                total_collisions += 1

        img_entry: dict[str, Any] = {
            "storage_ref": storage_ref,
            "suggested_example_key": suggested_key,
            "size_bytes": size_bytes,
            "key_status": key_status,
            "existing_storage_ref": existing_storage_ref,
        }

        images.append(img_entry)

    return {
        "path": str(target),
        "images": images,
        "skipped": skipped,
        "total_images": len(images),
        "total_skipped": len(skipped),
        "total_collisions": total_collisions,
    }

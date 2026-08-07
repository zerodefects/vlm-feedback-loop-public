# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization-bound reads for persisted file references.

Persisted paths are references, not durable filesystem grants. This module
resolves the current grant, then opens the canonical target through a rooted
``openat`` walk that refuses symlinks in every component. Callers consume the
resulting descriptor instead of reopening the mutable pathname.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:
    from types import TracebackType

    from vlm_feedback_loop.config import Settings

_LOOPBACK_ADDRESSES = frozenset({"127.0.0.1", "::1", "localhost"})


def check_image_access_allowed(settings: Settings) -> str | None:
    """Return why persisted-image access is disabled, or ``None``."""
    if settings.IMAGE_ROOT is None and settings.BIND_HOST not in _LOOPBACK_ADDRESSES:
        return (
            "Filesystem image access is disabled. Configure IMAGE_ROOT when "
            "the backend is network-accessible."
        )
    return None


@dataclass
class OpenedRegularFile:
    """One owned descriptor bound to an authorized regular file inode."""

    fd: int
    canonical_path: Path
    stat_result: os.stat_result
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            os.close(self.fd)
            self._closed = True

    def open_binary(self) -> BinaryIO:
        """Return an independently closable binary stream for this inode."""
        if self._closed:
            raise ValueError("Authorized file descriptor is closed")
        return os.fdopen(os.dup(self.fd), "rb")

    def __enter__(self) -> OpenedRegularFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _resolve_beneath(path: Path, root: Path) -> tuple[Path, Path]:
    """Resolve an existing regular-file candidate beneath an existing root."""
    try:
        canonical_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PermissionError(
            f"Authorized file root cannot be resolved: {root}"
        ) from exc

    # Check containment before an existence-revealing strict resolution. An
    # out-of-root reference must remain a policy denial whether or not its
    # target happens to exist on this host.
    try:
        candidate = path.resolve(strict=False)
        candidate.relative_to(canonical_root)
    except ValueError as exc:
        raise PermissionError(f"Path is outside authorized root: {path}") from exc
    except (OSError, RuntimeError) as exc:
        raise PermissionError(f"File path cannot be resolved safely: {path}") from exc

    try:
        canonical_target = path.resolve(strict=True)
        canonical_target.relative_to(canonical_root)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {path}") from exc
    except ValueError as exc:
        raise PermissionError(f"Path is outside authorized root: {path}") from exc
    except (OSError, RuntimeError) as exc:
        raise PermissionError(f"File path cannot be resolved safely: {path}") from exc
    return canonical_target, canonical_root


def _secure_open_flags(*, directory: bool) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory_flag is None or os.open not in os.supports_dir_fd:
        raise PermissionError(
            "Secure filesystem reads require POSIX openat, O_NOFOLLOW, and O_DIRECTORY"
        )
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= directory_flag
    else:
        # Opening a FIFO or device with plain O_RDONLY can block before the
        # fstat check below has a chance to reject it. O_NONBLOCK is inert for
        # regular files and lets the type check fail safely for special files.
        flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _open_canonical_regular_file(
    target: Path,
    root: Path,
    *,
    display_path: Path,
) -> OpenedRegularFile:
    """Open a canonical target without following any mutable path component."""
    relative = target.relative_to(root)
    if not relative.parts:
        raise FileNotFoundError(f"Path is not a regular file: {display_path}")
    if root.anchor != os.sep:
        raise PermissionError("Secure filesystem reads require an absolute POSIX root")

    directory_flags = _secure_open_flags(directory=True)
    file_flags = _secure_open_flags(directory=False)
    with ExitStack() as stack:
        try:
            parent_fd = os.open(os.sep, directory_flags)
            stack.callback(os.close, parent_fd)
            for part in root.parts[1:]:
                child_fd = os.open(
                    part,
                    directory_flags,
                    dir_fd=parent_fd,
                )
                stack.callback(os.close, child_fd)
                parent_fd = child_fd
            for part in relative.parts[:-1]:
                child_fd = os.open(
                    part,
                    directory_flags,
                    dir_fd=parent_fd,
                )
                stack.callback(os.close, child_fd)
                parent_fd = child_fd
            file_fd = os.open(relative.parts[-1], file_flags, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
                raise FileNotFoundError(
                    f"File not found or changed during authorization: {display_path}"
                ) from exc
            if exc.errno in {errno.ENODEV, errno.ENXIO}:
                raise FileNotFoundError(
                    f"Path is not a regular file: {display_path}"
                ) from exc
            raise PermissionError(
                f"File path changed during authorization or is unreadable: {display_path}"
            ) from exc

        try:
            stat_result = os.fstat(file_fd)
            if not stat.S_ISREG(stat_result.st_mode):
                raise FileNotFoundError(f"Path is not a regular file: {display_path}")
            return OpenedRegularFile(
                fd=file_fd,
                canonical_path=target,
                stat_result=stat_result,
            )
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(file_fd)
            raise


def open_regular_file_beneath(path: Path, root: Path) -> OpenedRegularFile:
    """Authorize and open one regular file beneath a backend-owned root."""
    target, canonical_root = _resolve_beneath(path, root)
    return _open_canonical_regular_file(
        target,
        canonical_root,
        display_path=path,
    )


def open_authorized_image(path: str | Path, settings: Settings) -> OpenedRegularFile:
    """Open a persisted image reference under the current ``IMAGE_ROOT`` policy."""
    disabled_error = check_image_access_allowed(settings)
    if disabled_error is not None:
        raise PermissionError(disabled_error)

    candidate = Path(path)
    if not candidate.is_absolute():
        raise PermissionError(f"Persisted image path must be absolute: {candidate}")
    root = (
        Path(settings.IMAGE_ROOT) if settings.IMAGE_ROOT is not None else Path(os.sep)
    )
    try:
        return open_regular_file_beneath(candidate, root)
    except PermissionError as exc:
        if "outside authorized root" in str(exc):
            raise PermissionError(f"Path is outside IMAGE_ROOT: {candidate}") from exc
        raise


__all__ = [
    "OpenedRegularFile",
    "check_image_access_allowed",
    "open_authorized_image",
    "open_regular_file_beneath",
]

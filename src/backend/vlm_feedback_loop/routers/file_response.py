# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starlette file responses bound to an already-authorized descriptor."""

from __future__ import annotations

from pathlib import Path

from starlette.responses import FileResponse
from starlette.types import Receive, Scope, Send

from vlm_feedback_loop.services.authorized_file import OpenedRegularFile


class FileDescriptorResponse(FileResponse):
    """Stream an authorized inode while preserving FileResponse semantics."""

    def __init__(
        self,
        opened_file: OpenedRegularFile,
        *,
        media_type: str,
        filename: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        descriptor_path = Path(f"/proc/self/fd/{opened_file.fd}")
        if not descriptor_path.parent.is_dir():
            opened_file.close()
            raise RuntimeError("Descriptor-backed responses require Linux procfs")
        self._opened_file = opened_file
        try:
            super().__init__(
                path=descriptor_path,
                media_type=media_type,
                filename=filename,
                headers=headers,
                stat_result=opened_file.stat_result,
            )
        except BaseException:
            opened_file.close()
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Disable the optional path-send extension: the ASGI server could open
        # the procfs path only after this response's finally block closes the
        # descriptor. Starlette's normal streaming/range paths open it while
        # the descriptor is still owned here.
        safe_scope: Scope = dict(scope)
        extensions = dict(scope.get("extensions", {}))
        extensions.pop("http.response.pathsend", None)
        safe_scope["extensions"] = extensions
        try:
            await super().__call__(safe_scope, receive, send)
        finally:
            self._opened_file.close()


__all__ = ["FileDescriptorResponse"]

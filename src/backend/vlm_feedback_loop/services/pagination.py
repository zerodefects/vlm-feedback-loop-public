# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared keyset (seek) pagination cursor helpers.

A cursor encodes the ``(timestamp, id)`` of the last row on a page. The
``after_position`` predicate then selects rows strictly *after* that position
under a ``(timestamp DESC, id DESC)`` ordering; ``after_position_asc`` is the
variant for lists whose same-timestamp tie-break is ``id ASC``.

Comparing the full ``(timestamp, id)`` tuple — not the timestamp alone — is
what keeps same-timestamp rows from being skipped or duplicated across pages.
A bare ``created_at < cursor`` filter (the bug this consolidates away) drops
the boundary row and every row sharing its timestamp; keying only on a random
UUID while ordering by timestamp is incoherent entirely.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from sqlalchemy import ColumnElement, and_, or_


class InvalidCursorError(ValueError):
    """Raised when a pagination cursor is malformed (→ HTTP 400, not 500)."""


def encode_cursor(ts: str | None, id_: str) -> str:
    """Encode a ``(timestamp, id)`` position as an opaque base64 cursor."""
    return base64.urlsafe_b64encode(json.dumps([ts, id_]).encode()).decode()


def decode_cursor(cursor: str) -> tuple[str | None, str]:
    """Decode a cursor to ``(timestamp, id)``; raise on any malformed input."""
    try:
        ts, id_ = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
    except (ValueError, TypeError) as exc:
        # base64 (binascii.Error is a ValueError), UTF-8, JSON, or unpack
        # failure — a client-supplied bad cursor is a 400, not a 500.
        raise InvalidCursorError(f"invalid pagination cursor: {cursor!r}") from exc
    return ts, id_


def after_position(
    ts_col: Any,
    id_col: Any,
    cursor_ts: str | None,
    cursor_id: str,
) -> ColumnElement[bool]:
    """Predicate: rows strictly after ``(cursor_ts, cursor_id)``.

    Assumes the query orders by ``ts_col DESC, id_col DESC`` — the ordering
    every paginated list in this app uses. "After" therefore means an older
    timestamp, or the same timestamp with a smaller id.
    """
    return or_(
        ts_col < cursor_ts,
        and_(ts_col == cursor_ts, id_col < cursor_id),
    )


def after_position_asc(
    ts_col: Any,
    id_col: Any,
    cursor_ts: str | None,
    cursor_id: str,
) -> ColumnElement[bool]:
    """Predicate: rows strictly after ``(cursor_ts, cursor_id)``, ASC tie-break.

    For queries ordered by ``ts_col DESC, id_col ASC`` — newest-first lists
    whose same-timestamp tie-break is ascending id. "After" therefore means
    an older timestamp, or the same timestamp with a *larger* id.
    """
    return or_(
        ts_col < cursor_ts,
        and_(ts_col == cursor_ts, id_col > cursor_id),
    )

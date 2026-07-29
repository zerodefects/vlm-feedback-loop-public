# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared keyset-pagination cursor helpers.

The DESC tie-break variant (``after_position``) is pinned through its consumer
in ``test_student_model_service.py::test_keyset_pagination_no_drops_or_dupes_
same_timestamp`` — no duplicate here. These tests cover what no consumer test
reaches: the codec's malformed-input contract, the ``None``-timestamp and
URL-safety guarantees, and the ASC tie-break variant (``after_position_asc``),
whose same-timestamp branch no tao-job or training-suite test exercises.
"""

from __future__ import annotations

import base64
from urllib.parse import parse_qsl

import pytest
from sqlalchemy import Column, MetaData, String, Table, create_engine, select

from vlm_feedback_loop.services.pagination import (
    InvalidCursorError,
    after_position_asc,
    decode_cursor,
    encode_cursor,
)


class TestCursorCodec:
    """The cursor is an exact, URL-transparent round-trip of (timestamp, id)."""

    @pytest.mark.parametrize(
        ("ts", "id_"),
        [
            ("2026-05-01T10:00:00Z", "run-001"),
            # None timestamp is real: the examples list encodes it for rows in
            # the null-verified_at section and branches on it after decode.
            (None, "example-key-9"),
        ],
    )
    def test_round_trip_preserves_timestamp_and_id(self, ts, id_):
        """decode(encode(ts, id)) returns exactly what was encoded — a None
        timestamp comes back as None, not a "null"/"None" string."""
        assert decode_cursor(encode_cursor(ts, id_)) == (ts, id_)

    def test_cursor_survives_a_query_string_verbatim(self):
        """A cursor pasted into a URL query string unencoded (curl, browser)
        must arrive intact — the codec's alphabet may not contain characters
        that query-string parsing mutates, such as '+' (decoded as a space)."""
        ts, id_ = "2026-05-01T10:00:00Z", "key>2"
        # Fixture guard: this payload is adversarial — standard (non-urlsafe)
        # base64 of it does emit '+' or '/', so a codec regression is visible.
        std = base64.b64encode(f'["{ts}", "{id_}"]'.encode()).decode()
        assert "+" in std or "/" in std

        cursor = encode_cursor(ts, id_)
        ((_, received),) = parse_qsl(f"cursor={cursor}")
        assert received == cursor
        assert decode_cursor(received) == (ts, id_)


class TestDecodeCursorRejectsMalformed:
    """Every malformed-cursor class raises InvalidCursorError — the exact type
    the app-level 400 handler and the tao-job / training-suite service catches
    key on. Any other exception type escapes those catches and becomes a 500.
    Only the not-base64 class is covered by the per-router garbage-cursor
    tests; the rest would 500 silently if the except clause were narrowed.
    """

    @pytest.mark.parametrize(
        "cursor",
        [
            pytest.param("", id="empty-string"),
            pytest.param("!!!not-base64!!!", id="not-base64"),
            pytest.param(
                base64.urlsafe_b64encode(b"\xff\xfe\xfd").decode(),
                id="not-utf8",
            ),
            pytest.param(
                base64.urlsafe_b64encode(b"not json at all").decode(),
                id="not-json",
            ),
            pytest.param(
                base64.urlsafe_b64encode(b"42").decode(),
                id="json-not-unpackable",
            ),
            pytest.param(
                base64.urlsafe_b64encode(b'["ts", "id", "extra"]').decode(),
                id="json-wrong-arity",
            ),
        ],
    )
    def test_malformed_cursor_raises_invalid_cursor_error(self, cursor):
        with pytest.raises(InvalidCursorError):
            decode_cursor(cursor)


class TestAfterPositionAsc:
    """Paging with the ASC tie-break variant (tao jobs, training suites) must
    visit every row exactly once when a same-timestamp run straddles a page
    boundary — the case where 'after' means *larger* id, the opposite of
    ``after_position``. A copy-paste 'fix' of the comparison direction, or a
    bare timestamp filter, silently drops the rest of the run.
    """

    def test_no_drops_or_dupes_across_same_timestamp_boundary(self):
        engine = create_engine("sqlite://")
        metadata = MetaData()
        table = Table(
            "rows",
            metadata,
            Column("ts", String, nullable=False),
            Column("id", String, primary_key=True),
        )
        metadata.create_all(engine)

        # Three rows share 09:00:00; the page boundary (limit=2) falls after
        # the first of them, so page 2 exists only if the tie-break is right.
        rows = [
            ("2026-05-01T10:00:00Z", "m"),
            ("2026-05-01T09:00:00Z", "a"),
            ("2026-05-01T09:00:00Z", "b"),
            ("2026-05-01T09:00:00Z", "c"),
            ("2026-05-01T08:00:00Z", "z"),
        ]
        with engine.begin() as conn:
            conn.execute(table.insert(), [{"ts": ts, "id": id_} for ts, id_ in rows])

        limit = 2
        seen: list[tuple[str, str]] = []
        cursor: str | None = None
        for _ in range(10):  # safety bound
            stmt = select(table.c.ts, table.c.id).order_by(
                table.c.ts.desc(), table.c.id.asc()
            )
            if cursor is not None:
                cur_ts, cur_id = decode_cursor(cursor)
                stmt = stmt.where(
                    after_position_asc(table.c.ts, table.c.id, cur_ts, cur_id)
                )
            with engine.connect() as conn:
                page = [tuple(r) for r in conn.execute(stmt.limit(limit)).all()]
            seen.extend(page)
            if len(page) < limit:
                break
            cursor = encode_cursor(page[-1][0], page[-1][1])

        # Exactly once each, in (ts DESC, id ASC) order — `rows` is pre-sorted.
        assert seen == rows

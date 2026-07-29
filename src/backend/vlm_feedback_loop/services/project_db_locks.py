# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-project asyncio.Lock for serializing concurrent writers to a single
project SQLite database.

Several classes of writer can hit one project's ``project.db`` in normal
operation:

* The ``POST /v1/projects/{id}/examples:ingest`` request handler
  (``ingest_examples`` inside ``to_thread``).
* The ``ingest_sweeper_service._ingest_worker`` (UPDATE ``example.phash``).
* The ``clip_embedding_service._embedding_worker`` (INSERT ``clip_embedding``).
* Interactive Guidance, label, project-settings, and review-selector writes.

SQLite WAL allows one writer at a time and the ``busy_timeout=5000ms``
PRAGMA covers short contention windows — but when multiple long-running
workers run concurrently against the same DB, the busy_timeout can be
exceeded, producing ``OperationalError: database is locked``. Without
this lock, ingesting under load produces dozens of such errors across
concurrent sweeper + CLIP worker tasks.

App-layer serialization via the shared lock returned here eliminates
that class of failure and prevents an interactive save from losing a
SQLite lock race to a background batch. Only one cooperating writer holds
the project's write window at a time. Reads remain concurrent (WAL allows
concurrent readers with one writer).

The lock is held *only* across the actual write transaction — not
across network calls (e.g. NVCLIP embedding requests) or CPU-bound
work (pHash compute, base64 encoding). Each writer should:

  # Heavy/network work outside the lock
  vectors = await fetch_embeddings(images)

  # Short write window inside the lock
  async with get_project_write_lock(project_id):
      with Session(engine) as session:
          # INSERT/UPDATE
          session.commit()
"""

from __future__ import annotations

import asyncio

# Module-level registry. Locks are created lazily on first request per
# project_id and never freed — projects are long-lived and the cost of
# a stale ``asyncio.Lock`` reference is a few hundred bytes. Cleared in
# tests by reaching into ``_locks`` directly when needed (rare).
_locks: dict[str, asyncio.Lock] = {}


def get_project_write_lock(project_id: str) -> asyncio.Lock:
    """Return the process-shared asyncio.Lock for the given project's
    SQLite write transactions.

    The same lock is returned for the same ``project_id`` across every
    call. Different projects get independent locks — writes to project
    A do not block writes to project B.
    """
    lock = _locks.get(project_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[project_id] = lock
    return lock

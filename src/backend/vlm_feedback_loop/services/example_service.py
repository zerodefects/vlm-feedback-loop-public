# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example ingestion, querying, and retrieval service.

Handles batch image ingestion with per-item validation, pHash computation,
idempotency checking, and example querying with cursor pagination.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.audit_event import AuditEvent
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.services import filesystem_service
from vlm_feedback_loop.services.authorized_file import open_authorized_image

# PIL format names that map to our supported extensions — canonical set
# lives in ``image_transport``.
from vlm_feedback_loop.services.image_transport import ACCEPTED_PIL_FORMATS
from vlm_feedback_loop.services.pagination import (
    decode_cursor,
    encode_cursor,
)
from vlm_feedback_loop.services.phash import compute_phash
from vlm_feedback_loop.services.project_service import get_project_engine

logger = logging.getLogger("vlm_feedback_loop.example_service")

# 20 MB in bytes
_SIZE_WARNING_BYTES = 20 * 1024 * 1024
# 8192 px longest edge
_SIZE_WARNING_PX = 8192


# ── Image validation ────────────────────────────────────────────────────────


def _validate_image(
    storage_ref: str,
    *,
    full_decode: bool = True,
    settings: Settings | None = None,
) -> tuple[Image.Image | None, str | None, list[str]]:
    """Validate an image file for ingestion.

    Returns ``(pil_image, error_message, warnings)``.
    On success ``pil_image`` is an open PIL Image and ``error_message`` is None.
    On failure ``pil_image`` is None and ``error_message`` describes the problem.

    ``full_decode=False`` uses Pillow's structural ``verify()`` pass instead
    of expanding every source pixel. The production skeleton-row ingest path
    does not consume pixels until the background pHash sweep, so decoding a
    4032×3024 JPEG here only burns CPU and, more importantly, lengthens the
    serialized SQLite write window by tens of milliseconds per image.
    ``full_decode=True`` remains available to callers that immediately consume
    the returned pixels (the legacy inline-pHash path).
    """
    warnings: list[str] = []
    if settings is None:
        from vlm_feedback_loop.config import get_settings

        settings = get_settings()

    try:
        opened = open_authorized_image(storage_ref, settings)
    except (FileNotFoundError, PermissionError) as exc:
        return None, str(exc), warnings

    stream = opened.open_binary()
    try:
        img = Image.open(stream)
    except Exception as exc:
        stream.close()
        opened.close()
        return None, f"Cannot read image: {storage_ref} ({exc})", warnings

    try:
        pil_format = img.format
        if pil_format not in ACCEPTED_PIL_FORMATS:
            img.close()
            stream.close()
            return (
                None,
                f"Unsupported image format: {pil_format or 'unknown'} ({storage_ref})",
                warnings,
            )

        # Multi-page TIFF rejection
        if pil_format == "TIFF":
            try:
                img.seek(1)
                img.seek(0)
                img.close()
                stream.close()
                return (
                    None,
                    f"Multi-page TIFFs are not supported ({storage_ref})",
                    warnings,
                )
            except EOFError:
                pass

        file_size = opened.stat_result.st_size
        display_name = Path(storage_ref).name
        if file_size > _SIZE_WARNING_BYTES:
            warnings.append(
                f"{display_name} {file_size / (1024 * 1024):.1f} MB (exceeds 20 MB)"
            )

        longest_edge = max(img.width, img.height)
        if longest_edge > _SIZE_WARNING_PX:
            warnings.append(
                f"{display_name} {longest_edge} px longest edge "
                f"(exceeds {_SIZE_WARNING_PX} px)"
            )

        try:
            if full_decode:
                cast("Any", img).load()
            else:
                # verify() invalidates the Pillow object. Reopen from a fresh
                # duplicate of the same authorized inode for the caller.
                img.verify()
                img.close()
                stream.close()
                stream = opened.open_binary()
                img = Image.open(stream)
        except Exception as exc:
            img.close()
            stream.close()
            return None, f"Cannot read image: {storage_ref} ({exc})", warnings

        return img, None, warnings
    finally:
        opened.close()


# ── Ingestion ───────────────────────────────────────────────────────────────

# States an ingested Example may be created in. Anything else silently drops
# out of the review loop (the review selector serves only Unlabeled/Auto-
# Labeled), so reject it at ingest instead of persisting an unreachable row.
_INGESTIBLE_STATES = frozenset({"Unlabeled", "Auto-Labeled", "Verified", "Omitted"})


def _validate_ingest_identity(example_key: str, state: str) -> str | None:
    """Return an error message if the key/state are unaddressable, else None.

    ``example_key`` is used verbatim in per-example URL paths
    (``/{example_key}/image``) and as a filter, so an empty key or one
    containing a path/URL separator creates a row no endpoint can address.
    """
    if not example_key:
        return "example_key must be non-empty"
    if len(example_key) > 512:
        return "example_key exceeds 512 characters"
    if any(c in example_key for c in ("/", "\\", "\x00")):
        return "example_key must not contain path separators or null bytes"
    if state not in _INGESTIBLE_STATES:
        return f"invalid state {state!r}: not in {sorted(_INGESTIBLE_STATES)}"
    return None


def _ingest_single(
    session: Session,
    project_id: str,
    item: dict[str, Any],
    *,
    compute_phash_inline: bool = True,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Process a single ingestion item within an open session.

    ``compute_phash_inline=False`` (the fast skeleton-row path) skips the pHash
    compute block: image validation still runs so bad files surface as
    ``status="error"`` and never become rows, but the created row's
    ``phash`` is left ``None`` for the background sweeper to populate
    (``ingest_sweeper_service``). The PIL image handle is closed
    immediately after validation rather than being held open through a
    skipped compute step.

    ``settings`` enables the ``IMAGE_ROOT`` boundary check on ``storage_ref``.
    Enforcing at ingest is the door: images are stored by
    reference, so a row whose ``storage_ref`` escapes the configured root would
    let a later Teacher/embedding call base64-ship an arbitrary host file
    to a NIM endpoint. Rejecting here means no out-of-root reference is
    ever persisted. An unset root (loopback single-user default) is a no-op,
    identical to the browse endpoints.
    """
    example_key: str = item.get("example_key", "")
    storage_ref: str = item.get("storage_ref", "")
    source_metadata: dict[str, Any] = item.get("source_metadata", {})
    extra_metadata: dict[str, Any] | None = item.get("metadata")
    state: str = item.get("state", "Unlabeled")

    if extra_metadata:
        source_metadata = {**source_metadata, **extra_metadata}

    # Reject unaddressable keys / unreachable states before creating a row.
    identity_err = _validate_ingest_identity(example_key, state)
    if identity_err is not None:
        return {
            "example_key": example_key,
            "status": "error",
            "error": identity_err,
            "warnings": [],
            "example": None,
        }

    # Reject a storage_ref outside IMAGE_ROOT before creating a row (see
    # docstring). No-op when the root is unconfigured on loopback.
    if settings is not None:
        allow_err = filesystem_service.check_path_allowed(Path(storage_ref), settings)
        if allow_err is not None:
            return {
                "example_key": example_key,
                "status": "error",
                "error": allow_err,
                "error_code": "path_not_allowed",
                "warnings": [],
                "example": None,
            }

    # Idempotency check
    existing = session.execute(
        select(Example).where(
            Example.project_id == project_id,
            Example.example_key == example_key,
        )
    ).scalar_one_or_none()

    if existing is not None:
        if existing.storage_ref == storage_ref:
            # True idempotent re-ingest
            session.expunge(existing)
            return {
                "example_key": example_key,
                "status": "exists",
                "error": None,
                "warnings": [],
                "example": _example_to_dict(existing),
            }
        # Key collision — different path
        return {
            "example_key": example_key,
            "status": "error",
            "error": (
                f"Generated example_key collided with an existing example "
                f"from a different path. Existing: {existing.storage_ref}. "
                f"New: {storage_ref}."
            ),
            "error_code": "example_key_collision",
            "warnings": [],
            "example": None,
        }

    # A caller can choose its own example_key and scans from older releases
    # generated keys relative to the selected scan directory. Treat the
    # persisted source path as a second idempotency boundary so the same image
    # cannot enter one project twice under different keys.
    existing_path = session.execute(
        select(Example).where(
            Example.project_id == project_id,
            Example.storage_ref == storage_ref,
        )
    ).scalar_one_or_none()
    if existing_path is not None:
        return {
            "example_key": example_key,
            "status": "error",
            "error": (
                "Image path is already ingested under a different example_key. "
                f"Existing key: {existing_path.example_key}. Path: {storage_ref}."
            ),
            "error_code": "storage_ref_already_ingested",
            "warnings": [],
            "example": None,
        }

    img, error, warnings = _validate_image(
        storage_ref,
        full_decode=compute_phash_inline,
        settings=settings,
    )
    if error is not None or img is None:
        return {
            "example_key": example_key,
            "status": "error",
            "error": error or "Image validation produced no image",
            "warnings": warnings,
            "example": None,
        }

    phash_value: str | None = None
    if compute_phash_inline:
        # Compute pHash inline (failure → null, not crash)
        try:
            phash_value = compute_phash(img)
        except Exception:
            logger.warning(
                "pHash computation failed for %s", storage_ref, exc_info=True
            )
            warnings.append(f"pHash computation failed for {Path(storage_ref).name}")

    # Close the image now that validation (and optionally pHash) is done.
    img.close()

    now = utc_now()
    example = Example(
        example_key=example_key,
        project_id=project_id,
        storage_ref=storage_ref,
        ingested_at=now,
        source_metadata=source_metadata,
        state=state,
        phash=phash_value,
    )
    session.add(example)
    session.flush()  # assign to DB but don't commit yet (batch commit)

    session.expunge(example)
    return {
        "example_key": example_key,
        "status": "created",
        "error": None,
        "warnings": warnings,
        "example": _example_to_dict(example),
    }


def ingest_examples(
    project_id: str,
    workspace_root: str,
    items: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Ingest a batch of images into a project.

    Returns a per-item result list with ``status``, ``error``, ``warnings``,
    and ``example`` fields. Processing is per-item: one failure does not
    block others (partial success). When ``settings`` is supplied the
    ``IMAGE_ROOT`` boundary is enforced on each ``storage_ref`` (see
    :func:`_ingest_single`).

    pHash is NOT computed inline: rows are created with ``phash=None`` and
    the background ``ingest_sweeper_service`` worker populates it after the
    endpoint returns, keeping large ingests fast. Image validation still
    runs synchronously so a bad file surfaces as ``status="error"`` instead
    of creating an unworkable row.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return [
            {
                "example_key": item.get("example_key", ""),
                "status": "error",
                "error": "Project not found",
                "warnings": [],
                "example": None,
            }
            for item in items
        ]

    results: list[dict[str, Any]] = []
    with Session(engine) as session:
        for item in items:
            result = _ingest_single(
                session,
                project_id,
                item,
                compute_phash_inline=False,
                settings=settings,
            )
            results.append(result)
        session.commit()
    return results


def _example_to_dict(ex: Example) -> dict[str, Any]:
    """Serialise an Example ORM object to a plain dict."""
    return {
        "example_key": ex.example_key,
        "project_id": ex.project_id,
        "storage_ref": ex.storage_ref,
        "ingested_at": ex.ingested_at,
        "source_metadata": ex.source_metadata,
        "state": ex.state,
        "phash": ex.phash,
        "clip_embedding_present": ex.clip_embedding_present,
        "clip_embedding_dim": ex.clip_embedding_dim,
        "clip_embedding_model_id": ex.clip_embedding_model_id,
        "embedding_provider": ex.embedding_provider,
        "omitted_source": ex.omitted_source,
        "omitted_at": ex.omitted_at,
        "prior_verified_label_ref": ex.prior_verified_label_ref,
        "prior_verified_outcome": ex.prior_verified_outcome,
    }


# ── Retrieval ───────────────────────────────────────────────────────────────


def get_example(
    project_id: str,
    workspace_root: str,
    example_key: str,
) -> Example | None:
    """Look up a single Example by (project_id, example_key)."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None

    with Session(engine) as session:
        row = session.execute(
            select(Example).where(
                Example.project_id == project_id,
                Example.example_key == example_key,
            )
        ).scalar_one_or_none()
        if row is not None:
            session.expunge(row)
        return row


# ── Query ───────────────────────────────────────────────────────────────────

# Cursor codec + error are shared with the other paginated lists (one source
# of truth). This list keys on (timestamp, example_key) with the key ordered
# ASC, so it keeps its own inline "after cursor" predicate rather than using
# pagination.after_position (which assumes a DESC secondary key).


def query_examples(
    project_id: str,
    workspace_root: str,
    *,
    limit: int = 50,
    cursor: str | None = None,
    state: str | None = None,
    verified_after: str | None = None,
    verified_before: str | None = None,
    verified_outcome: str | None = None,
    guidance_id: str | None = None,
    pool_membership: str | None = None,
    include: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Query examples with cursor pagination and optional filters.

    Returns ``(items, next_cursor)``.  Each item is a dict with ``example``
    and optionally ``verified_label`` keys.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return [], None

    needs_label_join = any(
        [
            verified_after,
            verified_before,
            verified_outcome,
            guidance_id,
            pool_membership,
            include == "verified_label",
        ]
    )

    with Session(engine) as session:
        if needs_label_join:
            return _query_with_label_join(
                session,
                project_id,
                limit=limit,
                cursor=cursor,
                state=state,
                verified_after=verified_after,
                verified_before=verified_before,
                verified_outcome=verified_outcome,
                guidance_id=guidance_id,
                pool_membership=pool_membership,
                include=include,
            )
        return _query_simple(
            session,
            project_id,
            limit=limit,
            cursor=cursor,
            state=state,
        )


def _query_simple(
    session: Session,
    project_id: str,
    *,
    limit: int,
    cursor: str | None,
    state: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Query examples without label join — ordered by ingested_at DESC, example_key ASC."""
    stmt = select(Example).where(Example.project_id == project_id)

    if state is not None:
        stmt = stmt.where(Example.state == state)

    # Cursor: (ingested_at, example_key)
    if cursor is not None:
        cursor_ts, cursor_key = decode_cursor(cursor)
        if cursor_ts is not None:
            stmt = stmt.where(
                (Example.ingested_at < cursor_ts)
                | (
                    (Example.ingested_at == cursor_ts)
                    & (Example.example_key > cursor_key)
                )
            )
        else:
            stmt = stmt.where(Example.example_key > cursor_key)

    stmt = stmt.order_by(Example.ingested_at.desc(), Example.example_key.asc())
    stmt = stmt.limit(limit + 1)  # fetch one extra to detect next page

    rows = list(session.execute(stmt).scalars().all())

    has_next = len(rows) > limit
    if has_next:
        rows = rows[:limit]

    items: list[dict[str, Any]] = []
    for ex in rows:
        session.expunge(ex)
        items.append({"example": _example_to_dict(ex), "verified_label": None})

    next_cursor: str | None = None
    if has_next and rows:
        last = rows[-1]
        next_cursor = encode_cursor(last.ingested_at, last.example_key)

    return items, next_cursor


def _query_with_label_join(
    session: Session,
    project_id: str,
    *,
    limit: int,
    cursor: str | None,
    state: str | None,
    verified_after: str | None,
    verified_before: str | None,
    verified_outcome: str | None,
    guidance_id: str | None,
    pool_membership: str | None,
    include: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Query with label join — ordered by verified_at DESC NULLS LAST, example_key ASC."""
    # Main query: Example LEFT JOIN Label
    stmt = (
        select(Example, Label)
        .outerjoin(
            Label,
            (Label.project_id == Example.project_id)
            & (Label.example_key == Example.example_key)
            & (Label.label_status == "verified"),
        )
        .where(Example.project_id == project_id)
    )

    if state is not None:
        stmt = stmt.where(Example.state == state)

    # Label-specific filters
    if verified_after is not None:
        stmt = stmt.where(Label.verified_at > verified_after)
    if verified_before is not None:
        stmt = stmt.where(Label.verified_at < verified_before)
    if verified_outcome is not None:
        stmt = stmt.where(Label.verified_outcome == verified_outcome)
    if guidance_id is not None:
        stmt = stmt.where(Label.guidance_id == guidance_id)
    if pool_membership == "test_pool":
        stmt = stmt.where(Label.pool_assignment == "test_pool")
    elif pool_membership == "none":
        stmt = stmt.where(
            (Label.pool_assignment.is_(None)) | (Label.label_id.is_(None))
        )
    elif pool_membership == "any":
        stmt = stmt.where(Label.pool_assignment.isnot(None))

    # Ordering: verified_at DESC NULLS LAST, example_key ASC
    # SQLite sorts NULLs last in DESC naturally
    stmt = stmt.order_by(
        Label.verified_at.desc().nullslast(),
        Example.example_key.asc(),
    )

    # Cursor
    if cursor is not None:
        cursor_ts, cursor_key = decode_cursor(cursor)
        if cursor_ts is not None:
            stmt = stmt.where(
                (Label.verified_at < cursor_ts)
                | (
                    (Label.verified_at == cursor_ts)
                    & (Example.example_key > cursor_key)
                )
                | (Label.verified_at.is_(None))
            )
        else:
            # Cursor was in the null-verified_at section
            stmt = stmt.where(
                (Label.verified_at.is_(None)) & (Example.example_key > cursor_key)
            )

    stmt = stmt.limit(limit + 1)

    rows = list(session.execute(stmt).all())

    has_next = len(rows) > limit
    if has_next:
        rows = rows[:limit]

    items: list[dict[str, Any]] = []
    for row in rows:
        ex: Example = row[0]
        lbl: Label | None = row[1]
        session.expunge(ex)
        if lbl is not None:
            session.expunge(lbl)

        item: dict[str, Any] = {"example": _example_to_dict(ex)}
        if include == "verified_label" and lbl is not None:
            item["verified_label"] = _label_to_dict(lbl)
        else:
            item["verified_label"] = None

        items.append(item)

    next_cursor: str | None = None
    if has_next and rows:
        last_row = rows[-1]
        last_lbl: Label | None = last_row[1]
        last_ex: Example = last_row[0]
        va = last_lbl.verified_at if last_lbl is not None else None
        next_cursor = encode_cursor(va, last_ex.example_key)

    return items, next_cursor


def _label_to_dict(lbl: Label) -> dict[str, Any]:
    """Serialise a Label ORM object to a plain dict."""
    return {
        "label_id": lbl.label_id,
        "example_key": lbl.example_key,
        "project_id": lbl.project_id,
        "label_status": lbl.label_status,
        "guidance_id": lbl.guidance_id,
        "inference_invocation_id": lbl.inference_invocation_id,
        "label_json": lbl.label_json,
        "labeled_at": lbl.labeled_at,
        "verified_outcome": lbl.verified_outcome,
        "verified_at": lbl.verified_at,
        "edited_core_fields": lbl.edited_core_fields,
        "edited_aux_fields": lbl.edited_aux_fields,
        "rationale_source": lbl.rationale_source,
        "batch_label_run_id": lbl.batch_label_run_id,
        "pool_assignment": lbl.pool_assignment,
    }


# ── Path Remapping ─────────────────────────────────────────────────────────


def remap_paths(
    project_id: str,
    workspace_root: str,
    old_prefix: str,
    new_prefix: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Bulk-remap storage_ref paths by prefix replacement.

    Returns a dict with either dry-run preview data or commit results.

    Raises:
        ValueError: when inputs are invalid or zero-resolve on commit.
    """
    if not old_prefix or not old_prefix.startswith("/"):
        raise ValueError("old_prefix must be a non-empty absolute path")
    if not new_prefix or not new_prefix.startswith("/"):
        raise ValueError("new_prefix must be a non-empty absolute path")

    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        raise ValueError("Project not found")

    with Session(engine) as session:
        matching = list(
            session.execute(
                select(Example).where(
                    Example.project_id == project_id,
                    # Byte-literal prefix match. LIKE (which .startswith
                    # compiles to) treats '_'/'%' as wildcards and folds
                    # ASCII case, so /data/My_Data would also match
                    # /data/my_data — and the rewrite below would splice
                    # the wrong tail onto those unrelated rows.
                    func.substr(Example.storage_ref, 1, len(old_prefix)) == old_prefix,
                )
            )
            .scalars()
            .all()
        )

        matched_count = len(matching)

        total_count = session.execute(
            select(func.count())
            .select_from(Example)
            .where(Example.project_id == project_id)
        ).scalar_one()
        unmatched_count = total_count - matched_count

        # Build sample remappings (up to 10)
        sample = matching[:10]
        sample_remappings: list[dict[str, str]] = []
        for ex in sample:
            new_ref = new_prefix + ex.storage_ref[len(old_prefix) :]
            sample_remappings.append(
                {
                    "example_key": ex.example_key,
                    "old_storage_ref": ex.storage_ref,
                    "new_storage_ref": new_ref,
                }
            )

        # Validation: spot-check up to 10 remapped paths
        sample_checked = len(sample_remappings)
        sample_resolved = 0
        sample_missing = 0
        missing_examples: list[str] = []
        for item in sample_remappings:
            if Path(item["new_storage_ref"]).is_file():
                sample_resolved += 1
            else:
                sample_missing += 1
                missing_examples.append(item["example_key"])

        if dry_run:
            return {
                "dry_run": True,
                "matched_count": matched_count,
                "sample_remappings": sample_remappings,
                "unmatched_count": unmatched_count,
                "validation": {
                    "sample_checked": sample_checked,
                    "sample_resolved": sample_resolved,
                    "sample_missing": sample_missing,
                    "missing_examples": missing_examples,
                },
            }

        # Commit mode
        if matched_count > 0 and sample_resolved == 0:
            raise ValueError(
                "None of the sampled remapped paths resolve to existing "
                "files. Verify new_prefix is correct."
            )

        for ex in matching:
            ex.storage_ref = new_prefix + ex.storage_ref[len(old_prefix) :]

        now = utc_now()
        audit = AuditEvent(
            audit_event_id=generate_uuid4(),
            project_id=project_id,
            event_type="storage_ref_remap",
            event_data={
                "old_prefix": old_prefix,
                "new_prefix": new_prefix,
                "remapped_count": matched_count,
                "timestamp": now,
            },
            created_at=now,
        )
        session.add(audit)
        session.commit()

        return {
            "dry_run": False,
            "remapped_count": matched_count,
            "audit_event_id": audit.audit_event_id,
        }

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-applicable deployment secrets.

Two-layer secret resolution: a process-level override layer sits in
front of the ``Settings`` LRU singleton (``config.py``). Services
read effective secret values via :func:`get_effective_secret`, which
returns the override if set and otherwise falls through to the
``Settings`` value loaded from ``~/.vlm_feedback_loop/.env``.

The override layer lets the SME paste a key into the UI and have it
take effect on the next NIM call (embedding worker spawn, Teacher
proposal, local NIM image pull) WITHOUT a backend restart — closing
the FTU friction where a no-keys project would otherwise need an
``.env`` edit and bounce to recover.

Two persistence shapes are supported via :func:`set_runtime_secret` and
:func:`persist_secret_to_env`:

* **Session-only** (default): held in process memory; lost on backend
  restart. Sufficient for trying out a key without committing it to
  disk; the right posture for shared installs.
* **Persisted**: written to the canonical ``.env`` file (gated by
  ``ALLOW_UI_SECRET_PERSIST``; opt-in from the Apply Controls UI).
  Survives restart. After persist, the same value is re-installed as
  the runtime override so in-flight work holding a stale ``Settings``
  reference still resolves the new credential; the disk value becomes
  canonical at the next process start (see rationale in
  :func:`persist_secret_to_env`).

Audit-event emission lives in the calling router; this module is pure
state plumbing. Key VALUES are never logged or returned in error paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from vlm_feedback_loop.config import (
    get_canonical_env_file_path,
    get_config_file_path,
    init_settings,
)

logger = logging.getLogger("vlm_feedback_loop.services.runtime_secrets")

#: Canonical names of the secrets this module manages. Restricted to the
#: three deployment-scoped credentials. Adding new entries
#: requires a security review (the override mechanism elevates a
#: pasted-key UI affordance into a process-wide effective credential).
ALLOWED_SECRETS: Final[frozenset[str]] = frozenset(
    {"NVIDIA_API_KEY", "NGC_API_KEY", "TAO_API_KEY"}
)


class InvalidSecretNameError(ValueError):
    """Raised when a caller supplies a name outside :data:`ALLOWED_SECRETS`."""


_runtime_overrides: dict[str, str] = {}


def _validate_name(name: str) -> None:
    if name not in ALLOWED_SECRETS:
        raise InvalidSecretNameError(
            f"Unknown secret name {name!r}; allowed: {sorted(ALLOWED_SECRETS)}"
        )


def _validate_value(value: str) -> None:
    """Reject an empty value or one carrying control characters.

    A newline in the value would inject extra ``KEY=VALUE`` lines when
    persisted to ``.env`` — smuggling settings past the ``ALLOWED_SECRETS``
    name allowlist (the loader applies every ``.env`` key with precedence
    over ``config.yaml``) — and any control char corrupts both the file
    and the auth header the value is interpolated into. Real credentials
    contain none, so this rejects rather than silently strips.
    """
    if not value:
        raise ValueError("Refusing to set an empty secret value")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError(
            "Secret value contains control characters (newlines, tabs, or "
            "similar); paste the credential without surrounding whitespace"
        )


def get_effective_secret(
    name: str,
    settings: object | None = None,
) -> str | None:
    """Resolve the effective value of a deployment-scoped secret.

    Runtime override wins over the ``Settings``-loaded value. Returns
    ``None`` when neither layer has a value (e.g. no key configured at
    all — the cold-start FTU case).

    ``settings`` is an optional :class:`vlm_feedback_loop.config.Settings`
    instance used as the fallback when no runtime override is set. When
    omitted, the cached process-wide ``Settings`` singleton is used —
    the right default for production paths. Passing an explicit
    instance is useful in two cases:

    * Test code that constructs an ad-hoc ``Settings`` (e.g. via a
      helper factory) and wants the live values reflected without
      mutating process state.
    * Production paths that have already received a snapshot
      ``Settings`` and want to avoid re-fetching the singleton.

    Typed as ``object`` to avoid a circular import at module load
    (``Settings`` lives in ``config`` which imports nothing from
    services). The runtime ``getattr`` is duck-typed.
    """
    _validate_name(name)
    override = _runtime_overrides.get(name)
    if override is not None:
        return override
    if settings is None:
        # Late import to avoid circular dependency at module load.
        from vlm_feedback_loop.config import get_settings

        settings = get_settings()
    value = getattr(settings, name, None)
    if value is None:
        return None
    return str(value) if not isinstance(value, str) else value


def set_runtime_secret(name: str, value: str) -> None:
    """Install ``value`` into the process-level override layer.

    Subsequent :func:`get_effective_secret` calls return this value until
    the process exits. Also invalidates the environment-assessment cache
    so the model-catalog list endpoint picks up the new credential on its
    next call without waiting for the TTL.
    """
    _validate_name(name)
    _validate_value(value)
    _runtime_overrides[name] = value
    logger.info("Runtime secret %s set (length=%d)", name, len(value))
    # Late import to avoid environment.py loading at process start.
    from vlm_feedback_loop.services.environment import invalidate_env_cache

    invalidate_env_cache()


def persist_secret_to_env(name: str, value: str, env_path: Path | None = None) -> Path:
    """Upsert ``name=value`` in ``env_path`` and reload Settings.

    Preserves other lines (other env vars, comments, blank lines) and the
    file's existing ordering. If the file is missing, creates it (and the
    parent directory) with ``600`` / ``700`` permissions.

    After the disk write, calls :func:`init_settings` against the same
    path so subsequent ``get_settings()`` calls see the persisted value
    (skipped when no canonical ``config.yaml`` exists — the loader would
    fail-fast mid-request), then re-installs the value as the runtime
    override so in-flight work holding a stale ``Settings`` reference
    still resolves the new credential (see the inline rationale below).

    Returns the resolved ``env_path``.

    Raises :class:`InvalidSecretNameError` for unknown names and
    :class:`OSError` if the disk write fails. Callers must enforce the
    ``ALLOW_UI_SECRET_PERSIST`` deployment flag before invoking — this
    function does not consult it.
    """
    _validate_name(name)
    _validate_value(value)

    target = env_path if env_path is not None else get_canonical_env_file_path()

    # Ensure parent directory exists with user-only permissions.
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    # Read existing lines (if any) and replace any prior `NAME=...` line.
    existing_lines: list[str] = []
    if target.exists():
        existing_lines = target.read_text(encoding="utf-8").splitlines()

    new_lines: list[str] = []
    replaced = False
    prefix = f"{name}="
    for line in existing_lines:
        # Match `NAME=...` (exact key, ignoring leading whitespace; tolerates
        # `export NAME=...` shape by stripping the prefix). Anything that
        # doesn't match passes through verbatim — comments, blanks, other
        # vars all preserved.
        stripped = line.lstrip()
        normalised = (
            stripped[len("export ") :] if stripped.startswith("export ") else stripped
        )
        if normalised.startswith(prefix):
            new_lines.append(f"{name}={value}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        # Append at end. Ensure a trailing newline for cleanliness.
        if new_lines and new_lines[-1] != "":
            new_lines.append("")
        new_lines.append(f"{name}={value}")

    # Compose final content with a trailing newline.
    content = "\n".join(new_lines)
    if not content.endswith("\n"):
        content += "\n"

    target.write_text(content, encoding="utf-8")
    # User-only file permissions.
    try:
        target.chmod(0o600)
    except OSError:
        # Permission setting may fail on filesystems that don't support it
        # (e.g. mounted FAT). The write succeeded; log and continue.
        logger.warning(
            "Could not chmod 0o600 on %s; file written but permissions "
            "could not be tightened.",
            target,
        )

    # Reload Settings so subsequent get_settings() calls pick up the new
    # value. We deliberately do NOT clear the runtime override — see the
    # rationale below. Skipped when the canonical config.yaml is absent
    # (env-only hosts, hermetic tests): the loader would SystemExit
    # mid-request, and the runtime override installed below already makes
    # the new value effective for this process while the .env write above
    # covers the next boot.
    if get_config_file_path().exists():
        init_settings(cli_env_file=str(target))
    # Re-install the (now-persisted) value into the runtime override so
    # it remains canonical for the rest of the session.
    #
    # Why: ``init_settings`` constructs a NEW Settings instance and
    # rebinds the ``get_settings()`` singleton. But any code that
    # captured a Settings reference earlier in the request lifecycle —
    # e.g., an in-flight background ``deploy_local_nim`` task that
    # received ``settings: Settings`` from its FastAPI dependency at
    # submission time — still holds the OLD instance with the OLD
    # credential. ``get_effective_secret(name, settings=<stale_old>)``
    # would then return the OLD value, breaking the "paste a new key
    # and it applies immediately" property for any background work
    # already queued.
    #
    # Evidence: with a fresh NGC key persisted to ``.env``, in-flight
    # local NIM teacher + embedding deploys queued seconds earlier from
    # the NIM Configuration screen still fail with 401 from nvcr.io
    # because they use the
    # captured stale Settings.
    #
    # Keeping the runtime override post-persist means
    # ``get_effective_secret`` always returns the persisted value first,
    # regardless of which Settings ref a caller is holding. The disk
    # value remains canonical at process restart; the override is the
    # in-process bridge that keeps stale Settings refs from poisoning
    # background work. This trade-off is deliberate.
    _runtime_overrides[name] = value
    logger.info(
        "Persisted secret %s to %s (replaced existing=%s)",
        name,
        target,
        replaced,
    )
    return target


def reset_overrides_for_testing() -> None:
    """Clear all runtime overrides. Test-only helper.

    Exists for test fixtures that need a clean slate between cases.
    """
    _runtime_overrides.clear()

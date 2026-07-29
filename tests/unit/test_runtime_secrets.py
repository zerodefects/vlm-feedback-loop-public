# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the runtime-secrets override layer."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlm_feedback_loop.services import runtime_secrets

#: A stand-in for a Settings snapshot with no key configured. Because
#: ``get_effective_secret`` duck-types its ``settings`` argument, resolving
#: a value against this stub proves the value came from the runtime
#: override layer, not from ``Settings``.
_EMPTY_SETTINGS = SimpleNamespace(NVIDIA_API_KEY=None)


@pytest.fixture(autouse=True)
def reset_overrides():
    """Each test starts with a clean override layer to avoid bleed-through."""
    runtime_secrets.reset_overrides_for_testing()
    yield
    runtime_secrets.reset_overrides_for_testing()


class TestRuntimeOverride:
    def test_get_effective_secret_falls_through_when_no_override(self, isolated_config):
        """When no override is set, ``get_effective_secret`` returns the
        ``Settings``-loaded value (test confirms equality with the
        fallback path; under the isolated config there is no ``.env``,
        so both sides resolve to the field default)."""
        from vlm_feedback_loop.config import get_settings

        # Sanity: no override installed.
        assert "NVIDIA_API_KEY" not in runtime_secrets._runtime_overrides
        # Effective value matches the Settings-loaded value (which is
        # whatever the developer has in their canonical .env — may be
        # None or a real key; both are valid).
        assert (
            runtime_secrets.get_effective_secret("NVIDIA_API_KEY")
            == get_settings().NVIDIA_API_KEY
        )

    def test_set_runtime_secret_then_get_returns_override(self):
        runtime_secrets.set_runtime_secret("NVIDIA_API_KEY", "nvapi-test")
        assert runtime_secrets.get_effective_secret("NVIDIA_API_KEY") == "nvapi-test"
        # The value resolves even against a Settings snapshot with no key
        # — it came from the override layer.
        assert (
            runtime_secrets.get_effective_secret("NVIDIA_API_KEY", _EMPTY_SETTINGS)
            == "nvapi-test"
        )

    def test_invalid_name_rejected(self):
        with pytest.raises(runtime_secrets.InvalidSecretNameError):
            runtime_secrets.set_runtime_secret("FOO_BAR", "v")

    def test_empty_value_rejected(self):
        with pytest.raises(ValueError):
            runtime_secrets.set_runtime_secret("NVIDIA_API_KEY", "")

    def test_control_character_value_rejected(self):
        """A newline in the value would inject extra KEY=VALUE lines when
        persisted, smuggling arbitrary settings past the ALLOWED_SECRETS
        name allowlist. The service rejects control characters and leaves
        no override behind."""
        injection = "nvapi-real\nIMAGE_ROOT=/"
        with pytest.raises(ValueError):
            runtime_secrets.set_runtime_secret("NVIDIA_API_KEY", injection)
        assert "NVIDIA_API_KEY" not in runtime_secrets._runtime_overrides


# persist_secret_to_env reloads Settings via init_settings, so every test in
# this class needs the hermetic temp config.yaml (CI runners have none).
@pytest.mark.usefixtures("isolated_config")
class TestPersistSecretToEnv:
    def test_writes_new_file_with_secure_permissions(self, tmp_path: Path):
        env_path = tmp_path / "subdir" / ".env"
        runtime_secrets.persist_secret_to_env(
            "NVIDIA_API_KEY", "persisted-value", env_path=env_path
        )
        assert env_path.exists()
        # 600 file (user-only).
        file_perms = stat.S_IMODE(os.stat(env_path).st_mode)
        assert file_perms == 0o600, f"expected 600, got {file_perms:o}"
        # 700 parent dir.
        parent_perms = stat.S_IMODE(os.stat(env_path.parent).st_mode)
        assert parent_perms == 0o700, f"expected 700, got {parent_perms:o}"
        # Line present.
        assert "NVIDIA_API_KEY=persisted-value" in env_path.read_text()

    def test_newline_injection_rejected_and_file_untouched(self, tmp_path: Path):
        """A newline-bearing value must not reach the .env writer: it would
        add extra KEY=VALUE lines that the loader applies with precedence
        over config.yaml, overriding non-secret settings the name allowlist
        was supposed to protect."""
        env_path = tmp_path / ".env"
        env_path.write_text("LOG_LEVEL=info\n")
        with pytest.raises(ValueError):
            runtime_secrets.persist_secret_to_env(
                "NVIDIA_API_KEY",
                "nvapi-real\nBIND_HOST=0.0.0.0",
                env_path=env_path,
            )
        # The pre-existing file is left exactly as it was — no partial write.
        assert env_path.read_text() == "LOG_LEVEL=info\n"

    def test_preserves_other_lines_and_comments(self, tmp_path: Path):
        env_path = tmp_path / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(
            "# top-level comment\nLOG_LEVEL=debug\n\n# another\nWORKSPACE_ROOT=/tmp\n"
        )
        runtime_secrets.persist_secret_to_env(
            "NVIDIA_API_KEY", "the-key", env_path=env_path
        )
        content = env_path.read_text()
        assert "# top-level comment" in content
        assert "LOG_LEVEL=debug" in content
        assert "# another" in content
        assert "WORKSPACE_ROOT=/tmp" in content
        assert "NVIDIA_API_KEY=the-key" in content

    def test_idempotent_no_duplicate_lines(self, tmp_path: Path):
        env_path = tmp_path / ".env"
        runtime_secrets.persist_secret_to_env("NVIDIA_API_KEY", "v1", env_path=env_path)
        runtime_secrets.persist_secret_to_env("NVIDIA_API_KEY", "v1", env_path=env_path)
        # Exactly one occurrence — no duplicate appends.
        content = env_path.read_text()
        assert content.count("NVIDIA_API_KEY=") == 1

    def test_updates_existing_line(self, tmp_path: Path):
        env_path = tmp_path / ".env"
        runtime_secrets.persist_secret_to_env("NVIDIA_API_KEY", "v1", env_path=env_path)
        runtime_secrets.persist_secret_to_env("NVIDIA_API_KEY", "v2", env_path=env_path)
        content = env_path.read_text()
        # Old value replaced, not appended.
        assert content.count("NVIDIA_API_KEY=") == 1
        assert "v2" in content
        assert "v1" not in content

    def test_persist_retains_runtime_override_with_persisted_value(
        self, tmp_path: Path
    ):
        """Persist retains the runtime override (set to the persisted
        value) so background tasks holding stale Settings references
        don't read the OLD value via ``get_effective_secret(name,
        settings=<stale>)``. The override is the in-process bridge that
        guarantees "paste a new key and it applies immediately" for any
        already-queued work.
        """
        env_path = tmp_path / ".env"
        runtime_secrets.set_runtime_secret("NVIDIA_API_KEY", "session-value")
        assert (
            runtime_secrets.get_effective_secret("NVIDIA_API_KEY", _EMPTY_SETTINGS)
            == "session-value"
        )
        runtime_secrets.persist_secret_to_env(
            "NVIDIA_API_KEY", "persisted-value", env_path=env_path
        )
        # After persist, the override is RETAINED (with the persisted
        # value) so stale Settings references can't shadow the update:
        # even a Settings snapshot with no key resolves the new value.
        assert (
            runtime_secrets.get_effective_secret("NVIDIA_API_KEY", _EMPTY_SETTINGS)
            == "persisted-value"
        )
        assert (
            runtime_secrets.get_effective_secret("NVIDIA_API_KEY") == "persisted-value"
        )

    def test_persist_with_stale_settings_still_returns_new_value(self, tmp_path: Path):
        """Regression: a background task holds a Settings reference
        captured at request-dispatch time. The user pastes a new key
        (set + persist). The background task should read the NEW value
        via get_effective_secret(name, settings=<stale_old_settings>)
        even though the captured Settings still has the old value.
        """
        env_path = tmp_path / ".env"
        env_path.write_text("NVIDIA_API_KEY=old-value\n")
        # Background task captures Settings reflecting old-value at
        # request-dispatch time. ``get_effective_secret`` is duck-typed
        # on the ``settings`` arg (declared as ``object``), so a
        # SimpleNamespace stands in for a real Settings snapshot.
        stale_settings = SimpleNamespace(NVIDIA_API_KEY="old-value")

        # User pastes a new key and persists.
        runtime_secrets.set_runtime_secret("NVIDIA_API_KEY", "new-value")
        runtime_secrets.persist_secret_to_env(
            "NVIDIA_API_KEY", "new-value", env_path=env_path
        )

        # Background task does its credential lookup. Even with the
        # stale Settings reference it captured earlier, it MUST see the
        # new value (runtime override wins).
        effective = runtime_secrets.get_effective_secret(
            "NVIDIA_API_KEY", stale_settings
        )
        assert effective == "new-value"


class TestSecretsRouter:
    def test_persist_false_only_runtime(self, test_app_client):
        resp = test_app_client.post(
            "/v1/secrets:set",
            json={"name": "NVIDIA_API_KEY", "value": "nvapi-x", "persist": False},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["effective"] is True
        assert body["persisted"] is False
        assert body["env_path"] is None
        assert runtime_secrets._runtime_overrides.get("NVIDIA_API_KEY") == "nvapi-x"

    def test_persist_blocked_when_disabled(self, test_app_client):
        # test_app_client injects a Settings instance via dependency
        # override. Reach into the override and flip the flag — same
        # pattern used elsewhere in this file's fixtures.
        from vlm_feedback_loop.main import app
        from vlm_feedback_loop.routers.projects import get_current_settings

        injected_settings = app.dependency_overrides[get_current_settings]()
        injected_settings.ALLOW_UI_SECRET_PERSIST = False

        resp = test_app_client.post(
            "/v1/secrets:set",
            json={
                "name": "NVIDIA_API_KEY",
                "value": "nvapi-x",
                "persist": True,
            },
        )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "ui_secret_persist_disabled"
        # Runtime override should NOT be installed when persist was
        # requested but blocked — the SME's all-or-nothing intent is
        # preserved.
        assert "NVIDIA_API_KEY" not in runtime_secrets._runtime_overrides

        # Restore flag for downstream tests sharing the fixture
        injected_settings.ALLOW_UI_SECRET_PERSIST = True

    def test_invalid_name_returns_422(self, test_app_client):
        resp = test_app_client.post(
            "/v1/secrets:set",
            json={"name": "FOO_BAR", "value": "x", "persist": False},
        )
        assert resp.status_code == 422
        # Rejected at the schema edge — nothing installed in the
        # override layer.
        assert not runtime_secrets._runtime_overrides

    def test_empty_value_returns_422(self, test_app_client):
        resp = test_app_client.post(
            "/v1/secrets:set",
            json={"name": "NVIDIA_API_KEY", "value": "", "persist": False},
        )
        assert resp.status_code == 422
        # An empty paste must not clobber (or create) the override.
        assert "NVIDIA_API_KEY" not in runtime_secrets._runtime_overrides

    def test_control_character_value_returns_422(self, test_app_client):
        """A newline-injecting value is rejected at the API boundary (422),
        not surfaced as a 500 or persisted."""
        resp = test_app_client.post(
            "/v1/secrets:set",
            json={
                "name": "NVIDIA_API_KEY",
                "value": "nvapi-x\nIMAGE_ROOT=/",
                "persist": False,
            },
        )
        assert resp.status_code == 422
        assert "NVIDIA_API_KEY" not in runtime_secrets._runtime_overrides

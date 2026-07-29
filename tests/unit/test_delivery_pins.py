# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift guards pinning delivery surfaces to the backend's canonical
catalog constants.

The repo ships operator-facing launch helpers (``scripts/setup-local.sh``,
``docker-compose.yml``) and live smokes that carry copies of values whose
single source of truth is backend code. Each test here fails when a copy
drifts from its authoritative constant — the failure mode that shipped a
setup script pre-pulling a container tag the backend itself classifies as
known-bad.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import yaml

from vlm_feedback_loop._defaults import DEFAULTS
from vlm_feedback_loop.model_catalog_constants import EMBEDDING_NIM_IMAGE
from vlm_feedback_loop.services.project_service import SEEDED_MODEL_CATALOG

_REPO_ROOT = Path(__file__).resolve().parents[2]

_EMBED_IMAGE_RE = re.compile(
    r"nvcr\.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:[0-9][\w.\-]*"
)


class TestSetupScriptImagePins:
    def test_scripts_reference_only_the_canonical_embedding_image(self):
        """Every embedding-NIM image literal under scripts/ must equal the
        backend's ``EMBEDDING_NIM_IMAGE``. A drifted tag makes the
        documented pre-pull fetch (or fail on) an image the first real
        deploy never uses, defeating the cache-warming purpose."""
        literals: dict[str, set[str]] = {}
        for path in sorted((_REPO_ROOT / "scripts").rglob("*")):
            if path.suffix not in {".sh", ".py", ".md"} or not path.is_file():
                continue
            found = set(_EMBED_IMAGE_RE.findall(path.read_text(errors="replace")))
            if found:
                literals[str(path.relative_to(_REPO_ROOT))] = found

        assert literals, "no script references the embedding NIM image anymore"
        for rel_path, found in literals.items():
            assert found == {EMBEDDING_NIM_IMAGE}, (
                f"{rel_path} pins {sorted(found)} but the canonical image is "
                f"{EMBEDDING_NIM_IMAGE} (model_catalog_constants.py)"
            )


class TestSetupScriptSecretForwarding:
    def test_precache_docker_forwards_ngc_by_name(self):
        """The pre-cache container inherits NGC without a literal argv value."""
        script = (_REPO_ROOT / "scripts" / "setup-local.sh").read_text()
        assert re.search(r"-e\s+NGC_API_KEY(?:\s|\\)", script)
        assert not re.search(r"-e\s+NGC_API_KEY\s*=", script)


class TestComposeSecretPersistDefault:
    def test_compose_defaults_ui_secret_persist_off(self):
        """The shipped compose must disable UI secret persistence: no
        volume covers /home/appuser, so a key 'saved to .env' from the UI
        would land in the container's ephemeral layer and silently vanish
        on the next recreate. ``${ALLOW_UI_SECRET_PERSIST:-false}`` keeps
        it operator-overridable for persistent-home mounts."""
        compose = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text())
        environment = compose["services"]["backend"]["environment"]
        assert (
            "ALLOW_UI_SECRET_PERSIST=${ALLOW_UI_SECRET_PERSIST:-false}" in environment
        )


class TestHostedSeededModelsSmokeRoster:
    @staticmethod
    def _import_smoke_module():
        path = _REPO_ROOT / "scripts" / "hosted_seeded_models_smoke.py"
        spec = importlib.util.spec_from_file_location(
            "hosted_seeded_models_smoke", path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # Registration is required for the module's dataclass to resolve
        # its stringified annotations during exec.
        sys.modules[spec.name] = module
        # Module-level code runs the roster drift guard (no I/O happens
        # before argparse in main()); a stale roster raises SystemExit
        # here, which is exactly the DOA state this test pins against.
        spec.loader.exec_module(module)
        return module

    def test_roster_resolves_against_current_catalog(self):
        """The live smoke must be launchable at HEAD: its covered-model
        roster resolves every name against SEEDED_MODEL_CATALOG (a stale
        roster kills the script at import, before even --help)."""
        module = self._import_smoke_module()
        catalog_names = {entry["model_name"] for entry in SEEDED_MODEL_CATALOG}
        covered = set(module._COVERED_MODEL_NAMES)
        assert module.SEEDED_MODELS, "smoke covers no models"
        assert covered <= catalog_names, (
            f"smoke roster names not in SEEDED_MODEL_CATALOG: "
            f"{sorted(covered - catalog_names)}"
        )

    def test_roster_covers_the_default_teacher(self):
        """The smoke's purpose is verifying the seeded hosted models an
        operator actually hits — at minimum the shipped default Teacher.
        This fails when a default-Teacher reseat forgets the roster."""
        module = self._import_smoke_module()
        assert DEFAULTS["DEFAULT_TEACHER_MODEL"] in module._COVERED_MODEL_NAMES

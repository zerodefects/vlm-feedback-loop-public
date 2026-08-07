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
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
import yaml

import vlm_feedback_loop.services.local_nim_deploy_generator  # noqa: F401
from vlm_feedback_loop._defaults import DEFAULTS, SECRET_KEYS
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER_GPU_MIN_GB,
    COSMOS3_REASONER_NIM_IMAGE,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_2B_GPU_MIN_GB,
    COSMOS_REASON2_2B_NIM_IMAGE,
    COSMOS_REASON2_8B,
    EMBEDDING_NIM_IMAGE,
    MISTRAL_MEDIUM_3_5,
    NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN,
    NEMOTRON_3_NANO_OMNI_GPU_MIN_GB,
    NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
    NEMOTRON_3_NANO_OMNI_REASONING,
    NEMOTRON_NANO_12B_VL,
    STEP_3_7_FLASH,
)
from vlm_feedback_loop.services.action_requests import generate_action_request
from vlm_feedback_loop.services.project_service import SEEDED_MODEL_CATALOG

_REPO_ROOT = Path(__file__).resolve().parents[2]

_NIM_IMAGE_RE = re.compile(
    r"nvcr\.io/nim/nvidia/[a-z0-9][a-z0-9._-]*:[a-z0-9][a-z0-9._-]*",
    re.IGNORECASE,
)


def _shell_assignment(path: Path, name: str) -> str:
    match = re.search(
        rf'^\s*{re.escape(name)}=(?:"([^"]*)"|([^\s#]+))\s*$',
        path.read_text(),
        re.MULTILINE,
    )
    assert match is not None, f"{path.name} does not assign {name}"
    value = match.group(1) if match.group(1) is not None else match.group(2)
    assert value is not None
    return value


def _shell_function(path: Path, name: str) -> str:
    lines = path.read_text().splitlines()
    start = lines.index(f"{name}() {{")
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == "}")
    return "\n".join(lines[start : end + 1])


def _import_script_module(filename: str) -> ModuleType:
    path = _REPO_ROOT / "scripts" / filename
    module_name = f"delivery_pin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve stringified annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestSetupScriptImagePins:
    def test_local_setup_images_match_the_backend_catalog(self):
        """Pre-pull must warm the same pinned images that deployment uses."""
        path = _REPO_ROOT / "scripts" / "setup-local.sh"
        expected = {
            "COSMOS_REASON2_2B_NIM_IMAGE": COSMOS_REASON2_2B_NIM_IMAGE,
            "COSMOS3_REASONER_NIM_IMAGE": COSMOS3_REASONER_NIM_IMAGE,
            "NEMOTRON_3_NANO_OMNI_NIM_IMAGE": NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
            "EMBEDDING_NIM_IMAGE": EMBEDDING_NIM_IMAGE,
        }

        assert {name: _shell_assignment(path, name) for name in expected} == expected
        assert set(_NIM_IMAGE_RE.findall(path.read_text())) == set(expected.values())

    def test_dev_setup_access_probe_uses_a_deployable_catalog_image(self):
        """The lightweight NGC probe must follow the pinned CR2 image tag."""
        path = _REPO_ROOT / "scripts" / "setup-dev.sh"
        assert (
            _shell_assignment(path, "ACCESS_TEST_IMAGE") == COSMOS_REASON2_2B_NIM_IMAGE
        )
        assert set(_NIM_IMAGE_RE.findall(path.read_text())) == {
            COSMOS_REASON2_2B_NIM_IMAGE
        }

    def test_local_setup_selection_floors_match_the_backend_catalog(self):
        """Shell hardware tiers must not diverge from onboarding."""
        path = _REPO_ROOT / "scripts" / "setup-local.sh"
        expected = {
            "COSMOS_REASON2_2B_GPU_MIN_GB": COSMOS_REASON2_2B_GPU_MIN_GB,
            "COSMOS3_NANO_REASONER_GPU_MIN_GB": COSMOS3_NANO_REASONER_GPU_MIN_GB,
            "NEMOTRON_3_NANO_OMNI_GPU_MIN_GB": NEMOTRON_3_NANO_OMNI_GPU_MIN_GB,
            "NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN": (
                NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN
            ),
        }

        assert {
            name: float(_shell_assignment(path, name)) for name in expected
        } == expected

    @pytest.mark.parametrize(
        ("gpu_inventory", "expected_image"),
        [
            ("81920, 9.0\n", NEMOTRON_3_NANO_OMNI_NIM_IMAGE),
            ("81920, 8.0\n", COSMOS3_REASONER_NIM_IMAGE),
            ("57344, 8.9\n", COSMOS3_REASONER_NIM_IMAGE),
            ("81920, N/A\n", COSMOS3_REASONER_NIM_IMAGE),
            ("40960, 8.9\n", COSMOS_REASON2_2B_NIM_IMAGE),
            ("24576, 8.9\n", ""),
            (
                "57344, 8.9\n81920, 9.0\n",
                NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
            ),
            # Match the backend's 1% nominal-capacity tolerance.
            ("81101, 9.0\n", NEMOTRON_3_NANO_OMNI_NIM_IMAGE),
        ],
    )
    def test_local_setup_selects_the_backend_quality_order(
        self, gpu_inventory: str, expected_image: str
    ):
        """Pre-pull chooses one quality-ranked Teacher across all GPUs."""
        path = _REPO_ROOT / "scripts" / "setup-local.sh"
        variable_names = (
            "COSMOS_REASON2_2B_NIM_IMAGE",
            "COSMOS3_REASONER_NIM_IMAGE",
            "NEMOTRON_3_NANO_OMNI_NIM_IMAGE",
            "COSMOS_REASON2_2B_GPU_MIN_GB",
            "COSMOS3_NANO_REASONER_GPU_MIN_GB",
            "NEMOTRON_3_NANO_OMNI_GPU_MIN_GB",
            "NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN",
        )
        program = "\n".join(
            [
                *(
                    f"{name}={shlex.quote(_shell_assignment(path, name))}"
                    for name in variable_names
                ),
                _shell_function(path, "vfl_gpu_memory_meets_model_floor"),
                _shell_function(path, "vfl_compute_capability_meets_model_floor"),
                _shell_function(path, "vfl_select_teacher_prepull_image"),
                "vfl_select_teacher_prepull_image",
            ]
        )

        result = subprocess.run(
            ["bash", "-c", program],
            input=gpu_inventory,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected_image

    def test_cr3_profile_precache_is_guarded_by_the_selected_teacher(self):
        """Omni and CR2 pre-pulls must not run the shared-image Nano recipe."""
        script = (_REPO_ROOT / "scripts" / "setup-local.sh").read_text()
        guard = 'if [ "$SELECTED_TEACHER_IMAGE" = "$COSMOS3_REASONER_NIM_IMAGE" ]; then'
        assert script.count("download-to-cache --profile") == 1
        assert script.index(guard) < script.index("download-to-cache --profile")


class TestLocalNimPrerequisiteFloorPins:
    def test_generated_handoff_matches_shared_setup_floors(self):
        """Operator handoffs and setup scripts must name identical minimums."""
        common = _REPO_ROOT / "scripts" / "setup-common.sh"
        floors = {
            "nvidia_driver": _shell_assignment(common, "VFL_NIM_MIN_DRIVER_VERSION"),
            "docker": _shell_assignment(common, "VFL_NIM_MIN_DOCKER_VERSION"),
            "nvidia_container_toolkit": _shell_assignment(
                common, "VFL_NIM_MIN_CTK_VERSION"
            ),
        }
        generated = generate_action_request(
            request_type="student_nim_deploy",
            project_name="Floor drift guard",
            project_id="unused",
            context={},
        )
        prerequisites = generated["technical_requirements"]["host_prerequisites"]

        for key, version in floors.items():
            assert prerequisites[key] == f"{version}+"
            assert f"{version}+" in generated["rendered_text"]


class TestSetupScriptSecretForwarding:
    def test_precache_docker_forwards_ngc_by_name(self):
        """The pre-cache container inherits NGC without a literal argv value."""
        script = (_REPO_ROOT / "scripts" / "setup-local.sh").read_text()
        assert re.search(r"-e\s+NGC_API_KEY(?:\s|\\)", script)
        assert not re.search(r"-e\s+NGC_API_KEY\s*=", script)


class TestLocalNimDriverFloor:
    @pytest.mark.parametrize(
        ("driver_version", "supported"),
        [
            ("570.195.03", False),
            ("580.65.05", False),
            ("580.65.06", True),
            ("590.44.01", True),
        ],
    )
    def test_shared_driver_check_enforces_cuda_13_floor(
        self, driver_version: str, supported: bool
    ):
        """Both GPU setup paths must reject drivers below the VLM NIM floor."""
        common_script = _REPO_ROOT / "scripts" / "setup-common.sh"
        result = subprocess.run(
            [
                "bash",
                "-c",
                '. "$1"; vfl_driver_meets_nim_minimum "$2"',
                "driver-check",
                str(common_script),
                driver_version,
            ],
            check=False,
        )
        assert (result.returncode == 0) is supported

    def test_setup_entry_points_use_the_shared_driver_check(self):
        """The tolerant dev bootstrap and strict NIM setup share one floor."""
        for filename in ("setup-dev.sh", "setup-local.sh"):
            script = (_REPO_ROOT / "scripts" / filename).read_text()
            assert "vfl_driver_meets_nim_minimum" in script
            assert "535.104.05" not in script


class TestLocalNimContainerRuntimeFloors:
    @staticmethod
    def _run_common(function: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        common_script = _REPO_ROOT / "scripts" / "setup-common.sh"
        return subprocess.run(
            [
                "bash",
                "-c",
                f'. "$1"; {function} "${{@:2}}"',
                "runtime-version-check",
                str(common_script),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    @pytest.mark.parametrize(
        ("function", "version", "supported"),
        [
            ("vfl_docker_meets_nim_minimum", "29.3.9", False),
            ("vfl_docker_meets_nim_minimum", "29.4.0", True),
            ("vfl_docker_meets_nim_minimum", "29.4.1", True),
            ("vfl_docker_meets_nim_minimum", "30.0.0", True),
            ("vfl_ctk_meets_nim_minimum", "1.18.0", False),
            ("vfl_ctk_meets_nim_minimum", "1.19.0", True),
            ("vfl_ctk_meets_nim_minimum", "1.19.1", True),
            ("vfl_ctk_meets_nim_minimum", "2.0.0", True),
        ],
    )
    def test_shared_runtime_checks_enforce_supported_floors(
        self, function: str, version: str, supported: bool
    ):
        """Setup accepts the boundary release and rejects older runtimes."""
        result = self._run_common(function, version)
        assert (result.returncode == 0) is supported

    @pytest.mark.parametrize(
        ("raw_version", "expected"),
        [
            ("29.4.0", "29.4.0"),
            ("29.4.0+azure-1", "29.4.0"),
            ("Docker version 29.4.0, build abc", "29.4.0"),
            ("NVIDIA Container Toolkit CLI version 1.19.0", "1.19.0"),
            ("1.19.0-1", "1.19.0"),
        ],
    )
    def test_version_parser_normalizes_supported_command_output(
        self, raw_version: str, expected: str
    ):
        """Package suffixes and command banners do not hide a valid release."""
        result = self._run_common("vfl_extract_version", raw_version)
        assert result.returncode == 0
        assert result.stdout.strip() == expected

    @pytest.mark.parametrize("raw_version", ["", "unknown", "29.4"])
    def test_version_parser_fails_closed_on_ambiguous_output(self, raw_version: str):
        """An unreadable runtime version cannot be treated as compatible."""
        result = self._run_common("vfl_extract_version", raw_version)
        assert result.returncode != 0

    def test_setup_entry_points_use_shared_runtime_checks(self):
        """Both setup policies consume one Docker and Toolkit compatibility rule."""
        for filename in ("setup-dev.sh", "setup-local.sh"):
            script = (_REPO_ROOT / "scripts" / filename).read_text()
            assert "vfl_docker_meets_nim_minimum" in script
            assert "vfl_ctk_meets_nim_minimum" in script
            assert "VFL_NIM_MIN_DOCKER_VERSION=" not in script
            assert "VFL_NIM_MIN_CTK_VERSION=" not in script

        dev_script = (_REPO_ROOT / "scripts" / "setup-dev.sh").read_text()
        assert "dpkg-query" not in dev_script

    def test_dev_setup_does_not_report_failed_gpu_passthrough_as_ready(self):
        """A failed container GPU probe must block the local-NIM summary."""
        script = (_REPO_ROOT / "scripts" / "setup-dev.sh").read_text()
        passthrough_branch = script.split(
            "if $DOCKER_TEST_CMD run --rm --runtime=nvidia --gpus all "
        )[1].split("        fi", maxsplit=1)[0]
        failure_branch = passthrough_branch.split("        else", maxsplit=1)[1]
        assert "GPU_RUNTIME_READY=false" in failure_branch


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

    def test_compose_forwards_quantization_calibration_size(self):
        """The documented Compose override must reach backend Settings."""
        compose = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text())
        environment = compose["services"]["backend"]["environment"]
        assert (
            "TAO_QUANTIZATION_CALIBRATION_SAMPLES="
            "${TAO_QUANTIZATION_CALIBRATION_SAMPLES:-128}" in environment
        )


class TestPublicConfigurationExamples:
    def test_examples_document_every_supported_setting_at_the_correct_scope(self):
        """Operators must be able to discover every accepted setting.

        A Settings field that silently disappears from both public examples is
        effectively private, while a secret shown in config.yaml encourages
        unsafe persistence. Keep the searchable catalog complete and scoped.
        """
        config_text = (_REPO_ROOT / "config.yaml.example").read_text()
        env_text = (_REPO_ROOT / ".env.example").read_text()
        config_keys = set(
            re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)\s*:", config_text, re.MULTILINE)
        )
        env_keys = set(re.findall(r"^#\s*([A-Z][A-Z0-9_]+)=", env_text, re.MULTILINE))

        expected_config = (set(DEFAULTS) - SECRET_KEYS) | {"WORKSPACE_ROOT"}
        assert config_keys >= expected_config
        assert config_keys.isdisjoint(SECRET_KEYS)
        assert env_keys >= SECRET_KEYS

        # Credential examples remain comments/placeholders, never active
        # assignments that a copied template could mistake for real values.
        for key in SECRET_KEYS:
            assert not re.search(rf"^{key}=", env_text, re.MULTILINE)


class TestContainerHealthProbeDependencySurface:
    def test_backend_runtime_uses_the_audited_minimal_base(self):
        """The published backend must not reintroduce the vulnerable Debian
        utility and Perl surface removed from its non-root runtime image."""
        dockerfile = (_REPO_ROOT / "src" / "backend" / "Dockerfile").read_text()
        runtime_marker = "FROM python:3.12-alpine3.24 AS runtime"
        builder, runtime = dockerfile.split(runtime_marker, maxsplit=1)

        assert "FROM python:3.12-alpine3.24 AS builder" in builder
        assert "apk add --no-cache build-base linux-headers cmake git" in builder
        assert "apk add --no-cache libstdc++" in runtime
        assert "build-base" not in runtime
        assert "linux-headers" not in runtime
        assert "cmake" not in runtime
        assert "git" not in runtime

    def test_backend_health_probes_reuse_python_without_curl(self):
        """The backend image must not install a second HTTP client solely to
        probe itself; Python is already present and avoids shipping curl's
        additional runtime dependency and vulnerability surface."""
        dockerfile = (_REPO_ROOT / "src" / "backend" / "Dockerfile").read_text()
        compose = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text())
        healthcheck = compose["services"]["backend"]["healthcheck"]["test"]

        assert "apt-get install" not in dockerfile
        assert "CMD python -c" in dockerfile
        assert "urllib.request.urlopen" in dockerfile
        assert healthcheck[:3] == ["CMD", "python", "-c"]
        assert "urllib.request.urlopen" in healthcheck[3]
        assert "curl" not in " ".join(healthcheck)


class TestBackendContainerSignalDelivery:
    def test_backend_execs_uvicorn_after_expanding_runtime_bind(self):
        """Docker stop must signal Uvicorn directly without hard-coding bind config."""
        dockerfile = (_REPO_ROOT / "src" / "backend" / "Dockerfile").read_text()
        command_line = next(
            line for line in dockerfile.splitlines() if line.startswith("CMD [")
        )
        command = json.loads(command_line.removeprefix("CMD "))

        assert command[:2] == ["sh", "-c"]
        assert command[2].startswith("exec uvicorn vlm_feedback_loop.main:app --host ")
        assert "${BIND_HOST}" in command[2]
        assert "${BIND_PORT}" in command[2]


class TestUiRuntimePrivilegeBoundary:
    def test_static_ui_drops_root_before_starting_nginx(self):
        """The port-3000 static UI has no reason to retain root at runtime."""
        dockerfile = (_REPO_ROOT / "src" / "ui" / "Dockerfile").read_text()

        assert "COPY --chown=nginx:nginx nginx.conf" in dockerfile
        assert "COPY --from=builder --chown=nginx:nginx" in dockerfile
        assert "chown -R nginx:nginx /var/cache/nginx /run" in dockerfile
        assert "sed -i '/^user  nginx;$/d'" in dockerfile
        assert "USER nginx" in dockerfile


class TestPublicIngestDocumentationContract:
    def test_docs_describe_async_phash_after_accepted_ingest(self):
        """Operators must not wait for hashes the 202 response does not carry."""
        overview = (_REPO_ROOT / "docs" / "Overview.md").read_text()
        brief = (_REPO_ROOT / "docs" / "Engineering_Spec_Brief.md").read_text()
        spec = (_REPO_ROOT / "docs" / "Engineering_Spec.md").read_text()

        assert "background sweep after ingest" in overview
        assert "phash=null" in brief
        assert "return **202 Accepted**" in spec
        for document in (overview, brief, spec):
            assert "pHash is computed inline" not in document
            assert "pHash computed inline" not in document

    def test_api_walkthrough_uses_uuid4_resource_ids(self):
        """Paste-ready API examples use IDs accepted by the public contract."""
        api = (_REPO_ROOT / "docs" / "API.md").read_text()
        project_ids = re.findall(r'"project_id": "([^"]+)"', api)
        invocation_ids = re.findall(r'"inference_invocation_id": "([^"]+)"', api)

        assert project_ids
        assert invocation_ids
        for value in project_ids + invocation_ids:
            parsed = UUID(value)
            assert parsed.version == 4


class TestHostedSeededModelsSmokeRoster:
    @staticmethod
    def _import_smoke_module():
        # Module-level code runs the roster drift guard (no I/O happens before
        # argparse in main()); a stale roster raises SystemExit here, which is
        # exactly the DOA state this test pins against.
        return _import_script_module("hosted_seeded_models_smoke.py")

    def test_roster_resolves_against_current_catalog(self):
        """The live smoke covers every hosted-compatible seeded Teacher.

        Local-only Cosmos entries must not be called through the hosted API,
        and a newly seeded hosted Teacher must not escape release acceptance.
        """
        module = self._import_smoke_module()
        catalog_names = {
            entry["model_name"]
            for entry in SEEDED_MODEL_CATALOG
            if "teacher" in entry["eligible_roles"]
            and entry.get("hosted_compatible", True)
        }
        covered = set(module._COVERED_MODEL_NAMES)
        assert module.SEEDED_MODELS, "smoke covers no models"
        assert covered == catalog_names

    def test_capability_expectations_come_from_current_catalog(self):
        """Every live capability verdict is compared with the seeded truth."""
        module = self._import_smoke_module()
        catalog = {entry["model_name"]: entry for entry in SEEDED_MODEL_CATALOG}
        for model in module.SEEDED_MODELS:
            entry = catalog[model.name]
            assert (
                model.structured_generation_support
                == entry["structured_generation_support"]
            )
            assert model.thinking_toggle_support == entry["thinking_toggle_support"]
            assert model.visual_budget_support == entry["visual_budget_support"]

    def test_hosted_deadline_matches_interactive_product_budget(self):
        """Slow but healthy hosted alternates are not failed by a 60s smoke cap."""
        module = self._import_smoke_module()
        assert module.DEADLINE_S >= 180.0

    def test_roster_covers_the_default_teacher(self):
        """The smoke's purpose is verifying the seeded hosted models an
        operator actually hits — at minimum the shipped default Teacher.
        This fails when a default-Teacher reseat forgets the roster."""
        module = self._import_smoke_module()
        assert DEFAULTS["DEFAULT_TEACHER_MODEL"] in module._COVERED_MODEL_NAMES


class TestLiveAcceptanceTeacherRosters:
    """Live release harnesses must not launch with retired hosted Teachers."""

    _HOSTED_TEACHERS = {
        "step": STEP_3_7_FLASH,
        "mistral_medium": MISTRAL_MEDIUM_3_5,
        "nemotron": NEMOTRON_NANO_12B_VL,
        "omni": NEMOTRON_3_NANO_OMNI_REASONING,
    }
    _DEFAULT_TEACHERS = ",".join(_HOSTED_TEACHERS)

    def test_hosted_harnesses_cover_current_canonical_teacher_matrix(self):
        """The no-argument hosted runs cover every supported seeded Teacher,
        start with Step, and cannot silently retain a retired alias."""
        cold = _import_script_module("cold_start_smoke.py")
        realistic = _import_script_module("icl_loop_realistic_smoke.py")
        profile_b = _import_script_module("profile_b_live_validation.py")

        assert cold.TEACHER_LOOKUP == self._HOSTED_TEACHERS
        assert {
            label: model_and_cap[0]
            for label, model_and_cap in realistic.TEACHER_LOOKUP.items()
        } == self._HOSTED_TEACHERS
        assert profile_b.TEACHER_LOOKUP == self._HOSTED_TEACHERS
        assert cold.DEFAULT_TEACHERS == self._DEFAULT_TEACHERS
        assert realistic.DEFAULT_TEACHERS == self._DEFAULT_TEACHERS
        assert profile_b.DEFAULT_TEACHER == "step"

    def test_baseline_icl_harness_adds_only_explicit_local_teachers(self):
        """The baseline smoke extends the hosted matrix only with the two
        explicit local Cosmos choices; retired hosted aliases stay absent."""
        module = _import_script_module("icl_loop_smoke.py")

        hosted = {
            label: model
            for label, model in module.TEACHER_LOOKUP.items()
            if not label.endswith("_local")
        }
        assert hosted == self._HOSTED_TEACHERS
        assert module.DEFAULT_TEACHERS == self._DEFAULT_TEACHERS
        assert module.TEACHER_LOOKUP["cosmos_8b_local"] == COSMOS_REASON2_8B
        assert module.TEACHER_LOOKUP["cosmos_2b_local"] == COSMOS_REASON2_2B

    def test_image_cap_probe_covers_current_hosted_teacher_matrix(self):
        """The cap probe must not default to a retired or partial roster."""
        module = _import_script_module("probe_hosted_image_caps.py")

        assert module.MODEL_LOOKUP == self._HOSTED_TEACHERS
        assert list(self._HOSTED_TEACHERS) == module.DEFAULT_MODEL_LABELS
        assert max(module.DEFAULT_LADDER) > max(
            entry["max_images_per_request"]
            for entry in SEEDED_MODEL_CATALOG
            if entry["model_name"] in self._HOSTED_TEACHERS.values()
        )

    def test_db_audit_harnesses_do_not_reconstruct_the_workspace(self):
        """Direct SQLite audits must follow the project directory returned by
        project creation instead of guessing a machine-specific workspace."""
        for filename in (
            "cold_start_smoke.py",
            "icl_loop_realistic_smoke.py",
            "profile_b_live_validation.py",
        ):
            source = (_REPO_ROOT / "scripts" / filename).read_text()
            assert "/tmp/vlm_workspace" not in source
            assert "WORKSPACE_ROOT" not in source

    def test_cold_start_uses_real_images_beneath_the_operator_root(self, tmp_path):
        """The cold-start harness must honor the selected RPS data root.

        This keeps its ingest references inside a containment-enabled
        deployment instead of silently creating fixtures under an unrelated
        temporary directory.
        """
        module = _import_script_module("cold_start_smoke.py")
        rock = tmp_path / "rock" / "testrock01-00.png"
        paper = tmp_path / "paper" / "testpaper01-00.png"
        rock.parent.mkdir()
        paper.parent.mkdir()
        rock.write_bytes(b"rock")
        paper.write_bytes(b"paper")

        assert module._resolve_rps_images(tmp_path) == [
            ("rock", rock),
            ("paper", paper),
        ]

        paper.unlink()
        with pytest.raises(FileNotFoundError, match="RPS image.*missing"):
            module._resolve_rps_images(tmp_path)

    def test_schema_evolution_uses_the_current_default_teacher(self):
        """The live schema-evolution journey must not pin a retired endpoint."""
        module = _import_script_module("schema_evolution_smoke.py")
        assert DEFAULTS["DEFAULT_TEACHER_MODEL"] == module.TEACHER_MODEL

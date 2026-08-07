# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the configuration system — the five-level precedence loader."""

from __future__ import annotations

import pytest

from vlm_feedback_loop.config import Settings, load_settings

# ── config.yaml missing → fail fast ────────────────────────────────────


class TestMissingConfigYaml:
    """Verify: backend fails fast with bootstrap message."""

    def test_missing_config_yaml_exits(self, patch_config_paths, tmp_config_dir):
        # config.yaml does not exist in the temp dir
        with pytest.raises(SystemExit) as exc_info:
            load_settings()
        assert exc_info.value.code == 1

    def test_missing_config_yaml_mentions_init(
        self, patch_config_paths, tmp_config_dir, capsys
    ):
        with pytest.raises(SystemExit):
            load_settings()
        captured = capsys.readouterr()
        assert "vlm-feedback-loop init" in captured.err


# ── Five-level precedence ──────────────────────────────────────────────


class TestFiveLevelPrecedence:
    """Verify: process env > explicit .env > canonical .env > yaml > default."""

    def test_all_five_levels(
        self,
        patch_config_paths,
        write_config,
        write_env,
        tmp_config_dir,
        tmp_path,
        monkeypatch,
    ):
        # Level 5: built-in default for LOG_LEVEL is "info"
        # Level 4: yaml overrides to "debug"
        write_config(overrides={"LOG_LEVEL": "debug"})
        # Level 3: canonical .env overrides to "warn"
        write_env({"LOG_LEVEL": "warn"})
        # Level 2: explicit .env overrides to "error"
        explicit_env = write_env({"LOG_LEVEL": "error"}, filename="custom.env")
        # Level 1: process env var overrides to "critical"
        monkeypatch.setenv("LOG_LEVEL", "critical")

        # With all 5 levels active, process env wins
        settings = load_settings(cli_env_file=str(explicit_env))
        assert settings.LOG_LEVEL == "critical"

    def test_remove_level_1(
        self,
        patch_config_paths,
        write_config,
        write_env,
        tmp_path,
        monkeypatch,
    ):
        """Remove process env → explicit .env wins."""
        write_config(overrides={"LOG_LEVEL": "debug"})
        write_env({"LOG_LEVEL": "warn"})
        explicit_env = write_env({"LOG_LEVEL": "error"}, filename="custom.env")
        # No monkeypatch.setenv — level 1 absent

        settings = load_settings(cli_env_file=str(explicit_env))
        assert settings.LOG_LEVEL == "error"

    def test_remove_levels_1_and_2(
        self,
        patch_config_paths,
        write_config,
        write_env,
    ):
        """Remove process env + explicit .env → canonical .env wins."""
        write_config(overrides={"LOG_LEVEL": "debug"})
        write_env({"LOG_LEVEL": "warn"})

        settings = load_settings()  # no cli_env_file
        assert settings.LOG_LEVEL == "warn"

    def test_remove_levels_1_2_3(
        self,
        patch_config_paths,
        write_config,
        tmp_config_dir,
    ):
        """Remove env + both .env → yaml wins."""
        write_config(overrides={"LOG_LEVEL": "debug"})
        # No .env written

        settings = load_settings()
        assert settings.LOG_LEVEL == "debug"

    def test_remove_levels_1_2_3_4(
        self,
        patch_config_paths,
        write_config,
    ):
        """Remove all overrides → built-in default wins."""
        write_config()  # no LOG_LEVEL override

        settings = load_settings()
        assert settings.LOG_LEVEL == "info"  # built-in default


# ── canonical .env loaded without explicit override ────────────────────


class TestCanonicalEnv:
    def test_canonical_env_loaded(self, patch_config_paths, write_config, write_env):
        write_config()
        write_env({"NVIDIA_API_KEY": "nvapi-test123"})

        settings = load_settings()  # no cli_env_file
        assert settings.NVIDIA_API_KEY == "nvapi-test123"

    def test_unknown_env_key_is_ignored_not_crash(
        self, patch_config_paths, write_config, write_env
    ):
        """A .env is often shared/reused and may carry unrelated shell vars.
        Under extra='forbid' an unknown key would crash startup; it must be
        filtered out (the same filter process env uses), while known keys
        still load."""
        write_config()
        write_env({"NVIDIA_API_KEY": "nvapi-ok", "SOME_UNRELATED_SHELL_VAR": "x"})

        settings = load_settings()  # must not raise SystemExit
        assert settings.NVIDIA_API_KEY == "nvapi-ok"

    def test_active_env_file_is_the_cli_env_file(
        self, patch_config_paths, write_config, write_env, tmp_path
    ):
        """get_canonical_env_file_path (the persist-write target) must return
        the SAME file the process loaded, honoring --env-file — otherwise a
        persisted secret is written to a different .env than the one read."""
        from vlm_feedback_loop.config import get_canonical_env_file_path

        write_config()
        explicit = write_env({"NVIDIA_API_KEY": "nvapi-x"}, filename="custom.env")
        load_settings(cli_env_file=str(explicit))
        assert get_canonical_env_file_path() == explicit


# ── --env-file overrides canonical location ────────────────────────────


class TestEnvFileOverride:
    def test_explicit_env_file_wins(
        self,
        patch_config_paths,
        write_config,
        write_env,
        tmp_path,
    ):
        write_config()
        # Canonical has one value
        write_env({"NVIDIA_API_KEY": "canonical-key"})
        # Explicit file has a different value
        custom_env = tmp_path / "custom.env"
        custom_env.write_text("NVIDIA_API_KEY=custom-key\n")

        settings = load_settings(cli_env_file=str(custom_env))
        assert settings.NVIDIA_API_KEY == "custom-key"

    def test_explicit_env_file_excludes_canonical(
        self,
        patch_config_paths,
        write_config,
        write_env,
        tmp_path,
    ):
        """When --env-file is set, canonical .env MUST NOT be loaded."""
        write_config()
        write_env({"NGC_API_KEY": "from-canonical"})
        # Explicit file does NOT contain NGC_API_KEY
        custom_env = tmp_path / "custom.env"
        custom_env.write_text("NVIDIA_API_KEY=from-custom\n")

        settings = load_settings(cli_env_file=str(custom_env))
        assert settings.NVIDIA_API_KEY == "from-custom"
        assert settings.NGC_API_KEY is None  # NOT loaded from canonical

    def test_vlm_feedback_loop_env_file_var(
        self,
        patch_config_paths,
        write_config,
        write_env,
        tmp_path,
        monkeypatch,
    ):
        """VLM_FEEDBACK_LOOP_ENV_FILE env var works like --env-file."""
        write_config()
        write_env({"NVIDIA_API_KEY": "canonical-key"})
        custom_env = tmp_path / "custom.env"
        custom_env.write_text("NVIDIA_API_KEY=env-var-key\n")
        monkeypatch.setenv("VLM_FEEDBACK_LOOP_ENV_FILE", str(custom_env))

        settings = load_settings()  # no cli_env_file arg
        assert settings.NVIDIA_API_KEY == "env-var-key"


# ── backend does NOT search CWD, WORKSPACE_ROOT, or project dirs ──────


class TestNoUnwantedEnvSearch:
    def test_cwd_env_not_loaded(
        self,
        patch_config_paths,
        write_config,
        tmp_workspace,
        tmp_path,
        monkeypatch,
    ):
        write_config()
        # Place .env in CWD
        cwd_env = tmp_path / ".env"
        cwd_env.write_text("NVIDIA_API_KEY=from-cwd\n")
        monkeypatch.chdir(tmp_path)

        settings = load_settings()
        assert settings.NVIDIA_API_KEY is None

    def test_workspace_env_not_loaded(
        self,
        patch_config_paths,
        write_config,
        tmp_workspace,
    ):
        write_config()
        # Place .env in WORKSPACE_ROOT
        (tmp_workspace / ".env").write_text("NVIDIA_API_KEY=from-workspace\n")

        settings = load_settings()
        assert settings.NVIDIA_API_KEY is None

    def test_project_dir_env_not_loaded(
        self,
        patch_config_paths,
        write_config,
        tmp_workspace,
    ):
        write_config()
        # Place .env in a project directory
        project_dir = tmp_workspace / "projects" / "test"
        project_dir.mkdir(parents=True)
        (project_dir / ".env").write_text("NVIDIA_API_KEY=from-project\n")

        settings = load_settings()
        assert settings.NVIDIA_API_KEY is None


# ── missing .env → startup continues ───────────────────────────────────


class TestMissingEnv:
    def test_no_env_file_startup_continues(self, patch_config_paths, write_config):
        write_config()
        # No .env written anywhere
        settings = load_settings()
        assert settings.NVIDIA_API_KEY is None
        assert settings.NGC_API_KEY is None
        assert settings.TAO_API_KEY is None

    def test_explicit_env_file_nonexistent_continues(
        self, patch_config_paths, write_config, tmp_path
    ):
        """When --env-file points to a nonexistent file, startup continues."""
        write_config()
        settings = load_settings(cli_env_file=str(tmp_path / "nonexistent.env"))
        assert settings.NVIDIA_API_KEY is None


# ── all built-in defaults present and correct ─────────────────────


class TestAllDefaults:
    def test_defaults_load_with_representative_values(
        self, patch_config_paths, write_config
    ):
        """A bare config.yaml loads and lands on the built-in defaults.

        One representative value per value *shape* — string, int, bool,
        None, nested preset dict, list. Echoing every individual default
        byte-for-byte would only ever fail when a default was changed on
        purpose; the full catalog lives in ``_defaults.py``. Loader
        behavior (precedence, coercion, unknown keys) is covered by the
        other classes in this file.
        """
        write_config()
        settings = load_settings()

        assert settings.EMBEDDING_PROVIDER == "auto"  # string
        assert settings.BIND_PORT == 8000  # int
        assert settings.THINKING_DEFAULT_ON is True  # bool
        assert settings.ICL_MAX_EXAMPLES is None  # nullable
        # Nested preset dict survives the YAML round-trip intact.
        assert settings.LABELING_PRESETS["precise"] == {
            "temperature": 0.0,
            "top_p": 1.0,
        }
        assert settings.STUDENT_LATENCY_TEST_CONCURRENCIES == [1, 8, 24]  # list
        assert settings.IMAGE_ROOT is None  # loopback resolves this to "/"


# ── type validation at load time ──────────────────────────────────────


class TestTypeValidation:
    def test_array_as_string_fails(self, patch_config_paths, write_config, capsys):
        """Array field set to a plain string → fail fast naming the key."""
        write_config(overrides={"STUDENT_LATENCY_TEST_CONCURRENCIES": "not_an_array"})
        with pytest.raises(SystemExit):
            load_settings()
        captured = capsys.readouterr()
        assert "STUDENT_LATENCY_TEST_CONCURRENCIES" in captured.err

    def test_int_as_string_coerced(self, patch_config_paths, write_config):
        """YAML delivers ints natively, but env-sourced string '1024' coerces."""
        write_config()
        settings = load_settings()
        assert isinstance(settings.EMBEDDING_DIM, int)

    def test_bool_as_string_from_env(
        self, patch_config_paths, write_config, monkeypatch
    ):
        """Boolean env values like 'true'/'false' should coerce correctly."""
        write_config()
        monkeypatch.setenv("THINKING_DEFAULT_ON", "false")
        settings = load_settings()
        assert settings.THINKING_DEFAULT_ON is False

    def test_list_from_env_json(self, patch_config_paths, write_config, monkeypatch):
        """List fields from env vars must be JSON-parseable."""
        write_config()
        monkeypatch.setenv("STUDENT_LATENCY_TEST_CONCURRENCIES", "[2, 4, 8]")
        settings = load_settings()
        assert settings.STUDENT_LATENCY_TEST_CONCURRENCIES == [2, 4, 8]

    @pytest.mark.parametrize(
        "field",
        [
            "EMBEDDING_DIM",
            "EMBEDDING_CONCURRENCY_HOSTED",
            "EMBEDDING_BATCH_SIZE_HOSTED",
            "EMBEDDING_CONCURRENCY_SELF_HOSTED",
            "EMBEDDING_BATCH_SIZE_SELF_HOSTED",
            "ICL_MAX_EXAMPLES",
            "BATCH_LABEL_RUN_LIMIT",
            "BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD",
            "BATCH_LABEL_CONCURRENCY_HOSTED",
            "BATCH_LABEL_CONCURRENCY_SELF_HOSTED",
            "EVAL_CONCURRENCY_HOSTED",
            "EVAL_CONCURRENCY_SELF_HOSTED",
            "NIM_STARTUP_TIMEOUT_S",
            "NIM_BENCHMARK_TIMEOUT_S",
            "HTTP_DEADLINE_INTERACTIVE_S",
            "HTTP_DEADLINE_BACKGROUND_S",
            "HTTP_MAX_RETRIES",
            "RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE",
            "BASE_OUTPUT_TOKENS_FLOOR",
            "RATIONALE_NOTE_ESTIMATE_TOKENS",
            "DEFAULT_UNBOUNDED_STRING_BUDGET",
            "MODEL_REASONING_HEADROOM_TOKENS",
        ],
    )
    def test_non_positive_operational_setting_fails_fast(
        self, field, patch_config_paths, write_config, capsys
    ):
        write_config(overrides={field: 0})

        with pytest.raises(SystemExit):
            load_settings()

        assert field in capsys.readouterr().err

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("JSON_STRUCTURAL_OVERHEAD_TOKENS", -1),
            ("RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN", 0),
            ("RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN", 1.1),
            ("MAX_OUTPUT_FRACTION", 0),
            ("MAX_OUTPUT_FRACTION", 1.1),
            ("MODEL_REASONING_HEADROOM_TOKENS", 4095),
            ("ICL_SIM_GAP", -0.1),
            ("ICL_ABS_THRESHOLD", -1.1),
            ("ICL_ABS_THRESHOLD", 1.1),
        ],
    )
    def test_bounded_inference_setting_fails_fast(
        self, field, value, patch_config_paths, write_config, capsys
    ):
        write_config(overrides={field: value})

        with pytest.raises(SystemExit):
            load_settings()

        assert field in capsys.readouterr().err

    @pytest.mark.parametrize(
        "presets",
        [
            {},
            {"precise": {"temperature": -0.1, "top_p": 1.0}},
            {"precise": {"temperature": 0.0, "top_p": 0.0}},
            {"precise": {"temperature": 0.0}},
            {"precise": {"temperature": 0.0, "top_p": 1.0, "seed": 1.0}},
        ],
    )
    def test_invalid_labeling_preset_fails_fast(
        self, presets, patch_config_paths, write_config, capsys
    ):
        write_config(overrides={"LABELING_PRESETS": presets})

        with pytest.raises(SystemExit):
            load_settings()

        assert "LABELING_PRESETS" in capsys.readouterr().err

    @pytest.mark.parametrize("value", [[], [1, 0], [1, -1]])
    def test_benchmark_concurrencies_require_positive_entries(
        self, value, patch_config_paths, write_config, capsys
    ):
        write_config(overrides={"STUDENT_LATENCY_TEST_CONCURRENCIES": value})

        with pytest.raises(SystemExit):
            load_settings()

        assert "STUDENT_LATENCY_TEST_CONCURRENCIES" in capsys.readouterr().err

    def test_invalid_operational_environment_override_fails_fast(
        self, patch_config_paths, write_config, monkeypatch, capsys
    ):
        write_config()
        monkeypatch.setenv("HTTP_MAX_RETRIES", "0")

        with pytest.raises(SystemExit):
            load_settings()

        assert "HTTP_MAX_RETRIES" in capsys.readouterr().err

    def test_nested_dict_preserved_from_yaml(self, patch_config_paths, write_config):
        """Visual budget presets (deeply nested) survive YAML round-trip."""
        write_config()
        settings = load_settings()
        balanced = settings.VISUAL_BUDGET_PRESETS["balanced"]
        assert balanced["mm_processor_size"]["size"] == {
            "shortest_edge": 1568,
            "longest_edge": 131072,
        }
        assert balanced["mm_processor_tiles"]["max_num_tiles"] == 16

    def test_unknown_key_rejected(self, patch_config_paths, write_config, capsys):
        """Extra keys in config.yaml are rejected (extra='forbid')."""
        write_config(overrides={"TOTALLY_UNKNOWN_KEY": "oops"})
        with pytest.raises(SystemExit):
            load_settings()
        captured = capsys.readouterr()
        assert "TOTALLY_UNKNOWN_KEY" in captured.err

    def test_image_root_must_be_absolute(
        self, patch_config_paths, write_config, capsys
    ):
        """A relative root would depend on the launch directory and is rejected."""
        write_config(overrides={"IMAGE_ROOT": "images"})
        with pytest.raises(SystemExit):
            load_settings()
        captured = capsys.readouterr()
        assert "IMAGE_ROOT must be an absolute path" in captured.err

    def test_removed_config_key_is_rejected(
        self, patch_config_paths, write_config, capsys
    ):
        """Current config files reject obsolete keys instead of ignoring them."""
        write_config(overrides={"ICL_SELECTION_POLICY": "relevance"})

        with pytest.raises(SystemExit):
            load_settings()

        assert "ICL_SELECTION_POLICY" in capsys.readouterr().err


# ── TAO workspace settings ────────────────────────────────────


class TestTaoWorkspaceSettings:
    """Only TAO workspace secrets belong in Settings."""

    def test_tao_workspace_secrets_load_from_env(
        self, patch_config_paths, write_config, monkeypatch
    ):
        """Process env loads the two secret fields."""
        write_config()
        monkeypatch.setenv("TAO_WORKSPACE_S3_ACCESS_KEY", "seaweedfs")
        monkeypatch.setenv("TAO_WORKSPACE_S3_SECRET_KEY", "seaweedfs123")

        settings = load_settings()
        assert settings.TAO_WORKSPACE_S3_ACCESS_KEY == "seaweedfs"
        assert settings.TAO_WORKSPACE_S3_SECRET_KEY == "seaweedfs123"

    def test_s3_secrets_are_in_secret_keys(self):
        """Both S3 credentials are registered as secrets (never in yaml)."""
        from vlm_feedback_loop._defaults import SECRET_KEYS

        assert "TAO_WORKSPACE_S3_ACCESS_KEY" in SECRET_KEYS
        assert "TAO_WORKSPACE_S3_SECRET_KEY" in SECRET_KEYS


class TestReservedVisualBudgetPresets:
    def test_native_preset_survives_operator_override(self, tmp_path):
        """An operator config.yaml that lists its own preset ladder replaces
        VISUAL_BUDGET_PRESETS wholesale; the reserved ``native`` preset
        (Student training-parity serving) must be merged back or every
        Student serving eval fails instantly on preset validation — observed
        live on a deployment whose config predated the preset."""
        s = Settings(
            WORKSPACE_ROOT=str(tmp_path),
            VISUAL_BUDGET_PRESETS={
                "fast": {"mm_processor_size": {"size": {"shortest_edge": 672}}}
            },
        )
        assert "native" in s.VISUAL_BUDGET_PRESETS
        assert s.VISUAL_BUDGET_PRESETS["native"] == {}
        assert "fast" in s.VISUAL_BUDGET_PRESETS

    def test_operator_visual_budget_values_are_preserved(self, tmp_path):
        """The loader enforces the reserved preset without rewriting overrides."""
        s = Settings(
            WORKSPACE_ROOT=str(tmp_path),
            VISUAL_BUDGET_PRESETS={
                "high_detail": {
                    "mm_processor_size": {
                        "size": {"shortest_edge": 1568, "longest_edge": 4096}
                    },
                    "mm_processor_pixels": {
                        "images_kwargs": {"min_pixels": 4, "max_pixels": 32768}
                    },
                    "mm_processor_tiles": {"max_num_tiles": 99},
                }
            },
        )
        high = s.VISUAL_BUDGET_PRESETS["high_detail"]
        assert high["mm_processor_size"]["size"] == {
            "shortest_edge": 1568,
            "longest_edge": 4096,
        }
        assert high["mm_processor_pixels"]["images_kwargs"] == {
            "min_pixels": 4,
            "max_pixels": 32768,
        }
        # A genuine operator customization is preserved.
        assert high["mm_processor_tiles"]["max_num_tiles"] == 99

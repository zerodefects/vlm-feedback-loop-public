# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the real boto3 construction path behind ``_boto3_shim``.

Every other TAO test fakes the S3 client (``support.FakeS3Client``) or
monkey-patches ``build_s3_client``, so this file is the only place the
shim's re-exports and the shared client factory are exercised for real.
Client construction is purely local — boto3 performs no network I/O
until an operation is invoked.
"""

from __future__ import annotations

import pytest

from conftest import make_stub_settings
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.services.tao_dataset_upload_service import (
    S3ClientProtocol,
    build_s3_client,
)

_ENDPOINT = "http://127.0.0.1:8333"


def _make_deployment_config(
    *, endpoint_external: str | None = _ENDPOINT
) -> TAODeploymentConfig:
    """In-memory TAODeploymentConfig with just what the factory reads."""
    return TAODeploymentConfig(
        tao_deployment_config_id="cfg-1",
        tao_workspace_id="ws-1",
        tao_workspace_bucket="vlm-bucket",
        tao_workspace_s3_endpoint_url_external=endpoint_external,
    )


def _make_settings_with_creds():
    return make_stub_settings(
        TAO_WORKSPACE_S3_ACCESS_KEY="test-access-key",
        TAO_WORKSPACE_S3_SECRET_KEY="test-secret-key",
    )


class TestRealClientConstruction:
    def test_client_satisfies_consumer_protocol_with_sigv4_and_bounded_retries(self):
        """The factory yields a genuine boto3 S3 client aimed at the external
        endpoint, speaking SigV4 with bounded standard retries, and exposing
        every operation both workspace-S3 consumers call — uploads via
        ``S3ClientProtocol``, artifact downloads via ``download_fileobj``.
        """
        import botocore.client

        client = build_s3_client(
            _make_deployment_config(), settings=_make_settings_with_creds()
        )

        assert isinstance(client, botocore.client.BaseClient)
        # @runtime_checkable Protocol: checks the upload method surface exists.
        assert isinstance(client, S3ClientProtocol)
        assert hasattr(client, "download_fileobj")  # polling artifact downloads
        assert client.meta.endpoint_url == _ENDPOINT
        assert client.meta.config.signature_version == "s3v4"
        # botocore normalizes max_attempts=3 to total_max_attempts=4
        # (initial try + 3 retries); "standard" mode bounds transient retries.
        assert client.meta.config.retries == {
            "mode": "standard",
            "total_max_attempts": 4,
        }

    def test_upload_path_needs_no_host_aws_configuration(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ):
        """The upload path omits ``region_name``; construction must succeed on
        a fresh Blueprint host with zero AWS setup (no env vars, no ~/.aws) —
        not raise ``NoRegionError``.
        """
        for var in ("AWS_DEFAULT_REGION", "AWS_REGION", "AWS_PROFILE"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
        monkeypatch.setenv(
            "AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-creds")
        )

        client = build_s3_client(
            _make_deployment_config(), settings=_make_settings_with_creds()
        )

        assert client.meta.endpoint_url == _ENDPOINT

    def test_explicit_region_overrides_host_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The polling path pins ``region_name="us-east-1"`` so SigV4 signing
        stays consistent with the workspace bucket even when the host has a
        different AWS default region configured.
        """
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-3")

        client = build_s3_client(
            _make_deployment_config(),
            settings=_make_settings_with_creds(),
            region_name="us-east-1",
        )

        assert client.meta.region_name == "us-east-1"


class TestFailFastGuardrails:
    @pytest.mark.parametrize(
        ("endpoint", "settings_kwargs", "match"),
        [
            pytest.param(
                None,
                {
                    "TAO_WORKSPACE_S3_ACCESS_KEY": "ak",
                    "TAO_WORKSPACE_S3_SECRET_KEY": "sk",
                },
                "tao_workspace_s3_endpoint_url_external",
                id="missing-endpoint",
            ),
            pytest.param(
                _ENDPOINT,
                {},
                "TAO_WORKSPACE_S3_ACCESS_KEY / TAO_WORKSPACE_S3_SECRET_KEY",
                id="missing-credentials",
            ),
        ],
    )
    def test_misconfiguration_raises_actionable_value_error(
        self, endpoint: str | None, settings_kwargs: dict[str, str], match: str
    ):
        """A host missing the workspace endpoint or S3 credentials gets a
        ``ValueError`` naming the missing setting — not a confusing boto3
        auth/connection failure mid-upload. ``tao_polling_service`` documents
        and relies on this exact contract.
        """
        with pytest.raises(ValueError, match=match):
            build_s3_client(
                _make_deployment_config(endpoint_external=endpoint),
                settings=make_stub_settings(**settings_kwargs),
            )

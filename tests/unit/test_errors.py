# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the centralized API error helpers."""

from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from vlm_feedback_loop.services.errors import (
    APIError,
    conflict,
    map_service_error,
    validation_failed,
)


class TestHelperStatusCodes:
    """Each helper produces the expected HTTP status code and detail."""

    def test_validation_failed_is_400(self) -> None:
        err = validation_failed("teacher_model_config_id missing")
        assert err.status_code == 400
        assert err.detail == "teacher_model_config_id missing"

    def test_conflict_is_409(self) -> None:
        err = conflict("proposal superseded by newer retry")
        assert err.status_code == 409
        assert err.detail == "proposal superseded by newer retry"


class TestAPIErrorBehaviour:
    """APIError integrates cleanly with FastAPI's exception handling."""

    def test_is_http_exception_subclass(self) -> None:
        assert issubclass(APIError, HTTPException)
        assert isinstance(conflict("x"), HTTPException)

    def test_is_raisable(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            raise conflict("stale proposal")
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == "stale proposal"

    def test_logs_at_construction(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="vlm_feedback_loop.errors")
        validation_failed("bad request body")
        matching = [
            r
            for r in caplog.records
            if r.name == "vlm_feedback_loop.errors"
            and "api_error" in r.message
            and "status=400" in r.message
            and "bad request body" in r.message
        ]
        assert len(matching) == 1, "expected exactly one log entry at construction"

    def test_each_construction_logs_independently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="vlm_feedback_loop.errors")
        validation_failed("c")
        conflict("d")
        entries = [
            r
            for r in caplog.records
            if r.name == "vlm_feedback_loop.errors" and "api_error" in r.message
        ]
        assert len(entries) == 2


class TestMapServiceErrorPrecedence:
    """An explicit prefix is the author's classification and always wins.

    The substring fallback scans the whole message, and service messages
    interpolate user-controlled content — a legal enum vocabulary like
    ['found', 'not found'] or a value like 'insufficient_stock' inside a
    validation error must not flip the status to 404/409."""

    def test_validation_prefix_beats_not_found_in_user_vocabulary(self) -> None:
        msg = (
            "validation: label values do not match the schema — "
            "'label_state': value 'unknown' not in allowed values "
            "['found', 'not found']"
        )
        assert map_service_error(msg).status_code == 400

    def test_validation_prefix_beats_conflict_markers_in_user_vocabulary(
        self,
    ) -> None:
        msg = (
            "validation: label values do not match the schema — "
            "'stock': value 'x' not in allowed values "
            "['in_stock', 'insufficient_stock']"
        )
        assert map_service_error(msg).status_code == 400

    def test_conflict_prefix_wins_over_not_found_substring(self) -> None:
        msg = "conflict: proposal artifact for invocation abc is missing or unreadable"
        assert map_service_error(msg).status_code == 409

    def test_unprefixed_legacy_strings_keep_substring_fallback(self) -> None:
        assert map_service_error("Example not found: img_001").status_code == 404
        assert (
            map_service_error("run is already terminal; cannot cancel").status_code
            == 409
        )
        assert map_service_error("something else went wrong").status_code == 400


class TestFastAPIIntegration:
    """When raised inside an endpoint, APIError serializes as {'detail': ...}
    with the carried status code — identical to HTTPException behaviour."""

    @pytest.fixture
    def client(self) -> TestClient:
        from fastapi import FastAPI

        app = FastAPI()

        @app.get("/unreachable")
        def _unreachable() -> None:
            raise map_service_error("tao_unreachable: nim down")

        @app.get("/timeout")
        def _timeout() -> None:
            raise map_service_error("tao_timeout: deadline")

        @app.get("/bad")
        def _bad() -> None:
            raise validation_failed("bad body")

        @app.get("/conflict")
        def _conflict() -> None:
            raise conflict("stale")

        return TestClient(app)

    def test_503_response_shape(self, client: TestClient) -> None:
        res = client.get("/unreachable")
        assert res.status_code == 503
        assert res.json() == {"detail": "tao_unreachable: nim down"}

    def test_504_response_shape(self, client: TestClient) -> None:
        res = client.get("/timeout")
        assert res.status_code == 504
        assert res.json() == {"detail": "tao_timeout: deadline"}

    def test_400_response_shape(self, client: TestClient) -> None:
        res = client.get("/bad")
        assert res.status_code == 400
        assert res.json() == {"detail": "bad body"}

    def test_409_response_shape(self, client: TestClient) -> None:
        res = client.get("/conflict")
        assert res.status_code == 409
        assert res.json() == {"detail": "stale"}

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one service-error → HTTP-status classifier.

map_service_error replaced seven per-router substring mappers that had
drifted apart (the same error produced 400 on one endpoint and 422 on
another). These tests pin the consolidated rules; if a rule changes, every
endpoint changes together — that is the point.
"""

from __future__ import annotations

import pytest

from vlm_feedback_loop.services.errors import (
    APIError,
    map_service_error,
)


class TestPrefixRules:
    def test_not_found_prefix_is_404(self):
        assert map_service_error("not found: Project p1").status_code == 404

    def test_prose_not_found_is_404(self):
        # Older services say "Project not found: p1" rather than the prefix.
        assert map_service_error("Project not found: p1").status_code == 404

    def test_conflict_prefix_is_409(self):
        assert (
            map_service_error("conflict: Run r1 already in terminal state").status_code
            == 409
        )

    def test_superseded_is_409(self):
        assert (
            map_service_error(
                "Stale proposal conflict: invocation superseded by later retry"
            ).status_code
            == 409
        )

    def test_insufficient_is_409(self):
        assert map_service_error("Insufficient examples: 2 of 5").status_code == 409

    def test_validation_prefix_is_400(self):
        assert (
            map_service_error("validation: invalid training_preset 'x'").status_code
            == 400
        )

    def test_unprefixed_domain_error_falls_back_to_400(self):
        """422 is reserved for Pydantic schema validation (CLAUDE.md error
        contract); every service-produced string is a domain error → 400."""
        assert map_service_error("eligible_roles must be non-empty").status_code == 400
        assert (
            map_service_error("invalid generation_preset_key 'bogus'").status_code
            == 400
        )

    def test_no_active_guidance_maps_400_everywhere(self):
        """Pre-consolidation, this exact string was 400 on batch/eval/prompt
        endpoints but 422 on training-suite create — the audit's marquee
        divergence example."""
        assert map_service_error("No active Guidance configured").status_code == 400


class TestMachineTokens:
    @pytest.mark.parametrize(
        ("token_string", "expected"),
        [
            ("validation: VISUAL_BUDGET_PARITY_VIOLATION (suite=s1)", 400),
            ("conflict: INFERENCE_CONTRACT_MISMATCH", 409),
            ("conflict: student_nim_not_deployed:sm-1", 409),
            ("conflict: tao_eval_already_in_suite:run-1", 409),
            ("conflict: student_nim_serving_run_missing:sm-1", 409),
            (
                "tao_dataset_upload_failed: retry training-suite creation",
                409,
            ),
        ],
    )
    def test_token_statuses(self, token_string: str, expected: int):
        assert map_service_error(token_string).status_code == expected


class TestUpstreamContract:
    """503/504/502 for upstream TAO failures — the documented contract
    (docs/API.md error table) that previously had no implementation."""

    def test_nim_unreachable_is_503(self):
        assert (
            map_service_error(
                "nim_unreachable: selected Teacher endpoint is disabled"
            ).status_code
            == 503
        )

    def test_tao_unreachable_is_503(self):
        assert (
            map_service_error("tao_unreachable: TAO cancel failed: ...").status_code
            == 503
        )

    def test_tao_timeout_is_504(self):
        assert (
            map_service_error("tao_timeout: TAO cancel failed: ...").status_code == 504
        )

    def test_tao_provider_error_is_502(self):
        assert (
            map_service_error("tao_error: TAO configuration incomplete").status_code
            == 502
        )


class TestPrecedence:
    def test_not_found_wins_over_validation_fallback(self):
        assert (
            map_service_error("not found: dataset_export(s) not found: a").status_code
            == 404
        )

    def test_token_wins_over_prose(self):
        # The conflict prefix and the token agree here, but the token rule
        # must fire first so token semantics never depend on prose wording.
        err = map_service_error("conflict: INFERENCE_CONTRACT_MISMATCH")
        assert isinstance(err, APIError)
        assert err.status_code == 409

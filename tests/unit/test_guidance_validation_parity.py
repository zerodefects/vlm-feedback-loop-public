# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend↔frontend contract for the reserved guidance field name.

Guidance schema validation is backend-only: the builder UI renders issues
straight from ``guidance:validate_draft`` and re-implements no rules in
TypeScript (the former client-side ruleset and its code/regex parity checks
were removed with it). The ONE remaining cross-language constant is the
reserved field name ``rationale_note``: the frontend uses it to lock the
reserved row, pin its ordering, and detect its absence, so it must match
``schema_core.RESERVED_FIELD_NAME`` exactly. This test parses the committed
TypeScript source and fails when either side drifts.
"""

from __future__ import annotations

import re
from pathlib import Path

from vlm_feedback_loop.services import schema_core

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO_ROOT / "src" / "ui" / "src" / "lib" / "guidance-templates.ts"


class TestGuidanceReservedFieldParity:
    def test_reserved_field_name_matches_backend(self):
        assert _TEMPLATES.is_file(), (
            f"parity anchor moved: {_TEMPLATES} not found — update this test"
        )
        ts = _TEMPLATES.read_text(encoding="utf-8")
        m = re.search(r'RATIONALE_NOTE_FIELD_NAME = "([^"]+)"', ts)
        assert m, "RATIONALE_NOTE_FIELD_NAME not found in guidance-templates.ts"
        assert m.group(1) == schema_core.RESERVED_FIELD_NAME

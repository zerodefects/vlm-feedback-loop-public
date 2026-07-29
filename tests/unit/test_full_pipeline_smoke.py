# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the final-integration-checkpoint script's pure logic.

The HTTP orchestration in ``run_full_pipeline_smoke()`` is exercised by the
hosted-NIM integration test; this file covers the deterministic bits that
don't need a backend:

  * ``_validate_cosmos_rl_archive`` — the Cosmos-RL wire-format check applied to
    on-disk ``.tar.gz`` archives.
  * ``_ground_truth_for_image`` — deterministic filename-derived ground
    truth (rock_*, paper_*, scissors_* + hash bucket fallback).

These are the bits where regressions would silently hide a Cosmos-RL
contract violation behind a green CI run.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from full_pipeline_smoke import (  # noqa: E402
    _ground_truth_for_image,
    _validate_cosmos_rl_archive,
)


def _write_archive(
    tmp_path: Path, annotations: object, *, archive_name: str = "ds.tar.gz"
) -> Path:
    """Synthesize a .tar.gz with the given annotations.json contents."""
    payload = json.dumps(annotations).encode("utf-8")
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_bytes(payload)
    archive_path = tmp_path / archive_name
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(annotations_path, arcname="annotations.json")
    return archive_path


# ── Valid archive ───────────────────────────────────────────────────────────


_GOOD_SAMPLE = {
    "id": "ex-1",
    "images": ["images/ex-1.png"],
    "conversations": [
        {"from": "human", "value": "<image>\nClassify the gesture."},
        {"from": "gpt", "value": json.dumps({"gesture": "rock"})},
    ],
}


def test_validates_well_formed_archive(tmp_path):
    archive = _write_archive(tmp_path, [_GOOD_SAMPLE])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok, detail
    assert "validated 1 samples" in detail


def test_validates_multiple_samples(tmp_path):
    samples = [
        {**_GOOD_SAMPLE, "id": f"ex-{i}", "images": [f"images/ex-{i}.png"]}
        for i in range(1, 6)
    ]
    archive = _write_archive(tmp_path, samples)
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok, detail


# ── Missing / unreadable archive ────────────────────────────────────────────


def test_missing_archive_fails(tmp_path):
    ok, detail = _validate_cosmos_rl_archive(tmp_path / "nope.tar.gz")
    assert ok is False
    assert "not found" in detail


def test_archive_without_annotations_fails(tmp_path):
    archive_path = tmp_path / "empty.tar.gz"
    junk = tmp_path / "junk.txt"
    junk.write_text("not annotations")
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(junk, arcname="junk.txt")
    ok, detail = _validate_cosmos_rl_archive(archive_path)
    assert ok is False
    assert "annotations.json" in detail


def test_archive_with_invalid_json_fails(tmp_path):
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text("{not json")
    archive_path = tmp_path / "bad.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(annotations_path, arcname="annotations.json")
    ok, detail = _validate_cosmos_rl_archive(archive_path)
    assert ok is False
    assert "parse error" in detail or "JSONDecodeError" in detail


# ── Top-level shape ─────────────────────────────────────────────────────────


def test_rejects_top_level_object_not_array(tmp_path):
    archive = _write_archive(tmp_path, {"not": "an array"})
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "top-level array" in detail


def test_rejects_empty_array(tmp_path):
    archive = _write_archive(tmp_path, [])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "empty" in detail


# ── Per-sample required fields ──────────────────────────────────────────────


def test_rejects_sample_missing_id(tmp_path):
    sample = {**_GOOD_SAMPLE}
    sample.pop("id")
    archive = _write_archive(tmp_path, [sample])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "missing 'id'" in detail


@pytest.mark.parametrize(
    "images_value, expected_msg",
    [
        ("images/ex-1.png", "must be a length-1 array"),  # str, not list
        ([], "must be a length-1 array"),
        (
            ["images/a.png", "images/b.png"],
            "must be a length-1 array",
        ),
    ],
)
def test_rejects_bad_images_shape(tmp_path, images_value, expected_msg):
    sample = {**_GOOD_SAMPLE, "images": images_value}
    archive = _write_archive(tmp_path, [sample])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert expected_msg in detail


def test_rejects_wrong_conversation_length(tmp_path):
    sample = {**_GOOD_SAMPLE, "conversations": [_GOOD_SAMPLE["conversations"][0]]}
    archive = _write_archive(tmp_path, [sample])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "length-2 array" in detail


def test_rejects_wrong_turn_1_role(tmp_path):
    sample = {
        **_GOOD_SAMPLE,
        "conversations": [
            {"from": "system", "value": "<image>\nfoo"},
            _GOOD_SAMPLE["conversations"][1],
        ],
    }
    archive = _write_archive(tmp_path, [sample])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "from=human" in detail


def test_rejects_human_turn_missing_image_token(tmp_path):
    sample = {
        **_GOOD_SAMPLE,
        "conversations": [
            {"from": "human", "value": "Classify."},  # no <image>
            _GOOD_SAMPLE["conversations"][1],
        ],
    }
    archive = _write_archive(tmp_path, [sample])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "<image>" in detail


def test_rejects_wrong_turn_2_role(tmp_path):
    sample = {
        **_GOOD_SAMPLE,
        "conversations": [
            _GOOD_SAMPLE["conversations"][0],
            {"from": "assistant", "value": "{}"},
        ],
    }
    archive = _write_archive(tmp_path, [sample])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "from=gpt" in detail


def test_rejects_gpt_value_not_string(tmp_path):
    """The gpt turn's value MUST be a JSON string, not a nested object."""
    sample = {
        **_GOOD_SAMPLE,
        "conversations": [
            _GOOD_SAMPLE["conversations"][0],
            {"from": "gpt", "value": {"gesture": "rock"}},  # wrong: dict, not string
        ],
    }
    archive = _write_archive(tmp_path, [sample])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "JSON string" in detail


def test_rejects_gpt_value_unparseable_json(tmp_path):
    sample = {
        **_GOOD_SAMPLE,
        "conversations": [
            _GOOD_SAMPLE["conversations"][0],
            {"from": "gpt", "value": "{not parseable"},
        ],
    }
    archive = _write_archive(tmp_path, [sample])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "not valid JSON" in detail


def test_rejects_gpt_value_parses_to_array(tmp_path):
    """The gpt turn's parsed value MUST be an object."""
    sample = {
        **_GOOD_SAMPLE,
        "conversations": [
            _GOOD_SAMPLE["conversations"][0],
            {"from": "gpt", "value": json.dumps(["rock"])},
        ],
    }
    archive = _write_archive(tmp_path, [sample])
    ok, detail = _validate_cosmos_rl_archive(archive)
    assert ok is False
    assert "not an object" in detail


# ── Ground truth derivation ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("rock_001.png", "rock"),
        ("paper_42.jpg", "paper"),
        ("scissors_a.webp", "scissors"),
        ("ROCK_HASH.png", "rock"),  # case-insensitive
    ],
)
def test_ground_truth_from_prefixed_filename(filename, expected):
    gt = _ground_truth_for_image(Path(filename))
    assert gt == {"gesture": expected}


def test_ground_truth_falls_back_to_hash_bucket():
    """Non-prefixed files get a deterministic bucket assignment."""
    gt1 = _ground_truth_for_image(Path("img-abc.png"))
    gt2 = _ground_truth_for_image(Path("img-abc.png"))
    assert gt1 == gt2  # deterministic
    assert gt1["gesture"] in {"rock", "paper", "scissors"}

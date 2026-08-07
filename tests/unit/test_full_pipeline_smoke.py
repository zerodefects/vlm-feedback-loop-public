# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the final-integration-checkpoint script's pure logic.

The HTTP orchestration in ``run_full_pipeline_smoke()`` is exercised by the
hosted-NIM integration test; this file covers the deterministic bits that
don't need a backend:

  * public dataset-export download metadata, checksums, and Cosmos-RL content.
  * exact per-item ingestion and requested-label cardinality.
  * image discovery and deterministic class-balanced cohort selection.
  * ``_ground_truth_for_image`` — class-directory/filename-derived ground
    truth with a deterministic fallback.

These are the bits where regressions would silently hide a Cosmos-RL
contract violation behind a green CI run.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from full_pipeline_smoke import (  # noqa: E402
    _example_key_for_image,
    _gather_image_paths,
    _ground_truth_for_image,
    _parse_args,
    _step_dataset_export,
    _step_ingest,
    _step_label_loop,
    _validate_archive_download,
    _validate_cosmos_rl_archive,
)


def _write_archive(
    tmp_path: Path, annotations: object, *, archive_name: str = "ds.tar.gz"
) -> bytes:
    """Synthesize a .tar.gz with the given annotations.json contents."""
    payload = json.dumps(annotations).encode("utf-8")
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_bytes(payload)
    archive_path = tmp_path / archive_name
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(annotations_path, arcname="annotations.json")
    return archive_path.read_bytes()


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


def test_validates_public_archive_download_contract(tmp_path):
    archive = _write_archive(tmp_path, [_GOOD_SAMPLE], archive_name="export.tar.gz")
    checksum = hashlib.sha256(archive).hexdigest()
    response = httpx.Response(
        200,
        content=archive,
        headers={
            "Content-Type": "application/gzip",
            "Content-Disposition": 'attachment; filename="export.tar.gz"',
            "X-Checksum-SHA256": checksum,
        },
    )

    ok, detail = _validate_archive_download(
        response,
        expected_checksum=checksum,
        expected_filename="export.tar.gz",
    )

    assert ok, detail
    assert "checksum-validated" in detail


@pytest.mark.parametrize(
    ("header_name", "header_value", "expected_detail"),
    [
        ("Content-Type", "application/zip", "Content-Type"),
        ("Content-Disposition", 'inline; filename="export.tar.gz"', "Disposition"),
        ("X-Checksum-SHA256", "0" * 64, "X-Checksum-SHA256"),
    ],
)
def test_rejects_invalid_public_archive_headers(
    tmp_path, header_name, header_value, expected_detail
):
    archive = _write_archive(tmp_path, [_GOOD_SAMPLE], archive_name="export.tar.gz")
    checksum = hashlib.sha256(archive).hexdigest()
    headers = {
        "Content-Type": "application/gzip",
        "Content-Disposition": 'attachment; filename="export.tar.gz"',
        "X-Checksum-SHA256": checksum,
    }
    headers[header_name] = header_value
    response = httpx.Response(200, content=archive, headers=headers)

    ok, detail = _validate_archive_download(
        response,
        expected_checksum=checksum,
        expected_filename="export.tar.gz",
    )

    assert ok is False
    assert expected_detail in detail


def test_rejects_archive_bytes_that_do_not_match_export_checksum(tmp_path):
    archive = _write_archive(tmp_path, [_GOOD_SAMPLE], archive_name="export.tar.gz")
    expected_checksum = hashlib.sha256(archive).hexdigest()
    response = httpx.Response(
        200,
        content=archive + b"corrupt",
        headers={
            "Content-Type": "application/gzip",
            "Content-Disposition": 'attachment; filename="export.tar.gz"',
            "X-Checksum-SHA256": expected_checksum,
        },
    )

    ok, detail = _validate_archive_download(
        response,
        expected_checksum=expected_checksum,
        expected_filename="export.tar.gz",
    )

    assert ok is False
    assert "archive bytes" in detail


@pytest.mark.asyncio
async def test_dataset_export_downloads_archive_through_public_endpoint(tmp_path):
    archive = _write_archive(tmp_path, [_GOOD_SAMPLE], archive_name="export.tar.gz")
    checksum = hashlib.sha256(archive).hexdigest()
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "dataset_export_id": "export-1",
                    "status": "completed",
                    "example_count": 1,
                    "artifact_refs": {
                        "archive_path": "/backend/exports/export.tar.gz",
                        "checksum_sha256": checksum,
                    },
                },
            )
        return httpx.Response(
            200,
            content=archive,
            headers={
                "Content-Type": "application/gzip",
                "Content-Disposition": 'attachment; filename="export.tar.gz"',
                "X-Checksum-SHA256": checksum,
            },
        )

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        result, downloaded = await _step_dataset_export(
            client,
            project_id="project-1",
            export_field_mode="all",
        )

    assert result.ok, result.detail
    assert downloaded == archive
    assert requested_paths[-1] == (
        "/v1/projects/project-1/dataset_exports/export-1/archive"
    )


@pytest.mark.asyncio
async def test_ingest_uses_unique_resolved_path_keys_for_same_stem(tmp_path):
    image_paths = [
        tmp_path / "rock" / "shared.png",
        tmp_path / "paper" / "shared.png",
    ]
    captured_items: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_items.extend(body["examples"])
        return httpx.Response(
            202,
            json={
                "results": [
                    {"example_key": item["example_key"], "status": "created"}
                    for item in body["examples"]
                ]
            },
        )

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        result = await _step_ingest(
            client,
            project_id="project-1",
            image_paths=image_paths,
        )

    keys = [item["example_key"] for item in captured_items]
    assert result.ok, result.detail
    assert keys == [_example_key_for_image(path) for path in image_paths]
    assert len(set(keys)) == len(image_paths)
    assert all(len(key) == 64 and key.isalnum() for key in keys)
    assert all("/" not in key and "\\" not in key for key in keys)


@pytest.mark.asyncio
async def test_ingest_rejects_non_created_per_item_result(tmp_path):
    image_paths = [tmp_path / "rock.png", tmp_path / "paper.png"]

    def handler(request: httpx.Request) -> httpx.Response:
        items = json.loads(request.content)["examples"]
        return httpx.Response(
            202,
            json={
                "results": [
                    {"example_key": items[0]["example_key"], "status": "created"},
                    {"example_key": items[1]["example_key"], "status": "exists"},
                ]
            },
        )

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        result = await _step_ingest(
            client,
            project_id="project-1",
            image_paths=image_paths,
        )

    assert result.ok is False
    assert "2 per-item created results" in result.detail
    assert result.metrics["created"] == 1


@pytest.mark.asyncio
async def test_label_loop_rejects_fewer_queried_examples_than_requested():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected HTTP request: {request.url}")

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        result = await _step_label_loop(
            client,
            project_id="project-1",
            examples=[{"example_key": "only-one"}],
            image_paths=[],
            expected_count=2,
        )

    assert result.ok is False
    assert result.metrics == {"queried": 1, "requested": 2, "saved": 0}
    assert "queried 1 unlabeled examples; expected 2" in result.detail


@pytest.mark.asyncio
async def test_label_loop_fails_when_fewer_labels_are_saved(tmp_path):
    image_paths = [tmp_path / "rock_1.png", tmp_path / "paper_1.png"]
    examples = [
        {"example_key": f"example-{index}", "storage_ref": str(path.resolve())}
        for index, path in enumerate(image_paths)
    ]
    save_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal save_calls
        if request.url.path.endswith("/proposals"):
            return httpx.Response(
                200,
                json={
                    "inference_invocation_id": f"invocation-{save_calls}",
                    "invocation_status": "completed",
                    "schema_valid_core": True,
                    "proposal_json": {"gesture": "rock"},
                },
            )
        save_calls += 1
        if save_calls == 2:
            return httpx.Response(500, text="save failed")
        return httpx.Response(200, json={"label_id": "label-1"})

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        result = await _step_label_loop(
            client,
            project_id="project-1",
            examples=examples,
            image_paths=image_paths,
            expected_count=2,
        )

    assert result.ok is False
    assert result.metrics["saved"] == 1
    assert "save failed" in result.detail


# ── Missing / unreadable archive ────────────────────────────────────────────


def test_empty_archive_bytes_fail():
    ok, detail = _validate_cosmos_rl_archive(b"")
    assert ok is False
    assert "parse error" in detail


def test_archive_without_annotations_fails(tmp_path):
    archive_path = tmp_path / "empty.tar.gz"
    junk = tmp_path / "junk.txt"
    junk.write_text("not annotations")
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(junk, arcname="junk.txt")
    ok, detail = _validate_cosmos_rl_archive(archive_path.read_bytes())
    assert ok is False
    assert "annotations.json" in detail


def test_archive_with_invalid_json_fails(tmp_path):
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text("{not json")
    archive_path = tmp_path / "bad.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(annotations_path, arcname="annotations.json")
    ok, detail = _validate_cosmos_rl_archive(archive_path.read_bytes())
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
    "image_path, expected",
    [
        ("rock_001.png", "rock"),
        ("paper_42.jpg", "paper"),
        ("scissors_a.webp", "scissors"),
        ("ROCK_HASH.png", "rock"),
        ("testrock04-11.png", "rock"),
        ("testpaper04-08.png", "paper"),
        ("testscissors02-01.png", "scissors"),
        ("paper/unknown-name.png", "paper"),
        ("scissors/rock_misleading.png", "scissors"),
    ],
)
def test_ground_truth_uses_truthful_path_conventions(image_path, expected):
    gt = _ground_truth_for_image(Path(image_path))
    assert gt == {"gesture": expected}


def test_ground_truth_falls_back_to_hash_bucket():
    """Non-prefixed files get a deterministic bucket assignment."""
    gt1 = _ground_truth_for_image(Path("img-abc.png"))
    gt2 = _ground_truth_for_image(Path("img-abc.png"))
    assert gt1 == gt2  # deterministic
    assert gt1["gesture"] in {"rock", "paper", "scissors"}


# ── Image cohort discovery ─────────────────────────────────────────────────


def test_gather_images_recurses_and_balances_nested_classes(tmp_path):
    """A canonical dir-per-class root yields a stable, balanced cohort."""
    for gesture in ("rock", "paper", "scissors"):
        class_dir = tmp_path / gesture
        class_dir.mkdir()
        for index in range(5):
            (class_dir / f"sample-{index}.png").touch()
    (tmp_path / "not-an-image.txt").touch()

    first = _gather_image_paths(tmp_path, max_count=8)
    second = _gather_image_paths(tmp_path, max_count=8)

    assert first == second
    assert len(first) == 8
    assert [path.parent.name for path in first] == [
        "rock",
        "paper",
        "scissors",
        "rock",
        "paper",
        "scissors",
        "rock",
        "paper",
    ]
    assert {_ground_truth_for_image(path)["gesture"] for path in first} == {
        "rock",
        "paper",
        "scissors",
    }


def test_gather_images_supports_bundled_nested_sample():
    """The shipped first-run sample is a valid, exactly balanced source."""
    selected = _gather_image_paths(REPO_ROOT / "deploy" / "example-images", 15)
    counts = Counter(_ground_truth_for_image(path)["gesture"] for path in selected)

    assert len(selected) == 15
    assert counts == Counter({"rock": 5, "paper": 5, "scissors": 5})


def test_gather_images_retains_flat_arbitrary_image_support(tmp_path):
    """Flat images without RPS naming conventions remain usable."""
    for filename in ("frame-z.png", "frame-a.jpg", "frame-m.webp"):
        (tmp_path / filename).touch()

    selected = _gather_image_paths(tmp_path, max_count=10)

    assert sorted(path.name for path in selected) == [
        "frame-a.jpg",
        "frame-m.webp",
        "frame-z.png",
    ]


def test_image_limit_is_optional_for_existing_cli_callers(tmp_path):
    """Omitting the new cohort cap preserves the label-count-sized cohort."""
    args = _parse_args(["--image-source-dir", str(tmp_path), "--label-count", "7"])

    assert args.image_limit is None
    assert (args.image_limit if args.image_limit is not None else args.label_count) == 7


@pytest.mark.parametrize("image_limit", ["0", "6"])
def test_explicit_image_limit_must_cover_requested_labels(tmp_path, image_limit):
    """An explicit cohort cap cannot be empty or undersupply the review."""
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--image-source-dir",
                str(tmp_path),
                "--label-count",
                "7",
                "--image-limit",
                image_limit,
            ]
        )

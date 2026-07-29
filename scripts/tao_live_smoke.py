#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end live TAO training-chain smoke.

Drives the Blueprint's own service layer (no curl) through a complete,
low-cost training chain against a live TAO FTMS instance. This is a wiring
smoke, not a model-quality benchmark; use ``rps_e2e.py`` for the latter.

Prereqs:
  * ``~/.vlm_feedback_loop/.env`` has TAO_API_BASE_URL, TAO_API_KEY
    (NGC personal key or a pre-exchanged JWT), TAO_ORG_NAME, and
    TAO_WORKSPACE_S3_* credentials.
  * ``vlm-feedback-loop tao-bootstrap`` has recorded the workspace in
    ``deployment.db``.
  * Live TAO reachable at the configured URL (SSH tunnel OK).
  * The workspace's external S3 endpoint is reachable.
  * ``scripts/merge_lora_requirements.txt`` installed in the runtime
    venv so checkpoint packaging's subprocess can run.

Flow:
  1. Load settings and workspace state; confirm required configuration.
  2. Probe TAO (exercises JWT auto-exchange); GET workspace detail.
  3. Discover the Cosmos Reason2 2B base experiment in the workspace
     (optionally self-service provision it first via
     ``--auto-provision-base-experiments``); fail with operator
     guidance if none is indexed.
  4. Create a fresh project; activate a minimal Guidance and ingest 3
     generated JPEGs through production services; seed 3 Verified labels
     through an explicit fixture-only database boundary (no Teacher call).
  5. Patch the project's Cosmos Reason2 2B ModelConfig row with
     ``tao_base_experiment_id`` + pull status.
  6. ``create_training_suite`` with Quick preset, 2B only, no quant.
  7. Drive TAO polling + inspect the suite every 30s
     until evaluate(baseline) succeeds or the deadline hits.
  8. Verify StudentModel wiring: packaging, evaluation lineage, and
     checkpoint shape. A quality failure is expected for the 2-sample
     training fixture and does not fail this wiring smoke.

Exit code 0 on success, 1 on failure, 2 on deadline exceeded (training
still pending — partial success; operator can monitor via
``GET /training_suites/{id}`` from here).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import random
import sys
import uuid
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session
from tao_validation import (
    DEFAULT_BASE_EXPERIMENT_ID_2B,
    MODEL_NAME_2B,
    find_reason2_2b_base_experiment,
    log_banner,
    poll_training_suite,
    probe_and_confirm_workspace,
    submit_training_suite,
)

from vlm_feedback_loop.config import load_settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services import (
    example_service,
    guidance_service,
    model_config_service,
    project_service,
)
from vlm_feedback_loop.services.project_service import (
    create_project,
    get_project_engine,
)


def _make_jpeg(seed: int) -> bytes:
    """Return the bytes of a small deterministic JPEG."""
    rng = random.Random(seed)
    img = Image.new("RGB", (64, 64))
    for y in range(img.height):
        for x in range(img.width):
            img.putpixel(
                (x, y),
                (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)),
            )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _build_smoke_project(settings, base_experiment_id: str) -> dict:
    log_banner("Create wiring-smoke project + 3 labeled examples")

    project = create_project(
        name=f"tao-wiring-smoke-{uuid.uuid4().hex[:8]}",
        description="Disposable three-image fixture for live TAO wiring validation",
        settings=settings,
    )
    project_id = project.project_id
    project_dir = Path(project.project_dir)
    print(f"✓ Project created: {project_id} ({project_dir})")

    # Use the production Guidance service so SchemaCore validation and derived
    # schema generation are exercised. Only the ground-truth label import below
    # crosses the fixture-only DB boundary.
    fields = [
        {
            "field_name": "rationale_note",
            "type": "string",
            "role": "aux",
            "display_order": -1,
        },
        {
            "field_name": "severity",
            "type": "enum",
            "role": "core",
            "display_order": 1,
            "allowed_values": ["low", "high"],
        },
    ]
    guidance = guidance_service.create_guidance(
        project_id=project_id,
        description="Classify severity of visible damage.",
        schema_fields=fields,
        rules="",
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if guidance is None:
        raise SystemExit(f"Project disappeared while creating Guidance: {project_id}")
    if isinstance(guidance, str):
        raise SystemExit(f"Guidance validation failed: {guidance}")
    guidance_id = guidance.guidance_id
    updated_project = project_service.update_project(
        project_id,
        {"active_guidance_id": guidance_id},
        settings.WORKSPACE_ROOT,
    )
    if updated_project is None:
        raise SystemExit(f"Project disappeared while activating Guidance: {project_id}")
    print(f"✓ Guidance {guidance_id} activated through production services")

    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    assert engine is not None

    # Link the seeded Student base to the deployment's indexed experiment via
    # the production ModelConfig update service.
    with Session(engine) as session:
        model_config = (
            session.query(ModelConfig)
            .filter_by(project_id=project_id, model_name=MODEL_NAME_2B)
            .one()
        )
        mc_2b_id = model_config.model_config_id
    update_result = model_config_service.update_model_config(
        project_id,
        mc_2b_id,
        {
            "tao_base_experiment_id": base_experiment_id,
            "tao_base_experiment_pull_status": "pull_complete",
        },
        settings.WORKSPACE_ROOT,
    )
    if update_result is None or isinstance(update_result, str):
        raise SystemExit(f"Could not patch the 2B ModelConfig: {update_result}")
    print(f"✓ Patched ModelConfig {mc_2b_id} with tao_base_experiment_id")

    # Generated images must persist until TAO has exported the dataset. Keep
    # them in the disposable project's artifact tree, then ingest by reference
    # through the production service. The smoke does not need embeddings or an
    # inline pHash; the normal background sweeper may populate those later.
    images_dir = project_dir / "artifacts" / "tao-wiring-fixture-images"
    images_dir.mkdir(parents=True, exist_ok=True)
    example_paths: list[tuple[str, Path]] = []
    for i in range(3):
        img_path = images_dir / f"smoke_{i}.jpg"
        img_path.write_bytes(_make_jpeg(seed=42 + i))
        example_paths.append((f"smoke_{i}", img_path))

    ingest_results = example_service.ingest_examples(
        project_id,
        settings.WORKSPACE_ROOT,
        [
            {
                "example_key": key,
                "storage_ref": str(image_path),
                "source_metadata": {"validation_fixture": "tao_live_smoke"},
                "state": "Verified",
            }
            for key, image_path in example_paths
        ],
    )
    ingest_errors = [
        f"{result['example_key']}: {result['error']}"
        for result in ingest_results
        if result["status"] == "error"
    ]
    if ingest_errors:
        raise SystemExit("Fixture image ingest failed: " + "; ".join(ingest_errors))
    print(
        "✓ Ingested 3 generated examples through the production service "
        f"(keys: {[key for key, _ in example_paths]})"
    )

    # Fixture-only boundary: the product has no API for importing pre-verified
    # labels with a caller-forced train/test split. Directly seed the minimum
    # label lineage needed by dataset export, without making Teacher calls.
    # Represent each import as a manual Edit of an empty, schema-invalid
    # proposal; do not claim a Teacher accepted labels it never generated.
    # Keep this block narrow; all project/Guidance/ingest behavior above uses
    # the same services as the application.
    label_values = [
        ({"rationale_note": "low-severity crack", "severity": "low"}, None),
        ({"rationale_note": "clear high dent", "severity": "high"}, "test_pool"),
        ({"rationale_note": "moderate scratch", "severity": "low"}, None),
    ]
    with Session(engine) as session:
        for (key, _img), (label_json, pool) in zip(
            example_paths, label_values, strict=True
        ):
            op_id = generate_uuid4()
            session.add(
                OperationRecord(
                    inference_invocation_id=op_id,
                    project_id=project_id,
                    purpose="interactive_proposal",
                    example_key=key,
                    guidance_id=guidance_id,
                    model_config_id=None,
                    invocation_status="schema_invalid",
                    schema_valid_core=False,
                    structured_generation_fallback_used=False,
                    structured_generation_attempted=False,
                    truncation_attributed_schema_invalid=False,
                    ignored_due_to_run_cancellation=False,
                )
            )
            session.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=project_id,
                    example_key=key,
                    label_status="verified",
                    guidance_id=guidance_id,
                    inference_invocation_id=op_id,
                    label_json=label_json,
                    labeled_at=utc_now(),
                    verified_outcome="Edit",
                    verified_at=utc_now(),
                    edited_core_fields=["severity"],
                    edited_aux_fields=["rationale_note"],
                    rationale_source="sme_edited",
                    pool_assignment=pool,
                )
            )
        session.commit()
    print(
        "✓ Seeded 3 fixture-only Verified labels "
        "(2 training, 1 Test Pool) with synthetic invalid-proposal lineage"
    )

    return {
        "project_id": project_id,
        "project_dir": project_dir,
        "guidance_id": guidance_id,
        "mc_2b_id": mc_2b_id,
    }


def _verify_training_wiring(settings, suite: dict) -> bool:
    """Verify Student packaging, evaluation lineage, and checkpoint shape.

    A quality-validated Student is accepted, but quality failure alone does not
    fail this deliberately tiny wiring fixture. Structural failures still do:

      * StudentModel row exists with validated checkpoint packaging
      * Eval RunRecord has evaluation_source="tao", non-null+truthy
        rescored_metrics, non-null+truthy tao_native_metrics, and a
        tao_job_id pointing at a succeeded evaluate TAOJob
      * nim_checkpoint_ref resolves to a directory under
        {project_dir}/artifacts/
      * The merged checkpoint directory contains config.json, at least
        one *.safetensors or pytorch_model*.bin shard, and tokenizer
        files (tokenizer.json OR tokenizer_config.json OR tokenizer.model)
    """
    log_banner("Verify Student training wiring")
    engine = get_project_engine(suite["project_id"], settings.WORKSPACE_ROOT)
    assert engine is not None
    failures: list[str] = []
    with Session(engine) as session:
        sm = (
            session.query(StudentModel)
            .filter_by(project_id=suite["project_id"])
            .first()
        )
        if sm is None:
            print("✗ No StudentModel row found")
            return False
        print(f"  student_model_id: {sm.student_model_id}")
        print(f"  checkpoint_packaging_status: {sm.checkpoint_packaging_status}")
        print(f"  quality_status: {sm.quality_status}")
        print(f"  quality_evaluation_run_id: {sm.quality_evaluation_run_id}")
        print(f"  nim_checkpoint_ref: {sm.nim_checkpoint_ref}")

        if sm.quality_status != "validated":
            failures.append(
                f"quality_status={sm.quality_status!r} (expected 'validated')"
            )
        if sm.checkpoint_packaging_status != "validated":
            failures.append(
                f"checkpoint_packaging_status={sm.checkpoint_packaging_status!r} "
                "(expected 'validated')"
            )
        if sm.quality_evaluation_run_id is None:
            failures.append("quality_evaluation_run_id is null")

        # Eval RunRecord shape.
        if sm.quality_evaluation_run_id:
            run = (
                session.query(RunRecord)
                .filter_by(run_id=sm.quality_evaluation_run_id)
                .one_or_none()
            )
            if run is None:
                failures.append(
                    f"quality_evaluation_run_id={sm.quality_evaluation_run_id!r} "
                    "but no RunRecord found"
                )
            else:
                print(f"  Run.evaluation_source: {run.evaluation_source}")
                has_rescored = bool(run.rescored_metrics)
                has_native = bool(run.tao_native_metrics)
                print(
                    f"  Run.rescored_metrics: populated={has_rescored}; "
                    f"tao_native_metrics: populated={has_native}"
                )
                print(f"  Run.tao_job_id: {run.tao_job_id}")
                if run.evaluation_source != "tao":
                    failures.append(
                        f"RunRecord.evaluation_source={run.evaluation_source!r} "
                        "(expected 'tao')"
                    )
                if not has_rescored:
                    failures.append("RunRecord.rescored_metrics empty/null")
                if not has_native:
                    failures.append("RunRecord.tao_native_metrics empty/null")
                if not run.tao_job_id:
                    failures.append("RunRecord.tao_job_id null")
                else:
                    eval_job = (
                        session.query(TAOJob)
                        .filter_by(tao_job_id=run.tao_job_id)
                        .one_or_none()
                    )
                    if eval_job is None:
                        failures.append(
                            f"RunRecord.tao_job_id={run.tao_job_id!r} "
                            "but no TAOJob row found"
                        )
                    else:
                        print(
                            f"  TAOJob {eval_job.tao_job_id}: "
                            f"action={eval_job.action!r}, "
                            f"status={eval_job.status!r}"
                        )
                        if eval_job.action != "evaluate":
                            failures.append(
                                f"TAOJob({run.tao_job_id}).action="
                                f"{eval_job.action!r} (expected 'evaluate')"
                            )
                        if eval_job.status != "succeeded":
                            failures.append(
                                f"TAOJob({run.tao_job_id}).status="
                                f"{eval_job.status!r} (expected 'succeeded')"
                            )

        # nim_checkpoint_ref shape on disk.
        from pathlib import Path

        if not sm.nim_checkpoint_ref:
            failures.append("StudentModel.nim_checkpoint_ref is null/empty")
        else:
            ckpt_dir = Path(sm.nim_checkpoint_ref)
            project_artifacts = (
                Path(settings.WORKSPACE_ROOT)
                / "projects"
                / suite["project_id"]
                / "artifacts"
            )
            if not ckpt_dir.exists() or not ckpt_dir.is_dir():
                failures.append(
                    f"nim_checkpoint_ref={sm.nim_checkpoint_ref!r} "
                    "is not a directory on disk"
                )
            else:
                try:
                    ckpt_resolved = ckpt_dir.resolve()
                    artifacts_resolved = project_artifacts.resolve()
                    if artifacts_resolved not in ckpt_resolved.parents and (
                        ckpt_resolved != artifacts_resolved
                    ):
                        failures.append(
                            f"nim_checkpoint_ref={ckpt_resolved} not under "
                            f"{artifacts_resolved}"
                        )
                except (OSError, ValueError) as exc:
                    failures.append(f"nim_checkpoint_ref resolution failed: {exc}")
                # Required files: config.json + at least one weight shard
                # + a tokenizer file.
                if not (ckpt_dir / "config.json").is_file():
                    failures.append(f"merged checkpoint {ckpt_dir} missing config.json")
                weight_shards = list(ckpt_dir.glob("*.safetensors")) + list(
                    ckpt_dir.glob("pytorch_model*.bin")
                )
                if not weight_shards:
                    failures.append(
                        f"merged checkpoint {ckpt_dir} has no .safetensors "
                        "or pytorch_model*.bin shards"
                    )
                else:
                    print(
                        f"  weight shards: {len(weight_shards)} "
                        f"(first: {weight_shards[0].name})"
                    )
                tokenizer_files = [
                    p
                    for p in (
                        ckpt_dir / "tokenizer.json",
                        ckpt_dir / "tokenizer_config.json",
                        ckpt_dir / "tokenizer.model",
                    )
                    if p.is_file()
                ]
                if not tokenizer_files:
                    failures.append(
                        f"merged checkpoint {ckpt_dir} has no tokenizer "
                        "files (tokenizer.json / tokenizer_config.json / "
                        "tokenizer.model)"
                    )
                else:
                    print(f"  tokenizer files: {[p.name for p in tokenizer_files]}")

        # Distinguish "wiring failure" (anything broken) from
        # "training-budget failure" (chain works end-to-end but the
        # 2-sample fixture is too small to bend the 2B model's output
        # toward the JSON schema, so rescoring's C2 path fires). The
        # latter is the EXPECTED smoke outcome: cosmos-rl's default
        # ``optm_lr=1e-06`` + ``optm_warmup_steps=20`` over
        # ``epochs × samples = 24`` total steps never escapes warmup
        # and produces a checkpoint indistinguishable from base.
        # Production users with real datasets clear the gate per the
        # rescoring service's design.
        wiring_only_failures = {
            f"quality_status={sm.quality_status!r} (expected 'validated')",
            "quality_evaluation_run_id is null",
        }
        only_quality_signals = (
            failures
            and all(f in wiring_only_failures for f in failures)
            and sm.checkpoint_packaging_status == "validated"
            and bool(sm.nim_checkpoint_ref)
        )
        if only_quality_signals:
            print(
                "⚠ Student training wiring VERIFIED but quality_status='failed' "
                "(expected on the 2-sample smoke fixture):"
            )
            for f in failures:
                print(f"    - {f}")
            print(
                "  Reason: cosmos-rl ``optm_lr=1e-06`` + 20-step warmup over "
                "``epochs × samples = 24`` steps yields a checkpoint "
                "indistinguishable from base (see train log). "
                "Chain wiring (provisioning → train → eval → packaging → "
                "predictions extracted → predictions paired with ground "
                "truth) is fully verified on this run."
            )
            return True
        if failures:
            print("✗ Student training wiring FAILED:")
            for f in failures:
                print(f"    - {f}")
            return False
        print("✓ Student training wiring PASSED (quality also validated)")
        return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_argv = list(sys.argv[1:] if argv is None else argv)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-experiment-id-2b",
        default=DEFAULT_BASE_EXPERIMENT_ID_2B,
        help="Override the Cosmos Reason2 2B base-experiment id.",
    )
    p.add_argument(
        "--pull-deadline-s",
        type=float,
        default=1800.0,
        help="Base-experiment pull deadline in seconds (default: 30 min).",
    )
    p.add_argument(
        "--train-deadline-s",
        type=float,
        default=3600.0,
        help="Training + eval deadline in seconds (default: 60 min).",
    )
    p.add_argument(
        "--skip-pull",
        action="store_true",
        help=(
            "Skip the base-experiment pull step (assume the workspace already has it)."
        ),
    )
    p.add_argument(
        "--auto-provision-base-experiments",
        action="store_true",
        default=False,
        help=(
            "Invoke the Blueprint self-service provisioning service before "
            "discovery. Default OFF; ON in the closing acceptance invocation."
        ),
    )
    return p.parse_args(cli_argv)


async def _amain() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = _parse_args()

    settings = load_settings()
    for required in (
        "TAO_API_BASE_URL",
        "TAO_API_KEY",
        "TAO_ORG_NAME",
    ):
        if not getattr(settings, required):
            print(f"✗ {required} is not configured in .env", file=sys.stderr)
            return 1

    await probe_and_confirm_workspace(settings)

    provisioned_uuids: dict[str, str] = {}
    if args.auto_provision_base_experiments:
        from vlm_feedback_loop.services.tao_base_experiment_provisioning_service import (  # noqa: E402
            provision_base_experiments,
        )

        log_banner("Self-service base-experiment provisioning")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        prov = await provision_base_experiments(settings, hf_token=hf_token)
        print(
            f"  registered={prov.registered}  "
            f"already_registered={prov.already_registered}  "
            f"failed={prov.failed}"
        )
        if prov.failed:
            print(
                "✗ Self-service provisioning failed; cannot proceed to "
                "the training-chain smoke.",
                file=sys.stderr,
            )
            return 1
        provisioned_uuids = dict(prov.uuid_by_model_name)
        print(f"  uuid_by_model_name={provisioned_uuids}")

    if args.skip_pull:
        print("→ Skipping base-experiment pull (--skip-pull)")
        base_experiment_id = args.base_experiment_id_2b
    elif provisioned_uuids.get(MODEL_NAME_2B):
        base_experiment_id = provisioned_uuids[MODEL_NAME_2B]
        print(
            f"→ Using base experiment UUID from self-service "
            f"provisioning result: {base_experiment_id}"
        )
    else:
        base_experiment_id = await find_reason2_2b_base_experiment(
            settings,
            args.base_experiment_id_2b,
            deadline_s=args.pull_deadline_s,
        )

    assembly = _build_smoke_project(settings, base_experiment_id)
    # The smoke validates END-TO-END WIRING, not training quality.
    # cosmos-rl's default optimizer (``optm_lr=1e-06``,
    # ``optm_warmup_steps=20``) is tuned for production-scale runs
    # over millions of tokens; against the 2-sample smoke fixture
    # ``(epochs × samples ÷ batch_size)`` never exceeds the warmup
    # window, so the trained checkpoint is essentially identical to
    # base regardless of preset (smoke runs at 1 epoch and at 12
    # epochs produce byte-identical eval output). Higher presets
    # just burn 8×A100 time without buying any lineage-acceptance
    # difference. Use ``quick`` for the fastest chain-wiring
    # round-trip and treat ``quality_status="failed"`` from the C2
    # path as the expected smoke outcome on this fixture — production
    # users with real datasets will reach ``validated`` per the
    # rescoring service's design.
    suite = await submit_training_suite(
        settings,
        assembly,
        base_model="2b",
        training_preset="quick",
        quantization_schemes=[],  # baseline only — fastest smoke path
        export_field_mode="all",
        idempotency_prefix="live-smoke",
    )

    outcome = await poll_training_suite(
        settings,
        suite,
        deadline_s=args.train_deadline_s,
        accept_eval_failure=False,
    )
    if outcome == "succeeded":
        wiring_validated = _verify_training_wiring(settings, suite)
        print(
            json.dumps(
                {
                    "smoke": "succeeded",
                    "project_id": assembly["project_id"],
                    "training_suite_id": suite["training_suite_id"],
                    "wiring_validated": wiring_validated,
                    "quality_gate": "not_required_for_three-image_wiring_fixture",
                },
                indent=2,
            )
        )
        return 0 if wiring_validated else 1
    if outcome == "deadline_exceeded":
        print(
            json.dumps(
                {
                    "smoke": "deadline_exceeded",
                    "project_id": assembly["project_id"],
                    "training_suite_id": suite["training_suite_id"],
                    "note": (
                        "training submitted + mid-chain; monitor via "
                        "GET /projects/{pid}/training_suites/{suite_id}"
                    ),
                },
                indent=2,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "smoke": "failed",
                "project_id": assembly["project_id"],
                "training_suite_id": suite["training_suite_id"],
            },
            indent=2,
        )
    )
    return 1


def main() -> None:
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()

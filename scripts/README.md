# scripts/

Product and operations tooling lives here.

## Product & operations — start here

| Script | Purpose |
|---|---|
| `dev.sh` | Single-command dev launch: backend :8000 + frontend :5173 |
| `setup-dev.sh` / `setup-local.sh` / `setup-common.sh` | Fresh-machine toolchain bootstrap (CPU-only dev vs GPU/local-NIM host) |
| `ci-local.sh` | Reproduce the CI pipeline locally before pushing |
| `export_public_snapshot.py` | Export a committed, curated public-repository snapshot; AutoRun is excluded by default and can be restored with `--include-autorun` |
| `generate_third_party_licenses.py` | Regenerate `LICENSE-3rd-party.txt` after dependency changes |
| `render_architecture_diagram.py` | Re-render `docs/images/architecture.png` from the `.mmd` source |
| `merge_lora.py` (+ `merge_lora_requirements.txt`) | **Runtime dependency** — spawned by the backend to merge LoRA adapters after Student training |
| `pull_base_experiments.py` (+ `pull_base_experiments_requirements.txt`) | **Runtime dependency** — spawned by the backend's TAO base-experiment provisioning |

## Live smokes & validation

Operator-run gates against live services. Most hosted-NIM smokes need
`NVIDIA_API_KEY`; TAO smokes need a bootstrapped TAO workspace and the
credentials documented in `docs/tao-ftms-install.md`.

The TAO validation scripts have intentionally different jobs:

| Script | Gate | Dataset |
|---|---|---|
| `tao_live_smoke.py` | Fast TAO wiring: provisioning/discovery → train → evaluate → checkpoint packaging and lineage | Three generated images; a Student quality failure is expected and is not this smoke's pass criterion |
| `rps_e2e.py` | Real training quality, 2B/8B selection, and optional quantization matrix | Complete `rps-test-set`: 372 images, 124 per class; defaults to `~/rps-test-set` |
| `full_stack_validation.py` | Trained Student deployment, serving, and benchmark handoff | Existing trained project/checkpoint |
| `capture_tao_fixtures.py` | Refresh committed FTMS wire-contract fixtures after an intentional TAO compatibility update | Live FTMS responses, not an image dataset |

The repository does bundle 15 RPS images (5 per class) under
`deploy/example-images`, but those are first-run product samples—not enough to
run the 372-image `rps_e2e.py` quality gate. Obtain the complete test split from
the source named in `deploy/example-images/LICENSE.DATA`, arrange it as
`{rock,paper,scissors}/*.png`, and pass its path with `--rps-root` if it is not
at the default location.

Other live drivers:

`full_pipeline_smoke.py` · `full_stack_validation.py` ·
`hosted_seeded_models_smoke.py` · `cold_start_smoke.py` ·
`icl_loop_smoke.py` · `icl_loop_realistic_smoke.py` ·
`profile_b_live_validation.py` · `schema_evolution_smoke.py` ·
`rps_e2e.py` · `tao_live_smoke.py` · `capture_tao_fixtures.py` ·
`probe_hosted_image_caps.py` · `probe_image_caps_sweep.py`

`smoke_helpers.py` contains the small health-check and model-resolution helpers
shared by several hosted-NIM smokes. `tao_validation.py` contains the TAO
workspace, suite-submission, polling, and terminal-state mechanics shared by
the two TAO training drivers; it is a library module, not a standalone command.

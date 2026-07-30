# Changelog

Notable user-facing changes to the Interactive VLM Feedback Loop are recorded
here. The public release begins with v1.0; private development history and
pre-release database revisions are intentionally excluded.

## 2026-07-30 — LoRA baseline merge and evaluation

- LoRA training chains now merge the adapter with its gated base on the
  Blueprint host and automatically evaluate the resulting full-precision
  checkpoint through the local Student NIM. The Training monitor records this
  as the real baseline evaluation instead of canceling TAO's unsupported
  adapter-only evaluate action.
- Quantized evaluation remains TAO-native because TAO quantization already
  emits a merged quantized checkpoint.
- Student Training readiness now requires `HF_TOKEN` and a working isolated
  LoRA merge runtime when LoRA is enabled. `scripts/setup-dev.sh` provisions
  the shared runtime automatically.

## 2026-07-29 — Public FTUE and release hardening

- Made Docker Compose with hosted NVIDIA endpoints the primary SME quick
  start, with a loopback-only default, trusted-network warning, and documented
  stop, resume, persistence, and reset behavior.
- Added a bundled-sample shortcut, rock-paper-scissors Guidance preset,
  150-image recommendation, and an automated browser walkthrough through a
  truthful Verified Edit and subsequent ICL use.
- Reconciled embedding-provider state after background activation so ingestion
  cannot report hosted or local embeddings as unavailable once active.
- Added authoritative TAO/data preflight and training counts, a minimal
  one-base Quick + FP8 validation workflow, exact four-job confirmation, and a
  structured infrastructure handoff.
- Clarified temporary Student serving validation versus externally owned
  production deployment throughout the product and documentation.
- Added offline public-snapshot structure/link validation and a separate,
  sanitized TAO FTMS operator guide.

## v1.0 — Initial public release

### Interactive labeling and in-context learning

- Teacher VLM proposals support SME Accept, Edit, and Skip decisions.
- Accepted and edited labels become verified ground truth.
- Verified edits feed subsequent proposals through relevance- and
  diversity-aware in-context learning.
- Guidance and SchemaCore evolution preserve explicit labeling semantics.

### Evaluation and scale-up

- Held-out Test Pool evaluation reports exact match, per-field match, coverage,
  and operational health.
- A five-criterion readiness gate controls Batch Labeling and Student Training.
- Batch Labeling produces reviewable machine labels and reproducible dataset
  exports.

### Student training and deployment

- Training suites prepare datasets and submit Cosmos-RL/TAO fine-tuning jobs.
- Trained Students can be deployed behind local NIM.
- Compare benchmarks Teacher and Student quality and serving performance.

### Delivery and operations

- Local-source development runs through `./scripts/dev.sh`.
- The containerized application runs through `docker compose up --build`.
- Each project uses a WAL-mode SQLite database with automatic integrity checks,
  backups before upgrades, and an Alembic version marker.
- The public database lineage starts at `v1_0001`, a single baseline containing
  the complete v1 project schema. Private pre-release databases are unsupported
  and require a fresh workspace.
- Curated public snapshots exclude the optional AutoRun operator feature and
  its dedicated tests by default. Maintainers can opt back in with
  `--include-autorun`.
- The repository includes a licensed RPS sample for the first-run workflow.
- TAO operator validation separates a generated three-image wiring smoke from
  the complete 372-image RPS quality/quantization gate; shared polling and
  terminal-state logic lives in one non-executable helper module.

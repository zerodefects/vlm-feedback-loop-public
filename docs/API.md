# API Reference

The Interactive VLM Feedback Loop backend is a FastAPI application. All endpoints live under the `/v1` prefix.

**Base URLs**

- **Local source** (`./scripts/dev.sh`): `http://localhost:8000/v1`
- **Containerized** (`docker compose up --build`, behind nginx): `http://localhost:3000/v1`

**Authentication**: none. v1 is a single-user tool intended for a trusted network — there is no auth, RBAC, or multi-reviewer support, and one backend process owns a project database at a time.

This page is a curated map of the **core product-loop endpoints**. The full interactive surface — every endpoint with request/response schemas — is served by the running backend at `/docs` (Swagger UI) and `/redoc`. The normative API contract is [`docs/Engineering_Spec.md`](Engineering_Spec.md) §10; section numbers below (for example §10.2.3) refer to that document.

## Conventions

**Errors.** Errors are structured JSON with specific status codes:

| Status | Meaning |
|---|---|
| `400` | Domain validation failed — the client could fix the input (invalid enum value, missing reference, out-of-range integer) |
| `404` | Project, example, or resource not found |
| `409` | Conflict — a state transition violates an invariant (stale proposal, terminal-state cancel, occupied GPU, project locked by another process) |
| `422` | Pydantic schema validation (FastAPI automatic) — reserved for request-shape errors; domain errors use `400` |
| `502` | Upstream TAO service refused a request |
| `503` | Upstream NIM/TAO endpoint unreachable (connection error) |
| `504` | Upstream endpoint reachable but timed out |

One classifier (`services/errors.py::map_service_error`) produces these statuses for every endpoint, so the same error maps to the same code across the whole API. Interactive proposal endpoints are the deliberate exception for *attempted upstream* failures: they return `200` with `invocation_status: "timeout" | "endpoint_error" | "rate_limited"` (the last distinguishes retry-exhausted hosted-NIM 429s) so the failed invocation is recorded and the UI can offer retry affordances. A Teacher endpoint already persisted as disabled or hard-unhealthy is rejected before dispatch with `503`; the backend never follows its stale URL merely because a reused host port answers.

The standard body is `{"detail": "<message>"}`. Some 409 responses carry a machine-readable code alongside the message — either as a sibling field (`{"detail": "...", "code": "project_busy"}` on project archive conflicts) or as a structured detail object (`{"detail": {"code": "gpu_occupied", "message": "..."}}` on local NIM deploy conflicts).

**Custom verbs.** Non-CRUD operations use a `:verb` path suffix, e.g. `POST /v1/projects/{id}:archive`, `POST /v1/projects/{id}/examples:ingest`.

**SSE is a hint channel.** `GET /v1/projects/{project_id}/events` streams project-scoped server-sent events (ingest/embedding progress, run lifecycle, NIM benchmark progress). REST is authoritative: clients reconcile from REST on page load, reconnect, and terminal events rather than trusting the stream alone.

**Pagination.** List endpoints take `limit` and `cursor` query parameters and return `{"items": [...], "next_cursor": "..." | null}`. Pass `next_cursor` back as `cursor` to fetch the next page.

## Projects (§10.2.13, §13.4)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects` | Create a project (201); selects and attaches an exact compatible running Teacher when available |
| `GET` | `/v1/projects` | List projects with counts; `include_archived=true` to include soft-archived. Every response carries a workspace-global `has_archived: bool` (cheap marker-file scan) so clients can offer "Show archived" without an archived-inclusive fetch |
| `GET` | `/v1/projects/{project_id}` | Get one project with example-state counts |
| `PATCH` | `/v1/projects/{project_id}` | Partial update — Teacher selection pointer, first Guidance activation (later switches 400 — use the guidance edit endpoint), generation controls, gate thresholds, feature flags |
| `POST` | `/v1/projects/{project_id}:archive` | Soft-archive; 409 `project_busy` while runs/jobs are in flight |
| `POST` | `/v1/projects/{project_id}:unarchive` | Reverse a soft archive; 409 `not_archived` otherwise |
| `POST` | `/v1/projects/{project_id}:mark_setup_completed` | Stamp first-time-setup acknowledgment (idempotent) |
| `GET` | `/v1/projects/{project_id}/events` | SSE stream of project events |
| `POST` | `/v1/projects/{project_id}/action_requests:generate` | Render a pre-filled infrastructure handoff Action Request (§10.2.23) |
| `POST` | `/v1/projects/{project_id}/action_requests:log_copy` | Audit that an Action Request was copied |

Each project owns its own SQLite database under `{WORKSPACE_ROOT}/projects/{project_id}/`. Opening a project that another backend process holds returns `409` ("already open in another process").

## Image ingestion & examples (§10.2.1, §10.2.8–§10.2.11)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/examples:ingest` | Batch-ingest images by filesystem reference (202); pHash + CLIP embeddings compute in background workers |
| `GET` | `/v1/projects/{project_id}/examples` | Query examples with filters (`state`, `verified_outcome`, `guidance_id`, `pool_membership`, date bounds) and cursor pagination |
| `GET` | `/v1/projects/{project_id}/examples/{example_key}/image` | Stream the image bytes from the persisted `storage_ref` |
| `GET` | `/v1/projects/{project_id}/examples:embedding_status` | Authoritative embedding-completion status (REST counterpart to `embedding_progress` SSE); polling also restarts the embedding worker when examples are pending but no worker is running (rate-limited per project) |
| `POST` | `/v1/projects/{project_id}/examples:remap_paths` | Bulk prefix-remap `storage_ref` paths after moving a dataset; `dry_run` (default true) previews first |

Images are stored **by reference** — the request carries an `example_key` and an absolute `storage_ref` path; files are never copied into the workspace. Ingestion is per-item with partial success: each item returns `status` of `created`, `exists` (idempotent re-ingest), or `error` with an `error_code`. Error codes include `example_key_collision` (same key, different path), `storage_ref_already_ingested` (same source path under a different key), and `path_not_allowed` (the `storage_ref` falls outside `IMAGE_ROOT`). Scan-generated keys are anchored to configured `IMAGE_ROOT`, or to the normalized absolute path when the root is unset, so selecting a nested directory does not change an image's identity. The root boundary is enforced at ingest and re-applied whenever persisted image bytes are read. Serving, inference, embeddings, pHash, benchmark workloads, and exports consume an already-authorized regular-file descriptor rather than checking and later reopening a mutable path. It is unrestricted only for a loopback/single-user backend with `IMAGE_ROOT` unset.

## Guidance (§10.2.2, §4.4)

Guidance versions are immutable: the task description, label schema, and rules that drive prompt construction and validation.

Field-name note: **requests** send the field list as `schema` (`{description, schema, rules}`); **responses** return it as `schema_fields` (alongside the backend-derived `derived_json_schema`). The asymmetry is intentional and stable — `schema` shadows Pydantic's `BaseModel.schema` internally, so the Python attribute is `schema_fields` with a request-side alias, and renaming either side now would break existing clients.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/guidance` | Create a new Guidance version (201); body is `{description, schema, rules}` |
| `GET` | `/v1/projects/{project_id}/guidance` | List versions newest-first |
| `GET` | `/v1/projects/{project_id}/guidance/{guidance_id}` | Get one version with derived JSON Schema |
| `POST` | `/v1/projects/{project_id}/guidance:validate_draft` | Validate a draft without saving; returns issues, derived schema, `save_allowed` |
| `POST` | `/v1/projects/{project_id}/guidance:edit` | Edit the active Guidance; `dry_run=true` classifies the change (in-place vs semantic) and previews affected label counts, `dry_run=false` executes |
| `GET` | `/v1/projects/{project_id}/guidance:icl_count` | Count of ICL-eligible Edits (non-pool Verified Edits under the active Guidance) |
| `GET` | `/v1/projects/{project_id}/guidance:reminder_status` | Schema-refinement reminder state (§6.8) |
| `POST` | `/v1/projects/{project_id}/guidance:dismiss_reminder` | Dismiss the current refinement reminder |

`description` and `rules` may be empty; saving requires at least one valid Core field (`NO_CORE_FIELDS`).

Rationale notes are disabled by default. Their Guidance toggle is represented
by schema presence: omit `rationale_note` to disable it; include the reserved
`{"field_name":"rationale_note","type":"string","role":"aux"}` field to enable
it. Adding or removing that Aux field is an in-place edit and does not
invalidate labels.

## Proposals & labeling (§10.2.3, §10.2.4, §10.2.14, §10.2.15)

The interactive loop: pick the next image, get a Teacher proposal, then Accept/Edit (save) or Skip.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/projects/{project_id}/review_selector/next` | Diversity-driven selection of the next example to review (pHash or CLIP mode); carries the example's `storage_ref` and prior-label snapshot ref |
| `POST` | `/v1/projects/{project_id}/proposals` | Request a Teacher proposal for an example — runs ICL selection → prompt render → Teacher invocation → schema validation |
| `POST` | `/v1/projects/{project_id}/labels` | Save a Verified label; the backend diffs proposal vs submission to classify Accept (no diff) or Edit |
| `POST` | `/v1/projects/{project_id}/examples/{example_key}:skip` | Omit an example; atomically discard its Auto-Labeled machine proposal, if present |
| `POST` | `/v1/projects/{project_id}/examples:restore_omitted` | Bulk-restore all Omitted examples to clean Unlabeled state |
| `POST` | `/v1/projects/{project_id}/examples/{example_key}:regenerate_rationale` | When rationale notes are enabled, generate a fresh note by independently inspecting the image; returns 409 while disabled |

Proposal requests accept per-call overrides (`teacher_model_config_id_override`, `guidance_id_override`, `generation_preset_key_override`, `thinking_mode_override`, `visual_budget_preset_key_override`) and `use_existing_label=true` to return a stored Auto-Label without a fresh Teacher call. Retries pass `retry_of_inference_invocation_id`; a proposal superseded by a later retry saves as `409`.

When rationale notes are disabled, proposal schemas and prompts omit the field,
and label saves require only `example_key`, `inference_invocation_id`, and
`label_json`. When enabled, saves additionally require `rationale_source`;
edited labels cannot use `teacher_proposal`.

## Evaluation & Scale-Up gate (§10.2.16, §10.2.17, §7)

Background evaluation scores the current Teacher + Guidance + ICL setup against the held-out Test Pool.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/evaluation_runs` | Start an evaluation run (201); `icl_mode` of `enabled`/`disabled`, plus diagnostic overrides. `icl_max_examples` unset defers the ICL depth cap to the Teacher model's `default_icl_max_examples` (§6.2); setting it overrides the model default in either direction |
| `GET` | `/v1/projects/{project_id}/evaluation_runs` | List runs (`status` filter; `basis` filter — `gate` for gate-basis Teacher runs, `benchmark` for Student benchmark runs; cursor pagination) |
| `GET` | `/v1/projects/{project_id}/evaluation_runs/{run_id}` | Run status, persisted progress (`processed` / frozen `total`, retained after terminal status), and metrics |
| `POST` | `/v1/projects/{project_id}/evaluation_runs/{run_id}:cancel` | Cancel a running evaluation; 409 if terminal |
| `GET` | `/v1/projects/{project_id}/evaluation_trigger_status` | Whether labeling progress has tripped an evaluation trigger (§7.1) |
| `POST` | `/v1/projects/{project_id}/evaluation_trigger_status:dismiss` | Dismiss a pending trigger |
| `GET` | `/v1/projects/{project_id}/scaleup_gate` | Evaluate the 5-criteria Scale-Up Readiness Gate (§7.3) |

The gate response reports `gate_status` (`ready` / `not_ready`) and per-criterion detail for `overall_exact_match`, `per_field_match`, `min_per_value_f1`, `accept_rate`, and `min_test_pool_size`. The overall criterion's `details` identifies the selected evaluation run and its snapshotted Teacher, and reports whether current project settings differ; this prevents historical quality from being attributed to a newly selected Teacher. Passing the gate unlocks Batch Labeling. Student Training navigation is independent of the Teacher-quality criteria; `POST .../training_suites` performs its own fail-closed TAO/workspace/timeout/data/role validation, including the configured Test Pool minimum, and creates a conditional provisioning stage for selected missing bases. Gate thresholds are per-project settings (`PATCH /v1/projects/{id}`).

## Batch labeling (§10.2.6)

Auto-label remaining Unlabeled examples at scale with the trusted Teacher setup.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/batch_label_runs` | Start a run (201); optional `run_limit`, `include_auto_labeled`, `concurrency` (dispatch-width override; default is provider-aware — hosted 1 / self-hosted 8), ingest-date bounds, `icl_mode` (`enabled`/`disabled`, default `enabled` — disable to batch-label at the Teacher's zero-shot form for ICL-negative teachers, §8.3) |
| `GET` | `/v1/projects/{project_id}/batch_label_runs` | List runs |
| `GET` | `/v1/projects/{project_id}/batch_label_runs/{run_id}` | Run status, counters, and the snapshotted `circuit_breaker_threshold` used to explain a pause (null on legacy runs) |
| `POST` | `/v1/projects/{project_id}/batch_label_runs/{run_id}:resume` | Resume a paused run |
| `POST` | `/v1/projects/{project_id}/batch_label_runs/{run_id}:cancel` | Cancel a run |
| `GET` | `/v1/projects/{project_id}/batch_label_runs/{run_id}/schema_invalid_manifest` | Manifest of items whose Teacher output failed Core schema validation (§10.2.24) |

## Dataset exports (§10.2.18)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/dataset_exports` | Create a dataset export (201, `status: "running"`; the archive builds in the background — poll the record or follow `export_*` SSE). **409** while another export is already building for the project. `dataset_intent`, `label_tier_filter`, `export_field_mode`, optional `batch_label_run_id` and selection filters |
| `GET` | `/v1/projects/{project_id}/dataset_exports` | List exports (`dataset_intent` filter) |
| `GET` | `/v1/projects/{project_id}/dataset_exports/{dataset_export_id}` | Get an export record (status, progress, paths, counts) |
| `GET` | `/v1/projects/{project_id}/dataset_exports/{dataset_export_id}/archive` | Download a completed `.tar.gz` archive (`application/gzip`, attachment filename, and `X-Checksum-SHA256`). Running/failed exports return 409; the backend streams only a project-contained artifact. |

Exports with `dataset_intent=evaluation` or `testing` require
`label_tier_filter=verified_only`; requests that would use Auto-Labeled
predictions as an answer key are rejected.

Set a non-empty `batch_label_run_id` to export only Auto-Labeled outputs
produced by that Batch run. The Batch run status screen supplies this filter
for completed and partial-run exports and tracks only an export with matching
lineage; omitting it deliberately selects matching labels across the project,
as required by the direct export and Training Suite workflows. An empty string
is rejected rather than interpreted as omission. The completed export's
selection snapshot and manifest record the effective run, Guidance, tier,
field-mode, and additional selection filters.

The archive builds in a background task — the create response returns immediately with `status: "running"` and `artifact_refs: null`. Poll the GET endpoint (or follow the `export_progress` / `export_completed` / `export_failed` SSE events) until the record reaches `completed`, at which point `artifact_refs` carries the archive path and SHA-256 checksum. Do not re-POST after a slow export: each POST builds a new multi-GB archive. Export archives land under the project's `exports/` directory; formats are specified in §8 and §9.3.

Use the `/archive` endpoint for downloads in both local-source and Compose
mode. Do not treat `artifact_refs.archive_path` as a client-visible filesystem
path: in Compose it belongs to the backend container's named workspace volume.
The endpoint confines the artifact to that project's `exports/` directory and
streams the authorized inode, including standard HTTP byte-range responses;
changing the stored path after authorization cannot change the downloaded
bytes.

## Student training (§10.2.7, §10.2.22, §9)

Fine-tune Student models (Cosmos-RL / TAO) from Verified data. A **training suite** is the composite one-POST path: optional first-use base provisioning, dataset exports, pre-created TAO chain jobs (train → baseline evaluate → quantize → quantized evaluate), and first-chain kickoff. Individual TAO jobs can also be managed directly.

Before any workspace-S3 request, the backend requires each training and Test
Pool export to be `completed`, opens its archive and required
`annotations.json` sidecar beneath that project's `exports/` directory, and
validates both through the same file descriptors later uploaded. The archive
must match its recorded SHA-256, and the sidecar JSON must be type-sensitively
equal to the archive's root `annotations.json` (object key order and JSON
formatting may differ; array order and JSON value types may not). The upload
rehashes the exact bytes sent, so an in-place file change after validation
fails a single PUT before dispatch or aborts multipart before completion. The two
objects form one idempotent upload pair; lineage is persisted only after both
are present at distinct object keys with matching hashes.

For an ordinary suite whose bases are already ready, an export-integrity or
workspace-transfer failure returns HTTP 409 with error code
`tao_dataset_upload_failed`. If first-use base provisioning already returned a
provisional suite, the same later failure makes that suite `failed`; poll the
suite GET endpoint and inspect `setup_error_ref` for the actionable error.

The backend persists a `preparing` TrainingSuite before building exports and
links each completed DatasetExport before uploading it. When a transfer or
workspace-configuration failure occurs before any chain exists, repeating the
identical request with the same `idempotency_key` atomically resumes that suite,
reuses its frozen export IDs and object keys, and uploads only a missing or
mismatched pair member. A changed request body conflicts. An integrity failure
is not resumable because the frozen local evidence is no longer trustworthy;
create fresh exports with a new idempotency key after correcting the cause.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/training_preflight` | Authoritative TAO/workspace/base/data readiness and selected-data counts used by Scale-Up and Student Training |
| `POST` | `/v1/projects/{project_id}/training_presets:resolve` | Resolve read-only per-model hyperparameter patches for the Training screen |
| `POST` | `/v1/projects/{project_id}/tao_base_experiment_provisioning` | Ensure selected missing Student bases in a tracked background run (202); already-ready selections complete immediately |
| `GET` | `/v1/projects/{project_id}/tao_base_experiment_provisioning/{provisioning_run_id}` | Poll first-use base provisioning until `succeeded` or `failed` |
| `POST` | `/v1/projects/{project_id}/training_suites` | Validate and launch a training suite (201); returns a provisional `status="provisioning"` suite when selected bases need first-use setup; idempotent by `idempotency_key` |
| `GET` | `/v1/projects/{project_id}/training_suites` | List suites newest-first |
| `GET` | `/v1/projects/{project_id}/training_suites/{training_suite_id}` | Suite with live chain status |
| `POST` | `/v1/projects/{project_id}/training_suites/{training_suite_id}:cancel` | Best-effort cancel every remaining setup task and TAO job, then terminalize the suite locally |
| `POST` | `/v1/projects/{project_id}/tao_jobs` | Submit a single TAO job (201) |
| `GET` | `/v1/projects/{project_id}/tao_jobs` | List TAO jobs |
| `GET` | `/v1/projects/{project_id}/tao_jobs/{tao_job_id}` | Get a job; `?refresh=true` polls the external TAO service |
| `POST` | `/v1/projects/{project_id}/tao_jobs/{tao_job_id}:cancel` | Cancel a job and halt downstream chain siblings; `?force_local=true` cancels the local row when TAO is unreachable |

TrainingSuite create/get/list responses include
the persisted `enable_lora` training mode, `training_example_count`,
`evaluation_example_count`, `evaluation_dataset_checksum_sha256`,
`student_model_ids[]`, provisioning lineage, and `setup_error_ref` alongside
the dataset-export and chain lineage. The derived `setup_retryable` boolean is
the authoritative signal for offering a same-key recovery; clients do not
infer safety from error prose. Counts/checksum are nullable while setup is still
provisioning or when historical export evidence is unavailable.

`training_preflight` accepts `student_base_model_config_ids[]`,
`include_auto_labeled`, `enable_lora` (default `true`), and
`quantization_schemes` (default `["FP8_DYNAMIC"]`). LoRA readiness also
checks the Blueprint's local merge runtime and `HF_TOKEN`, because the
full-precision baseline is merged and evaluated through the local Student NIM
instead of TAO's adapter-incompatible evaluate action. The per-model
`training_mode_compatible` check fails before suite creation when the selected
mode is unsupported. For the qualified `6.26.3-cosmos-rl` runtime, Cosmos 3
Super rejects LoRA wrappers during its required tensor-parallel partitioning;
direct API callers may send `enable_lora=false` for that base. The Training UI
does not expose Super or Full-weight until the broader Full-weight matrix is
qualified. In addition to structured checks, `verified_train_examples`
requires one active-Guidance Verified example outside the Test Pool and
`min_test_pool_size` requires the
active-Guidance Test Pool to reach
`max(1, project.scaleup_min_test_pool_size)`. Dataset failures block suite
launch without producing a TAO setup request. `data_summary` returns
`verified_training_count`, `test_pool_count`, `required_test_pool_count`,
`auto_labeled_eligible_count`, `auto_labeled_included_count`,
`excluded_test_pool_count`, `excluded_auto_labeled_count`, and
`usable_training_count`. Test Pool examples are always excluded from training
and frozen separately for every automatic Student evaluation in the suite.
The per-model `quantization_compatible` check also fails when Cosmos 3 Super
is selected with any quantization scheme. Super remains supported as a
full-precision Student: send `enable_lora=false` and
`quantization_schemes=[]` for its qualified baseline training and NIM
deployment path.

When selected bases are missing, the suite response is provisional and the
Training Jobs screen polls it while the backend drives the durable provisioning
run and subsequent suite materialization. An interrupted provisioning run and
provisional suite are marked `failed` on backend restart. A restart-interrupted
dataset transfer with both exports already linked can be resumed with its
identical request and idempotency key; base provisioning itself starts a new
suite. The Training Jobs monitor offers **Retry Dataset Upload** only for that
safe transfer-recovery state and reconstructs the exact request from persisted
suite fields. Suite creation takes `student_base_model_config_ids`,
`training_preset`, `include_auto_labeled`, `enable_lora`, `export_field_mode`, and
`quantization_schemes` (omission defaults to `["FP8_DYNAMIC"]`). The
suite-level cancel endpoint preserves completed jobs, requests cancellation
for every known active external TAO job, cancels all remaining local jobs, and
sets the suite to `canceled`. The response reports canceled/terminal counts
and `remote_cancel_failures[]`. A remote failure does not keep the project
locked in Training Jobs; clients must surface the warning because TAO work may
still be running.

TAO `status="succeeded"` means the remote executor finished; it does not by
itself mean the Blueprint has downloaded and processed the result. TAO job
detail and every `training_suites[].chains[].jobs[]` summary also expose
`outputs_fetch_status` (`pending`, `in_progress`, `completed`, or `failed`) and
sanitized `outputs_fetch_error_ref`. Clients must keep polling a succeeded job
while artifact processing is pending/in progress, show that interval as
**Finalizing**, and surface an artifact-processing failure instead of claiming
the downstream job is merely waiting for TAO.

TAO wire format is specified in §9.7.

## Student deploy & benchmark (§10.2.20, §10.2.21, §9.5)

Trained Students are deployed behind a NIM and benchmarked against the Teacher on the Compare screen, which reads the Teacher baseline and each Student's quality/serving results directly from evaluation runs. Every suite-created Student returns `training_suite_id`; ad-hoc legacy Students return null. Models & Results remains project-wide and groups these records newest-run first. It flags mixed Guidance/output-contract/evaluation-set evidence as directional and suppresses a Student-vs-Teacher delta when the two Run Records do not share Guidance, Pool version, and effective Inference Contract.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/projects/{project_id}/student_models` | List StudentModel records; each response derives `serving_benchmark_current` and a nullable `serving_benchmark_blocker` from the referenced serving run |
| `GET` | `/v1/projects/{project_id}/student_models/{student_model_id}` | Single Student (quality/serving status, current-benchmark assessment, checkpoint state) |
| `POST` | `/v1/projects/{project_id}/student_models/{student_model_id}:deploy_nim` | Trigger the Student NIM lifecycle (202, runs in background); `nim_endpoint_url: null` = local Docker orchestration, a URL = external endpoint and requires `benchmark_kv_cache_reuse: "disabled"` |
| `POST` | `/v1/projects/{project_id}/student_models/{student_model_id}:deployment_handoff` | Generate the deployment-handoff Action Request; 409 unless quality and serving are `validated` and the serving run satisfies the current AIPerf contract |
| `GET` | `/v1/projects/{project_id}/student_models/{student_model_id}/deployment_bundle` | Stream the gated portable NIM deployment bundle as `application/x-tar`; applies the same current-AIPerf gate and includes the validated Student payload, pinned launch contract, verification tooling, checksums, and lineage |
| `POST` | `/v1/projects/{project_id}/student_models/{student_model_id}:rerescore` | Replay the quality rescore for a `failed` Student after a scoring fix (409 otherwise) |
| `POST` | `/v1/projects/{project_id}/student_models/{student_model_id}:repackage` | Replay checkpoint packaging for a Student whose `checkpoint_packaging_status` is `failed`, after an environment fix — e.g. the LoRA-merge interpreter was unprovisioned (409 otherwise) |

The local `:deploy_nim` lifecycle is preflight → cache-disabled `docker run` → health → smoke → temp endpoint registration → evaluation → production benchmark sweep → stop, with progress on SSE (`nim_benchmark_progress` / `nim_benchmark_completed`). External endpoints run evaluation plus the same sweep. The benchmark replays a deterministic sample of up to 200 frozen Test Pool images with the active Guidance-derived Student prompt, no output-token cap, and pinned AIPerf. The serving run persists `metrics.benchmark_workload` and `metrics.benchmarks`; every configured concurrency must pass for `serving_status="validated"`.

Upgraded workspaces can retain a historical `serving_status="validated"` from
the pre-AIPerf synthetic latency sweep. The list/get responses report
`serving_benchmark_current=false` (with a machine-readable blocker) for that
evidence. It remains visible for comparison, but `:deployment_handoff` and
`deployment_bundle` return `409 conflict:
serving_benchmark_requires_aiperf` until `:deploy_nim` completes the current
real Test Pool AIPerf workload.

The deployment bundle contains the validated NIM-loadable checkpoint payload,
excluding TAO completion-only `status.json` and `microservices_log.txt`, plus `manifest.json`,
`SHA256SUMS`, `README.md`, `run-nim.sh`, `request-template.json`, and
`verify-nim.sh`. The request template is reconstructed from the Guidance and
effective controls of the serving evaluation that passed the deployment gate;
it intentionally omits `max_tokens`/`max_completion_tokens` and its launcher
sets `NIM_ENABLE_KV_CACHE_REUSE=0`, matching the benchmark contract;
shared-image Students retain the base ModelConfig's `NIM_MODEL_SIZE` and pinned
`NIM_MODEL_PROFILE`, plus validated non-secret runtime overrides such as the
Cosmos 3 Super context clamp, in the manifest and launcher while keeping their
Student-specific served name;
the verifier injects a caller-supplied image and fails unless the response
contains parseable structured JSON. The bundle does not contain the licensed
NIM image or credentials; the launch script pulls the pinned NGC image with an
operator-provided `NGC_API_KEY` and mounts the checkpoint read-only. Its
evaluation snapshot includes the SHA-256 of the held-out `evaluation`/`testing`
export referenced by the TAO evaluate job paired with the Student's
artifact-producing train or quantize job; the Student's own
`dataset_export_ids` continue to identify training data only.

## Local NIM lifecycle (§1.5)

Deploy and manage local NIM containers (Teacher or embedding) on this host's GPUs.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/local_nim/preflight` | Dry-run preflight checks (Docker, GPU memory, NGC credential) without starting a container |
| `POST` | `/v1/projects/{project_id}/local_nim/deploy` | Queue or reuse a local NIM (201). A new deploy returns `disposition="queued"` plus a persisted `status="starting"` reservation, then runs preflight/startup in the background; `preflight.resolved_port` and `deployment.host_port` both identify the host port actually reserved, which may differ from `preferred_port`. An exact compatible running Teacher returns `disposition="reused"`, `deployment=null`, and a safe `resident` summary. Body: `{role: "teacher"\|"embedding", model_config_id, gpu_assignment?, preferred_port?, replace_resident?, activate_on_success?}` |
| `GET` | `/v1/projects/{project_id}/local_nim/deployments` | List deployments for the project |
| `GET` | `/v1/projects/{project_id}/local_nim/deployments/{deployment_id}` | Single deployment status |
| `POST` | `/v1/projects/{project_id}/local_nim/deployments/{deployment_id}:stop` | Stop a running deployment |

The **one-NIM-per-GPU invariant** (§1.5) is enforced here. A Teacher request first reuses an exact compatible `running` Blueprint-managed Teacher, including one owned by another project; the requesting project is attached to its endpoint without taking lifecycle ownership. Otherwise an occupied target returns `409` with `detail.code = "gpu_occupied"` (or `"gpu_exhausted"` when every GPU is occupied), a non-secret `resident` summary, `matches_requested_model`, and `can_replace`. An exact compatible resident that is still starting returns `"resident_starting"` with `can_replace=false`. Pass `"replace_resident": true` only after operator confirmation to stop the named resident and start the requested NIM; the displacement is audited (§13.15). For post-onboarding model changes, Teacher-only `"activate_on_success": true` persists the selection intent and changes `project.teacher_model_config_id` only after verified health (or exact healthy-resident reuse). A failed confirmed replacement best-effort requeues the displaced resident. Preflight and auto-placement enforce both the model's memory floor and optional compute-capability floor.

Fresh-project creation computes the host's quality-ranked local recommendation
after seeding the project catalog. When an exact running resident matches that
recommendation, its 201 response already points `teacher_model_config_id` at
the reused local Teacher. A different resident remains visible through
`GET /v1/environment` so onboarding can ask whether to keep or explicitly
replace it.

For `disposition="queued"`, the response's `preflight` block reports that work
was queued; it is not the final preflight result. Poll the returned deployment
with `GET .../local_nim/deployments/{deployment_id}` until `running`, `failed`,
or `stopped`. A failed background preflight or container start is persisted as
`status="failed"` with an actionable `status_reason`. For
`disposition="reused"`, no polling or stop is required or permitted through the
requesting project: `deployment` is null and the named owner project retains
lifecycle control.

## Examples

The walkthrough below follows the hosted Docker Compose quick start. The edge
proxy is at `http://localhost:3000`, the bundled sample is mounted at
`/data/images`, and a valid `NVIDIA_API_KEY` was present when the stack
started. IDs and timestamps are illustrative.

Create a project:

```bash
curl -s -X POST http://localhost:3000/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "rps-demo", "description": "Rock/paper/scissors labeling"}'
```

```json
{
  "project_id": "a1b2c3d4-e5f6-4789-8abc-def012345678",
  "name": "rps-demo",
  "description": "Rock/paper/scissors labeling",
  "project_dir": "/data/workspace/projects/a1b2c3d4-e5f6-4789-8abc-def012345678",
  "counts": {"verified": 0, "unlabeled": 0, "auto_labeled": 0, "omitted": 0, "pending_relabel": 0, "prior_relabeled": 0},
  "teacher_model_config_id": "9f9c7c2e-3ad9-4b7e-9d2f-825b6eed4500",
  "active_guidance_id": null,
  "embedding_provider": "hosted_nvclip",
  "test_pool_fraction": 0.40,
  "created_at": "2026-07-10T17:03:12Z",
  "...": "..."
}
```

Project creation probes and resolves the embedding provider before returning.
The keyed hosted path reports `hosted_nvclip`; a healthy configured local
deployment reports `self_hosted_nvclip`; `none` means no usable provider was
found.

Ingest images by reference (returns 202; pHash and embeddings compute in the background):

```bash
curl -s -X POST http://localhost:3000/v1/projects/a1b2c3d4-e5f6-4789-8abc-def012345678/examples:ingest \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {"example_key": "rock_001", "storage_ref": "/data/images/rock/testrock04-11.png"},
      {"example_key": "paper_001", "storage_ref": "/data/images/paper/testpaper04-08.png"}
    ]
  }'
```

```json
{
  "results": [
    {
      "example_key": "rock_001",
      "status": "created",
      "error": null,
      "warnings": [],
      "example": {"example_key": "rock_001", "state": "Unlabeled", "phash": null, "clip_embedding_present": false, "...": "..."}
    },
    {"example_key": "paper_001", "status": "created", "...": "..."}
  ]
}
```

Create the same Guidance used by the bundled UI preset:

```bash
curl -s -X POST http://localhost:3000/v1/projects/a1b2c3d4-e5f6-4789-8abc-def012345678/guidance \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Classify the hand gesture in each image as rock, paper, or scissors.",
    "schema": [{
      "field_name": "category",
      "type": "enum",
      "role": "core",
      "allowed_values": ["rock", "paper", "scissors"],
      "display_order": 0
    }],
    "rules": "Use rock for a closed fist, paper for an open hand, and scissors for two separated fingers."
  }'
```

```json
{
  "guidance_id": "7d6c5b4a-3210-4fed-8cba-9876543210ab",
  "project_id": "a1b2c3d4-e5f6-4789-8abc-def012345678",
  "version_number": 1,
  "schema_fields": [{"field_name": "category", "type": "enum", "role": "core", "...": "..."}],
  "...": "..."
}
```

Creating a Guidance version does not activate it. Select it explicitly:

```bash
curl -s -X PATCH http://localhost:3000/v1/projects/a1b2c3d4-e5f6-4789-8abc-def012345678 \
  -H "Content-Type: application/json" \
  -d '{"active_guidance_id": "7d6c5b4a-3210-4fed-8cba-9876543210ab"}'
```

Now request a Teacher proposal. The project's seeded Teacher model config is
already selected by project creation:

```bash
curl -s -X POST http://localhost:3000/v1/projects/a1b2c3d4-e5f6-4789-8abc-def012345678/proposals \
  -H "Content-Type: application/json" \
  -d '{"example_key": "rock_001"}'
```

```json
{
  "inference_invocation_id": "9f8e7d6c-5b4a-4321-8fed-cba987654321",
  "example_key": "rock_001",
  "proposal_json": {"category": "rock"},
  "schema_valid_core": true,
  "validation_errors_core": [],
  "validation_errors_aux": [],
  "invocation_status": "success",
  "latency_ms_end_to_end": 2140,
  "icl_images_attached_count": 0,
  "icl_example_keys_used": [],
  "used_existing_label": false
}
```

Save the label (an unchanged `label_json` classifies as Accept; any diff classifies as Edit):

```bash
curl -s -X POST http://localhost:3000/v1/projects/a1b2c3d4-e5f6-4789-8abc-def012345678/labels \
  -H "Content-Type: application/json" \
  -d '{
    "example_key": "rock_001",
    "inference_invocation_id": "9f8e7d6c-5b4a-4321-8fed-cba987654321",
    "label_json": {"category": "rock"}
  }'
```

```json
{
  "example_key": "rock_001",
  "label_status": "verified",
  "verified_outcome": "Accept",
  "verified_at": "2026-07-10T17:09:44Z",
  "edited_core_fields": [],
  "edited_aux_fields": [],
  "pool_assignment": null
}
```

The first Verified example is not assigned to the Test Pool because
`floor(1 × 0.40) = 0`; later saves grow the pool toward the configured target.

Check the Scale-Up Readiness Gate:

```bash
curl -s http://localhost:3000/v1/projects/a1b2c3d4-e5f6-4789-8abc-def012345678/scaleup_gate
```

```json
{
  "gate_status": "not_ready",
  "criteria": [
    {"criterion_name": "overall_exact_match", "passed": false, "current_value": 0.0, "threshold": 0.8, "message": "No completed evaluation run found. Run an evaluation to measure quality.", "details": {"no_completed_run": true}},
    {"criterion_name": "per_field_match", "passed": false, "current_value": 0.0, "threshold": 0.8, "message": "Depends on evaluation results.", "details": {"failing_fields": [], "blocked_by": "overall_exact_match"}},
    {"criterion_name": "min_per_value_f1", "passed": false, "current_value": 0.0, "threshold": 0.6, "message": "Depends on evaluation results.", "details": {"failing_values": [], "blocked_by": "overall_exact_match"}},
    {"criterion_name": "accept_rate", "passed": true, "current_value": 1.0, "threshold": 0.8, "message": "Accept rate: 100% over last 1 labels (need 80%). Passed.", "details": null},
    {"criterion_name": "min_test_pool_size", "passed": false, "current_value": 0, "threshold": 60, "message": "Test Pool: 0 examples (need 60). Continue labeling to grow the pool.", "details": {"pool_target": 0, "test_pool_fraction": 0.4, "total_verified": 1}}
  ],
  "evaluated_at": "2026-07-10T17:12:03Z"
}
```

## Supporting endpoints not covered here

The following routers back setup screens and utilities rather than the core loop. Their full contracts are in the runtime `/docs` (Swagger UI):

- **Filesystem** — `GET /v1/filesystem/browse`, `POST /v1/filesystem/scan`: server-side directory browsing and recursive image scans (with suggested example keys) for the ingestion picker, constrained by one `IMAGE_ROOT`. Omit the browse `path` query parameter to open the deployment root automatically.
- **Secrets** — `POST /v1/secrets:set`: apply a deployment secret (e.g. `NVIDIA_API_KEY`) at runtime, optionally persisting to `~/.vlm_feedback_loop/.env` when `ALLOW_UI_SECRET_PERSIST` allows.
- **Model configs** — `/v1/projects/{project_id}/model_configs` create/list/get/update plus `POST .../{model_config_id}:reprobe` for capability re-probing (§10.2.12). Each entry carries `default_icl_max_examples` (int | null): the per-model default ICL depth cap applied to proposals, evaluations, and batch labeling when no explicit `icl_max_examples` override is present (§6.2). Seeded from the July 2026 cross-model depth studies (Nemotron Nano VL 2 · Nemotron 3 Nano Omni 4 · Cosmos CR3 8 · CR2-2B 8 · CR2-8B 16; null for unmeasured current alternates); settable at create and PATCHable for re-tuning. Historical/operator-created MiniMax M3 records retain their depth cap, but MiniMax is not in the fresh-project commercial seed. Evaluation and Batch runs persist a credential-free snapshot of every model and endpoint value they consume, so later updates, re-probes, or local-NIM rebinding affect future runs only.
- **NIM endpoints & environment** — `/v1/projects/{project_id}/nim_endpoints` create/list/get/update for registered inference endpoints. Every endpoint response includes derived `usage_policy`: a hosted endpoint on `integrate.api.nvidia.com` is `evaluation_only`; all others are `operator_managed`, which makes no claim that the operator has a production entitlement. The Batch pre-run UI uses this field for its trial-use confirmation; the Batch API contract is unchanged. `POST /v1/projects/{project_id}/nim_endpoints:configure_self_hosted_teacher` re-verifies an exact cataloged vision Teacher, idempotently creates/reuses its credential-free endpoint, binds it, selects it, and refreshes capabilities; `GET /v1/environment` returns current deployment state over a process-cached Docker/toolkit/GPU snapshot, and `GET /v1/environment?refresh_hardware=true` explicitly rebuilds that machine snapshot after an operator changes the host; `POST /v1/nim/test_connection` / `test_ngc_credential` / `test_nvidia_credential` provide connectivity checks. Its embedding `fits` flag is conservative: a currently claimable GPU must meet the configured memory floor and have an exact detected-name match in the pinned support matrix. Local NIM preflight always performs its own live checks and does not trust the cached recommendation. `PATCH /v1/embedding_deployment_config` provides low-level deployment configuration and `POST /v1/embedding_deployment_config:configure_self_hosted` is the UI's verified path. The latter repeats a real credential-free NeMo Retriever embedding request, requires one finite 2,048-dimensional vector, then persists the normalized URL and re-resolves every project. A healthy local embedding NIM recorded here is the default embedding provider — the background worker sends embedding traffic to it, falling back to the hosted endpoint, then pHash (see [deployment.md](deployment.md#gpu--local-nim-runtime-behavior)).
- **Testing** — `POST /v1/testing/projects/{project_id}/events:emit`: test-only SSE event injection. Mounted only when the backend starts with `VLM_ENABLE_TESTING_ROUTES=1`; by default the route does not exist (404) and is absent from `/docs` — the gate keeps SSE event spoofing out of production deployments.

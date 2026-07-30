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

One classifier (`services/errors.py::map_service_error`) produces these statuses for every endpoint, so the same error maps to the same code across the whole API. Interactive proposal endpoints are the deliberate exception for *upstream* failures: they return `200` with `invocation_status: "timeout" | "endpoint_error" | "rate_limited"` (the last distinguishes retry-exhausted hosted-NIM 429s) so the failed invocation is recorded and the UI can offer retry affordances.

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

Images are stored **by reference** — the request carries an `example_key` and an absolute `storage_ref` path; files are never copied into the workspace. Ingestion is per-item with partial success: each item returns `status` of `created`, `exists` (idempotent re-ingest), or `error` with an `error_code`. Error codes include `example_key_collision` (same key, different path) and `path_not_allowed` (the `storage_ref` falls outside `IMAGE_ROOT`). The root boundary is enforced at ingest so an out-of-root image is never persisted. It is unrestricted only for a loopback/single-user backend with `IMAGE_ROOT` unset.

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

The gate response reports `gate_status` (`ready` / `not_ready`) and per-criterion detail for `overall_exact_match`, `per_field_match`, `min_per_value_f1`, `accept_rate`, and `min_test_pool_size`. Passing the gate unlocks Batch Labeling. Student Training navigation is independent; `POST .../training_suites` performs its own fail-closed TAO/workspace/timeout/data/role validation and creates a conditional provisioning stage for selected missing bases. Gate thresholds are per-project settings (`PATCH /v1/projects/{id}`).

## Batch labeling (§10.2.6)

Auto-label remaining Unlabeled examples at scale with the trusted Teacher setup.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/batch_label_runs` | Start a run (201); optional `run_limit`, `include_auto_labeled`, `concurrency` (dispatch-width override; default is provider-aware — hosted 1 / self-hosted 8), ingest-date bounds, `icl_mode` (`enabled`/`disabled`, default `enabled` — disable to batch-label at the Teacher's zero-shot form for ICL-negative teachers, §8.3) |
| `GET` | `/v1/projects/{project_id}/batch_label_runs` | List runs |
| `GET` | `/v1/projects/{project_id}/batch_label_runs/{run_id}` | Run status and counters |
| `POST` | `/v1/projects/{project_id}/batch_label_runs/{run_id}:resume` | Resume a paused run |
| `POST` | `/v1/projects/{project_id}/batch_label_runs/{run_id}:cancel` | Cancel a run |
| `GET` | `/v1/projects/{project_id}/batch_label_runs/{run_id}/schema_invalid_manifest` | Manifest of items whose Teacher output failed Core schema validation (§10.2.24) |

## Dataset exports (§10.2.18)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/dataset_exports` | Create a dataset export (201, `status: "running"`; the archive builds in the background — poll the record or follow `export_*` SSE). **409** while another export is already building for the project. `dataset_intent`, `label_tier_filter`, `export_field_mode`, optional `batch_label_run_id` and selection filters |
| `GET` | `/v1/projects/{project_id}/dataset_exports` | List exports (`dataset_intent` filter) |
| `GET` | `/v1/projects/{project_id}/dataset_exports/{dataset_export_id}` | Get an export record (status, progress, paths, counts) |

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

## Student training (§10.2.7, §10.2.22, §9)

Fine-tune Student models (Cosmos-RL / TAO) from Verified data. A **training suite** is the composite one-POST path: optional first-use base provisioning, dataset exports, pre-created TAO chain jobs (train → quantize → evaluate), and first-chain kickoff. Individual TAO jobs can also be managed directly.

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

`training_preflight` accepts `student_base_model_config_ids[]`,
`include_auto_labeled`, and `enable_lora` (default `true`). LoRA readiness also
checks the Blueprint's local merge runtime and `HF_TOKEN`, because the
full-precision baseline is merged and evaluated through the local Student NIM
instead of TAO's adapter-incompatible evaluate action. In addition to
structured checks, `data_summary`
returns `verified_training_count`, `test_pool_count`,
`auto_labeled_eligible_count`, `auto_labeled_included_count`,
`excluded_test_pool_count`, `excluded_auto_labeled_count`, and
`usable_training_count`. Test Pool examples are always excluded.

When selected bases are missing, the suite response is provisional and the
Training Jobs screen polls it while the backend drives the durable provisioning
run and subsequent suite materialization. An interrupted provisioning run and
provisional suite are marked `failed` on backend restart and can be retried
safely. Suite creation takes `student_base_model_config_ids`,
`training_preset`, `include_auto_labeled`, `export_field_mode`, and
`quantization_schemes` (omission defaults to `["FP8_DYNAMIC"]`). The
suite-level cancel endpoint preserves completed jobs, requests cancellation
for every known active external TAO job, cancels all remaining local jobs, and
sets the suite to `canceled`. The response reports canceled/terminal counts
and `remote_cancel_failures[]`. A remote failure does not keep the project
locked in Training Jobs; clients must surface the warning because TAO work may
still be running.

TAO wire format is specified in §9.7.

## Student deploy & benchmark (§10.2.20, §10.2.21, §9.5)

Trained Students are deployed behind a NIM and benchmarked against the Teacher on the Compare screen, which reads the Teacher baseline and each Student's quality/serving results directly from evaluation runs.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/projects/{project_id}/student_models` | List StudentModel records |
| `GET` | `/v1/projects/{project_id}/student_models/{student_model_id}` | Single Student (quality/serving status, checkpoint state) |
| `POST` | `/v1/projects/{project_id}/student_models/{student_model_id}:deploy_nim` | Trigger the Student NIM lifecycle (202, runs in background); `nim_endpoint_url: null` = local Docker orchestration, a URL = register an external endpoint |
| `POST` | `/v1/projects/{project_id}/student_models/{student_model_id}:deployment_handoff` | Generate the deployment-handoff Action Request; 409 unless both quality and serving gates are `validated` |
| `POST` | `/v1/projects/{project_id}/student_models/{student_model_id}:rerescore` | Replay the quality rescore for a `failed` Student after a scoring fix (409 otherwise) |
| `POST` | `/v1/projects/{project_id}/student_models/{student_model_id}:repackage` | Replay checkpoint packaging for a Student whose `checkpoint_packaging_status` is `failed`, after an environment fix — e.g. the LoRA-merge interpreter was unprovisioned (409 otherwise) |

The local `:deploy_nim` lifecycle is preflight → `docker run` → health → smoke → temp endpoint registration → evaluation → benchmark sweep → stop, with progress on SSE (`nim_benchmark_progress` / `nim_benchmark_completed`). It defaults to replace semantics on an occupied GPU and auto-restores a displaced local Teacher afterwards (§9.5.2).

## Local NIM lifecycle (§1.5)

Deploy and manage local NIM containers (Teacher or embedding) on this host's GPUs.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/projects/{project_id}/local_nim/preflight` | Dry-run preflight checks (Docker, GPU memory, NGC credential) without starting a container |
| `POST` | `/v1/projects/{project_id}/local_nim/deploy` | Queue or reuse a local NIM (201). A new deploy returns `disposition="queued"` plus a persisted `status="starting"` reservation, then runs preflight/startup in the background. An exact compatible running Teacher returns `disposition="reused"`, `deployment=null`, and a safe `resident` summary. Body: `{role: "teacher"\|"embedding", model_config_id, gpu_assignment?, preferred_port?, replace_resident?, activate_on_success?}` |
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

Create a project:

```bash
curl -s -X POST http://localhost:8000/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "rps-demo", "description": "Rock/paper/scissors labeling"}'
```

```json
{
  "project_id": "a1b2c3d4e5f6",
  "name": "rps-demo",
  "description": "Rock/paper/scissors labeling",
  "project_dir": "/home/user/.vlm_feedback_loop/workspace/projects/a1b2c3d4e5f6",
  "counts": {"verified": 0, "unlabeled": 0, "auto_labeled": 0, "omitted": 0, "pending_relabel": 0, "prior_relabeled": 0},
  "teacher_model_config_id": "9f9c7c2e-3ad9-4b7e-9d2f-825b6eed4500",
  "active_guidance_id": null,
  "embedding_provider": "hosted",
  "test_pool_fraction": 0.40,
  "created_at": "2026-07-10T17:03:12Z",
  "...": "..."
}
```

Ingest images by reference (returns 202; pHash and embeddings compute in the background):

```bash
curl -s -X POST http://localhost:8000/v1/projects/a1b2c3d4e5f6/examples:ingest \
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

Request a Teacher proposal (after creating a Guidance and selecting a Teacher model config):

```bash
curl -s -X POST http://localhost:8000/v1/projects/a1b2c3d4e5f6/proposals \
  -H "Content-Type: application/json" \
  -d '{"example_key": "rock_001"}'
```

```json
{
  "inference_invocation_id": "inv_9f8e7d6c",
  "example_key": "rock_001",
  "proposal_json": {"gesture": "rock"},
  "schema_valid_core": true,
  "validation_errors_core": [],
  "validation_errors_aux": [],
  "invocation_status": "success",
  "latency_ms_end_to_end": 2140,
  "icl_images_attached_count": 3,
  "icl_example_keys_used": ["scissors_017", "paper_042", "rock_088"],
  "used_existing_label": false
}
```

Save the label (an unchanged `label_json` classifies as Accept; any diff classifies as Edit):

```bash
curl -s -X POST http://localhost:8000/v1/projects/a1b2c3d4e5f6/labels \
  -H "Content-Type: application/json" \
  -d '{
    "example_key": "rock_001",
    "inference_invocation_id": "inv_9f8e7d6c",
    "label_json": {"gesture": "rock"}
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
  "pool_assignment": "test_pool"
}
```

Check the Scale-Up Readiness Gate:

```bash
curl -s http://localhost:8000/v1/projects/a1b2c3d4e5f6/scaleup_gate
```

```json
{
  "gate_status": "not_ready",
  "criteria": [
    {"criterion_name": "overall_exact_match", "passed": true, "current_value": 0.92, "threshold": 0.9, "message": "Latest evaluation exact match 92.0% meets threshold 90.0%", "details": null},
    {"criterion_name": "per_field_match", "passed": true, "current_value": 0.95, "threshold": 0.85, "message": "...", "details": null},
    {"criterion_name": "min_per_value_f1", "passed": true, "current_value": 0.88, "threshold": 0.7, "message": "...", "details": null},
    {"criterion_name": "accept_rate", "passed": false, "current_value": 0.78, "threshold": 0.9, "message": "Accept rate over the last 20 labels is 78.0%; threshold 90.0%", "details": null},
    {"criterion_name": "min_test_pool_size", "passed": true, "current_value": 72, "threshold": 60, "message": "...", "details": null}
  ],
  "evaluated_at": "2026-07-10T17:12:03Z"
}
```

## Supporting endpoints not covered here

The following routers back setup screens and utilities rather than the core loop. Their full contracts are in the runtime `/docs` (Swagger UI):

- **Filesystem** — `GET /v1/filesystem/browse`, `POST /v1/filesystem/scan`: server-side directory browsing and recursive image scans (with suggested example keys) for the ingestion picker, constrained by one `IMAGE_ROOT`. Omit the browse `path` query parameter to open the deployment root automatically.
- **Secrets** — `POST /v1/secrets:set`: apply a deployment secret (e.g. `NVIDIA_API_KEY`) at runtime, optionally persisting to `~/.vlm_feedback_loop/.env` when `ALLOW_UI_SECRET_PERSIST` allows.
- **Model configs** — `/v1/projects/{project_id}/model_configs` create/list/get/update plus `POST .../{model_config_id}:reprobe` for capability re-probing (§10.2.12). Each entry carries `default_icl_max_examples` (int | null): the per-model default ICL depth cap applied to proposals, evaluations, and batch labeling when no explicit `icl_max_examples` override is present (§6.2). Seeded from the July 2026 cross-model depth studies (Nemotron Nano VL 2 · Nemotron 3 Nano Omni 4 · Cosmos CR3 8 · CR2-2B 8 · CR2-8B 16 · MiniMax-M3 8 · Mistral Large 2); settable at create and PATCHable for re-tuning.
- **NIM endpoints & environment** — `/v1/projects/{project_id}/nim_endpoints` create/list/get/update for registered inference endpoints; `GET /v1/environment` and `POST /v1/nim/test_connection` / `test_ngc_credential` / `test_nvidia_credential` connectivity checks; `PATCH /v1/embedding_deployment_config` (deployment-scoped embedding-NIM config; a healthy local embedding NIM recorded here is the default embedding provider — the background worker sends embedding traffic to it, falling back to the hosted endpoint, then pHash — see [deployment.md](deployment.md#gpu--local-nim-runtime-behavior)).
- **Testing** — `POST /v1/testing/projects/{project_id}/events:emit`: test-only SSE event injection. Mounted only when the backend starts with `VLM_ENABLE_TESTING_ROUTES=1`; by default the route does not exist (404) and is absent from `/docs` — the gate keeps SSE event spoofing out of production deployments.

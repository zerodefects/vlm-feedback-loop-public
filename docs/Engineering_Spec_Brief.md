# Interactive VLM Feedback Loop - Engineering Specification (Brief)

*A structure-mirroring brief of `Engineering_Spec.md`. Each section is either an **exact copy**, a **summary with reference**, or a **reference only**. Section numbers match the full spec; cross-references like "§4.3" resolve to the matching section in `Engineering_Spec.md`. Read the full section there when implementing or modifying the referenced subsystem.*

**Treatment legend (shown on each heading):** `[EXACT]`, `[SUMMARY]`, `[REFERENCE]`.

---

## Decisions (mandatory)  `[EXACT]`

- **SchemaCore supports schema-evolution-based Core changes.** A small set of non-semantic Core edits (field rename, exact 1:1 enum value rename, presentation metadata) are allowed in place. All other Core changes (add/remove field, type change, constraint change, allowed-value change, role change) trigger label invalidation: existing Verified and Auto-Labeled labels are deleted, examples return to `Unlabeled` with prior label data preserved on the Example record, and the standard labeling loop re-labels them with prior-label hints (annotated reference, "Adopt prior" per-field action). The review selector prioritizes examples with prior labels (prior Edits first to seed ICL). There is no separate re-verification state or workflow. Aux-only additive extensions remain allowed in place without label invalidation.
- **Core/Aux roles define requiredness and evaluation**
  - `role="core"` fields are **required** and **evaluated**
  - `role="aux"` fields are **optional** and **not evaluated**
- **Structured generation runtime failure triggers per-invocation fallback, not permanent downgrade.** A runtime `json_schema` rejection after a prior successful probe is treated as a transient or schema-specific issue. For interactive invocations, the system retries once with prompt-only JSON generation and surfaces a warning. For evaluation and Batch Labeling, the system MUST NOT silently mix structured and prompt-only modes within the same run; the run fails and the SME restarts under prompt-only mode. The model config remains `supported`; structured generation is still attempted on subsequent invocations. Permanent downgrade to `unsupported` requires a re-probe failure. See §6.2 for the runtime rejection policy.
- **Exact Match boolean normalization is strict:** canonical boolean values are JSON `true` and `false` only. String proxies (e.g., `"true"`, `"false"`, `"yes"`, `"no"`) and numeric proxies (e.g., `1`, `0`) are not canonical and MUST be treated as schema-invalid. This aligns with JSON Schema `boolean` typing and NVIDIA structured generation. The system MUST use a single backend-canonical boolean normalizer across all evaluation paths: interactive proposal validation, evaluation, Batch Labeling validation, and TAO per-sample re-scoring (Appendix A.2). Canonical stored labels (`label_json`, `normalized_json_ref`) MUST contain actual JSON booleans; string/numeric representations are preserved only in the raw artifact (`raw_model_response_ref`) for audit. When structured generation is enabled, boolean fields are schema-constrained to JSON booleans by the derived JSON Schema. When structured generation is unavailable (prompt-only fallback), a model response containing a string proxy for a boolean field is classified as schema-invalid — evaluation semantics are identical regardless of the generation mode used.
- **Cold-start UI verification is UI-driven:** Verified examples are created by Accept/Edit in the UI. The system MUST function when `Verified = 0` (ICL degrades to zero examples).
- **Model selection is catalog-driven**
  - Every model used for inference, evaluation, or batch labeling MUST reference a catalog entry with the appropriate `eligible_roles` value (`teacher` or `student_base`).
  - Student Training MAY produce multiple Student variants; evaluations MUST record which Student variant was used.
- **Prompt budgets are per-model:** prompt packing MUST derive effective input budgets from the selected model's configured context window.
- **Comparison reads from evaluation runs directly:** the Compare screen's Teacher baseline is the most recent completed Teacher-contract evaluation run; each Student's quality/serving results come from the runs referenced on its StudentModel record. The project-wide registry is grouped by immutable `training_suite_id`, newest first; run-qualified chart labels disambiguate repeated candidates. Cross-run evidence is labeled directional when Guidance, Inference Contract, or frozen evaluation evidence differs, and Teacher deltas fail closed on context mismatch. (Pre-v1.0 evaluation-suite execution grouping remains removed.)
- **Batch Labeling is Teacher-run:** the batch fully synthetic label generation workflow in this spec runs the **Teacher**.
- **Retry is a first-class interactive action:** the UI MUST offer **Save / Skip / Retry** per image. Retry MUST produce a new proposal attempt for the same example after applying a user-selected Teacher model and/or Guidance. Retry MUST NOT create a Verified label by itself.
- **Skip omits the image:** Skip MUST record no label and MUST omit the image from the workflow so it is not presented again.
- **Review selector selection is diversity-driven whenever a signal is available:** a restartable background pHash sweep provides the baseline signal. When per-image CLIP embeddings exist, the selector upgrades to CLIP-diverse mode (semantic diversity). While all candidate signals are still pending, the selector uses deterministic `example_key` order; there is no random selection mode.
- **All persisted entities are project-scoped:** every Example/Label/Guidance/Pool/Run/Operation/ModelConfig MUST reference `project_id`.
- **Cosmos-RL / TAO VLM jobs are first-class and poll-tracked:** every Student Training fine-tuning execution MUST create a **TAOJob** record with an explicit TAOJob state machine, persist the exact job configuration payload used to start the training backend, and track (by reference) required outputs: **artifacts, metrics/progress, and logs**. Status MUST be updated via polling the TAO Jobs API and mapped deterministically into the TAOJob state machine.
- **Rationale notes are opt-in auxiliary data, never prediction authority:** new Guidance disables `rationale_note` by default. The SME may toggle the reserved Aux string field freely without label invalidation (§4.4). When enabled, the production Teacher requests it last and the existing review/provenance workflow applies. When disabled, prompts, model contracts, proposals, review, save validation, and new Verified labels bypass it completely. Production Teacher ICL remains Core-only. Export field mode controls serialization of rationale data that exists (§9.3).
- **Embedding computation is default-on; the provider is a fallback cascade (§5.5.1):** a healthy local embedding NIM deployment (live-verified at probe time) is the default whenever present; the hosted embedding NIM (hosted NIM API key) is the fallback; `none` (pHash-diverse selection) is the last resort. The supported Blueprint model is **NVIDIA NeMo Retriever VL 1B v2** (`nvidia/llama-nemotron-embed-vl-1b-v2`, 2048-dim, requires `input_type` in the request body). Embedding computation MUST NOT block interactive labeling (§5.5.2).
- **Pool management is fully automatic:** a configured fraction sets the Test Pool target, and the next Verified example fills an immediate shortfall regardless of verification outcome. Rebalancing uses CLIP-diverse selection once enough embeddings exist and pHash-diverse selection otherwise. Both Accepts and Edits are candidates for the Test Pool; ICL draws exclusively from non-pool Edits. Evaluation runs auto-create frozen pool version snapshots for reproducibility. No user action is required (§4.3).
- **Scale-Up Readiness Gate governs Batch Labeling:** the Batch Labeling CTA MUST be gated on configurable, system-evaluated criteria (evaluation overall Exact Match, per-core-field match rate, minimum per-value F1, Accept rate over a rolling window, and minimum Test Pool size). Clicking [Run Batch Labeling] when criteria pass IS the SME's confirmation. The gate MUST NOT block Interactive Labeling or evaluation runs. Thresholds are configurable per project (§7.3).
- **Student Training has an independent data-readiness gate:** Teacher Exact Match, per-field, per-value F1, and Accept-rate criteria do not gate Student Training. Start requires at least one active-Guidance non-pool Verified training example plus `max(1, project.scaleup_min_test_pool_size)` active-Guidance Test Pool examples for automatic held-out evaluation. The backend repeats these checks after base provisioning (§9.7.8.3, §10.2.22).
- **Generation Controls are preset-driven:** labeling workflows expose Output Stability presets and a Thinking toggle as the only user-facing sampling knobs; raw parameter editing is not permitted. Effective values are always persisted in Operation Records. See §6.7.
- **Visual Budget Controls are capability-gated and preset-driven:** image preprocessing (visual token spend) is managed via named presets (**Fast** / **Balanced** / **High Detail**) that resolve to model-specific `mm_processor_kwargs`. Controls are only exposed when the active Teacher supports them (`visual_budget_mode` ≠ `none`). The effective `mm_processor_kwargs` sent per invocation are persisted on Operation Records. Evaluation and Batch Labeling use the project's visual budget preset (no per-example variation) for apples-to-apples reproducibility. See §6.9.
- **Student Training backend is Cosmos-RL / TAO VLM supervised fine-tuning:** Student Training is not a generic TAO operation. It is an explicit Cosmos-RL / TAO VLM supervised fine-tuning workflow with TAO-native dataset, config, checkpoint, quantization, and NIM deployment handoff requirements. Only catalog entries with the `student_base` role are eligible (currently Cosmos Reason2 8B/2B and Cosmos 3 Nano/Super reasoner). The default training policy is `sft` (supervised fine-tuning); implementations MUST NOT assume RLHF-like behavior from the "RL" in the product name.
- **Student Training is LoRA-first by default.** `enable_lora=true` is the default training mode. LoRA is the parameter-efficient path suited to iterative, multi-variant, cost-conscious VLM fine-tuning and comparison. Cosmos-RL does not require LoRA, but this system's default training configuration uses it. The `lora_config` on every TAOJob (§9.7.3.2) records the LoRA parameters used. Full-weight fine-tuning (if ever needed) would set `enable_lora=false`, but this is not the default path and requires the operator to understand the increased resource and packaging implications.
- **The Training UI exposes the qualified LoRA matrix only.** It keeps `enable_lora=true`, hides the unqualified Full-weight selector, and filters Cosmos 3 Super from new UI training runs. Backend preflight and suite creation retain `enable_lora=false` and Super compatibility handling for API compatibility, historical evidence, and future qualification.
- **Full-precision baseline NIM packaging owns the LoRA merge path.** For quantized variants, TAO `quantize` auto-merges the LoRA adapter when `enable_lora=true` and `base_model_path` is set (§9.8). For the full-precision baseline variant, no quantize step occurs, so no auto-merge happens. Checkpoint packaging (§9.5.1) MUST handle this: it first validates whether the training output is already a NIM-loadable HuggingFace checkpoint directory; if the output is adapter-only, packaging MUST merge the adapter into the base model using the persisted `base_model_path` (sourced from the training TAOJob's `resolved_training_fields.policy.model_name_or_path`) and produce a merged checkpoint. If `base_model_path` is unavailable or merge fails, packaging fails with a clear error. `export_safetensors: true` in the training config controls checkpoint serialization format (SafeTensors vs PyTorch .bin); it does NOT guarantee LoRA adapter merge.
- **Runtime stack is mandated; background tasks use in-process execution with SSE notification:** the system MUST use Python + FastAPI (backend), React + TypeScript (frontend), and SQLite (per-project database). Evaluations, CLIP embedding computation, and Batch Labeling MUST run as in-process background tasks with durable run records. On restart, the backend MUST detect incomplete background tasks and either resume idempotently or mark them failed. The backend MUST support SSE for pushing progress and completion events to the UI; REST status endpoints MUST be retained for initial state load and reconnect fallback (§1.3, §1.8).
- **JSON Schema derivation is backend-canonical:** the backend MUST be the sole canonical implementation of SchemaCore → JSON Schema derivation. The frontend performs lightweight local form validation for immediate UX feedback (§6.6.6) and requests backend canonical validation for schema derivation, JSON Schema compilation, and save readiness via the draft validation endpoint (§10.2.2). The same backend derivation function MUST be used by both the draft validation endpoint and the Guidance save endpoint. No shared cross-language derivation library is required.
- **Post-training evaluation is automatic and TAO-backed:** when a TAO training job succeeds, the system MUST automatically submit a TAO `evaluate` job against the Test Pool export. When a TAO quantize job succeeds, the system MUST automatically submit a TAO `evaluate` job against the quantized checkpoint. TAO per-sample predictions MUST be re-scored by the system's canonical Core-field evaluator (Appendix A.2) to preserve product-specific correctness semantics. TAO's native aggregate metrics are informational; the system's re-scored metrics are authoritative. This chain (train → evaluate → quantize → evaluate) runs without SME intervention after "Start Training."
- **First-run quantization is FP8_DYNAMIC only:** Validate training setup
  selects one recommended small base, Quick, and baseline + FP8, expanding to
  four jobs. Multi-base and additional quantization comparison is an explicit
  advanced intent. Every selected scheme remains a separate TAO `quantize`
  job followed by a TAO `evaluate` job.
- **Student quality and serving readiness are separate gates:** Student quality is validated by TAO evaluation (accuracy/F1/precision/recall against the Test Pool, re-scored by the canonical evaluator). Student serving readiness is validated by NIM deployment + evaluation (latency, throughput, profile metadata). Quality validation does not require NIM. Serving validation does. The `deployment_handoff` Action Request (§10.3) requires both gates to pass.
- **Quality status is four-state, not three (F35):** `quality_status ∈ {pending, validated, partial, failed}`. `partial` is **informational only** — it surfaces "the model serves but did not produce parseable output on every example" as a yellow badge in the UI, but the `deployment_handoff` dual gate STILL requires `validated`. Only NIM-eval can land `partial` (when run finishes `incomplete` AND parseable rate ≥ `STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD`, default 0.90); TAO rescore stays binary success/fail. The audit invariant "validated is never demoted" holds — `validated → partial` is illegal, the promotion helper is a no-op when prior status is `validated`. F33 `:rerescore` rejects `partial` (§9.7.6) — the remediation path is to re-run NIM eval, not to replay a TAO rescore.
- **NIM deployment for Student evaluation uses local orchestration with Action Request fallback:** the system attempts to deploy Student NIM containers locally on the backend host via Docker (§9.5.2). A NIM deployment preflight checks prerequisites (Docker, NVIDIA Container Toolkit, GPU memory, NGC API key). If preflight passes, the system orchestrates the full container lifecycle (start → health poll → evaluate → production-representative benchmark → stop) per variant, sequentially. If preflight fails, the system generates an Action Request with the exact cache-disabled `docker run` command and prerequisites so the SME can hand off to infrastructure. Student NIM endpoints are temporary evaluation infrastructure, not production deployments.
- **One-NIM-per-GPU invariant (F49 amendment, 2026-05-19; §1.5, §9.5.2):** at most one NIM container is `starting | running` on any GPU at any time. Fresh-project creation reuses and selects an exact running Teacher only when it matches the current quality recommendation; a different resident becomes an explicit keep/replace choice. Later deploys repeat the exact-runtime reuse check. To start a different NIM on an occupied GPU, the orchestrator MUST first stop the resident (the *replace* semantics, opt-in via `replace_resident=true` on `POST :deploy`). Multi-GPU placement is deterministic (lowest free index first). Single-GPU local-only hosts support exactly one NIM at a time: FTUE Run-locally is Teacher-only with pHash diversity fallback (§5.6); Student NIM benchmarking displaces the Teacher in step 0 and best-effort auto-restores it in step 9. Empirical motivation: Cosmos Reason2 NIM's hardcoded `gpu_memory_utilization=0.9` profile-selector floor (README "One-NIM-per-GPU policy").

---

## 1. Introduction & Scope

### 1.1 Purpose  `[SUMMARY]`

Implementation contract for data states/transitions, ICL selection and prompt rendering, inference + review workflows (Retry/Skip), evaluation workflows and metrics, optional Batch Labeling (Teacher-run synthetic labels), optional Student Training and Student tracking, diversity-driven review selectors (pHash baseline → CLIP upgrade), catalog-driven multi-model support, project/workspace management. Supports both NVIDIA-hosted NIM (API Catalog) and self-hosted NIM. **v1 is single-user, single-session-per-project; no auth, no RBAC, no multi-reviewer — trusted-network assumption.** Multi-user is post-v1. See §1.1 for full prose.

### 1.2 Reuse-first Principle  `[EXACT]`

Implementations MUST follow these reuse boundaries:

- Model serving MUST use NVIDIA NIM for Teacher/Student endpoints.
- Embedding computation MUST use NVIDIA NeMo Retriever VL 1B v2 NIM (`/v1/embeddings`, 2048-dim) for the supported Blueprint path. Legacy `*_nvclip` provider enum names remain only for database/API compatibility.
- Student Training fine-tuning MUST use Cosmos-RL / TAO VLM fine-tuning via NVIDIA TAO Toolkit (when enabled).

### 1.3 Execution Model  `[EXACT]`

- Primary API is synchronous request/response (REST). No WebSockets.
- The backend MUST support Server-Sent Events (SSE) for real-time progress and completion notifications on background tasks. SSE is the primary UI update channel for background work. REST status endpoints MUST be retained for initial state load, reconnect after SSE disconnect, and polling fallback.
- Required SSE event types: `evaluation_started`, `evaluation_progress`, `evaluation_completed`, `embedding_progress`, `embedding_completed`, `batch_label_progress`, `batch_label_completed`, `export_progress`, `export_completed`, `export_failed`, `run_failed`. Events are project-scoped.
- **SSE is best-effort and non-authoritative in v1.** The authoritative source of state is the REST API backed by persisted project records. SSE provides inexpensive live progress updates for long-running work (evaluations, embeddings, batch labeling, TAO polling, NIM benchmarking) but missed events during a disconnection are not replayed in v1. No replay buffer, `Last-Event-ID` replay semantics, or `reconnect_stale` protocol are required. The browser's EventSource reconnects automatically on connection loss; the frontend treats reconnect as a cue to refresh state from REST, not to expect event replay.
- **Frontend recovery contract:** on page load, fetch REST snapshot first. While SSE is healthy, use SSE as a hint/update channel. On any SSE disconnect or error, if active background work exists, begin short-interval REST polling (e.g., 5s). On SSE reconnect/open, immediately refresh relevant REST state. On terminal SSE events (`*_completed`, `run_failed`), immediately refresh REST state. This preserves a live feel without requiring replay correctness.
- Evaluations, CLIP embedding computation, and Batch Labeling run as **in-process background tasks** within the backend process. Run/task records MUST be persisted in the project database for observability and recovery. On backend restart, the system MUST detect incomplete background tasks. Recovery is workload-specific and governed by the run state machines (§13.2.1, §13.2.2): **evaluation** runs in non-terminal states transition to `failed` with `status_reason="backend_restart_interrupted"` (the user re-triggers); **CLIP embedding** computation resumes automatically from the first example without an embedding; **Batch Labeling** runs in `queued` or `running` state transition to `queued` with `recovered_from_restart=true` and auto-resume; runs in `paused` state remain `paused`.
- **Single-process concurrency model (v1):** v1 supports one backend process per project. The backend MAY optionally acquire a simple file lock when a project is opened; if a second process attempts to open the same project, it receives a hard error: *"This project is already open in another process."* No override path, no warning-plus-continue flow. Multi-session UX is post-v1. Within a single process, the backend has multiple concurrent writers (interactive labeling, evaluation, embedding, batch labeling); see §1.8 SQLite write-discipline rules.
- **Graceful shutdown:** on backend process exit, in-flight background tasks are canceled after a short grace period. The system persists partial results and marks interrupted run states before exiting. Browser tab closure does not stop the backend; background tasks continue until the backend process itself exits.
- Student Training (when enabled) runs on **external executors** and is observed via polling/completion checks. Student Training MUST be represented as a poll-tracked TAOJob with durable state, logs, metrics, and artifact references. **TAO recovery on backend restart:** the system MUST reconcile all persisted TAO job chains. For any TAOJob in `submitting` with `tao_external_job_id = null`, transition to `failed` with `status_reason="submission_interrupted"`. For any TAOJob in a non-terminal state (`submitted`, `queued`, `running`, `paused`) with a known `tao_external_job_id`, the system MUST resume polling because TAO jobs execute externally and continue regardless of backend availability. For any chain where the job at `chain_sequence = N` has `status=succeeded` and the next job at `chain_sequence = N+1` has `status=not_started`, the system MUST submit that next job. If any prior job in the chain has `status=failed` or `status=canceled`, the chain remains halted. **Outputs-fetch recovery (Phase 12 amendment 2026-05-05, §13.12):** for any TAOJob in `status="succeeded"` whose `outputs_fetch_status` is still `pending` or `in_progress` (the fingerprint of a backend crash mid-multi-GB-artifact-download), the polling tick MUST re-fire the success handler — fetch artifacts, merge resolved fields, emit `tao_job_completed`, run downstream Student registration / re-scoring, and re-attempt chain advance. Idempotent: artifact downloads use atomic `.part` files; downstream services tolerate re-execution; chain advance only submits jobs in `not_started`. `outputs_fetch_status="failed"` is terminal — operator intervention required.
- **Training Jobs exposes post-success work honestly:** TAO job detail and suite
  chain summaries carry `outputs_fetch_status` and sanitized
  `outputs_fetch_error_ref`. The monitor keeps polling a TAO-succeeded job while
  Blueprint artifact handling is active, shows **Finalizing**, explains the
  downstream artifact wait, and renders terminal artifact handling failure as
  **Artifact Failed** rather than a misleading completed/stalled chain.
- **Foreground priority on shared endpoints:** two workload classes. **Foreground:** `interactive_proposal`, `retry`, `rationale_regeneration`. **Background:** `evaluation`, `batch_label`, `embedding`. When one or more foreground requests are in flight, the system MUST hold dispatch of new background HTTP requests until foreground demand clears. Already-in-flight background requests MAY complete naturally. This is a simple dispatch hold, not preemption. Background work may complete more slowly while the SME is actively labeling; that tradeoff is intentional.

### 1.4 Optimization Objectives  `[SUMMARY]`

Minimize feedback→behavior latency (edits influence next call); maximize label efficiency via ICL under token constraints; quick onboarding (hosted NIM + API key); diversity-driven SME throughput (background pHash baseline, CLIP upgrade); scale via Batch Labeling with lineage; tradeoff decisions via multi-Student compare; interactive iteration via Retry; Skip keeps Omitted images hidden; strict project scoping; explainability via lineage/artifacts/audit; full TAO observability. See §1.4.

### 1.5 NIM Endpoint Modes  `[SUMMARY]`

Three modes supported:

- **Mode A — Hosted NIM (API Catalog):** base URL `https://integrate.api.nvidia.com/v1`, `Authorization: Bearer $NVIDIA_API_KEY`. `GET /v1/models` when supported.
- **Mode B — Self-hosted NIM:** deployment-defined base URL (e.g., `http://0.0.0.0:8000/v1`). No credential attached — expected on a trusted private network or behind an external gateway (`auth_mode="none"`).
- **Mode C — Local NIM Deployment (System-Managed):** Docker orchestration for the NeMo Retriever VL 1B v2 embedding NIM and seeded local Teacher/Student entries carrying `local_deploy_metadata`.

Environment assessment produces `EnvironmentAssessment` (§10.2.25): credentials, Docker/toolkit, GPU memory + compute capability, deployable models, placement-aware embedding availability, and active Blueprint-managed NIM residents. It is deployment-scoped: the browser warms it once without blocking the Project List, and the backend caches the expensive Docker/toolkit/GPU snapshot for its process lifetime while recomposing current credentials, embedding configuration, and resident state on every response. `refresh_hardware=true` explicitly re-probes after an operator changes the machine; live deployment preflight remains uncached. Project routes that do not consume machine capabilities do not wait for it. Hardware gates eligibility; a curated evidence-backed quality rank selects Omni on supported ≥80 GB / cc≥9.0 GPUs, CR3-Nano at ≥56 GB when Omni is ineligible, and CR2-2B at 36–55 GB. Memory-floor checks allow a 1% `nvidia-smi` reporting tolerance so a nominal 80 GB card reported near 79.6 GiB remains in its documented tier. Super and CR2-8B remain selectable. An exact healthy Teacher resident outranks a hosted key only when it matches that recommendation: fresh-project creation attaches its endpoint and selects that model before returning. A different resident remains visible for an explicit keep/replace choice. Without a preferred resident, a fitting local Teacher is recommended for lower latency; an API key supplies an immediate hosted bridge during download, and the project activates local atomically only after verification. Hosted-only remains available. Supported GPUs in the 24–35 GB tier recommend hosted Teacher + local NeMo Retriever VL embeddings; below 24 GB, embeddings are hosted or pHash. Automatic embedding recommendations require both the memory floor and an exact detected GPU-name match in the pinned NIM support matrix; unrecognized hardware is left to an explicit manual fallback attempt.

Post-onboarding NIM Configuration joins `local_deployable_models[].fits` with the project model catalog and lists every compatible local Teacher, with the quality recommendation and running resident marked. Choosing a model sends an exact `model_config_id`; the first deploy request never replaces a resident. A structured 409 names the exact resident/GPU/project before the SME may confirm `replace_resident=true`. `activate_on_success=true` is durable deployment intent: the project switches Teachers only after the new/reused endpoint is verified. Failed replacement preflight, startup, health, or served-model verification best-effort requeues every displaced resident from the displacement audit.

The same screen's self-hosted embedding override is deployment-scoped. Test
and Save each call the actual `/embeddings` operation using the seeded NeMo
Retriever model and require one finite 2,048-dimensional vector. Only Save
persists the credential-free URL and re-resolves every non-archived project. A
running Blueprint-managed embedding resident permits an idempotent recheck of
its exact URL but blocks switching to a different URL until the SME explicitly
stops that resident.

Mode C preflight (per-model): Docker available, NVIDIA toolkit GPU passthrough, GPU memory ≥ `nim_gpu_memory_minimum_gb`, `NGC_API_KEY` configured, model-specific profile compatible (`list-model-profiles`), container image pullable. An exact missing-utility result from a published single-model image is inconclusive and proceeds to bounded serve health and served-model verification; shared-image selections remain blocked without their size-aware probe. Container lifecycle: pinned image, name-only `-e NGC_API_KEY` forwarding, NIM cache mount, loopback-only `127.0.0.1:{host_port}→8000` publication, `--runtime=nvidia --gpus device=N`, `--shm-size=32GB`, named container. The Docker client receives the credential only through its private child environment; the value and `KEY=value` assignment never enter argv or operator-visible handoffs. Startup bounded by `NIM_STARTUP_TIMEOUT_S` (default 1200s) polling `/v1/health/ready`; after a backend restart, a persisted `starting` Teacher/embedding with a live but not-ready named container resumes that poll for a fresh configured window instead of being torn down. Teacher adoption (startup poll and restart-recovery rebind) is additionally gated on served-model verification plus a 1-token inference probe — a dead vLLM engine keeps serving health/metadata 200s from the surviving HTTP front-end; probe timeout = inconclusive pass. Every transition to a terminal state for a row whose container may still exist (health timeout, verification/probe failure, recovery of a previously `running` but now unready deployment) best-effort stops+removes the container so a terminal row never hides a live GPU resident from the placement scan. On healthy: auto-register endpoint in catalog (Teacher) or set `embedding_provider=self_hosted_nvclip` and re-resolve the provider for every non-archived project, draining pending embedding work (embedding NIM, §5.5.1); on stop/displacement/startup failure the embedding config resets to `provider=none`. Port allocation: `LOCAL_NIM_TEACHER_PORT=8000`, `LOCAL_NIM_NVCLIP_PORT=8001`; falls back to next available if occupied.

**One-NIM-per-GPU invariant (F49 amendment, 2026-05-19):** at most one NIM container is `starting | running` on any GPU at any time. Fresh-project creation reuses an exact compatible `running` Blueprint-managed Teacher across projects only when it matches the current quality recommendation; later explicit deploy requests reuse any exact requested runtime. The consumer gets a project-local endpoint attachment but does not own or stop the resident. A different resident is replaced only after explicit user confirmation and `replace_resident=true`. Owner state is durable in project SQLite; named containers survive restart and recovery health-verifies/re-adopts them. When an exact replacement/restored resident becomes healthy, projects still selecting a disabled managed attachment for that runtime are reattached; proposal dispatch never follows a disabled or hard-unhealthy attachment merely because its old port answers. See §1.5 for placement, single-GPU splits, displacement, and restore semantics.

### 1.6 TAO Endpoint Configuration  `[SUMMARY]`

Deployment-level (shared across projects), not per-project. Required when Student Training enabled:

```
TAO_API_BASE_URL    # base URL for TAO API v2 — no default
TAO_API_KEY         # bearer token for TAO calls — stored as secret (§12)
TAO_ORG_NAME        # organization name for /api/v2/orgs/{org_name}/...
```

`TAO_API_KEY` is an **application-level abstraction**: the value used as the `Authorization: Bearer` token. On stock FTMS v2, native auth is two-step: NGC Personal API Key → `POST /api/v2/login` → JWT → use JWT on subsequent calls. `TAO_API_KEY` stores the appropriate bearer token (typically the JWT). The Blueprint's `services/tao_auth.py` auto-exchanges NGC keys when detected.

Connection probe: `GET /api/v2/orgs/{org_name}/jobs?limit=1`. Failure surfaces: *"Could not connect to TAO. Verify the API URL, key, and organization name."* TAO FTMS manages artifact storage; this system retrieves artifacts via TAO Jobs API and imports them into `{project_dir}/artifacts/`. The system MUST NOT assume direct filesystem access to TAO's workspace storage.

Workspace bootstrap remains a deployment-time operation. Base experiments use two paths: **self-service (default)**, launched automatically for selected missing bases by **Start Training** (or explicitly via `vlm-feedback-loop tao-pull-base-experiments`), and **admin handoff (fallback)** for air-gapped or policy-separated deployments. Self-service invokes `nvidia-tao-core.microservices.pretrained_models` out-of-process in an isolated `uv` environment, uploads HF-sourced checkpoints to workspace S3, calls `POST /jobs:load_airgapped`, and records `ModelConfig.tao_base_experiment_id`. See §1.6 and §9.7.8.

**TAO workspace state:** `deployment.db.tao_deployment_configs` (§13.19) is the sole source of truth for non-secret TAO workspace identity, bucket, and endpoint URLs. `vlm-feedback-loop tao-bootstrap` writes those values directly there. S3 credentials come from the process environment or canonical `.env`; the DB stores only their env-var-name references.

### 1.7 Workspace and Project Storage  `[EXACT]`

The system stores all persistent data under a **workspace root** directory, configured once at install or first launch (e.g., `~/.vlm_feedback_loop/` or a user-chosen path). The workspace root is a deployment-level setting, not per-project.

Layout:

```text
{workspace_root}/
  projects/
    {project_id}/
      project.db          # SQLite database (WAL mode); all records for this project
      exports/            # dataset export archives
      artifacts/          # large artifacts referenced by _ref fields on records
      logs/               # structured operational logs (JSONL)
        operations/{inference_invocation_id}/
        runs/{run_id}/
```

Each project is a self-contained directory under `{workspace_root}/projects/`. On project creation, the system creates the project directory and initializes its database.

**Artifact storage:** large artifacts (raw model responses, parsed/normalized JSON, validation reports, run logs) MUST be stored as files under `{project_dir}/artifacts/`, not as database blobs. Database records store only metadata and file-path references (`_ref` fields).

**Images are not copied into the project directory.** The system records the original filesystem path as `storage_ref` on the Example record (§13.8). The backend provides a filesystem browse endpoint (§10.2.10). Images MUST remain accessible at their recorded paths for the lifetime of the project. If paths change, a bulk path remapping endpoint is available (§10.2.11). When the backend runs in a container, mount image directories at identical paths (e.g., `-v /data/images:/data/images`) so `storage_ref` values are valid in both contexts. A persisted reference is not a durable grant: every production read re-applies the current `IMAGE_ROOT`, requires a regular file, and consumes the already-authorized descriptor so path replacement cannot redirect the bytes.

Required configuration:

- `WORKSPACE_ROOT`: absolute path to the workspace root directory. No default; MUST be set before first use. The system MUST create the directory structure if it does not exist.

### 1.8 Implementation Stack  `[EXACT]`

**Frontend:** React + TypeScript + Vite. Client-rendered SPA served as static assets. **UI library: NVIDIA KUI Foundations + Tailwind.** KUI provides component primitives and NVIDIA design language; Tailwind provides layout composition. Aligns with Retail Catalog Enrichment, Retail Agentic Commerce, RAG Blueprint.

**Backend:** Python + FastAPI. Pydantic for request/response validation and record schemas. SQLAlchemy for ORM. httpx for outbound async HTTP. Jinja2 for prompt templates (Appendix D). Standard `logging` with JSON formatter writing to stdout and `{project_dir}/logs/*.jsonl`. tiktoken for token counting (`encoding_for_model` when model maps cleanly, `cl100k_base` fallback); `RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN` absorbs tokenizer mismatch.

**Database:** SQLite per project at `{project_dir}/project.db` (§1.7), WAL mode enabled. SQLAlchemy models MUST avoid SQLite-specific query behavior to preserve portability. The public Alembic lineage starts at `v1_0001`, which creates the complete v1 schema; private pre-release revisions and databases are unsupported and require a fresh workspace. On startup, the backend MUST detect pending post-v1 migrations and apply them before proceeding. Schema version is tracked in the database. Before revision discovery, a nonempty schema MUST contain one canonical, nonempty revision row; otherwise the backend creates a validated recovery backup and refuses the database without migration. Only an empty schema or a sole canonical empty `alembic_version` table may initialize as fresh. Public v1+ project databases are upgradeable, not disposable; labeled data, Guidance versions, evaluation history, and training lineage MUST be preserved across schema changes. Before applying an upgrade, the backend MUST copy `project.db` to `project.db.backup.{ISO 8601 timestamp}`. If an upgrade fails, the backend MUST surface the error and the backup file path. On startup, the backend MUST run `PRAGMA quick_check`; on failure, surface a clear error with the database path (no auto-repair).

**SQLite write-discipline (v1):** multiple concurrent writers exist within a single backend process. Guardrails:

- SQLite `busy_timeout` MUST be ≥ 5000 ms.
- All write transactions MUST be short-lived. Background tasks and request handlers MUST NOT hold a write transaction open across outbound HTTP calls, model inference calls, retry/backoff waits, polling waits, sleeps, or any other long-running operation. Pattern: perform the long-running operation, collect results in memory, then open a short write transaction to persist them.
- Implementation MAY optionally use a simple in-process async mutex per project to serialize writes, but this is not required in v1 if write transactions are consistently short-lived.

**Background execution:** in-process asyncio task orchestration for I/O-bound work. Small thread pool for CPU-bound work that MUST NOT block the event loop (pHash, file I/O). No external task queue. Bounded concurrency for evaluations and batch labeling is provider-aware: `EVAL_CONCURRENCY_HOSTED` / `BATCH_LABEL_CONCURRENCY_HOSTED` (default 1) against hosted endpoints, `EVAL_CONCURRENCY_SELF_HOSTED` / `BATCH_LABEL_CONCURRENCY_SELF_HOSTED` (default 8) against self-hosted/local NIMs. Durable run/task records enable recovery on restart (§1.3).

**Package management:** Python with **uv** (`pyproject.toml` + `uv.lock`). Frontend with **pnpm**.

**Local launch:** single command MUST start both frontend and backend for local development. Docker Compose MAY be used for packaged deployment but is not required for development.

**Local NIM prerequisites script (`setup-local.sh`):** optional, one-time run by SME/admin before first launch. NOT auto-triggered from web UI. Installs Docker + NVIDIA Container Toolkit, configures GPU runtime, validates GPU access, optionally pre-pulls NIM images. Referenced by the onboarding flow when prerequisites are missing.

**Development frontend/backend routing:** when frontend (Vite dev server) and backend (FastAPI) run on different origins (e.g., `localhost:5173` / `localhost:8000`), the implementation MUST provide a deployment-scoped development routing strategy. **Preferred:** Vite proxy for `/v1/` and SSE path with SSE passthrough. **Alternative:** FastAPI `CORSMiddleware` with explicit allowlist. Wildcard `*` MUST NOT be used with credentials.

### 1.9 Configuration  `[EXACT]`

Two configuration sources with distinct scopes.

**Environment variables / `.env` file** for secrets and deployment-bound endpoints:

- `NVIDIA_API_KEY`: hosted NVIDIA NIM API key (§1.5 Mode A, §5.5)
- `NGC_API_KEY`: NGC API key for pulling NIM container images (§1.5 Mode C); required for local NIM deployment.
- `TAO_API_KEY`: TAO API credential (§1.6)
- `TAO_API_BASE_URL`: TAO API base URL (§1.6)
- `TAO_ORG_NAME`: TAO organization name (§1.6)

**Canonical `.env` location:** `~/.vlm_feedback_loop/.env`, colocated with `~/.vlm_feedback_loop/config.yaml`. Deployment-scoped, not project-scoped.

The backend MUST NOT search for `.env` in the current working directory, `WORKSPACE_ROOT`, or project directories. An explicit override MAY be provided via CLI `--env-file <absolute-path>` or env var `VLM_FEEDBACK_LOOP_ENV_FILE=<absolute-path>`. When an override is provided, the backend MUST load only that file and MUST NOT fall back to the canonical location. If no `.env` file exists at the selected location, startup continues; missing required secrets are surfaced only when a workflow requiring them is used.

For containerized/CI use, secrets are injected as standard process environment variables; no `.env` file is needed. See §12.1 for security requirements.

**YAML config file** for non-secret application settings at `~/.vlm_feedback_loop/config.yaml`. Settings include `workspace_root`, `log_level`, UI defaults, thresholds, feature flags. API keys MUST NOT be stored here.

**Precedence (highest wins):**

1. Explicit process environment variables
2. Explicit env file path from `--env-file` or `VLM_FEEDBACK_LOOP_ENV_FILE`
3. Default env file at `~/.vlm_feedback_loop/.env`
4. Config file values from `~/.vlm_feedback_loop/config.yaml`
5. Built-in defaults (Appendix A.4)

**First-launch behavior (CLI/bootstrap-first):** if `~/.vlm_feedback_loop/config.yaml` does not exist, the backend MUST fail fast with a clear message directing the user to run a bootstrap command (e.g., `vlm-feedback-loop init`). The bootstrap command prompts for `WORKSPACE_ROOT`, writes a commented `config.yaml` template (`WORKSPACE_ROOT` is the only active key; other defaults appear as commented documentation lines), generates a commented skeleton `.env` at `~/.vlm_feedback_loop/.env`, and exits. The backend then starts normally. The application does **not** include a first-launch web setup wizard.

After backend startup, service connection setup remains in the web app: new
projects use the split NIM setup chain, while the richer NIM Configuration
screen is the post-onboarding edit surface. A self-hosted Teacher becomes
active only after the backend re-verifies the exact catalog model and durably
binds its credential-free endpoint.

---

## 2. Conventions

### 2.1 Reproducibility and Determinism  `[EXACT]`

This system does not assume model inference is deterministic. Instead, it guarantees deterministic scoring, reproducible prompt construction, and recorded lineage for auditability.

**Deterministic (MUST):**

- Schema validation and normalization (Core validity classification)
- Exact Match scoring on Core fields (and all derived metrics)
- JSON schema derivation from SchemaCore
- Pool membership lists (immutable per pool version)
- Dataset/export manifests (stable ordering and contents for a given export definition)

**Prompt reproducibility (MUST):**

- For every model invocation, persist enough lineage to reconstruct the exact prompt context: `model_config_id`, `guidance_id`, decoding params, `icl_example_keys_used[]`, and a `prompt_hash` or rendered prompt reference.

**Selection policies:**

- ICL selection MUST be deterministic for evaluation and batch labeling. Interactive proposals MAY use stochastic selection, but MUST persist `icl_example_keys_used[]`.
- Review selector is always diversity-driven (pHash or CLIP); system MUST record enough state to reproduce the sequence (Section 13.3).

**Seed policy (Generation Controls):**

- Interactive proposals: do NOT set `seed` (allow natural variation).
- Evaluation + Batch Labeling: set a deterministic per-example seed:
  - `seed_effective = derive_seed(scope_id, example_key)`
  - `derive_seed(scope_id, example_key) = abs(int32(first_4_bytes(sha256("<scope_id>:<example_key>"))))`
  - where `scope_id` is `evaluation_run_id` or `batch_label_run_id`.
- Persisted in `sampling_params_effective.seed` on Operation Record (§13.1).

### 2.2 Entity Identifier Format  `[EXACT]`

All backend-generated entity identifiers (`project_id`, `guidance_id`, `model_config_id`, `run_id`, `tao_job_id`, `dataset_export_id`, `pool_id`, `audit_event_id`, `student_model_id`, `local_nim_deployment_id`, `inference_invocation_id`, and similar record IDs) MUST use **UUID4** in canonical lowercase hyphenated string format (e.g., `f47ac10b-58cc-4372-a567-0e02b2c3d479`).

Content-derived identities (e.g., `schema_hash`, `prompt_hash`, artifact checksums) use hashes, not UUIDs. Human-readable display labels (e.g., Guidance `version_number`, model `model_name`) are separate fields, not encoded into the ID.

### 2.3 Key Definitions  `[EXACT]`

**Schema-invalid output** — Response cannot be normalized into schema-valid JSON for all **Core fields** (invalid JSON, missing Core, invalid types/ranges/enums, conflicting duplicates preventing resolution). Aux field errors MUST NOT block proposal validity and MUST NOT change schema-valid vs schema-invalid classification. Implementations MUST produce a structured validation report separating Core vs Aux errors (Sections 6.3.1, 13.1).

**Timeout** — Model invocation exceeded enforced deadline with no usable result.

**State** — Canonical example state: `Unlabeled` / `Auto-Labeled` / `Verified` / `Omitted`.

**Omitted** — Example excluded from workflow: MUST NOT be shown in SME review selector; excluded from Batch Labeling input selection by default. Omission is set by SME Skip.

**Test Pool** — A reserved subset of Verified examples used for evaluation and excluded from ICL. Serves both iteration feedback and Scale-Up gating.

**Pool membership** — Tag indicating a Verified example is assigned to the Test Pool. Managed automatically by the system via pool assignment routing (§4.3).

**Pool assignment routing** — Automatic routing based on the configured target fraction: the next Verified example fills an immediate shortfall, while later rebalancing uses the review selector's CLIP/pHash switchover. Verification outcome (Accept/Edit) does not influence pool assignment (§4.3.1–4.3.2).

**Evaluation snapshot (pool version)** — Frozen, immutable capture of pool membership at the time an evaluation run starts. Ensures reproducible evaluation against a fixed example set (§4.3.3).

**ICL eligibility** — Verified examples with `verified_outcome=Edit` that are not assigned to the Test Pool (see Section 6.2). Accepted examples are never selected for ICL.

**Auto-Labeled label** — Schema-validated model output produced without SME review (typically Batch Labeling). Stored with lineage; MAY be exported for training; is NOT ground truth (§3.2).

**Batch Labeling run** — Long-running job applying a selected Teacher to many Unlabeled examples to produce Auto-Labeled labels + run-level metadata.

**Model endpoint** — Configured OpenAI-compatible NIM base URL plus auth mode (`bearer` for hosted, `none` for self-hosted/local).

**Model name** — Identifier string used in OpenAI-compatible NIM `model` parameter; shown in UI.

**Model config** — Catalog entry binding endpoint + `model_name` + operational metadata (context window tokens required), `eligible_roles[]` declaring which roles the entry may serve (`teacher`, `student_base`), optional deployment metadata (NIM profile and precision), and persisted structured generation capability probe status (Section 6.2). Active selections per role are stored on the Project record (Sections 4.8, 13.4).

**NIM model profile** — Deployment-time selection that determines which engine NIM uses and how it is optimized. A profile implies: backend (TensorRT-LLM or vLLM), precision, optimization target (latency vs throughput), and tensor parallelism. For Cosmos Reason2, documented packaged profile precisions are BF16 and FP8 (vLLM backend). Profiles can be pinned with `NIM_MODEL_PROFILE`; otherwise NIM automatically selects and logs at startup.

**Model precision / quantization (NIM)** — The effective precision of a deployed Student. For packaged NIM profiles, precision is a property of the chosen profile (BF16, FP8). For TAO-quantized checkpoints, precision is a property of the quantization method used (FP8_DYNAMIC, W4A16), served through `NIM_MODEL_NAME` on the vLLM backend.

**Profile metadata recording (required for deployed Students)** — Any deployed Student endpoint MUST persist: `nim_model_profile_requested` (nullable), `nim_model_profile_selected` (required; observed from deployment/startup logs), `nim_profile_metadata` (at minimum: backend/engine, precision, TP, optimization target), GPU type/count, and `quantization_method` (e.g., `FP8_DYNAMIC`, `W4A16`, or `none` for full-precision). Attached to any latency/throughput benchmark outputs.

**Embedding provider** — Configured source for per-image CLIP-style embedding computation. Values: `self_hosted_nvclip` (the default whenever a healthy local embedding NIM deployment exists), `hosted_nvclip` (fallback via the hosted NIM API key), `none` (last resort — pHash diversity). Names retained for backwards compatibility; the actual model is governed by `embedding_model_id`. Resolution cascade in §5.5.1; consistency enforced per project (§5.5.3).

**Embedding NIM endpoint** — NVIDIA NIM embedding endpoint exposing OpenAI-compatible `/v1/embeddings` with image input support. Hosted: `https://integrate.api.nvidia.com/v1/embeddings`. **Supported model: `nvidia/llama-nemotron-embed-vl-1b-v2`** (2048-dim; requires `input_type` in the request body, default `"passage"`). Image input: base64 data URL.

**CLIP embedding** — Fixed-length vector representation of an image used for review selector scheduling and optional filtering. "CLIP" names the algorithm class; the default provider is the hosted NeMo Retriever VL embedding NIM.

**pHash (perceptual hash)** — A compact 64-bit perceptual fingerprint computed by a restartable CPU-only background sweep after ingest (no external dependency). Similarity measured by hamming distance. Baseline diversity signal for the review selector. See §5.6.

**pHash-diverse review selector** — Baseline selection strategy using available pHash values and hamming distance. When all candidate hashes are still pending or failed, it falls back to deterministic `example_key` order. Deterministic given persisted scheduler state.

**CLIP-diverse review selector** — Selection strategy using CLIP embedding cosine similarity. Preferred over pHash-diverse when CLIP embeddings are available. Deterministic given persisted scheduler state.

**Visual budget mode** — Per-model declaration of which visual token control shape the model accepts. Values: `none`, `mm_processor_size`, `mm_processor_pixels`, `mm_processor_tiles`. Stored on ModelConfig. See §6.9.

**Visual budget preset** — Named preset (**Fast** / **Balanced** / **High Detail**) resolving to model-specific `mm_processor_kwargs` values based on `visual_budget_mode`. See §6.9.

**Retry (interactive)** — User-initiated re-run of proposal generation for the same image after selecting a different Teacher model config and/or Guidance.

**Rationale note** — An opt-in reserved `role="aux"` string field (`rationale_note`) controlled in Guidance and disabled by default. When enabled it is requested last, excluded from Teacher ICL, and never evaluated; when disabled it is bypassed end to end.

**Export field mode** — Per-export setting controlling which field groups are included in the gpt turn of training dataset exports and Auto-Labeled output: `all` (rationale + Aux + Core; default), `aux_and_core` (Aux + Core, no rationale), or `core_only` (Core only). See §9.3.

### 2.4 Timestamps  `[EXACT]`

All persisted timestamps MUST be stored in UTC using ISO 8601 with a `Z` suffix (e.g., `2026-03-30T14:22:07Z`). The frontend converts timestamps to local time for display.

---

## 3. System Invariants  `[EXACT]`

### 3.1 Data Model Invariants

- Exactly four example states: `Unlabeled`, `Auto-Labeled`, `Verified`, `Omitted`.
- The Test Pool is a pool membership automatically managed by the system (§4.3). Pool members are reserved for evaluation and excluded from ICL. Both Accepts and Edits are candidates for the Test Pool. Pool assignments are durable once made (exception: a semantic Core change deletes Label records, which clears pool assignments; §4.4.1).
- Verified is the ground-truth source of record for evaluation, ICL selection, and default training split definition (excluding pools).
- Auto-Labeled labels change example state. Batch Labeling MAY store Auto-Labeled labels for Unlabeled examples, transitioning those examples to `Auto-Labeled` until SME review yields Verified. An `auto_labeled` Label is live only while its exact owning Example remains `state="Auto-Labeled"`; Skip discards that machine proposal (§4.5).
- SchemaCore stability invariant (mandatory):
  - Each field has an immutable `field_id` that persists across renames.
  - In-place Core edits are limited to: field rename, exact 1:1 enum value rename, and presentation metadata changes.
  - All other Core changes (add/remove field, type change, constraint change, allowed-value change, role change) are semantic and MUST trigger the schema evolution flow (§4.4.1).
  - SchemaCore may also be extended by adding new fields with `role="aux"` in place (no label invalidation required).
- Guidance versioning preserves SchemaCore identity within a project: Description/Rules may change freely; semantic Core changes trigger label invalidation and return examples to `Unlabeled` (§4.4.1).
- Schema evolution invariant: when a semantic Core change occurs, all Verified labels and Auto-Labeled labels are deleted. Verified examples transition to `Unlabeled` with prior label data preserved as reference on the Example record (`prior_verified_label_ref`). Auto-Labeled examples transition to `Unlabeled` (their Operation Records preserved for audit).
- Cold start invariant: system MUST operate with `Verified = 0` and MUST degrade to zero ICL examples until Verified exists.
- Catalog-driven model identity: every inference/evaluation/batch labeling operation MUST reference a specific `model_config_id`.
- Omission invariant: `state=Omitted` examples MUST NOT be presented in the labeling review selector and are excluded from Batch Labeling input selection by default. Omission is set by SME Skip.
- Project partitioning invariant: all record families MUST include `project_id`, and every query/update MUST be scoped to exactly one `project_id`.

### 3.2 Review & Evaluation Invariants

1. Interactive UI actions per image are exactly those defined in §4.5.
2. Review outcomes that create Verified are exactly: `Accept` and `Edit`. `Skip` and `Retry` do not create Verified.
3. Exact Match (Core Fields) is deterministic and schema-driven (Appendix A.2).
4. Auto-Labeled labels MUST NOT be used as evaluation ground truth; evaluation compares to Verified only.
5. Skip MUST persist `state=Omitted` per §4.5.

### 3.3 Selection & Prompt Invariants

1. ICL eligibility: only Edits not assigned to the Test Pool. Accepted examples are never ICL-eligible.
2. Prompt lineage is recorded: every inference/evaluation/batch labeling call persists ordered `icl_example_keys_used[]`.
3. Marker rule: model-visible prompt MUST NOT include `example_key`; examples use markers `E01..`.
4. Token budget enforcement MUST degrade gracefully by dropping the least query-relevant ICL examples (relevance-tail pruning, §6.2) rather than blocking progress.
5. Per-model prompt budgets: prompt packing MUST respect selected model context window and output cap.
6. Generation Controls persistence: sampling parameters (`temperature`, `top_p`, `seed`) and thinking toggle state MUST be persisted per invocation in Operation Records (§13.1).
7. Visual Budget persistence: `visual_budget_preset_key` and `visual_budget_params_effective` MUST be persisted per invocation. When the model does not support visual budget controls, both are null.

### 3.4 Failure & Integrity Invariants

1. Failure semantics distinguish schema-invalid vs timeout vs endpoint/transport error.
2. Student Training execution is validated server-side independently of the Scale-Up Teacher-quality criteria: fine-tuning cannot start until TAO/workspace checks pass, at least one active-Guidance non-pool Verified training example exists, the active-Guidance Test Pool reaches the project-configured minimum, and each selected base model has the `student_base` role. A selected missing TAO base is non-blocking because first-use provisioning is the first conditional Training Jobs stage.
3. Review selector selections MUST be reproducible (Section 13.3).

### 3.5 Operation Record Invariants

1. Invocation record persistence: system MUST generate `inference_invocation_id` before invoking model and MUST persist an operation record for all outcomes.
2. Evaluation uses the same prompt pipeline as Interactive Labeling.
3. User-requested interactive Retry and evaluation's explicit sequential retry pass produce distinct invocation records linked to the prior attempt. Resuming one unfinished Batch Labeling item reuses that item's pending invocation record.
4. (removed)
5. TAO job observability is persisted: TAO jobs MUST have durable records, deterministic state mapping, and durable output references.

---

## 4. Data Model

### 4.1 Example States  `[EXACT]`

**Unlabeled** — Image exists without human-confirmed label. Eligible for interactive labeling review selector. Not used for ICL and not eligible for pools.

**Auto-Labeled** — Image+label produced without SME review, with one live `label_status=auto_labeled` Label for the exact Example. Used for training scale. MUST NOT be used for ICL and not eligible for pools.

**Verified (Ground Truth)** — Image+label confirmed by human via Accept/Edit. Used for ICL selection (unless pooled), evaluation reference labels, and default training split (excluding pools).

**Omitted** — Image excluded from workflow (Skip), with no live machine Label; batch invocation history remains auditable. MUST NOT be presented in interactive labeling review selector. Excluded from Batch Labeling input selection by default.

### 4.2 Omission Metadata  `[SUMMARY]`

Example record minimum fields: `project_id`, `example_key` (unique within project), `storage_ref` (absolute filesystem path), `ingested_at`, `source_metadata` (JSON, may be `{}`), `state`. Omission provenance (required when `state=Omitted`, null otherwise): `omitted_source` ∈ {`sme_skip`, null}, `omitted_at`. pHash field required: `phash` (nullable while the post-ingest sweep is pending or after a per-image failure; otherwise hex-encoded, §5.6). Embedding fields: `clip_embedding_present` (default false), `clip_embedding_dim`, `clip_embedding_model_id`, `embedding_provider`. Prior-label reference fields (set during semantic Core schema change, null otherwise): `prior_verified_label_ref` (JSON snapshot of prior label data), `prior_verified_outcome` (`Accept` / `Edit` / null). System MUST NOT assume embeddings exist for all examples. Omission is set by SME Skip; `omitted_source` and `omitted_at` MUST be recorded automatically. See §4.2 for full field list.

### 4.3 Test Pool  `[SUMMARY]`

The Test Pool is a reserved subset of Verified used to measure quality without contaminating ICL context. **Pool management is fully automatic** — no user action required.

**Routing (§4.3.1):** when an example becomes Verified:
1. If Test Pool is below its target count → assign to Test Pool.
2. Else → non-pool (available for ICL if eligible per §6.2).

Verification outcome (Accept/Edit) does not influence pool assignment. Target count:

```
test_pool_target = floor(total_verified × TEST_POOL_FRACTION)    # default 0.40
```

When CLIP embeddings are available and multiple candidates could fill the next slot, select the most dissimilar from current pool members. When embeddings are unavailable, assign the next Verified example directly. Resulting `pool_assignment` is persisted on the Label record (§13.7).

**Rebalancing (§4.3.2):** after each new Verified label, promote non-pool Verified examples if the pool is below target. Same diversity-mode switchover as the review selector (Appendix A.3, A.5): CLIP when ≥ `CLIP_SWITCHOVER_MIN_COUNT` eligible embeddings, else pHash. No mixed-mode scoring in one pass. Pool assignments are durable: once assigned, never demoted. **Exception:** a semantic Core change deletes Label records (and thus pool assignments). The pool rebuilds as examples are re-labeled.

**Order-sensitivity (§4.3.2 diagnostic guard):** the Test/non-pool split is labeling-order sensitive — a class-clustered order (manifest-order autorun, class-sorted import) can make the two sets hold disjoint classes, which structurally floors relevance-ICL metrics for any Test class absent from the non-pool Edits. The CLIP-diverse review selector (default) keeps the split class-representative; pHash diversity does not. Routing is unchanged; a diagnostic WARNING fires at evaluation finalization (ICL runs only) when a degenerate split is detected.

**Snapshots (§4.3.3):** when an evaluation run starts, the system auto-creates a frozen **pool version snapshot**. Snapshots are immutable; `pool_version` monotonically increasing within the project. Evaluation runs reference the snapshot, not the live pool. If the Test Pool has no members, evaluation MUST NOT proceed.

### 4.4 Guidance and Schema  `[EXACT]`

Guidance records are immutable versions. Editing Guidance creates a new Guidance record with a new `guidance_id`. Projects select active version via `project.active_guidance_id`.

Minimum required fields on Guidance:

- `project_id`, `guidance_id` (unique within project), `version_number: int` (1-based, monotonically increasing, backend-assigned, immutable; user-visible as `v{version_number}`), `description`, `schema`, `rules`, `created_at`.

**SchemaCore edit policy (mandatory):**

**In-place edits (no label invalidation):**

- **Add Aux field:** add new fields with `role="aux"`.
- **Rename Core field:** change `field_name` while preserving `field_id`. Deterministic rename propagation MUST update all Verified labels, ICL references, and export templates within the project.
- **Rename enum value (exact 1:1):** rename an allowed value in an enum or enum_set field where the mapping is exactly one old value → one new value with no change in meaning. Deterministic value rename propagation MUST update all Verified labels.
- **Edit presentation metadata:** changes to `display_order`, display labels, helper text, tooltips, or other non-semantic presentation properties.

**Semantic Core edits (trigger label invalidation; see §4.4.1):**

- Add Core field
- Remove Core field
- Change Core field type
- Change any Core constraint (integer min/max, string minLength/maxLength)
- Change allowed enum/enum_set values (add value, remove value, merge, split, or redefine meaning)
- Change Core ↔ Aux role (promote Aux to Core, or demote Core to Aux)

The decision rule: **if the edit changes what a correct label means, it MUST trigger the schema evolution flow (§4.4.1).** Practically: would a person need to reconsider the image to know the correct answer under the new Core? If yes, labels are invalidated.

**Reserved Aux field `rationale_note`:**

- Presence in the active Guidance schema is the feature flag: optional reserved `role="aux"`, type `string`; absent by default. Enabling or disabling is an in-place Aux edit and never invalidates labels.
- When enabled, the production Teacher requests it last. When disabled, prompts, schemas, proposals, review, save gating, and newly Verified data omit it. It never affects evaluation or Core validity.
- **Base serialization order is backend-canonical:**
  ```
  generation_order = [
    ...["rationale_note"] if enabled,
    ...remaining_aux_fields_sorted_by_display_order,
    ...core_fields_sorted_by_display_order
  ]
  ```
  Stable base order for persisted schema metadata and backend transforms. Inference Contracts may filter it, and the production Teacher deterministically moves rationale to the end of prompt-visible, guided-decoding, and Student conversation output.
- When present, `rationale_note` has the lowest `display_order`; Aux-before-Core group ordering is system-enforced.
- **Correctness MUST NOT depend on downstream preservation of JSON object member order.** Property order is a compatibility hint and deterministic serialization aid.
- When enabled, Labeling orders Core first and Aux second, with rationale hidden until Edit; every newly Verified example has a reviewed note. On Accept it is retained silently. On Edit the SME writes it or explicitly approves regeneration.
- When disabled, Labeling has no rationale display/panel or rationale save gate; `label_json` omits the key and rationale provenance is null.
- Toggling does not backfill or erase historical Label JSON, so re-enabling may yield a tolerated mix of examples with and without notes. Active contracts filter through current Guidance, and Teacher ICL always excludes rationale.
- **`rationale_source`** ∈ {`teacher_proposal`, `sme_edited`, `teacher_regenerated_approved`} only when enabled.
- Rationale is excluded from production Teacher ICL (§6.2) and included in training export only when selected by export field mode (§9.3).

**SchemaCore type system (normative):**

Five user-facing field types, each mapping deterministically to a JSON Schema representation used by structured generation (§6.2) and Exact Match normalization (Appendix A.2):

- **Enum** (single choice): `string` with `enum`. `allowed_values[]` required (≥2 values, no empty strings, unique after trim).
- **Enum Set** (multi-select): `array` of `string` with `uniqueItems: true`, items constrained by `enum`. Same `allowed_values[]` constraints.
- **Boolean**: `boolean`. No additional constraints.
- **Integer**: `integer` with optional `minimum`/`maximum`. If both set: `minimum ≤ maximum`.
- **String**: `string` with optional `minLength`/`maxLength`. If both set: `minLength ≤ maxLength`.

No other field types are permitted in SchemaCore.

**SchemaCore field record (minimum per-field):**

- `field_id: string` (required; immutable; system-generated; persists across renames; unique across Core + Aux)
- `field_name: string` (required; editable; must match `^[a-zA-Z_][a-zA-Z0-9_]*$`; max 64 characters; unique across Core + Aux; `rationale_note` is reserved)
- `type: "enum" | "enum_set" | "boolean" | "integer" | "string"` (required)
- `role: "core" | "aux"` (required)
- `allowed_values: string[]` (required when type is `enum` or `enum_set`)
- `minimum: integer | null`, `maximum: integer | null` (applicable when type is `integer`)
- `min_length: integer | null`, `max_length: integer | null` (applicable when type is `string`)
- `display_order: integer` (required; controls within-group ordering)

`field_name` values are label-schema data keys, not runtime identifiers. Names like `type`, `id`, or `class` are valid because they appear only as JSON property keys and database column values, never as code identifiers.

#### 4.4.1 Schema Evolution: Semantic Core Changes  `[EXACT]`

When a user requests a Core edit classified as semantic, the system invalidates existing labels and returns affected examples to `Unlabeled`. The standard Interactive Labeling loop handles these examples with prior-label hints for efficiency. No separate re-verification state or workflow.

**Semantic Core change flow:**

1. User edits Core in the Guidance editor.
2. Backend classifies the edit against the in-place vs. semantic boundary.
3. If the edit is semantic, the system presents a confirmation dialog (§6.6.8).
4. On confirmation, the backend applies the following changes atomically:
   a. Creates a new Guidance version with the Core change applied, sets it as `active_guidance_id`, and records `semantic_core_change_from_guidance_id` on the new Guidance record pointing to the prior version.
   b. All examples with `state="Verified"`: prior Label data (label JSON, verified outcome, guidance_id) is copied to `prior_verified_label_ref` and `prior_verified_outcome` on the Example record; Label record is deleted; Example transitions to `state="Unlabeled"`. Omitted examples remain Omitted.
   c. All examples with `state="Auto-Labeled"`: Label records deleted; Example transitions to `state="Unlabeled"`. Operation Records preserved for audit. System surfaces: *"{M} Auto-Labeled examples reverted to Unlabeled. Re-run Batch Labeling when ready."*
   d. Test Pool assignments cleared. Pool rebuilds automatically.
   e. Evaluation-trigger baselines reset: `icl_recommendation_dismissed_at_count=0`, `scaleup_accept_rate_window` resets, any in-progress evaluation canceled. Auto-Evaluate trigger counters rebuild from zero under the new Guidance.
   f. Review selector scheduler state reinitialized: recent window cleared, persisted selector-history reset (§13.3).
   g. If the SME was reviewing an example, `schema_change_context_example_key` is recorded on the Project record so the selector starts from that example (§6.5).
   h. Schema refinement reminder counters reset to `0` (§6.8).

**How re-labeling works (no special mode):** after the schema change, the system is in a state equivalent to cold start. The standard Interactive Labeling loop runs normally with two enhancements:

- **Review selector priority:** the selector presents Unlabeled examples with `prior_verified_label_ref` before standard Unlabeled. Among those: `schema_change_context_example_key` first (if set), then prior Edits (strongest corrective signal for ICL seeding), then prior Accepts. After all prior-label examples re-labeled, selector returns to standard diversity-driven ordering.
- **Prior-label hints on labeling screen:** when an example has `prior_verified_label_ref`, the UI shows the prior label as annotated reference alongside the VLM proposal (read-only):
  - Fields the SME previously edited are badged: *"You changed this from {original_VLM_proposal} to {your_correction}"*
  - Prior values schema-invalid under the new schema are highlighted.
  - VLM agree/disagree with prior is indicated.
  - **"Adopt prior" action per field:** one-click replace of VLM value with prior value. Counts as an Edit; rationale review is required only when enabled (§6.3.2).
- **Save / Skip / Retry**: same three actions as standard review. Any modification requires rationale review before Save only when rationale notes are enabled.

ICL rebuilds progressively from zero as the SME labels via Edit. Evaluation, Test Pool, and Auto-Evaluate follow standard rules. The SME can freely interleave re-labeling prior examples and labeling newly ingested images.

### 4.5 Interactive Actions and Review Outcomes  `[EXACT]`

The proposal is displayed in an editable form. The UI presents three actions:

- **Save** — store the current label as ground truth. If the SME modified any fields, `verified_outcome=Edit`; if unmodified, `verified_outcome=Accept`. When rationale notes are enabled and fields are modified, the rationale must be addressed before Save (§6.3.2); when disabled, there is no rationale step. Example state after: `Verified`.
- **Skip** — omit this image from workflow; no label recorded; not shown again. Example state after: `Omitted`.
- **Retry** — re-run proposal for same image after changing Teacher model and/or Guidance. Example state after: remains `Unlabeled` or `Auto-Labeled` until Save.

Additionally, a **Reset** action is available when the SME has modified any field. Reset restores all fields to the VLM's original proposal values, collapses the rationale panel, and returns to the unmodified state.

**Skip semantics:**

- Skip MUST set `state="Omitted"` on Example.
- Skip MUST set `omitted_source = "sme_skip"`, `omitted_at = now()`.
- If the Example is `Auto-Labeled`, Skip MUST delete its `label_status=auto_labeled` Label in the same transaction. The batch-label Operation Record remains for audit.
- Skip does not require a reason.
- Omitted examples MUST NOT be presented again by the review selector.
- **Restore Omitted:** the SME MAY bulk-restore all Omitted examples to `state="Unlabeled"` (clearing `omitted_source` and `omitted_at`). Available from the labeling screen when the queue is empty and Omitted examples exist. Restore does not resurrect a discarded Auto-Labeled proposal.

**Retry specifics:**

- Retry MUST trigger a new Teacher invocation for the same `example_key`.
- Retry controls MUST be pre-populated with the current project settings so the SME changes only what they want.
- Retry MUST allow SME to change:
  - `teacher_model_config_id` (from entries with `teacher` role) and/or
  - `guidance_id` (existing Guidance version compatible with SchemaCore policy)
- After completion, UI MUST present the same three actions again.
- System MUST persist an invocation record for each proposal attempt, including Retry attempts.

### 4.6 Verification Metadata (Accept/Edit Only)  `[SUMMARY]`

Persist with the Label record (§13.7), at minimum: `project_id`, `example_key`, `verified_outcome` (Accept/Edit), `inference_invocation_id` (invocation whose proposal was accepted/edited), `verified_at`. Skip does not create Verified label, but MUST persist omission fields on Example (§4.2).

### 4.7 Label Storage (Unified)  `[EXACT]`

All labels — whether produced by Batch Labeling or confirmed by SME review — are stored as **Label** records (§13.7) with a `label_status` discriminator (`verified` or `auto_labeled`). Single record family, single table, single code path. The status discriminator governs which fields are populated and which downstream behaviors apply.

**Auto-Labeled labels** are created by Batch Labeling and stored with `label_status=auto_labeled`. Invocation lineage (model, guidance, ICL, validation) is stored on the Operation Record with `purpose=batch_label`; the Label record stores the label JSON and links to that Operation Record.

**Verified labels** are created by SME Accept/Edit and stored with `label_status=verified`. Verification-specific fields (`verified_outcome`, `edited_core_fields[]`, `rationale_source`, `pool_assignment`) are populated on verification.

**Promotion:** when the SME reviews an Auto-Labeled example and clicks Save, `label_status` transitions from `auto_labeled` to `verified`, verification fields are populated, and `inference_invocation_id` is updated per the save provenance rule below. The prior Auto-Labeled label JSON is preserved on the Operation Record for audit. `batch_label_run_id` is retained for upstream provenance; `inference_invocation_id` is the reviewed-proposal pointer (the exact proposal the SME saved). These two fields serve different purposes and do not conflict.

**Rejection by Skip:** when the SME skips an Auto-Labeled example, its machine Label is deleted and the Example becomes `Omitted` atomically. Its Operation Record remains as audit history. Restore returns a clean `Unlabeled` Example.

**Label save provenance after Retry:** when a Label is created or promoted to `verified`, `inference_invocation_id` MUST reference the exact proposal invocation whose contents the SME saved. If the SME saved the surfaced Auto-Labeled proposal without Retry, `inference_invocation_id` remains the original batch-label invocation. If the SME performed one or more Retry actions, `inference_invocation_id` MUST reference the most recent Retry proposal displayed at the time of Save.

Auto-Labeled labels MUST NOT be used as evaluation ground truth, ICL context, or pool candidates. Only `label_status=verified` labels serve those purposes.

### 4.8 Model Catalog and Project Selections  `[EXACT]`

Each model config entry binds an endpoint + `model_name` + operational metadata (`context_window_tokens` required; optional `model_quantization` and `nim_model_profile` metadata) and declares **`eligible_roles[]`**: `teacher`, `student_base`.

**`context_window_tokens` semantics:** exact integer value used by the prompt-budget logic (§6.2). Seeded catalog entries preserve the vendor's documented value rather than normalizing shorthand (e.g., "128K"). Some vendors document input-only budgets, others document combined input+output. The §6.2 formula subtracts `max_output_tokens` and applies `safety_margin` to derive `effective_max_input_tokens`.

- MUST be project-scoped via `project_id`.

**Rules:**

- `model_config_id` MUST be backend-generated and unique within the project.
- Fresh-project seeds MUST have published model terms permitting commercial
  use at catalog-review time. Non-commercial-only or unknown-term models are
  excluded from the curated seed; operators may add them explicitly and
  historical records remain inspectable.
- `eligible_roles[]` MUST be non-empty. Role filtering is the sole mechanism:
  - Teacher selection: entries where `teacher ∈ eligible_roles`. A model assigned as `teacher_model_config_id` MUST have `supports_image_input=true`. Enforced at every write site (seeding, `POST /v1/projects`, `PATCH /v1/projects/{id}` — the top-bar Teacher picker, `TeacherModelPicker.tsx`, writes to this endpoint).
  - Student Training base: entries where `student_base ∈ eligible_roles`. Currently limited to the seeded Cosmos bases (Cosmos Reason2 8B/2B, Cosmos 3 Nano/Super reasoner).
- Model configs persist structured generation support status: `unknown` / `supported` / `unsupported` (via probe §6.2).
- Model configs include thinking toggle metadata (§6.7):
  - `thinking_toggle.mode`: `"none"` | `"qwen_enable_thinking"` | `"kimi_thinking"`
  - `thinking_toggle_support`: `"unknown"` | `"supported"` | `"unsupported"`
  - Seeding: Qwen → `mode="qwen_enable_thinking"`; Mistral VLMs → `mode="none"`.
- Model configs include image input capability:
  - `supports_image_input: boolean` (distinct from `visual_budget_mode`).
  - Seeding: every current catalog entry accepts images; this includes the Cosmos families, Nemotron Nano VL/Omni, Step 3.7 Flash, and Mistral Medium 3.5.
- Model configs include visual budget metadata (§6.9):
  - `visual_budget_mode`: `"none"` | `"mm_processor_size"` | `"mm_processor_pixels"` | `"mm_processor_tiles"`
  - `visual_budget_support`: `"unknown"` | `"supported"` | `"unsupported"`
  - Seeding: Cosmos Reason2 → `mm_processor_size`; Nemotron Nano VL → `mm_processor_tiles`; Mistral / Qwen / Nemotron 3 Nano Omni → `none`.
- Model configs with `student_base` role include TAO base-experiment metadata (§9.7.8):
  - `tao_base_experiment_id: string | null` (UUID in the bootstrapped TAO workspace; null until first-use automatic provisioning, the equivalent CLI, or admin handoff).
  - `tao_base_experiment_pull_status: "unknown" | "starting" | "in_progress" | "pulling" | "pull_complete" | "invalid_pull" | "failed" | null`.
  - Suite-launch validation marks a valid missing selected base as `provisioning_required`; job-chain creation remains protected until its id is non-null and status is `pull_complete`.
  - Non-`student_base` entries MUST have `tao_base_experiment_id: null`.

**Each project MUST store active selections:**

- `teacher_model_config_id` (entry with `teacher` role)
- `active_guidance_id`
- `active_student_model_config_id` (nullable; entry with `student_base` role)

**Seeded catalog entries:**

- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (128000) | teacher | vision: yes | media: `none` | local: `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant`, ≥80 GB + compute capability ≥9.0 | `qwen_enable_thinking` | ***recommended supported high-GPU local Teacher; Teacher-only***
- `nvidia/cosmos3-nano-reasoner` (131072) | teacher, student_base | vision: yes | media: `mm_processor_size` | local: `nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0` (`NIM_MODEL_SIZE=nano`, pinned `NIM_MODEL_PROFILE`), ≥56 GB GPU · *recommended local Teacher when Omni is ineligible*
- `nvidia/cosmos3-super-reasoner` (131072) | teacher, student_base | vision: yes | media: `mm_processor_size` | local: `nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0` (`NIM_MODEL_SIZE=super`), ≥88 GB GPU · *selectable, not auto-selected by memory size*
- `nvidia/cosmos-reason2-8b` (256000) | teacher, student_base | vision: yes | media: `mm_processor_size` | local: `nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0`, ≥56 GB GPU (fully selectable; no longer the auto-recommended big-GPU default — superseded by CR3-Nano)
- `nvidia/cosmos-reason2-2b` (256000) | teacher, student_base | vision: yes | media: `mm_processor_size` | local: `nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0`, ≥36 GB GPU · *recommended local Teacher default on 36–55 GB GPUs*
- `nvidia/nemotron-nano-12b-v2-vl` (128000) | teacher | vision: yes | media: `mm_processor_tiles`
- `stepfun-ai/step-3.7-flash` (262144) | teacher | vision: yes | media: `none` | ***default hosted Teacher*** *(selected 2026-08-06 as the strongest commercially permitted reachable hosted Teacher in the retained campaign; `always_on_reasoning`; image cap 8)*
- `mistralai/mistral-medium-3.5-128b` (262144) | teacher | vision: yes | media: `none` | *near-ceiling Mistral-family alternate (image cap 10)*

Entries without `local:` metadata do not support system-managed local deployment in v1 (remote endpoint only). Cosmos Reason2 8B stays seeded because it is a `student_base`. Omni uses its specialized local NIM and supported Qwen-style thinking switch. The completed long-horizon matrix makes Omni the supported high-GPU Teacher and CR3-Nano the robust fallback; Omni remains Teacher-only while Cosmos supplies every Student base. Qwen 3.5 is absent because its hosted API retired. Mistral Large 3 is also absent from new-project seeds after its free endpoint was deprecated and began returning HTTP 410. MiniMax M3 is absent from new-project seeds because its published terms restrict it to non-commercial use. Schema revision `v1_0004` selects Step for a pre-change MiniMax project only when Step is present and the project has no Label rows; projects with labels are untouched, and historical/operator-created configurations remain inspectable.

Implementations MUST seed these entries and set `teacher_model_config_id` to the effective `DEFAULT_TEACHER_MODEL` — `stepfun-ai/step-3.7-flash` by default. An override outside the curated commercial seed fails project creation with an actionable error. The value is config-overridable and exposed to the UI as `EnvironmentResponse.default_teacher_model_name` (no frontend hardcode). Default Teacher selection prioritises commercially permitted model terms, certified multi-domain accuracy, ICL-over-time depth capacity, live-measured interactive latency, and live-probed reliability (image caps, fail rates).

Backend MUST validate that any active selection references an entry with the corresponding role and that the model is reachable and compatible.

**Capability re-probe:** the three capability fields (`structured_generation_support`, `thinking_toggle_support`, `visual_budget_support`) are checked once and persisted. Re-probe action (§10.2.12) resets all three to `unknown` and re-runs probes. It remains blocked while the model config is referenced by a `queued`, `running`, or `canceling` evaluation/Batch run, or an active training job; a paused Batch run does not block it. Evaluation and Batch runs own credential-free runtime snapshots, so later catalog or endpoint changes affect future runs only, including after restart or explicit Resume.

Retry MAY override Teacher and/or Guidance per-attempt.

---

## 5. System Roles

### 5.1 Teacher (Served via NVIDIA NIM)  `[EXACT]`

- Generates proposals during Interactive Labeling.
- Batch Labeling invokes Teacher to generate Auto-Labeled fully synthetic labels from Unlabeled images.
- MUST be selected from catalog entries with the `teacher` role.
- MUST be reachable via OpenAI-compatible `/v1/chat/completions`.

### 5.2 Guidance-Author (removed)  `[EXACT]`

> **Removed before v1.0:** the Guidance-Author role and AI Guidance Rewrite feature (§6.4) are absent from the public schema.

### 5.3 Student (Fine-tuned via Cosmos-RL / TAO VLM; Served via NIM)  `[EXACT]`

- Produced via Student Training (optional), possibly multiple variants.
- Evaluated using same Exact Match rules (Appendix A.2) against the Test Pool.
- Used for deployment/serving decisions and model comparisons.
- Batch Labeling in this spec is not a Student operation.

### 5.4 Pre-ingest Curation (Future)  `[REFERENCE]`

Not in v1. See §5.4 (NeMo Curator integration possible in a future version).

### 5.5 Embedding Computation and Indexing  `[SUMMARY]`

**Provider resolution is a fallback cascade (`EMBEDDING_PROVIDER=auto`):** (1) a healthy local embedding NIM deployment (recorded on `EmbeddingDeploymentConfig`, live-verified at probe time; the local NIM is unauthenticated — a keyless GPU-only host gets semantic embeddings) is the default whenever present; (2) the hosted NeMo Retriever VL embedding NIM at the same base URL as Teacher (`/v1/embeddings`), using the same hosted NIM API key; (3) `none` (selector falls back to pHash). Model `nvidia/llama-nemotron-embed-vl-1b-v2`, 2048-dim, `input_type: "passage"`. Probed at project creation or first ingest, and re-resolved for every non-archived project when the local NIM turns healthy. **Local embedding NIM deployment (§1.5 Mode C):** pinned image `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0`; 1B params, 2048-dim, and a 24 GB eligibility floor because the smallest GPU SKUs in the 2.0.0 support matrix are the 24 GB L4 and A10G. The launch pins `NIM_PRECISION=fp16` to avoid the broken unset-precision SM120 cuDNN-plan path and persists `NIM_MODEL_PATH` below `/opt/nim/.cache`; live RTX PRO 6000 Blackwell validation served a finite 2048-dim vector at 6.3 GiB. Automatic setup chooses the lowest-index compatible free GPU whose detected name exactly matches that pinned matrix and whose memory meets the seeded floor — including supported hosts whose every GPU is below the Teacher floors (local embeddings + hosted Teacher). Unrecognized hardware is never recommended from memory alone; an operator may deliberately test NVIDIA's unverified fallback through NIM Configuration. On successful local deploy, `embedding_provider=self_hosted_nvclip` and endpoint URL configured automatically; on stop, displacement, or startup failure the config resets to `provider=none`.

**Background computation (non-blocking):** runs after image ingestion; MUST NOT block labeling start or first proposal. **State-independent:** embeddings MUST be computed for all ingested examples regardless of state (embeddings are a property of the image, not the example state). **Computation order:** pHash-diverse (same order the review selector would present under pHash mode), so by switchover threshold the first `CLIP_SWITCHOVER_MIN_COUNT` embeddings correspond to images the SME is most likely to see first. Incremental ingestion appends new images to the queue. Resumable on restart.

**CLIP switchover:** selector upgrades from pHash-diverse to CLIP-diverse when eligible examples with CLIP embeddings ≥ `CLIP_SWITCHOVER_MIN_COUNT` (default 50).

**Consistency (§5.5.3):** all embeddings within a project MUST use the same model + dimension. Invalidated and recomputed when effective model identity changes. Endpoint location alone (hosted → self-hosted, same model/dim) does NOT require recomputation.

**Storage (§5.5.5):** embeddings stored in a dedicated `ClipEmbedding` table (§13.16), NOT on the Example record. Each embedding row keyed one-to-one to Example with a float32 vector payload (~4 KB at 1024-dim). Example record retains only lightweight presence/summary fields. In-memory cache loaded with a single bulk SELECT on project open; incremental updates as new embeddings complete. Cache rebuilds from table on restart.

See §5.5 for full contract.

### 5.6 pHash Computation (Background Sweep after Ingest)  `[SUMMARY]`

- **Computation (F46 amendment, 2026-05-14):** moved from inline-at-ingest to a background sweeper modeled on `clip_embedding_service.py`. The `:ingest` endpoint creates skeleton `Example` rows with `phash=null` and returns **202 Accepted** in ~1s; `services/ingest_sweeper_service.py` sweeps `Example WHERE phash IS NULL` in multi-pass batches via `compute_phash_from_path`. Restart-recoverable via `recover_ingest_tasks` from the lifespan startup hook. SSE: `ingest_progress {processed, total, pass_index}` per batch + `ingest_completed {processed, total}` at the end. CPU-only, no external dependency. DCT-based 64-bit hash (`dct_phash_64`). Stored as hex on Example record. Algorithm recorded on Project record (`phash_algorithm`) so migrations can detect algorithm changes. Per-row pHash failure leaves `phash=null` and the sweep continues (no `run_failed`). The review selector uses available hashes and falls back to deterministic `example_key` order when every candidate hash is null, so hashing never blocks labeling.
- **Similarity:** `sim_phash(a, b) = 1 - hamming_distance(phash(a), phash(b)) / hash_bits` (hash_bits = 64). Higher = more similar. Used by pHash-diverse selector (Appendix A.3) and pool rebalancing (Appendix A.5).
- **Schema migration:** pHash values unaffected by schema changes (the image doesn't change). pHash remains valid across Guidance versions.

---

## 6. Interactive Labeling Workflows

### 6.1 Guidance Creation and Image Ingestion  `[SUMMARY]`

Create Guidance (new immutable version). Ingest images (Example records: `storage_ref`, `ingested_at`, `source_metadata`). pHash and CLIP embedding computation are triggered in background. Cold start supported: projects may begin with `Verified = 0`. See §6.1.

### 6.2 ICL Selection and Prompt Rendering  `[EXACT]`

**ICL eligibility definition:**

```
ICL_ELIGIBLE = Label WHERE label_status = 'verified' AND pool_assignment IS NULL AND verified_outcome = 'Edit'
```

Only Edited examples are ICL-eligible. Accepted examples are never selected for ICL. The model already produced correct output for those images, so they add no corrective signal and would consume context window tokens without improving proposal quality. The prompt template (Appendix D.1) provides schema and format guidance independently; ICL is supplementary, not structural.

- ICL eligibility requires `guidance_id = project.active_guidance_id`: only labels created under the current Guidance are eligible. After a semantic Core change (§4.4.1), old labels are deleted, so ICL naturally starts from zero.
- If no Edits exist (cold start or all-Accept run), selection returns empty list and the system proceeds with zero ICL examples.

**Selection algorithm (deterministic, relevance-only — Selective-K):**

1. **Relevance rank:** order all eligible Edits by descending CLIP cosine similarity to the QUERY image (per-query nearest Edits), degrading to newest-first order (ties broken by `example_key`) when embeddings are unavailable.
2. **Adaptive-K:** keep the leading prefix of neighbors within `ICL_SIM_GAP` of the best (optional `ICL_ABS_THRESHOLD` floor; top-1 always kept).
3. **Depth cap:** keep the head within the effective cap (explicit `icl_max_examples`/`ICL_MAX_EXAMPLES` override, else the model's `default_icl_max_examples`, else uncapped).

Selection is always relevance-ranked; there are no per-run selection-policy or pinned-recency configuration paths. Empirical boundary: relevance lifts visual-classification macro-F1 2.7–4.5× on Cosmos but is weak on pure OCR/text-extraction. Token/image budgets prune from the tail of the selection order (least query-relevant first).

**Selection determinism by purpose:**

- **Evaluation and Batch Labeling:** ICL selection MUST use the deterministic algorithm above (apples-to-apples comparisons).
- **Interactive proposals:** MAY use stochastic selection for exploration, but MUST persist `icl_example_keys_used[]`.

**ICL selection size:**

- **Per-model depth default (`ModelConfig.default_icl_max_examples: int | null`, §6.2):** effective selection cap = explicit `icl_max_examples` override (per-run API field or non-null `ICL_MAX_EXAMPLES`) when present — the override wins in either direction; else the model default; else uncapped. Resolved centrally in `prompt_service.invoke_teacher` so proposal/eval/batch cannot drift. Adaptive-K trims within the cap; token/image pruning still applies after. Seeded (July 2026 depth studies): Nemotron Nano VL 2 · Nemotron 3 Nano Omni 4 · CR3 nano/super 8 · CR2-2B 8 · CR2-8B 16 · null for unmeasured current alternates. Historical/operator-created MiniMax M3 records retain 8 and Mistral Large records retain 2.
- If `ICL_MAX_EXAMPLES` is set, selection MUST return at most `ICL_MAX_EXAMPLES` examples (explicit override; replaces any per-model default).
- If the selected ICL set exceeds the per-model token budget or image budget (`max_images_per_request − 1`), prompt packing MUST drop examples from the END of the selection-ordered list — **relevance-tail pruning**: the tail is the least query-similar exemplar (the oldest, in the embedding-less newest-first degrade), so budget enforcement removes the weakest corrective signal first. Token pruning may empty the ICL set entirely; the invocation then proceeds at the cold-start render.
- After all pruning, bookend presentation keeps rank-1 first and moves rank-2 last; stable middle order is preserved. Zero-to-two examples are unchanged. Pruning always sees the original relevance order.

**ICL field rendering:**

- The Teacher uses `output_field_mode=all` and `icl_field_mode=core_only`; it requests `rationale_note` last only when active Guidance enables it.
- Production Teacher ICL renders Core fields only in canonical order; rationale and other Aux prose are excluded. Students run without ICL in v1.0 and retain their training field mode as provenance.

**Prompt stability:** given the same `guidance_id`, ordered ICL set, and query context, rendering MUST be stable and include a compact prompt-visible SchemaCore contract (field names, required/optional, types, enum values, bounds), `E01..` markers, strict JSON instruction, and concise header/pre-query ICL-use directives whenever examples exist. Each example label is scoped to its paired image; examples teach field meanings and decision boundaries, while every query value is derived independently from the query image. The prompt never tells the model to prefer or copy a retrieved label based on visual proximity. The production prompt omits verbose Guidance Description/Rules prose. Prompt-only mode receives the same schema contract and MUST NOT contain a dangling reference to an absent schema. Cold start omits the ICL block/directives. Reference: Appendix D.1.

**Structured generation:**

- Backend MUST attempt `response_format` with `type="json_schema"` when supported. NVIDIA recommends `json_schema` over `json_object`.
- Truncated JSON MUST be treated as schema-invalid (Core invalid).
- Support status persisted per `model_config_id`: `unknown` / `supported` / `unsupported`.
- **Probe (normative):** a single `/v1/chat/completions` with a fixed minimal schema `{"ok": boolean}`, `max_tokens: 16`, bounded by `HTTP_DEADLINE_INTERACTIVE_S`. Classification: accept + parse → `supported`; 4xx from `response_format` → `unsupported`; timeout/5xx → remains `unknown`.
- If probe `unsupported`, omit `response_format` and proceed with prompt-only JSON generation.
- **Runtime `json_schema` rejection after `supported` probe:** per-invocation fallback (does not change probe status):
  - **Interactive invocations:** retry once with prompt-only, surface inline warning. `structured_generation_fallback_used=true` on Operation Record.
  - **Evaluation and Batch Labeling:** MUST NOT silently mix modes within a run. If rejection occurs mid-run with `structured_generation_mode=auto`, run transitions to `failed` with `status_reason="structured_generation_rejected"`. SME restarts with `structured_generation_mode=prompt_only`.
- **Downgrade to `unsupported`:** only via explicit re-probe (§10.2.12) failing with clear `json_schema` rejection. Runtime failures do not change probe status.

**Structured generation mode (run-level control):**

Evaluation and Batch Labeling accept explicit `structured_generation_mode` for the entire run:

- `auto` (default): attempt when `supported`; prompt-only when `unsupported`. Mid-run rejections fail the run (uniform mode required).
- `prompt_only`: never send `response_format` for this run.

Effective mode MUST be snapshotted on the Run Record (`structured_generation_mode_effective`). Interactive proposals do not use this field. When the model is `unsupported`, the UI MUST pin to `prompt_only` and explain why.

**JSON schema derivation (backend-canonical):**

- Top-level `object` with `properties` for all SchemaCore fields, `additionalProperties: false`.
- **Property ordering:** base derivation follows `generation_order`; when enabled, production Teacher invocation moves `rationale_note` last in both `properties` and `x-generation-order` before prompt/guided decoding. When disabled it is absent. Best-effort hint only.
- **`x-generation-order` extension:** derived schema MUST include `"x-generation-order": [...]` as a top-level extension for debugging, introspection, and test verification. Runtime correctness MUST NOT depend on a consumer honoring this extension.
- `required` includes all `role="core"` fields and excludes `role="aux"` fields.
- Each field schema reflects SchemaCore type + constraints.

**Per-model budget enforcement (schema-aware output budget):**

```
schema_output_estimate =
  JSON_STRUCTURAL_OVERHEAD_TOKENS
  + sum(per_field_worst_case_estimate(field) for field in SchemaCore)

base_output_tokens =
  max(BASE_OUTPUT_TOKENS_FLOOR, 2 * schema_output_estimate)

reasoning_headroom_tokens =
  0                                if thinking_mode_effective = "off"
  MODEL_REASONING_HEADROOM_TOKENS  if thinking_mode_effective = "on"

max_output_tokens =
  RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE
  ?? min(
       base_output_tokens + reasoning_headroom_tokens,
       floor(context_window_tokens * MAX_OUTPUT_FRACTION)
     )
```

**Per-field worst-case token estimation:**

- `rationale_note`: `RATIONALE_NOTE_ESTIMATE_TOKENS` (default 160)
- `enum`: key overhead + max tokenized value + 6
- `enum_set`: key overhead + sum of all tokenized values + array syntax
- `boolean`: 6 tokens
- `integer`: 8 tokens
- `string` with `maxLength`: key overhead + ceil(maxLength/4)
- `string` without `maxLength`: `DEFAULT_UNBOUNDED_STRING_BUDGET` (default 200)

`JSON_STRUCTURAL_OVERHEAD_TOKENS` covers braces, whitespace, trailing syntax. The `× 2` multiplier absorbs tokenizer mismatch, model verbosity, and structured generation framing. `RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE` is a deployment-level escape hatch.

**Reasoning headroom (Thinking=ON):** `MODEL_REASONING_HEADROOM_TOKENS` (default 16384) is a fixed headroom added to the schema-derived output budget. On `finish_reason="length"` with `thinking_mode_effective="on"`, surface a non-blocking warning and the automatic retry on truncation handles recovery. `truncation_attributed_schema_invalid` (§13.1) provides diagnostic signal.

`thinking_mode_effective` is resolved per taxonomy and capability: `none` → always `"off"`; `always_on_reasoning` → always `"on"`; request-based Qwen/Kimi modes follow user input only when `thinking_toggle_support=supported`. With support `unknown`/`unsupported`, omit override fields, hide the control, treat natural default as ON, and allocate headroom.

**Input budget:**

```
safety_margin = RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN (default 0.85)

effective_max_input_tokens =
  floor((context_window_tokens - max_output_tokens) * safety_margin)
```

Fixed prompt cost is tokenized from the actual rendered zero-ICL message envelope, including the compact field schema. Raw Guidance length plus a magic constant MUST NOT be used. Per-example estimates add rendered labels/markers; the safety margin absorbs tokenizer and multimodal framing differences.

When Thinking=ON, `max_output_tokens` is larger (reasoning headroom added), which reduces `effective_max_input_tokens`. The system handles this by relevance-tail pruning of the ICL set. Token budget enforcement MUST be resilient to estimation error by dropping ICL deterministically rather than blocking progress.

### 6.3 Inference, Validation, and Review  `[SUMMARY]`

**Image transport (§6.3):** all outbound requests with images MUST follow normative transport rules:

- **Format normalization:** backend MUST transcode unsupported formats to a model-supported format (default: PNG) before sending. MUST NOT fail silently.
- **Chat completions:** base64 data URL (`data:image/<format>;base64,<data>`) in OpenAI-compatible `image_url` content part.
- **Embeddings:** base64 data URL within the `input[]` string array. Batched.
- **Direct URL:** only when the image already exists at a fully qualified, runtime-reachable URL. `storage_ref` values MUST NOT be passed as URLs.
- **Hosted/self-hosted/local NIM:** inline base64 always, regardless of image size. The NVCF Asset API large-image path was **removed 2026-06-02** — the hosted OpenAI-compatible endpoint only ever accepted inline base64, and NVIDIA's June 2026 NVCF deprecation retired the `/v2/nvcf/assets` endpoints (replacement: "send payloads directly during invocation"). See full spec §6.3.
- **ICL example images** follow the same rules. All images for one invocation (ICL + query) are prepared in a single batched call before the model request is dispatched.
- **Persistence:** Operation Record MUST persist `image_transport_mode` ∈ {`base64_inline`, `direct_url`} and `image_format_transmitted` when the invocation includes images.

#### 6.3.1 Proposal Attempt (Teacher Invocation)  `[SUMMARY]`

Per example: (1) determine effective Teacher, Guidance, Generation Controls, Visual Budget (project defaults + per-attempt overrides); (2) resolve endpoint + `model_name` + context window; (3) render prompt + compute `icl_example_keys_used[]`; (4) persist Operation Record before model call; (5) construct and invoke NIM request per §6.7.6 with deadline + bounded retries; (6) persist attempt artifacts and validation report (Core vs Aux separated) — `raw_model_response_ref`, `normalized_json_ref`, `schema_valid_core`, `validation_errors_core[]`, `validation_errors_aux[]`, `validation_report_ref`. Core errors determine schema validity; Aux errors MUST NOT block validity. UI presents proposal + Aux warnings on success, failure state with Save/Retry on failure. Operation Records MUST be persisted for all outcomes. See §6.3.1.

#### 6.3.2 Review Loop (Save / Skip / Retry)  `[EXACT]`

After each proposal attempt, the proposal is displayed in an editable form with three actions: **Save / Skip / Retry** (§4.5). Additional requirements:

The rationale requirements below apply only when `rationale_note` is enabled. When disabled, the system bypasses rationale prompting, review, regeneration, save gating, and provenance.

**Rationale grounding (normative when enabled).** Rationale notes describe concrete, image-specific evidence relevant to the active Task and Rules, using natural vocabulary appropriate to that domain. Quantities, identities, comparisons, and locations MAY be used when visibly supported and useful. A rationale MUST NOT merely echo field values, recite boilerplate, or invent support. Automatic regeneration receives task context and Core schema but neither the original proposal nor the SME correction. It describes the subject the task asks about and distinguishes a physical carrier from depicted content only when that distinction matters to the task, preserving OCR, scene, and anomaly semantics. Ambiguity is stated rather than hidden.

- **Edit detection and diff:** the backend MUST compute a deterministic diff between the proposal normalized JSON and the final SME label JSON and persist `edited_core_fields[]` and `edited_aux_fields[]` for the Verified label. If no fields differ, `verified_outcome=Accept`; if any field differs, `verified_outcome=Edit`. Both diff arrays are required for audit and provenance.
- **Rationale display ordering (anti-anchoring):** when `rationale_anti_anchoring` is enabled (default: `true`), the UI MUST NOT show `rationale_note` alongside the initial proposal. The rationale panel is hidden until the SME modifies a field value. On Save without modifications, the rationale is never shown; the Teacher's version is retained silently. This prevents the model's reasoning from biasing the SME's independent judgment.
- **Rationale is mandatory when enabled and fields are modified.** The Save button is disabled until the rationale is in an approved state. Single editable textarea pre-populated with the model's original rationale. Three states:
  - **"Needs review"** (initial state when a field is modified): textarea contains model's original rationale, editable. Save disabled until meaningfully changed. Two actions:
    - **Edit directly:** state → **"Edited"**. Save enabled.
    - **Generate AI Rationale** button: system calls Teacher with image + task context (Appendix D.3); the request carries neither original nor corrected values. Inline loading replaces textarea content. On completion, state → **"AI-regenerated, review required"**.
  - **"AI-regenerated, review required"**: regenerated rationale in textarea (editable). Save still disabled. Either edit → **"Edited"** (Save enabled) or click **Approve AI Rationale** → **"Approved"** (Save enabled).
  - **"Edited"** or **"Approved"**: Save enabled.
  - Whitespace-only changes do not transition from "Needs review" to "Edited".
- **Reset:** available when any field modified. Restores all fields to VLM's original proposal, collapses rationale panel, returns to unmodified state.
- **Rationale on Edit (backend):** `rationale_source="sme_edited"` if SME wrote/modified; `"teacher_regenerated_approved"` if AI regenerated and explicitly approved. Regeneration invocation persisted as Operation Record with `purpose="rationale_regeneration"` and linked via `rationale_regeneration_invocation_id`.
- **Skip** advances the review selector; **Retry** does not.
- SME may Retry multiple times; only final Save creates Verified label.

### 6.4 Guidance Rewrite (removed)  `[SUMMARY]`

> **Removed before v1.0:** the AI Guidance Rewrite flow and its Guidance-Author role are absent; Guidance improvement is manual editing via the Edit Guidance screen (§4.4).

### 6.5 Labeling Review Selector  `[SUMMARY]`

**Eligibility:** MUST draw only from `state="Unlabeled"` and `state="Auto-Labeled"` examples. After semantic Core change (§4.4.1), prioritizes Unlabeled with `prior_verified_label_ref` (prior Edits first, then prior Accepts), starting from `schema_change_context_example_key` if set.

**Strategy:** CLIP-diverse when available and mode is `auto`/`clip_diverse`; pHash-diverse otherwise (always available).

Skip advances selector and transitions to `Omitted`. Retry does not advance selector. Full algorithm: Appendix A.3.

### 6.6 Create Guidance Screen (UI Contract)  `[REFERENCE]`

Screen-level presentation and interaction model for the Create/Edit Guidance UI: sticky header, two-section field builder (Core + Aux), template selector, backend-driven validation (the builder renders draft-validation issues; no frontend rules), Derived JSON Schema preview, Example label output preview, schema change confirmation dialog. See §6.6; the implemented screen contract is `CreateGuidancePage.tsx` / `EditGuidancePage.tsx`.

### 6.7 Generation Controls  `[EXACT]`

Generation Controls provide two user-facing knobs (**Output Stability presets** and a **Thinking toggle**) for tuning creativity, determinism, latency, and reasoning behavior during labeling, evaluation, and batch labeling. No raw parameter editing is permitted in these flows.

#### 6.7.1 Output Stability Presets (Labeling)

Applies to: interactive proposals, evaluation, batch labeling.

- **Precise** (default): `temperature=0.0`, `top_p=1.0`
- **Explore**: `temperature=0.3`, `top_p=0.9`

Presets map to **only** `temperature` and `top_p`. All other sampling parameters are system-controlled (§6.7.5).

#### 6.7.2 Output Stability Presets (Guidance Rewrite) (removed)

> **Removed 2026-07-21:** the rewrite-specific presets (Conservative / Balanced / Creative) were removed with the AI Guidance Rewrite feature (§6.4).

#### 6.7.3 Thinking Toggle

`ModelConfig.thinking_toggle.mode` taxonomy:

| Value | Toggle? | Reasons? | UI | `thinking_mode_effective` |
|---|---|---|---|---|
| `none` | No | No | Hidden | `"off"` always |
| `always_on_reasoning` | No | Yes (always) | Hidden | `"on"` always |
| `qwen_enable_thinking` | Yes | Yes | Probe-supported only | Follows user input only when supported |
| `kimi_thinking` | Yes | Yes | Probe-supported only | Follows user input only when supported |

- **Default: ON.** For probe-supported toggle models, no override is sent on ON; specific fields disable thinking on OFF (Qwen → `chat_template_kwargs: {"enable_thinking": false}`; Kimi → `chat_template_kwargs: {"thinking": false}`).
- The toggle MUST be **hidden** in the UI for `none` (Mistral — no toggle, no reasoning) and `always_on_reasoning` (Step 3.7 Flash — no working toggle, always reasons). Omni uses `qwen_enable_thinking` and exposes the toggle.
- The split between `none` and `always_on_reasoning` is load-bearing for the §6.2 budget: `always_on_reasoning` requires `MODEL_REASONING_HEADROOM_TOKENS` allocation regardless of user input; `none` does not.

#### 6.7.4 Thinking Override Acceptance Check and Runtime Gating

**Thinking override acceptance check:** applies only when `thinking_toggle.mode` is `qwen_enable_thinking` or `kimi_thinking`. Single `/v1/chat/completions` request with the model-specific thinking-off fields, `max_tokens: 4`, bounded by `HTTP_DEADLINE_INTERACTIVE_S`. Classification: non-error → `supported`; 4xx attributable to `chat_template_kwargs` → `unsupported`; timeout/5xx → remains `unknown`. If `thinking_toggle.mode` is `"none"` or `"always_on_reasoning"`, set `thinking_toggle_support=unsupported` without checking — there is no runtime toggle to probe in either case.

Only `thinking_toggle_support=supported` authorizes `chat_template_kwargs`.
Unknown/unsupported support omits the override, hides the toggle, treats the
natural default as Thinking ON, and reserves reasoning headroom across every
Teacher path, including rationale regeneration.

**Runtime rejection handling:**

1. Mark `thinking_toggle_support="unsupported"`.
2. **Interactive proposals:** auto-retry once without the thinking override, show a warning banner.
3. **Evaluation / Batch Labeling:** fail the run (no silent fallback; reproducibility matters).

#### 6.7.5 System-Controlled and Excluded Parameters

**System-controlled (always set by the system):**

- `response_format`: system chooses `json_schema` when supported (§6.2).
- `max_tokens`: controlled by the prompt budget system; NIM default is 16 so the system MUST always set it explicitly (§6.2).

**Excluded from user controls:**

- `stop`, `ignore_eos`: can break JSON, intended for benchmarking.
- `min_tokens`: can cause repetitive content.
- `logprobs`, `prompt_logprobs`: observability features.

#### 6.7.6 Request Construction (Normative)

For any label-generating invocation, the system constructs the OpenAI-compatible request as follows:

**Step 1, sampling parameters from preset:** set `temperature` and `top_p` from the resolved preset (§6.7.1).

**Step 2, thinking toggle → model-specific fields:** if Thinking=OFF and model supports the toggle (`thinking_toggle.mode` is `qwen_enable_thinking` or `kimi_thinking`):
- `qwen_enable_thinking` → add `chat_template_kwargs: {"enable_thinking": false}`
- `kimi_thinking` → add `chat_template_kwargs: {"thinking": false}`

If Thinking=ON, or `thinking_toggle.mode` is `"none"` or `"always_on_reasoning"`: send no thinking override. The two no-toggle modes never inject `chat_template_kwargs`; the §6.2 budget honors the resolved `thinking_mode_effective` per the §6.7.3 taxonomy.

**Step 3, seed injection by purpose:**
- Purpose `evaluation` or `batch_label`: include `seed = seed_effective` per §2.1.
- Purpose `interactive_proposal`: omit `seed`.

**Step 4, visual budget → `mm_processor_kwargs`:** resolve effective `visual_budget_preset_key` (project default or per-attempt override). Look up model's `visual_budget_mode`. If `visual_budget_mode ≠ none` and `visual_budget_support = supported`: resolve preset to model-specific `mm_processor_kwargs` per §6.9.3. Else: omit. Persist `visual_budget_preset_key` and `visual_budget_params_effective`.

**Step 5, output budget + structured generation:** always set `max_tokens = max_tokens_effective` from the budget system (§6.2). If structured generation is supported, set `response_format: {type: "json_schema", json_schema: ...}`.

**Step 6, automatic retry on truncation:** if `finish_reason="length"` or output appears truncated, retry once with increased output budget (budget-system change only). Re-pack ICL as needed. Do NOT change the sampling preset, thinking toggle, or visual budget preset on automatic retry.

#### 6.7.7 UX Contract

**Labeling screen:** Output Stability (Precise | Explore), Thinking (ON/OFF, default ON; hidden if unsupported).

Teacher and all visible generation controls are persisted project defaults for
the next proposal; they do not re-run the proposal already displayed. Label
actions wait for any in-progress header write, and a failed write is visible
instead of silently falling back to the previous defaults.

### 6.8 Schema Refinement Reminders  `[SUMMARY]`

Two dismissable hints on the labeling screen, each fires at most once per project (reset on semantic Core change):

- **First (default threshold 10):** early signal. *"Need to adjust your schema? Fewer labels to re-do now."* [Review Schema] | [Dismiss].
- **Second (default threshold 35):** last call. *"You have {N} labels. Schema changes mean more images to re-label."*

Rules: neither fires if the user already edited Guidance post-save. If count crosses both thresholds before either fires, only the higher applicable reminder fires. **Collision with the first-pool evaluation banner (§7.1):** while either reminder is visible and undismissed, the first-pool evaluation banner MUST be suppressed (one nudge at a time; schema refinement wins because semantic Core changes invalidate labels). Config: `SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1` (default 10), `SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2` (default 35). Set to `0` to disable. See §6.8.

### 6.9 Visual Budget Controls  `[SUMMARY]`

Controls image preprocessing (visual token spend). Capability-gated: controls only available when the active model supports them. Distinct from text-side Generation Controls.

**Modes (§6.9.1, on ModelConfig):**

- `none`: no controls. Cosmos Reason2 → `mm_processor_size`; Nemotron Nano VL → `mm_processor_tiles`; Mistral / Qwen → `none`.

**Two-stage runtime probe (§6.9.2):** uses a deterministic 512×512 RGB PNG probe image. Stage 1 baseline (no `mm_processor_kwargs`); Stage 2 capability (mode-specific kwargs from NVIDIA docs). Classification: baseline + capability success → `supported`; baseline success + capability fail → `unsupported`; baseline fail → `unknown`. If `visual_budget_mode=none`, auto-set `unsupported` without probing.

**Presets (§6.9.3):** Fast / Balanced / High Detail (default). Each resolves to model-specific `mm_processor_kwargs` values per `visual_budget_mode`. For `mm_processor_size`, the misleadingly named `shortest_edge` / `longest_edge` values are pixel-area bounds, not linear dimensions; the shipped 64K / 128K / 256K upper-bound ladder preserves progressively larger visual-token budgets. High Detail is the default because image fidelity is the dominant perception lever for tunable models (measured in an A/B study; evidence retained in the project's internal engineering archive). When `visual_budget_support=unsupported` and mode is not `none`: omit `mm_processor_kwargs` and show non-blocking warning (interactive) or omit with `visual_budget_params_effective=null` (evaluation/batch).

**Deterministic policy (§6.9.4):** evaluation and Batch Labeling MUST use the project's `visual_budget_preset_key` (no per-example variation). Evaluation runs MUST snapshot `visual_budget_preset_key`. Different presets across runs MUST be explicitly labeled in comparison views.

**Per-invocation persistence (§6.9.5):** `visual_budget_preset_key`, `visual_budget_params_effective` on Operation Record.

### 6.10 Context Budget Indicator (Removed in v1.0)  `[SUMMARY]`

**Removed in v1.0** with the Selective-K simplification (see the full Spec §6.10 tombstone): relevance-ranked selection made "room for more examples" a non-signal, and enforcement stays observable via the ICL-selection log point and `icl_images_attached_count`. Section number retained.

### 6.11 Inference Contract  `[EXACT]`

The system defines an **Inference Contract** for each runnable model configuration. The Inference Contract specifies the effective output format, ICL demonstration format, and ICL sizing controls that apply at inference time. This is a first-class concept — not an implicit side-effect of scattered settings — and it governs how prompts are rendered, how ICL examples are formatted, and what evaluation runs snapshot for reproducibility.

**Inference Contract fields:**

- `output_field_mode` ∈ {`all`, `aux_and_core`, `core_only`}: which field groups the model is expected to produce in its output.
- `icl_field_mode` ∈ {`all`, `aux_and_core`, `core_only`}: which field groups are included when rendering ICL demonstration examples.
- `icl_max_examples: int | null`: maximum number of ICL examples (from `ICL_MAX_EXAMPLES` or override; null defers to the model's `default_icl_max_examples`, §6.2).

**Teacher contract (fixed):** `output_field_mode=all`, `icl_field_mode=core_only`. Applies to Interactive Labeling, evaluation, and Batch Labeling. Teacher output retains the active schema; rationale is requested last when enabled and omitted when disabled. ICL demonstrates only Core correction fields.

**Student contract (derived from training):**

- `output_field_mode` = the `export_field_mode` from the training DatasetExport.
- `icl_field_mode` = `output_field_mode` by default.

This ensures the model is conditioned on demonstration examples in the same format it learned during training. Running a `core_only`-trained Student with `all`-field ICL examples creates a format mismatch that distorts evaluation results and deployment behavior.

**Invariant:** a model is run and evaluated under the same field-mode contract it was designed or trained for. Two evaluation runs are only directly comparable if their effective Inference Contracts match.

**Sole authority for ICL field rendering:** the Inference Contract is the single source of truth for which fields appear in ICL demonstrations. No separate config key, feature flag, or per-invocation override exists for ICL rendering mode. All runtime ICL rendering decisions MUST be attributable to the effective Inference Contract only.

**Persistence:** effective Inference Contract MUST be snapshotted on evaluation Run Records (§13.2.3) and persisted as part of Student training lineage (§13.13). Per-invocation `icl_example_keys_used[]` on Operation Records remains the deepest truth for exact prompt reconstruction; the Inference Contract captures the evaluation regime.

---

## 7. Evaluation Workflows

### 7.1 Evaluation Runs  `[SUMMARY]`

Re-runs the current Teacher+Guidance+ICL setup against the Test Pool. Three triggers drive recommendations or auto-runs based on the **Auto-Evaluate** toggle (default OFF):

1. **First evaluation:** when Test Pool reaches `EVAL_FIRST_POOL_SIZE` (default 5). OFF → banner; ON → auto-run.
2. **Configuration change:** when any tracked config changes (Guidance, Teacher model, Generation Controls, Visual Budget) and pool has members.
3. **ICL growth:** when ICL-eligible Edits doubled since last baseline (last completed eval or last dismissed recommendation). Baseline tracked via `icl_eligible_count_at_completion` (Run Record) and `icl_recommendation_dismissed_at_count` (Project).

SME can manually trigger at any time regardless of toggle. If a gate-basis evaluation is already running when a new gate-basis run starts, the in-progress run transitions `running → canceling → canceled` with `status_reason="superseded_by_newer_evaluation"` and a new run starts. The supersede scopes to gate-basis (interactive Teacher) runs: Student benchmark runs (`student_model_config_id` set, §9.5.2) snapshot immutable configs, so they neither fire it nor receive it. SME may also cancel manually via [Cancel].

**Execution:**

- **Configuration snapshot** on Run Record: model, Guidance, preset keys, thinking mode, effective Inference Contract (§6.11), `icl_eligible_count_at_start`, and credential-free version-2 `runtime_config_snapshot` for every consumed model/endpoint value plus concrete sampling, visual, token-budget, ICL, and image-downscale inputs. Later semantic changes affect future runs only. Credentials, filesystem authorization, timeouts, retries, concurrency, rate limiting, and emergency capability kill switches remain live operational policy.
- Background; MUST NOT block Interactive Labeling.
- Inferences run concurrently; provider-aware: `EVAL_CONCURRENCY_HOSTED` (default 1) for hosted endpoints, `EVAL_CONCURRENCY_SELF_HOSTED` (default 8) for self-hosted/local NIMs.
- **Sequential retry pass** after concurrent burst: failed examples retried at concurrency=1 (avoids rate-limit storm). If any still fails, the run is marked **incomplete**.
- On start, system auto-creates a frozen pool version snapshot (§4.3.3). Run references the snapshot.
- If Test Pool has no members, evaluation MUST NOT proceed.
- Uses the same Interactive Labeling prompt pipeline.
- For "without ICL", forces ICL to zero.
- Persists per-example Operation Records (`purpose=evaluation`) including `invocation_status`, `exact_match_pass`, `latency_ms_end_to_end`.
- **Incomplete runs do not satisfy the Scale-Up Readiness Gate (§7.3).**

Evaluation uses only Verified examples under the current Guidance. After a semantic Core change, old labels are deleted; as examples are re-labeled and the pool rebuilds, evaluation becomes available naturally. No special lockout.

See §7.1 for full prose.

#### 7.1.1 Configuration Change Detection  `[SUMMARY]`

Tracked fields compared (most recent completed eval vs project's current): `guidance_id`, `model_config_id`, `generation_preset_key`, `thinking_mode_effective` (mapped from `thinking_default_on`), `visual_budget_preset_key`. If any differ, config has changed. Lightweight (persisted values only). The Scale-Up Readiness Gate always uses the most recent completed evaluation's overall metrics. See §7.1.1.

#### 7.1.2 Returning vs New Metric Split  `[EXACT]`

Because the Test Pool grows over time, a drop in overall accuracy can mean the model degraded or simply that a harder example joined the pool. To separate these signals, each evaluation reports metrics in three buckets.

**Bucket definitions:**

When an evaluation completes, the system compares the current pool version snapshot against the previous completed evaluation's snapshot:

- **Returning**: examples present in both the current and previous snapshots. Metrics on this set are the regression signal.
- **New**: examples present in the current but not the previous. These entered the pool since the last evaluation.
- **Overall**: all examples in the current snapshot (Returning + New combined).

On the first evaluation (no previous snapshot), all examples are reported as Overall only; there is no Returning/New split. After a semantic Core change creates a new Guidance version, evaluation history under the prior Guidance is not used as the "previous" baseline. The first evaluation under the new Guidance is treated as a fresh first evaluation (Overall only). Prior Run Records remain in history for audit.

**Required metrics per bucket:** Exact Match rate, per-core-field match rate, per-value precision/recall/F1 for categorical Core fields (Appendix A.2).

**Persistence:** the Run Record (§13.2) MUST persist:

- `previous_pool_version: int | null` (null on first evaluation)
- `returning_example_keys[]: string[]`
- `new_example_keys[]: string[]`
- `icl_eligible_count_at_completion: int`
- Per-bucket aggregate metrics (Returning, New, Overall)
- `previous_overall_exact_match: float | null` (so comparison can be rendered without loading the previous Run Record)

**UX contract:** the core of the display is comparing previous Overall against current Returning — the same images re-tested under the current setup:

`Previous: 82% (15) → Same images now: 85% (15) · New: 60% (5 new) · Overall: 78% (20)`

**Previous** = Overall accuracy from last completed evaluation. **Same images now** (Returning) = accuracy on those same images under current setup — the fundamental comparison. A green delta (↑3%) or red delta (↓5%) makes direction immediately visible. **New** and **Overall** are secondary context. First evaluation: `Accuracy: 80% (5 images)`.

A **[Results]** button opens a detail panel with full breakdown: Returning/New/Overall comparison, per-core-field match rates, expandable per-value P/R/F1, coverage gaps. Panel includes the run's configuration snapshot.

#### 7.1.3 Test Pool Coverage Warnings  `[SUMMARY]`

After evaluation completes, the system checks whether test pool ground truth covers all schema values. Per-field: Enum (check all `allowed_values[]`), Boolean (check true + false), Integer with range (report observed vs schema range), Enum Set (each allowed value appears in at least one label). String: no check. Non-blocking warning. Persisted on Run Record as `coverage_gaps[]: {field_name, field_type, missing_values[]}`. See §7.1.3.

### 7.2 Evaluation Sources (`evaluation_source`)  `[SUMMARY]`

Every evaluation run records `evaluation_source`:

- **`tao`**: quality evaluation inside TAO against a checkpoint. Accuracy metrics only (Appendix A.2), no serving metrics. Available automatically after training/quantization (§9.7.6). Re-scored by canonical evaluator.
- **`nim`**: evaluation against a deployed NIM endpoint. Both accuracy and serving metrics (latency, throughput, NIM profile). Available after NIM deployment (§9.5.2).

Teacher runs (§7.1) carry a null source. The Compare screen reads the Teacher baseline from the most recent completed Teacher-contract run and each Student's results from `quality_evaluation_run_id` / `serving_evaluation_run_id`. **Absent from v1.0:** the evaluation-suite grouping and the Student+ICL evaluation arm (Students deploy and are evaluated bare). Full rationale in the main Spec §7.2.

### 7.3 Scale-Up Readiness Gate  `[EXACT]`

Determines whether the project meets configurable quality criteria for Batch Labeling. System-evaluated; re-evaluates automatically when underlying data changes. The gate MUST NOT block Interactive Labeling, evaluation runs, or any other workflow.

#### 7.3.1 Gate Criteria

All five criteria MUST pass for system-ready:

**1. Overall Exact Match.** Most recent completed evaluation's overall Exact Match MUST meet `SCALEUP_EXACT_MATCH_THRESHOLD` (default 0.80). If no completed evaluation exists, criterion fails.

**2. Per-core-field match rate.** In same evaluation, every Core field's individual match rate MUST meet `SCALEUP_PER_FIELD_MATCH_THRESHOLD` (default 0.80). Reports which field(s) failed and current rates.

**3. Minimum per-value F1.** Every value of every categorical Core field (enum, enum_set, boolean) MUST have F1 meeting `SCALEUP_MIN_PER_VALUE_F1_THRESHOLD` (default 0.60). Catches systematic misclassification even when overall field metrics pass. Reports failing values with F1 + precision + recall (diagnostic context).

**4. Accept rate (rolling window).** Rolling Accept rate over most recent `SCALEUP_ACCEPT_RATE_WINDOW` Verified labels (default 50) MUST meet `SCALEUP_ACCEPT_RATE_THRESHOLD` (default 0.80):

```
accept_rate = count(verified_outcome = Accept) / count(all outcomes in window)
```

If fewer than window exist, denominator is actual count; threshold still applies. Zero Verified → fails.

**5. Minimum Test Pool size.** Live Test Pool MUST contain ≥ `SCALEUP_MIN_TEST_POOL_SIZE` members (default 60).

#### 7.3.2 Gate Evaluation Timing

Re-evaluate: after each new Verified label, after each completed evaluation run, on Scale-Up hub page load. Lightweight — queries persisted metrics and counts only. The gate reads from the most recent completed evaluation; does not trigger evaluations itself.

#### 7.3.4 Gate Status Reporting

Structured object:

- `gate_status` ∈ {`not_ready`, `ready`}
- Per-criterion detail: `{criterion_name, passed: bool, current_value, threshold, message}`

UI MUST present in plain language. MUST NOT expose raw metric names or MLOps jargon. Example messages: *"Model accuracy: 88% overall (need 80%). Passed."* / *"Per-field quality: 'damage_severity' at 72% (need 80%). Continue labeling or refine Guidance for this field."* / *"Per-value quality: 'scratch' in 'damage_type' has F1 67% (need 80%, precision 75%, recall 60%). The model is missing this category. Add more examples or refine Guidance."* / *"Accept rate: 65% over last 50 labels (need 80%). Continue Interactive Labeling."* / *"Test Pool: 24 examples (need 60). Continue labeling to grow the pool."* / *"No evaluation run found. Run Evaluation to measure quality."*

---

## 8. Batch Labeling Workflows (Teacher-run)  `[SUMMARY]`

Generates large volumes of Auto-Labeled synthetic data. Starting a run requires Scale-Up Readiness Gate `ready` (§7.3).

The pre-run screen resolves the selected Teacher endpoint's derived
`usage_policy`. A seeded hosted API Catalog endpoint is `evaluation_only` and
requires a launch-time confirmation explaining that trial credits do not
authorize production, with Continue evaluation / Configure production endpoint
/ Cancel actions and a link to NVIDIA's API Trial Terms. Other endpoints are
`operator_managed`: the Blueprint makes no entitlement claim and does not show
the trial confirmation.

**§8.1 Purpose/non-goals:** generate large dataset without per-item SME validation; use same Guidance+ICL that performed well. Auto-Labeled labels are NOT ground truth; system MUST NOT convert to Verified without SME review.

**§8.2 Execution (8 steps):**

1. Verify Scale-Up Readiness Gate `ready` (409 if not).
2. Generate `batch_label_run_id`.
3. **Snapshot config** on Run Record: model, Guidance, preset keys, thinking mode, and credential-free version-2 `runtime_config_snapshot` for every consumed model/endpoint value plus concrete sampling, visual, token-budget, ICL, and image-downscale inputs. All steps use snapshotted semantic values; later changes affect future runs only. Live credentials, authorization, timeout/retry, concurrency, rate-limit, and emergency-kill-switch policy remains operational. The v1_0003 migration wrote model/endpoint-only version 1; startup upgrades resumable Batch snapshots exactly once or fails that run closed.
4. Resolve Generation Controls + Visual Budget. Seed injection uses `batch_label_run_id` as `scope_id`. Visual budget uses snapshotted preset for ALL examples (no per-example variation).
5. **Input selection:** default = `state="Unlabeled"` excluding `Omitted`. When `include_auto_labeled=true`, also include `Auto-Labeled` — existing Label records replaced with new Auto-Labeled output (prior Labels overwritten; prior Operation Records preserved). Useful when Guidance improved without Core schema change. `BATCH_LABEL_RUN_LIMIT` caps run size when set (ingestion order).
6. For each selected example (provider-aware dispatch width: `BATCH_LABEL_CONCURRENCY_HOSTED` default 1, `BATCH_LABEL_CONCURRENCY_SELF_HOSTED` default 8 — same policy as evaluation §7.1; per-run `concurrency` override persisted on the Run Record so restart recovery resumes at the same width): render prompt with `active_guidance_id`; select ICL per §6.2 unless the run's `icl_mode` is `disabled`; construct NIM request per §6.7.6; persist Operation Record with `purpose=batch_label`, `label_tier="auto_labeled"`, `batch_label_run_id`; validate/normalize. Only `schema_valid_core=true` outputs produce Label records (`label_status=auto_labeled`, `label_json`, `batch_label_run_id`, `inference_invocation_id`). Schema-invalid outputs recorded on Operation Records only. Example transitions to `state="Auto-Labeled"`.
7. **Idempotent keying:** `{batch_label_run_id}:{example_key}` is the logical item key. Restart resume validates the unique input snapshot and frozen total, then reuses each pending invocation ID. A successful item atomically commits its terminal Operation outcome, exact-run/exact-invocation/exact-Guidance Auto-Labeled Label, and `Auto-Labeled` Example state; later `Verified`/`Omitted` SME state supersedes the machine label. Recovery repairs torn or mismatched success under the original ID and fails closed on duplicate, foreign, invalid-status, or ambiguous lineage. Re-execution MUST NOT duplicate outcomes.
8. **Circuit breaker:** consecutive failure counter, completion-ordered under concurrent dispatch (`timeout` → increment, `endpoint_error` → increment, `schema_invalid` → ignored, success → reset to 0). The run snapshots `BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD` (default 10), and each authoritative item transaction persists the new streak plus a tripped latch with its outcome. The latch survives a later in-flight success that drains and resets the streak, so restart cannot reset the spend boundary. At the threshold, new dispatch stops (including work released from a foreground hold), in-flight requests complete and are recorded, then the run transitions to `paused` with `paused_reason="circuit_breaker_threshold_reached"`. User: [Resume] (`paused` → `queued` → executor-owned `running`, explicitly reset streak/latch, continue from next unprocessed) or [Cancel] (directly to terminal `canceled`, persisted results retained). Prevents burning rate limits/credits on a down endpoint.

**§8.3 ICL:** enabled by default for Batch Labeling; deterministic algorithm per §6.2; subject to per-model context window, prompt packing token budget, per-model depth default (`default_icl_max_examples`, §6.2), `ICL_MAX_EXAMPLES` when set. Per-run `icl_mode="disabled"` on the start request (F-S9 amendment) skips ICL selection — the supported path for ICL-negative teachers, snapshotted on the Run Record and restart-safe. Total run size governed by `BATCH_LABEL_RUN_LIMIT`, not ICL settings.

**§8.4 Auto-Labeled dataset export:** uses Cosmos-RL format (§9.3: `annotations.json` + media archive). Recommended: export Verified train separately from Auto-Labeled (`verified_train_llava.tar` / `auto_labeled_batch_labeled_llava.tar`); downstream may merge with weighting. DatasetExport records (§13.5) with manifest ref. Auto-Labeled entries are training data only and require both a `label_status=auto_labeled` Label and an exact owning Example with `state="Auto-Labeled"`. Evaluation/testing exports require `verified_only`; Auto-Labeled predictions are never an answer key. After semantic Core change, only examples re-labeled under new Guidance are exportable.

---

## 9. Student Training Workflows (Cosmos-RL / TAO VLM)

Student Training is optional.

### 9.1 Enablement Gate  `[SUMMARY]`

Requires Student Training capability (Cosmos-RL / TAO VLM integration configured). Preflight MUST validate: (1) TAO endpoint reachable (§1.6); (2) base model has `student_base` role (§4.8). Hardware/environment constraints (GPU count/memory, vGPU, driver/CUDA, disk) are TAO's responsibility — failures surface through TAO job polling (§9.7.4). Known constraints for reference: NIM for VLMs does not support vGPU; Cosmos Reason2 8B SFT: 8× A100 80 GB minimum; 200 GB installation floor (500 GB practical for one retained full suite; 1 TB preferred for variants/retries); Ubuntu 22.04 LTS is NVIDIA's recommended FTMS 6.26.3 baseline and Ubuntu 24.04 LTS is Blueprint-live-validated; Driver 580.65.06+ (R580 branch), CUDA 13.0+. See §9.1.

### 9.2 Training Data Sources  `[SUMMARY]`

Required base: Verified excluding Test Pool (only labels verified under current Guidance). Optional: Auto-Labeled (batch-labeled, at minimum Core schema-valid). See §9.2.

### 9.3 Dataset Export Contract  `[EXACT]`

When enabled, system MUST export in the Cosmos-RL TAO-native dataset format. The wire format is derived from NVIDIA's Cosmos-RL documentation and upstream LLaVA custom-data guidance.

#### 9.3.1 Archive Layout

Each export artifact is a dataset folder containing:

- `images.tar.gz` (or `images/` directory): all referenced image files, organized by relative path.
- `annotations.json`: the annotation file defining samples and conversations.

Implementations MUST produce archives where every `images[*]` path in `annotations.json` resolves to an existing file in the media payload.

#### 9.3.2 `annotations.json` Structure (Normative)

`annotations.json` is a **top-level JSON array** of sample objects. Each sample object has:

- `id: string` (required). Maps to `example_key`.
- `images: string[]` (required). Array of relative paths to image files within the archive. For this system's single-image workflow, `images` MUST contain exactly one entry. The path is relative to the archive root (e.g., `"images/example_000001.png"`). Note: NVIDIA Cosmos-RL documentation uses `images` (plural array), not `image` (singular). This spec follows the NVIDIA Cosmos-RL convention because TAO is the downstream consumer.
- `conversations: object[]` (required). Array of exactly two conversation turn objects:
  - **Turn 1 (human):** `{"from": "human", "value": "{rendered_serving_prompt}"}` — the rendered §6/D.1 serving prompt in zero-shot self-contained form (F-W7 amendment 2026-07-14: trains Students on the prompt distribution they serve with; the prior short `task_prompt` form cost a measured 3.4× EM at deployment). `<image>` appears within it as a **literal token**, exactly once, at the query-image position. Training and evaluation exports render identical human turns. Canonical renderer: `services.prompt_service.render_training_conversation_prompt`.
  - **Turn 2 (assistant):** `{"from": "gpt", "value": "{label_json_string}"}` where `{label_json_string}` is the label serialized as a JSON **string** (see serialization rule below). Content is governed by `export_field_mode`.

Turn field names are `from` and `value`, not `role` / `content`. This matches both NVIDIA Cosmos-RL documentation and upstream LLaVA examples.

**Label serialization rule:** the assistant turn's `value` field is always a text string. Serialize deterministic compact JSON in serving-prompt order: other Aux/Core fields retain base relative order and `rationale_note` is last when included. The assistant turn MUST NOT contain a nested JSON object.

**Normative example:**

```json
[
  {
    "id": "example_000001",
    "images": ["images/example_000001.png"],
    "conversations": [
      {"from": "human", "value": "<image>\nClassify the visible damage and return JSON only."},
      {"from": "gpt", "value": "{\"rationale_note\":\"dent visible on front-left corner\",\"damage_type\":\"dent\",\"severity\":2}"}
    ]
  }
]
```

#### 9.3.3 TAO Config Field Naming

Each cosmos-rl action class consumes a different config schema (live-verified 2026-04-29 against TAO FTMS 6.26.3 + cosmos-rl 6.26.3):

- **`train`** — `specs.custom.train_dataset.{media_path, annotation_path}` (cosmos-rl `CustomConfig` Pydantic schema requires the `train_dataset` key by name).
- **`evaluate`** — top-level `specs.dataset.{media_dir, annotation_path}` (cosmos-rl `ITSEvaluator` reads `config["dataset"]`).
- **`quantize`** — top-level `specs.dataset.{media_dir, annotation_path}` (F11 amendment 2026-05-05: cosmos-rl-quantize's CLI accepts `--media_dir` and rejects `--media_path`; moved from the train-side `custom.train_dataset` binding). Plus `specs.quantization_scheme = <FP8_DYNAMIC|...>` (NOT `quantization_method`) and `specs` MUST OMIT `policy.model_name_or_path`, `train.*`, and `validation.*` — the cosmos-rl-quantize CLI rejects every train-side flag.
- **`inference`** — no dataset binding.

Archive layout is the same regardless of action; only the spec key the worker reads differs. Apply the correct binding per action via `services.tao_job_service.apply_dataset_binding(action=…)` (single source of truth).

#### 9.3.4 Export Validation Requirement  `[SUMMARY]`

Before dataset export integration is implementation-complete, the following MUST be validated against the pinned `cosmos_rl_container_tag`: parse `annotations.json` as JSON array, every `images[*]` resolves to an existing file, run one TAO/Cosmos-RL smoke test (1-epoch `train` against pinned container). Until this passes, the export contract is decisioned but not implementation-closed. See §9.3.4.

#### 9.3.5 Export Field Mode  `[EXACT]`

**Export field mode** (`export_field_mode`):

Controls which field groups are included in the gpt turn JSON. Field ordering matches serving: other Aux/Core retain base relative order and `rationale_note` is last when included.

- `all` (default): gpt turn contains available `rationale_note` + other Aux + Core, retaining auxiliary/audit data without making it Teacher ICL authority.
- `aux_and_core`: gpt turn contains Aux (excluding `rationale_note`) + Core. Observational scaffolding without explicit reasoning.
- `core_only`: gpt turn contains Core only. Smallest output format; lowest inference cost.

Mode is set per export and persisted on DatasetExport. Same `export_field_mode` MUST apply consistently across all examples within a single export artifact.

**Training consistency:** a Student training run MUST use a single `export_field_mode` across all included datasets. If multiple datasets are combined (e.g., Verified + Auto-Labeled), they MUST share the same field mode, or be normalized to a single selected mode before training. Mixing modes within one Student training run is not allowed — the model would learn an inconsistent output contract. `export_field_mode` is stored on DatasetExport records only (single source of truth, §13.5); the Student's Inference Contract (§6.11) derives its field mode from the referenced training DatasetExport records at query time.

TAO distinguishes dataset intents (training, evaluation, testing). Export layer MUST emit explicit dataset manifests per intent so TAO dataset bindings can be wired correctly for each job action.

Verified (train) and Auto-Labeled MUST be exportable as separate artifacts.

Exports MUST create DatasetExport record (§13.5) with: `dataset_export_id`, `dataset_intent` ∈ {`training`, `evaluation`, `testing`}, `export_field_mode`, status lifecycle (`running | completed | failed` — the standalone API endpoint builds the archive in a background task and populates artifact refs at completion; §10.2.18), artifact refs (tar + checksums), manifest ref, linkage to `guidance_id`, selection definition snapshot.

### 9.4 Multi-model Training Suites  `[SUMMARY]`

Trigger Cosmos-RL / TAO VLM fine-tuning for one or more Student bases under
one sequential suite. Validate training setup defaults to one recommended
small base; Compare candidate variants is the explicit multi-base intent.
At least one base must remain selected. Start also requires a non-empty
Verified Training Pool and the project-configured Test Pool minimum; these are
dataset-readiness checks, not Teacher-quality gates. See §9.4.

### 9.5 Student Registry and Deployment  `[SUMMARY]`

Each training run records: `guidance_id`, pool version, training split definition, dataset export refs, resulting Student identifier, `nim_vlm_release_version`. Each variant deployable behind a NIM-compatible inference endpoint.

**Two distinct evaluation/deployment phases:**

- **Quality validation (TAO-backed, preferred):** TAO `evaluate` runs automatically after training and after each quantization (§9.7.6). Accuracy metrics. No NIM deployment required. System re-scores TAO per-sample predictions with the canonical Core-field evaluator for authoritative metrics. On success, flips Student `quality_status="validated"` with `quality_evaluation_run_id` pointing at the TAO RunRecord.
- **Quality validation (NIM-backed fallback, narrow):** NIM-source eval MAY satisfy the quality gate **only** when (a) `quality_status="pending"` (cold start, no TAO eval has terminated) OR (b) `quality_status="failed"` AND the prior failed TAO `evaluate`'s failure evidence (`error_ref`, `poll_error_ref`, `chain_halted_reason`, or `outputs.tao_logs_text`) matches a known upstream loader-gap pattern from `services.tao_failure_classifier.MODEL_LOADER_FAILURE_PATTERNS`. Other TAO failures (dataset shape, OOM, transient infra, schema mismatch) leave `quality_status="failed"` — NIM eval is NOT a generic rescue. Failure-evidence capture: polling service MUST best-effort fetch `:logs` (TAO REST API Overview) on **any** TAO action failure and persist the tail on `TAOJob.outputs.tao_logs_text` (F42 amendment 2026-05-13 extended from evaluate-only to train + quantize + evaluate). Audit invariant: TAO-validated audit pointer (`quality_evaluation_run_id`) is preserved when set. Implementation: `services/student_nim_lifecycle._promote_quality_from_nim_eval` + `services/tao_failure_classifier`. Two canonical pattern entries today: (1) cosmos-rl 6.26.3 + Cosmos-Reason2 + Qwen3-VL-dense gap (documented 2026-05-04 — registry miss + `qwen2_5_vl.py` fallback + `vocab_parallel_embedding.py:457` assert; evidence retained in the project's internal engineering archive); (2) **F43 (2026-05-13) — `"Following weights were not initialized from checkpoint"`** signature, Cosmos-Reason2-**8B**-specific vLLM weight-init validation that NIM 1.6.0's vLLM does NOT exhibit on the same checkpoint (full report retained in the project's internal engineering archive). Live-validated on 2026-05-13 — flipped 8B Student `dd05ae9f` from `failed → validated` after a clean 84/84 NIM eval rerun unlocked the §9.5 condition-(b) gate.
- **Pending Student UI path:** a packaged `quality_status="pending"` Student remains visible on Compare with **Deploy and benchmark**. This makes the permitted NIM-only cold-start validation path reachable without an API-only workaround.
- **Serving validation (NIM-backed):** NIM deployment enables a pinned-AIPerf comparison over a deterministic sample of up to 200 real frozen Test Pool images, using the evaluated Guidance-derived Student prompt/schema and Inference Contract with no output-token cap. Local and external endpoints use the same cache-disabled workload (§9.5.2). Every configured concurrency must complete its exact request count with zero failures; all cells, including failures, remain durable evidence. Serving status and `serving_evaluation_run_id` remain independent of the quality path.

Quality validation establishes "did fine-tuning produce a better model?" Serving validation establishes "can this model serve in production, and at what cost?" The `deployment_handoff` Action Request (§10.3) requires `quality_status="validated"` AND `serving_status="validated"`; either source for quality is acceptable.

**F14 (Phase 12 amendment 2026-05-05) — different strictness for the two gates against the same NIM Run Record.** Serving validation accepts `run_record.status IN (completed, incomplete)` — the gate confirms the container served and produced parseable output for some invocations; model accuracy is a separate concern. Quality validation (`_promote_quality_from_nim_eval`) accepts `run_record.status == "completed"` ONLY — conservative because model accuracy is the load-bearing signal for the Scale-Up Readiness Gate (§7.3) and the deployment handoff's customer-visible metrics. A Student whose post-fine-tune model emits some schema-invalid examples can still reach `serving_status="validated"` but stays at `quality_status="failed"` until a clean run lands. The `deployment_handoff`'s dual gate prevents shipping a Student that serves but is too inaccurate.

**F35 (Phase 12 closeout amendment 2026-05-06) — partial quality_status for incomplete-but-mostly-parseable NIM evals.** Adds a third `quality_status` value `partial` (enum is now `pending | validated | partial | failed`). When a NIM-source eval finishes `incomplete` AND parseable rate ≥ `STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD` (default 0.90), the paired Student lands at `quality_status="partial"` instead of staying at the prior value. **Informational, not gate-passing** — `deployment_handoff` still requires `validated` (a `partial` Student returns 409 `conflict: quality_status_partial`, distinct from `conflict: quality_status_not_validated`). Audit invariants: `validated → partial` is illegal (helper is no-op); `pending → partial` and `failed → partial` are the only valid promotion transitions. F33 `:rerescore` rejects `partial` with the standard `student_not_failed` 409 — partial is set by NIM eval, not by a stale TAO rescore. Implementation: `services.student_nim_lifecycle._promote_quality_to_partial` mirrors `_promote_quality_from_nim_eval`'s shape; the lifecycle's post-eval gate routes between `validated` / `partial` / no-promotion in BOTH local-mode and external-mode (F35 unifies F14 across the two paths). UI: yellow KUI `<Badge>` "Quality: Partial" + helper line on `StudentVariantCard`. Live evidence motivating the amendment: 8B baseline `ab007102` 65% NIM exact-match on 135/148 (91% parseable) — under F14 alone stuck at `failed`; under F35 lands at `partial`.

TAO Inference Microservices are NOT part of this system's serving story. Production Student serving MUST use NIM.

See §9.5 for full contract.

#### 9.5.1 Checkpoint Handoff (TAO → NIM)  `[EXACT]`

After a TAO training job succeeds (or after quantization via either lane in §9.8), a **packaging step** MUST verify the output is a NIM-loadable checkpoint before the Student can be registered:

- For Cosmos Reason2 (vLLM backend): `NIM_MODEL_NAME` must point to a directory containing a HuggingFace checkpoint or quantized checkpoint. Expected structure: HF model files at root (`config.json`, `generation_config.json`, safetensor shards, tokenizer files, `runtime_params.json`).
- System MUST validate checkpoint directory structure before registration succeeds.
- **LoRA merge-or-validate (full-precision baseline):** if LoRA was used (`enable_lora=true`), packaging MUST first check whether the training output is already a NIM-loadable HuggingFace checkpoint directory. If it is (full model shards present), no merge is performed. If the output is adapter-only (LoRA weights without merged base), packaging MUST merge the adapter into the base model using the persisted `base_model_path` (sourced from the training TAOJob's `resolved_training_fields.policy.model_name_or_path`), materialize a merged HuggingFace checkpoint directory, then validate the result. If `base_model_path` is unavailable or merge fails, packaging MUST fail with a clear error identifying the missing prerequisite. Note: `export_safetensors: true` controls serialization format only; it does not guarantee adapter merge.
- **LoRA merge for quantized variants:** TAO `quantize` auto-merges when `enable_lora=true` and `base_model_path` is set (§9.8). No separate merge step is needed for quantized variants.

**LoRA deployment policy:** NIM VLM does not document runtime LoRA adapter serving for Cosmos Reason2. The deployment input to NIM MUST be a merged HuggingFace checkpoint or quantized checkpoint. Do not specify runtime LoRA adapter serving for VLM NIM unless NVIDIA later documents it explicitly.

Persist on Student registry record:

- `checkpoint_packaging_status` ∈ {`pending`, `validated`, `failed`}
- `nim_checkpoint_ref: string` (path/URI to NIM-loadable checkpoint directory)

Portable handoff provenance follows two distinct lineages: the Student's
`dataset_export_ids[]` remains training-only, while the Test Pool SHA-256 is
resolved from the held-out `evaluation`/`testing` export on the evaluate job
paired with the artifact-producing train or quantize job. The same checksum is
persisted on Student serving runs and the deployment bundle manifest.

#### 9.5.2 NIM Local Deployment Orchestration  `[SUMMARY]`

System attempts to deploy Student NIM containers locally for evaluation. Temporary infrastructure, not production. Shared-image Students inherit the base ModelConfig's `NIM_MODEL_SIZE`, pinned `NIM_MODEL_PROFILE`, and validated non-secret container env in preflight, live launch, handoff, and portable bundle while retaining a Student-specific served name. Cosmos 3 Super carries the live-validated `NIM_MAX_MODEL_LEN=65536` clamp needed for its BF16 checkpoint on a 96 GB GPU. A durable server-side `serving_status="pending"` remains visibly in progress after refresh and blocks duplicate deployment actions.

**Preflight (6 checks):** (1) Docker available, (2) NVIDIA Container Toolkit (GPU passthrough), (3) GPU memory sufficient (seeded BF16 nominal floors: 8B ≥56 GB, 2B ≥36 GB, with the shared 1% reporting tolerance; FP8 uses model-specific `nim_gpu_memory_minimum_gb`), (4) NGC_API_KEY valid, (5) NIM container image pullable, (6) checkpoint exists with `checkpoint_packaging_status="validated"`. Persisted: `nim_preflight_status`, `nim_preflight_details`, `nim_preflight_at`.

**If preflight passes (Tier 1 local orchestration):** **(0) acquire GPU (F49 amendment, 2026-05-19)** — stop every `LocalNimDeployment` row in `starting | running` on the target GPU before docker_run; each displaced row's `displaced_by_deployment_id` (the Student's deployment_id) + `displaced_at` are persisted on §13.15; on multi-GPU hosts where auto-placer found a free device, step 0 is a no-op; (1) construct `docker run` from StudentModel (checkpoint path, served model name, NIM cache, NGC key, port, `--shm-size=32GB`, `-u $(id -u)`, GPU allocation, `NIM_ENABLE_KV_CACHE_REUSE=0`); (2) start, poll `/v1/health/ready` up to `NIM_STARTUP_TIMEOUT_S` (default 1200s); (3) smoke inference; (4) register temporary endpoint as ModelConfig; (5) run NIM evaluation against Test Pool with `evaluation_source="nim"`; (6) build one deterministic real-image workload from the frozen Test Pool and evaluated Student production contract, then replay it exactly once at each `STUDENT_LATENCY_TEST_CONCURRENCIES` level (default [1,8,24]) through pinned `aiperf==0.10.0`; persist workload provenance, integer-ms latency percentiles, achieved RPS, exact failures, token means when present, and best-effort Prometheus evidence; (7) stop container (persistent NIM cache retained for next startup); (8) repeat next variant; **(9) auto-restore displaced residents (F49 amendment, 2026-05-19)** — iterate step 0's displaced deployments and best-effort re-deploy each (same role / model_config_id / gpu_assignment), health-poll up to `NIM_STARTUP_TIMEOUT_S`; failure surfaces as warning on `serving_evaluation_run_id` summary but does NOT fail `serving_status`. Container lifecycle events logged on StudentModel.

**If preflight fails (Tier 2 Action Request fallback):** generate `student_nim_deploy` Action Request (§10.3) with exact cache-disabled `docker run` command, checkpoint path, GPU requirements, NIM release, env vars, mount paths, health check, smoke test, temporary-infra note. SME hands to infrastructure; once the endpoint is running with KV-cache reuse disabled, the SME registers its URL and explicitly confirms the cache policy; the system runs evaluation plus the same real-image benchmark against that endpoint.

**NIM configuration defaults (Cosmos Reason2 custom deployment):** vLLM backend, `--shm-size=32GB`, persistent NIM cache at `/opt/nim/.cache`, `NIM_CUSTOM_MODEL_NAME` for cached engine, port 8000.

**Sequential variant benchmarking:** SME selects which variants to benchmark (all / selected / individual). Sequential, one container at a time. Per-variant timeouts: startup 1200s, benchmark `NIM_BENCHMARK_TIMEOUT_S` (default 1200s). On timeout: variant marked, queue continues. Results progressive. No global workflow timeout in v1.

See §9.5.2 for full details.

### 9.6 Student Inference Modes  `[EXACT]`

Student inference MUST use the Inference Contract derived from training lineage (§6.11):

- `output_field_mode` and `icl_field_mode` both default to the `export_field_mode` from the training DatasetExport. A `core_only`-trained Student produces Core-only output.
- Students run without ICL: TAO evaluation does not support ICL injection, and v1.0 deploys, serves, and evaluates Students bare (`icl_mode="disabled"`).
- Release gating MUST match the intended Inference Contract: the Student MUST be evaluated under the same contract it will use in production.

### 9.7 Cosmos-RL / TAO VLM Job Triggering and Tracking  `[SUMMARY]`

#### 9.7.1 TAO Execution Substrate

TAO as **remote external executor** via REST API. Preferred: TAO FTMS TAO API v2 (unified Jobs API; supports pause/resume/cancel/delete; logs/files/outputs; metadata may include progress/metrics). Pinned versions: `tao_release_version: "6.26.3"`, `cosmos_rl_container_tag: "6.26.3-cosmos-rl"`. Persisted on every TAOJob. Implementations targeting a different release MUST update pins and revalidate export format (§9.3.4).

All TAO calls use `TAO_API_BASE_URL` + authenticate with `TAO_API_KEY`. API paths use `TAO_ORG_NAME` (e.g., `GET {TAO_API_BASE_URL}/orgs/{TAO_ORG_NAME}/jobs/{job_id}`).

Training backend identifier: `cosmos_rl_tao_vlm` (persisted on every TAOJob). Cosmos-RL exposes four task families via `tao-client cosmos-rl`: **train, evaluate, inference, quantize**. VLM fine-tuning only through TAO Toolkit API + `tao-client` (no launcher-based interface).

#### 9.7.2 TAOJob State Machine  `[EXACT]`

Canonical statuses: `not_started`, `submitting`, `submitted`, `queued`, `running`, `paused`, `succeeded`, `failed`, `canceled`, `deleted`.

- `not_started`: local state; pre-created chain jobs awaiting predecessor. No TAO-side equivalent.
- `submitting`: local state; HTTP POST to TAO in flight, no confirmed `tao_external_job_id` yet. On restart: `submitting` + `tao_external_job_id=null` → `failed` with `status_reason="submission_interrupted"` (§9.7.7).

Terminal statuses: `succeeded`, `failed`, `canceled`, `deleted`. Once terminal, MUST NOT transition to non-terminal.

Allowed transitions:

```
not_started -> submitting | failed | canceled
submitting  -> submitted | failed
submitted   -> queued | running | failed | canceled | succeeded   ← F9 (skip running)
queued      -> running | failed | canceled | succeeded            ← F9 (skip running)
running     -> succeeded | failed | paused | canceled
paused      -> running | canceled | succeeded                     ← F9 (skip running)
(any)       -> deleted  (only if system performs delete)
```

`not_started → failed` when `chain_halted_reason` is set (predecessor failed). `not_started → canceled` when chain canceled before start. `submitting → failed` on POST failure or backend restart before `tao_external_job_id` persisted.

**F9 (Phase 12 amendment 2026-05-05) — fast-completion races skip ``running``.** TAO can move through ``Queued → Running → Done`` faster than the polling cadence (30s for `submitted`, 60s for `running`) can observe — a 3-epoch SFT on a small dataset finishes in ~7 min. ``submitted/queued/paused → succeeded`` direct transitions are explicitly allowed so observed reality (TAO returns ``Done``) wins over an unobserved intermediate ``running``. ``not_started`` and ``submitting`` are NOT included — those predate any ``tao_external_job_id`` round-trip; succeeded without an external_id would be incoherent.

**Mapping TAO raw status → canonical (case-insensitive):**

- `Done` → `succeeded`
- `Failed` → `failed`
- `Running` → `running`
- `Queued` / `Pending` → `queued`
- `Paused` → `paused`
- `Canceled` / `Cancelled` → `canceled`

Unknown raw status: set `running` if metadata indicates active progress; `queued` if not started; else `running` as conservative non-terminal default. `tao_status_raw` persisted alongside canonical.

#### 9.7.3 TAOJob Configuration Payload Structure  `[SUMMARY]`

TAOJob persists two configuration objects:

1. `job_config` — high-level config used by this system (human-meaningful). Required fields: `project_id`, `training_backend="cosmos_rl_tao_vlm"`, `student_base_model_config_id`, `dataset_export_ids[]`, `guidance_id`, `training_preset`, `training_policy_type="sft"`, `lora_config` (§9.7.3.2), `hyperparameters` (opaque TAO specs patch; may be `{}`), `dataset_refs`, `intended_outputs`, `tao_release_version`, `cosmos_rl_container_tag`. Cosmos-RL resolved fields (persisted in `resolved_training_fields`): `policy.model_name_or_path`, `policy.model_max_length`, `policy.model_gradient_checkpointing`, `policy.parallelism.*` (TP/PP/DP), `train.ckpt.*`, `train.train_policy.*`, `validation.*`, `results_dir`. Parallelism: `parallelism_config`, `num_nodes`, `num_gpus_per_node`, `redis_config`.
2. `tao_create_job_request` — exact payload submitted to TAO (opaque JSON) for reproducibility/audit. **Required top-level fields** (FTMS 6.26.3, live-verified): `kind="experiment"`, `action`, `name`, `network_arch="cosmos-rl"`, `workspace`, `base_experiment_ids`, `timeout_minutes` (default 1440; configured by `TAO_JOB_TIMEOUT_MINUTES`), `specs`, action-specific dataset binding (§9.3.3). Training preflight fails closed unless TAO's v2 `ExperimentJobReq` declares the timeout field (install guide §13.15). For chain-advanced jobs (evaluate, quantize): top-level `parent_job_id` (§9.7.6). For gated base experiments: `docker_env_vars.HF_TOKEN` (§9.7.3.4). **Fields that MUST be omitted:** `policy.parallelism` and `num_gpu` except explicit per-model large-tier overrides (TAO otherwise auto-injects `tp_size=1, dp_shard_size=NUM_GPU_PER_NODE`).

**Hyperparameter resolution:** MUST resolve `tao_create_job_request.specs` by starting from TAO default `specs` for the workflow/action/base experiment, then applying `job_config.hyperparameters` as a deterministic patch. `{}` means "use TAO defaults".

See §9.7.3 for full field list.

#### 9.7.3.1 Training Preset System  `[SUMMARY]`

Single dropdown **Training Intensity**: **Quick** (validation default),
**Standard** (production-candidate comparison default), **High Quality**,
**Max Quality**. Power-user "Advanced" accordion shows resolved patch JSON
read-only.

Stored on `job_config`: `training_preset` (user-facing, auditable) + `hyperparameters` (actual patch, reproducible, deterministic).

Preset targets Cosmos-RL fields: `train.epoch`, `train.resume`, `train.ckpt.enable_checkpoint`, `train.ckpt.save_freq_in_epoch`, `train.ckpt.max_keep`, `train.ckpt.export_safetensors`.

Epoch calibration (based on NVIDIA's Cosmos-Reason2 examples: default 1 epoch; Cookbook 8B recipe: 3 epochs; per-epoch checkpointing with best_model selection):

- **Quick**: 1 epoch. "Fast run for a quick signal."
- **Standard**: 3 epochs. "Recommended default."
- **High Quality**: 9 epochs. "More training for better accuracy."
- **Max Quality**: 18 epochs. "Large-epoch training for best quality (slowest)."

All include `resume=false`, `enable_checkpoint=true`, `save_freq_in_epoch=1`, `export_safetensors=true`, **`max_keep=1` when `resume=false`** (F27 amendment 2026-05-05 — was 8; the Blueprint selector only fetches the latest epoch so additional retained epochs uploaded ~8× more bytes to workspace S3 with no Blueprint-visible benefit; live-surfaced as 117.8 GB / ~50 min upload on an 8B high_quality train). Future `resume=true` flows MUST emit `max_keep=8` so cosmos-rl's resume machinery has fallback checkpoints. **F24 amendment 2026-05-05** — epoch counts above are calibrated for Cookbook-scale data (≥1000 examples); for ≤500-example projects pick `max_quality` (24ep on 2B / 18ep on 8B) as the first attempt; Quick/Standard sanity-check the pipeline but produce non-converged Students.

Optional model-aware mapping (internal policy): 2B may use higher epochs (Quick=1, Standard=3, High=12, Max=24); 8B lower (High=9, Max=18). Deterministic lookup by `student_base_model_config_id`.

Resume semantics: `train.resume=true` resumes from latest checkpoint (restores weights, optimizer state, training progress). Continuation, not a "new run with changed knobs".

See §9.7.3.1 for backend resolution flow.

#### 9.7.3.2 LoRA Configuration  `[SUMMARY]`

Every TAOJob MUST persist `lora_config` in `job_config` with at minimum:

- `enable_lora: boolean` (required; default true)
- `lora_rank: int`, `lora_alpha: int`, `lora_dropout: float`
- `lora_target_modules: string[]` (e.g., attention projection modules)
- `modules_to_save: string[] | null` (e.g., `["visual"]` for full vision-side fine-tuning)

Notes:

- `modules_to_save: ["visual"]` fully fine-tunes vision encoder. Compatibility constraints between some attention target modules and saving `visual` exist; implementations validate before submission.
- If LoRA is used and quantization is requested, adapter MUST be merged before quantization (TAO requires `base_model_path` when `enable_lora=true` during `quantize`; §9.8).
- LoRA config persisted as part of student lineage.
- **NIM serving contract:** VLM NIM serves merged checkpoints only; runtime LoRA adapter loading is documented for LLM NIM but NOT for VLM NIM. Ensure adapter merge before any NIM deployment (§9.5.1).

#### 9.7.3.3 Cosmos-RL SFT Spec Overrides (Required for train + evaluate; F11 amendment for quantize)  `[EXACT]`

Every cosmos-rl `train` and `evaluate` action MUST emit these spec overrides on top of the preset patch (live-verified 2026-04-29; each prevents a specific cosmos-rl 6.26.3 worker failure):

- `train.compile=false` — torch.compile unsupported for HFModel; default true crashes the worker with AssertionError.
- `train.train_policy.type="sft"` — cosmos-rl defaults to GRPO; missing this exits the SFT helper silently.
- `train.train_policy.dataloader_drop_last=false` — small datasets otherwise yield zero full batches and crash checkpoint at step 0 on `NoneType.state_dict`.
- `train.ckpt.save_mode="sync"` — async save races with step-0 path on small datasets.
- `validation.enable=false` — cosmos-rl asserts on missing `val_dataset` when validation is on; the chain emits a separate evaluate job (§9.7.6).

**F11 (Phase 12 amendment 2026-05-05): the `quantize` action MUST NOT emit any of these overrides.** cosmos-rl-quantize's argparse CLI rejects `--compile`, `--type`, `--dataloader_drop_last`, `--save_mode`, `--enable` as unrecognized. Quantize spec is minimal: `specs.quantization_scheme` (NOT `quantization_method`; cosmos-rl-quantize's CLI flag is `--quantization_scheme`) plus `specs.dataset.{media_dir, annotation_path}` per §9.3.3, with `policy.model_name_or_path` omitted (`--model_path` is auto-injected from `parent_tao_job_id`). `job_config.quantization_method` (the persisted internal field) is unchanged — it aligns with `StudentModel.quantization_method`; only the wire-level spec key is renamed. **Residual gap is operator-side, not upstream cosmos-rl**: the parallelism args (`--n_init_replicas/--tp_size/--cp_size/--dp_shard_size/--dp_replicate_size/--pp_size`) come from a deployment-time custom patch (`cosmosrl_parallelism_defaults.py`, applied to TAO-FTMS for Issue #8 to give `train` explicit parallelism defaults). cosmos-rl-quantize is single-process and has no parallelism arguments. The fix is operator-side: make the patch action-aware so parallelism defaults are injected only for `train`. With the Blueprint-side F11 emitting a correct quantize spec, no further Blueprint changes are needed.

Apply via a single helper after the preset patch is computed and BEFORE checksum (`services.training_suite_service._build_train_payload`). The quantize path is `_build_quantize_payload` (no SFT overrides).

#### 9.7.3.4 Environment Variables (`docker_env_vars`)  `[EXACT]`

`docker_env_vars.HF_TOKEN` MUST be injected on every chain submission. Even when the base experiment is registered via `:load_airgapped`, the cosmos-rl worker's `transformers.from_pretrained()` still authenticates against `huggingface.co` for gated-repo verification — without `HF_TOKEN` the worker hits HTTP 401 and the job retry-loops on "config.json not found". TAO FTMS 6.26.3 only accepts `HF_TOKEN` (NOT `HF_HOME`/`HUGGING_FACE_HUB_TOKEN`/`HUGGINGFACE_TOKEN`/`HF_HUB_TOKEN`/`HF_ACCESS_TOKEN` — those are rejected as `Invalid enum member`). Token is sourced from the Blueprint's `HF_TOKEN` setting; `services.tao_job_service._submit_to_tao` injects it uniformly (train, evaluate, quantize). The diagnostic POST log redacts the token to `***`.

#### 9.7.4 Polling Contract  `[SUMMARY]`

Backend polls TAO for TAOJob records in non-terminal statuses. Updates: `status`, `tao_status_raw`, `progress`, `started_at`/`completed_at`, `last_polled_at`, `error` fields, `outputs` refs. Polling APIs (TAO v2; live-verified against FTMS 6.26.3):

- Job metadata/status: `GET /api/v2/orgs/{org_name}/jobs/{job_id}`
- Logs: `GET /api/v2/orgs/{org_name}/jobs/{job_id}:logs`
- List files: `GET /api/v2/orgs/{org_name}/jobs/{job_id}:list_files` (returns JSON array of workspace-relative keys)
- Artifact retrieval: see §9.7.5. Direct boto3 `GetObject` against the workspace S3 bucket using keys from `:list_files`. FTMS 6.26.3's `:download_selective_files` is unsuitable for cosmos-rl outputs (POST → 405; GET requires `best_model`/`latest_model` aliases that cosmos-rl does not produce).

**Status mapping (case-insensitive):** `Done → succeeded`, `Failed → failed`, `Error`/`Errored → failed` (FTMS 6.26.3 surfaces container-level worker failures as `"Error"`, not `"Failed"`; without this mapping the polling loop never detects the terminal state), `Running → running`, `Queued`/`Pending → queued`, `Paused → paused`, `Canceled`/`Cancelled → canceled`.

**Frequency (deterministic policy):** `submitted/queued` 30–60s; `running/paused` 60–180s; terminal — poll once to finalize outputs then stop. Rate-limited per job.

**On-demand refresh:** `GET /v1/projects/{project_id}/tao_jobs/{tao_job_id}?refresh=true` forces a poll-update (rate-limited).

#### 9.7.5 Required Outputs to Track  `[SUMMARY]`

TAOJob MUST track by reference. Cosmos-RL writes its outputs directly into the workspace S3 bucket; the Blueprint reads from there (live-verified 2026-04-29):

1. **Artifacts:** action-specific layout — train and quantize emit DIFFERENT shapes. **`train`** → nested merged HF checkpoint at `results/<tao_external_job_id>/<timestamp>/safetensors/epoch_<N>/{config.json, *.safetensors, tokenizer.json, …}`; the Blueprint mirrors the latest epoch directory flat into `{project_dir}/artifacts/tao_jobs/{tao_job_id}/`. **`quantize`** → **F31 amendment 2026-05-05**: emits the merged HF checkpoint FLAT at the job root `results/<tao_external_job_id>/{config.json, model-XXXXX-of-NNNNN.safetensors, recipe.yaml, tokenizer.json, …}` (no `<timestamp>/safetensors/epoch_<N>/` nesting because quantize is single-shot). `_select_hf_checkpoint_keys` MUST recognize both shapes — train wins when both appear, quantize is the fallback when no `safetensors/epoch_<N>/` matches. Live-surfaced on a 8B FP8 quantize (72db4400) where the artifacts were present from TAO Done time but the train-only glob silently returned empty, exhausting F25's full retry budget needlessly. **`evaluate`** → single tarball at `results/<tao_external_job_id>/evaluate_results.tar.gz`; downloaded + extracted under the cache dir, per-sample predictions translated to a synthesized `per_sample_predictions` JSON file (see §9.7.5.1). Retrieval: `GET :list_files` enumerates keys → boto3 `GetObject` against the workspace's external S3 endpoint. Plus exact training config (`tao_create_job_request` + resolved specs). **F25 (Phase 12 amendment 2026-05-05):** TAO can mark a job `Done` before SeaweedFS finalizes the upload; the first `:list_files` after status-flip can return zero matching artifacts. `_fetch_tao_artifacts` MUST retry the listing-and-action-aware-selection step with bounded backoff when caller passed a non-null `local_cache_dir` and selection returned empty. **F28 (Phase 12 amendment 2026-05-05):** the schedule is widened to `[10, 20, 40, 60, 90, 150, 240, 360]`s (~970s ≈ 16 min). Metadata-only callers (no `local_cache_dir`) MUST NOT retry. **F32 (Phase 12 amendment 2026-05-05):** the retry trigger narrows to GENUINELY-EMPTY listings (`len(keys) == 0`) only. Non-empty listings without checkpoint shards (e.g., `status.json` + `microservices_log.txt` only) are terminal — cosmos-rl finished, emitted completion metadata, and never wrote checkpoint shards; retrying burns the full ~970s budget for no benefit and hangs unit-test runners. Live cases F28 was widened for (ce4d95da, 8a379810) were both empty listings, so the narrowing preserves F28's intent on the real-world race signature.
2. **Logs:** `logs_ref` (stored in system) or `tao_logs_ref` (snapshot of `GET .../jobs/{id}:logs`).
3. **Metrics / Progress:** `epoch_current`, `epoch_total`, `eta_seconds`, `metrics_latest` (object), `metrics_history_ref`. All nullable.

Completed locally retrieved outputs remain internal inputs to evaluation,
recovery, Student NIM validation, and portable NIM deployment bundle
construction. Raw TAO files have no standalone public download endpoint; the
gated portable NIM deployment bundle is the single trained-model delivery
artifact.

#### 9.7.5.1 Evaluate Results Translation (Required)  `[EXACT]`

For evaluate jobs, the cosmos-rl tarball is extracted under `{project_dir}/artifacts/tao_jobs/{tao_job_id}/`. Per-sample prediction files at `<prefix>/freeform/<eval_set>/images/<key>.json` carry cosmos-rl's `{video_id, correct_answer, answer, reasoning, full_response}` shape. The `<prefix>` is `epoch_<N>` for train-checkpoint evaluates and the `<parent_external_id>` for quantize-checkpoint evaluates (**F29 amendment 2026-05-05**: cosmos-rl-evaluate of a quantized checkpoint emits `<parent_external_id>/freeform/...` rather than `epoch_<N>/freeform/...`; the synthesizer's glob MUST match both shapes — old hardcoded `epoch_*/*/*/images/*.json` produced empty `per_sample_predictions` and a rescore C2 on every quantized-model eval). The polling service synthesizes a single canonical `per_sample_predictions` JSON file in the rescoring-service-expected `{id, prediction, …}` shape:

- `video_id` is normalized to the bare `example_key` by stripping the `images/` directory prefix and the file extension (e.g. `images/smoke_1.jpg` → `smoke_1`) so it pairs with `dataset_export_service`'s ground-truth annotation `id` field.
- `answer` → `prediction`.
- Auxiliary fields (`video_id`, `datasource`, `correct_answer`, `reasoning`, `full_response`) preserved for debugging.

Canonical implementation: `services.tao_polling_service._materialize_evaluate_predictions`.

**F10 (Phase 12 amendment 2026-05-05) — `prediction` field MAY carry a markdown-fenced JSON wrapper.** cosmos-rl 6.26.3's vLLM-served `cosmos-rl-evaluate` does NOT enable structured generation; Cosmos-Reason2 emits its instruction-tuned default — JSON wrapped in a markdown code fence (e.g. ```` ```json\n{...}\n``` ````). Implementations of the rescoring service MUST strip a leading/trailing markdown code-fence wrapper before `json.loads`. Canonical implementation: `services.tao_rescoring_service._strip_markdown_fences`; accepts both fenced and bare shapes, idempotent. Without stripping, every prediction parses as schema-invalid → run C2 (zero parseable predictions). Live-validated 2026-05-05 (pre-fix 0/84 parseable; post-fix 84/84 parseable, 28/84 exact-match).

#### 9.7.6 Automatic Post-Training and Post-Quantization Evaluation  `[EXACT]`

When a TAO training job reaches `succeeded`, the system MUST automatically submit a TAO `evaluate` job against the Test Pool export. When a TAO `quantize` job reaches `succeeded`, the system MUST automatically submit a TAO `evaluate` job against the quantized checkpoint. No SME interaction is required for these submissions.

**Test Pool export for TAO evaluation:**

The system MUST export the reserved Test Pool as a distinct Cosmos-RL evaluation dataset (§9.3) and MUST exclude those examples from all training exports. TAO evaluation operates only on the explicitly provided `annotation_path` and `media_dir`; it does not infer pool membership from system records.

Export rules:

- Verified non-pool examples → training export
- Reserved Test Pool examples → evaluation export only
- These MUST never be mixed

The evaluation dataset export MUST use the same Cosmos-RL LLaVA format as training exports (§9.3). The `conversations` human turn contains the task prompt; the gpt turn contains the ground-truth label JSON string. TAO reads the gpt turn as the expected answer. The evaluation export MUST use the same `export_field_mode` as the training exports in the same chain — scoring a `core_only`-trained Student against `all`-field ground truth would produce systematic mismatch.

**TAO evaluate configuration (live-verified against FTMS 6.26.3 + cosmos-rl 6.26.3, 2026-04-29):**

- `action: "evaluate"`.
- Top-level `parent_job_id` (REQUIRED for chain advancement) → predecessor's `tao_external_job_id`. FTMS's `infer_parent_model_folder` resolves the trained-checkpoint folder onto cosmos-rl's `model.model_name` from this field. MUST be at the top of the POST body (NOT nested under `specs`); without it the worker crashes in `cosmos_rl/evaluation/base.py` with `TypeError: ... NoneType` because `model.model_name` is None.
- `specs.dataset.{media_dir, annotation_path}` (top-level `dataset` shape, not `custom.val_dataset` — see §9.3.3).
- `specs.policy.model_name_or_path` MUST be OMITTED. cosmos-rl reads it first and only falls back to FTMS-injected `model.model_name` when absent; explicit values override `parent_job_id` resolution and force eval to load the BASE experiment instead of the trained student.
- §9.7.3.3 SFT spec overrides still apply (`compile=false`, `train_policy.type=sft`, `dataloader_drop_last=false`, `ckpt.save_mode=sync`, `validation.enable=false`).
- TAO uploads results as a single tarball at `results/<eval_external_job_id>/evaluate_results.tar.gz`; see §9.7.5.1 for the tarball-extract + per-sample translation contract.

**Re-scoring with canonical evaluator:**

TAO's native aggregate metrics may not match the system's exact Core-field normalization semantics. The system MUST:

1. Retrieve TAO's per-sample prediction outputs after the evaluate job succeeds.
2. Parse each prediction as a JSON label.
3. Re-score every prediction against the Test Pool ground truth using the canonical Exact Match evaluator (Appendix A.2).
4. Produce authoritative metrics: overall Exact Match rate, per-core-field match rate, per-value precision/recall/F1.
5. Persist both the TAO-native metrics (informational) and the re-scored metrics (authoritative) on the evaluation Run Record.

The frozen evaluation export is the scoring population. Before a TAO quality
RunRecord is created or a Student is validated, the archive MUST match its
recorded SHA-256, contain exactly `DatasetExport.example_count` valid
ground-truth rows with each key appearing once, and provide exactly one
materialized prediction for every ground-truth key. Missing or malformed
digests, unreadable archives, count/key gaps, or duplicate ground-truth or
frozen prediction keys take C2 rather than shrinking the denominator or
selecting by artifact order; outside prediction keys are ignored, while
present schema-invalid predictions remain zero-match denominator entries.

TAO's aggregate metrics are useful for quick directional checks; the system's re-scored metrics are the authoritative source for quality comparison and gate decisions.

**F33 (Phase 12 amendment 2026-05-05) — operator-driven re-rescore.** When the rescoring service is amended (e.g., F10 markdown-fence stripping, F29 quantize-parent prefix glob), Students whose original rescore returned a C2 path under the old code remain `quality_status="failed"` even though the on-disk per-sample predictions would now parse cleanly. New endpoint `POST /v1/projects/{project_id}/student_models/{student_model_id}:rerescore` (§10.2.20) replays the canonical rescore against the existing evaluate TAOJob. Safety guard: `quality_status` MUST be `"failed"` (HTTP 409 otherwise) — F33 never overwrites `"validated"`, `"pending"`, or (after F35) `"partial"`. The endpoint locates the paired evaluate TAOJob via `parent_tao_job_id` from the Student's `quantize_tao_job_id` (quantized variants) or `tao_job_id` (baselines), then invokes `tao_rescoring_service.rescore_evaluate_job`. Idempotent against TAO predictions — F33 does NOT re-fetch artifacts from workspace S3. **F33 + F35 interaction:** `partial` Students cannot use `:rerescore` because `partial` is set by NIM-eval (no on-disk TAO rescore to replay); the remediation path is to re-run NIM eval until it lands `completed`.

#### 9.7.7 Job Chaining  `[EXACT]`

The post-training workflow is a deterministic chain of TAO jobs. The chain is defined at "Start Training" time based on the selected base models and quantization schemes. All jobs in a chain share a `chain_id` and are sequenced by `chain_sequence`.

**Chain structure per base model:**

```
1. train (action: "train")
2. evaluate baseline (action: "evaluate", parent: train job)
3. quantize scheme A (action: "quantize", parent: train job)
4. evaluate scheme A (action: "evaluate", parent: quantize job)
5. quantize scheme B (action: "quantize", parent: train job)  [if selected]
6. evaluate scheme B (action: "evaluate", parent: quantize job)  [if selected]
...
```

**Chaining rules:**

- Each job in the chain is a distinct TAOJob record with its own `tao_job_id`. `chain_id` groups them; `chain_sequence` orders them for display.
- At "Start Training" time, the system pre-creates TAOJob records for every job in every chain with `status=not_started` (§9.7.2). Only the first job in each chain is immediately submitted. The full chain definition is persisted so the UI can show expected vs completed jobs.
- **Eligibility (chain isolation, amended 2026-05-04):** advancement is by **dependency satisfaction**, not chain_sequence ordering. A `not_started` job is eligible when its `parent_tao_job_id` is null (chain root) OR points at a `succeeded` predecessor. Among eligible jobs, the lowest `chain_sequence` is submitted first.
- **Halt scope on failure:** when a job reaches `failed` / `canceled`, the system marks as halted **only the transitive dependents** of the failed job. Independent siblings whose parent is a different (still-`succeeded`) job remain `not_started` and become eligible. Concretely: a failed `evaluate baseline` does NOT halt `quantize FP8 / W8A16` (those parent on the still-succeeded `train`). A failed `train` DOES halt every job (everything depends on it transitively). This makes chain advancement DAG-correct rather than sequence-conservative — quantize jobs proceed when sibling evaluate fails for an upstream-loader reason that doesn't affect the trained checkpoint (§9.5 NIM-eval-as-quality-fallback validates the resulting quantized Students). Implementation: `services/tao_polling_service._transitive_dependents_in_session` + `_find_next_eligible_in_session`.
- Chains for different base models run sequentially (one model's full chain completes before the next begins).
- System-generated chained TAO jobs MUST omit `force_create`.

**Submission protocol:**

1. Transition the TAOJob to `submitting`.
2. Send the TAO create-job POST.
3. On success: persist `tao_external_job_id` from the response and transition to `submitted`.
4. On failure: transition to `failed` with `error_ref` describing the submission error.

**Submission interrupted recovery:** if the backend crashes between steps 1 and 3, the TAOJob remains in `submitting` with `tao_external_job_id = null`. On restart, these jobs transition to `failed` with `status_reason="submission_interrupted"`. The UI MUST allow the SME to retry the failed step or restart the chain from the failed step.

**Chain advancement from persisted state:** the system MUST be able to derive the next required submission from persisted chain state after restart (§1.3). On scan, for each chain the system identifies the lowest-`chain_sequence` `not_started` job whose dependency is satisfied (parent is null or `succeeded`) and submits it — the chain-isolation eligibility rule applies on restart too.

#### 9.7.8 TAO Provisioning Model  `[SUMMARY]`

Stock TAO FTMS deployments require three resources before any training suite can execute: a **workspace** (S3-backed storage), one **base experiment per trainable student base** (Cosmos Reason2 8B/2B, Cosmos 3 Nano/Super reasoner — registered in the workspace), and the **training / Test Pool dataset archives uploaded** into the workspace's S3 bucket. A fresh FTMS instance has none of these.

**Two provisioning paths** (same end state: populated workspace with base experiments keyed by `tao_base_experiment_id` on each `student_base` ModelConfig):

**Self-service (§9.7.8.1a, DEFAULT, Blueprint-driven):** when the Blueprint host has outbound access to Hugging Face (or mirror) and workspace S3. The normal SME path selects any missing catalog base and clicks **Start Training**; the Blueprint creates a durable Training Suite, shows one conditional **Provision Student Bases** stage in Training Jobs, performs all selected missing pulls in one background run, and then prepares datasets and job chains automatically. The operator CLI remains available for eager provisioning. The helper invokes `nvidia-tao-core.microservices.pretrained_models` out-of-process in an isolated `uv` environment with `AIRGAPPED_MODE=true`, uploads to `s3://{bucket}/shared-storage/models/`, calls `POST /orgs/{org}/jobs:load_airgapped`, and patches `ModelConfig.tao_base_experiment_id` across every project database. Each staged checkpoint file is hash/size/upload-bound to one descriptor opened beneath the disposable stage root, and the exact uploaded byte stream must retain that hash. The FastAPI process never imports the packaging stack.

**Admin handoff (§9.7.8.1b, FALLBACK):** for air-gapped or policy-separated deployments. Admin uses `nvidia-tao-core` CLI with NGC-TAO registry credentials or a hand-carried airgapped bundle off-Blueprint. Resulting base-experiment UUIDs communicated back to Blueprint operator, who patches via `vlm-feedback-loop tao-bootstrap --admin-managed --base-experiment-id-2b=<UUID> --base-experiment-id-8b=<UUID>`.

The diagnostics endpoint (§10.2.22) reports the same readiness checks that
**Start Training** enforces internally and is rendered on both the Scale-Up Hub
and Student Training screen.

**Per-training dataset upload (§9.7.8.2, Blueprint-automated):** persist or adopt a durable `preparing` TrainingSuite before export work, link each completed training + Test Pool DatasetExport before upload, and obtain its archive and sidecar only from `artifact_refs`. Open both as regular files beneath the owning project's `exports/` root, then use those same descriptors to verify the recorded archive SHA-256, parse the root archive `annotations.json`, and require exact type-sensitive JSON equality with the sidecar before any S3 request. Rehash the exact bytes sent, rejecting changed single-PUT bodies before dispatch and aborting changed multipart streams before completion. Upload the pair to distinct keys under `s3://{bucket}/vlm-feedback-loop/projects/{project_id}/dataset_exports/{dataset_export_id}/`; use multipart above 8 MiB, reuse each object only when its hash metadata matches, and persist lineage only after both succeed. A failed pre-chain transfer may resume only under the identical request and idempotency key, reusing the linked export IDs and repairing only missing or mismatched objects; an integrity failure requires corrected fresh exports and a new key. TAOJob specs reference those remote archive/JSON paths, never the Blueprint host filesystem. Ready-base synchronous failures return 409 `tao_dataset_upload_failed`; failures after a provisional first-use suite was returned set the suite to `failed` and expose `setup_error_ref` through GET.

**Preflight additions (§9.7.8.3):** `tao_workspace_reachable` (bootstrap_status=bootstrapped AND `GET /orgs/{org}/workspaces/{workspace_id}` returns 200), `tao_base_experiment_ready` (non-null UUID AND pull status `pull_complete`), `verified_train_examples` (at least one active-Guidance non-pool Verified example), and `min_test_pool_size` (active-Guidance Test Pool count ≥ `max(1, project.scaleup_min_test_pool_size)`). Dataset failures direct the SME to labeling and never generate a TAO setup request. Suite materialization repeats both dataset checks after any base provisioning and rejects an undersized final export before upload or TAO job creation.

See §9.7.8 for full details.

### 9.8 Quantization Workflow (Post-Training, Two Lanes)  `[SUMMARY]`

Quantization is an explicit post-training stage. Uses TAO Cosmos-RL `quantize` action exclusively. Full Student lifecycle:

`fine_tune → merge_and_evaluate_baseline_via_student_nim → quantize(optional) → evaluate_quantized_via_tao → package_for_nim → deploy`

For a LoRA chain, TAO evaluate cannot consume the adapter-only post-train
checkpoint. The Blueprint merges the adapter locally, serves the merged
full-precision checkpoint through the temporary Student NIM, evaluates the
frozen Test Pool, and records that Run on the baseline job. Quantized
evaluation remains TAO-native because TAO quantize already performs the merge.
The training readiness gate therefore requires both `HF_TOKEN` and the local
merge runtime whenever LoRA is enabled.

**Default quantization:** FP8_DYNAMIC only for Validate training setup,
producing baseline + FP8. W4A16, W8A8, and W8A16 are explicit additional
comparison choices. Each scheme is a separate TAO `quantize` job.

**Cosmos 3 Super compatibility:** Super remains baseline-only at the backend
API layer. Full-weight training, packaging, local NIM evaluation/benchmark,
and handoff are supported; every post-training quantization scheme is blocked
by the backend-owned `quantization_compatible` check after the qualified
FP8_DYNAMIC action failed on a CPU-offloaded meta-device weight. The Training
UI excludes Super because its LoRA path is incompatible and Full-weight has not
been qualified across the UI-supported Student matrix.

**§9.8.1 TAO `quantize` action:** documented methods FP8_DYNAMIC (default), W8A8, W8A16, W4A16. Spec structure: `model` (trained or merged LoRA checkpoint), `calibration_dataset`, `quantization_method`. Rules:

- If `enable_lora=true` on training job, `base_model_path` required so TAO merges adapter before quantization.
- MUST be separate TAOJob record with `action: "quantize"` + `parent_tao_job_id` linkage.
- Accuracy verification automatic via TAO `evaluate` after each quantize succeeds (§9.7.6, §9.7.7).
- TAO `quantize` produces quantized checkpoint artifact, NOT a packaged NIM profile. Must pass §9.5.1 packaging step before registration.
- Persist `quantization_method` on Student variant record (`FP8_DYNAMIC`, `W4A16`, or `none` for baseline).

---

## 10. Architecture & Interfaces

### 10.1 Components (Conceptual)  `[EXACT]`

- **Frontend** (React + TypeScript SPA) ↔ **Backend** (Python + FastAPI; REST + SSE) ↔ NIM endpoints (Teacher, Student, embedding NIM)
- **Domain store** (SQLite per project): projects, examples, labels, guidance, pools, lineage, operation records, run records, model catalog, dataset exports, TAO job records, audit events
- **Background task executor** (in-process asyncio): evaluations, CLIP embedding computation, Batch Labeling
- Optional adjacent systems: Cosmos-RL / TAO VLM (TAO Toolkit)

### 10.2 Key Interface Requirements  `[REFERENCE]`

All endpoints are defined in §10.2.1–§10.2.26 of `Engineering_Spec.md`. All non-project resources MUST be accessed in a project scope; non-project-scoped endpoints are the environment assessment (§10.2.25) and filesystem browse (§10.2.10). All APIs MUST enforce project scoping; cross-project IDs MUST be rejected. Endpoint groups:

- **Projects:** `POST /v1/projects`, `GET /v1/projects/{project_id}`, `PATCH /v1/projects/{project_id}`, `GET /v1/projects` (list with summary counts; optional `include_archived: bool = false`; every response carries workspace-global `has_archived: bool` from a marker-file scan so the UI's "Show archived" affordance needs no archived-inclusive fetch) — §10.2.13, §13.4.2. Soft-archive verbs `POST /v1/projects/{id}:archive` and `:unarchive` with a busy gate (non-terminal RunRecord/TAOJob/LocalNimDeployment → 409 `project_busy`), `require_not_archived` cross-mutation guard on PATCH + start-runs endpoints, and `.archived` marker file as a lazy index for list and worker scans (drift against the `archived_at` column self-heals: the column wins and the marker is rewritten/removed) — §10.2.13.1
- **Background task SSE stream:** `GET /v1/projects/{project_id}/events` — event types listed in §1.3 + new types added by later phases (open event type dispatcher)
- **Image ingestion + filesystem:** `POST /v1/projects/{project_id}/examples:ingest` (§10.2.1, 202 skeleton response, partial success, idempotent re-ingest, structural format validation, restartable pHash/CLIP background work); `GET /v1/filesystem/browse` + `POST /v1/filesystem/scan` (§10.2.10, one `IMAGE_ROOT`, omitted browse path opens that root, deterministic `suggested_example_key`; **F47 amendment 2026-05-15** — scan is no longer a user-facing preview step, only an internal helper called by [Ingest Selected] to expand a checked directory into the per-image `(storage_ref, suggested_example_key)` rows that ingest needs; the response's `skipped[]` array is not rendered to the SME); `POST /v1/projects/{project_id}/examples:remap_paths` (§10.2.11, dry-run default, validation, AuditEvent); `GET /v1/projects/{project_id}/examples/{example_key}/image` (§10.2.9, streams from `storage_ref`, never accepts arbitrary paths); `GET /v1/projects/{project_id}/examples` (§10.2.8, cursor pagination, state/verified/pool_membership filters)
- **Guidance:** `POST/GET/LIST /v1/projects/{project_id}/guidance` (§10.2.2, immutable versioning, monotonic `version_number`); `POST /v1/projects/{project_id}/guidance:validate_draft` (§10.2.2, same canonical `validate_and_derive` used by save); `GET /v1/projects/{project_id}/guidance:icl_count` (§10.2.5, `{"eligible_count": int}` — non-pool Verified Edits under the active Guidance, feeds the Batch pre-run ICL row)
- **Interactive proposal:** `POST /v1/projects/{project_id}/proposals` (§10.2.3, overrides, `retry_of_inference_invocation_id`, `use_existing_label` for Auto-Labeled)
- **Review selector:** `GET /v1/projects/{project_id}/review_selector/next` (§10.2.4)
- **Labels:** `POST /v1/projects/{project_id}/labels` (§10.2.14, backend computes deterministic diff, stale-proposal 409, conditional `rationale_source` validation, Auto-Labeled → Verified promotion); `POST /v1/projects/{project_id}/examples/{example_key}:skip` (atomically discards any machine Label); `POST /v1/projects/{project_id}/examples:restore_omitted` (restores clean Unlabeled state)
- **Rationale regeneration:** `POST /v1/projects/{project_id}/examples/{example_key}:regenerate_rationale` (§10.2.15, Appendix D.3; 409 while disabled)
- **Evaluation:** `POST/GET/LIST /v1/projects/{project_id}/evaluation_runs` + `:cancel` (§10.2.16, state machine §13.2.1)
- **Scale-Up Gate:** `GET /v1/projects/{project_id}/scaleup_gate` (§10.2.17)
- **Batch Labeling:** `POST/GET/LIST /v1/projects/{project_id}/batch_label_runs` + `:resume` + `:cancel` (§10.2.6, state machine §13.2.2); `GET .../batch_label_runs/{run_id}/schema_invalid_manifest` (§10.2.24)
- **Dataset exports:** `POST/GET/LIST /v1/projects/{project_id}/dataset_exports` (§10.2.18)
- **Model catalog:** `POST/GET/LIST/PATCH /v1/projects/{project_id}/model_configs` (§10.2.19, role filtering; **F22 amendment 2026-05-05** — GET response exposes `tao_base_experiment_id` and `tao_base_experiment_pull_status` as read-only fields so operators can verify TAO provisioning state without direct sqlite3 access; PATCH continues to reject both as `extra_forbidden`); `POST .../model_configs/{model_config_id}:reprobe` (§10.2.12, 409 while referenced by a queued/running/canceling run or active training job; paused Batch runs use their snapshot)
- **Student models:** `GET .../student_models` + `.../{id}` (§10.2.20; includes nullable immutable `training_suite_id`); `POST .../student_models/{id}:deploy_nim` (local Tier 1 or external Tier 2)
- **Training readiness and counts:** `POST
  /v1/projects/{project_id}/training_preflight` (§10.2.22) is rendered by
  Scale-Up and Student Training. It returns the launch checks plus
  authoritative Verified Training Pool, Test Pool, effective required Test
  Pool minimum, Auto-Labeled eligible/included, excluded, and final usable
  counts. The request includes the exact `quantization_schemes`; the response
  fails `quantization_compatible` for Cosmos 3 Super plus any scheme while
  allowing its empty-scheme, baseline-only path.
- **TAO jobs:** `POST/GET/LIST /v1/projects/{project_id}/tao_jobs` (§10.2.7); raw outputs remain internal and have no standalone download endpoint; `POST .../tao_jobs/{tao_job_id}:cancel?force_local=<bool>` (transitions to `canceled`, halts downstream `not_started` siblings with `chain_halted_reason`; **F17 amendment 2026-05-05** — `force_local=true` skips the TAO POST entirely so a job whose `tao_external_job_id` was orphaned by a TAO server rebuild can still cancel locally; the canceled row's `poll_error_ref` is stamped `forced_local_cancel: external TAO unreachable or external_id orphaned` for audit, and a `warning`-level log line is emitted on every use); training suite endpoint `POST/GET/LIST /v1/projects/{project_id}/training_suites` (Phase 11 Step 11.2, idempotency key; suite owner precedes export work, each export row and suite link commit atomically in a worker-thread session per §1.8 write discipline, and chain rows + suite initialization commit in one transaction). `POST .../training_suites/{training_suite_id}:cancel` terminalizes the suite first, best-effort cancels setup plus every remaining remote job, locally cancels all remaining jobs even when TAO cannot confirm, and returns the remote failures for operator warning; late workers MUST NOT reactivate the suite or advance a chain. **Training-suite response top-level keys** (canonical names — the lifecycle key is `status`, NOT `suite_status`): `training_suite_id`, `project_id`, `idempotency_key`, `guidance_id`, `training_preset`, `export_field_mode`, `include_auto_labeled`, `enable_lora`, `quantization_schemes`, `training_dataset_export_id`, `evaluation_dataset_export_id`, nullable `training_example_count`, nullable `evaluation_example_count`, nullable `evaluation_dataset_checksum_sha256`, `student_model_ids[]`, `selected_student_base_model_config_ids`, `chain_ids_ordered`, `chains[]` (each: `chain_id`, `student_base_model_config_id`, `base_model_name`, `jobs[]` — each: `tao_job_id`, `action`, `chain_sequence`, `status`, `tao_external_job_id`, `chain_halted_reason`), nullable `provisioning_run_id`, `provisioning_model_names[]`, nullable `setup_error_ref`, derived `setup_retryable`, `status`, `created_at`, `started_at`, `completed_at`.
- **Action requests:** `POST /v1/projects/{project_id}/action_requests:generate` (§10.2.23); `POST .../action_requests:log_copy`
- **Environment:** `GET /v1/environment` (§10.2.25, deployment-scoped)
- **NIM endpoints:** `POST/GET/LIST/PATCH /v1/projects/{project_id}/nim_endpoints` (§10.2.26; no standalone probe verb — endpoint health is auto-probed on create/update); `POST /v1/nim/test_connection` (transient connection test proxy; credential held in request memory only, never persisted)

### 10.3 Action Requests (Structured Handoff)  `[SUMMARY]`

**Purpose (§10.3.1):** when SME is blocked by an infrastructure dependency (self-hosted NIM needed, TAO not configured, NIM deployment preflight failed, etc.), system generates pre-filled, read-only message with technical requirements and current diagnostic context. SME copies to clipboard and pastes to organization's channel. NOT a ticketing system — no state tracking, no integration with external trackers.

**Content (§10.3.2):** `request_type`, `project_id`+`project_name`, `generated_at`, `technical_requirements` (type-specific), `current_environment` (relevant detected state). MUST NOT contain secrets.

**Request types (§10.3.3):**

- `tao_setup`: TAO endpoint or workspace setup is needed (Scale-Up Hub, training preflight); an ordinary missing base experiment uses the automatic Start Training path.
- `nim_setup`: self-hosted NIM endpoint needed (onboarding step 2).
- `nim_issue`: NIM endpoint unreachable (labeling screen proposal failure).
- `missing_files`: image files not found (labeling screen).
- `student_nim_deploy`: Student NIM deployment preflight failed (Compare & Deploy).
- `tao_issue`: TAO job failure (Training Job Monitor).
- `deployment_handoff`: Student production deployment (per-variant [Request Production Deployment] on `StudentVariantCard.tsx`). Distinct from `student_nim_deploy` above (per-variant [Deploy for serving validation], same card) — the "Production" qualifier separates the production-shipping intent from the temporary-evaluation NIM deployment the fallback path requests, so screenshots and audit logs disambiguate the two affordances. Rich content: `nim_checkpoint_ref`, NIM config (`NIM_MODEL_NAME`, `NIM_MODEL_PROFILE`, backend, release, env vars), model metadata (base, quantization, TP, GPU), evaluation snapshot (Exact Match, per-field, per-value P/R/F1 for categorical Core, latency p50/p90/p99, pool version, ICL mode, Guidance), training lineage (TAO job, quantize job, datasets, preset, LoRA config). **MUST return 409 if Student lacks both `quality_status=validated` AND `serving_status=validated`.**

**UX (§10.3.4):** inline CTAs at blocker screens (not in settings). Contextual labels ("Request TAO Setup", "Report NIM Issue", etc.). Pre-filled content with [Copy to Clipboard]. No SME form fields. After copy, AuditEvent `event_type="action_request_copied"` logged.

### 10.4 Prompt Package Export (Removed in v1.0)  `[SUMMARY]`

**Absent from v1.0.** A direct "deploy Teacher as-is" `.zip` recipe could not reproduce the live loop's per-query ICL example selection (§6.2), and one-NIM-per-GPU (§1.5) makes a standing deployed Teacher expensive. The public schema has no `prompt_packages` table. Full rationale in the main Spec §10.4.

## 11. Error Handling, Retries, and Idempotency  `[EXACT]`

**Deadlines**

- Every external invocation (Teacher/Student model calls; TAO API calls) MUST enforce a finite deadline.
- Default deadlines (configurable): interactive proposals `HTTP_DEADLINE_INTERACTIVE_S` (default: 180s — accommodates thinking-mode models whose reasoning phase adds ~60–90s), evaluation / batch labeling / embedding `HTTP_DEADLINE_BACKGROUND_S` (default: 300s).

**Retry classes (distinct semantics)**

1. **Automatic retry (system-initiated):**
  - MUST be bounded: at most `HTTP_MAX_RETRIES` attempts (default: 3).
  - MUST occur within a single `inference_invocation_id` attempt.
  - MUST NOT create additional Operation Records.
  - MAY include prompt budget fallback retries (diversity-based ICL pruning on context overflow, increasing `max_output_tokens` on truncation), but remains bounded and within the same invocation.
  - **Retryable errors:** `429`, `502`, `503`, `504`, transient connection failures. Exponential backoff with jitter (base intervals: 1s, 2s, 4s).
  - **Non-retryable errors:** `400`, `401`, `403`, `404`, `422`, schema-invalid outputs, deterministic validation failures. Fail immediately.
  - **Rate limiting:** on `429` from hosted NIM, backoff is the primary mitigation. Existing concurrency controls (`EVAL_CONCURRENCY_HOSTED` / `EVAL_CONCURRENCY_SELF_HOSTED`, bounded background tasks) limit concurrent requests. For sustained high-volume workloads, self-hosted NIM is recommended.

2. **Retry (UI action):**
  - MUST create a new proposal attempt (new `inference_invocation_id`) for the same `example_key`.
  - MUST set `retry_of_inference_invocation_id` to link the new attempt to the prior attempt.
  - MUST NOT advance the review selector and MUST NOT create a Verified label (only Accept/Edit do).

**Idempotency / dedupe keys**

- **Batch Labeling per-example operation ID MUST be:**

```
operation_id = "{batch_label_run_id}:{example_key}"
```

- Re-execution of a Batch Labeling run MUST NOT duplicate persisted outcomes for the same `operation_id`.

**Failure handling**

- **Interactive proposals:** for `schema_invalid` / `timeout` / `endpoint_error`, UI MUST surface the failure state and allow **Edit** (manual label) and **Retry**.
- **Skip/Omission:** per §4.5.
- **Missing image files:** if `storage_ref` is not found or unreadable at runtime, image serving endpoint (§10.2.9) returns an error. Labeling screen shows a broken-image placeholder and offers **Skip**. The system MUST NOT crash or block the labeling session.
- **Evaluation:** after concurrent burst, any failed examples retried sequentially (concurrency=1; §7.1). If any still fails, run is marked **incomplete** (no pass/gate). No configurable failure tolerance. Partial diagnostic metrics MAY be displayed but MUST be labeled as diagnostic only.
- **Batch Labeling:** per-example failures do not make the run incomplete; run completes when every example reaches terminal state. Circuit breaker pauses run after consecutive endpoint failures (§8.2 step 8). Failures MUST persist failure metadata + best-available artifacts/refs; exports include only `schema_valid_core=true` outputs.
- **Review selector:** if CLIP embeddings are missing/unavailable, selection MUST use available pHash values (§5.6); when every candidate hash is still pending or failed, it MUST use deterministic `example_key` order.
- **Embedding computation:** probe failures (403/no model access, network errors) MUST set `embedding_provider=none`; selector continues with pHash. Per-image CLIP failures (payload too large, rate limits, transient errors) MUST be handled gracefully: log failure, skip embedding, continue.
- **TAO jobs:** submission failures MUST set TAOJob `status="failed"` with error ref; polling failures MUST NOT corrupt last-known-good status and MUST record/update `poll_error_ref`.

**Operational Logging**

Structured JSON logs at key decision points for debugging, development, and LLM-assisted development. Each entry MUST include: `timestamp` (ISO 8601), `level` ∈ {`debug`, `info`, `warn`, `error`}, `component`, `project_id`, `correlation_id`, `message`, `details`.

**Correlation:** a single user action triggers a chain (ICL selection → token budget → prompt rendering → model invocation → schema validation). All entries share the same `inference_invocation_id` so the full decision trace is recoverable with a single query.

**Required log points** (8 total):

**Default level: `info`** (always visible):

1. **Model invocation**: model name, endpoint, sampling params, visual budget params, `finish_reason`, latency, input/output token counts.
2. **Schema validation**: validity classification, Core error count + specifics, Aux error count, normalization steps.
3. **Gate evaluation**: each criterion's current value, threshold, pass/fail; overall `gate_status`.

**Default level: `debug`** (visible when `LOG_LEVEL=debug`; escalate to `info` on material changes):

4. **ICL selection**: candidate pool size, selected `example_key[]`, total ICL count, pruned count/keys; explicit cold-start log. Escalate to `info` when pruning drops examples.
5. *(Removed in v1.0 — token-budget utilization reporting retired with the Context Budget Indicator, §6.10; enforcement observability lives in log point 4's pruned keys/count. Number retained so log points 6–8 keep their identities.)*
6. **Capability probes**: probe type, request summary, response status, classification, ModelConfig field updated. Escalate to `info` on status flip.
7. **Review selector**: selection mode, candidate set size, selected key, diversity score. Escalate to `info` on CLIP↔pHash mode switch.
8. **Pool routing**: routing decision, outcome, pool counts, rebalancing triggered. Escalate to `info` when rebalancing changes membership.

**Security constraints:** logs MUST NOT contain API keys, raw image bytes, full prompt text, or user-identifiable information beyond `project_id` and `example_key`. Prompt content referenced by `prompt_hash` (§2.1); image content by `storage_ref`. Approved-fields approach: only emit structured fields defined above. Regex redaction filter for API key patterns and bearer tokens MUST be applied as safety net.

---

## 12. Security and Privacy (Blueprint-level, Basic)  `[EXACT]`

### 12.1 Secrets Storage

Secrets (API keys, authentication credentials) MUST be stored in environment variables or the canonical `.env` file (§1.9), following NVIDIA Blueprint conventions. Required secrets: `NVIDIA_API_KEY` (hosted NIM), `NGC_API_KEY` (local NIM container pulls), `TAO_API_KEY` (Student Training).

The canonical `.env` file location is `~/.vlm_feedback_loop/.env` (§1.9), colocated with `config.yaml`. The file MUST be:

- Created by the bootstrap command (`vlm-feedback-loop init`) as a commented skeleton documenting required variables without values.
- Readable only by the local user where practical (file permissions `600`); the bootstrap command MUST create `~/.vlm_feedback_loop/` with user-only permissions (`700`) where supported by the host OS.
- Outside the repository — canonical location is not within any repo working directory. If developers use the explicit override (`--env-file` or `VLM_FEEDBACK_LOOP_ENV_FILE`) to point at a repo-local `.env`, that file MUST be gitignored.

The `.env` file contains secrets in plain text. For production or managed deployments, operators MAY bypass the file entirely and provide secrets through process environment variables or a secrets manager.

API keys MUST NOT be stored in the YAML config file (§1.9).

Credential values MUST NOT be accepted through CLI arguments or embedded in subprocess argv. Children receive private values through a copied child environment (or stdin when required); parent environments remain unchanged, and supplied private values are redacted from child/spawn diagnostics. Operator-visible Docker commands use name-only `-e NGC_API_KEY` forwarding.

**Deployment-scoped secret persistence (FTU Phase G amendment, 2026-05-12):** `NVIDIA_API_KEY`, `NGC_API_KEY`, `TAO_API_KEY` support three persistence shapes — selected per call:

- (a) **One-time connection test** — held only in request memory; never written to any persistent store.
- (b) **Runtime override** — `POST /v1/secrets:set` with `persist=false` installs the key into the deployment-scoped runtime-override layer (§12.1.2). Applies to the next NIM call without a backend restart. Lost on process exit.
- (c) **Persisted to .env** — `POST /v1/secrets:set` with `persist=true` AND deployment-level `ALLOW_UI_SECRET_PERSIST=true` atomically upserts the line in `~/.vlm_feedback_loop/.env` (`0600` perms) and reloads `Settings`. It retains the same runtime override so already-queued work with a stale `Settings` snapshot sees the new value; failed writes leave the prior file intact and the new value session-only. Container/production deployments set `ALLOW_UI_SECRET_PERSIST=false` to disable this path; endpoint returns 403 `ui_secret_persist_disabled`.

Audit trail: structured `info`-level logs (`secret_runtime_set`, `secret_persisted`) per §12.1.1. Key VALUES never appear in logs.

Runtime-override layer (§12.1.2): services read via `get_effective_secret(name, settings)` instead of `settings.NAME` directly so override path is honored. Implementation: `services/runtime_secrets.py`.

### 12.2 General Requirements

- MUST NOT log API keys (NIM or TAO).
- MUST NOT embed API keys in persisted prompts/invocations/UI payloads.
- MUST NOT log raw image bytes in operational logs.
- Filesystem operations (§10.2.10) MUST enforce one `IMAGE_ROOT`. When the backend is network-accessible (non-loopback bind), it MUST be explicitly configured. Symlinks escaping it MUST be rejected. The initial browse request omits `path`, so the backend reports the effective root; an unset loopback root means `/`. When the bundled sample and its parent are in scope, the UI may immediately open that parent while retaining the backend root as the browse boundary so the sample directory itself is selectable.
- Inference, embeddings, pHash, image serving, benchmark workloads, and
  dataset exports MUST re-authorize persisted `storage_ref` values at read
  time and bind authorization and consumption to one regular-file descriptor.
  Dataset export downloads apply the same descriptor rule beneath the
  project-owned `exports/` directory.

---

## 13. Required Records  `[REFERENCE]`

Twenty numbered sections describe 19 active persisted record families; §13.14
documents the removed `PromptPackage` and is not an active family. Full schemas
with minimum field lists, constraints, and notes are in
`Engineering_Spec.md §13.1–§13.20`. The state machines (§13.2.1, §13.2.2) are
kept EXACT below because they are cited cross-repo; all other record families
are referenced.

| # | Section | Record family | Purpose |
|---|---------|---------------|---------|
| 1 | §13.1 | **Operation Record** | Single per-invocation record across interactive proposals, evaluation, batch labeling, rationale regeneration. Generation Controls / Visual Budget / image transport / completion / validation fields. |
| 2 | §13.2 | **Run Record** | Evaluation runs and batch labeling runs. Run-type-specific fields + state machines (§13.2.1, §13.2.2). |
| 3 | §13.3 | **Review Selector Scheduler State** | On Project record; reset on semantic Core change. |
| 4 | §13.4 | **Project Record** | Top-level partition boundary. Default selections, Generation/Visual Budget defaults, embedding config, pHash algorithm, feature flags, Test Pool + Scale-Up Gate thresholds. Optional `archived_at: timestamp \| null` for soft-archive (§10.2.13.1). Optional `setup_completed_at: timestamp \| null` — null until the SME walks Screen 2/3 once, then stamped via `POST :mark_setup_completed`. ``ProjectIndexRedirect`` gates the setup route on this field. |
| 5 | §13.5 | **DatasetExport** | Versioned export artifacts (training / evaluation / testing) with manifests + lineage. `.tar.gz` format. |
| 6 | §13.6 | **AuditEvent** | Optional; notable system/user actions (skip, guidance_save, semantic_core_change, action_request_copied, storage_ref_remap, etc.). |
| 7 | §13.7 | **Label** | All labels — Verified + Auto-Labeled — with `label_status` discriminator. Promotion transitions `auto_labeled → verified`; Skip deletes the machine Label while retaining its Operation Record. |
| 8 | §13.8 | **Example** | Each ingested image. Includes `state`, `storage_ref`, `phash`, embedding presence fields, omission provenance, prior-label reference fields (set during semantic Core change). |
| 9 | §13.9 | **Guidance** | Immutable Guidance version (Description / SchemaCore / Rules) with `version_number`, `semantic_core_change_from_guidance_id`, `schema_change_summary`. |
| 10 | §13.10 | **ModelConfig** | Catalog entry with `endpoint_id`, `model_name`, `context_window_tokens`, `eligible_roles[]`, capability probes, `local_deploy_metadata`, `tao_base_experiment_id`. |
| 11 | §13.11 | **Pool** | Frozen Test Pool version snapshots. Auto-created at evaluation run start. |
| 12 | §13.12 | **TAOJob** | Cosmos-RL / TAO VLM job state (10 statuses), configuration payload, tracked outputs, chain fields (`chain_id`, `chain_sequence`, `chain_halted_reason`, `parent_tao_job_id`), and the `outputs_fetch_status` lifecycle marker (Phase 12 amendment 2026-05-05): `pending` → `in_progress` → `completed` \| `failed`. Polling tick re-fires `_handle_succeeded` for any `succeeded` job in `pending`/`in_progress` so a backend restart mid-artifact-download doesn't silently halt the chain. F8 (2026-05-05): post-success flow runs as a non-blocking background task (`background_manager.register("post-success-{tao_job_id}", ...)`) — the polling tick MUST NOT await multi-GB downloads inline, or one project's safetensors fetch head-of-line-blocks every other project. |
| 13 | §13.13 | **StudentModel** | Trained/deployed Student with immutable nullable `training_suite_id` lineage, `checkpoint_packaging_status`, `quality_status` (4-state F35: `pending`/`validated`/`partial`/`failed`), `serving_status`, NIM deployment state, `quantization_method`. |
| 14 | §13.14 | **PromptPackage** | Absent from v1.0. |
| 15 | §13.15 | **LocalNimDeployment** | Tracks system-managed local NIM containers (§1.5 Mode C). F49 amendment (2026-05-19): adds `displaced_by_deployment_id` + `displaced_at` diagnostic fields for one-NIM-per-GPU replace semantics. |
| 16 | §13.16 | **ClipEmbedding** | Dedicated float32 vector storage, one-to-one with Example. Cascade delete. |
| 17 | §13.17 | **EmbeddingDeploymentConfig** | Deployment-scoped (singleton). NeMo Retriever VL 1B v2 embedding NIM desired-state config. |
| 18 | §13.18 | **NimEndpoint** | Configured NIM endpoint (hosted / self-hosted / local_system_managed). Model identity stays on ModelConfig. |
| 19 | §13.19 | **TAODeploymentConfig** | Deployment-scoped (singleton, required when Student Training enabled). Workspace + S3 credentials + bootstrap status. |
| 20 | §13.20 | **TAOBaseExperimentProvisioningRun** | Deployment-scoped tracked first-use base registration (`queued/running/succeeded/failed`) with selected targets and redacted outcome. |

**Partitioning (§13.4.3):** every project-scoped record family MUST include `project_id`. Cross-project reads/writes are forbidden. Deployment-scoped exceptions: `EmbeddingDeploymentConfig` (§13.17), `TAODeploymentConfig` (§13.19), and `TAOBaseExperimentProvisioningRun` (§13.20) are shared across projects. TAO endpoint configuration (§1.6) is deployment-scoped but stored in env vars, not as a DB record.

### 13.2.1 Evaluation Run State Machine  `[EXACT]`

Statuses:

```
queued | running | canceling | completed | incomplete | canceled | failed
```

- `queued`: Run record exists, configuration snapshot and pool version snapshot taken, worker not yet started.
- `running`: Actively issuing evaluation inferences and/or performing the sequential retry pass.
- `canceling`: Cancellation requested (supersession or manual cancel). Two-phase: transition to `canceling`, stop new dispatch, then wait for in-flight work before `canceled`. Only an outcome committed after the durable `canceling` transition is marked `ignored_due_to_run_cancellation=true`; finalization MUST NOT bulk-reclassify earlier outcomes. Ignored records remain diagnostic and non-authoritative.
- `completed`: Run finished cleanly; all pool examples reached a terminal per-example outcome with no persistent failures. Results are authoritative and satisfy the Scale-Up Readiness Gate.
- `incomplete`: Run finished, but at least one example still failed after the sequential retry pass. Metrics are diagnostic only and MUST NOT satisfy the Scale-Up Readiness Gate (§7.1).
- `canceled`: Run was intentionally stopped, either manually by the SME or because a newer evaluation superseded it. All in-flight tasks have settled. Operation Records persisted during the `canceling` window are retained for audit but are non-authoritative. Canceled runs MUST NEVER produce authoritative aggregate metrics, even if some child Operation Records finished successfully. Evaluation metrics, run summaries, and gate checks MUST be computed only from Operation Records whose `evaluation_run_id` matches a `completed` run.
- `failed`: Unrecoverable execution failure (backend restart interruption, database/persistence failure, snapshot creation failure, uncaught runtime exception).

Transitions:

```
queued     -> running | canceling | failed
running    -> completed | incomplete | canceling | failed
canceling  -> canceled | failed
```

Terminal states: `completed`, `incomplete`, `canceled`, `failed`.

Restart recovery: evaluation runs in `queued`, `running`, or `canceling` state MUST transition to `failed` with `status_reason="backend_restart_interrupted"`. The user re-triggers evaluation.

### 13.2.2 Batch Labeling Run State Machine  `[EXACT]`

Statuses:

```
queued | running | paused | canceling | completed | canceled | failed
```

- `queued`: Run record exists, input set and configuration snapshot taken, worker not yet started. Also used after restart recovery (with `recovered_from_restart=true`) before auto-resume.
- `running`: Actively processing examples.
- `paused`: Processing intentionally halted because the circuit breaker fired (§8.2 step 8). Resumable from the next unprocessed example. `paused_reason` records the cause (e.g., `"circuit_breaker_threshold_reached"`).
- `canceling`: Cancellation requested; no new examples may start. Only outcomes committed after the durable transition are marked ignored, and ignored successes create no Auto-Labeled Label; finalization preserves earlier outcomes.
- `completed`: All selected examples reached a terminal per-example outcome (success, schema-invalid, timeout, or endpoint_error). This does NOT mean every example succeeded — use counters to show the breakdown. The run itself finished.
- `canceled`: User stopped the run (from paused or running state); already-persisted results are retained.
- `failed`: Unrecoverable run-level failure (database/persistence failure, corrupted run state, uncaught runtime exception that prevents safe resume).

Transitions:

```
queued     -> running | canceling | failed
running    -> paused | completed | canceling | failed
paused     -> queued | canceled | failed
canceling  -> canceled | failed
```

Terminal states: `completed`, `canceled`, `failed`.

Restart recovery: batch labeling runs in `queued` or `running` state MUST transition to `queued` with `recovered_from_restart=true`, then auto-resume to `running` from the next unprocessed example using idempotent per-example persistence and the durable breaker streak/latch (§8.2 steps 7–8). A latched trip pauses before new dispatch even if a drained success reset the streak. Runs in `paused` state remain `paused` (do not auto-resume). Runs in `canceling` state transition to `canceled` when `cancel_requested_at` and exact lineage are present, rebuilding counters from non-ignored authoritative items; missing intent fails with `backend_restart_interrupted`, and malformed or invalid durable state fails only that run with `batch_recovery_state_invalid`.

Additional batch labeling run fields: `paused_reason`, the run-start
`circuit_breaker_threshold` snapshot exposed by the detail response,
the internal durable `circuit_breaker_consecutive` streak and
`circuit_breaker_tripped` latch,
`examples_succeeded`, `examples_schema_invalid`, `examples_timeout`,
`examples_endpoint_error`, `examples_total`.

**Design note: why `incomplete` exists for evaluation but not batch labeling.** Evaluation has a special "finished but not trustworthy" outcome because the spec requires that one persistently failed example makes results unsuitable for gating (§7.1). Batch labeling is different: per-example failures are expected, the run can finish successfully as a whole, and exports filter to `schema_valid_core=true` outputs. So batch labeling uses `completed` plus counters; evaluation uses `completed` vs `incomplete`.

**Design note: why `paused` exists for batch labeling but not evaluation.** The spec defines a circuit breaker with Resume/Cancel for batch labeling (§8.2 step 8). Evaluation has cancel/restart behavior but no pause concept.

### 13.2.3 Required Fields by Run Type  `[REFERENCE]`

See §13.2.3 for per-run-type required fields (`evaluation_run`: pool version, guidance_id, model_config_id, icl_mode, evaluation_source ∈ {`tao`, `nim`}, generation_preset_key, thinking_mode_effective, visual_budget_preset_key, structured_generation_mode_effective, inference_contract, credential-free versioned runtime_config_snapshot for NIM inference, icl_eligible_count_at_start, icl_eligible_count_at_completion, tao_job_id, tao_native_metrics, rescored_metrics, **plus per-status counters** `examples_total` / `_succeeded` / `_schema_invalid` / `_timeout` / `_endpoint_error` (F45 amendment 2026-05-13: aggregated from OperationRecord rows during Phase G finalize so F35's `_compute_parseable_rate` and downstream `_promote_quality_to_partial` see accurate values; pre-F45 these stayed at `0` defaults even when the OperationRecords had the data); `batch_label_run`: guidance_id, teacher model_config_id, runtime_config_snapshot, ICL settings snapshot, input selector definition, structured_generation_mode_effective). The executable `RuntimeConfigSnapshot` schema owns the exact field set. Version 2 includes model/endpoint lineage and capabilities plus concrete sampling, visual, token-budget, ICL, and image-downscale inputs—never credentials or live operational policy.

---

## Appendix A: Algorithms and Contracts

### A.1 Markers and IDs  `[EXACT]`

1. Backend MUST NOT include `example_key` in model-visible prompt text.
2. ICL examples in prompt MUST be labeled `E01`, `E02`, …
3. Backend MUST persist `icl_example_keys_used[]` for every call (interactive, evaluation, batch labeling).

### A.2 Exact Match (Core Fields) (Normative Summary)  `[EXACT]`

- Deterministic, schema-driven comparison
- Core fields only
- Normalization:
  - enums: trim/lowercase and match allowed values
  - enum sets: normalize, dedupe, sort, compare as sets
  - booleans: valid values are JSON `true` and `false` only. String proxies (`"true"`, `"false"`) and numeric proxies (`1`, `0`) are not canonical and are treated as schema-invalid. The same boolean normalization logic MUST be used in interactive validation, evaluation, Batch Labeling validation, and TAO re-scoring (single canonical normalizer; see Decisions).
  - integers: validate min/max; out of range invalid (not clamped)
  - strings: trimmed exact match (no fuzzy matching)

**Derived accuracy metrics (MUST be computed for every evaluation run):**

- **Overall Exact Match rate**: fraction of examples where all Core fields match after normalization.
- **Per-core-field match rate**: for each Core field independently, the fraction of examples where that field matches. Identifies which fields the model handles well vs. poorly.
- **Per-value precision, recall, and F1** (categorical Core fields only — enum, enum set, boolean): for each allowed value, compute precision (true positives / predicted positives), recall (true positives / actual positives), and F1 (harmonic mean). For enum set fields, membership of each value is evaluated independently.

All derived metrics MUST be reported alongside the overall Exact Match rate in evaluation results (§7) and Student comparison results (§5.4).

### A.3 Diversity-Driven Review Selector Scheduling  `[SUMMARY]`

Greedy max-min diversity with two tiers of similarity signal. Algorithm structure identical; only the similarity function differs.

**Mode selection** (when `REVIEW_SELECTION_MODE=auto`):

1. If number of eligible examples with CLIP embeddings ≥ `CLIP_SWITCHOVER_MIN_COUNT` (default 50) → CLIP-diverse.
2. Else → pHash-diverse (pHash always available; §5.6).

Forced modes: `clip_diverse` (falls back to pHash if below threshold); `phash_diverse` (forces pHash).

**Similarity functions:**

- CLIP: `sim_clip(a, b) = dot(emb(a), emb(b)) / (||emb(a)|| * ||emb(b)||)`
- pHash: `sim_phash(a, b) = 1 - hamming_distance(phash(a), phash(b)) / hash_bits`

**Prior-label priority rule (§4.4.1, §6.5):** before diversity selection, selector partitions eligible set into **Tier 1 (prior-label)** — examples with `prior_verified_label_ref` (starting with `schema_change_context_example_key` if set, then prior Edits, then prior Accepts) — and **Tier 2 (standard)** — examples with no prior label. Exhausts Tier 1 before Tier 2. Within each tier, diversity selection applies normally.

**Greedy diversity selection** (per tier):

1. Candidate set `C = { e ∈ tier | signal(e) is not null }`.
2. Recent window `H_sig = [h ∈ H | signal(h) is not null]`.
3. If `H_sig` empty: select first candidate by ascending `example_key`.
4. Else for each `e ∈ C`, compute `score(e) = max_{h in H_sig} sim(e, h)`. Select `e* = argmin_e score(e)`. Tie-break by ascending `example_key`.
5. If `C` empty (no candidates have active signal): select first eligible by ascending `example_key`.

Reproducibility: candidate sets from persisted state, similarity retrieval stable for given ref/hash, scoring deterministic, tie-breakers explicit.

See §A.3 for full pseudocode.

### A.4 Reference Configuration Defaults  `[SUMMARY]`

Canonical defaults list — full yaml in `Engineering_Spec.md Appendix A.4`. Settings may be null/optional overrides; when null, system derives effective values. Key highlights:

**Embedding:** `EMBEDDING_PROVIDER: auto`, `EMBEDDING_MODEL_ID: nvidia/llama-nemotron-embed-vl-1b-v2`, `EMBEDDING_DIM: 2048`, `EMBEDDING_INPUT_TYPE: passage`, `EMBEDDINGS_AUTO_COMPUTE: true`. Provider-aware worker shape (§5.5.2): `EMBEDDING_CONCURRENCY_HOSTED: 1` / `EMBEDDING_BATCH_SIZE_HOSTED: 8` (avoid hammering shared `build.nvidia.com`) and `EMBEDDING_CONCURRENCY_SELF_HOSTED: 4` / `EMBEDDING_BATCH_SIZE_SELF_HOSTED: 1` (saturate local GPU). The `*_nvclip` provider values and `LOCAL_NIM_NVCLIP_PORT` name remain for compatibility only.

**Generation Controls:** `THINKING_DEFAULT_ON: true`. Default preset key is a per-project setting (Project record §13.4): `labeling_generation_preset_key: precise`.

**Preset params:** `precise: {temperature: 0.0, top_p: 1.0}`, `explore: {temperature: 0.3, top_p: 0.9}`.

**Visual Budget:** per-project `visual_budget_preset_key: high_detail` (Project record §13.4). Per-mode preset values for `mm_processor_size`, `mm_processor_pixels`, `mm_processor_tiles` — see §A.4.

**Prompt budgets:** `RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN: 0.85`, `BASE_OUTPUT_TOKENS_FLOOR: 256`, `JSON_STRUCTURAL_OVERHEAD_TOKENS: 48`, `MAX_OUTPUT_FRACTION: 0.25`, `RATIONALE_NOTE_ESTIMATE_TOKENS: 160`, `DEFAULT_UNBOUNDED_STRING_BUDGET: 200`, `MODEL_REASONING_HEADROOM_TOKENS: 16384`. Startup validates positive token budgets, fractions/margins in `(0, 1]`, non-negative structural overhead, reasoning headroom ≥4096, and valid `temperature`/`top_p` sampling pairs.

**ICL:** `ICL_MAX_EXAMPLES: null` (or a positive integer; zero-shot runs use `icl_mode="disabled"`).

**Export:** per-project `export_field_mode: all`.

**Test Pool:** per-project `test_pool_fraction: 0.40` (Project record §13.4).

**Scale-Up Gate:** per-project settings (Project record §13.4): `scaleup_exact_match_threshold: 0.80`, `scaleup_per_field_match_threshold: 0.80`, `scaleup_min_per_value_f1_threshold: 0.60`, `scaleup_accept_rate_threshold: 0.80`; `scaleup_accept_rate_window: 50`; `scaleup_min_test_pool_size: 60`.

**Batch labeling:** `BATCH_LABEL_RUN_LIMIT: null` (or a positive integer), `BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD: 10`, `BATCH_LABEL_CONCURRENCY_HOSTED: 1`, `BATCH_LABEL_CONCURRENCY_SELF_HOSTED: 8`; the threshold and concurrencies must be positive.

**TAO:** `TAO_API_BASE_URL`, `TAO_API_KEY`, `TAO_ORG_NAME`, `TAO_WORKSPACE_S3_ACCESS_KEY`, and `TAO_WORKSPACE_S3_SECRET_KEY` are unset until explicitly configured. Non-secret workspace identity, bucket, and endpoint URLs live only in `deployment.db` after `vlm-feedback-loop tao-bootstrap`.

**Student training:** `TAO_RELEASE_VERSION: "6.26.3"`, `COSMOS_RL_CONTAINER_TAG: "6.26.3-cosmos-rl"`, `STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD: 0.90` (F35 — parseable-rate threshold for `quality_status="partial"` promotion on incomplete NIM evals).

**NIM benchmark:** `NIM_STARTUP_TIMEOUT_S: 1200`, `NIM_BENCHMARK_TIMEOUT_S: 1200`, `STUDENT_LATENCY_TEST_CONCURRENCIES: [1,8,24]`.

**Review selector:** `REVIEW_SELECTION_MODE: auto`, `REVIEW_RECENT_WINDOW_K: 20`, `CLIP_SWITCHOVER_MIN_COUNT: 50`.

**Schema reminders:** `SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1: 10`, `SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2: 35`.

**Rationale:** per-project `rationale_anti_anchoring: true`.

**Evaluation:** `EVAL_CONCURRENCY_HOSTED: 1`, `EVAL_CONCURRENCY_SELF_HOSTED: 8`, `EVAL_FIRST_POOL_SIZE: 5`.

**HTTP:** `HTTP_DEADLINE_INTERACTIVE_S: 180`, `HTTP_DEADLINE_BACKGROUND_S: 300`, `HTTP_MAX_RETRIES: 3`.

**Logging:** `LOG_LEVEL: info`.

**Workspace/server:** `WORKSPACE_ROOT: unset`, `BIND_HOST: 127.0.0.1`, `BIND_PORT: 8000`, `IMAGE_ROOT: unset` (effective `/` on loopback), `HOSTED_NIM_BASE_URL: https://integrate.api.nvidia.com/v1`, `LOCAL_NIM_TEACHER_PORT: 8000`, `LOCAL_NIM_NVCLIP_PORT: 8001`.

### A.5 Pool Diversity Selection (Rebalancing)  `[SUMMARY]`

When rebalancing promotes non-pool Verified into the Test Pool (§4.3.2), uses the review selector's CLIP/pHash switchover — single diversity signal per rebalancing pass, no mixed-mode scoring.

Same greedy max-min diversity approach as review selector (A.3), applied to pool construction. Inputs: candidate set (non-pool Verified, both Accept + Edit), current pool members (may be empty on first fill), `n_needed`, per-example CLIP + pHash. Algorithm: determine active function via switchover rule, iterate `n_needed` times computing `score(e) = max_{r ∈ P_ref} sim(e, r)`, select most dissimilar (argmin), tie-break by `example_key`. If `n_needed > |C|`, fill remaining from candidates without active signal by `example_key`.

See §A.5 for pseudocode.

---

## Appendix B: Acceptance Tests / Validation Checklist  `[REFERENCE]`

Full verification checklist — 1000+ items covering every normative section. See `Engineering_Spec.md Appendix B`. Do not duplicate content here; when implementing a change, load the relevant Appendix B section from the full spec.

Notable subsections:

- **Action Requests (§10.3)** — 11 verify items including the four §10.3.3 customer-facing handoff keys (`nim_model_profile_recommended`, `gpu_requirements`, `tensor_parallelism`, `nim_env_vars_recommended`) and the canonical `docker_run_args` provenance via `local_nim_service.build_student_docker_run_*`.
- **Full-stack validation (Phase 12.4)** — C20 (handoff `docker run` re-execution proof for both 2B and 8B; optional RPS evidence uses baseline representatives and gates on one raw/fenced-JSON schema-parseable prediction per class while recording accuracy separately), C21 (2B vs 8B handoff content differentiation, accepting two intentionally unset custom-checkpoint profiles while requiring distinct images/GPU floors/served-model environments), final integration checkpoint, Cosmos-RL format validator, `LOCAL_NIM_MOCK_ENDPOINT_URL` test seam. Mock-validated in `tests/unit/test_full_stack_validation_mock.py`; live-validated via a rental-window operator runbook (retained in the project's internal engineering archive).
- **Release automation and static analysis** — verify the eight CI jobs
  (backend lint/tests/typecheck, frontend lint/tests/build, locked dependency
  audit, and Compose smoke), seven pre-commit hooks, serial operator-run
  integration suite, strict pyright configuration at the Python 3.11 floor,
  and the org-gated SonarQube delegate. The current commands and invariants are
  listed in the full Spec; historical job and suppression counts are not part
  of the contract.

---

## Appendix C: Minimal Configuration (Interactive Labeling Only)  `[REFERENCE]`

Fastest path: set `WORKSPACE_ROOT` + `NVIDIA_API_KEY`, run bootstrap, ingest images, label. See `Engineering_Spec.md Appendix C`.

---

## Appendix D: Default Prompts  `[REFERENCE]`

- **D.1** Teacher Interactive Proposal Prompt
- **D.2** Guidance Rewrite Prompt (removed 2026-07-21 with the AI Guidance Rewrite feature; D.1/D.3 numbering unchanged)
- **D.3** Rationale Regeneration Prompt

Full templates with normative requirements (compact prompt-visible schema, Core-only bookended ICL with `E01..` markers, rationale-last request, and task-conditional grounding) are in `Engineering_Spec.md` Appendix D.

---

## Appendix E: Student Precision/Quantization Evaluation Suite  `[REFERENCE]`

Five experiments for comparing Student variants:

- **E.1** Quality Delta vs 16-bit Baseline (TAO or NIM)
- **E.2** Min Latency + Peak Memory (NIM at concurrency=1)
- **E.3** Concurrency Scaling and Reliability (NIM via pinned AIPerf; exact request counts and zero-failure cell gate)
- **E.4** Efficiency Normalization
- **E.5** Robustness (Long Input / Long Output)

Apples-to-apples rules require the identical deterministic real Test Pool workload, a Guidance-derived uncapped production request whose prompt hash matches the serving evaluation, disabled KV-cache reuse, selected NIM profile, and decoding/visual controls across variants. Required provenance includes `student_model_config_id`, `nim_model_profile_*`, `quantization_method`, GPU model/count, dataset manifest SHA-256, workload hash, prompt/contract lineage, and driver version. See `Engineering_Spec.md Appendix E`.

---

## End of Engineering Specification Brief

*This brief mirrors the structure of `Engineering_Spec.md`. For normative detail on any summarized or referenced section, open the corresponding section in the full spec.*

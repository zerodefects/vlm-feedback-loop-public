# Interactive VLM Feedback Loop - Engineering Specification

*Labeling with ICL; Optional Fine-Tuning*

**Version 1.10.1**

> Amendment markers such as "F-W7 amendment (2026-07-14)" or "Phase 12
> amendment" are stable IDs from the project's engineering finding ledger,
> kept as anchors for cross-references within this document. Every
> amendment's text is self-contained — no external ledger is needed to
> apply it.

## Decisions (mandatory)

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
- **Prompt budgets are per-model:** prompt packing MUST derive effective input budgets from the selected model’s configured context window.
- **Comparison reads from evaluation runs directly:** the Compare screen's Teacher baseline is the most recent completed Teacher-contract evaluation run, and each Student's quality/serving results come from the runs referenced on its StudentModel record (`quality_evaluation_run_id` / `serving_evaluation_run_id`). The page remains one project-wide comparison and groups retained Students by immutable `training_suite_id`, newest run first. Repeated base/precision combinations use run-qualified chart identities. Cross-run score differences are explicitly directional when Guidance, effective Inference Contract, or frozen evaluation evidence differs; Student-vs-Teacher deltas are hidden unless both evaluation contexts match. The pre-v1.0 evaluation-suite *execution* grouping was removed with the Student+ICL evaluation arm (§7.2 note); Training Suite headings are provenance, not a replacement evaluation-suite subsystem.
- **Batch Labeling is Teacher-run:** the batch fully synthetic label generation workflow in this spec runs the **Teacher**.
- **Retry is a first-class interactive action:** the UI MUST offer **Save / Skip / Retry** per image. Retry MUST produce a new proposal attempt for the same example after applying a user-selected Teacher model and/or Guidance. Retry MUST NOT create a Verified label by itself.
- **Skip omits the image:** Skip MUST record no label and MUST omit the image from the workflow so it is not presented again.
- **Review selector selection is diversity-driven whenever a signal is available:** a restartable background pHash sweep provides the baseline signal. When per-image CLIP embeddings exist, the selector upgrades to CLIP-diverse mode (semantic diversity). While all candidate signals are still pending, the selector uses deterministic `example_key` order; there is no random selection mode.
- **All persisted entities are project-scoped:** every Example/Label/Guidance/Pool/Run/Operation/ModelConfig MUST reference `project_id`.
- **Cosmos-RL / TAO VLM jobs are first-class and poll-tracked:** every Student Training fine-tuning execution MUST create a **TAOJob** record with an explicit TAOJob state machine, persist the exact job configuration payload used to start the training backend, and track (by reference) required outputs: **artifacts, metrics/progress, and logs**. Status MUST be updated via polling the TAO Jobs API and mapped deterministically into the TAOJob state machine.
- **Rationale notes are opt-in auxiliary data, never prediction authority:** Guidance disables `rationale_note` by default. The SME MAY enable or disable the reserved `role="aux"` string field at any time without label invalidation (§4.4). When enabled, the production Teacher requests it last and the existing review/provenance workflow applies. When disabled, the field is absent from prompts, model contracts, proposals, review, validation, and newly Verified labels; neither the model nor the SME supplies it. Core validity and evaluation never depend on it. Production Teacher ICL is Core-only and never feeds rationale text back as a labeling signal. An export field mode (`all`, `aux_and_core`, or `core_only`; default `all`) independently controls serialization of rationale data that is present (§9.3).
- **Embedding computation is default-on via hosted embedding NIM:** when a hosted NIM API key is configured, the system MUST automatically compute CLIP-style image embeddings for ingested images (§5.5). The supported Blueprint model is **NVIDIA NeMo Retriever VL 1B v2** (`nvidia/llama-nemotron-embed-vl-1b-v2`, 2048-dim, requires `input_type` in the request body). Embedding computation MUST NOT block interactive labeling (§5.5.2).
- **Pool management is fully automatic:** a configured fraction sets the Test Pool target, and the next Verified example fills an immediate shortfall regardless of verification outcome. Rebalancing uses CLIP-diverse selection once enough embeddings exist and pHash-diverse selection otherwise. Both Accepts and Edits are candidates for the Test Pool; ICL draws exclusively from non-pool Edits. Evaluation runs auto-create frozen pool version snapshots for reproducibility. No user action is required (§4.3).
- **Scale-Up Readiness Gate governs Batch Labeling:** the Batch Labeling CTA MUST be gated on configurable, system-evaluated criteria (evaluation overall Exact Match, per-core-field match rate, minimum per-value F1, Accept rate over a rolling window, and minimum Test Pool size). Clicking [Run Batch Labeling] when criteria pass IS the SME's confirmation. The gate MUST NOT block Interactive Labeling or evaluation runs. Thresholds are configurable per project (§7.3).
- **Student Training has an independent data-readiness gate:** the Scale-Up Teacher-quality criteria (Exact Match, per-field match, per-value F1, and Accept rate) MUST NOT block Student Training navigation or launch. **Start Training** MUST, however, fail closed until the active Guidance has at least one non-pool Verified training example and the Test Pool reaches the project's configured `scaleup_min_test_pool_size` (default: 60; effective Student-evaluation floor: one). The same backend preflight check MUST protect initial launch and post-provisioning suite materialization (§9.7.8.3, §10.2.22).
- **Generation Controls are preset-driven:** labeling workflows expose Output Stability presets and a Thinking toggle as the only user-facing sampling knobs; raw parameter editing is not permitted. Effective values are always persisted in Operation Records. See §6.7.
- **Visual Budget Controls are capability-gated and preset-driven:** image preprocessing (visual token spend) is managed via named presets (**Fast** / **Balanced** / **High Detail**) that resolve to model-specific `mm_processor_kwargs`. Controls are only exposed when the active Teacher supports them (`visual_budget_mode` ≠ `none`). The effective `mm_processor_kwargs` sent per invocation are persisted on Operation Records. Evaluation and Batch Labeling use the project's visual budget preset (no per-example variation) for apples-to-apples reproducibility. See §6.9.
- **Student Training backend is Cosmos-RL / TAO VLM supervised fine-tuning:** Student Training is not a generic TAO operation. It is an explicit Cosmos-RL / TAO VLM supervised fine-tuning workflow with TAO-native dataset, config, checkpoint, quantization, and NIM deployment handoff requirements. Only catalog entries with the `student_base` role are eligible (currently Cosmos Reason2 8B/2B and Cosmos 3 Nano/Super reasoner). The default training policy is `sft` (supervised fine-tuning); implementations MUST NOT assume RLHF-like behavior from the "RL" in the product name.
- **Student Training is LoRA-first by default.** `enable_lora=true` is the default training mode. LoRA is the parameter-efficient path suited to iterative, multi-variant, cost-conscious VLM fine-tuning and comparison. Cosmos-RL does not require LoRA, but this system's default training configuration uses it. The `lora_config` on every TAOJob (§9.7.3.2) records the LoRA parameters used. Full-weight fine-tuning (if ever needed) would set `enable_lora=false`, but this is not the default path and requires the operator to understand the increased resource and packaging implications.
- **Full-precision baseline NIM packaging owns the LoRA merge path.** For quantized variants, TAO `quantize` auto-merges the LoRA adapter when `enable_lora=true` and `base_model_path` is set (§9.8). For the full-precision baseline variant, no quantize step occurs, so no auto-merge happens. Checkpoint packaging (§9.5.1) MUST handle this: it first validates whether the training output is already a NIM-loadable HuggingFace checkpoint directory; if the output is adapter-only, packaging MUST merge the adapter into the base model using the persisted `base_model_path` (sourced from the training TAOJob's `resolved_training_fields.policy.model_name_or_path`) and produce a merged checkpoint. If `base_model_path` is unavailable or merge fails, packaging fails with a clear error. `export_safetensors: true` in the training config controls checkpoint serialization format (SafeTensors vs PyTorch .bin); it does NOT guarantee LoRA adapter merge.
- **Runtime stack is mandated; background tasks use in-process execution with SSE notification:** the system MUST use Python + FastAPI (backend), React + TypeScript (frontend), and SQLite (per-project database). Evaluations, CLIP embedding computation, and Batch Labeling MUST run as in-process background tasks with durable run records. On restart, the backend MUST detect incomplete background tasks and either resume idempotently or mark them failed. The backend MUST support SSE for pushing progress and completion events to the UI; REST status endpoints MUST be retained for initial state load and reconnect fallback (§1.3, §1.8).
- **JSON Schema derivation is backend-canonical:** the backend MUST be the sole canonical implementation of SchemaCore → JSON Schema derivation. The frontend renders the issues returned by the draft validation endpoint (§10.2.2) — debounced while editing, immediate on save — and implements no validation rules of its own (§6.6.6). The same backend derivation function MUST be used by both the draft validation endpoint and the Guidance save endpoint. No shared cross-language derivation library is required.
- **Post-training evaluation is automatic and TAO-backed:** when a TAO training job succeeds, the system MUST automatically submit a TAO `evaluate` job against the Test Pool export. When a TAO quantize job succeeds, the system MUST automatically submit a TAO `evaluate` job against the quantized checkpoint. TAO per-sample predictions MUST be re-scored by the system's canonical Core-field evaluator (Appendix A.2) to preserve product-specific correctness semantics. TAO's native aggregate metrics are informational; the system's re-scored metrics are authoritative. This chain (train → evaluate → quantize → evaluate) runs without SME intervention after "Start Training."
- **First-run quantization is FP8_DYNAMIC only:** the default **Validate
  training setup** workflow selects one recommended small base, the Quick
  preset, and one FP8 variant beside the full-precision baseline. This expands
  to exactly four jobs. Multi-base and additional quantization comparison is
  an explicit advanced intent. Each selected scheme remains a separate TAO
  `quantize` job followed by a TAO `evaluate` job.
- **Student quality and serving readiness are separate gates:** Student quality is validated by TAO evaluation (accuracy/F1/precision/recall against the Test Pool, re-scored by the canonical evaluator). Student serving readiness is validated by NIM deployment + evaluation (latency, throughput, profile metadata) and the current real Test Pool AIPerf workload. Quality validation does not require NIM. Serving validation does. The `deployment_handoff` Action Request (§10.3) requires both gates to pass; a legacy synthetic serving result remains historical evidence but cannot satisfy the serving gate.
- **NIM deployment for Student evaluation uses local orchestration with Action Request fallback:** the system attempts to deploy Student NIM containers locally on the backend host via Docker (§9.5.2). A NIM deployment preflight checks prerequisites (Docker, NVIDIA Container Toolkit, GPU memory, NGC API key). If preflight passes, the system orchestrates the full container lifecycle (start → health poll → evaluate → benchmark → stop) per variant, sequentially. If preflight fails, the system generates an Action Request with the exact `docker run` command and prerequisites so the SME can hand off to infrastructure. Student NIM endpoints are temporary evaluation infrastructure, not production deployments.
- **One-NIM-per-GPU invariant (F49 amendment, 2026-05-19; §1.5, §9.5.2):** at most one NIM container is `starting` or `running` on any GPU at any time. Fresh-project creation and later Teacher deploys reuse an exact compatible running Blueprint-managed Teacher, including across projects; the fresh project selects it before returning. A different resident is never stopped implicitly: the API names it and requires an explicit `replace_resident=true`; the FTUE asks the SME whether to keep it or stop it and start the requested NIM. Multi-GPU placement is deterministic: lowest free index first. Student NIM benchmarking (§9.5.2 step 0) defaults to replace semantics and best-effort auto-restores the displaced Teacher after stop (§9.5.2 step 9). On single-GPU local-only hosts, image embedding falls back to pHash diversity (§5.6) until the operator enables a hybrid path (`NVIDIA_API_KEY` for hosted embeddings) or explicitly stops the Teacher. Empirical motivation: Cosmos Reason2 NIM's hardcoded `gpu_memory_utilization=0.9` profile-selector floor (README "One-NIM-per-GPU policy").

---

## 1. Introduction & Scope

### 1.1 Purpose

This document is the implementation contract for the Interactive VLM Feedback Loop system. It specifies required behavior for:

- Data states and transitions
- ICL selection and prompt rendering
- Inference + review workflows (including **Retry** and **Skip**)
- Evaluation workflows and metrics
- Optional **Batch Labeling** runs (Teacher-run fully synthetic label generation) to create large Auto-Labeled datasets from Unlabeled images
- Optional Student Training fine-tuning integration and Student tracking
- Review selector scheduling for Unlabeled and Auto-Labeled examples, using pHash-diverse selection (immediate) with automatic upgrade to CLIP-diverse when embeddings are available
- Catalog-driven multi-model support (multiple Teachers, multiple Students, side-by-side comparisons)
- Project/workspace management (Project CRUD + defaults + strict scoping/partitioning)

This spec supports either:

- **NVIDIA-hosted NIM APIs** (via API Catalog), or
- **Self-hosted (or otherwise hosted) NIM** endpoints you operate.

Goal: fast Interactive Labeling setup, scalable dataset generation via Batch Labeling, model tradeoff decisions via multi-Student comparison, rapid iteration via Retry, improved SME throughput via diversity-driven review selectors (pHash baseline, CLIP upgrade), and fully specified Cosmos-RL / TAO VLM job triggering + tracking.

**v1 scope: single-user tool.** v1 is a single-user, project-scoped labeling tool, not a collaborative platform. One active session per project (§1.3). No user authentication, no RBAC, no multi-reviewer workflows. Remote access assumes a trusted network. Multi-user and auth are post-v1.

### 1.2 Reuse-first Principle

Implementations MUST follow these reuse boundaries:

- Model serving MUST use NVIDIA NIM for Teacher/Student endpoints.
- Embedding computation MUST use NVIDIA NeMo Retriever VL 1B v2 NIM (`/v1/embeddings`, 2048-dim) for the supported Blueprint path. Legacy `*_nvclip` provider enum names remain only for database/API compatibility.
- Student Training fine-tuning MUST use Cosmos-RL / TAO VLM fine-tuning via NVIDIA TAO Toolkit (when enabled).

### 1.3 Execution Model

- Primary API is synchronous request/response (REST). No WebSockets.
- The backend MUST support Server-Sent Events (SSE) for real-time progress and completion notifications on background tasks. SSE is the primary UI update channel for background work. REST status endpoints MUST be retained for initial state load, reconnect after SSE disconnect, and polling fallback.
- Required SSE event types: `evaluation_started`, `evaluation_progress`, `evaluation_completed`, `embedding_progress`, `embedding_completed`, `batch_label_progress`, `batch_label_completed`, `export_progress`, `export_completed`, `export_failed`, `run_failed`. Events are project-scoped.
- **SSE is best-effort and non-authoritative in v1.** The authoritative source of state is the REST API backed by persisted project records. SSE provides inexpensive live progress updates for long-running work (evaluations, embeddings, batch labeling, TAO polling, NIM benchmarking) but missed events during a disconnection are not replayed in v1. No replay buffer, `Last-Event-ID` replay semantics, or `reconnect_stale` protocol are required. The browser's EventSource reconnects automatically on connection loss; the frontend treats reconnect as a cue to refresh state from REST, not to expect event replay.
- **Frontend recovery contract:** on page load, fetch REST snapshot first. While SSE is healthy, use SSE as a hint/update channel. On any SSE disconnect or error, if active background work exists, begin short-interval REST polling (e.g., 5s). On SSE reconnect/open, immediately refresh relevant REST state. On terminal SSE events (`*_completed`, `run_failed`), immediately refresh REST state. This preserves a live feel without requiring replay correctness.
- Evaluations, CLIP embedding computation, and Batch Labeling run as **in-process background tasks** within the backend process. Run/task records MUST be persisted in the project database for observability and recovery. On backend restart, the system MUST detect incomplete background tasks. Recovery is workload-specific and governed by the run state machines (§13.2.1, §13.2.2): **evaluation** runs in non-terminal states transition to `failed` with `status_reason="backend_restart_interrupted"` (the user re-triggers); **CLIP embedding** computation resumes automatically from the first example without an embedding; **Batch Labeling** runs in `queued` or `running` state transition to `queued` with `recovered_from_restart=true` and auto-resume; runs in `paused` state remain `paused`.
- **Single-process concurrency model (v1):** v1 supports one backend process per project. The backend MAY optionally acquire a simple file lock when a project is opened; if a second process attempts to open the same project, it receives a hard error: *"This project is already open in another process."* No override path, no warning-plus-continue flow. Multi-session UX is post-v1. Within a single process, the backend has multiple concurrent writers (interactive labeling, evaluation, embedding, batch labeling); see §1.8 SQLite write-discipline rules.
- **Graceful shutdown:** on backend process exit, in-flight background tasks are canceled after a short grace period. The system persists partial results and marks interrupted run states before exiting. Browser tab closure does not stop the backend; background tasks continue until the backend process itself exits.
- Student Training (when enabled) runs on **external executors** and is observed via polling/completion checks. Student Training MUST be represented as a poll-tracked TAOJob with durable state, logs, metrics, and artifact references. **TAO recovery on backend restart:** the system MUST reconcile all persisted TAO job chains. For any TAOJob in `submitting` with `tao_external_job_id = null`, transition to `failed` with `status_reason="submission_interrupted"`. For any TAOJob in a non-terminal state (`submitted`, `queued`, `running`, `paused`) with a known `tao_external_job_id`, the system MUST resume polling because TAO jobs execute externally and continue regardless of backend availability. For any chain where the job at `chain_sequence = N` has `status=succeeded` and the next job at `chain_sequence = N+1` has `status=not_started`, the system MUST submit that next job. If any prior job in the chain has `status=failed` or `status=canceled`, the chain remains halted.

- **Foreground priority on shared endpoints:** the system defines two workload classes. **Foreground:** `interactive_proposal`, `retry`, `rationale_regeneration`. **Background:** `evaluation`, `batch_label`, `embedding`. When one or more foreground requests are in flight, the system MUST hold dispatch of new background HTTP requests until foreground demand clears. Already-in-flight background requests MAY complete naturally. This is a simple dispatch hold, not preemption. When foreground demand clears, background dispatch resumes from the remaining not-yet-started work units. Completed work is not retried; failed in-flight work follows the normal retry policy (§11). The dispatch hold does not change run state machines: evaluation remains `running`, batch labeling remains `running`. Batch labeling `paused` is reserved for the circuit-breaker behavior (§8.2 step 8), not for foreground-priority holds. Background work may complete more slowly while the SME is actively labeling; that tradeoff is intentional.

### 1.4 Optimization Objectives

- Minimize time from feedback to changed behavior (edits influence subsequent calls).
- Maximize label efficiency via ICL under token constraints.
- Quick onboarding (hosted NIM + API key sufficient).
- Improve SME throughput via diversity-driven selection as background pHash/CLIP signals become available, with a deterministic no-signal fallback from the first image.
- Scale dataset creation via Batch Labeling with lineage + safeguards.
- Tradeoff decision support: compare Students with accuracy + latency.
- Interactive iteration speed: Retry per image by changing Teacher and/or Guidance.
- Reduce wasted SME time: Skip images and ensure they never reappear.
- Workspace hygiene: strict project scoping; no cross-project leakage.
- Explainability: persist lineage, artifacts, attempts, and audit events.
- Student Training observability: complete Cosmos-RL / TAO VLM job config + state machine + output tracking.

### 1.5 NIM Endpoint Modes

The system MUST support configuring Teacher/Student endpoints in one of these modes.

#### Mode A: NVIDIA-hosted NIM APIs (API Catalog)

Base URL typically:

```text
https://integrate.api.nvidia.com/v1
```

Auth:

```text
Authorization: Bearer $NVIDIA_API_KEY
```

Model list (when supported): `GET /v1/models`

#### Mode B: Self-hosted (or otherwise hosted) NIM you operate

Base URL is deployment-defined, e.g.:

```text
http://0.0.0.0:8000/v1
```

No credential is attached: self-hosted NIMs are expected to run on a trusted private network or behind an external gateway (`auth_mode="none"`).

#### Mode C: Local NIM Deployment (System-Managed)

The system can deploy NIM containers locally on the backend host for models with known deployment metadata in the catalog. This reuses the same Docker orchestration pattern as Student NIM evaluation (§9.5.2) but for persistent local endpoints rather than temporary evaluation containers.

Local NIM deployment is available for seeded catalog entries that carry `local_deploy_metadata` (§13.10): container image ref, GPU memory minimum, and preferred host port. In v1, this covers the NeMo Retriever VL 1B v2 embedding NIM and the seeded Teacher models (Cosmos Reason2 8B/2B, Cosmos 3 Nano/Super reasoner, and Nemotron 3 Nano Omni). Arbitrary user-added catalog entries are remote-only unless they include full local deployment metadata.

**Environment assessment (deployment-scoped, warmed at application start):**

The system probes the local environment to determine which NIM deployment modes are available. The browser starts this deployment-scoped assessment once at application startup without blocking the Project List; project creation and direct API clients lazily initialize the same backend snapshot when it is not already warm. The expensive machine capability portion (Docker, NVIDIA Container Toolkit, and GPU inventory) is ephemeral but cached for the backend process lifetime rather than repeated per project or per request. Cheap dynamic state — configured credentials, embedding deployment configuration, and active Blueprint-managed NIM residents — is composed fresh on every response. Routes whose behavior does not depend on machine capabilities MUST NOT wait for the assessment.

`GET /v1/environment?refresh_hardware=true` explicitly discards and rebuilds the process-local machine snapshot for the uncommon case where an operator installs or changes Docker, the NVIDIA runtime, drivers, or GPUs without restarting the backend. This read-side cache is recommendation state, not a deployment safety gate: every local NIM preflight still re-checks the applicable live hardware and runtime requirements immediately before launch.

Checks:

1. `NVIDIA_API_KEY` present → hosted NIM available for inference and embeddings.
2. `NGC_API_KEY` present → NIM container image pulls possible.
3. `docker info` succeeds → Docker available.
4. NVIDIA Container Toolkit test (`docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`) → GPU passthrough available.
5. `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader` → GPU inventory (model, total memory, and compute capability per GPU).
6. Cross-reference GPU inventory against seeded catalog `local_deploy_metadata` → which models can deploy locally (one GPU meets the model's memory and optional compute-capability floors).

Output: a structured `EnvironmentAssessment` available via `GET /v1/environment` (§10.2):

- `hosted_nim_available: boolean` (NVIDIA_API_KEY present)
- `local_deploy_available: boolean` (Docker + NVIDIA toolkit + at least one GPU detected)
- `docker_available: boolean`
- `nvidia_toolkit_available: boolean`
- `nvidia_api_key_configured: boolean`
- `ngc_api_key_configured: boolean`
- `gpus[]: { name: string, memory_total_gb: float, compute_capability: float | null }` (empty when no GPU detected)
- `local_deployable_models[]: { model_name: string, nim_container_image: string, gpu_memory_minimum_gb: int, compute_capability_minimum: float | null, fits: boolean }` (seeded catalog entries with `local_deploy_metadata`, each annotated with whether one detected GPU meets both declared floors. This deployment-scoped response intentionally carries no project-local ModelConfig IDs; NIM Configuration joins by `model_name` against `GET /projects/{id}/model_configs`. This covers Teacher/Student models only — VLM inference models from the model catalog.)
- `embedding_deployment: { model_name: string, nim_container_image: string, gpu_memory_minimum_gb: int, fits: boolean, provider: string }` (Embedding NIM deployment availability, sourced from `EmbeddingDeploymentConfig` §13.17, not from the model catalog. `fits` requires a currently claimable GPU whose memory meets the floor and whose detected name exactly matches the pinned NIM support matrix. Reported separately because the embedding NIM is infrastructure for background embedding computation, not a user-selectable inference model.)
- `missing_prerequisites[]: { check: string, install_hint: string }` (what's missing and how to fix it)
- `recommended_teacher_mode: "hosted" | "local" | "none"` (system recommendation based on available capabilities)
- `recommended_embedding_mode: "hosted" | "local" | "none"` (system recommendation; prefers the local embedding NIM whenever a supported GPU will actually be free for it — placement-aware: devices holding an active Teacher/Student NIM are excluded, when a local Teacher is recommended but not yet deployed the auto-placer's pick is reserved for it, and automatic eligibility requires both the floor from `EmbeddingDeploymentConfig` and an exact detected-name match in the pinned support matrix)
- `active_local_nim_residents[]: { project_id, project_name, local_nim_deployment_id, role, model_name, nim_container_image, gpu_assignment, status }` (non-secret summaries of Blueprint-managed `starting | running` NIMs; used to explain reuse and replacement choices)

Recommendation logic: first compute the quality-ranked local Teacher whose hardware contract this host satisfies. An exact healthy Blueprint-managed Teacher resident is auto-reused only when it matches that recommendation; a different resident remains visible and the SME chooses whether to keep or explicitly replace it. Otherwise prefer hosted Teacher (instant, no startup wait) when `NVIDIA_API_KEY` is configured, while still surfacing the local option. Prefer local embedding NIM when local deployment is available and a suitable GPU will remain free for it (eliminates rate limits, no hosted transport constraints). A host whose every GPU is below the Teacher floors but has a supported GPU at/above the 24 GB embedding floor still recommends local embeddings — that host class runs local embeddings with a hosted Teacher. When `NVIDIA_API_KEY` is not configured and local deployment is available, recommend local Teacher.

**F-amendment NIM-FTU-Local-Peer (2026-05-18).** Restores spec compliance after a Phase H (2026-05-12) regression that had over-simplified `recommended_teacher_mode` to always `"hosted"`. Three normative additions:

1. **Auto-pick by validated quality, then gate by hardware.** Filter to entries with `"teacher" ∈ eligible_roles`, non-null `local_deploy_metadata`, and at least one GPU satisfying `nim_gpu_memory_minimum_gb` plus optional `nim_compute_capability_minimum`. Sort by the curated quality rank first; memory is eligibility, not a quality proxy. The 2026-07-24 policy is: Nemotron 3 Nano Omni on ≥80 GB, compute capability ≥9.0; CR3-Nano on ≥56 GB when Omni is ineligible; Cosmos Reason2 2B on 36–55 GB. GPU memory comparisons allow a 1% reporting tolerance because `nvidia-smi` can expose a nominal 80 GB card as approximately 79.6 GiB; this recognizes the documented card tier without admitting a meaningfully smaller device. CR3-Super and Cosmos Reason2 8B remain selectable but do not become defaults merely by consuming more memory. This ranking follows the completed 3-model × 3-dataset long-horizon ICL matrix: Omni led the two multiclass datasets, while CR3-Nano beat Super on two of three datasets and Super won only VisA. Omni's 0.68–1.57% schema-error rate remains a disclosed tradeoff; CR3 had zero. On supported 24–35 GB GPUs no local Teacher fits, but the embedding-only path below remains available. Canonical implementation: `services.environment._pick_local_teacher_recommendation` (`_LOCAL_TEACHER_PREFERENCE_RANK`).
2. **Hybrid-by-default when both paths are viable but no preferred Teacher is resident.** When `NVIDIA_API_KEY` is configured AND a local Teacher fits, `recommended_teacher_mode` stays `"hosted"` (ready instantly via the seeded hosted default, Step 3.7 Flash — §4.8) BUT the local-teacher recommendation fields stay populated so the FTUE can offer the measured-best local Teacher as a peer card. An exact compatible running resident overrides this theoretical-fit rule only when it matches the current quality recommendation: `recommended_teacher_mode="local"`, and fresh-project creation attaches and selects the matching local config before returning. A different resident is reported as a keep/replace choice rather than silently overriding the current policy.
3. **New `EnvironmentAssessment` fields (additive).** Three optional fields surface the concrete recommendation to the frontend without it having to re-derive: `recommended_local_teacher_model_name: str | None`, `recommended_local_teacher_image: str | None`, `recommended_local_teacher_gpu_memory_minimum_gb: int | None`. Null when no teacher-eligible local model fits the GPU.

Fresh-project adoption and background-deploy dispatch: after seeding the project catalog, `POST /v1/projects` MUST scan durable active residents. When an exact compatible running Teacher matches the host's current quality recommendation, it MUST attach a project-local endpoint and set `teacher_model_config_id` to that matching config before returning. The FTUE queues no Teacher deploy and auto-skips when embeddings need no action; on a multi-GPU host it may continue solely to collect/dispatch a separately recommended embedding NIM. A different resident is not silently adopted: Screen 2C names it and obtains the SME's keep/replace decision. Otherwise Screen 2C fires one `POST /v1/projects/{id}/local_nim/deploy` per queued model. The deploy endpoint repeats the exact-reuse check to cover later model changes and races. No second Teacher container, model reload, or ownership transfer occurs on reuse.

**Preflight (per-model, runs at deploy time; reuses §9.5.2 checks 1–4 plus model-specific validation):**

1. Docker available (`docker info`).
2. NVIDIA Container Toolkit (GPU passthrough confirmed).
3. GPU memory: fast-fail against the model's published support-matrix minimum (`nim_gpu_memory_minimum_gb`).
4. `NGC_API_KEY` configured (required to pull NIM container images).
5. Docker registry authenticated to `nvcr.io` with the configured NGC key (password on stdin, never argv).
6. Container image pullable from NGC.
7. Model-specific profile check: run `list-model-profiles` against the target image to confirm a compatible profile exists on the current hardware. A shared-image Teacher probe MUST receive the same `NIM_MODEL_SIZE`, pinned `NIM_MODEL_PROFILE` (when configured), and size-specific `NIM_SERVED_MODEL_NAME` as the real deploy, so the probe cannot validate the image's default sibling model. A custom-checkpoint Student retains the base `NIM_MODEL_SIZE` but MUST NOT inherit the base model's pinned `NIM_MODEL_PROFILE`: that profile selects the image's bundled weights and conflicts with the read-only checkpoint selected by `NIM_MODEL_NAME`. Its probe and real launch let NIM select a checkpoint-compatible profile, and its real launch uses the Student-specific served name. If no runnable profile is available, deployment fails with a clear error. A published single-model NIM that does not ship the optional standalone utility is inconclusive rather than incompatible: continue to the bounded serve health and served-model checks, which remain authoritative. This exception never applies to a shared-image selection.

If preflight fails, the system generates an Action Request with the exact `docker run` command and prerequisites (existing §10.3 pattern) so the SME can hand off to infrastructure.

**Container lifecycle (persistent, named):**

1. Construct `docker run` with: pinned container image, name-only `-e NGC_API_KEY` forwarding, NIM cache mount (`~/.cache/nim:/opt/nim/.cache`), loopback-only host port → container port 8000 (`127.0.0.1:{host_port}:8000`), `--runtime=nvidia`, GPU assignment (`--gpus`), `--shm-size=32GB`, named container (`--name`). System-managed NIMs are unauthenticated implementation services and MUST NOT be published on every host interface. The Docker client subprocess receives the credential through its private child environment; neither the value nor a `KEY=value` assignment may enter the process argument vector or operator-visible command. Most supported images run as the host UID so the cache stays writable. Nemotron 3 Nano Omni `1.7.0-variant` MUST retain its declared `nvs` user because its startup calls `getpwuid()` and exits for an arbitrary host UID; the orchestrator MUST make only the shared cache parent/scratch directories writable to that user, without recursively changing cached model artifacts. NeMo Retriever VL NIM 2.0.0 MUST receive `NIM_PRECISION=fp16` (the unset-precision SM120 path requires a cuDNN plan directory its entrypoint skips creating) and a `NIM_MODEL_PATH` below `/opt/nim/.cache` so model downloads survive container replacement.
2. Start container. Poll `/v1/health/ready` up to `NIM_STARTUP_TIMEOUT_S` (default: 1200s). Every transition to a terminal `failed` state for a deployment whose container may still exist (health timeout, served-model verification failure, inference-probe failure) MUST best-effort stop and remove the named container: a terminal row is invisible to the one-NIM-per-GPU placement scan, so an untouched container would keep holding the GPU's VRAM and port as an unmanaged resident. Teardown targets the persisted container id when available (container names are project+role scoped and reused across deployment generations).
3. **Served-model verification (Teacher deployments).** Before marking the deployment healthy, the system MUST confirm the container is genuinely serving the *requested* model — `/v1/health/ready` returning and `/v1/models` reporting the requested name are NOT proof the right weights are loaded. Verification queries `/v1/metadata` for the actually-loaded model slug and confirms the requested model has real weight files in the NIM cache (not a config-only stub). On a contradiction — a `/v1/metadata` loaded-model mismatch, or zero non-trivial weight files for the requested model — the deployment is marked `failed` with an actionable `status_reason` instead of being marked healthy. This guards the silent wrong-model fallback observed with `cosmos3-reasoner` at `NIM_MODEL_SIZE=nano`: when the nano NGC profile fetch timed out the NIM logged `Detected 0 compatible profile(s)`, silently fell back to the cached SUPER weights, and kept reporting `served_model_name=…nano…` (the nano cache was a 52K config-only stub with zero weight files). Verification is Teacher-role only (Students mount checkpoints; the embedding NIM uses a different cache). Passing verification is followed by a minimal real-inference probe (a 1-token `/v1/chat/completions` call carrying the Blueprint source header): `/v1/health/ready`, `/v1/models`, and `/v1/metadata` keep answering 200 from the surviving HTTP front-end after a vLLM EngineCore death (observed live: CUDA illegal-memory-access under load), so only the inference path proves the engine is alive. An affirmative probe failure (HTTP error, connection refusal) marks the deployment `failed` with an actionable `status_reason` and stops the container; a probe timeout is inconclusive and passes adoption — a busy but healthy NIM queues completions for tens of seconds under load, and the probe must never tear down a serving deployment on slowness alone. The same verification + probe gate runs when restart recovery re-adopts a running container. The probe is Teacher-role only: embedding NIMs do not serve `/v1/chat/completions` (their inference path is live-verified at provider resolution, §5.5.1), and the Student lifecycle runs its own smoke inference (§9.5.2 step 3). On the same healthy transition the system also auto-sets the endpoint's per-prompt image cap (`NIM_MAX_IMAGES_PER_PROMPT`) from the served NIM, so ICL is not silently truncated by a version-specific default.
4. On healthy: auto-register the local endpoint in the model catalog (Teacher) or set `embedding_provider=self_hosted_nvclip` with the local endpoint URL (embedding NIM — enum value name retained for backwards compatibility); the embedding NIM's healthy transition also re-resolves the provider for every non-archived project and drains pending embedding work (§5.5.1). The SME does not manually configure URLs.
5. Container persists until explicitly stopped by the SME or backend shutdown.

**Port allocation:** each model type has a preferred host port (Teacher: `LOCAL_NIM_TEACHER_PORT`, default 8000; embedding NIM: `LOCAL_NIM_NVCLIP_PORT`, default 8001 — env var name retained for backwards compatibility). The container always listens on port 8000 internally. If the preferred host port is occupied, the system uses the next available port. The resolved endpoint URL is persisted.

**One-NIM-per-GPU invariant (F49 amendment, 2026-05-19):** at most one NIM container is `starting` or `running` on any GPU at any time, regardless of how much VRAM the math suggests is free. This is a backend-enforced invariant, not a frontend convention.

Before placement, a Teacher deploy with `replace_resident=false` MUST scan all non-archived project databases for Blueprint-managed active residents. A `running` Teacher with the exact same runtime identity — model name, pinned container image, model size/profile, and other container-affecting catalog environment — is reused across projects. The consuming project gets its own enabled `local_system_managed` endpoint attachment linked to the owner deployment; the owner keeps lifecycle control. If that resident later stops or fails, every attached project endpoint is disabled. When a new generation of that exact runtime later passes the healthy-adoption gate, the system MUST reattach projects whose selected Teacher still points at a disabled Blueprint-managed attachment from the prior resident; hosted, self-hosted, and differently selected configs are not changed. Proposal dispatch MUST reject disabled or hard-unhealthy endpoint state before making an HTTP request, even when the old host port currently answers.

Fresh-project creation performs the same exact scan immediately after catalog seeding. If a match exists, it attaches the endpoint and selects the matching Teacher before the create response returns; a hosted API key does not outrank already-running local infrastructure. Project creation remains available if the host scan itself fails, falling back to the effective seeded default and the normal setup choice.

To start a different NIM on an occupied GPU, the orchestrator MUST first transition the resident `LocalNimDeployment` to `stopped` — the **replace** semantics. Replacement is opt-in per call (`replace_resident=true` on the `POST :deploy` request, §10.2.X — defaulted by the Student NIM lifecycle, §9.5.2). Without it, the router returns `409 gpu_occupied` or `409 gpu_exhausted` with a non-secret `resident` summary, `matches_requested_model`, and `can_replace`. The FTUE MUST present the named model and owning project and let the SME choose **Keep current** or **Replace NIM**; it MUST NOT replace on the first click. If an exact compatible Teacher is still `starting`, return `409 resident_starting` with `can_replace=false` so the SME can wait/check again rather than destructively restarting it. Each displaced deployment's `displaced_by_deployment_id` and `displaced_at` are persisted on its `LocalNimDeployment` row for audit (§13.15).

**Post-onboarding compatible-model chooser and safe replacement:** NIM Configuration MUST list every Teacher entry for which `local_deployable_models[].fits=true` and a project ModelConfig with `teacher` role plus `local_deploy_metadata` exists. It marks `recommended_local_teacher_model_name`, the running resident, and each hardware floor, but deploys only the explicit selection. The first request always sends `replace_resident=false`; only a structured 409 naming the exact floor-qualified replacement target may open the confirmation. A confirmed retry pins that `gpu_assignment` and sends `replace_resident=true`. For a model change, `activate_on_success=true` is persisted on `LocalNimDeployment`; the Project's `teacher_model_config_id` changes only after the local endpoint passes health, served-model, and inference verification (or immediately after exact healthy-resident reuse). When a confirmed replacement fails during background preflight, container startup, health polling, served-model verification, or restart recovery, the orchestrator MUST best-effort requeue every displaced Teacher/embedding resident identified by `displaced_by_deployment_id`. Restoration failure is logged and surfaced operationally but does not conceal the requested deployment's original failure.

NIM Configuration MUST also expose the deployment-scoped NeMo Retriever VL
embedding lifecycle after onboarding. Its first deploy request sends
`role="embedding"` and `replace_resident=false` even when the placement-aware
`embedding_deployment.fits` value is false: that value means no GPU currently
meets the conservative automatic recommendation contract (free placement,
memory floor, and exact pinned-matrix name), not that a manual fallback attempt
is prohibited. The backend remains authoritative and either returns a
floor/placement failure or a
structured 409 naming the exact replaceable resident. Only the latter opens a
Keep-versus-stop-and-start confirmation; the confirmed retry pins the returned
GPU and sends `replace_resident=true`. A running embedding deployment owned by
the current project exposes the normal `:stop` lifecycle action. Deploy and
stop transitions invalidate environment/project state so provider
re-resolution and the hosted-or-pHash fallback remain visible without a
backend restart. A different project's embedding resident is named but its
lifecycle stays with the owning project.

The self-hosted embedding override on the same screen MUST prove the embedding
operation rather than treating a generic model-list response as sufficient.
Both Test Connection and Save send a credential-free request to
`POST {base_url}/embeddings` using the deployment config's exact NeMo Retriever
model and `input_type`, and require one finite vector of the configured 2,048
dimensions. Test is transient. Save repeats the proof before persisting
`provider=self_hosted_nvclip` and the normalized URL, then immediately
re-resolves every non-archived project. It MUST NOT switch to a different URL
while a Blueprint-managed embedding deployment is active; the SME stops that
resident explicitly first. Re-saving the exact active managed URL is an
idempotent live verification and preserves its GPU assignment.

**Placement policy (deterministic):** a new deployment claims the lowest-indexed GPU whose `LocalNimDeployment` rows are all in a terminal state (`stopped` / `failed`). The auto-placer never returns a GPU whose residents are active. Embedding deployments additionally skip free GPUs below the embedding memory floor (`EmbeddingDeploymentConfig.gpu_memory_minimum_gb`, seeded 24 GB for NIM 2.0.0), so a heterogeneous host (say 80 GB + 8 GB) never lands the embedding NIM on a device below the eligibility floor; when free GPUs exist but none meets the floor, the placer raises `GpuExhaustedError` pointing at the real fixes (larger GPU or the hosted provider) rather than suggesting replace semantics. Memory alone does not establish embedding compatibility: automatic recommendations also require the detected GPU name to exactly match a SKU in the pinned support matrix. Unrecognized hardware stays hosted/pHash unless the operator deliberately requests the documented fallback path, whose live preflight remains authoritative. When every GPU has an active resident, the auto-placer raises `GpuExhaustedError` and the router returns `409 gpu_exhausted`; the caller chooses whether to explicitly opt into replace semantics.

**Single-GPU hosts:** support exactly one NIM at a time, which yields three viable splits. (1) The FTUE Screen 2A "Run locally" path deploys the Teacher only; image embedding falls back to pHash diversity (§5.6) until the operator either switches to a hybrid path where the embedding NIM runs hosted via `NVIDIA_API_KEY`, or explicitly stops the Teacher and runs an embedding deploy. (2) Hybrid: local Teacher on the GPU, hosted embeddings via `NVIDIA_API_KEY`. (3) The inverse split for the small-GPU host class: one supported GPU below every Teacher floor but at/above the embedding floor (24 GB seeded) runs the local embedding NIM with a hosted Teacher — `recommended_embedding_mode="local"` covers exactly this case. Below 24 GB, use hosted embeddings when configured or pHash diversity. Student NIM benchmarking on a single-GPU host explicitly invokes the replace semantics: the resident Teacher is stopped, the Student runs evaluation + benchmark, the Student stops, and the orchestrator best-effort auto-restores the displaced Teacher on the same GPU (§9.5.2 step 0 and step 9).

**Multi-GPU hosts:** each NIM gets its own GPU. Every role claims the lowest-indexed compatible free device at dispatch time; roles do not own fixed device numbers. Embedding placement additionally skips free devices below its configured eligibility floor. Co-location on the same GPU is not supported in v1.

**Preflight:** for every deployment, preflight attempts to verify a runnable model/profile on the assigned GPU via `list-model-profiles`. If a single-model published image omits that optional utility, the probe is recorded as inconclusive and the bounded serve health plus served-model checks become authoritative; a shared image must never use this exception. On occupied-GPU replace, preflight runs AFTER the resident is stopped so the GPU-memory check reflects post-displacement free memory.

**Empirical motivation:** the empirical 2026-05-19 finding (Cosmos Reason2 8B / A100-80GB, recorded in README under "One-NIM-per-GPU policy") demonstrated that Cosmos Reason2's NIM container hardcodes `gpu_memory_utilization=0.9` in its vLLM backend; its profile selector reads currently-free GPU memory and asserts each profile's requirement against it, so any neighbor process — including a 12 GB embedding NIM — drops "free" below the 90%-of-total threshold every profile requires, and Cosmos refuses to start with `Detected 0 compatible profile(s)`. Re-confirmed 2026-07-13 on Cosmos 3 nano / H100 NVL 96 GB: a second teacher-class NIM on an occupied GPU refuses startup (`Free memory … less than desired GPU memory utilization (0.9, 83.78 GiB)`; container exits 0), while the small TRT-based embedding NIM can physically co-locate in the leftover headroom on the larger card — the invariant is therefore enforced as policy, not just physics. The motivation is recorded in README; the invariant above is the contract.

**Restart recovery:** `LocalNimDeployment` ownership is durable in each project's SQLite database; the named Docker container also survives a Blueprint backend restart. On backend startup, the system inspects only containers it previously launched (by persisted container name). If the container is still running and `/v1/health/ready` returns ready, re-run the healthy-adoption gate (served-model verification + inference probe, lifecycle item 3) and on pass rebind it and restore the endpoint registration. Existing consumer-project attachments remain valid because they reference the same persisted deployment id and endpoint URL. If the container is running but not ready, stop and remove it, mark the row stopped, and disable all consumer attachments — a legitimately mid-startup container lands here too (recovery probes once), and a deterministic teardown + "Redeploy locally" beats leaving an unmanaged container to finish starting outside the placement accounting. If the container is not running, mark the row stopped and disable its attachments. The system MUST NOT perform generic Docker orphan discovery across all containers on the host.

**Deployment state persistence:** each active local deployment is tracked as a `LocalNimDeployment` record (§13.15) with container name, image ref used, role, host port, GPU assignment, endpoint URL, and status.

### 1.6 TAO Endpoint Configuration (Required When Student Training Enabled)

Student Training uses the TAO API as a remote execution backend. TAO FTMS (Fine-Tuning Microservices) does not need to run on the same machine as this system; the system communicates with TAO entirely via its REST API. TAO typically runs on GPU-equipped infrastructure (DGX, cloud GPU instances, on-prem clusters) while this system may run on separate compute.

**Artifact storage model:** TAO FTMS manages job execution and artifact generation. Durable outputs (checkpoints, logs, metrics) are stored in TAO's configured workspace backing storage (cloud storage or mounted filesystem), not on the FTMS host's local disk. This system is a downstream consumer: it retrieves the subset of artifacts it needs via the TAO Jobs API (download, selective download, logs endpoints) and imports them into the project's local artifact store (`{project_dir}/artifacts/`). The system MUST NOT assume direct filesystem access to TAO's workspace storage.

TAO endpoint configuration is **deployment-level** (shared across all projects), not per-project.

Required configuration:

```text
TAO_API_BASE_URL: https://<tao-host>/api/v2
TAO_API_KEY: <auth credential>
TAO_ORG_NAME: <organization name>
TAO workspace identity: provisioned in deployment.db by `vlm-feedback-loop tao-bootstrap`
```

- `TAO_API_BASE_URL`: base URL for the TAO API v2. No default; MUST be explicitly configured when Student Training is enabled.
- `TAO_API_KEY`: authentication credential for the TAO API. This is an **application-level abstraction** — the system uses this value as the `Authorization: Bearer` token for all TAO REST API calls. On a stock FTMS v2 deployment, native authentication is a two-step flow: the client sends the NGC Personal API Key to `POST /api/v2/login`, receives a JWT, and then uses that JWT as the bearer token for `/orgs/{org_name}/...` calls. The raw NGC Personal API Key is generally not the same token used for Jobs API requests. `TAO_API_KEY` stores the appropriate bearer token for the operator's deployment (typically the JWT). MUST be stored as a secret (same policy as NIM API keys, §12). MAY be pre-configured via environment variable (`TAO_API_KEY`) or the canonical `.env` file (§1.9).
- `TAO_ORG_NAME`: organization name used in TAO API paths (`/api/v2/orgs/{org_name}/...`). Required for all TAO API calls.
- TAO workspace identity is the UUID of the workspace that owns all datasets and jobs created by this Blueprint deployment. It is stored in `deployment.db` by `vlm-feedback-loop tao-bootstrap`; see §9.7.8.

**Single source of truth.** The deployment-scoped `TAODeploymentConfig` singleton (§13.19) is the only persistent store for `tao_workspace_id`, `tao_workspace_name`, `tao_workspace_bucket`, and the two `tao_workspace_s3_endpoint_url_*` fields. `vlm-feedback-loop tao-bootstrap` writes those values directly there; runtime services read that record exclusively. The S3 credentials (`TAO_WORKSPACE_S3_ACCESS_KEY`, `TAO_WORKSPACE_S3_SECRET_KEY`) come from the process environment or canonical `.env`; the DB stores only their env-var-name references.

Connection test:

- The system probes the TAO endpoint on configuration (`GET /api/v2/orgs/{org_name}/jobs?limit=1`) to confirm access. On auth failure, show: *"Could not connect to TAO. Verify the API URL, key, and organization name."*
- Probe result determines Student Training availability in the Scale Up hub (Overview §7) and the Student Training screen.

TAO endpoint setup is an infrastructure task completed before the Scale-Up onboarding flow, similar to self-hosted NIM setup (Overview §6 note). When TAO is not configured, the system surfaces an Action Request CTA (§10.3) so the SME can generate a structured handoff to their administrator with the required config fields pre-filled.

**Workspace + base-experiment prerequisites (§9.7.8).** On a stock TAO FTMS deployment, a workspace (S3-backed storage owner for datasets and job outputs) and the selected Cosmos base experiments MUST exist before a training suite can run. Workspace bootstrap remains a deployment-operator task. Base experiments support two paths: **self-service (default)**, where **Start Training** provisions each selected missing base in a tracked background run (with `vlm-feedback-loop tao-pull-base-experiments` retained for eager/operator use); and **admin handoff (fallback)**, where an administrator provisions base experiments off-Blueprint for air-gapped or policy-separated deployments. The resulting IDs are recorded on the corresponding `ModelConfig.tao_base_experiment_id` rows across projects.

### 1.7 Workspace and Project Storage

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

**Artifact storage:** large artifacts (raw model responses, parsed/normalized JSON, validation reports, run logs) MUST be stored as files under `{project_dir}/artifacts/`, not as database blobs. Database records store only metadata and file-path references (`_ref` fields). This keeps SQLite small and portable while preserving the full evidence trail for auditability and reproducibility.

**Images are not copied into the project directory.** The system records the original filesystem path as `storage_ref` on the Example record (§13.8). The backend provides a filesystem browse endpoint (§10.2.10) that the frontend uses to let the user navigate the backend host's filesystem and select image directories or files. The backend resolves all selections to absolute paths. A direct path entry field is also available for scripting and power-user workflows. Images MUST remain accessible at their recorded paths for the lifetime of the project. If paths change (drive reorganization, NFS remount, container bind-mount change), a bulk path remapping endpoint is available (§10.2.11). `storage_ref` values reflect the backend process's filesystem view; when the backend runs in a container, mount image directories at identical paths (e.g., `-v /data/images:/data/images`) so that `storage_ref` values are valid in both contexts.

`storage_ref` is a locator, not a durable authorization decision. Every
production read MUST re-apply the current §10.2.10 `IMAGE_ROOT` policy, open a
regular file without following mutable path components, and consume the bytes
from that authorized descriptor. A check-then-reopen sequence is not
sufficient: a rename or symlink retarget between those operations could change
the inode being served or sent to an external service.

Required configuration:

- `WORKSPACE_ROOT`: absolute path to the workspace root directory. No default; MUST be set before first use. The system MUST create the directory structure if it does not exist.

### 1.8 Implementation Stack

**Frontend:** React + TypeScript + Vite. Client-rendered single-page application served as static assets. No server-side rendering required. **UI library: NVIDIA KUI Foundations + Tailwind.** KUI Foundations (`@kui/foundations-react-external`, `@kui/foundations-design-tokens`) provides component primitives (buttons, inputs, dialogs, tabs, cards, badges, progress, tables) and the NVIDIA design language. Tailwind provides layout composition (page shells, split panes, sticky headers/footers, responsive review layouts, spacing). This combination aligns with recent NVIDIA Blueprint frontends (Retail Catalog Enrichment, Retail Agentic Commerce, RAG Blueprint) where KUI Foundations + Tailwind is the emerging standard. App-level wrapper components (schema field row, evaluation summary strip, gate status card, labeling action bar, action-request panel, compare-variant card) provide product-specific consistency on top of KUI primitives.

**Backend:** Python + FastAPI. Pydantic for request/response validation and record schemas. SQLAlchemy for ORM and database access. httpx for outbound async HTTP (NIM, TAO, embedding NIM endpoints). Jinja2 for prompt template rendering (Appendix D). Standard `logging` with a JSON formatter for structured operational logs (§11), writing to both stdout and `{project_dir}/logs/*.jsonl`. tiktoken for token counting (`encoding_for_model` when the model maps cleanly, `cl100k_base` fallback); the existing `RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN` (§6.2) absorbs tokenizer mismatch.

**Database:** SQLite per project, stored at `{project_dir}/project.db` (§1.7), with WAL mode enabled. SQLAlchemy models MUST avoid SQLite-specific query behavior to preserve database portability. The public Alembic lineage starts at `v1_0001`, which creates the complete v1 schema; private pre-release revisions and databases are unsupported and require a fresh workspace. On startup, the backend MUST detect pending post-v1 migrations for each opened project database and apply them before proceeding. The schema version MUST be tracked in the database. Before Alembic revision discovery, a nonempty schema MUST contain one canonical, nonempty revision row; otherwise the backend MUST create a validated recovery backup and refuse the database without applying migrations. Only an empty schema or a sole canonical empty `alembic_version` table may initialize as fresh. Public v1+ project databases are upgradeable, not disposable; labeled data, Guidance versions, evaluation history, and training lineage MUST be preserved across schema changes. Before applying a post-v1 migration, the backend MUST copy `project.db` to `project.db.backup.{ISO 8601 timestamp}` (e.g., `project.db.backup.2026-03-30T14-22-07Z`). If a migration fails, the backend MUST surface the error and the backup file path, and MUST NOT proceed with a partially migrated database. On startup, the backend MUST run `PRAGMA quick_check` on each project database; if the check fails, surface a clear error with the database path and instruct the user to restore from backup (do not attempt auto-repair).

**SQLite write-discipline (v1):** multiple concurrent writers exist within a single backend process (interactive labeling, evaluation result persistence, CLIP embedding updates, Batch Labeling progress writes). SQLite WAL mode permits concurrent reads alongside a single writer, but only one write transaction can make progress at a time. The following guardrails are normative:

- SQLite `busy_timeout` MUST be configured to at least 5000 ms.
- All write transactions MUST be short-lived. Background tasks and request handlers MUST NOT hold a write transaction open across outbound HTTP calls, model inference calls, retry/backoff waits, polling waits, sleeps, or any other long-running operation. The pattern is: perform the long-running operation, collect results in memory, then open a short write transaction to persist them.
- The implementation uses a simple in-process async mutex per project for the ingestion family (skeleton inserts, pHash updates, embedding updates) and latency-sensitive interactive writes (Guidance save/evolution, label save/skip/restore, project settings, and review-selector cursor persistence). These paths MUST share the same mutex. CPU/file validation, pHash computation, image normalization, base64 encoding, and outbound HTTP MUST happen before acquiring it; only the short SQLite read-modify-write window belongs inside. This prevents an interactive save from entering SQLite's multi-second busy wait behind a background batch. Other run-state writers remain safe under the short-transaction rule above.

**Background execution:** In-process asyncio task orchestration for I/O-bound work (NIM inference calls, CLIP embedding HTTP requests, evaluation concurrent inference). A small thread pool for CPU-bound work that MUST NOT block the event loop (pHash computation, file I/O). No external task queue (no Celery/Redis). Bounded concurrency for evaluations and batch labeling is provider-aware: `EVAL_CONCURRENCY_HOSTED` / `BATCH_LABEL_CONCURRENCY_HOSTED` (default: 1) against hosted endpoints, `EVAL_CONCURRENCY_SELF_HOSTED` / `BATCH_LABEL_CONCURRENCY_SELF_HOSTED` (default: 8) against self-hosted/local NIMs. Durable run/task records in the project database enable recovery on restart (§1.3).

**Package management:** Python dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`). Frontend dependencies are managed with **pnpm**. This aligns with recent NVIDIA Blueprint repositories (Retail Catalog Enrichment, Retail Agentic Commerce, RAG Blueprint).

**Local launch:** A single command MUST start both frontend and backend for local development. For packaged deployment, Docker Compose MAY be used but is not required for development.

**Local NIM prerequisites script (`setup-local.sh`):** an optional script shipped alongside the blueprint for preparing a bare-metal machine for local NIM deployment (§1.5 Mode C). The script is run once by the SME or administrator before first launch — it is NOT auto-triggered from the web UI. The script MUST: (1) check for NVIDIA drivers (fail with clear message if missing — driver install is outside scope); (2) install Docker if missing; (3) install NVIDIA Container Toolkit if missing; (4) configure the Docker runtime for NVIDIA GPU passthrough; (5) validate GPU access (`docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`); (6) optionally pre-pull NIM container images to save time on first deploy. DGX and cloud GPU instances typically come with Docker and toolkit pre-installed, so the script is mainly needed for workstations. When the onboarding flow detects missing prerequisites (Overview §6 step 2), it references this script.

**Development frontend/backend routing:** when the React/Vite frontend and FastAPI backend run on different origins during development (e.g., `localhost:5173` and `localhost:8000`), the implementation MUST provide a deployment-scoped development routing strategy so browser requests succeed. **Preferred:** configure the Vite dev server to proxy backend paths (at minimum `/v1/` API routes and the SSE events path) to the FastAPI origin. The proxy MUST be configured to support SSE passthrough (long-lived chunked connections). **Alternative:** configure FastAPI `CORSMiddleware` with an explicit allowlist of development frontend origins (e.g., `http://localhost:5173`). If credentials are used, wildcard origins (`*`) MUST NOT be used. In production, when the frontend and backend are served from the same origin or behind a reverse proxy, neither a dev proxy nor CORS is required. If production deployment remains cross-origin, CORS configuration MUST be provided. Development routing and CORS configuration are deployment-scoped, not per-project.

### 1.9 Configuration

The system uses two configuration sources with distinct scopes.

**Environment variables / `.env` file** for secrets and deployment-bound endpoints:

- `NVIDIA_API_KEY`: hosted NVIDIA NIM API key (§1.5 Mode A, §5.5)
- `NGC_API_KEY`: NGC API key for pulling NIM container images (§1.5 Mode C); required for local NIM deployment. Generated at `https://org.ngc.nvidia.com/setup/api-key`.
- `TAO_API_KEY`: TAO API credential (§1.6)
- `TAO_API_BASE_URL`: TAO API base URL (§1.6)
- `TAO_ORG_NAME`: TAO organization name (§1.6)

**Canonical `.env` location:** `~/.vlm_feedback_loop/.env`, colocated with `~/.vlm_feedback_loop/config.yaml`. This is a deployment-scoped file, not a project-scoped file.

The backend MUST NOT search for `.env` in the current working directory, `WORKSPACE_ROOT`, or project directories. An explicit override MAY be provided via CLI `--env-file <absolute-path>` or environment variable `VLM_FEEDBACK_LOOP_ENV_FILE=<absolute-path>`. When an override is provided, the backend MUST load only that file and MUST NOT fall back to the canonical location. If no `.env` file exists at the selected location, startup continues normally; missing required secrets are surfaced only when a workflow requiring them is used.

For containerized/CI use, secrets are injected as standard process environment variables and no `.env` file is needed. See §12.1 for `.env` file security requirements.

**YAML config file** for non-secret application settings:

Location: `~/.vlm_feedback_loop/config.yaml`. Settings include `workspace_root`, `log_level`, UI defaults, thresholds, and feature flags. Secrets are stored separately (§12.1).

**Precedence (highest wins):**

1. Explicit process environment variables
2. Explicit env file path from `--env-file` or `VLM_FEEDBACK_LOOP_ENV_FILE`
3. Default env file at `~/.vlm_feedback_loop/.env`
4. Config file values from `~/.vlm_feedback_loop/config.yaml`
5. Built-in defaults (Appendix A.4)

**First-launch behavior (CLI/bootstrap-first):** deployment-level configuration must exist before normal backend startup. If `~/.vlm_feedback_loop/config.yaml` does not exist, the backend MUST fail fast with a clear message directing the user to run a bootstrap command (e.g., `vlm-feedback-loop init`). The bootstrap command prompts for `WORKSPACE_ROOT`, writes a commented `config.yaml` template (`WORKSPACE_ROOT` is the only active key; every other Appendix A.4 default appears as a commented documentation line, so un-overridden settings keep tracking the shipped defaults across upgrades), generates a commented skeleton `.env` file at `~/.vlm_feedback_loop/.env` (documenting required variables without values), and exits. The backend then starts normally. The application does **not** include a first-launch web setup wizard.

After backend startup, service connection setup remains in the web app. A new
project enters the split NIM setup chain (`NIMNvidiaKeyPage` → optional
`NIMNgcKeyPage` → `NIMSetupGatePage`) when a credential or deployment decision
is needed; fully resolved configurations auto-skip it. The richer
`NIMConnectionPage` is the post-onboarding **NIM Configuration** surface.

---

## 2. Conventions

### 2.1 Reproducibility and Determinism

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

### 2.2 Entity Identifier Format

All backend-generated entity identifiers (`project_id`, `guidance_id`, `model_config_id`, `run_id`, `tao_job_id`, `dataset_export_id`, `pool_id`, `audit_event_id`, `student_model_id`, `local_nim_deployment_id`, `inference_invocation_id`, and similar record IDs) MUST use **UUID4** in canonical lowercase hyphenated string format (e.g., `f47ac10b-58cc-4372-a567-0e02b2c3d479`).

Content-derived identities (e.g., `schema_hash`, `prompt_hash`, artifact checksums) use hashes, not UUIDs. Human-readable display labels (e.g., Guidance `version_number`, model `model_name`) are separate fields, not encoded into the ID.

### 2.3 Key Definitions

**Schema-invalid output**
Response cannot be normalized into schema-valid JSON for all **Core fields** (invalid JSON, missing Core, invalid types/ranges/enums, conflicting duplicates preventing resolution).
Aux field errors MUST NOT block proposal validity and MUST NOT change schema-valid vs schema-invalid classification. Implementations MUST produce a structured validation report separating Core vs Aux errors (Sections 6.3.1, 13.1).

**Timeout**
Model invocation exceeded enforced deadline with no usable result.

**State**
Canonical example state: `Unlabeled` / `Auto-Labeled` / `Verified` / `Omitted`.

**Omitted**
Example excluded from workflow: MUST NOT be shown in SME review selector; excluded from Batch Labeling input selection by default. Omission is set by SME Skip.

**Test Pool**
A reserved subset of Verified examples used for evaluation and excluded from ICL. Serves both iteration feedback and Scale-Up gating.

**Pool membership**
Tag indicating a Verified example is assigned to the Test Pool. Managed automatically by the system via pool assignment routing (§4.3).

**Pool assignment routing**
Automatic routing based on the configured target fraction: the next Verified example fills an immediate shortfall, while later rebalancing uses the review selector's CLIP/pHash switchover. Verification outcome (Accept/Edit) does not influence pool assignment (§4.3.1–4.3.2).

**Evaluation snapshot (pool version)**
Frozen, immutable capture of pool membership at the time an evaluation run starts. Ensures reproducible evaluation against a fixed example set (§4.3.3).

**ICL eligibility**
Verified examples with `verified_outcome=Edit` that are not assigned to the Test Pool (see Section 6.2). Accepted examples are never selected for ICL.

**Auto-Labeled label**
Schema-validated model output produced without SME review (typically Batch Labeling). Stored with lineage; MAY be exported for training; is NOT ground truth (§3.2).

**Batch Labeling run**
Long-running job applying a selected Teacher to many Unlabeled examples to produce Auto-Labeled labels + run-level metadata.

**Model endpoint**
Configured OpenAI-compatible NIM base URL plus auth settings.

**Model name**
Identifier string used in OpenAI-compatible NIM `model` parameter; shown in UI.

**Model config**
Catalog entry binding endpoint + `model_name` + operational metadata (context window tokens required), `eligible_roles[]` declaring which roles the entry may serve (`teacher`, `student_base`), optional deployment metadata (NIM profile and precision; see §2.3 "NIM model profile" and "Profile metadata recording"), and persisted structured generation capability probe status (Section 6.2). Active selections per role are stored on the Project record (Sections 4.8, 13.4).

**NIM model profile**
Deployment-time selection that determines which engine NIM uses and how it is optimized. A profile implies: backend (TensorRT-LLM or vLLM), precision, optimization target (latency vs throughput), and tensor parallelism. For Cosmos Reason2, the documented packaged profile precisions are BF16 and FP8 (vLLM backend). Profiles can be pinned with `NIM_MODEL_PROFILE`; otherwise NIM automatically selects a compatible profile and logs the selection at startup.

**Model precision / quantization (NIM)**
The effective precision of a deployed Student. For packaged NIM profiles, precision is a property of the chosen profile (e.g., BF16, FP8). For TAO-quantized checkpoints, precision is a property of the quantization method used (e.g., FP8_DYNAMIC, W4A16), served through `NIM_MODEL_NAME` on the vLLM backend. The system MUST record profile identity/metadata for profile-based deployments, and quantization method for quantized deployments.

**Profile metadata recording (required for deployed Students)**
Any deployed Student endpoint MUST persist: `nim_model_profile_requested` (nullable; what was set, if any), `nim_model_profile_selected` (required; observed from deployment/startup logs), `nim_profile_metadata` (at minimum: backend/engine, precision, TP, optimization target), GPU type/count, and `quantization_method` (e.g., `FP8_DYNAMIC`, `W4A16`, or `none` for full-precision). This metadata MUST be attached to any latency/throughput benchmark outputs for reproducibility.

**Embedding provider**
Configured source for per-image CLIP-style embedding computation. Values: `hosted_nvclip` (default — name retained from the original NV-CLIP-only design; the actual model is governed by `embedding_model_id`), `self_hosted_nvclip`, `none`. Consistency enforced per project (§5.5.3).

**Embedding NIM endpoint**
NVIDIA NIM embedding endpoint exposing OpenAI-compatible `/v1/embeddings` with image input support. Hosted endpoint: `https://integrate.api.nvidia.com/v1/embeddings`. **Supported model: `nvidia/llama-nemotron-embed-vl-1b-v2`** (NVIDIA NeMo Retriever VL 1B v2; 2048-dimensional float vectors; requires `input_type` in the request body, default `"passage"`). Image input: base64 data URL (`data:image/<format>;base64,<data>`).

**CLIP embedding**
Fixed-length vector representation of an image used for review selector scheduling and optional filtering. "CLIP" here names the algorithm class (contrastive image-text embedding) rather than a specific model — the default provider is hosted NeMo Retriever VL embedding NIM; see "Embedding NIM endpoint" above for the supported alternates.

**pHash (perceptual hash)**
A compact 64-bit perceptual fingerprint computed inline at ingest (CPU-only, no external dependency). Similarity measured by hamming distance. Used as the baseline diversity signal for the review selector. See §5.6.

**pHash-diverse review selector**
Baseline selection strategy using available pHash values and hamming distance to avoid showing visually similar images consecutively. When no candidate hash is ready, selection falls back to deterministic `example_key` order. Deterministic given persisted scheduler state (Section 2.1).

**CLIP-diverse review selector**
Selection strategy using CLIP embedding cosine similarity to avoid showing semantically similar images consecutively. Preferred over pHash-diverse when CLIP embeddings are available. Deterministic given persisted scheduler state (Section 2.1).

**Visual budget mode**
Per-model declaration of which visual token control shape the model accepts. Values: `none` (no visual budget controls), `mm_processor_size` (accepts `size.shortest_edge`/`size.longest_edge`), `mm_processor_pixels` (accepts `images_kwargs.min_pixels`/`max_pixels`), `mm_processor_tiles` (accepts `max_num_tiles`). Stored on ModelConfig. See §6.9.

**Visual budget preset**
Named preset (**Fast** / **Balanced** / **High Detail**) that resolves to model-specific `mm_processor_kwargs` values based on `visual_budget_mode`. Controls per-image visual token spend, a cost/quality lever distinct from text-side sampling controls. See §6.9.

**Retry (interactive)**
User-initiated re-run of proposal generation for the same image after selecting a different Teacher model config and/or Guidance.

**Rationale note**
An opt-in `role="aux"` string field (`rationale_note`) controlled in Guidance; disabled by default. When enabled, it is requested last by the production Teacher, excluded from Teacher ICL, and never evaluated. When disabled, it is bypassed end to end. See §4.4.

**Export field mode**
Per-export setting controlling which field groups are included in the gpt turn of training dataset exports and Auto-Labeled output: `all` (rationale + Aux + Core; default), `aux_and_core` (Aux + Core, no rationale), or `core_only` (Core only). See §9.3.

### 2.4 Timestamps

All persisted timestamps MUST be stored in UTC using ISO 8601 with a `Z` suffix (e.g., `2026-03-30T14:22:07Z`). The frontend converts timestamps to local time for display.

---

## 3. System Invariants

### 3.1 Data Model Invariants

- Exactly four example states: `Unlabeled`, `Auto-Labeled`, `Verified`, `Omitted`.
- The Test Pool is a pool membership automatically managed by the system (§4.3). Pool members are reserved for evaluation and excluded from ICL. Both Accepts and Edits are candidates for the Test Pool. Pool assignments are durable once made (exception: a semantic Core change deletes Label records, which clears pool assignments; §4.4.1).
- Verified is the ground-truth source of record for evaluation, ICL selection, and default training split definition (excluding pools).
- Auto-Labeled labels change example state. Batch Labeling MAY store Auto-Labeled labels for Unlabeled examples, transitioning those examples to `Auto-Labeled` until SME review yields Verified. An `auto_labeled` Label is live only while its exact owning Example remains `state="Auto-Labeled"`; Skip discards that machine proposal (§4.5).
- SchemaCore stability invariant (mandatory):
  - Each field has an immutable `field_id` that persists across renames.
  - In-place Core edits are limited to: field rename, exact 1:1 enum value rename, and presentation metadata changes.
  - All other Core changes (add/remove field, type change, constraint change, allowed-value change, role change) are semantic and MUST trigger the schema evolution flow (§4.4.1): existing Verified and Auto-Labeled labels are deleted, examples return to `Unlabeled`, and the system preserves prior label data on the Example record for labeling-screen hints.
  - SchemaCore may also be extended by adding new fields with `role="aux"` in place (no label invalidation required).
- Guidance versioning preserves SchemaCore identity within a project: Description/Rules may change freely; semantic Core changes trigger label invalidation and return examples to `Unlabeled` (§4.4.1).
- Schema evolution invariant: when a semantic Core change occurs, all Verified labels and Auto-Labeled labels are deleted. Verified examples transition to `Unlabeled` with prior label data preserved as reference on the Example record (`prior_verified_label_ref`). Auto-Labeled examples transition to `Unlabeled` (labels generated under a stale schema are not trustworthy; their Operation Records are preserved for audit). The standard Interactive Labeling loop then re-labels these examples under the new schema, with prior-label hints for efficiency. Auto-Labeled data "is NOT ground truth" (§3.2); re-running Batch Labeling under the new schema is the correct recovery path.
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

1. ICL eligibility: only Edits not assigned to the Test Pool (definition and selection rules are in Section 6.2). Accepted examples are never ICL-eligible.
2. Prompt lineage is recorded: every inference/evaluation/batch labeling call persists ordered `icl_example_keys_used[]`.
3. Marker rule: model-visible prompt MUST NOT include `example_key`; examples use markers `E01..`.
4. Token budget enforcement MUST degrade gracefully by dropping the least query-relevant ICL examples (relevance-tail pruning, §6.2) rather than blocking progress.
5. Per-model prompt budgets: prompt packing MUST respect selected model context window and output cap.
6. Generation Controls persistence: sampling parameters (`temperature`, `top_p`, `seed`) and thinking toggle state MUST be persisted per invocation in Operation Records (§13.1).
7. Visual Budget persistence: `visual_budget_preset_key` and `visual_budget_params_effective` (the exact `mm_processor_kwargs` sent) MUST be persisted per invocation in Operation Records (§13.1). When the model does not support visual budget controls, both are null.

### 3.4 Failure & Integrity Invariants

1. Failure semantics distinguish schema-invalid vs timeout vs endpoint/transport error.
2. Student Training execution is validated independently of the Scale-Up Teacher-quality criteria: the suite-launch endpoint fails closed until TAO/workspace/timeout checks pass, at least one active-Guidance Verified training example remains outside the Test Pool, the active-Guidance Test Pool reaches the configured minimum, and each selected base model has the `student_base` role. A selected missing TAO base is non-blocking because first-use provisioning is the conditional first Training Jobs stage (§9.7.8).
3. Review selector selections MUST be reproducible (Section 13.3).

### 3.5 Operation Record Invariants

1. Invocation record persistence: system MUST generate `inference_invocation_id` before invoking model and MUST persist an operation record for all outcomes.
2. Evaluation uses the same prompt pipeline as Interactive Labeling.
3. User-requested interactive Retry and evaluation's explicit sequential retry pass produce distinct invocation records linked to the prior attempt. Resuming one unfinished Batch Labeling item is not a new retry: it MUST reuse that item's pending invocation record.
4. (removed)
5. TAO job observability is persisted: TAO jobs MUST have durable records, deterministic state mapping, and durable output references.

---

## 4. Data Model

### 4.1 Example States

**Unlabeled**

- Image exists without human-confirmed label.
- Eligible for interactive labeling review selector.
- Not used for ICL and not eligible for pools.

**Auto-Labeled**

- Image+label produced without SME review.
- Has one live `label_status=auto_labeled` Label for the exact Example.
- Used for training scale.
- MUST NOT be used for ICL and not eligible for pools.

**Verified (Ground Truth)**

- Image+label confirmed by human via Accept/Edit.
- Used for ICL selection (unless pooled), evaluation reference labels, and default training split (excluding pools).

**Omitted**

- Image excluded from workflow (Skip).
- Has no live machine Label; batch invocation history remains auditable.
- MUST NOT be presented in interactive labeling review selector.
- Excluded from Batch Labeling input selection by default.

### 4.2 Omission Metadata

Examples carry omission provenance when omitted.

Minimum required fields on the Example record:

- `project_id` (required)
- `example_key` (required; unique within project)
- `storage_ref` (required; absolute filesystem path to the original image; §1.7)
- `ingested_at: timestamp` (required; server timestamp)
- `source_metadata: object` (required; JSON; may be `{}`)
- `state: "Unlabeled" | "Auto-Labeled" | "Verified" | "Omitted"` (required)

Omission provenance fields (required when `state="Omitted"`):

- `omitted_source: "sme_skip" | null` (required when Omitted; MUST be null otherwise)
- `omitted_at: timestamp | null` (required when Omitted; MUST be null otherwise)

pHash field (required; populated by the restartable post-ingest sweep, §5.6):

- `phash: string` (required; 64-bit perceptual hash, hex-encoded; computed inline during ingestion)

Embedding fields (required when embeddings computed; see §5.5.5 for storage contract):

- `clip_embedding_present: boolean` (default: `false`; set `true` when embedding is computed and stored)
- `clip_embedding_dim: int | null`
- `clip_embedding_model_id: string | null`
- `embedding_provider: "hosted_nvclip" | "self_hosted_nvclip" | null`

Prior-label reference fields (set during semantic Core schema change, §4.4.1; null otherwise):

- `prior_verified_label_ref: string | null` (JSON snapshot of the prior Verified label data: label JSON, optional rationale, field values; null for examples that were never verified under a prior schema)
- `prior_verified_outcome: "Accept" | "Edit" | null` (the `verified_outcome` from the prior label; used by the review selector to prioritize prior Edits; null when no prior label exists)

Notes:

- When the embedding provider is the hosted embedding NIM, defaults are per §2.3 "Embedding NIM endpoint."
- System MUST NOT assume embeddings exist for all examples.
- Omission is set by SME Skip; `omitted_source` and `omitted_at` MUST be recorded automatically.

### 4.3 Test Pool

The Test Pool is a reserved subset of Verified used to measure quality without contaminating examples used as ICL context.

**Pool management is fully automatic.** The system routes newly Verified examples and rebalances pool membership continuously. No user action is required. The Test Pool grows alongside Verified labels, providing both iteration feedback during Interactive Labeling and quality metrics for the Scale-Up Readiness Gate (§7.3).

#### 4.3.1 Pool Assignment Routing

When an example becomes Verified, the system assigns it to exactly one destination:

1. If the Test Pool is below its target count → assign to Test Pool.
2. Else → non-pool (available for ICL selection if eligible per §6.2).

Verification outcome (`Accept` or `Edit`) does not influence pool assignment. Both Accepts and Edits are candidates for the Test Pool. This ensures the Test Pool is representative of the full task difficulty distribution, including the hard cases the model gets wrong, rather than biased toward examples the model already handles correctly.

Target count:

```text
test_pool_target = floor(total_verified × TEST_POOL_FRACTION)
```

When CLIP embeddings are available and multiple examples could fill the next slot, select the candidate most dissimilar from current pool members (maximize intra-pool diversity). When embeddings are unavailable, assign the next Verified example directly.

Routing is evaluated at the moment of verification. The resulting `pool_assignment` is persisted on the Label record (§13.7).

**Growth disclosure (F-S7 amendment, 2026-07-15).** Because the target tracks `total_verified`, late verification at scale grows the "fixed" holdout — by design (the pool stays representative as data grows), but the growth re-bases evaluation metrics between runs, and at batch-import scale it is easy to trip silently (July 2026: a 15× ingest at the default fraction would have ballooned a 120-key holdout toward ~1,900). The Scale-Up gate's `min_test_pool_size` criterion MUST expose `details.pool_target` (plus `test_pool_fraction`, `total_verified`), and the labeling screen's Test Pool chip MUST disclose impending growth when `pool_target > pool_count`. The documented lever for pinning a fixed benchmark is lowering `test_pool_fraction` (PATCH §13.4.2) so the target stays at or below current membership; per-run pool snapshots (§4.3.3) and the returning-vs-new decomposition remain the comparability mechanism across growth.

#### 4.3.2 Automatic Rebalancing

As total Verified count grows, the Test Pool target size grows proportionally. After each new Verified label, the system MUST check whether the Test Pool is below its target and promote non-pool Verified examples if needed.

Rebalancing rules:

1. Candidates for promotion are all non-pool Verified examples (both Accept and Edit).
2. Pool rebalancing uses the same diversity-mode switchover policy as the review selector (Appendix A.3): when the number of eligible examples with CLIP embeddings is at least `CLIP_SWITCHOVER_MIN_COUNT` (default: 50), use CLIP-diverse selection; otherwise use pHash-diverse selection. Mixed CLIP/pHash scoring within a single rebalancing pass is not allowed. See Appendix A.5 for the selection algorithm.

Pool assignments are durable: once assigned, an example remains in the Test Pool and MUST NOT be demoted. Exception: a semantic Core change (§4.4.1) deletes Label records (which carry `pool_assignment`), so pool assignments are cleared as a consequence. The pool rebuilds automatically as examples are re-labeled under the new schema. This is the only operation that resets the pool.

**Order-sensitivity note (diagnostic guard).** The split between the Test Pool and the non-pool ("Train Pool") set is sensitive to *labeling order*: the Test Pool fills from the earliest-labeled examples until `test_pool_target`, plus diversity rebalancing. A **class-clustered** labeling order — whole classes labeled contiguously, as produced by a manifest-order autorun or a class-sorted batch import — can therefore make the Test Pool and the non-pool set hold **disjoint class sets**. This structurally penalizes relevance-ICL evaluation: relevance ICL draws exemplars exclusively from the non-pool Edits (§6.2), so any Test-Pool class with no non-pool representation can only ever retrieve wrong-class exemplars and its metrics understate true accuracy. The **CLIP-diverse review selector** (§6.5, Appendix A.3) — the product default for choosing the next image to review — interleaves classes and keeps the split class-representative, so normal use does not hit this. pHash-diversity rebalancing does **not** prevent it (visual diversity ≠ class coverage). The routing algorithm above is unchanged; the system instead emits a **diagnostic WARNING** at evaluation finalization (ICL runs only) when it detects a degenerate, class-disjoint split — naming the unscorable classes — so the artifact is interpreted correctly rather than mistaken for a true accuracy floor.

#### 4.3.3 Evaluation Snapshots

For evaluation reproducibility, the system auto-creates a frozen **pool version snapshot** when an evaluation run starts:

- The snapshot captures the exact membership list at that point.
- Snapshots are immutable once created.
- Each snapshot has a monotonically increasing `pool_version` within the project.
- Evaluation runs reference the snapshot (Pool Record, §13.11), not the live pool.

If the Test Pool has no members, evaluation MUST NOT proceed (surface a message: *"Not enough Verified labels for the Test Pool. Continue labeling."*).

### 4.4 Guidance and Schema

Guidance records are immutable versions. Editing Guidance creates a new Guidance record with a new `guidance_id`. Projects select active version via `project.active_guidance_id`.

Minimum required fields on Guidance:

- `project_id`
- `guidance_id` (unique within project)
- `version_number: int` (1-based, monotonically increasing within project, backend-assigned, immutable; user-visible as `v{version_number}`)
- `description`
- `schema`
- `rules`
- `created_at: timestamp`

SchemaCore edit policy (mandatory):

**In-place edits (no label invalidation):**

- **Add Aux field:** add new fields with `role="aux"`.
- **Rename Core field:** change `field_name` while preserving `field_id`. Deterministic rename propagation MUST update all Verified labels, ICL references, and export templates within the project.
- **Rename enum value (exact 1:1):** rename an allowed value in an enum or enum_set field where the mapping is exactly one old value → one new value with no change in meaning. Deterministic value rename propagation MUST update all Verified labels within the project.
- **Edit presentation metadata:** changes to `display_order`, display labels, helper text, tooltips, or other non-semantic presentation properties.

**Semantic Core edits (trigger label invalidation + return to Unlabeled; see §4.4.1):**

- Add Core field
- Remove Core field
- Change Core field type
- Change any Core constraint (integer min/max, string minLength/maxLength)
- Change allowed enum/enum_set values (add value, remove value, merge, split, or redefine meaning)
- Change Core ↔ Aux role (promote Aux to Core, or demote Core to Aux)

The decision rule: **if the edit changes what a correct label means, it MUST trigger the schema evolution flow (§4.4.1).** Practically: would a person need to reconsider the image to know the correct answer under the new Core? If yes, labels are invalidated.

Presentation metadata may change without affecting parsing/evaluation and without label invalidation.

Reserved Aux field:

- Implementations MUST support an optional reserved `role="aux"` field named `rationale_note` of type `string`. Its presence in the active Guidance schema is the canonical feature flag: present means enabled; absent means disabled. New Guidance and every starter template MUST omit it by default.
- The SME MAY enable or disable rationale notes from the Guidance builder at any time. Enabling adds the reserved field; disabling removes it. Both changes follow the normal Aux-field rule: they are in-place edits, do not invalidate Verified labels, and may be repeated freely.
- When enabled, the production Teacher MUST request `rationale_note` after the label fields. When disabled, the backend MUST omit it from the prompt-visible schema, guided-decoding schema, output requirements, proposals, label review, and new Verified labels. A stale client-supplied rationale key or provenance value MUST be ignored rather than persisted. Because the field is Aux, it MUST NOT affect evaluation or Core validity in either state.
- **Base serialization order is backend-canonical.** The backend MUST derive a canonical ordered field list (`generation_order`) from SchemaCore:

  ```text
  generation_order = [
    ...["rationale_note"] if enabled,
    ...remaining_aux_fields_sorted_by_display_order,
    ...core_fields_sorted_by_display_order
  ]
  ```

  This list is the stable base order for persisted schema metadata and backend
  transforms. Inference Contracts may filter it for ICL rendering, and the
  production Teacher moves `rationale_note` to the end of prompt-visible,
  guided-decoding, and Student conversation output (§6.2, §9.3). Those
  transforms MUST be backend-owned and deterministic.

- When present, `rationale_note` appears first in the base order (lowest `display_order`, system-enforced), then remaining Aux fields by `display_order`, then Core fields by `display_order`. The `display_order` value controls ordering within each group (Aux-to-Aux, Core-to-Core); the Aux-before-Core group ordering is system-enforced and not user-overridable.
- **Correctness MUST NOT depend on downstream preservation of JSON object member order.** JSON objects are unordered per RFC 8259. Property order is a compatibility hint and deterministic serialization aid, not a correctness mechanism.
- When disabled, rationale processing ends here: models are not required to return it, the SME sees no rationale panel or rationale save gate, `rationale_source` and `rationale_regeneration_invocation_id` are null, and `label_json` has no rationale field.
- Toggling does not backfill or destructively rewrite historical Verified label JSON. Re-enabling may therefore produce a corpus containing examples with and without notes. This accepted tradeoff avoids SME rework for a low-use, high-cost Aux feature. Active prompt/schema/export transforms use the active Guidance and include only available values; production Teacher ICL excludes rationale in every state.
- When enabled, the labeling screen displays Core fields first and other Aux fields second, with `rationale_note` hidden until the SME engages with the label to prevent anchoring bias (see Overview §6). Every newly Verified example has an SME-reviewed rationale note. On Accept, the Teacher's `rationale_note` is retained. On Edit, the SME MUST address it before Save:
  - **SME writes directly:** the SME edits the rationale text to explain the corrected label. Save is enabled.
  - **AI regeneration + explicit approval:** the SME clicks **Generate AI Rationale** → the system calls the Teacher with the image and active task context, withholding both the prior proposal and the SME's correction (Appendix D.3) → the SME waits for the result → the regenerated rationale is displayed → the SME reviews and either edits further or clicks **Approve AI Rationale** to confirm. Save is enabled only after approval. No prior or reviewed field value is rendered into the regeneration prompt.
  - Whitespace-only modifications do not count as meaningful edits and do not enable Save.
  - No unreviewed rationale enters persisted Verified data. Rationale text is not rendered into production Teacher ICL.
- **Rationale source tracking:** when enabled, the Label MUST persist `rationale_source` ∈ {`teacher_proposal`, `sme_edited`, `teacher_regenerated_approved`} to distinguish how the rationale was produced. When disabled, it MUST be null.
- Rationale is excluded from production Teacher ICL (§6.2). Training export includes it only when selected by `export_field_mode` (§9.3).

SchemaCore type system (normative):

Five user-facing field types are supported. Each maps deterministically to a JSON Schema representation used by structured generation (§6.2) and Exact Match normalization (Appendix A.2).

- **Enum** (single choice): JSON Schema `string` with `enum`. Constraints: `allowed_values[]` (required, ≥2 values, no empty strings, unique after trim).
- **Enum Set** (multi-select): JSON Schema `array` of `string` with `uniqueItems: true`, items constrained by `enum`. Constraints: `allowed_values[]` (required, ≥2 values, no empty strings, unique after trim).
- **Boolean**: JSON Schema `boolean`. No additional constraints.
- **Integer**: JSON Schema `integer` with optional `minimum`/`maximum`. Constraints: optional `minimum`, optional `maximum` (if both set: `minimum` ≤ `maximum`).
- **String**: JSON Schema `string` with optional `minLength`/`maxLength`. Constraints: optional `minLength`, optional `maxLength` (if both set: `minLength` ≤ `maxLength`).

No other field types are permitted in SchemaCore.

SchemaCore field record (minimum per-field):

- `field_id: string` (required; immutable; system-generated at field creation; persists across renames; unique across Core + Aux within the schema)
- `field_name: string` (required; editable; must match `^[a-zA-Z_][a-zA-Z0-9_]*$`; max length 64 characters; unique across Core + Aux within the schema; `rationale_note` is reserved and MUST NOT be used for user-defined fields)
- `type: "enum" | "enum_set" | "boolean" | "integer" | "string"` (required)
- `role: "core" | "aux"` (required)
- `allowed_values: string[]` (required when type is `enum` or `enum_set`)
- `minimum: integer | null`, `maximum: integer | null` (applicable when type is `integer`)
- `min_length: integer | null`, `max_length: integer | null` (applicable when type is `string`)
- `display_order: integer` (required; controls within-group ordering for model output and prompt rendering, i.e. Aux fields among Aux, Core fields among Core; §4.4)

`field_name` values are label-schema data keys, not runtime identifiers. Implementations MUST NOT map them directly to Python attribute names, SQLAlchemy mapped attributes, or generated Pydantic field names. No broad reserved-word list is required beyond `rationale_note`; names such as `type`, `id`, or `class` are valid field names because they appear only as JSON property keys and database column values, never as code identifiers.

#### 4.4.1 Schema Evolution: Semantic Core Changes

When a user requests a Core edit classified as semantic (see edit policy above), the system invalidates existing labels and returns affected examples to `Unlabeled` so they can be re-labeled under the new schema. There is no separate re-verification state or workflow — the standard Interactive Labeling loop handles these examples with prior-label hints for efficiency.

**Semantic Core change flow:**

1. User edits Core in the Guidance editor.
2. Backend classifies the edit against the in-place vs. semantic boundary.
3. If the edit is semantic, the system presents a confirmation dialog (§6.6.8).
4. On confirmation, the backend applies the following changes atomically:
   a. Creates a new Guidance version with the Core change applied, sets it as `active_guidance_id`, and records `semantic_core_change_from_guidance_id` on the new Guidance record pointing to the prior version.
   b. All examples with `state="Verified"`: their prior Label data (label JSON, verified outcome, guidance_id) is copied to `prior_verified_label_ref` and `prior_verified_outcome` on the Example record; the Label record is then deleted; the Example transitions to `state="Unlabeled"`. Omitted examples remain Omitted — they are unaffected.
   c. All examples with `state="Auto-Labeled"`: their Label records are deleted; the Example transitions to `state="Unlabeled"`. Auto-Labeled labels generated under a stale schema are not trustworthy; their Operation Records (with `purpose=batch_label`) are preserved for audit. The system surfaces a notice: *"{M} Auto-Labeled examples reverted to Unlabeled. Re-run Batch Labeling when ready."*
   d. Test Pool assignments are cleared (Label records deleted → pool assignments gone). The pool rebuilds automatically as examples are re-labeled under the new schema.
   e. Evaluation-trigger baselines reset: `icl_recommendation_dismissed_at_count` resets to `0`, `scaleup_accept_rate_window` resets (labels under old Guidance are not counted), and any in-progress evaluation OR batch-labeling run (including paused runs, which would otherwise resume under the new schema) is transitioned to `failed` with `status_reason="schema_evolution_canceled"`. Non-semantic edits (in-place, no_change) likewise stop active runs — with `status_reason="guidance_edited_during_run"`, keeping their already-written (re-pointed) labels — because every edit mints a new active version that an in-flight run would otherwise orphan. Prior evaluation Run Records remain in history for audit but are not used as the "previous" baseline for evaluations under the new Guidance (§7.1.2). Auto-Evaluate trigger counters (first pool threshold, configuration change, ICL growth) rebuild from zero under the new Guidance.
   f. Review selector scheduler state is reinitialized: the recent window is cleared to empty, and any persisted selector-history state is reset (§13.3).
   g. If the SME was reviewing an example when the schema change was triggered, `schema_change_context_example_key` is recorded on the Project record so the review selector can start from that example (§6.5).
   h. Schema refinement reminder counters reset to `0` (§6.8).

**How re-labeling works (no special mode):**

After the schema change, the system is in a state equivalent to cold start: many Unlabeled examples, zero Verified, zero ICL. The standard Interactive Labeling loop (§3 Steps 2–5) runs normally with two enhancements:

- **Review selector priority:** the selector presents Unlabeled examples that have `prior_verified_label_ref` before standard Unlabeled examples (§6.5). Among those with prior labels, the example recorded in `schema_change_context_example_key` is presented first (if set), then prior Edits (strongest corrective signal for ICL seeding), then prior Accepts. After all examples with prior labels have been re-labeled, the selector returns to standard diversity-driven ordering over remaining Unlabeled examples.
- **Prior-label hints on the labeling screen:** when an example has `prior_verified_label_ref`, the labeling UI shows the prior label as annotated reference alongside the VLM proposal (read-only):
  - Fields the SME previously edited are badged: *"You changed this from {original_VLM_proposal} to {your_correction}"*
  - Prior values that are schema-invalid under the new schema are highlighted (e.g., removed enum value, out-of-range integer, type mismatch).
  - Where the VLM's new proposal agrees or disagrees with the prior value is indicated.
  - **"Adopt prior" action per field:** each field with a schema-valid prior value shows a one-click action to replace the VLM's value with the prior value. The SME does not retype corrections they already made. Adopting a prior value counts as an Edit; rationale review is required only when rationale notes are enabled (§6.3.2).
- **Save / Skip / Retry**: same three actions as standard review (§4.5). Save without modifications means the VLM proposal is correct on all fields. Any modification, including adopting prior values, requires rationale review before Save only when rationale notes are enabled.

ICL rebuilds progressively from zero as the SME labels examples via Edit (same progressive improvement as initial cold start). Evaluation, Test Pool management, and Auto-Evaluate all follow their standard rules — no special gating or lockout. The SME can freely interleave re-labeling prior examples (with hints) and labeling newly ingested images.

### 4.5 Interactive Actions and Review Outcomes

The proposal is displayed in an editable form. The UI presents three actions:

- **Save**
  - Meaning: store the current label as ground truth. If the SME modified any fields, `verified_outcome=Edit`; if unmodified, `verified_outcome=Accept`. When rationale notes are enabled and fields are modified, the rationale must be addressed before Save is enabled (§6.3.2). When disabled, there is no rationale step.
  - Example state after: `Verified`
  - Verification metadata: yes (`verified_outcome=Accept` or `Edit`)
- **Skip**
  - Meaning: omit this image from workflow; no label recorded; not shown again
  - Example state after: `Omitted`
  - Verification metadata: no
- **Retry**
  - Meaning: re-run proposal for same image after changing Teacher model and/or Guidance; return to the same review
  - Example state after: remains `Unlabeled` or `Auto-Labeled` until Save
  - Verification metadata: no

Additionally, a **Reset** action is available when the SME has modified any field. Reset restores all fields to the VLM's original proposal values, collapses the rationale panel when present, and returns to the unmodified state.

Skip semantics:

- Skip MUST set `state="Omitted"` on Example.
- Skip MUST set:
  - `omitted_source = "sme_skip"`
  - `omitted_at = now()`
- If the Example is `Auto-Labeled`, Skip MUST delete its `label_status=auto_labeled` Label in the same transaction as the Omitted transition. The batch-label Operation Record remains for audit.
- Skip does not require a reason; optional reason fields default null unless set by implementation.
- Omitted examples MUST NOT be presented again by the review selector (across all SME sessions).
- **Restore Omitted:** the SME MAY bulk-restore all Omitted examples to `state="Unlabeled"` (clearing `omitted_source` and `omitted_at`). Available from the labeling screen when the queue is empty and Omitted examples exist. Restored examples re-enter the review selector without resurrecting a discarded Auto-Labeled proposal.

Retry specifics:

- Retry MUST trigger a new Teacher invocation for the same `example_key`.
- Retry controls MUST be pre-populated with the current project settings (Teacher model, Guidance version, Output Stability, Thinking, Visual Budget) so the SME sees what was used for the current proposal and changes only what they want.
- Retry MUST allow SME to change:
  - `teacher_model_config_id` (from catalog entries with `teacher` role) and/or
  - `guidance_id` (select an existing Guidance version compatible with SchemaCore policy)
- After completion, UI MUST present the same three actions again.
- System MUST persist an invocation record for each proposal attempt, including Retry attempts.

### 4.6 Verification Metadata (Accept/Edit Only)

Persist verification metadata with the Label record (§13.7), at minimum:

- `project_id`
- `example_key`
- `verified_outcome` (`Accept`/`Edit`)
- `inference_invocation_id` (invocation whose proposal was accepted/edited)
- `verified_at`

Skip does not create Verified label, but MUST persist omission fields on Example (Section 4.2). Audit/event record for Skip is optional (Section 13.6).

### 4.7 Label Storage (Unified)

All labels — whether produced by Batch Labeling or confirmed by SME review — are stored as **Label** records (§13.7) with a `label_status` discriminator (`verified` or `auto_labeled`). This is a single record family, single table, single code path. The status discriminator governs which fields are populated and which downstream behaviors apply.

**Auto-Labeled labels** are created by Batch Labeling and stored as Label records with `label_status=auto_labeled`. Invocation lineage (model, guidance, ICL, validation) is stored on the Operation Record with `purpose=batch_label`; the Label record stores the label JSON and links to that Operation Record.

**Verified labels** are created by SME Accept/Edit and stored as Label records with `label_status=verified`. Verification-specific fields (`verified_outcome`, `edited_core_fields[]`, `pool_assignment`) are populated on verification; rationale provenance is populated only when the active Guidance enables it.

**Promotion:** when the SME reviews an Auto-Labeled example and clicks Save, the existing Label record's `label_status` transitions from `auto_labeled` to `verified`, verification fields are populated, and `inference_invocation_id` is updated per the save provenance rule below. The prior Auto-Labeled label JSON is preserved on the Operation Record for audit. `batch_label_run_id` is retained for upstream provenance (the Auto-Labeled lineage source that originally produced the machine label); `inference_invocation_id` is the reviewed-proposal pointer (the exact proposal the SME saved). These two fields serve different purposes and do not conflict.

**Rejection by Skip:** when the SME skips an Auto-Labeled example, the machine Label is deleted and the Example becomes `Omitted` atomically. Its Operation Record remains as audit history. Restore returns the Example to clean `Unlabeled` state; it does not restore the discarded proposal.

**Label save provenance after Retry:** when a Label is created or promoted to `verified`, `inference_invocation_id` MUST reference the exact proposal invocation whose contents the SME saved. If the SME saved the surfaced Auto-Labeled proposal without Retry, `inference_invocation_id` remains the original batch-label invocation. If the SME performed one or more Retry actions, `inference_invocation_id` MUST reference the most recent Retry proposal displayed at the time of Save. Earlier proposal attempts in the retry chain are superseded for save purposes.

Auto-Labeled labels MUST NOT be used as evaluation ground truth, ICL context, or pool candidates. Only `label_status=verified` labels serve those purposes. The `label_status` discriminator enforces this boundary — not a separate storage mechanism.

### 4.8 Model Catalog and Project Selections

Each model config entry binds an endpoint + `model_name` + operational metadata (`context_window_tokens` required; optional `model_quantization` and `nim_model_profile` metadata) and declares **`eligible_roles[]`**, the set of roles this entry may serve: `teacher`, `student_base`.

**`context_window_tokens` semantics:** this is the exact integer value used by this spec's prompt-budget logic (§6.2). Seeded catalog entries MUST preserve the vendor's documented value and meaning rather than normalizing shorthand labels (e.g., "128K", "256k") to a uniform convention. Vendor documentation is inconsistent across models: some document input-only token budgets, others document combined input+output context lengths. The seeded values below reflect each vendor's documented figure; the prompt-budget formula in §6.2 subtracts `max_output_tokens` and applies `safety_margin` to derive `effective_max_input_tokens`, which absorbs the difference in vendor semantics.

- MUST be project-scoped via `project_id`.

Rules:

- `model_config_id` MUST be backend-generated and unique within the project.
- **Commercial seed policy:** every model in the fresh-project seed MUST have
  published model terms that permit commercial use when the catalog is
  reviewed. A model with non-commercial-only or unknown model terms MUST NOT
  be preseeded, even when its hosted trial endpoint is technically reachable.
  Operators MAY add such a model explicitly after reviewing its terms;
  historical catalog and invocation records remain inspectable. Schema
  revision `v1_0004` makes one narrow selection repair for projects created
  before this policy: when MiniMax M3 is still the active Teacher, Step 3.7
  Flash exists in the same catalog, and the project has no Label rows, the
  upgrade selects Step. Projects with any Label row are not changed, and the
  MiniMax catalog record is never deleted or rewritten.
- `eligible_roles[]` MUST be non-empty. Role filtering is the sole mechanism for determining where a model may be used:
  - Teacher selection: entries where `teacher ∈ eligible_roles`. A model assigned as `teacher_model_config_id` MUST have `supports_image_input=true`; the backend MUST reject Teacher assignment for models that do not support image input. This invariant MUST be enforced at **every** write site that can set `teacher_model_config_id` — project seeding (§13.4), `POST /v1/projects`, and `PATCH /v1/projects/{id}` (§10.2.13) — so the UI top-bar Teacher picker (`TeacherModelPicker.tsx`) and the Retry per-attempt override (§10.2.3) both surface the same 422 error shape on invalid selections.
  - Student Training base: entries where `student_base ∈ eligible_roles`. Currently limited to the seeded Cosmos bases (Cosmos Reason2 8B/2B, Cosmos 3 Nano/Super reasoner); implementations MUST reject `student_base` role assignment for other models.
- Model configs MUST persist structured generation support status for `response_format` with `type="json_schema"`, with values `{unknown, supported, unsupported}` (updated via probe in §6.2).
- Model configs MUST include thinking toggle metadata (§6.7):
  - `thinking_toggle.mode`: `"none"` | `"qwen_enable_thinking"` | `"kimi_thinking"` (required; determines whether the Thinking toggle is shown and which request fields to send)
  - `thinking_toggle_support`: `"unknown"` | `"supported"` | `"unsupported"` (optional; seeded or updated on runtime rejection)
  - Seeding: Qwen models → `mode="qwen_enable_thinking"`; Mistral VLMs → `mode="none"` (request-level `chat_template_kwargs` override not supported).
- Model configs MUST include image input capability:
  - `supports_image_input: boolean` (required; whether the model accepts image content in messages). This is distinct from `visual_budget_mode`, which tracks visual *token control* support. A model may accept images (`supports_image_input=true`) without supporting request-level visual token controls (`visual_budget_mode="none"`).
  - Seeding: every current catalog entry accepts images; this includes the Cosmos families, Nemotron Nano VL/Omni, Step 3.7 Flash, and Mistral Medium 3.5.
- Model configs MUST include visual budget metadata (§6.9):
  - `visual_budget_mode`: `"none"` | `"mm_processor_size"` | `"mm_processor_pixels"` | `"mm_processor_tiles"` (required; determines which `mm_processor_kwargs` shape the model accepts)
  - `visual_budget_support`: `"unknown"` | `"supported"` | `"unsupported"` (required; seeded or updated via runtime probe)
  - Seeding: Cosmos Reason2 models → `mode="mm_processor_size"`; Nemotron Nano VL → `mode="mm_processor_tiles"`; Mistral VLMs → `mode="none"`.
- Model configs with the `student_base` role MUST include TAO base-experiment metadata (§9.7.8):
  - `tao_base_experiment_id: string | null` (nullable; UUID of the corresponding base experiment registered in the bootstrapped workspace). Null until first-use self-service provisioning (§9.7.8.1a), the equivalent eager CLI, or admin handoff (§9.7.8.1b). The suite launcher creates a provisional TrainingSuite when a selected value is null; TAO chain materialization still requires a non-null id with `pull_complete`.
  - `tao_base_experiment_pull_status: "unknown" | "starting" | "in_progress" | "pulling" | "pull_complete" | "invalid_pull" | "failed" | null` (nullable; cached registration/provisioning lifecycle). Seeding: all `student_base` entries seed to `null` + `"unknown"`; automatic first-use work sets `pulling`, then `pull_complete` or `failed`.
  - Non-`student_base` catalog entries (Teacher-only) MUST have `tao_base_experiment_id: null` — TAO training applies only to student bases.

Each project MUST store active selections:

- `teacher_model_config_id` (MUST reference entry with `teacher` role)
- `active_guidance_id`
- `active_student_model_config_id` (nullable; when set, MUST reference entry with `student_base` role)

Seeded catalog entries (model names are the exact `model` parameter sent in OpenAI-compatible NIM `/v1/chat/completions` requests; hosted catalog entries use provider-prefixed IDs):

- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (128000; 128K input + output tokens) | teacher | vision: yes | media: `none` | local: `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant`, ≥80 GB GPU and compute capability ≥9.0 | `qwen_enable_thinking`, image cap 8 | ***recommended local Teacher when supported***; Teacher-only (Cosmos remains the Student base)
- `nvidia/cosmos3-nano-reasoner` (131072) | teacher, student_base | vision: yes | media: `mm_processor_size` | local: `nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0` (`NIM_MODEL_SIZE=nano`, pinned `NIM_MODEL_PROFILE`), ≥56 GB GPU | *recommended local Teacher when Omni is ineligible*
- `nvidia/cosmos3-super-reasoner` (131072) | teacher, student_base | vision: yes | media: `mm_processor_size` | local: `nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0` (`NIM_MODEL_SIZE=super`), ≥88 GB GPU | *selectable; no longer auto-selected by its larger memory floor*
- `nvidia/cosmos-reason2-8b` (256000; up to 256K input tokens) | teacher, student_base | vision: yes | media: `mm_processor_size` | local: `nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0`, ≥56 GB GPU | *fully selectable; no longer the auto-recommended big-GPU default — superseded by CR3-Nano*
- `nvidia/cosmos-reason2-2b` (256000; up to 256K input tokens) | teacher, student_base | vision: yes | media: `mm_processor_size` | local: `nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0`, ≥36 GB GPU | *recommended local Teacher default on 36–55 GB GPUs*
- `nvidia/nemotron-nano-12b-v2-vl` (128000; 128K input + output tokens) | teacher | vision: yes | media: `mm_processor_tiles`
- `stepfun-ai/step-3.7-flash` (262144) | teacher | vision: yes | media: `none` | ***default Teacher on hosted-only deployments*** *(selected 2026-08-06 as the strongest commercially permitted reachable hosted Teacher in the retained campaign, avg attempted-EM 0.865; reasoning-by-default with no working toggle (`always_on_reasoning`); image cap 8; re-measured 21.0 s steady p50)*
- `mistralai/mistral-medium-3.5-128b` (262144) | teacher | vision: yes | media: `none` | *near-ceiling Mistral-family alternate (avg 0.821, 0.0% model-error); image cap 10; measured in the same congestion band as Mistral Large on 2026-07-21 — an accuracy alternate, not a latency escape*

Entries without `local:` metadata do not support system-managed local deployment in v1 (remote endpoint only). Cosmos Reason2 8B remains in the catalog because it is the primary `student_base` and is fully usable via local NIM deployment; hosted build.nvidia.com access is NVCF-account-gated. Nemotron 3 Nano Omni replaced `moonshotai/kimi-k2.5` as the seeded hosted reasoning Teacher after Kimi K2.5 reached end-of-life on 2026-04-30 (HTTP 410 on every call). Omni's specialized local NIM supports its Qwen-style thinking switch. The completed long-horizon local campaign used thinking OFF; the project-level default remains user-controllable. Omni's higher absolute accuracy earns the supported high-GPU tier, while its non-zero schema-error rate is why CR3-Nano remains the robust fallback. Qwen 3.5 is absent from the v1 seed because its hosted API retired with no NVIDIA-hosted successor. Mistral Large 3 was removed from new-project seeding on 2026-08-03 after NVIDIA marked its free endpoint deprecated, omitted it from the live hosted catalog, and returned HTTP 410. MiniMax M3 was removed from new-project seeding on 2026-08-06 because its published model terms restrict it to non-commercial use. Historical and operator-created configurations remain inspectable; the `v1_0004` zero-Label active-selection repair above does not remove them.

**Default Teacher selection and local availability.** Implementations MUST seed these entries and set `teacher_model_config_id` to the effective hosted `DEFAULT_TEACHER_MODEL` — `stepfun-ai/step-3.7-flash` by default — until a preferred local Teacher is selected or adopted. A configured default that is not in the commercially permitted fresh-project seed MUST fail project creation with an actionable configuration error; it MUST NOT silently reintroduce a removed seed. The value is config-overridable (`~/.vlm_feedback_loop/config.yaml`) and is exposed to the frontend as `EnvironmentResponse.default_teacher_model_name` so the Confirm Defaults preselect never hardcodes a model name. The local recommendation is a separate hardware-aware quality policy (§1.5): Omni when its specialized NIM contract is supported, CR3-Nano next, then Cosmos Reason2 2B. Cosmos remains the required base family for Student Training (only entries with the `student_base` role); Omni is Teacher-only. Default Teacher selection prioritises commercially permitted model terms, certified multi-domain accuracy, ICL-over-time behavior, live-measured interactive latency, and live-probed reliability (image caps, schema fail rates).

Backend MUST validate that any active selection references an entry with the corresponding role and that the model is reachable and compatible.

**Capability re-probe:** the three capability fields (`structured_generation_support`, `thinking_toggle_support`, `visual_budget_support`) are checked once and persisted. If the NIM endpoint is updated (new model version, new capabilities), the user MUST be able to re-probe without deleting and re-adding the model config. A re-probe action (§10.2.12) resets all three fields to `unknown` and re-runs: the structured-generation probe (§6.2), the thinking override acceptance check (§6.7.4, only when `thinking_toggle.mode` is request-based), and the visual-budget probe (§6.9.2). Re-probe MUST NOT be allowed while the model config is referenced by a `queued`, `running`, or `canceling` evaluation/Batch run, or by an active training job. A paused Batch run owns a runtime snapshot and does not block re-probe.

Retry MAY override Teacher and/or Guidance per-attempt; per-attempt override support is required.

---

## 5. System Roles

### 5.1 Teacher (Served via NVIDIA NIM)

- Generates proposals during Interactive Labeling.
- Batch Labeling invokes Teacher to generate Auto-Labeled fully synthetic labels from Unlabeled images.
- MUST be selected from catalog entries with the `teacher` role.
- MUST be reachable via OpenAI-compatible `/v1/chat/completions`.

### 5.2 Guidance-Author (removed)

> **Absent from v1.0:** the Guidance-Author role and AI Guidance Rewrite feature (§6.4) are not part of the public schema.

### 5.3 Student (Fine-tuned via Cosmos-RL / TAO VLM; Served via NIM)

- Produced via Student Training (optional), possibly multiple variants.
- Evaluated using same Exact Match rules (Appendix A.2) against the Test Pool.
- Used for deployment/serving decisions and model comparisons.
- Batch Labeling in this spec is not a Student operation.

### 5.4 Pre-ingest Curation (Future)

Pre-ingest deduplication, NSFW filtering, and metadata enrichment (e.g., via NeMo Curator) may be added in a future version. The system operates without curation in v1.

### 5.5 Embedding Computation and Indexing

#### 5.5.1 Embedding Provider Resolution: Local NIM Default, Hosted Fallback (NeMo Retriever VL 1B v2)

- Provider resolution under `EMBEDDING_PROVIDER=auto` (the default) is a fallback cascade: **(1)** a healthy local embedding NIM deployment — recorded on `EmbeddingDeploymentConfig` (§13.17) by the Mode C deploy flow and live-verified at probe time — is the **default whenever present**; **(2)** the hosted embedding NIM at the same base URL as Teacher (`/v1/embeddings`), using the same hosted NIM API key (no additional credential); **(3)** `none` — the review selector runs in pHash-diverse mode (§5.6). Explicit `EMBEDDING_PROVIDER` values pin a single arm and never fall through. The supported model is NVIDIA NeMo Retriever VL 1B v2 (`nvidia/llama-nemotron-embed-vl-1b-v2`, 2048-dim, requires `input_type` in the request body). Model details and image input format are defined in §2.3 "Embedding NIM endpoint."
- The system resolves the provider by probing at project creation or first ingest (lightweight call to `/v1/embeddings` with minimal input), walking the cascade in order. The local arm is verified with a live request against the recorded endpoint — never trusted from the config record alone — so a stale record (container stopped, displaced, or crashed) fails the verify and the cascade falls through to hosted. If no arm succeeds, set `embedding_provider=none`; the review selector continues with pHash-diverse mode (§5.6). The embedding worker sends requests to the resolved provider's endpoint: the local NIM is unauthenticated (no bearer header — a keyless GPU-only host gets semantic embeddings), while the hosted arm authenticates with the hosted NIM API key.
- **Re-resolution on the healthy transition:** when the local embedding NIM turns healthy, the system re-resolves the provider for every non-archived project and restarts workers where examples are still unembedded — projects created before the NIM finished starting flip to the local provider without waiting for the next ingest or backend restart. When the NIM stops or dies, the config reset (below) plus the live verify make the next probe fall back to hosted.
- **Local embedding NIM deployment (§1.5 Mode C):** the system deploys the embedding NIM locally using the same Docker orchestration as local Teacher deployment. Pinned image: `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0` (NeMo Retriever VL 1B v2; 1B parameters). NIM 2.0.0 preserves the model, modalities, and OpenAI-compatible API while replacing the legacy runtime with automatic architecture-aware kernels. Its support matrix validates specific GPU SKUs; the smallest listed devices, L4 and A10G, have 24 GB, so the Blueprint seeds a 24 GB eligibility floor. Memory alone does not establish support. The orchestrator pins `NIM_PRECISION=fp16` because the image's unset-precision SM120 path exits while requiring a cuDNN plan directory that its entrypoint skips creating, and points `NIM_MODEL_PATH` below the persistent `/opt/nim/.cache` mount instead of the disposable `/model/embed` default. Live RTX PRO 6000 Blackwell validation returned a finite 2,048-dimensional image vector at 6.3 GiB VRAM. The embedding NIM deploys to the lowest-indexed **free** GPU at/above that floor — including supported hosts whose every GPU is below the Teacher floors, which run local embeddings with a hosted Teacher (§1.5 placement policy). On successful deployment, the system sets `embedding_provider=self_hosted_nvclip` (enum value name retained for backwards compatibility) and configures the local endpoint URL automatically; on stop, displacement, or startup failure, the config resets to `provider=none` so the environment assessment and the probe cascade stop advertising a dead endpoint. Local embedding NIM deployment eliminates hosted API rate limits and hosted transport constraints for embedding computation. Switching between hosted and local endpoints serving the same model at the same dimension never invalidates stored embeddings — the no-recompute rule stays keyed on model identity (§5.5.3).

#### 5.5.2 Background Computation (Non-blocking)

- Embedding computation MUST run in the background after image ingestion.
- MUST NOT block the start of interactive labeling or the first proposal. The review selector uses pHash-diverse mode (§5.6) until enough CLIP embeddings are available.
- **State-independent computation:** CLIP embeddings MUST be computed for all ingested examples regardless of current example state. An example that transitions from Unlabeled to Verified before background computation reaches it MUST still have its embedding computed. Embeddings are a property of the image, not the example state, and are used by relevance-ranked ICL selection (§6.2), pool rebalancing (§4.3.2), and the review selector — all of which operate on examples in various states. The computation queue includes all examples without a CLIP embedding, not only Unlabeled examples.
- **Computation order:** CLIP embeddings MUST be computed in pHash-diverse order (the same order the review selector would present images under pHash-diverse mode). This aligns the embedding pipeline with the selector: by the time the CLIP switchover threshold is reached, the first `CLIP_SWITCHOVER_MIN_COUNT` embeddings correspond to the images the SME is most likely to see first.

  To produce the computation queue:

  1. At the start of background embedding computation (after ingestion completes), run the pHash-diverse selection algorithm (Appendix A.3, using `sim_phash`) over all examples without a CLIP embedding (regardless of state) with an empty history to produce a deterministic full ordering.
  2. Enqueue images for CLIP computation in that order.
  3. Process the queue sequentially (or in small batches preserving order). As each embedding completes, persist it in the `ClipEmbedding` table (§13.17) and set `clip_embedding_present=true` on the Example record.
  4. **Incremental ingestion:** if new images are ingested while CLIP computation is in progress, append the new images to the end of the queue. Their pHash values may still be pending in the independent ingest sweep, so the embedding worker MUST NOT assume the signal exists; simple append is the canonical behavior.
  5. **Resumability:** if computation is interrupted (system restart, transient errors), resume from the first example without a CLIP embedding in the original queue order. Already-computed embeddings are persisted and not recomputed.

- **CLIP switchover:** the selector upgrades from pHash-diverse to CLIP-diverse mode (when `REVIEW_SELECTION_MODE=auto`) once the number of eligible examples with CLIP embeddings reaches `CLIP_SWITCHOVER_MIN_COUNT` (default: 50). Below this count, pHash-diverse mode provides adequate visual diversity without constraining the selector to a small subset of CLIP-computed examples. Above this count, the semantic signal from CLIP is meaningfully better and covers enough of the corpus for effective diversity selection. The switchover is seamless; the selector checks the count before each selection.

- **Provider-aware worker concurrency and batch shape:** the embedding worker dispatches HTTP requests with provider-aware concurrency and batch size, because hosted and self-hosted endpoints have different optimal request patterns:
  - **Hosted (`provider=hosted_nvclip`, `integrate.api.nvidia.com`):** concurrency `1`, batch size `8`. Hosted endpoints are shared, rate-limited, and serve other tenants — keeping concurrency at 1 avoids competing with the operator's own foreground traffic and avoids tripping per-key rate limits. Larger batch sizes are more efficient under rate limits because each HTTP call carries more useful work.
  - **Self-hosted / local (`provider=self_hosted_nvclip` or `local_nvclip`):** concurrency `4`, batch size `1`. A local NIM has no rate-limit constraint; concurrent single-image requests saturate the GPU pipeline at lower per-request memory pressure than large server-side batches.

  Both pairs are configurable via `EMBEDDING_CONCURRENCY_HOSTED` / `EMBEDDING_BATCH_SIZE_HOSTED` / `EMBEDDING_CONCURRENCY_SELF_HOSTED` / `EMBEDDING_BATCH_SIZE_SELF_HOSTED` (Appendix A.4). The worker MUST keep SQLite write transactions short and serialized regardless of dispatch concurrency (see §1.8 SQLite write-discipline rules). Effective values are persisted on every project's first successful probe via `embedding_provider`, and re-stamped when the deployment-level provider changes (§5.5.1 re-resolution).

#### 5.5.3 Consistency

- All embeddings within a project MUST use the same embedding model (`clip_embedding_model_id`) and dimension (`clip_embedding_dim`).
- Embeddings MUST be invalidated and recomputed when the effective model identity changes (`clip_embedding_model_id` or `clip_embedding_dim` differs from stored values). A change in endpoint location alone (e.g., from `hosted_nvclip` to `self_hosted_nvclip` serving the same `nvidia/llama-nemotron-embed-vl-1b-v2` model at the same dimension) does NOT require recomputation — the embeddings are a property of the model, not the hosting location.
- If equivalence cannot be proven (different model version, unknown model identity), recomputation is required.

#### 5.5.4 Embedding Indexing (Implementation Detail)

CLIP-diverse review selector scheduling MAY use ANN indices derived from CLIP embeddings to accelerate similarity scoring at scale. No separate embedding index builder is required. The in-memory cache (§5.5.5) is sufficient for the expected project sizes (thousands to tens of thousands of images).

#### 5.5.5 Embedding Storage and Runtime Access

**Embeddings are not stored as artifact refs.** Although large artifacts generally use `_ref` fields and filesystem-backed artifact storage (§1.7), CLIP embeddings are an explicit exception because they are hot-path numeric data used for repeated similarity calculations during review selection, pool rebalancing, relevance-ranked ICL selection, and CLIP switchover counting. Storing one file per vector creates filesystem overhead and forces disk-I/O-shaped access patterns on operations that must be fast.

**Persistent storage:** CLIP embeddings MUST be stored in a dedicated `ClipEmbedding` table (§13.17), not on the `Example` record. Each embedding row is keyed one-to-one to an Example within a project and stores the float32 vector payload plus embedding metadata. Each embedding is ~4 KB as float32×1024, which is modest for SQLite. The `Example` record retains only lightweight presence/summary fields (`clip_embedding_present`, `clip_embedding_dim`, `clip_embedding_model_id`, `embedding_provider`) needed for filtering and selector-mode checks without joining or reading vector data. The `Example` record MUST NOT store the full vector payload. Provider/model invalidation (§5.5.3) is implemented as delete-or-replace on the `ClipEmbedding` table.

**Runtime access contract:** for review selection, pool rebalancing, and any other repeated similarity computations, the backend MUST maintain a project-scoped in-memory embedding cache (dense matrix or equivalent structure keyed by `example_key`). Similarity operations MUST NOT perform per-example database reads or per-example filesystem reads on the hot path.

Cache lifecycle:

- **On project open:** load all embeddings from the `ClipEmbedding` table with a single bulk `SELECT` into the in-memory cache. Build an `example_key → row_index` lookup.
- **On new embedding completion:** append/update the in-memory cache incrementally. Do not reload the full set.
- **On provider/model change (§5.5.3):** delete or replace rows in the `ClipEmbedding` table, then invalidate and rebuild the in-memory cache as embeddings are recomputed.
- **On restart:** cache rebuilds from the `ClipEmbedding` table on project open.

This design is the best fit for the spec's architecture: single-user, SQLite-per-project, thousands-to-tens-of-thousands scale, incremental ingest.

### 5.6 pHash Computation (Background Sweep after Ingest)

#### 5.6.1 Computation

- `POST .../examples:ingest` MUST create skeleton Example rows with
  `phash=null`, return **202 Accepted**, and dispatch restartable background
  pHash work. The worker MUST scan pending rows in bounded batches and persist
  each completed hash in a short transaction.
- pHash is a CPU-only operation with no external dependency (no NIM call, no network, no GPU).
- The system MUST use DCT-based perceptual hashing producing a 64-bit hash. This is the most widely used variant and provides the best quality among standard pHash algorithms. The algorithm is a permanent commitment: project databases are "upgradeable, not disposable" (§1.8), so persisted hashes must remain compatible across versions. The algorithm identifier `dct_phash_64` MUST be recorded on the Project record so that a future migration can detect and recompute if the algorithm ever changes.
- The resulting hash MUST be stored as a hex-encoded string on the Example record (`phash` field, §4.2).
- pHash computation failure for an individual image MUST NOT fail the ingestion; log the failure and leave `phash` null for that image. Pending/failed rows remain eligible through the selector's deterministic no-signal fallback but cannot contribute to or benefit from diversity scoring.

#### 5.6.2 Similarity

pHash similarity between two images is defined as:

```text
sim_phash(a, b) = 1 - hamming_distance(phash(a), phash(b)) / hash_bits
```

Where `hash_bits` is the bit length of the hash (e.g., 64). Higher values indicate greater visual similarity. This measure is used by the pHash-diverse review selector (Appendix A.3) and pHash-based pool rebalancing (Appendix A.5) in the same greedy max-min diversity algorithm used for CLIP.

#### 5.6.3 Schema Migration

pHash values are unaffected by schema changes because the image does not change. pHash remains valid across Guidance versions.

#### 5.6.4 F46 Amendment (2026-05-14) — Background sweep for pHash

To eliminate the wall-clock wait the SME experienced on the Ingest Images screen (batch-by-batch progress bar for 10k-image projects took several minutes), the `:ingest` endpoint now creates skeleton `Example` rows with `phash=null` and returns **202 Accepted** within ~1s. A background worker (`services/ingest_sweeper_service.py`) sweeps `Example WHERE phash IS NULL` in multi-pass batches, computes pHash via `services.phash.compute_phash_from_path` per row, and writes the value back in short transactions. The worker mirrors `clip_embedding_service.py`'s pattern exactly (multi-pass idempotent rescan, per-row error tolerance via `attempted_keys`, restart recovery via `recover_ingest_tasks` invoked from the lifespan startup hook in `main.py`).

The synchronous ingest validation is structural: Pillow opens the image,
checks the supported format/dimensions and calls `verify()` without expanding
the full raster. The background pHash sweep performs the first pixel decode.
This distinction is required for large source photos: full-decoding every image
inside a 200-row ingest request lengthens the serialized project-write window
and can stall Guidance saves and next-image selection for seconds. The
low-frequency DCT implementation computes only the 8×8 coefficient block
consumed by `dct_phash_64`; its loop order and persisted hash values remain
compatible with earlier project databases.

SSE events:

- `ingest_progress` — emitted per batch with `{processed, total, pass_index}`.
- `ingest_completed` — emitted once at the end with `{processed, total}` (covers all passes).
- `run_failed` — not emitted by this worker; per-row pHash failures leave the row at `phash=null` and the sweep continues. The review selector scores only rows with the active signal and falls back to deterministic `example_key` order when every candidate hash is null (Appendix A.3), so a permanently bad image remains reviewable rather than fatal to the project.

**Semantic guarantees preserved:**

- pHash work never blocks labeling. The selector uses available hashes and a deterministic no-signal fallback while rows are pending or permanently null.
- pHash is *still CPU-only with no external dependency* (§5.6.1 still describes the algorithm and storage format authoritatively).
- pHash computation failure for an individual image *still does not fail ingestion* (§5.6.1) — the failure mode shifts from "row created with phash=null inside the endpoint" to "row created with phash=null at the endpoint, sweep attempts compute, leaves at null on failure." End-state semantics identical.

**Frontend handoff (FTU 2026-05-14):** the IngestionSummary screen heading is "Images queued for processing" and renders the SME-facing reassurance line "{N} images queued. Processing continues in the background — you can start labeling now." The primary CTA is [Start labeling →]. No persistent progress chrome on subsequent screens (silent background per FTUE design).

---

## 6. Interactive Labeling Workflows (Interactive)

### 6.1 Guidance Creation and Image Ingestion

- Create Guidance (new Guidance version record; Guidance records are immutable once persisted).
- Ingest images into project (Example records: `storage_ref`, `ingested_at`, `source_metadata`). The 202 response MUST trigger restartable background pHash and optional CLIP computation (§§5.5–5.6).
- After ingestion, the system MUST trigger background CLIP-style embedding computation for newly ingested examples using the project's configured embedding provider (default: hosted embedding NIM, NeMo Retriever VL 1B v2). Embedding computation failure for individual images MUST NOT fail the ingestion and MUST NOT block labeling.
- Cold start supported: projects may begin with `Verified = 0`; first Verified examples created via Accept/Edit.

### 6.2 ICL Selection and Prompt Rendering

**ICL eligibility definition:**

```text
ICL_ELIGIBLE = Label WHERE label_status = ‘verified’ AND pool_assignment IS NULL AND verified_outcome = ‘Edit’
```

Only Edited examples are ICL-eligible. Accepted examples are never selected for ICL. The model already produced correct output for those images, so they add no corrective signal and would consume context window tokens without improving proposal quality. The prompt template (Appendix D.1) provides schema and format guidance independently; ICL is supplementary, not structural.

- ICL eligibility requires `guidance_id = project.active_guidance_id`: only labels created under the current Guidance are eligible. After a semantic Core change (§4.4.1), old labels are deleted, so ICL naturally starts from zero.
- If no Edits exist (cold start or all-Accept run), selection returns empty list and the system proceeds with zero ICL examples.

Selection algorithm (deterministic, relevance-only — Selective-K):

1. **Relevance rank:** order all eligible Edits by descending CLIP cosine similarity to the QUERY image, so the rendered ICL set is the per-query nearest Edits. This is the lever the prior ICL-improves study and the 2026-06-10 Cosmos-Reason generalization study identified as *the* mechanism that makes ICL improve accuracy on visual-classification tasks (relevance lifts macro-F1 2.7–4.5× over zero-shot on Cosmos; the lift is weak on pure OCR / text-extraction tasks, which is the boundary). Relevance degrades to newest-first order (most recent edits first, ties broken by `example_key`) when CLIP embeddings are unavailable. Selection is always relevance-ranked; there are no per-run selection-policy or pinned-recency configuration paths.
2. **Adaptive-K — per-query depth (similarity-gap stopping, default ON):** after relevance ranking, keep only the leading prefix of neighbors whose CLIP cosine similarity is within `ICL_SIM_GAP` of the best neighbor (top-1 always kept; stop at the first that drops further), with an optional absolute floor `ICL_ABS_THRESHOLD`. Governed by `ICL_SIM_GAP` (default `0.05`) and `ICL_ABS_THRESHOLD` (default unset) in `_defaults.py`/`config.py`, threaded through both `proposal_service` and `batch_label_service` into `invoke_teacher` (the eval path resolves its own from the run config). This bounds the per-query example count to the genuinely-similar neighbors and prunes the dissimilar tail that drives the ICL **depth-decline** at high K — the robust cross-model `max_icl` fix (gap 0.05 rescues +0.10–0.11 mean Exact-Match on Cosmos Reason, up to +0.41 on Nemotron Omni; measured in the July 2026 prompt-optimization study, evidence retained in the project's internal engineering archive). When both gap and floor are unset, or in the pHash / no-CLIP fallback (relevance → newest-first), it is a clean no-op (fixed-K). The gap default is CLIP-calibrated and operator-overridable — re-tune per embedder/domain.
3. **Depth cap:** keep the head of the ranked-and-trimmed list within the effective depth cap (per-model default / explicit override; resolution rules below).

Selection determinism by purpose:

- **Evaluation and Batch Labeling:** ICL selection MUST use the deterministic algorithm above (apples-to-apples comparisons).
- **Interactive proposals:** MAY use stochastic selection for exploration (e.g., sampling from eligible set), but MUST persist `icl_example_keys_used[]` so the prompt context is fully reproducible.

ICL selection size:

- **Per-model depth default (`ModelConfig.default_icl_max_examples: int | null`):** each catalog entry MAY carry a default ICL depth cap. The effective selection cap MUST be resolved as: an explicit `icl_max_examples` override (a per-run API field, or a non-null `ICL_MAX_EXAMPLES` setting) wins outright when present — in either direction, so diagnostic depth sweeps can exceed the model default; otherwise the model's `default_icl_max_examples` applies; otherwise (both null) selection is uncapped. Resolution MUST happen at the single shared invocation funnel (`prompt_service.invoke_teacher`) so the proposal, evaluation, and batch labeling pipelines cannot drift. Adaptive-K (step 2 above) still trims per query *within* the effective cap — the default bounds the adaptive mechanism, it does not replace it — and the token/image budget pruning below still applies after selection. *Empirical basis (July 2026 cross-model depth studies; evidence retained in the project's internal engineering archive): useful ICL depth is a model-family property, not a task property. Seeded defaults: Nemotron Nano VL 2; Nemotron 3 Nano Omni 4 (a substantive demonstration ceiling, with Adaptive-K retaining shallower per-query choices); Cosmos CR3 nano/super 8 (monotonic-up through the ~8-shot band in-capacity, overshoot only ≥16); Cosmos CR2-2B 8 (collapses at 16-shot); Cosmos CR2-8B 16 (the only model measured to keep gaining at 16). Historical/operator-created MiniMax M3 records retain 8 and retired Mistral Large records retain 2. Null for unmeasured current seeds.*
- If `ICL_MAX_EXAMPLES` is set, selection MUST return at most `ICL_MAX_EXAMPLES` examples (it is an explicit override: it also replaces any per-model default, above).
- If the selected ICL set exceeds the per-model token budget or image budget (`max_images_per_request − 1`, §6.7), prompt packing MUST drop examples from the END of the selection-ordered list — **relevance-tail pruning**: the tail is the least query-similar exemplar (oldest, in the embedding-less newest-first degrade), so budget enforcement removes the weakest corrective signal first. Token pruning may empty the ICL set entirely; the invocation then proceeds at the cold-start render.
- **Bookend presentation happens after all pruning.** For three or more retained examples, the most relevant example remains first and the second-most relevant moves to the final position; every other retained example keeps its relative order in the middle. Thus `[rank1, rank2, rank3, rank4]` renders as `[rank1, rank3, rank4, rank2]`. With zero, one, or two examples, order is unchanged. Selection, depth, token, and image pruning always operate on relevance order first, so bookending can never protect a weak example from pruning.

ICL field rendering:

- The production Teacher uses `output_field_mode=all`. When the active Guidance enables `rationale_note`, it requests the field last; when disabled, the field is absent. The runtime prompt/output-schema transform is deterministic and does not change the base SchemaCore serialization order (§4.4).
- ICL examples MUST render fields according to the effective `icl_field_mode` from the active Inference Contract (§6.11). For Teacher inference, `icl_field_mode=core_only`: only Core fields are demonstrated, in canonical Core order. Rationale and other Aux prose are omitted so the correction signal stays compact and does not teach generated explanations as label evidence. Students run without ICL in v1.0; their contracts still carry the field mode matching training provenance.

Given the same `guidance_id`, ordered ICL example set, and query context, prompt rendering MUST produce stable output, and MUST include:

- A compact, prompt-visible SchemaCore output contract containing every field name, required/optional status, type, allowed categorical values, and numeric/string bounds that apply
- ICL examples labeled `E01..`
- Strict instruction to output JSON matching schema
- **ICL-use directive (mandatory whenever any ICL example is rendered):** one concise header directive scopes every label to its paired example image and explains that examples demonstrate field meanings and decision boundaries; one concise pre-query directive requires the model to determine every value independently from the query image. The prompt MUST NOT tell the model to copy or prefer a retrieved example's label based on visual proximity. Cold-start prompts omit both. The reference rendering is Appendix D.1.

The production prompt intentionally omits the verbose Guidance Description and Rules blocks. Task/schema information is carried compactly in the field contract and Core-only demonstrations. Guidance remains versioned provenance and still governs schema construction, selection scope, review, and audit; it is not copied verbatim into the production Teacher request.

Structured generation:

- Backend MUST attempt OpenAI-compatible `response_format` with `type="json_schema"` when supported. NVIDIA explicitly recommends `json_schema` over `json_object` for structured outputs.
- Token cap caution: if `max_tokens` prevents the model from completing the JSON, the result may be truncated JSON. Truncated JSON MUST be treated as schema-invalid (Core invalid) by this system.
- Support status is persisted per `model_config_id`: `unknown` / `supported` / `unsupported`.
- **Structured-generation availability probe (normative):** the probe MUST use a fixed request to confirm the endpoint accepts `json_schema` and can satisfy a strict minimal schema. This is an availability smoke test, not a universal schema-compatibility guarantee; it does not prove support for every shape the derived schemas may later use.

  Probe request: one `/v1/chat/completions` request with the following fixed parameters:

  ```json
  {
    "model": "<model_name>",
    "messages": [
      { "role": "user", "content": "Return exactly this JSON object: {\"ok\": true}" }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "structured_probe",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": { "ok": { "type": "boolean" } },
          "required": ["ok"],
          "additionalProperties": false
        }
      }
    },
    "max_tokens": 16
  }
  ```

  Bounded by enforced deadline (`HTTP_DEADLINE_INTERACTIVE_S`).

  Classification:
  - Mark `supported` if endpoint accepts the request, returns a non-error response, and the output parses and validates against the probe schema.
  - Mark `unsupported` if endpoint rejects `json_schema` (4xx error attributable to `response_format`); record sanitized provider error for audit.
  - If the probe times out or returns a 5xx error, leave `structured_generation_support=unknown`; record the error for audit and retry on next probe trigger.
  - Persist result on `model_config_id`; reuse for subsequent invocations.
- If probe indicates `unsupported`, omit `response_format` and proceed with prompt-only JSON generation.
- Prompt-only generation MUST receive the same compact prompt-visible field contract used when guided decoding is attached. A prompt MUST NOT refer to a schema “above” unless that schema summary is actually present.
- **Runtime `json_schema` rejection after prior `supported` probe (per-invocation fallback):** a runtime rejection does not change the model's `supported` status. The probe is a smoke test; real derived schemas may expose incompatibilities that the probe did not cover. Behavior is workload-dependent:
  - **Interactive invocations** (`interactive_proposal`, `retry`, `rationale_regeneration`): retry the current invocation once with prompt-only JSON generation (omit `response_format`). Surface an inline warning: *"Structured generation failed for this request. Prompt-only fallback used."* The model config remains `supported`. The Operation Record persists `structured_generation_fallback_used: true`.
  - **Evaluation and Batch Labeling**: MUST NOT silently mix structured and prompt-only modes within the same run (reproducibility requires uniform generation mode). If a `json_schema` rejection occurs mid-run when `structured_generation_mode` is `auto`, the run transitions to `failed` with `status_reason="structured_generation_rejected"`. The SME restarts the run with `structured_generation_mode=prompt_only` to bypass structured generation. When `structured_generation_mode` is `prompt_only`, the system omits `response_format` from all invocations in the run from the first item, ensuring uniform generation mode.
- **Downgrade to `unsupported`:** a model config transitions from `supported` to `unsupported` only when the explicit capability re-probe (§10.2.12) fails with a clear `json_schema` rejection. Runtime failures do not change probe status; they are handled by the per-invocation fallback (interactive) or run failure (evaluation/batch).

**Structured generation mode (run-level control):**

Evaluation and Batch Labeling run creation requests accept an explicit `structured_generation_mode` that governs structured generation behavior for the entire run:

- `auto` (default): attempt structured generation when the model's `structured_generation_support` is `supported`; use prompt-only when `unsupported`. Mid-run `json_schema` rejections fail the run (uniform mode required).
- `prompt_only`: never send `response_format` for this run. All invocations use prompt-only JSON generation from the first item, ensuring uniform generation mode regardless of model probe state.

The effective mode MUST be snapshotted on the Run Record (`structured_generation_mode_effective`). Interactive proposals do not use this field; they retain the per-invocation fallback behavior above.

When the model is `unsupported`, the UI MUST pin to `prompt_only` and explain why.

JSON schema derivation: backend MUST derive the JSON schema deterministically from SchemaCore:

- Top-level schema is an `object` with `properties` for all SchemaCore fields and `additionalProperties: false`.
- **Property ordering (best-effort hint):** base derivation follows `generation_order` (§4.4). When enabled, the backend deterministically moves `rationale_note` to the end for production Teacher invocation in both `properties` and `x-generation-order` before building the prompt-visible contract or `response_format`. When disabled, it is absent from both. Correctness MUST NOT depend on downstream preservation of JSON object member order. The implementation MUST use ordered serialization and MUST NOT round-trip through a tool that reorders properties.
- **`x-generation-order` extension:** the derived schema MUST include the effective ordered field-name list as a top-level extension property for debugging, introspection, and test verification. Its first base entry is `rationale_note` only when that field is enabled. Runtime correctness MUST NOT depend on a consumer honoring this extension.
- `required` includes all `role="core"` fields and excludes `role="aux"` fields.
- Each field schema reflects SchemaCore type + constraints (enum, min/max, uniqueItems, etc).

Per-model budget enforcement:

**Output budget derivation (schema-aware, per invocation):**

`max_output_tokens` MUST be derived per invocation from the active SchemaCore. The system estimates the worst-case final JSON size from the schema and uses that to compute a base answer budget. If Thinking is ON, the system adds model-scoped reasoning headroom because reasoning tokens consume the normal generation budget on vLLM-backed reasoning models (reasoning is bounded by `max_tokens` unless `thinking_token_budget` is set separately). Bundling reasoning and answer estimates into one number would over-allocate for non-reasoning runs and make the formula opaque.

```text
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

Schema output estimation models the final JSON answer only; it does not predict hidden reasoning length. Reasoning headroom is a separate term because reasoning length varies by task difficulty and is not derivable from the schema.

**Per-field worst-case token estimation:**

- `rationale_note`: `RATIONALE_NOTE_ESTIMATE_TOKENS` (default: 160; covers the 80-word hard cap + JSON key/value overhead + tokenizer mismatch)
- `enum`: key overhead + `max(tokenize(v) for v in allowed_values)` + 6 (quotes, colon, comma)
- `enum_set`: key overhead + `sum(tokenize(v) for v in allowed_values)` + array syntax (worst case: all values selected)
- `boolean`: 6 tokens
- `integer`: 8 tokens
- `string` with `maxLength`: key overhead + `ceil(maxLength / 4)`
- `string` without `maxLength`: `DEFAULT_UNBOUNDED_STRING_BUDGET` tokens (default: 200)

`JSON_STRUCTURAL_OVERHEAD_TOKENS` covers braces, whitespace, and trailing syntax.

**Unbounded string fields are discouraged.** If a Core or Aux field is `string` with no `maxLength`, the system must invent a fallback budget (`DEFAULT_UNBOUNDED_STRING_BUDGET`), which weakens the estimate. Implementations MAY surface a soft warning in the Create Guidance UI when a string field has no `maxLength` set: *"String fields without a max length make it harder to predict output size. Consider adding a maxLength."*

The `× 2` multiplier on `schema_output_estimate` absorbs tokenizer mismatch (`cl100k_base` fallback for all seeded models), model verbosity, and structured generation framing. The `BASE_OUTPUT_TOKENS_FLOOR` prevents degenerate cases (a schema with one boolean field still needs room for well-formed JSON). `MAX_OUTPUT_FRACTION` caps the output budget so it cannot consume the entire context window.

`RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE` is a deployment-level escape hatch. When set, it replaces the derived value for troubleshooting. Under normal operation it is null and the system derives the value.

**Reasoning headroom (Thinking=ON):**

`MODEL_REASONING_HEADROOM_TOKENS` (default: 16384) is a fixed headroom added to the schema-derived output budget when `thinking_mode_effective="on"`. No adaptive tier system or rolling truncation tracking is required. The default is set conservatively to cover multi-field schemas with reasoning; `RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE` is the escape hatch for edge cases.

`thinking_mode_effective` is resolved per the §6.7.3 taxonomy and the persisted `thinking_toggle_support` capability:
- `thinking_toggle.mode = "none"` (e.g., Mistral) → always `"off"`; no headroom (the model does not reason).
- `thinking_toggle.mode = "always_on_reasoning"` (e.g., Step 3.7 Flash) → always `"on"`; headroom always allocated regardless of user input (the model reasons regardless and would otherwise burn its full `max_tokens` budget on internal `<think>` content, leaving none for visible output).
- `thinking_toggle.mode ∈ {qwen_enable_thinking, kimi_thinking}` with `thinking_toggle_support="supported"` → follows the user's `thinking_default_on` / per-attempt override and sends the corresponding request field.
- The same request-based modes with support `unknown` or `unsupported` → send no thinking override, hide the ineffective control, treat the model's natural default as `"on"`, and allocate reasoning headroom.

When `finish_reason="length"` occurs with `thinking_mode_effective="on"`, the system surfaces a non-blocking warning: *"Reasoning output was truncated. Consider switching to Thinking OFF or increasing the output budget."* The automatic retry on truncation (§6.7.6 Step 6) handles recovery for the current invocation. `truncation_attributed_schema_invalid` (§13.1) tracks when truncation caused a schema failure, providing diagnostic signal without requiring automated headroom adjustment.

**Input budget derivation:**

```text
safety_margin =
  RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN (default: 0.85; see Appendix A.4)

effective_max_input_tokens =
  floor((context_window_tokens - max_output_tokens) * safety_margin)
```

Note: `context_window_tokens` is the vendor-documented value stored on the ModelConfig (§4.8). Vendor semantics vary (input-only vs combined input+output); subtracting `max_output_tokens` and applying `safety_margin` absorbs this difference. Token counting itself is approximate (§1.8) because seeded models lack tiktoken encoder mappings and use the `cl100k_base` fallback; `safety_margin` MUST be set conservatively to absorb tokenizer mismatch.

When Thinking is ON, `max_output_tokens` is larger (reasoning headroom added), which reduces `effective_max_input_tokens`. This is correct: the model needs more output room for reasoning, so ICL capacity shrinks. The system handles this by relevance-tail pruning of the ICL set.

Before ICL pruning, the system MUST tokenize the actual rendered zero-ICL production message envelope, including the compact field schema. It MUST NOT estimate fixed prompt cost from raw Guidance length plus a magic constant. Per-example estimates then add the rendered field subset and marker overhead. Token budget enforcement MUST remain resilient to tokenizer and multimodal framing differences by applying the safety margin and dropping ICL deterministically rather than blocking progress.

---

### 6.3 Inference, Validation, and Review

**Image transport for model inference and embedding computation (normative):**

All outbound requests that include images — including interactive proposals, evaluation, batch labeling, rationale regeneration, and embedding NIM computation — MUST follow the transport rules in this section:

- **Format normalization:** the ingest pipeline accepts JPEG, PNG, WebP, BMP, and single-page TIFF (§10.2.1), but NIM endpoints may not support all of these at request time (e.g., Cosmos Reason2 documents JPEG and PNG only; hosted embedding NIMs typically accept JPG, JPEG, and PNG). Before sending an image to a model endpoint, the backend MUST transcode the image to a model-supported format if the source format is not natively supported. Default transcode target: PNG (lossless). The backend MUST NOT fail silently on unsupported formats; it MUST transcode or reject with a clear error.
- **Chat completions transport (base64 inline):** for `/v1/chat/completions` requests (Teacher, Student), images sourced from `storage_ref` are read, format-normalized if needed, and encoded as a base64 data URL (`data:image/<format>;base64,<data>`) in the OpenAI-compatible `image_url` content part. This is the standard path for all local-file inference.
- **Embeddings transport:** for embedding NIM `/v1/embeddings` requests, images are sent in the request body's `input` array, not as chat-completions message content parts. Each image input MUST be represented as a base64 data URL (`data:image/<format>;base64,<data>`) within the `input[]` string array. The `/v1/embeddings` endpoint supports batching (multiple images per request, response embeddings indexed in input order). NeMo Retriever VL requests MUST also include `"input_type"` (typical values `"query"` / `"passage"`). Hosted embedding NIMs typically accept JPG, JPEG, and PNG image inputs.
- **Direct URL transport:** a direct HTTP(S) URL MAY be used only when the image already exists at a fully qualified, runtime-reachable URL that the target NIM endpoint can download. Filesystem paths and `storage_ref` values MUST NOT be passed as URL references — they are local to the backend host and not resolvable by NIM.
- **Hosted NIM transport (inline base64 only):** hosted NVIDIA endpoints (API Catalog, `integrate.api.nvidia.com`) accept only inline base64 data URLs for image content — the OpenAI-compatible chat/completions and embeddings endpoints reject asset references with `"Only base64 data URLs are supported for now."`. The backend therefore always sends images inline as base64, on hosted, self-hosted, and local endpoints alike, regardless of image size. **Historical note (removed 2026-06-02):** earlier revisions of this spec defined a large-image path that uploaded images above `HOSTED_NIM_ASSET_UPLOAD_THRESHOLD_BYTES` to the NVCF Asset API (`POST /v2/nvcf/assets`), referenced them by asset ID, and deleted them after the invocation. That path was removed because the hosted OpenAI-compatible endpoint never accepted asset references, and NVIDIA's June 2026 NVCF deprecation retired the `/v2/nvcf/assets` endpoints outright — its documented replacement is to "send payloads directly during invocation," which is exactly what inline base64 does.
- **ICL example images** follow the same transport rules. Each ICL example's image is read from `storage_ref`, normalized, and dispatched as inline base64. All images for one invocation (ICL examples + query) are prepared in a single batched call before the model request is dispatched.
- **Persistence:** the Operation Record (§13.1) MUST persist `image_transport_mode` ∈ {`base64_inline`, `direct_url`} and `image_format_transmitted` (e.g., `"image/png"`, `"image/jpeg"`) for auditability.

#### 6.3.1 Proposal Attempt (Teacher Invocation)

Flow per example:

1. Determine effective Teacher, Guidance, Generation Controls, and Visual Budget:
  - Start from project defaults: `teacher_model_config_id`, `active_guidance_id`, `labeling_generation_preset_key`, `thinking_default_on`, `visual_budget_preset_key`
  - Apply per-attempt overrides (e.g., Retry; `generation_preset_key_override`, `thinking_mode_override`, `visual_budget_preset_key_override`)
  - Production Teacher ICL includes Core fields only and never includes `rationale_note` (§6.2)
2. Resolve effective `teacher_model_config_id` → endpoint + `model_name` + context window.
3. Render prompt and compute `icl_example_keys_used[]`.
4. Persist operation record before model call.
5. Construct and invoke NIM request per §6.7.6 (sampling params from preset, thinking toggle fields, visual budget params, seed injection by purpose) with deadline + bounded transport/system retries.
6. Persist attempt artifacts and validate/normalize output under SchemaCore; persist structured validation report separating Core vs Aux:
  - `raw_model_response_ref` (nullable)
  - `normalized_json_ref` (nullable)
  - `schema_valid_core: boolean`
  - `validation_errors_core[]`
  - `validation_errors_aux[]`
  - `validation_report_ref`
  Core errors determine schema-valid vs schema-invalid. Aux errors MUST NOT block validity and are shown as warnings.
7. UI presentation:
  - If Core valid: display proposal JSON + Aux warnings (if any)
  - If Core invalid / timeout / endpoint error: display failure state; allow manual labeling and Retry

Optional latency benchmarking: MAY measure p50/p99 end-to-end call latency for configured concurrencies.

Operation Records MUST be persisted for all outcomes (success/invalid/timeout/endpoint error), with best available artifacts and sanitized error payload refs (Section 13.1).

#### 6.3.2 Review Loop (Save / Skip / Retry)

After each proposal attempt (including after Retry), the proposal is displayed in an editable form with three actions: **Save / Skip / Retry** (§4.5). Additional requirements for this workflow:

The following rationale requirements apply only when `rationale_note` is enabled in the active Guidance. When it is disabled, the system bypasses all rationale prompting, review, regeneration, save gating, and provenance.

**Rationale grounding (normative when enabled).** Rationale notes describe concrete, image-specific evidence relevant to the active Task and Rules, using natural vocabulary appropriate to that domain. Quantities, identities, comparisons, and locations MAY be used when they are visibly supported and useful for the task; no universal prompt-level prohibition or prescribed visual-vocabulary palette applies across projects. A rationale MUST NOT merely echo field names or values, recite generic visual boilerplate, or invent details to make a field value sound supported. Automatic rationale regeneration MUST inspect independently: neither the original proposal nor the SME's corrected values are rendered into its writer prompt. The task, Rules, and Core schema provide context without supplying an answer. The rationale MUST describe the subject the task asks about and distinguish a physical carrier from depicted content only when that distinction matters to that task. This conditional rule supports object, OCR, scene, and anomaly tasks without imposing one ontology on all of them. When the image is ambiguous, it MUST state the uncertainty rather than invent support. The Teacher prompt (Appendix D.1) and rationale regeneration prompt (Appendix D.3) carry this contract into automatic rationale generation; SME-written rationales SHOULD follow the same pattern.

- **Edit detection and diff:** the backend MUST compute a deterministic diff between the proposal normalized JSON and the final SME label JSON and persist `edited_core_fields[]` and `edited_aux_fields[]` for the Verified label. If no fields differ, `verified_outcome=Accept`; if any field differs, `verified_outcome=Edit`. Both diff arrays are required for audit and provenance.
- **Rationale display ordering (anti-anchoring):** when `rationale_anti_anchoring` is enabled (default: `true`), the UI MUST NOT show `rationale_note` alongside the initial proposal. The rationale panel is hidden until the SME modifies a field value. On Save without modifications, the rationale is never shown; the Teacher's version is retained silently. This prevents the model's reasoning from biasing the SME's independent judgment. When anti-anchoring is disabled, the rationale is visible alongside the proposal from the start. Rationale review when fields are modified is still mandatory regardless of this display setting.
- **Rationale is mandatory when enabled and fields are modified.** The Save button is disabled until the rationale is in an approved state. The Teacher's original rationale explained a wrong answer and MUST NOT be saved as-is. The rationale panel shows a single editable textarea pre-populated with the model's original rationale. The SME edits the text directly — fixing what's wrong, keeping what's right. Three states:
  - **"Needs review"** (initial state when a field is modified): the textarea contains the model's original rationale, editable. Save is disabled until the text is meaningfully changed. Two actions are available:
    - **Edit directly:** the SME modifies the text in the textarea. State transitions to **"Edited"**. Save is enabled.
    - **Generate AI Rationale** button: the system calls the Teacher with the image and active task context (Appendix D.3). The request carries neither the original proposal nor the SME's corrected values. Inline loading replaces the textarea content while the SME waits. On completion, the regenerated text replaces the textarea content and state transitions to **"AI-regenerated, review required"**.
  - **"AI-regenerated, review required"**: the regenerated rationale appears in the textarea (editable). Save is still disabled. The SME either:
    - Edits the text → state transitions to **"Edited"**. Save is enabled.
    - Clicks **Approve AI Rationale** → state transitions to **"Approved"**. Save is enabled.
  - **"Edited"** or **"Approved"**: Save is enabled. The SME can continue editing or click Save.
  - Whitespace-only changes do not transition from "Needs review" to "Edited".
- **Reset:** available when any field has been modified. Restores all fields to the VLM's original proposal values, collapses the rationale panel, and returns to the unmodified state (Save becomes one-click again). Replaces the "Undo changes" button from the rationale panel — Reset covers the entire form, not just the rationale.
- **Rationale on Edit (backend):** if the SME wrote or modified the rationale, store with `rationale_source="sme_edited"`. If the SME used AI regeneration and explicitly approved, store with `rationale_source="teacher_regenerated_approved"`. The regeneration invocation is persisted as an Operation Record with `purpose="rationale_regeneration"` and linked via `rationale_regeneration_invocation_id`.
- **Skip** advances the review selector; **Retry** does not.
- SME may Retry multiple times; only final Save creates Verified label.

### 6.4 Guidance Rewrite (removed)

> **Absent from v1.0:** the AI Guidance Rewrite flow and its Guidance-Author role are not part of the public schema; Guidance improvement is manual editing via the Edit Guidance screen (§4.4).

### 6.5 Labeling Review Selector

Eligibility:

- Review selector MUST draw only from `state="Unlabeled"` and `state="Auto-Labeled"` examples during standard labeling.
- After a semantic Core change (§4.4.1), the selector prioritizes Unlabeled examples that have `prior_verified_label_ref` set (prior Edits first, then prior Accepts) before standard Unlabeled examples, starting from `schema_change_context_example_key` if set.

Selection strategy:

- If CLIP embeddings available and mode is `auto` or `clip_diverse`: use CLIP-diverse algorithm (Appendix A.3).
- Otherwise: use the pHash-diverse algorithm when hashes are available (Appendix A.3, substituting pHash hamming similarity for cosine similarity), with newest-first fallback while the background signal is pending (§5.6).

Skip advances review selector and transitions example to `Omitted`. Retry does not advance review selector.

### 6.6 Create Guidance Screen (UI Contract)

This section defines the presentation, interaction model, and local validation contract for the Create Guidance screen (onboarding step 5, Overview §6). The user defines the task Description, label Schema (SchemaCore), and Rules that govern all downstream labeling, evaluation, and model behavior.

Design priorities:

1. Get the user to their first Verified labels quickly (onboarding goal).
2. Make the Core/Aux distinction unmissable (semantic Core changes after save invalidate existing labels).
3. Validate SchemaCore locally before save (prevent broken schemas from reaching the backend).
4. Support expert users without burdening first-time users.

#### 6.6.1 Page Structure

The screen has a sticky header (status) and sticky footer (actions).

**Sticky header:**

- Left: title ("Create Guidance") and subtitle (`Project: <name>`)
- Right: live-updating schema status badge: `Valid` (no issues) or `N errors` (errors exist). The badge gives a persistent "am I done?" signal at all scroll positions.

**Sticky footer (primary actions), right-aligned:**

- **Cancel** (secondary/text button): returns to previous onboarding step
- **Save Guidance** (primary button): disabled when errors exist; enabled when 0 errors (warnings allowed)

**Body:**

Single-column layout. Three stacked cards (Description, SchemaCore, Rules), followed by collapsed previews (Derived JSON Schema, Example label output). Errors are shown inline on the offending field, not in a separate panel.

**Card layout:**

All three cards (Description, SchemaCore, Rules) are always visible. On first load, focus the Description card.

#### 6.6.2 Description Card

- **Label:** "Task Description"
- **Component:** multiline textarea (3–8 visible lines)
- **Helper text:** *"Describe the task and what the model should focus on in each image."*
- **Placeholder:** *"You are inspecting manufactured parts for surface damage. Focus on cracks, discoloration, and deformation. Minor cosmetic wear is acceptable…"*
- **Validation:** optional; may be empty. Save readiness is governed by backend validation (§6.6.6). Soft character counter shown for awareness (no hard limit enforced in UI).

**Starter template dropdown:**

A labeled dropdown (**"Start from:"**) above the three cards (not inside a card). Options: Blank (default), Classification, Rock, paper, scissors, Multi-label classification, Presence and count, Packaging information audit, Industrial anomaly inspection. Selecting a template pre-fills Description, proposes a starter schema in SchemaCore, and pre-fills Rules. The dropdown defaults to Blank. Everything remains fully editable. If the SME chooses another template after editing, the UI MUST ask before replacing the current Description, Schema, and Rules; canceling MUST preserve the draft and restore the applied selection.

The generic classification templates use unmistakable `replace_me_*` placeholder values. Dataset-referenced templates carry real task contracts: the repository's bundled rock-paper-scissors walkthrough, public Open Food Facts packaging images, and the public VisA industrial-anomaly dataset. The selector shows one concise explanation for the applied choice. Dataset-referenced choices additionally show the dataset name, scope, license, and an external source link when one exists; these details are supporting context and are not prefixed into the option label. The repository bundles no additional sample data for Open Food Facts or VisA.

Template definitions:

- **Blank:** empty Description, Schema, and Rules.
- **Classification:** Description: *"Classify each image into one category."* Core: `category: enum` [`replace_me_a`, `replace_me_b`]. Rules instruct the SME to replace both values, assign exactly one category, and Skip rather than invent a category when the image cannot be classified confidently.
- **Rock, paper, scissors:** Description: *"Classify the hand gesture in each image as rock, paper, or scissors."* Core: `category: enum` [rock, paper, scissors]. Rules define the visible gesture for each value, identify the primary foreground hand as the subject, and direct the SME to Skip occluded or ambiguous gestures. Dataset context: bundled 15-image walkthrough, CC BY 2.0.
- **Multi-label classification:** Description: *"Select all labels that apply to each image."* Core: `labels: enum_set` [`replace_me_a`, `replace_me_b`]. Rules instruct the SME to replace both values, select every visibly supported label, use an empty set when none apply, and avoid near-synonym labels.
- **Presence and count:** Description: *"Determine whether the target object is visible in each image and count the visible instances."* Core: `target_present: boolean`, `target_count: integer` (min: 0). Rules require the SME to name the real target, keep the boolean/count pair consistent, define how partial instances are counted, and Skip images that cannot be counted reliably.
- **Packaging information audit:** Description: *"Read each food-packaging photo, identify the dominant language of the visible text, and determine whether a nutrition-information panel is visible."* Core: `language_on_packaging: enum` [fr, en, es, de, it, nl, other], `contains_nutrition_table: boolean`. Rules define dominant language from legible words, the visible structure required for a nutrition panel, and when unreadable text requires Skip. Dataset context: Open Food Facts images, open product-packaging photos with extracted text, CC BY-SA images, source `https://openfoodfacts.github.io/openfoodfacts-server/api/aws-images-dataset/`; no images are bundled.
- **Industrial anomaly inspection:** Description: *"Inspect each product image from the VisA dataset, identify the object category, and determine whether a visible manufacturing anomaly is present."* Core: `object_category: enum` [candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum], `has_anomaly: enum` [no, yes]. Rules distinguish visible surface/structural flaws from pose, lighting, and ordinary appearance differences and direct the SME to Skip insufficient evidence. The template deliberately omits free-text defect description and the source annotation's 39-value defect taxonomy: prior Blueprint runs found that taxonomy synonym-fragmented and poorly suited to reliable exact-match evaluation, while optional rationale notes already cover explanatory evidence. Dataset context: Visual Anomaly (VisA), 10,821 images across 12 object categories, CC BY 4.0, source `https://github.com/amazon-science/spot-diff`; no images are bundled.

All templates leave rationale notes disabled. The SME may opt in with the Guidance toggle after selecting any template.

#### 6.6.3 SchemaCore Card (Field Builder)

**Explainer strip** (one line + tooltip) at the top of the card:

> **Core fields** = required + evaluated · **Aux fields** = optional + not evaluated

Tooltip (on info icon): *"After saving, renames apply directly. Changing what a correct answer means (types, constraints, values, adding/removing fields) invalidates existing labels. The system returns them to Unlabeled and shows your prior labels as reference when you re-label."*

**Core edit policy banner**, a concise info callout below the explainer strip, shown only on the create flow:

> After saving, you can rename fields and values at any time. Changing what a correct answer looks like (types, constraints, categories, or adding/removing fields) invalidates existing labels. The system returns them to Unlabeled, re-proposes labels under the new schema, and shows your prior labels as reference so you can focus on what changed.

**Two separated field sections:**

The card is divided into two visually distinct sections:

- **Core Fields (Required, Evaluated)**: section header with primary-color left border or background tint; contains all Core field rows; **+ Add Core Field** button at section bottom. Helper text below the section header: *"These are the critical answers. They are evaluated. Unlike other fields, changing Core fields after labeling starts means re-labeling your existing images."*
- **Aux Fields (Optional, Not Evaluated)**: section header with muted/secondary styling; contains a dedicated **Rationale notes** toggle followed by user-defined Aux field rows; **+ Add Aux Field** button at section bottom. The toggle defaults off. Helper text explains that enabling it asks the Teacher for visible evidence and requires SME review after a label edit.

This physical separation encodes the Core/Aux role at field creation time and makes the distinction structural.

**Field row design:**

Each field is a card or row within its section, containing:

- **Drag handle**: for reorder within section. Field order affects within-group prompt rendering order via `display_order` (§4.4). The Aux-before-Core group ordering is system-enforced.
- **Field name**: text input, required, validated identifier (§4.4 field record constraints).
- **Type**: dropdown, required. Options: Enum, Enum Set, Boolean, Integer, String (§4.4 type system).
- **Constraints**: contextual inline controls that appear based on selected type (see below).
- **Role badge**: chip, `Core` or `Aux`. Visible even though fields are grouped by section. Core = solid primary; Aux = outlined muted.
- **Row actions**: icon buttons: Delete, Move to Aux / Move to Core.

**Move to Aux / Move to Core** is available as a row action on the create flow (pre-save) and moves the field row to the other section. After save, changing Core ↔ Aux role is a semantic Core change that triggers label invalidation (§4.4.1).

**Reserved `rationale_note` control:**

- Presented as a dedicated **Rationale notes** toggle in the Aux section, not as a normal editable field row.
- Defaults off for new Guidance and templates.
- Enabling adds the canonical `rationale_note` / `string` / `aux` field. Disabling removes it.
- The SME cannot rename it, change its type or role, or manipulate it through normal row controls.
- Toggling either direction is an in-place Aux edit and never carries an invalidation marker.

**Contextual constraint controls (by type):**

Default constraints are shown inline beneath the row when applicable. Additional constraints are available behind a collapsed **"Advanced constraints"** expander on applicable types.

- **Enum** (single choice):
  - Allowed values: tag/chip input (default, always visible). Required ≥2 values. No empty strings. Unique after trim.
- **Enum Set** (multi-select):
  - Allowed values: tag/chip input (default, always visible). Required ≥2 values. No empty strings. Unique after trim.
- **Boolean:** no additional constraints.
- **Integer:**
  - `min`: number input (default, always visible). Optional. If both set: min ≤ max.
  - `max`: number input (default, always visible). Optional. If both set: min ≤ max.
- **String:**
  - `minLength`: number input (advanced, collapsed). Optional. If both set: minLength ≤ maxLength.
  - `maxLength`: number input (advanced, collapsed). Optional. If both set: minLength ≤ maxLength.
The "Advanced constraints" expander keeps the default view clean for onboarding while ensuring users who need `maxLength` on day one are not blocked.


#### 6.6.4 Rules Card

- **Label:** "Rules & Edge Cases"
- **Component:** multiline textarea (supports plain text; markdown optional)
- **Helper text:** *"Optional. How should the model handle ambiguous or tricky images? Edge cases tend to surface during labeling, so you can always add Rules later."*
- **Placeholder:** *"If damage is partially obscured, classify based on the visible portion only…"*
- **Validation:** optional. May be empty on first version (user can refine later by editing the Guidance).

#### 6.6.5 Previews

Below the three cards, two collapsed previews provide optional detail for power users and engineers.

**JSON Schema preview** (collapsed by default):

- Toggle: "Derived JSON Schema"
- Shows the canonical derived JSON Schema from the backend validation endpoint (§10.2.2), updating on each backend validation response.
- Read-only. Intended for engineers integrating validators, exporters, or structured generation.

**Example output preview** (collapsed by default):

- Toggle: "Example label output"
- Shows a representative JSON object with all Core fields populated with placeholder values matching their types and constraints (e.g., first enum value, mid-range integer), and Aux fields shown as optional.
- Intended for quick verification: "Is this what I want the model to produce?"

#### 6.6.6 SchemaCore Validation Contract

The backend is the single validator: schema rules, derivation, JSON Schema compilation, and `SCHEMA_COMPILE_FAILURE` detection all live in the draft validation endpoint (§10.2.2). The frontend implements no validation rules of its own — it calls the endpoint debounced (~400ms) after edits and immediately on a save attempt, renders the returned `issues[]` inline (badge, per-row errors, `SCHEMA_COMPILE_FAILURE` banner above the SchemaCore card), and applies latest-wins on in-flight responses. This single backend codepath powers the inline error display, the Derived JSON Schema preview (§6.6.5), and save readiness (§6.6.7).

**On Save click:** the frontend MUST obtain a fresh backend validation response for the current draft and proceed only when it reports zero errors.

**Severity model:**

- **Errors:** block save. All errors come from backend validation.

**SchemaIssue structure:**

```text
severity: "error"
code: string           // e.g., "DUPLICATE_FIELD_NAME"
message: string        // user-friendly
field_path: string | null  // per-section indices, e.g. "core[2].name" /
                           // "aux[0].role"; enables click-to-scroll
```

**Validation errors (block save):**

- `NO_CORE_FIELDS`: "Add at least one Core field (required for evaluation)."
- `MISSING_FIELD_NAME`: "Field name is required."
- `DUPLICATE_FIELD_NAME`: "Duplicate field name: `{name}`." (names unique across Core + Aux)
- `INVALID_FIELD_NAME`: "Use only letters, numbers, and underscores. Must not start with a number." (must match `^[a-zA-Z_][a-zA-Z0-9_]*$`)
- `FIELD_NAME_TOO_LONG`: "Field name must be 64 characters or fewer."
- `MISSING_TYPE`: "Select a type."
- `ENUM_TOO_FEW_VALUES`: "Add at least two allowed values." (Enum/Enum Set)
- `ENUM_EMPTY_VALUE`: "Allowed values cannot be empty strings."
- `ENUM_DUPLICATE_VALUE`: "Duplicate value: `{value}`." (unique after trim)
- `MIN_EXCEEDS_MAX`: "Min must be ≤ Max." (Integer)
- `MINLENGTH_EXCEEDS_MAXLENGTH`: "minLength must be ≤ maxLength." (String)
- `RATIONALE_NOTE_WRONG_ROLE`: "`rationale_note` must be an Aux field."
- `RATIONALE_NOTE_WRONG_TYPE`: "`rationale_note` must be type String."

- `SCHEMA_COMPILE_FAILURE`: "Schema cannot be compiled (internal inconsistency). See details…" — catches contradictions the field-level rules might miss.

**Fix-it affordances:**

For select high-frequency errors, a one-click fix button appears next to the inline error message. Fix actions are deterministic and re-trigger backend validation on the next debounce cycle.

- `ENUM_DUPLICATE_VALUE`: button "Remove duplicates". Deduplicates the allowed values list (keeps first occurrence after trim).
- `MIN_EXCEEDS_MAX`: button "Swap min/max". Swaps the min and max values.

#### 6.6.7 Save Behavior

**Button states:**

- Errors exist → Save Guidance disabled; status badge shows error count.
- No issues (0 errors) → Save Guidance enabled; status badge shows "Valid".

**On Save:**

1. Re-validate the current draft via the backend endpoint (§6.6.6) and confirm zero errors.
2. If errors: scroll to first error; do not save.
3. If valid (0 errors):
  - Call the Guidance save endpoint. The backend MUST run the same `validate_and_derive` function used by the draft validation endpoint (§10.2.2) before persisting. This ensures preview and save use identical derivation logic.
  - Persist Guidance as a new immutable version (§4.4).
  - Set it as `project.active_guidance_id`.
  - Show confirmation toast: *"Guidance v1 saved."*
  - Navigate to the labeling screen (Screen 7).

Cancel returns to the previous onboarding step without saving.

#### 6.6.8 Post-Save SchemaCore Edit UX

On subsequent "Edit Guidance" screens (after the first save):

- **Description** and **Rules** textareas are editable (editing creates a new Guidance version per §4.4).
- **+ Add Aux Field** remains available without label invalidation.
- **SchemaCore section** supports the Core edit policy (§4.4) with visual indicators that distinguish in-place edits from label-invalidating edits at the individual control level, before the user commits to a change.

**Per-control visual indicators (required):**

Every editable control on a post-save Core field shows which edit category it belongs to:

- **In-place controls** (no label invalidation): normal styling, no additional indicator. These are:
  - Field name text input (rename)
  - Enum/enum_set value rename (exact 1:1 edit of an existing value chip)
  - Presentation metadata (display_order drag handle, display labels, helper text)
- **Label-invalidating controls**: a small warning icon or subtle visual marker (e.g., a refresh glyph, tinted border, or badge) adjacent to the control. Tooltip on hover/focus: *"Changing this invalidates your {N} verified labels. They return to Unlabeled for re-labeling."* These are:
  - Type dropdown
  - Constraint inputs (min/max, minLength/maxLength)
  - Allowed values "add" or "remove" actions (distinct from renaming an existing value)
  - **+ Add Core Field** button
  - Field delete action
  - Move to Aux / Move to Core action

This per-control distinction gives the user confidence to make in-place edits without hesitation, and clear forewarning before they touch something that invalidates labels.

**Schema change confirmation dialog (positive framing):**

When the user modifies a label-invalidating control, the confirmation dialog:

1. Explain the change and its concrete effect:
  - Title: *"Update schema and re-label?"*
  - Body (Verified only): *"Changing [specific thing, e.g., 'the allowed values for damage_type'] changes what a correct answer looks like. Your {N} labeled images will return to Unlabeled for re-labeling under the new schema. The model will re-propose labels and your prior labels are shown as reference."*
  - Body (Verified + Auto-Labeled): when Auto-Labeled examples exist, append: *"This will also revert {M} Auto-Labeled examples to Unlabeled. You can re-run Batch Labeling when ready. Your improved Guidance and ICL examples carry over to the new run."*
2. Show what happens next:
  - *"Your prior labels are preserved as read-only reference."*
  - *"The model re-proposes labels under the new schema."*
  - *"Prior edits are reviewed first to rebuild context quickly."*
3. Actions: **"Update and Re-label"** (primary) / **"Cancel"** (secondary)

Avoid language that implies the user made a mistake (no "warning", "caution", "are you sure", "irreversible").

**Post-save banner:**

> Renames apply directly. Changing what a correct answer looks like invalidates labels and returns them to Unlabeled. The model re-proposes labels under the new schema and shows your prior labels as reference. Look for the label-invalidation icon on controls that trigger it.

This surfaces the edit policy at the moment it matters: when the user tries to edit, not during initial onboarding.

#### 6.6.9 Accessibility

- Stable tab order across all controls (field builder rows, constraint inputs, action buttons).
- Error messages announced to screen readers via `aria-live` regions.
- On save with errors, the page scrolls to the first error and focuses the offending control.
- Drag-to-reorder supports keyboard alternatives (move up/down via arrow keys or action menu).
- Status badge and footer button states communicated via `aria-disabled` and `aria-label`.
- Card headers are keyboard-focusable.

#### 6.6.10 Wireframe (Illustrative)

```text
+-----------------------------------------------------------------------+
|  Create Guidance                                  Schema: Valid        |
|  Project: My Damage Inspection                                        |
+-----------------------------------------------------------------------+
|                                                                       |
|  Start from: [Blank                                               v]  |
|                                                                       |
|  DESCRIPTION                                                          |
|  Describe the task and what the model should focus on in each image.  |
|  +---------------------------------------------------------------+   |
|  | Classify each image by damage type and severity...             |   |
|  +---------------------------------------------------------------+   |
|                                                                       |
|  SCHEMACORE                                                           |
|  Core fields = required + evaluated                                   |
|  Aux fields = optional + not evaluated                                |
|                                                                       |
|  CORE FIELDS (Required, Evaluated)                                    |
|                                                                       |
|  ::  damage_type   [Enum     v]  Core            [del][>Aux]          |
|      Values: [crack] [dent] [+]                                       |
|                                                                       |
|  ::  severity      [Integer  v]  Core            [del][>Aux]          |
|      Min: [1]  Max: [5]                                               |
|                                                                       |
|  [+ Add Core Field]                                                   |
|                                                                       |
|  AUX FIELDS (Optional, Not Evaluated)                                 |
|                                                                       |
|  Rationale notes                                  [ Off ]             |
|  Optional. When enabled, the Teacher explains visible evidence.       |
|                                                                       |
|  [+ Add Aux Field]                                                    |
|                                                                       |
|                                                                       |
|  RULES & EDGE CASES                                                   |
|  +---------------------------------------------------------------+   |
|  | If damage is partially obscured, classify based on the         |   |
|  | visible portion only...                                        |   |
|  +---------------------------------------------------------------+   |
|                                                                       |
|  v  Derived JSON Schema                                               |
|  v  Example label output                                              |
|                                                                       |
+-----------------------------------------------------------------------+
|                                        [Cancel]  [Save Guidance]      |
+-----------------------------------------------------------------------+
```

### 6.7 Generation Controls

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

| Value | Runtime toggle? | Reasons by default? | UI visibility | `thinking_mode_effective` |
|---|---|---|---|---|
| `none` | No | No | Hidden | Always `"off"` |
| `always_on_reasoning` | No | Yes (always) | Hidden | Always `"on"` |
| `qwen_enable_thinking` | Yes (`chat_template_kwargs.enable_thinking`) | Yes (toggle controls) | Visible only when probe-supported | Follows user input only when probe-supported |
| `kimi_thinking` | Yes (`chat_template_kwargs.thinking`) | Yes (toggle controls) | Visible only when probe-supported | Follows user input only when probe-supported |

- **Default: ON.** When ON for a probe-supported toggle model, no override is sent (model uses its default reasoning behavior).
- **When OFF** for a toggle-supporting model, the system sends model-specific request fields to disable thinking:
  - `qwen_enable_thinking` → `chat_template_kwargs: {"enable_thinking": false}`
  - `kimi_thinking` → `chat_template_kwargs: {"thinking": false}`
- The toggle MUST be **hidden or disabled** in the UI for models where `ModelConfig.thinking_toggle.mode` is `"none"` (e.g., Mistral VLMs — no toggle, no reasoning) or `"always_on_reasoning"` (e.g., Step 3.7 Flash — no working toggle, always reasons). Omni uses `qwen_enable_thinking` and exposes the toggle.
- Shown only when the selected model's `thinking_toggle.mode` is `"qwen_enable_thinking"` or `"kimi_thinking"` **and** `thinking_toggle_support="supported"`.

The split between `none` and `always_on_reasoning` is load-bearing for the §6.2 token budget. Models in the `always_on_reasoning` class burn output tokens on internal `<think>` reasoning regardless of the user's input; the budget MUST allocate `MODEL_REASONING_HEADROOM_TOKENS` to keep visible content from being truncated. Models in the `none` class do not reason and do not need the headroom — allocating it would just shrink effective input tokens for no benefit.

#### 6.7.4 Thinking Override Acceptance Check and Runtime Gating

**Thinking override acceptance check (normative).** This check applies only when `ModelConfig.thinking_toggle.mode` is `qwen_enable_thinking` or `kimi_thinking`. It verifies that the endpoint accepts the model-specific request fields used to disable thinking. It does **not** verify reasoning quality and does **not** apply to models whose reasoning is controlled via prompts, environment variables, or other deployment-level mechanisms. If `thinking_toggle.mode` is `"none"` or `"always_on_reasoning"`, set `thinking_toggle_support=unsupported` without checking — there is no runtime toggle to probe in either case.

Check request: one `/v1/chat/completions` request with the model-specific thinking-off fields:

```json
{
  "model": "<model_name>",
  "messages": [
    { "role": "user", "content": "Reply with OK." }
  ],
  "max_tokens": 4,
  "chat_template_kwargs": { ...mode-specific thinking-off fields... }
}
```

Mode-specific fields:

- `qwen_enable_thinking` → `{"enable_thinking": false}`
- `kimi_thinking` → `{"thinking": false}`

Bounded by enforced deadline (`HTTP_DEADLINE_INTERACTIVE_S`).

Classification:

- Non-error response → mark `supported`.
- 4xx clearly attributable to the thinking override fields (`chat_template_kwargs`) → mark `unsupported`; record sanitized provider error for audit.
- Timeout or 5xx → leave `thinking_toggle_support=unknown`; record the error for audit and retry on next check trigger.
- Persist result on `model_config_id`; reuse for subsequent invocations.

Only `thinking_toggle_support="supported"` authorizes
`chat_template_kwargs` on normal inference. When support is `unknown` or
`unsupported`, the system MUST omit the override, hide the UI toggle, treat the
model's natural default as Thinking ON, and reserve reasoning headroom. This
same rule applies to interactive proposals, evaluation, Batch Labeling, and
rationale regeneration.

**Runtime rejection handling:**

If a request with thinking override is rejected by the endpoint at runtime:

1. Mark `ModelConfig.thinking_toggle_support="unsupported"`.
2. **Interactive proposals:** auto-retry once without the thinking override and show a warning banner to the user.
3. **Evaluation / Batch Labeling:** fail the run (no silent fallback; reproducibility matters).

#### 6.7.5 System-Controlled and Excluded Parameters

These NIM parameters are NOT user-controlled:

**System-controlled (always set by the system):**

- `response_format`: system chooses `json_schema` when supported (§6.2).
- `max_tokens`: controlled by the existing prompt budget system; NIM default is 16 so the system MUST always set it explicitly (§6.2).

**Excluded from user controls:**

- `stop`, `ignore_eos`: can break JSON, intended for benchmarking.
- `min_tokens`: can cause repetitive content if not paired carefully.
- `logprobs`, `prompt_logprobs`: observability features, not needed for Generation Controls.

#### 6.7.6 Request Construction (Normative)

For any model invocation that generates a label (interactive, evaluation, batch label), the system constructs the OpenAI-compatible request as follows:

**Step 1, sampling parameters from preset:**

Set `temperature` and `top_p` from the resolved preset (§6.7.1).

**Step 2, thinking toggle → model-specific fields:**

If Thinking=OFF and the model supports the toggle (`thinking_toggle.mode` is `qwen_enable_thinking` or `kimi_thinking`):

- `qwen_enable_thinking` → add `chat_template_kwargs: {"enable_thinking": false}`
- `kimi_thinking` → add `chat_template_kwargs: {"thinking": false}`

If Thinking=ON, or `thinking_toggle.mode` is `"none"` or `"always_on_reasoning"`: send no thinking override. The two no-toggle modes never inject `chat_template_kwargs` regardless of the user's `thinking_default_on` or per-attempt override — the model's reasoning behavior is fixed at inference (off for `none`, on for `always_on_reasoning`). The §6.2 budget honors the resolved `thinking_mode_effective` (per §6.7.3 taxonomy) when allocating reasoning headroom.

**Step 3, seed injection by purpose:**

- If purpose is `evaluation` or `batch_label`: include `seed = seed_effective` derived per §2.1 seed policy.
- If purpose is `interactive_proposal`: omit `seed`.

**Step 4, visual budget → `mm_processor_kwargs`:**

- Resolve the effective `visual_budget_preset_key` (project default or per-attempt override).
- Look up the model's `visual_budget_mode` on ModelConfig.
- If `visual_budget_mode` ≠ `none` and `visual_budget_support` = `supported`: resolve the preset to model-specific `mm_processor_kwargs` per §6.9.3 and include in the request body.
- If `visual_budget_mode` = `none` or `visual_budget_support` = `unsupported`: omit `mm_processor_kwargs`.
- Persist `visual_budget_preset_key` and `visual_budget_params_effective` on the Operation Record (§13.1).

**Step 5, output budget + structured generation:**

- Always set `max_tokens = max_tokens_effective` from the existing budget system (§6.2).
- If structured generation is supported for this model, set `response_format: { type: "json_schema", json_schema: ... }` (§6.2).

**Step 6, automatic retry on truncation:**

If `finish_reason="length"` or output appears truncated:

- Retry once with increased output budget (budget-system change only).
- Re-pack ICL as needed (relevance-tail pruning per §6.2).
- Do NOT change the sampling preset, thinking toggle, or visual budget preset on automatic retry.

#### 6.7.7 UX Contract

**Labeling screen** (top bar, adjacent to Teacher model selector):

- **Output Stability:** Precise | Explore (toggle or segmented control)
- **Thinking:** ON / OFF toggle (default ON; hidden if model does not support it)
- Teacher, Output Stability, Thinking, and Visual Budget are persisted project
  defaults for the next proposal. Changing a control MUST NOT re-run an
  in-flight or displayed proposal. Save, Skip, and Retry MUST NOT dispatch the
  next proposal until an in-progress top-bar settings write has completed; a
  failed write MUST be visible and MUST NOT silently fall back while presenting
  the rejected selection as active.

**Helper copy (non-blocking):**

- "Precise is best for stable JSON labels. Explore adds variation when the model is stuck in a rut."
- "Turning Thinking OFF can reduce latency and truncation on some reasoning models."

### 6.8 Schema Refinement Reminders

After early labeling, SMEs may realize their Core schema needs adjustment (wrong categories, missing fields, constraints too narrow/wide). The cost of re-labeling grows with label count: a schema change after 10 labels means 10 images to re-label; after 500, it means 500. The system nudges the user twice: once early when re-labeling is cheap, and again as a last call before the cost becomes significant.

#### 6.8.1 Triggers

Two reminders fire at configurable Verified count thresholds, each at most once per project:

- **First reminder** (default: `SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1 = 10`): early signal. The SME has seen enough variety to spot schema problems, and re-labeling cost is trivial.
- **Second reminder** (default: `SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2 = 35`): last call. The SME has real momentum; re-labeling is noticeable but still recoverable. After this, the system stops asking.

Rules:

- Each reminder MUST fire at most once per project.
- If the first reminder was dismissed and the Verified count later crosses the second threshold, the second reminder fires independently.
- Neither reminder fires if the user has already edited Guidance post-save in this project (they are already aware of the edit flow).
- If the Verified count crosses both thresholds before either fires (e.g., bulk verification), only the higher applicable reminder fires.

#### 6.8.2 Presentation

Both reminders are dismissable inline hints on the labeling screen (same visual pattern as the cold-start notice), shown after the Nth Accept/Edit:

**First reminder** (at threshold 1):

> *"Need to adjust your schema? Fewer labels to re-do now."*
>
> **[Review Schema]** (navigates to Edit Guidance) · **[Dismiss]**

**Second reminder** (at threshold 2):

> *"You have {N} labels. Schema changes mean more images to re-label."*
>
> **[Review Schema]** (navigates to Edit Guidance) · **[Dismiss]**

Each hint is brief, scannable, and non-blocking. It provides a direct link to Edit Guidance and never reappears in the same project after dismissal.

**Priority vs the first-pool evaluation banner (§7.1).** While either reminder is visible and undismissed, the first-pool evaluation recommendation banner (§7.1, rendered by `EvaluationStrip.tsx`) MUST be suppressed. The SME sees one nudge at a time; schema refinement wins the slot because a semantic Core change invalidates existing labels (§4.4.1), making the "fewer labels to re-do now" window the entire point of the reminder. Deferring the first evaluation by a handful of labels is acceptable because evaluation runs in the background and the persistent Test Pool counter keeps the option discoverable. Suppression applies only to the first-pool banner; `configuration_change` and `icl_growth` banners are unaffected.

#### 6.8.3 Configuration

- `SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1: int` (default: `10`; project-scoped).
- `SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2: int` (default: `35`; project-scoped).
- Set either threshold to `0` to disable that reminder. Set both to `0` to disable reminders entirely.
- Persisted dismissed state: `schema_refinement_reminders_dismissed: int` on the Project record (default: `0`; tracks how many reminders have been dismissed; reset to `0` on semantic Core change so the SME gets fresh reminders after a schema change).

### 6.9 Visual Budget Controls

Visual Budget Controls manage how images are preprocessed before model inference, directly influencing the number of visual tokens consumed per image. This is a distinct control plane from text-side Generation Controls (§6.7). Generation Controls govern output sampling behavior; Visual Budget Controls govern input image token spend.

Support for visual token controls varies by model and deployment. Some VLMs expose request-level `mm_processor_kwargs` with model-specific parameter shapes; others do not support any visual budget control. The system treats this as a capability-gated feature: controls are only available when the active model supports them.

#### 6.9.1 Visual Budget Modes

Each model's visual budget capability is described by `visual_budget_mode` on its ModelConfig (§13.10):

- `none`: model does not expose visual token controls. No `mm_processor_kwargs` sent.
- `mm_processor_size`: model accepts `mm_processor_kwargs` with `size.shortest_edge` and `size.longest_edge` (e.g., Cosmos Reason2). Despite their names, this processor uses both values as lower/upper **pixel-area bounds**, not linear edge lengths. NVIDIA documents that Cosmos Reason2 performs best with 16k multimodal tokens or fewer.
- `mm_processor_pixels`: model accepts `mm_processor_kwargs` with `images_kwargs.min_pixels` and `images_kwargs.max_pixels` (e.g., Cosmos Reason1-family). Each image maps to between 4 and 16,384 tokens depending on image size.
- `mm_processor_tiles`: model accepts `mm_processor_kwargs` with `max_num_tiles` (e.g., Nemotron Nano VL). Higher values allow higher-resolution media at higher compute cost.

Seeded defaults for the model catalog (§4.8):

- `nvidia/cosmos-reason2-8b` → `mm_processor_size`
- `nvidia/cosmos-reason2-2b` → `mm_processor_size`
- `mistralai/mistral-large-3-675b-instruct-2512` → `none` (does not support request-level `mm_processor_kwargs`)
- `nvidia/nemotron-nano-12b-v2-vl` → `mm_processor_tiles`

Container-level `NIM_MM_PROCESSOR_KWARGS` defaults MAY apply even when request-level parameters are not sent. The system records what it sent per invocation, not what the container default is.

#### 6.9.2 Visual Budget Support Probe

The system probes whether the active model actually accepts `mm_processor_kwargs` at runtime. Invalid `mm_processor_kwargs` can produce HTTP 500 from NIM, so a single failing request cannot reliably distinguish "model lacks visual budget support" from "image/request path is broken." The probe therefore runs in two stages.

`visual_budget_support` on ModelConfig: `unknown` / `supported` / `unsupported`.

If `visual_budget_mode=none`, set `visual_budget_support=unsupported` without probing.

**Probe image:** the backend MUST generate a deterministic, non-uniform **512×512 RGB PNG** with no alpha channel and no embedded metadata. The image MUST be generated in-process and MUST NOT depend on external URLs or files. The exact pixel pattern is implementation-defined and not normative. The image is sent as a base64 data URL (`data:image/png;base64,<PROBE_IMAGE_B64>`) in the `image_url` content part. The purpose is to verify image transport, decoding, and acceptance of `mm_processor_kwargs` — not model accuracy or visual understanding.

512×512 is chosen because it aligns with documented NVIDIA preprocessing defaults: Cosmos Reason2 documents `max_pixels: 262144` (= 512×512), and Nemotron Nano VL uses `image_size: 512` in its preprocessing config. This size avoids degenerate tiny-image edge cases while keeping the probe cheap.

**Probe request (normative):** both stages use the same minimal `/v1/chat/completions` request:

```json
{
  "model": "<model_name>",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Reply with OK." },
        {
          "type": "image_url",
          "image_url": { "url": "data:image/png;base64,<PROBE_IMAGE_B64>" }
        }
      ]
    }
  ],
  "max_tokens": 4,
  "stream": false
}
```

Both stages are bounded by the enforced deadline (`HTTP_DEADLINE_INTERACTIVE_S`).

**Stage 1 — baseline probe:** send the probe request **without** `mm_processor_kwargs`. This confirms the model accepts images and the endpoint is reachable.

**Stage 2 — capability probe:** send the same request again, adding mode-specific `mm_processor_kwargs`. The values MUST be documented non-default examples from NVIDIA VLM documentation:

- `mm_processor_size` → `{ "mm_processor_kwargs": { "size": { "shortest_edge": 1568, "longest_edge": 262144 } } }`
- `mm_processor_pixels` → `{ "mm_processor_kwargs": { "images_kwargs": { "min_pixels": 1568, "max_pixels": 262144 } } }`
- `mm_processor_tiles` → `{ "mm_processor_kwargs": { "max_num_tiles": 3 } }`

**Classification:**

- Baseline succeeds + capability succeeds → mark `supported`.
- Baseline succeeds + capability fails (4xx/5xx attributable to `mm_processor_kwargs`) → mark `unsupported`; record sanitized provider error for audit.
- Baseline fails → leave `visual_budget_support=unknown`; record the baseline error for audit. Do not attempt Stage 2.

Persist result on ModelConfig; reuse for subsequent invocations.

#### 6.9.3 Visual Budget Presets

User-facing presets (parallel to Output Stability presets, §6.7.1):

- **Fast**: fewer visual tokens, fastest inference, may miss fine detail.
- **Balanced**: moderate visual token spend, a faster lower-resolution option.
- **High Detail** (default): maximum visual tokens at the Cosmos-max resolution, best for small defects, text, or subtle cues; slowest inference. This is the project default because an A/B study (evidence retained in the project's internal engineering archive) found image resolution is the dominant perception lever for Cosmos-Reason / Qwen3-VL — maxing the visual budget lifted downstream accuracy at every ICL depth — so all tunable models default to the highest visual fidelity.

Each preset resolves to model-specific `mm_processor_kwargs` values based on the model's `visual_budget_mode`:

**mm_processor_size** (pixel-area bounds; internal policy defaults):

- Fast: `{ "mm_processor_kwargs": { "size": { "shortest_edge": 1568, "longest_edge": 65536 } } }`
- Balanced: `{ "mm_processor_kwargs": { "size": { "shortest_edge": 1568, "longest_edge": 131072 } } }`
- High Detail: `{ "mm_processor_kwargs": { "size": { "shortest_edge": 1568, "longest_edge": 262144 } } }`

**mm_processor_pixels** (internal policy defaults; adjust per deployment):

- Fast: `{ "mm_processor_kwargs": { "images_kwargs": { "min_pixels": 1568, "max_pixels": 65536 } } }`
- Balanced: `{ "mm_processor_kwargs": { "images_kwargs": { "min_pixels": 1568, "max_pixels": 131072 } } }`
- High Detail: `{ "mm_processor_kwargs": { "images_kwargs": { "min_pixels": 1568, "max_pixels": 262144 } } }`

Operator-provided preset values are preserved as explicit overrides. The
loader only ensures that the product-required `native` preset remains
available for Student training-parity evaluation.

**mm_processor_tiles** (internal policy defaults; adjust per deployment):

- Fast: `{ "mm_processor_kwargs": { "max_num_tiles": 8 } }`
- Balanced: `{ "mm_processor_kwargs": { "max_num_tiles": 16 } }`
- High Detail: `{ "mm_processor_kwargs": { "max_num_tiles": 32 } }`

**none**: no `mm_processor_kwargs` sent regardless of preset.

When `visual_budget_support=unsupported` (probe rejected) and `visual_budget_mode` is not `none`:

- Interactive proposals: omit `mm_processor_kwargs` and show a non-blocking warning: *"Visual budget controls not available for this model."*
- Evaluation / Batch Labeling: omit `mm_processor_kwargs`; persist `visual_budget_params_effective: null`.

#### 6.9.4 Deterministic Policy for Evaluation and Batch Labeling

Visual budget presets affect image preprocessing and therefore model output. To ensure apples-to-apples comparisons:

- Evaluation and Batch Labeling MUST use the project's `visual_budget_preset_key` (no per-example variation).
- Evaluation runs MUST snapshot `visual_budget_preset_key` alongside other configuration (Guidance, ICL settings, Generation Controls).
- Different visual budget presets across evaluation runs MUST be explicitly labeled in comparison views.

#### 6.9.5 Per-Invocation Persistence

Every model invocation that processes images MUST persist on the Operation Record (§13.1):

- `visual_budget_preset_key: string | null`: the preset key used (e.g., `"fast"`, `"balanced"`, `"high_detail"`); null when model does not support visual budget controls.
- `visual_budget_params_effective: object | null`: the exact `mm_processor_kwargs` object sent in the request; null when not sent.

#### 6.9.6 UX Contract

**Labeling screen** (top bar, adjacent to Generation Controls):

- **Visual Budget:** Fast | Balanced | High Detail (segmented control or dropdown; default High Detail).
- Hidden when the active Teacher's `visual_budget_support=unsupported` (which includes `visual_budget_mode=none` models, since §6.9.2 auto-sets `unsupported` for them).

**Retry:** Visual Budget preset is overridable alongside Teacher model, Guidance, and Generation Controls.

**Helper copy (non-blocking):**

- "Fast uses fewer visual tokens for quicker inference. High Detail maximizes resolution, best for fine defects or small text."

### 6.10 Context Budget Indicator (Removed in v1.0)

**Absent from v1.0.** The context-budget indicator (green/yellow top-bar dot) and its reporting pipeline are not part of the public API or Operation Record. July 2026 ICL depth studies established Selective-K: a small number of highly relevant exemplars beats context-filling, so "room for more examples" is not a signal the SME should act on, and the indicator's capacity guidance pointed in the measured-worse direction. Token- and image-budget **enforcement** remains (relevance-tail pruning, §6.2) and is observable via the ICL-selection log point and `icl_images_attached_count`; the adjacent ICL chip carries the one load-bearing top-bar signal (how many examples informed the proposal). Section number retained so existing cross-references stay resolvable.

### 6.11 Inference Contract

The system defines an **Inference Contract** for each runnable model configuration. The Inference Contract specifies the effective output format, ICL demonstration format, and ICL sizing controls that apply at inference time. This is a first-class concept — not an implicit side-effect of scattered settings — and it governs how prompts are rendered, how ICL examples are formatted, and what evaluation runs snapshot for reproducibility.

**Inference Contract fields:**

- `output_field_mode` ∈ {`all`, `aux_and_core`, `core_only`}: which field groups the model is expected to produce in its output.
- `icl_field_mode` ∈ {`all`, `aux_and_core`, `core_only`}: which field groups are included when rendering ICL demonstration examples.
- `icl_max_examples: int | null`: maximum number of ICL examples (from `ICL_MAX_EXAMPLES` on the project or override). When null, the Teacher model's `default_icl_max_examples` (§6.2) governs the selection cap at invocation time.

**Teacher contract (fixed):**

The Teacher always uses: `output_field_mode=all`, `icl_field_mode=core_only`. This applies to Interactive Labeling, evaluation, and Batch Labeling. The Teacher returns the active output schema; `rationale_note` is requested last only when Guidance enables it and is otherwise omitted. ICL examples demonstrate only the Core correction signal.

**Student contract (derived from training):**

A Student's Inference Contract MUST match the field mode used to train that Student:

- `output_field_mode` = the `export_field_mode` from the training DatasetExport.
- `icl_field_mode` = `output_field_mode` by default (ICL demonstrations match the format the Student was trained on; in v1.0 Students run without ICL, so the field is contract provenance).

This ensures the model is conditioned on demonstration examples in the same format it learned during training. Running a `core_only`-trained Student with `all`-field ICL examples creates a format mismatch that distorts evaluation results and deployment behavior.

**Invariant:** a model is run and evaluated under the same field-mode contract it was designed or trained for. Two evaluation runs are only directly comparable if their effective Inference Contracts match.

**Sole authority for ICL field rendering:** the Inference Contract is the single source of truth for which fields appear in ICL demonstrations. No separate config key, feature flag, or per-invocation override exists for ICL rendering mode. All runtime ICL rendering decisions MUST be attributable to the effective Inference Contract only.

**Persistence:** the effective Inference Contract MUST be snapshotted on evaluation Run Records (§13.2.3) and persisted as part of Student training lineage (§13.13). Per-invocation `icl_example_keys_used[]` on Operation Records remains the deepest truth for exact prompt reconstruction; the Inference Contract captures the evaluation regime.

---

## 7. Evaluation Workflows

### 7.1 Evaluation Runs

During Interactive Labeling, the system evaluates by re-running the current Teacher+Guidance+ICL setup against the Test Pool to verify that accuracy has not degraded as new images are introduced and Guidance is refined.

**Triggering:**

Three events recommend or trigger an evaluation: first pool threshold, configuration change, and ICL growth. The behavior depends on the **Auto-Evaluate** toggle (OFF by default, visible on the labeling screen alongside the evaluation summary):

- **Auto-Evaluate OFF (default):** the system shows non-interrupting inline recommendation banners. The SME decides when to run.
- **Auto-Evaluate ON:** the system silently runs evaluations in the background on each trigger instead of showing banners.

The SME MAY also trigger an evaluation manually at any time, regardless of the toggle.

If a **gate-basis** evaluation is already running when a new gate-basis run starts (trigger or manual), the in-progress run MUST transition to `canceling` → `canceled` (with `status_reason="superseded_by_newer_evaluation"`) and a new run started with the current configuration. This newest-config-wins rule scopes to gate-basis (interactive Teacher) evaluations only: a Student benchmark run (`student_model_config_id` set, §9.5.2) snapshots an immutable configuration at creation — staleness cannot apply — so starting one supersedes nothing and a benchmark run is never superseded (it ends by completion, manual cancel, or restart recovery). The SME MAY also cancel a running evaluation manually via a [Cancel] action on the evaluation strip (e.g., when settings have changed and the run is stale). Cancellation is cooperative: the run transitions to `canceling`, no new inferences start, in-flight work unwinds, and the run reaches `canceled` once cleanup completes (§13.2.1).

**1. First evaluation:** when the Test Pool reaches `EVAL_FIRST_POOL_SIZE` (default: 5). This gives the SME early directional feedback well before the Scale-Up gate minimum (`SCALEUP_MIN_TEST_POOL_SIZE`, default: 60).

- OFF: banner *"{N} images reserved for testing. Run an evaluation to measure quality."* **[Evaluate]** **[Dismiss]**. The banner's [Evaluate] button triggers an evaluation run directly; [Dismiss] hides the nudge (the strip's persistent [Evaluate] button remains available). The banner message MUST NOT expose raw thresholds ("(threshold: 5)" or similar) — see §7.3.4 no-jargon rule.
- ON: runs automatically in background.
- **Suppression when a schema refinement reminder (§6.8) is visible:** to keep a single nudge on screen at a time, the first-pool banner MUST be suppressed while either schema refinement reminder (thresholds at 10 and 35 Verified, §6.8) is visible and undismissed. Schema refinement wins the slot because semantic Core changes invalidate existing labels (§4.4.1) — higher stakes than deferring the first evaluation by a handful of labels. Suppression applies only to this banner; configuration-change and ICL-growth banners are unaffected because neither fires at the low verified counts where a schema reminder lives.

**2. Configuration change:** when any tracked configuration changes (Guidance, Teacher model, Generation Controls, Visual Budget) and the Test Pool has members.

- OFF: banner *"Settings changed since last evaluation. Run an evaluation to check accuracy with current settings."* **[Dismiss]**. Dismissing acknowledges the change; the next banner appears on the next configuration change.
- ON: runs automatically in background.

**3. ICL growth:** when the number of ICL-eligible Edits (non-pool Edits) has doubled since the last baseline.

- OFF: banner *"{N} new edits since last evaluation. Run an evaluation to see if accuracy improved."* **[Dismiss]**
- ON: runs automatically in background.

ICL baseline tracking: the baseline is the ICL-eligible count at the most recent of either (a) the last completed evaluation (`icl_eligible_count_at_completion` on the Run Record) or (b) the last dismissed recommendation (`icl_recommendation_dismissed_at_count` on the Project Record). On **[Evaluate]**, the completed evaluation updates the baseline. On **[Dismiss]**, the current count becomes the new baseline. The next recommendation fires when the count doubles from the new baseline. Triggers often early (5→10, 10→20) and rarely later (50→100).

**Execution:**

- **Configuration snapshot:** when an evaluation starts, all semantic project configuration referenced by the run MUST be read once and persisted on the Run Record: `model_config_id`, `guidance_id`, `generation_preset_key`, `thinking_mode_effective`, `visual_budget_preset_key`, the effective **Inference Contract** (`output_field_mode`, `icl_field_mode`, `icl_max_examples`; §6.11), and a credential-free `runtime_config_snapshot`. That snapshot contains every mutable, result-shaping ModelConfig and NIM endpoint input consumed by inference plus the concrete sampling, visual-budget, token-budget, ICL-selection, and image-downscale values. It MUST also record `icl_eligible_count_at_start` (the number of ICL-eligible Edits at run creation). Later semantic configuration changes, capability re-probes, and endpoint rebinding MUST NOT affect the run. Credentials, filesystem authorization, timeouts, retries, concurrency, rate limiting, and emergency capability kill switches remain live operational policy and MUST NOT be persisted in the snapshot. Each Operation Record persists the exact ICL example keys and prompt hash because eligible data and embedding availability may evolve while a run is active. Two runs are only directly comparable if their effective Inference Contracts match.
- Evaluations MUST run in the background and MUST NOT block Interactive Labeling.
- Evaluation inferences MUST run concurrently. Concurrency is provider-aware and configurable: hosted endpoints use `EVAL_CONCURRENCY_HOSTED` (default: 1, rate-limit politeness); self-hosted/local NIMs use `EVAL_CONCURRENCY_SELF_HOSTED` (default: 8).
- **Sequential retry pass:** after the concurrent burst completes, if any per-example inferences failed (`timeout`, `endpoint_error`, or `schema_invalid` after exhausting per-invocation retries per §11), the system MUST retry each failed example sequentially (concurrency=1). Sequential retry avoids the rate-limit storm that may have caused the original failure. Each retry creates a new Operation Record linked to the original via `retry_of_inference_invocation_id`. If any example still fails after the sequential retry, the run is marked **incomplete**.
- When an evaluation starts, the system auto-creates a frozen pool version snapshot from the current live Test Pool membership (§4.3.3). The run references this snapshot.
- If the Test Pool has no members, evaluation MUST NOT proceed (§4.3.3).
- MUST use the same Interactive Labeling prompt pipeline.
- For “without ICL,” MUST force ICL to zero.
- Persist per-example Operation Records (`purpose=evaluation`) including `invocation_status`, `exact_match_pass`, and `latency_ms_end_to_end`.
- Incomplete runs do not satisfy the Scale-Up Readiness Gate (§7.3).

Evaluation uses only Verified examples under the current Guidance: the Test Pool, ground-truth Verified labels, and ICL context MUST all use labels created under the current `active_guidance_id`. After a semantic Core change (§4.4.1), old labels are deleted; as examples are re-labeled and the Test Pool rebuilds, evaluation becomes available naturally. No special lockout exists.

#### 7.1.1 Configuration Change Detection

After the first completed evaluation, the system MUST detect when the project's active configuration has diverged from the configuration captured in the most recent evaluation. This drives the configuration change recommendation banner (§7.1).

**Tracked configuration fields:**

The system compares the following fields from the most recent completed evaluation's Run Record against the project's current values:

- `guidance_id` (evaluation run) vs `project.active_guidance_id`
- `model_config_id` (evaluation run) vs `project.teacher_model_config_id`
- `generation_preset_key` (from evaluation run's Operation Records) vs `project.labeling_generation_preset_key`
- `thinking_mode_effective` (from evaluation run's Operation Records) vs `project.thinking_default_on` (mapped: `true` → `"on"`, `false` → `"off"`)
- `visual_budget_preset_key` (from evaluation run's Operation Records) vs `project.visual_budget_preset_key`

If any tracked field differs, the configuration has changed.

Detection is lightweight: it compares persisted values only, with no model invocation.

**UX contract:**

- While an evaluation is running: the evaluation summary line shows *"Evaluating..."* with the snapshotted configuration (e.g., *"Evaluating with Cosmos Reason2 8B, Guidance v3, Precise"*). If the user has changed settings since the run started, a non-blocking note: *"Running with settings from {run_created_at}. Current settings differ."* The SME is not blocked.
- When the evaluation completes: the summary line updates with new metrics.
- When no evaluation has run yet: no summary line (the first evaluation recommendation from Overview §6 handles this case).

**Interactions with Scale-Up Readiness Gate (§7.3):**

The gate always uses the most recent completed evaluation's **overall** metrics. The gate updates when a new evaluation completes.

#### 7.1.2 Returning vs New Metric Split

Because the Test Pool grows over time, a drop in overall accuracy can mean the model degraded or simply that a harder example joined the pool. To separate these signals, each evaluation reports metrics in three buckets.

**Bucket definitions:**

When an evaluation completes, the system compares the current pool version snapshot against the previous completed evaluation's snapshot:

- **Returning**: examples present in both the current and previous snapshots. These are images the system has tested before. Metrics on this set are the regression signal.
- **New**: examples present in the current snapshot but not in the previous. These are images that entered the pool since the last evaluation.
- **Overall**: all examples in the current snapshot (Returning + New combined).

On the first evaluation (no previous snapshot), all examples are reported as Overall only; there is no Returning/New split. After a semantic Core change creates a new Guidance version, evaluation history under the prior Guidance is not used as the "previous" baseline. The first evaluation under the new Guidance is treated as a fresh first evaluation (Overall only, no Returning/New split). Prior evaluation Run Records remain in history for audit.

**Required metrics per bucket:**

Each bucket MUST report the same metrics as overall evaluation (Appendix A.2): Exact Match rate, per-core-field match rate, and per-value precision/recall/F1 for categorical Core fields.

**Persistence:**

The Run Record (§13.2) MUST persist:

- `previous_pool_version: int | null` (the pool version snapshot of the previous completed evaluation; null on first evaluation)
- `returning_example_keys[]: string[]` (examples in both snapshots)
- `new_example_keys[]: string[]` (examples in current but not previous)
- `icl_eligible_count_at_completion: int` (number of ICL-eligible Edits at the time this evaluation completed; used for the ICL growth recommendation threshold)
- Per-bucket aggregate metrics (Returning, New, Overall)

**UX contract:**

The core of the evaluation display is comparing the previous Overall against the current Returning. These are the same images re-tested under the current setup, which directly answers "did we get better or worse?"

The labeling screen summary line places these side by side:

`Previous: 82% (15) → Same images now: 85% (15) · New: 60% (5 new) · Overall: 78% (20)`

- **Previous** is the Overall accuracy from the last completed evaluation.
- **Same images now** (Returning) is accuracy on those same images under the current setup. This is the fundamental comparison. A green delta (↑3%) or red delta (↓5%) makes the direction immediately visible.
- **New** and **Overall** are secondary context.

On the first evaluation: `Accuracy: 80% (5 images)` (no previous to compare against).

A **[Results]** button on the evaluation strip opens a detail panel showing the full metric breakdown: Returning/New/Overall accuracy comparison, per-core-field match rates, expandable per-value precision/recall/F1 for categorical Core fields, and coverage gaps (§7.1.3). The detail panel includes the run's configuration snapshot (Teacher, Guidance version, Generation Controls, pool version) so the SME can see what settings produced these results. Shipped as `ResultsPanel.tsx`, opened from the evaluation strip (`EvaluationStrip.tsx`) on the labeling screen.

**Persistence:**

The Run Record MUST also persist `previous_overall_exact_match: float | null` so the comparison can be rendered without loading the previous Run Record.

#### 7.1.3 Test Pool Coverage Warnings

After an evaluation completes, the system checks whether the test pool's ground truth labels cover all possible values defined in the Core schema. This is a simple set difference with no model invocation.

**Checked field types:**

- **Enum**: compare `allowed_values[]` against values present in test pool labels. Report any allowed value with zero representation.
- **Boolean**: check that both `true` and `false` appear in test pool labels. Report if only one value is present.
- **Integer** (with `minimum`/`maximum`): report the observed range in test pool labels vs the schema range (e.g., "severity: values 1-3 observed, 0 and 4 not represented").
- **Enum Set**: for each allowed value, check that at least one test pool label includes it.
- **String**: no coverage check (open-ended values).

**UX contract:**

If any gaps are found, show a non-blocking coverage warning below the evaluation summary:

*"Test pool has no examples with {field_name}='{value}' [, ...]. Evaluation results don't cover these values."*

Multiple missing values are grouped per field. The warning is informational; it does not affect the evaluation metrics, the gate, or any workflow.

**Persistence:**

The Run Record MUST persist `coverage_gaps[]: {field_name, field_type, missing_values[]}` (empty array when no gaps).

### 7.2 Evaluation Sources (`evaluation_source`)

Student variants are evaluated against the Test Pool from two distinct sources, and every evaluation run records which source produced its results in `evaluation_source`:

- **`tao`**: quality evaluation run inside TAO against a checkpoint. Produces accuracy metrics (Appendix A.2) but no serving metrics. Available automatically after training and quantization complete (§9.7.6).
- **`nim`**: evaluation run against a deployed NIM endpoint. Produces both accuracy metrics and serving metrics (latency, throughput, NIM profile metadata). Available after NIM deployment (§9.5.2).

TAO-sourced quality results use the system's canonical Core-field evaluator for authoritative scoring: TAO generates per-sample predictions, the system re-scores them (§9.7.6). NIM-sourced results use the standard evaluation pipeline (§7.1). Teacher evaluation runs (§7.1) carry `evaluation_source="nim"` and no `student_model_config_id`; Student NIM serving runs carry `evaluation_source="nim"` with the Student's `student_model_config_id`; TAO-rescored Student quality runs carry `evaluation_source="tao"`. Consumers that need the Teacher setup (the Scale-Up gate, the Returning/New baseline, the evaluation-trigger baselines) MUST filter on both fields.

The Compare & Benchmark screen (`CompareBenchmarkPage.tsx`) reads results from evaluation runs directly: the Teacher accuracy baseline is the most recent completed run whose Inference Contract equals the fixed Teacher contract (§6.11) and that carries no `student_model_config_id`; each Student card reads its TAO-rescored quality run (`quality_evaluation_run_id`) and its NIM serving run (`serving_evaluation_run_id`, which carries the `metrics.benchmarks` latency sweep).

Every Teacher identity on Compare's baseline card and chart MUST come from the Teacher snapshotted by that selected run's `model_config_id`, never the project's mutable current Teacher. Student deltas MUST be computed from and described as relative to that historical Teacher baseline. If the current Teacher differs, the screen MUST keep the historical attribution, name the current Teacher separately, and direct the SME to run a fresh evaluation to refresh the baseline. Copy describing a forthcoming local-NIM displacement continues to name the current Teacher because that operation acts on live project configuration, not the historical baseline.

**Absent from v1.0 — evaluation suites and the Student+ICL arm.** Students deploy and are evaluated bare (`icl_mode="disabled"`; F-W7 in §9.5.2), so an evaluation-suite grouping has no purpose in the public schema. Section number retained so existing cross-references stay resolvable.

### 7.3 Scale-Up Readiness Gate

The Scale-Up Readiness Gate determines whether the project meets configurable quality criteria for Batch Labeling. The gate is system-evaluated and automatically re-evaluated when underlying data changes (new Verified labels, new evaluation runs). The gate MUST NOT block Interactive Labeling, evaluation runs, or any other workflow.

#### 7.3.1 Gate Criteria

All five criteria MUST pass for the gate to report system-ready:

**1. Overall Exact Match.** The most recent completed evaluation run's overall Exact Match rate MUST meet or exceed `SCALEUP_EXACT_MATCH_THRESHOLD`. Only Teacher-contract runs (§7.2 discriminator) of the current schema era (§4.4.1) qualify — a prior-era run scored labels that no longer exist. If no qualifying run exists, this criterion fails.

**2. Per-core-field match rate.** In the same evaluation run, every Core field's individual match rate MUST meet or exceed `SCALEUP_PER_FIELD_MATCH_THRESHOLD`. If any single Core field falls below the threshold, this criterion fails and the gate MUST report which field(s) failed and their current rates.

**3. Minimum per-value F1.** In the same evaluation run, every value of every categorical Core field (enum, enum set, boolean) MUST have an F1 score meeting or exceeding `SCALEUP_MIN_PER_VALUE_F1_THRESHOLD`. This catches cases where the model systematically misclassifies or ignores specific values even when overall field-level metrics pass. If any value falls below the threshold, the gate MUST report which value(s) failed, their current F1, and the associated precision and recall. Per-value precision and recall are reported alongside F1 for diagnostic context but are not independent gate criteria.

**4. Accept rate (rolling window).** The system MUST compute a rolling Accept rate over the most recent `SCALEUP_ACCEPT_RATE_WINDOW` Verified labels (ordered by `verified_at`):

```text
accept_rate = count(verified_outcome = Accept) / count(all outcomes in window)
```

The rate MUST meet or exceed `SCALEUP_ACCEPT_RATE_THRESHOLD`. If fewer than `SCALEUP_ACCEPT_RATE_WINDOW` Verified labels exist, the denominator is the actual count and the threshold still applies. If zero Verified labels exist, this criterion fails.

**5. Minimum Test Pool size.** The current live Test Pool MUST contain at least `SCALEUP_MIN_TEST_POOL_SIZE` members. Below this count, evaluation results lack statistical significance.

#### 7.3.2 Gate Evaluation Timing

The system MUST re-evaluate gate criteria:

- After each new Verified label (Accept rate and pool size may change).
- After each completed evaluation run (evaluation metrics change).
- On Scale-Up hub page load (present current state).

Gate evaluation MUST be lightweight (query persisted metrics and counts; no model invocation). The gate reads results from the most recent completed evaluation; it does not trigger evaluations itself.

#### 7.3.4 Gate Status Reporting

The gate MUST report status as a structured object:

- `gate_status` ∈ {`not_ready`, `ready`}
- Per-criterion detail: `{criterion_name, passed: bool, current_value, threshold, message}`

The UI MUST present gate status in plain language (see Overview §7). The UI MUST NOT expose raw metric names or technical jargon. Example criterion messages:

- Pass: *”Model accuracy: 88% overall (need 80%). Passed.”*
- Fail: *”Per-field quality: 'damage_severity' at 72% (need 80%). Continue labeling or refine Guidance for this field.”*
- Fail: *”Per-value quality: 'scratch' in 'damage_type' has F1 67% (need 80%, precision 75%, recall 60%). The model is missing this category. Add more examples or refine Guidance.”*
- Fail: *”Accept rate: 65% over last 50 labels (need 80%). Continue Interactive Labeling.”*
- Fail: *”Test Pool: 24 examples (need 60). Continue labeling to grow the pool.”*
- Fail: *”No evaluation run found. Run Evaluation to measure quality.”*

---

## 8. Batch Labeling Workflows (Teacher-run)

Batch Labeling generates large volumes of Auto-Labeled fully synthetic data from Unlabeled images. Starting a Batch Labeling run requires the Scale-Up Readiness Gate to be approved (§7.3).

### 8.1 Purpose and Non-goals

Purpose:

- Generate large fully synthetic labeled dataset without SME validating each item.
- Use same Guidance and ICL behavior that performed well in interactive labeling/evaluation.

Non-goals:

- Auto-Labeled labels are not ground truth (§3.2). System MUST NOT convert Auto-Labeled into Verified without SME review.

### 8.2 Batch Labeling Run Execution

**Endpoint-use confirmation (UI policy boundary).** The Batch pre-run screen
MUST resolve the selected Teacher's `NimEndpointResponse.usage_policy` before
enabling launch and display it in the configuration summary. When the policy is
`evaluation_only`, clicking **Run Batch Labeling** MUST open a confirmation
immediately before the create request. The confirmation states that NVIDIA API
Catalog credits, including additional trial credits, do not authorize
production use; offers **Continue evaluation**, **Configure production
endpoint**, and **Cancel**; and links NVIDIA's API Trial Terms. Continue is the
only action that creates the run. An `operator_managed` endpoint launches
without this confirmation. `operator_managed` means that entitlement is the
operator's responsibility; it is not a Blueprint assertion that the endpoint
is commercially licensed. This disclosure is separate from the Teacher-quality
gate and does not change the Batch API's server-side readiness rules.

A run MUST:

1. Verify the Scale-Up Readiness Gate: `gate_status` MUST be `ready`. If not, the system MUST reject the run request with a message referencing the gate (§7.3).
2. Generate `batch_label_run_id`.
3. **Snapshot configuration:** read and persist on the Run Record: `model_config_id`, `guidance_id`, `generation_preset_key`, `thinking_mode_effective`, `visual_budget_preset_key`, and a credential-free `runtime_config_snapshot` containing every mutable, result-shaping ModelConfig and NIM endpoint input consumed by inference plus the concrete sampling, visual-budget, token-budget, ICL-selection, and image-downscale values. All subsequent steps in this run MUST use those semantic values, not live project defaults. Credentials, filesystem authorization, timeouts, retries, concurrency, rate limiting, and emergency capability kill switches remain live operational policy. The v1_0003 migration freezes valid non-terminal legacy NIM runs in snapshot version 1; startup upgrades resumable Batch snapshots to version 2 with the then-current semantic Settings before serving, or fails the affected run closed.
4. Resolve Generation Controls and Visual Budget from the snapshotted values (§6.7, §6.9). Seed injection uses `batch_label_run_id` as `scope_id` per §2.1 seed policy. Visual budget MUST use the snapshotted preset for all examples (no per-example variation).
5. Select input examples: by default, all examples with `state="Unlabeled"` (excluding `state="Omitted"`). When `include_auto_labeled=true`, also include examples with `state="Auto-Labeled"` — their existing Label records are replaced with new Auto-Labeled labels from the current Teacher+Guidance+ICL setup (prior Label records are overwritten; prior Operation Records are preserved for audit). This is useful when Guidance has improved (Description/Rules refined, ICL pool grown) and the SME wants Auto-Labeled data regenerated under the improved setup without requiring a Core schema change. When `BATCH_LABEL_RUN_LIMIT` is set, cap the run at that many examples (selected in ingestion order). Default: null (all eligible).
6. For each selected example (dispatched with provider-aware concurrency: hosted endpoints use `BATCH_LABEL_CONCURRENCY_HOSTED` (default: 1, rate-limit politeness); self-hosted/local NIMs use `BATCH_LABEL_CONCURRENCY_SELF_HOSTED` (default: 8) — same policy as evaluation, §7.1. A per-run `concurrency` override on the start request wins over both and is persisted on the Run Record so restart recovery resumes at the same width):
  - Render prompt using `project.active_guidance_id`.
  - Select ICL examples from `ICL_ELIGIBLE` as defined in Section 6.2, subject to prompt packing token budget, the per-model depth default (`default_icl_max_examples`, §6.2), and `ICL_MAX_EXAMPLES` when set. Skipped entirely when the run's `icl_mode` is `"disabled"` (§8.3).
  - Construct and invoke NIM request per §6.7.6 with deadline + bounded retries.
  - Persist operation record with `purpose=batch_label`, `label_tier="auto_labeled"`, and `batch_label_run_id`.
  - Persist attempt artifacts and validate/normalize output under SchemaCore.
  - Persist validation report (Core vs Aux) and references (`validation_report_ref` recommended).
  - Create or update a Label record with `label_status=auto_labeled`, `label_json` from the normalized output, `batch_label_run_id`, and `inference_invocation_id` linking to the operation record. Only schema-valid Core outputs produce Label records; schema-invalid outputs are recorded on the Operation Record only.
  - Set Example `state="Auto-Labeled"`.
7. Provide resumability and idempotency:
  - `{batch_label_run_id}:{example_key}` identifies one durable logical item. Restart resume MUST reuse its pending `inference_invocation_id`, not create a second Operation Record.
  - A terminal success is complete only when the same transaction committed the Operation outcome, an Auto-Labeled Label carrying the exact run and invocation IDs, and `Example.state="Auto-Labeled"`. Later `Verified` or `Omitted` SME state also completes the item because human disposition supersedes the machine label.
  - Recovery MUST validate that the frozen input list contains unique strings, exactly matches `examples_total`, and still resolves to the same project Examples before dispatch. It MUST retry torn or mismatched success under its original invocation ID and MUST fail closed on duplicate, foreign, invalid-status, or ambiguous operation lineage.
8. **Circuit breaker:** the system maintains a consecutive failure counter that tracks endpoint-availability failures. Counter rules: `timeout` → increment; `endpoint_error` → increment; `schema_invalid` → ignore (does not increment, does not reset); successful example → reset to 0. With concurrent dispatch (step 6), "consecutive" is counted in completion order; when the counter trips, no new work is dispatched, including work released from a foreground-priority hold; already-in-flight requests complete and are recorded (a dispatch stop, not preemption — the same philosophy as the foreground-priority hold, §1.3), and the run pauses after they drain. The run snapshots `BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD` (default: 10), and each authoritative item transaction persists both the updated streak and a durable tripped latch with its Operation/domain outcome. The latch remains true if a later in-flight success drains and resets the streak, so automatic restart recovery preserves the dispatch stop. If the snapshotted threshold is reached, the run MUST transition to `paused` with `paused_reason="circuit_breaker_threshold_reached"` and notify the user: *"Endpoint appears unreachable. [Resume] [Cancel]."* On explicit **Resume**, the run transitions `paused` → `queued` and its executor claims `running`; that user action resets the streak and latch, and processing continues from the next unprocessed example. On **Cancel**, a paused run transitions directly to terminal `canceled`; already-persisted results are retained. The circuit breaker prevents burning through rate limits and API credits on a down endpoint. See §13.2.2 for the full batch labeling state machine.

### 8.3 ICL Behavior for Batch Labeling

ICL is enabled for Batch Labeling by default. Batch Labeling MUST use the deterministic ICL selection algorithm (Section 6.2) and prompt packing rules, subject to per-model context window, prompt packing token budget, the per-model depth default (`ModelConfig.default_icl_max_examples`, §6.2), and `ICL_MAX_EXAMPLES` when set. The Interactive Loop builds Edited examples that serve as the corrective ICL signal improving proposal quality. The total number of examples processed in a run is governed by `BATCH_LABEL_RUN_LIMIT` (§8.2 step 5), not by ICL settings.

**Per-run ICL disable (F-S9 amendment, 2026-07-15).** The start request accepts `icl_mode ∈ {"enabled", "disabled"}` (default `"enabled"`), mirroring the evaluation API's field. When `"disabled"`, the run MUST skip ICL selection entirely and label at the Teacher's zero-shot form; the mode is snapshotted on the Run Record so restart recovery resumes the run under the same mode, and echoed on create/detail responses. *Rationale: ICL-negative teachers demonstrably exist (July 2026 campaign — local Omni on freiburg: zero-shot 0.525 vs depth-1 0.217, monotonic to 0.000 at depth 24), and before this field the only lever was setting `ICL_MAX_EXAMPLES=0` process-wide, which silently degrades every other project, interactive proposal, and evaluation sharing the backend, and is not a validated value. The prior sentence here — "There is no option to disable ICL for Batch Labeling" — is superseded by this amendment.*

### 8.4 Auto-Labeled Dataset Export

System MUST export Auto-Labeled dataset artifact using the Cosmos-RL dataset format defined in §9.3 (`annotations.json` + media archive).

Recommended:

- Export Verified train separately: `verified_train_llava.tar`
- Export Auto-Labeled separately: `auto_labeled_batch_labeled_llava.tar`

Downstream may merge with controlled weighting.

Exports MUST be first-class DatasetExport records (Section 13.5) including manifest ref enumerating included items + filters used.

Verified exports include only Verified examples under the current Guidance. Auto-Labeled export entries are training data only and require both a `label_status=auto_labeled` Label and an exact owning Example with `state="Auto-Labeled"`. Evaluation/testing exports MUST use `label_tier_filter=verified_only`; Auto-Labeled predictions are never an answer key. After a semantic Core change, old labels are deleted (§4.4.1), so only examples re-labeled under the new Guidance are exportable.

---

## 9. Student Training Workflows (Cosmos-RL / TAO VLM Student Training)

Student Training is optional.

### 9.1 Enablement Gate

- Student Training requires Student Training capability to be available for the deployment (Cosmos-RL / TAO VLM integration configured).
- Preflight MUST validate:
  - (1) TAO endpoint is reachable (§1.6).
  - (2) Base model has the `student_base` role (§4.8).
- Hardware and environment constraints (GPU count/memory, vGPU, driver/CUDA, disk) are the responsibility of the TAO deployment. This system communicates with TAO via REST API and cannot inspect remote infrastructure. If a job fails due to hardware constraints, the error surfaces through TAO job polling (§9.7.4) and is recorded on the TAOJob record.
- Known TAO/NIM constraints for reference (not enforced by this system's preflight):
  - NIM for VLMs does not support vGPU environments.
  - Cosmos Reason2 training: 8× A100 80 GB minimum (per NVIDIA Cosmos-Reason documentation; applies to both 8B and 2B unless independently validated otherwise).
  - 200 GB free disk is the installation floor (500 GB practical minimum for one retained train/evaluate/quantize suite; 1 TB preferred for variants/retries). Ubuntu 22.04 LTS is NVIDIA's recommended FTMS 6.26.3 baseline; Ubuntu 24.04 LTS is also Blueprint-live-validated. Driver 580.65.06+ (R580 branch), CUDA 13.0+. Note: TAO FTMS prerequisites documentation may reference the R570 driver branch, but TAO 6.26.3 containers use CUDA 13.0 which requires R580+. The R570 branch is insufficient.

### 9.2 Training Data Sources

- Verified (required base): Verified excluding Test Pool members (§6.2). Only labels verified under the current Guidance are training data. After a semantic Core change, old labels are deleted; only examples re-labeled under the new Guidance are training data.
- Optional Auto-Labeled: Batch-labeled outputs filtered (at minimum Core schema-valid).

### 9.3 Dataset Export Contract

When enabled, system MUST export in the Cosmos-RL TAO-native dataset format. The wire format below is derived from NVIDIA's Cosmos-RL documentation and upstream LLaVA custom-data guidance. Because TAO/Cosmos-RL is the downstream consumer, this spec standardizes on the NVIDIA-documented contract.

#### 9.3.1 Archive Layout

Each export artifact is a dataset folder containing:

- `images.tar.gz` (or `images/` directory): all referenced image files, organized by relative path.
- `annotations.json`: the annotation file defining samples and conversations.

Implementations MUST produce archives where every `images[*]` path in `annotations.json` resolves to an existing file in the media payload.

Selection-time existence is not authority to read later. The exporter MUST
re-authorize every source image under the current `IMAGE_ROOT` while assembling
the archive and add bytes from that opened regular-file descriptor. A missing
source MAY be omitted from both annotations and media during selection; a
policy denial or a source that becomes unavailable after selection MUST fail
the build and remove partial artifacts.

#### 9.3.2 `annotations.json` Structure (Normative)

`annotations.json` is a **top-level JSON array** of sample objects. Each sample object has the following fields:

- `id: string` (required). Maps to `example_key`.
- `images: string[]` (required). Array of relative paths to image files within the archive. For this system's single-image workflow, `images` MUST contain exactly one entry. The path is relative to the archive root (e.g., `"images/example_000001.png"`). Note: NVIDIA Cosmos-RL documentation uses `images` (plural array), not `image` (singular string). Upstream LLaVA custom-data guidance uses singular `image`; this spec follows the NVIDIA Cosmos-RL convention because TAO is the downstream consumer.
- `conversations: object[]` (required). Array of exactly two conversation turn objects:
  - **Turn 1 (human):** `{"from": "human", "value": "{rendered_serving_prompt}"}` — the **rendered §6/D.1 serving prompt in its zero-shot, self-contained form** (F-W7 amendment 2026-07-14; see below). `<image>` appears within it as a **literal token** (the training framework replaces it with image embeddings), exactly once, at the query-image position the serving prompt uses.
  - **Turn 2 (assistant):** `{"from": "gpt", "value": "{label_json_string}"}` where `{label_json_string}` is the label serialized as a JSON **string** (see serialization rule below). Content is governed by `export_field_mode`.

**F-W7 amendment (2026-07-14) — the human turn is the serving prompt, not a short task sentence.** Students are served — and serving-evaluated (§9.5) — with the full D.1 proposal prompt. The original contract (`"<image>\n{task_prompt}"` with `{task_prompt}` derived from the Guidance Description) trained Students on a prompt distribution they never see at deployment; measured live (2026-07-14, freiburg CR2-2B bf16, same checkpoint, same 120 held-out keys): 0.317 EM under the training prompt vs 0.092 under the serving prompt — a 3.4× systematic loss attributable entirely to prompt skew. The human turn MUST therefore be the D.1 proposal prompt rendered exactly as a deployed Student receives it: zero ICL examples (Student serving evaluation runs `icl_mode="disabled"`), `STRUCTURED_GENERATION_ATTEMPTED=false`, the compact prompt-visible schema included, verbose Guidance prose omitted, and the query-image slot replaced by the literal `<image>` token. Two residual deltas from the live serving request are deliberate and accepted:

1. The SYSTEM message text is folded into the top of the human turn (this two-turn contract has no system slot); at serve time it arrives as a separate system message.
2. Serve-time MAY additionally attach the derived JSON schema via structured generation (§6.2), which cannot be represented in training data; the training prompt always renders the self-contained Output Contract variant.

The dual-path evaluation agreement (direct-harness vs product serving-eval on the same held-out keys) is the standing regression test for the size of these residual deltas. Both training and evaluation exports MUST render the identical human turn (§9.7.6) — an evaluation export carrying a different prompt would score the Student out of distribution on the FTMS evaluate leg. Canonical implementation: `services.prompt_service.render_training_conversation_prompt`.

Turn field names are `from` and `value`, not `role` / `content`. This matches both NVIDIA Cosmos-RL documentation and upstream LLaVA examples.

**Label serialization rule:** the assistant turn's `value` field is always a text string. When this system exports structured labels, the label object MUST be serialized into a deterministic JSON string using compact encoding (no extra whitespace). Its order MUST match the serving prompt: other Aux fields and Core fields retain their base relative order, with `rationale_note` last when included. The assistant turn MUST NOT contain a nested JSON object; it MUST contain a JSON string whose content happens to be valid JSON.

**Normative example:**

```json
[
  {
    "id": "example_000001",
    "images": [
      "images/example_000001.png"
    ],
    "conversations": [
      {
        "from": "human",
        "value": "You are a vision labeling assistant. Output valid JSON only.\n\nLabel schema:\n- damage_type (required): \"dent\" | \"scratch\"\n- severity (required): integer 1..3\n- rationale_note (optional): string\nUse only these fields.\n\nNow label the QUERY image below.\n<image>\n\nReturn one JSON object matching the label schema above.\nAdd rationale_note last. Keep it to one or two short, image-specific sentences.\nDescribe the subject the task asks about. Distinguish a physical carrier from content shown on it only when that distinction matters to the task.\nState uncertainty when needed. Do not invent evidence or merely repeat the label.\n\nReturn JSON only."
      },
      {
        "from": "gpt",
        "value": "{\"rationale_note\":\"dent visible on front-left corner\",\"damage_type\":\"dent\",\"severity\":2}"
      }
    ]
  }
]
```

(The gpt turn's field content remains governed by `export_field_mode`; a `core_only` export answers the same serving prompt with Core fields only.)

#### 9.3.3 TAO Config Field Naming (Implementation Note)

Each cosmos-rl action class consumes a different config schema. Implementations MUST bind the dataset under the keys the action's worker actually reads (live-verified 2026-04-29 against TAO FTMS 6.26.3 + cosmos-rl 6.26.3):

- **`train`** — `specs.custom.train_dataset.{media_path, annotation_path}`. cosmos-rl's `CustomConfig` Pydantic schema (`/opt/cosmos_rl/tao_sft_example.py`) requires the `train_dataset` key by name; the previously-documented `custom.dataset.media_path` shape surfaces in the container as `ValidationError: train_dataset Field required`.
- **`evaluate`** — top-level `specs.dataset.{media_dir, annotation_path}`. cosmos-rl's `ITSEvaluator` reads `config["dataset"]` directly (`cosmos_rl/evaluation/its_evaluator.py:107`); the train-side `custom.train_dataset` shape crashes evaluate with `KeyError: 'dataset'`.
- **`quantize`** — top-level `specs.dataset.{media_dir, annotation_path}`. **F11 (Phase 12 amendment 2026-05-05)**: live-verified that cosmos-rl-quantize's argparse CLI accepts ``--media_dir`` (matches `dataset.media_dir`) and rejects ``--media_path`` (the train-side key) with `unrecognized arguments`. The earlier guidance said quantize shared train's schema; that was incorrect — quantize's calibration replays the training corpus but the binding key naming follows the cosmos-rl-quantize CLI signature (which differs from cosmos-rl's CustomConfig). The Blueprint's `services.tao_job_service.apply_dataset_binding(action="quantize")` was updated to produce the same shape as evaluate.
- **`inference`** — no dataset binding (caller responsibility).

`media_path` (used by train) and `media_dir` (used by evaluate + quantize) are different keys with the same semantics: they point at the parent directory under which images resolve. The export archive layout is the same regardless of action; only the spec key the worker reads differs.

Implementations MUST keep the archive structure stable and apply the correct action-specific binding. Do not normalize these field names in the spec or codebase; preserve the worker-native names so that config payloads match each action's runtime schema exactly. The Blueprint's `services.tao_job_service.apply_dataset_binding(action=…)` is the single source of truth for this dispatch.

#### 9.3.4 Export Validation Requirement

Before the dataset export integration is considered implementation-complete, the following MUST be validated against the pinned `cosmos_rl_container_tag`:

1. Generate one sample export artifact from this system.
2. Validate that `annotations.json` parses as the expected top-level JSON array.
3. Validate that every `images[*]` path in every sample resolves to an existing file in the extracted media payload.
4. Run one TAO/Cosmos-RL smoke test (e.g., a 1-epoch `train` action on a minimal dataset) against the pinned container version and confirm the job accepts the dataset without format errors.

Until this validation passes, the export contract is decisioned but not implementation-closed.

#### 9.3.5 Export Field Mode

**Export field mode** (`export_field_mode`):

Controls which field groups are included in the gpt turn JSON of training exports and Auto-Labeled output. Field ordering matches the production serving prompt: base relative order for other Aux/Core fields, then `rationale_note` last when included.

- `all` (default): gpt turn contains available `rationale_note` + other Aux fields + Core fields. This retains auxiliary/audit data for Student experiments; it does not make rationale authoritative or part of Teacher ICL.
- `aux_and_core`: gpt turn contains Aux fields (excluding `rationale_note`) + Core fields. The Student learns the observational scaffolding pattern without explicit reasoning text. Useful when Aux fields carry valuable intermediate signals but rationale is not needed at inference time.
- `core_only`: gpt turn contains Core fields only. The Student learns to produce labels directly without scaffolding. Smallest output format; lowest inference cost.

The mode is set per export and persisted on the DatasetExport record. The same `export_field_mode` MUST apply consistently across all examples within a single export artifact.

**Training consistency:** a Student training run MUST use a single `export_field_mode` across all included training datasets. If multiple datasets are combined (e.g., Verified + Auto-Labeled), they MUST share the same field mode, or be normalized to a single selected mode before training. Mixing `all`, `aux_and_core`, and `core_only` within one Student training run is not allowed — the model would learn an inconsistent output contract. The `export_field_mode` is stored on the DatasetExport records only (single source of truth, §13.5); the Student's Inference Contract (§6.11) derives its field mode from the referenced training DatasetExport records at query time.

TAO distinguishes dataset intents (training, evaluation, testing). The export layer MUST emit explicit dataset manifests per intent so that TAO dataset bindings can be wired correctly for each job action.

Verified (train) and Auto-Labeled MUST be exportable as separate artifacts. If combined, entries MUST carry `label_tier` metadata only if downstream tolerates it.

Exports MUST create DatasetExport record (Section 13.5) including:

- `dataset_export_id`
- `dataset_intent` ∈ {`training`, `evaluation`, `testing`} (required)
- `export_field_mode` ∈ {`all`, `aux_and_core`, `core_only`} (required; the field scope used for this export)
- artifact refs (tar + checksums)
- manifest ref
- linkage to `guidance_id`
- selection definition snapshot (filters, pool exclusions, Core-valid requirement for Auto-Labeled, etc.)

### 9.4 Multi-model Training Suites

System MUST support triggering Cosmos-RL / TAO VLM fine-tuning for multiple
configured Student base models under one training suite, executed sequentially
(one job at a time). The default **Validate training setup** intent MUST
pre-select exactly one recommended small base (prefer Cosmos Reason2 2B when
available), the Quick preset, and `FP8_DYNAMIC`. The explicit **Compare
production candidates** intent MAY pre-select multiple bases and schemes.
Seeded entries that are not provisioned remain visible and selectable with
**Provisioned in Training Jobs**:

- `nvidia/cosmos-reason2-8b`
- `nvidia/cosmos-reason2-2b`
- `nvidia/cosmos3-nano-reasoner`
- `nvidia/cosmos3-super-reasoner`

Cosmos 3 Super remains a backend catalog/API Student base and historical
Student records remain fully inspectable, but it is not selectable for a new
run in the Training UI. Its qualified path requires Full-weight training and
no post-training quantization. The Training UI exposes only the qualified LoRA
matrix—Cosmos Reason2 2B/8B and Cosmos 3 Nano—until Full-weight is qualified
across that supported UI matrix. Direct API callers remain subject to the
backend training-mode and quantization compatibility checks below.

At least one base must remain selected. The Scale-Up Hub's **Train a Student**
button is navigation into this configuration and MUST remain enabled. Both the
hub and Student Training screen call the backend preflight and render checking,
ready, data-not-ready, and infrastructure-required states. **Start Training**
repeats fail-closed TAO/workspace/timeout/data/role validation, including the
configured Test Pool minimum for the suite's held-out evaluation export. An ordinary
missing selected base is non-blocking: the server creates a provisional
TrainingSuite, navigates to Training Jobs, and provisions every selected
missing base together as one setup stage.

### 9.5 Student Registry and Deployment

Each training run records:

- `guidance_id`, pool version, training split definition, dataset export refs
- resulting Student identifiers registered in the Student registry; every suite-created Student persists the immutable parent `training_suite_id`
- `nim_vlm_release_version: string` (required on deployed Student record; pin NIM release for reproducibility)

TrainingSuite responses MUST expose the training/evaluation example counts,
the evaluation archive SHA-256 when available, and `student_model_ids[]`. These
fields let Models & Results group the project-wide registry and determine
whether two run groups have compatible evaluation evidence without reading
workspace manifests in the browser. Existing suite-created Students are
backfilled from their training TAOJob's `chain_id`; genuinely ad-hoc Students
remain nullable and render in an unassigned historical group.

Each Student variant MUST be deployable behind a NIM-compatible inference endpoint.

TAO runs and Student registrations MUST be represented as first-class records with lineage as specified in §13.12 (TAOJob) and §13.13 (StudentModel).

**Two distinct evaluation and deployment phases:**

- **Quality validation (TAO-backed, preferred):** TAO `evaluate` runs automatically after training and after each quantization (§9.7.6). Produces accuracy metrics. No NIM deployment required. The system re-scores TAO per-sample predictions with the canonical Core-field evaluator for authoritative metrics. When TAO eval succeeds, the system flips the paired Student's `quality_status` to `"validated"` and records `quality_evaluation_run_id` pointing at the TAO RunRecord.
- **Quality validation (NIM-backed fallback, narrow):** NIM-source evaluation MAY satisfy the quality gate **only** when one of the following holds, and never as a generic rescue for arbitrary TAO failures:
  - **Cold start / pending:** `quality_status="pending"` — no TAO eval has terminated yet (e.g., the project is operator-driven through NIM only, or the TAO chain has not finished). A successful NIM-source eval against the Test Pool promotes `quality_status` to `"validated"`.
  - **Visible cold-start path:** a packaged Student with
    `quality_status="pending"` remains visible on Compare with a pending-quality
    explanation and **Deploy and benchmark** action. The action dispatches the
    normal Student NIM lifecycle; hiding this record would make the permitted
    NIM-only quality path unreachable from the product UI. Once dispatched,
    the durable `serving_status="pending"` is authoritative across refresh and
    SSE loss: Compare MUST keep polling, show an in-progress state, disable
    project-wide benchmark actions, and offer no duplicate per-Student deploy.
    A Blueprint-local NIM preflight, startup, or benchmark failure before a
    quality result exists leaves `quality_status="pending"`; it is an
    operational serving failure, not a measured quality failure. On retry the
    backend also repairs the narrow legacy state where such a synthetic
    `student_nim_local` evaluate row had incorrectly changed quality to
    `"failed"`, preserving the pending-quality NIM path.
  - **TAO failure with a known upstream model-loader signature:** `quality_status="failed"` AND the prior failed TAO `evaluate` job's failure evidence (`error_ref`, `poll_error_ref`, `chain_halted_reason`, or `outputs.tao_logs_text`) matches a known upstream loader-gap pattern from `services.tao_failure_classifier.MODEL_LOADER_FAILURE_PATTERNS`. Two canonical references today: (1) the cosmos-rl 6.26.3 + Cosmos-Reason2 + Qwen3-VL-dense gap (documented 2026-05-04; evidence retained in the project's internal engineering archive), and (2) the cosmos-rl 6.26.3 + Cosmos-Reason2-8B vLLM weight-init gap (F43, 2026-05-13) — `ValueError: Following weights were not initialized from checkpoint:` listing `visual.blocks.*` + `language_model.model.layers.*` weight names that the cosmos-rl-bundled vLLM doesn't enumerate but NIM 1.6.0's vLLM loads cleanly. The full TAO-team-facing report is retained in the project's internal engineering archive. Other failure classes (dataset shape, OOM, transient infra, schema-validation crashes, etc.) leave `quality_status="failed"` — NIM eval is NOT a fallback for those.
  - **Implementation:** `services/student_nim_lifecycle._promote_quality_from_nim_eval` consults `services.tao_failure_classifier.matches_known_loader_gap` before flipping `quality_status`. The classifier follows the TAO job that produced the served artifact: the `train` job for a baseline Student and the `quantize` job for a quantized Student, because the latter's failed paired `evaluate` is parented by `quantize`. It inspects the complete bounded 64 KB TAO failure-log payload retained by the polling service; a smaller classifier window MUST NOT discard a signature that the persisted evidence still contains. Conservative gate: empty/None failure evidence → no promotion. The promotion writes `quality_evaluation_run_id = <NIM_eval_run_id>` and the corresponding RunRecord carries `evaluation_source="nim"` with the same canonical Core-field metrics (Appendix A.2) plus the latency / throughput / profile-metadata that NIM eval provides for free.
  - **Visible recovery path:** a packaged Student with `quality_status="failed"` remains visible on Compare—even when every Student in the project is quality-failed—and, until serving is pending or validated, exposes **Deploy for serving validation**. This dispatches the normal Student NIM lifecycle and renders its live stages. A reopened screen reconciles a persisted `serving_status="pending"` as in progress and MUST NOT offer a duplicate deploy action. The UI never decides whether the NIM result may recover quality; the backend applies the signature gate above. A non-matching TAO failure can become serving-validated but remains quality-failed and ineligible for production handoff.
  - **Failure-evidence capture (REQUIRED for the gate to be useful):** when a TAO `evaluate` reaches `failed`, the polling service MUST best-effort fetch the job's `:logs` body (TAO REST API Overview — `GET /api/v2/orgs/{org_name}/jobs/{job_id}:logs`) and persist the tail (≤ 64 KB) on `TAOJob.outputs.tao_logs_text`. Without this capture, the failure signature cannot be classified and the fallback degrades to "fail conservatively."
  - **Audit invariant:** when `quality_status` is already `"validated"` from a prior TAO eval, the NIM-source eval MUST NOT overwrite `quality_evaluation_run_id` — preserves the audit pointer back to the TAO RunRecord. Both run histories remain queryable.
- **Serving validation (NIM-backed):** NIM deployment enables the production-representative real-image latency, throughput, and reliability comparison plus profile metadata collection. The system attempts local NIM orchestration (§9.5.2); if that fails, an Action Request provides cache-disabled deployment details for external infrastructure. Every configured load cell must complete its exact request count with zero failures before `serving_status` flips to `validated`; `serving_evaluation_run_id` and all failed-cell evidence are still recorded independently of the quality path. On an upgraded workspace, a persisted pre-AIPerf `serving_status="validated"` remains historical rather than being rewritten. Student API responses derive `serving_benchmark_current=false` and a blocker from its referenced run, Models & Results offers **Revalidate with AIPerf**, and both production handoff paths fail closed until the current contract passes.

Quality validation establishes "did fine-tuning produce a better model?" Serving validation establishes "can this model serve in production, and at what cost?" The `deployment_handoff` Action Request (§10.3) requires `quality_status="validated"`, `serving_status="validated"`, AND `serving_benchmark_current=true`; either source for quality is acceptable.

**F-B7 amendment (2026-07-15) — Student invocations run at native image resolution.** Fine-tuned Students MUST be evaluated — and SHOULD be served — with the `native` Visual Budget preset (no `mm_processor_kwargs`): §9.3 training consumes images at native size, so any serve-time resize puts the Student off its training distribution. Inheriting the project's then-current Teacher-oriented `high_detail` resize override was measured live to collapse a healthy freiburg CR2-2B student from 0.95 EM (native, matching its FTMS evaluate at 0.90) to 0.367 on the identical 120-key holdout and checkpoint — the resize, not the prompt, was the entire dual-path disagreement after the F-W7 export fix. The Student NIM lifecycle's serving evaluation pins `visual_budget_preset_key="native"` (alongside the existing `icl_mode="disabled"` — together these are the training-parity contract); the corrected area-budget ladder remains for un-tuned Teacher perception. Deployment guidance: production serving of a Student should likewise send no image-resize processor kwargs.

**F14 (Phase 12 amendment 2026-05-05) — serving and quality gates have different strictness contracts.** The serving and quality validation paths run against the SAME NIM-source evaluation Run Record but apply DIFFERENT acceptance criteria:

- **Serving validation gate** accepts `run_record.status IN (completed, incomplete)`. The serving gate confirms the container is reachable, accepts requests, and produces parseable output for SOME invocations. Model accuracy is a separate concern. A run finalizing as `incomplete` per §13.2.1 (one or more examples failed schema-validation after the retry pass — typically because a fine-tune emits some examples with the wrong field shape) MUST NOT block serving validation: the container served, that's the question this gate answers.
- **Quality validation gate** (`_promote_quality_from_nim_eval`) accepts `run_record.status == "completed"` only. Conservative by design: model accuracy promotion via NIM-eval-fallback (the narrow F11 path) requires a clean run because quality is the load-bearing signal for the Scale-Up Readiness Gate (§7.3) and the deployment handoff's customer-visible accuracy metrics.

Concretely: a Student whose post-fine-tune model emits 47/84 schema-invalid examples can still reach `serving_status="validated"` but stays at `quality_status="failed"` (or the prior-set status) until either a clean TAO eval succeeds or a clean NIM-source eval finishes (`completed`, not `incomplete`) AND the F11 loader-gap signature matches. The `deployment_handoff`'s dual gate (`quality + serving`) prevents shipping a Student that serves but is too inaccurate.

Live-validated against the 2026-05-05 closing smoke: the 8B baseline NIM eval finalized `incomplete` (47/84 schema-invalid because the 3-epoch LoRA had not internalized the schema enough to consistently emit `gesture` at top level vs nested in a `gestures` object); F14 let `serving_status` flip to `validated` while `quality_status` remained `validated` from the prior TAO eval (held independently via §9.5 quality path).

**F35 (Phase 12 closeout amendment 2026-05-06) — partial quality_status for incomplete-but-mostly-parseable NIM evals.** F14's strict `completed`-only gate is correct for the production-handoff bar but loses the operator signal that a model "mostly works." When a NIM-source eval finishes `incomplete` with parseable rate ≥ `STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD` (default 0.90, operator-overridable in `config.yaml`), the system promotes the paired Student to `quality_status="partial"` instead of leaving it at the prior value. `partial` is **informational, not gate-passing**:

- The `deployment_handoff` Action Request (§10.3) **STILL requires** `quality_status="validated"`. A `partial` Student returns 409 with `conflict: quality_status_partial` (distinct from `conflict: quality_status_not_validated` so the UI can render different messages).
- The Scale-Up Readiness Gate (§7.3) reads from completed evaluations only; `partial` does not satisfy any gate criterion.
- F33 `:rerescore` rejects `partial` Students (§9.7.6) — the remediation path is to re-run NIM eval, not to replay a TAO rescore.

Audit invariants (mandatory):

- `validated → partial` is **never** a legal transition. The promotion helper is a no-op when prior `quality_status="validated"` (preserves the audit pointer to the validated TAO RunRecord).
- `partial → partial` is idempotent (no-op on re-call).
- Only `pending → partial` and `failed → partial` are valid promotion transitions. Both write `quality_evaluation_run_id = <NIM_eval_run_id>`.
- `partial → failed` and `partial → validated` are reachable via subsequent NIM evals: a fresh `completed` run promotes to `validated`; a fresh `incomplete` run with parseable rate < threshold leaves `partial` unchanged (idempotent).

Implementation: `services.student_nim_lifecycle._promote_quality_to_partial` mirrors the shape of `_promote_quality_from_nim_eval`. The lifecycle's post-eval gate computes parseable rate from `RunRecord.examples_succeeded / examples_total` and routes to `_promote_quality_from_nim_eval` (on `completed`) or `_promote_quality_to_partial` (on `incomplete + parseable ≥ threshold`) consistently in BOTH the local-mode and external-mode paths. As a side-effect, the external-mode path now also honors F14's lenient serving gate (it previously refused both serving and quality on `incomplete` runs) — F14 and F35 are unified at both call sites. Live evidence motivating F35: the 8B baseline `ab007102` from the 2026-05-05 full-stack validation run (evidence retained in the project's internal engineering archive) — 65% NIM exact-match on 135/148 (91% parseable); under F14 alone it was stuck at `failed`, under F35 it would land at `partial` and surface as a yellow badge on the Compare & Benchmark card.

TAO Inference Microservices (persistent model loading for repeated inference) are NOT part of this system's serving story. Production Student serving MUST use NIM.

#### 9.5.1 Checkpoint Handoff (TAO → NIM)

After a TAO training job succeeds (or after quantization via either lane in §9.8), a **packaging step** MUST verify the output is a NIM-loadable checkpoint before the Student can be registered:

- For Cosmos Reason2 (vLLM backend): `NIM_MODEL_NAME` must point to a directory containing a HuggingFace checkpoint or quantized checkpoint. Expected structure: HF model files at root (`config.json`, `generation_config.json`, safetensor shards, tokenizer files, `runtime_params.json`).
- System MUST validate checkpoint directory structure before registration succeeds.
- **LoRA merge-or-validate (full-precision baseline):** if LoRA was used (`enable_lora=true`), packaging MUST first check whether the training output is already a NIM-loadable HuggingFace checkpoint directory. If it is (full model shards present), no merge is performed. If the output is adapter-only (LoRA weights without merged base), packaging MUST merge the adapter into the base model using the persisted `base_model_path` (sourced from the training TAOJob's `resolved_training_fields.policy.model_name_or_path`), materialize a merged HuggingFace checkpoint directory, then validate the result. If `base_model_path` is unavailable or merge fails, packaging MUST fail with a clear error identifying the missing prerequisite. Note: `export_safetensors: true` controls serialization format only; it does not guarantee adapter merge.
- **LoRA merge for quantized variants:** TAO `quantize` auto-merges when `enable_lora=true` and `base_model_path` is set (§9.8). No separate merge step is needed for quantized variants.

**LoRA deployment policy:** NIM VLM does not document runtime LoRA adapter serving for Cosmos Reason2. The deployment input to NIM MUST be a merged HuggingFace checkpoint or quantized checkpoint. Do not specify runtime LoRA adapter serving for VLM NIM unless NVIDIA later documents it explicitly. (Runtime LoRA adapter loading is documented for LLM NIM only.)

Persist on Student registry record:

- `checkpoint_packaging_status` ∈ {`pending`, `validated`, `failed`}
- `nim_checkpoint_ref: string` (path/URI to NIM-loadable checkpoint directory)

**Portable production deployment bundle.** A Student that passes the
`deployment_handoff` quality, serving, checkpoint-packaging, and Inference
Contract gates exposes
`GET /v1/projects/{project_id}/student_models/{student_model_id}/deployment_bundle`.
The response streams an `application/x-tar` attachment without materializing
a second checkpoint copy. It contains:

- the complete NIM-loadable payload from the validated checkpoint under
  `checkpoint/`; TAO completion-only `status.json` and
  `microservices_log.txt` artifacts are excluded because they are not model
  inputs and may contain infrastructure context;
- `manifest.json` with the pinned NIM image/release/profile, GPU and inference
  contract, evaluation snapshot, and TAO/dataset lineage;
- `SHA256SUMS` covering every checkpoint and generated deployment file;
- an executable `run-nim.sh` that mounts the bundled checkpoint read-only,
  forwards `NGC_API_KEY` by name only, and uses the same GPU, shared-memory,
  user, model-size selector, and profile contract as local Student deployment;
- `request-template.json`, reconstructed from the exact Guidance version and
  effective controls of the successful serving evaluation, with one explicit
  image-data placeholder and the authoritative structured-output schema;
- an executable health + real-image inference verifier that injects the image,
  calls NIM, parses the returned label JSON, and fails on a non-structured
  response, plus a concise README.

The bundle MUST NOT contain the licensed NIM runtime image or any credential.
Its launch script pulls the exact pinned image from NGC using an
operator-provided key. The checkpoint must resolve beneath the requested
project directory and the exporter MUST reject symlinks, special files, path
escape, a missing pinned runtime image, or any non-validated handoff state.
The evaluation snapshot's Test Pool checksum MUST resolve from the held-out
DatasetExport referenced by the evaluate job paired with the artifact-producing
train or quantize job. `StudentModel.dataset_export_ids[]` remains training-data
lineage and MUST NOT be overloaded with the held-out export merely to populate
this checksum. Both `dataset_intent="evaluation"` (training-suite contract) and
the public `dataset_intent="testing"` variant identify held-out Test Pool
exports for this purpose.

#### 9.5.2 NIM Local Deployment Orchestration

The system attempts to deploy Student NIM containers locally on the backend host for evaluation and benchmarking. This is temporary infrastructure for the Compare & Deploy workflow, not a production deployment. A Student based on a multi-model shared image inherits `NIM_MODEL_SIZE` and validated non-secret `extra_container_env` from its base ModelConfig for preflight, actual launch, canonical handoff, and portable bundle. It does not inherit the base model's pinned `NIM_MODEL_PROFILE`, because that selector targets bundled base weights rather than the read-only custom checkpoint mounted through `NIM_MODEL_NAME`; NIM instead selects a checkpoint-compatible profile. Its `NIM_SERVED_MODEL_NAME` remains the Student-specific identity. Cosmos 3 Super pins `NIM_MAX_MODEL_LEN=65536`: its full-weight BF16 checkpoint used 62.63 GiB on the RTX PRO 6000 96 GB host, while the runtime's native 262,144-token setting required another 64 GiB of KV cache and could not start; the clamp fits the measured 20.47 GiB available cache and exceeds Blueprint prompt budgets.

**NIM deployment preflight:**

Before attempting local deployment, the system runs an automated preflight check. All checks MUST pass for local orchestration to proceed:

1. **Docker available:** `docker info` succeeds.
2. **NVIDIA Container Toolkit:** `docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi` succeeds, confirming GPU passthrough works.
3. **GPU memory sufficient:** the system reads GPU memory from `nvidia-smi` output and compares against the minimum for the target model/precision. BF16 thresholds: Cosmos Reason2 8B ≥56 GB, 2B ≥36 GB (the catalog `nim_gpu_memory_minimum_gb` values; the fit check is inclusive, `gpu_gb >= min_gb`). FP8 is validated on specific GPU models only (H100 80 GB, L40S 48 GB, H200, etc. per the VLM support matrix) — the system uses the model's `nim_gpu_memory_minimum_gb` from the catalog, not a hardcoded FP8 threshold. GPU compute capability requirements: ≥ 7.0 (generic), ≥ 8.0 (BF16), ≥ 8.9 (FP8).
4. **NGC API key valid:** `NGC_API_KEY` is configured (required to pull the NIM container image).
5. **NIM container image pullable:** the system verifies the pinned NIM VLM release image is accessible (e.g., `docker pull` or image already cached).
6. **Checkpoint exists and is validated:** `checkpoint_packaging_status="validated"` and the directory at `nim_checkpoint_ref` exists.

Preflight result MUST be persisted: `nim_preflight_status` ∈ {`passed`, `failed`}, `nim_preflight_details` (structured: per-check pass/fail with diagnostic messages), `nim_preflight_at` (timestamp).

**If preflight passes — local orchestration (Tier 1):**

The system manages the full NIM container lifecycle per variant:

0. **Acquire GPU (F49 amendment, 2026-05-19).** Resolve the target device per §1.5. If the device has any `LocalNimDeployment` rows in `starting | running`, the orchestrator MUST stop them in lifecycle order (`running` before `starting`) before constructing the Student `docker run`. Each displaced deployment's `displaced_by_deployment_id` (the Student's `local_nim_deployment_id`) and `displaced_at` are persisted on its row for audit (§13.15). The Student preflight (above) is re-run AFTER displacement when the resident's role was a NIM on the target device so the GPU-memory check reflects post-displacement free memory. On multi-GPU hosts where the auto-placer found a free device, step 0 is a no-op.
1. **Construct `docker run` command** from the StudentModel record: checkpoint path (`NIM_MODEL_NAME`), served model name (`NIM_SERVED_MODEL_NAME`), NIM cache path, NGC API key, port, `--shm-size=32GB`, `-u $(id -u)`, GPU allocation, and `NIM_ENABLE_KV_CACHE_REUSE=0`. Cache reuse is disabled for the serving comparison because the same frozen workload is replayed at each concurrency; later cache hits would not be comparable to the first cell.
2. **Start container** and begin polling `/v1/health/ready` (NIM startup includes runtime artifact build; allow up to `NIM_STARTUP_TIMEOUT_S`, default: 1200s).
3. **Smoke inference:** send a minimal `/v1/chat/completions` request to confirm the model responds.
4. **Register temporary endpoint** as a model config entry with the NIM-served URL and Student metadata.
5. **Run NIM evaluation** against the Test Pool using the standard evaluation pipeline (§7.1) with `evaluation_source="nim"`.
6. **Run the production VLM serving benchmark** at configured concurrencies (Appendix E). Build the workload from the frozen Test Pool and the evaluated Student's active Guidance-derived prompt/schema and Inference Contract: select every image when the pool has at most 200 members, otherwise select 200 without replacement by a stable hash rank. The rendered production prompt hash MUST equal the prompt hash recorded by the immediately preceding serving evaluation; absent or drifted prompt provenance fails the benchmark before load is dispatched. Each concurrency replays that identical ordered image set exactly once through pinned AIPerf `raw_payload` mode. Requests use the Student's no-ICL/native-visual-budget serving controls and omit both `max_tokens` and `max_completion_tokens`. Collect client p50/p90/p99 latency, achieved RPS, request counts/failure percentage, token statistics when available, and best-effort `/metrics` evidence. Persist `metrics.benchmark_workload` provenance plus every `metrics.benchmarks` cell, including failed cells. Missing metrics are `null`, never fabricated zeroes. `serving_status="validated"` requires every configured concurrency to complete the exact request count with zero transport/HTTP/timeout failures and finite latency/RPS; quality promotion from step 5 remains independent when the serving sweep fails.
7. **Stop container** after evaluation and benchmarks complete. The persistent NIM cache (`~/.cache/nim` or configured path, mounted to `/opt/nim/.cache`) is retained so subsequent startups skip the build step.
8. **Repeat** for the next variant (one container at a time; GPU resources are shared).
9. **Auto-restore displaced residents (F49 amendment, 2026-05-19).** After the Student container is stopped, iterate the deployments displaced by step 0 and best-effort re-deploy each using the original `role`, `model_config_id`, and `gpu_assignment`. Each restored deployment is health-polled up to `NIM_STARTUP_TIMEOUT_S`. Restoration failure surfaces as a warning on the StudentModel's `serving_evaluation_run_id` summary but does NOT fail `serving_status` — the benchmark itself succeeded and the operator can manually re-deploy via the NIM Configuration page. Auto-restore is per displaced deployment; an embedding NIM that was displaced by a Student deploy will also be restored.

Container lifecycle events (start, ready, evaluate, benchmark, stop, displaced, restored) MUST be logged and persisted on the StudentModel record.

**If preflight fails — Action Request fallback (Tier 2):**

The system generates a `student_nim_deploy` Action Request (§10.3) containing:

- The exact `docker run` command with all flags pre-filled
- Checkpoint path and expected directory structure
- GPU memory requirements and NIM release version
- Environment variable requirements (`NGC_API_KEY`, `NIM_MODEL_NAME`, `NIM_SERVED_MODEL_NAME`)
- Mount paths for checkpoint and NIM cache
- Health check and smoke test commands
- A note that this is temporary evaluation infrastructure

The SME copies the Action Request and sends it to their infrastructure team. Once the NIM endpoint is running with `NIM_ENABLE_KV_CACHE_REUSE=0`, the SME registers the endpoint URL and confirms `benchmark_kv_cache_reuse="disabled"`. The system then runs the same NIM evaluation and production benchmark against that endpoint; an external endpoint is not exempt from serving measurements.

**NIM configuration defaults (Cosmos Reason2 custom deployment):**

- NIM VLM release: pinned to the release supporting Cosmos Reason2 (currently `1.6.0`).
- Backend: vLLM (custom Cosmos Reason2 deployments use the vLLM backend).
- `--shm-size=32GB` (NVIDIA recommendation for custom VLM deployments).
- NIM cache: persistent host directory mounted to `/opt/nim/.cache`. Without this, the container rebuilds runtime artifacts on every startup.
- `NIM_CUSTOM_MODEL_NAME`: set to a stable identifier so the locally built engine is cached and reusable across restarts.
- Port: `8000` (default; configurable).

**Sequential variant benchmarking (user-scoped):**

The SME selects which variants to benchmark (all, selected, or individual; see Overview §7 step 4). Selected variants run sequentially, one container at a time. Each container is stopped before the next starts (GPU resources shared). The NIM cache persists across variants, but each variant's runtime artifacts are variant-specific. Results are shown progressively as each variant completes. Per-variant timeouts: startup bounded by `NIM_STARTUP_TIMEOUT_S` (default: 1200s), benchmark bounded by `NIM_BENCHMARK_TIMEOUT_S` (default: 1200s). On timeout or failure, the variant is marked (`timed_out` or `failed`), diagnostics preserved, and the queue continues with the next selected variant. No global workflow timeout in v1.

### 9.6 Student Inference Modes

Student inference MUST use the Inference Contract derived from training lineage (§6.11):

- `output_field_mode` and `icl_field_mode` both default to the `export_field_mode` from the training DatasetExport. A `core_only`-trained Student produces Core-only output.
- Students run without ICL: TAO evaluation does not support ICL injection, and v1.0 deploys, serves, and evaluates Students bare (`icl_mode="disabled"`). The contract's `icl_field_mode` is retained as training provenance (§6.11).
- Release gating MUST match the intended Inference Contract: the Student MUST be evaluated under the same contract it will use in production.

### 9.7 Cosmos-RL / TAO VLM Job Triggering and Tracking

This section defines the Cosmos-RL / TAO VLM integration contract: TAOJob state machine, record schema, configuration payload structure, polling contract, and required tracked outputs.

#### 9.7.1 TAO Execution Substrate

When enabled, system MUST treat Cosmos-RL / TAO VLM (via TAO) as a **remote external executor** and implement triggering + polling via the TAO REST API. TAO does not need to run on the same machine as this system; it runs on GPU-equipped infrastructure and is accessed via the deployment-level endpoint configuration defined in §1.6 (`TAO_API_BASE_URL`, `TAO_API_KEY`, `TAO_ORG_NAME`).

Preferred: TAO Fine-Tuning Microservices (FTMS) TAO API v2 using unified Jobs API (job-centric; supports pause/resume/cancel/delete; logs/files/outputs APIs; metadata may include progress/metrics depending on job type). The pinned versions for this Blueprint are `tao_release_version: "6.26.3"` and `cosmos_rl_container_tag: "6.26.3-cosmos-rl"`. These are the versions the system is built and tested against. Implementations targeting a different TAO release MUST update these pins and revalidate the dataset export format (§9.3.4).

All TAO API calls MUST use the configured `TAO_API_BASE_URL` and authenticate with `TAO_API_KEY` (see §1.6 for the application-level auth abstraction). API paths use `TAO_ORG_NAME` as the organization scope (e.g., `GET {TAO_API_BASE_URL}/orgs/{TAO_ORG_NAME}/jobs/{job_id}`).

Training backend identifier: `cosmos_rl_tao_vlm`. TAOJob records MUST persist `training_backend: "cosmos_rl_tao_vlm"`. Cosmos-RL exposes four task families via `tao-client cosmos-rl`: **train, evaluate, inference, quantize**. No launcher-based interface exists; all interaction is through the TAO Toolkit API and `tao-client`. VLM fine-tuning is currently available only through this API path.

#### 9.7.2 TAOJob State Machine

Canonical TAOJob statuses:

- `not_started`, `submitting`, `submitted`, `queued`, `running`, `paused`, `succeeded`, `failed`, `canceled`, `deleted`

- `not_started`: job record exists as part of a chain definition but has not been submitted to TAO. Used for downstream jobs in a chain that are waiting for their predecessor to succeed. This is a local state only — it has no TAO-side equivalent.
- `submitting`: the system is making the HTTP POST to TAO but does not yet have a confirmed `tao_external_job_id`. This is a local state only. On restart, a TAOJob in `submitting` with `tao_external_job_id = null` transitions to `failed` with `status_reason="submission_interrupted"` (§9.7.7).

Terminal statuses: `succeeded`, `failed`, `canceled`, `deleted`
Once terminal, MUST NOT transition to non-terminal.

Allowed transitions:

```text
not_started -> submitting | failed | canceled
submitting  -> submitted | failed
submitted   -> queued | running | failed | canceled | succeeded   ← F9 (skip running)
queued      -> running | failed | canceled | succeeded            ← F9 (skip running)
running     -> succeeded | failed | paused | canceled
paused      -> running | canceled | succeeded                     ← F9 (skip running)
(any)       -> deleted  (only if system performs delete)
```

`not_started` → `failed` occurs when `chain_halted_reason` is set (a predecessor failed). `not_started` → `canceled` occurs when the chain is canceled before this job starts. `submitting` → `failed` occurs when the TAO submission POST fails or when the backend restarts before `tao_external_job_id` is persisted.

**F9 (Phase 12 amendment 2026-05-05) — fast-completion races skip ``running``.** TAO can transition a job through ``Queued → Running → Done`` faster than the polling cadence (30s for `submitted`, 60s for `running`) can observe — a 3-epoch SFT on a small dataset can complete in ~7 min, well within a single ``submitted → submitted`` poll window. To prevent the local row from getting permanently stuck reading ``submitted`` (or ``queued`` / ``paused``) when TAO already returns ``Done``, ``submitted → succeeded``, ``queued → succeeded``, and ``paused → succeeded`` transitions are explicitly allowed. Observed reality wins. ``not_started`` and ``submitting`` are NOT included — those states predate any TAO ``tao_external_job_id`` round-trip, and ``status=succeeded`` without an `external_id` would be incoherent. Live-validated against the 2026-05-05 closing-smoke run (evidence retained in the project's internal engineering archive), where a 2B SFT chain completed on TAO in 7 min and the local poller missed every intermediate status. See `services/tao_job_service.can_transition`.

Mapping TAO raw status → canonical status:

- Persist `tao_status_raw` exactly as returned.
- Deterministically map case-insensitively using configurable table that MUST include at minimum:
  - `Done` → `succeeded`
  - `Failed` → `failed`
  - `Running` → `running`
  - `Queued` / `Pending` → `queued`
  - `Paused` → `paused`
  - `Canceled` / `Cancelled` → `canceled`

If `tao_status_raw` unknown:

- Set `status=running` only if metadata indicates active progress; else
- Set `status=queued` if job not started; else
- Set `status=running` as conservative non-terminal default.
Raw status remains queryable.

#### 9.7.3 TAOJob Configuration Payload Structure

A TAOJob MUST persist two configuration objects:

1. `job_config` - high-level config used by this system (human-meaningful).
2. `tao_create_job_request` - exact payload submitted to TAO (opaque JSON) for reproducibility/audit.

Minimum required `job_config` fields:

- `project_id`
- `training_backend: "cosmos_rl_tao_vlm"` (required; machine-readable backend identity)
- `student_base_model_config_id`
- `dataset_export_ids[]`
- `guidance_id`
- `training_preset` (string; required)
- `training_policy_type: "sft"` (required; default and currently only supported value, supervised fine-tuning; MUST NOT assume RLHF-like behavior)
- `lora_config` (required; see §9.7.3.2)
- `hyperparameters` (opaque TAO specs patch; may be `{}`)
- `dataset_refs` (opaque; may include TAO dataset ids and/or cloud URIs)
- `intended_outputs` (declares outputs system will collect; see 9.7.5)
- `tao_release_version: string` (required; pin TAO release for reproducibility)
- `cosmos_rl_container_tag: string` (required; pin container image tag)

Cosmos-RL resolved training fields (required subset to persist in `job_config.resolved_training_fields`):

- `policy.model_name_or_path`
- `policy.model_max_length` (truncates sequences beyond this value)
- `policy.model_gradient_checkpointing`
- `policy.parallelism.*` (TP, PP, DP settings)
- `train.ckpt.*` (checkpoint config)
- `train.train_policy.*` (training policy config)
- `validation.*` (if validation enabled during training)
- `results_dir`

Parallelism and distributed-training fields (required in `job_config`):

- `parallelism_config` (normalized; at minimum `tensor_parallel`, `pipeline_parallel`, `data_parallel`)
- `num_nodes: int` (default 1)
- `num_gpus_per_node: int`
- `redis_config: object | null` (for distributed coordination in multi-node setups; nullable for single-node)

Minimum required `tao_create_job_request` fields:

- `kind` (e.g., `experiment`)
- `action` (e.g., `train`)
- `specs` (full TAO specs JSON object)
- plus any required top-level job metadata fields used by chosen TAO workflow/action (e.g., `name`, `network_arch`, `workspace`, dataset bindings, `base_experiment_ids`, timeouts, etc.)

Hyperparameter resolution:

- Implementations MUST resolve `tao_create_job_request.specs` by starting from the TAO-provided default `specs` for the selected workflow/action and base experiment/model (e.g., `jobs:schema` default), then applying `job_config.hyperparameters` as a deterministic patch.
- `job_config.hyperparameters` contains only overrides from TAO defaults; `{}` means “use TAO defaults.”
- Implementations MAY expose a small set of presets in the UI; selecting a preset MUST deterministically populate `job_config.training_preset` and the resolved `job_config.hyperparameters`.

#### 9.7.3.1 Training preset system

This section defines an SME-friendly preset system that lets an SME pick “how hard to train” while the backend produces a deterministic `job_config.hyperparameters` patch that is applied to `tao_create_job_request.specs`.

##### Preset shape (what SME sees)

Single dropdown:

**Training Intensity**

- **Quick** (fast iteration; default for Validate training setup)
- **Standard** (default for Compare candidate variants)
- **High Quality** (more training)
- **Max Quality** (large-epoch train)

Optional “Advanced” accordion (for power users) can still show the resolved patch JSON read-only.

##### Preset contract (what the API stores)

Add two fields under `job_config`:

```json
"job_config": {
  "training_preset": "standard",
  "hyperparameters": { ...resolved patch... }
}
```

Rules:

- `training_preset` is **user-facing** (auditable, easy to reason about).
- `hyperparameters` remains the **actual patch** (reproducible, deterministic, works with TAO schema).

##### Preset definitions (Cosmos-RL / TAO VLM)

Presets MUST resolve to a deterministic patch that targets Cosmos-RL training spec fields:

- `train.epoch`
- `train.resume`
- `train.ckpt.enable_checkpoint`
- `train.ckpt.save_freq_in_epoch`
- `train.ckpt.max_keep`
- `train.ckpt.export_safetensors`

Epoch counts below are calibrated against NVIDIA's official Cosmos-Reason2 examples (default config: 1 epoch; Cookbook 8B recipe: 3 epochs). Per-epoch checkpointing with `best_model` selection means higher epoch counts explore longer without risk of shipping an overfit checkpoint. Each preset is 3x the previous (1, 3, 9, 18). Adjust per project.

**Small-dataset guidance.** The epoch counts above are calibrated for
Cookbook-scale data, not the bundled sample. Quick is the deliberate
first-run wiring check. Usable training data below 150 MUST be called out as
exceptionally small, and no preset choice may represent a successful tiny-data
run as production-quality evidence. A meaningful fine-tune requires
domain-specific data-volume, balance, and evaluation judgment; choosing a
larger preset does not make an undersized dataset sufficient.

**F27 (Phase 12 amendment 2026-05-05) — `max_keep` is conditional on `resume`.** All presets share `train.ckpt.{enable_checkpoint=true, save_freq_in_epoch=1, export_safetensors=true}` and `train.resume=false` for first-time runs. The retention bound `train.ckpt.max_keep` is conditional on the resume flag:

- `resume=false` (current default for every Blueprint flow): `max_keep=1`. The Blueprint's checkpoint selector (`_select_hf_checkpoint_keys`, §9.7.5) only ever pulls down the latest epoch directory; retaining additional epochs uploads ~Nx more bytes to workspace S3 with no Blueprint-visible benefit. Live-surfaced 2026-05-05 on a 8B `high_quality` 9-epoch train: pre-fix `max_keep=8` produced a 117.8 GB safetensors upload (~50 min wall-clock) while the rest of the chain blocked on TAO's single-cluster gate. Post-fix the same run uploads only the latest epoch (~14.7 GB, ~6 min).
- `resume=true` (future feature for resuming interrupted runs): `max_keep=8`. Multiple historical checkpoints are needed for cosmos-rl's `train.resume` machinery to find a valid restart point if the latest is corrupted.

Until a resume-supporting flow lands in Blueprint, every training invocation hits the `max_keep=1` branch.

**1) Quick**

- Intent: sanity check. Is the pipeline working? Is loss decreasing?

Patch:

```json
{
  “train”: {
    “epoch”: 1,
    “resume”: false,
    “ckpt”: { “enable_checkpoint”: true, “save_freq_in_epoch”: 1, “max_keep”: 1, “export_safetensors”: true }
  }
}
```

**2) Standard (production-candidate comparison default)**

- Intent: recommended baseline (matches NVIDIA Cookbook 8B recipe)

Patch:

```json
{
  “train”: {
    “epoch”: 3,
    “resume”: false,
    “ckpt”: { “enable_checkpoint”: true, “save_freq_in_epoch”: 1, “max_keep”: 1, “export_safetensors”: true }
  }
}
```

**3) High Quality**

- Intent: extended training for stronger convergence

Patch:

```json
{
  “train”: {
    “epoch”: 9,
    “resume”: false,
    “ckpt”: { “enable_checkpoint”: true, “save_freq_in_epoch”: 1, “max_keep”: 1, “export_safetensors”: true }
  }
}
```

**4) Max Quality**

- Intent: thorough search for best quality (slowest); `best_model` checkpoint selection protects against overfitting

Patch:

```json
{
  “train”: {
    “epoch”: 18,
    “resume”: false,
    “ckpt”: { “enable_checkpoint”: true, “save_freq_in_epoch”: 1, “max_keep”: 1, “export_safetensors”: true }
  }
}
```

##### Resume semantics

If `train.resume` is set `true`, TAO resumes from the latest checkpoint in `output_dir` and restores model weights, optimizer state, and training progress. This MUST be treated as a continuation, not a “new run with changed knobs.”

##### Optional: model-aware epochs (internal policy; not NVIDIA-specified)

Epoch counts MAY vary by base model size (bigger model = fewer epochs to avoid overfit). Example mapping (still only 4 presets shown to SME):

- Quick: 2B=1, 8B=1
- Standard: 2B=3, 8B=3
- High Quality: 2B=12, 8B=9
- Max Quality: 2B=24, 8B=18

Implementation: preset resolves to epoch based on `student_base_model_config_id` (deterministic lookup table).

##### Deterministic resolution flow (backend)

1. Start from TAO default `specs` for this workflow/model.
2. Resolve `training_preset` → preset patch (and optional model-aware mapping).
3. Set:
  - `job_config.training_preset = <value>`
  - `job_config.hyperparameters = <resolved patch>`
4. Apply patch → `tao_create_job_request.specs`.

If user provides `job_config.hyperparameters` explicitly, you can either:

- treat it as “custom” and set `training_preset="custom"`, **or**
- disallow custom in the SME UI and reserve it for admin/advanced users.

##### Example TAOJob create body (SME path)

See §10.2.7 for the full normative create request body. The following shows only the `job_config` portion demonstrating a “Max Quality” preset resolution:

```json
“job_config”: {
  “training_preset”: “max_quality”,
  “hyperparameters”: {
    “train”: {
      “epoch”: 18,
      “resume”: false,
      “ckpt”: { “enable_checkpoint”: true, “save_freq_in_epoch”: 1, “max_keep”: 1, “export_safetensors”: true }
    }
  },
  “...”: “other fields omitted”
}
```

##### UI copy (SME-friendly, low-cognitive-load)

- **Quick**: “Fast run for a quick signal.”
- **Standard**: “Recommended default.”
- **High Quality**: “More training for better accuracy.”
- **Max Quality**: “Large-epoch training for best quality (slowest).”

Example `tao_create_job_request` (illustrative; see §10.2.7 for the full create body including `job_config`):

```json
{
  "kind": "experiment",
  "name": "vlm-fb-train-<project_id_8>-<chain_id_8>-01",
  "network_arch": "cosmos-rl",
  "workspace": "<workspace_uuid>",
  "action": "train",
  "base_experiment_ids": ["<cosmos_reason2_2b_uuid>"],
  "timeout_minutes": 1440,
  "specs": {
    "train": {
      "epoch": 3,
      "resume": false,
      "ckpt": {
        "enable_checkpoint": true,
        "save_freq_in_epoch": 1,
        "max_keep": 1,
        "export_safetensors": true,
        "save_mode": "sync"
      },
      "compile": false,
      "train_policy": {"type": "sft", "dataloader_drop_last": false}
    },
    "policy": {"model_name_or_path": "hf_model://nvidia/Cosmos-Reason2-2B"},
    "validation": {"enable": false},
    "custom": {
      "train_dataset": {
        "media_path": "<workspace_url>/dataset_exports/<id>/",
        "annotation_path": "<workspace_url>/dataset_exports/<id>/<id>_annotations.json"
      }
    }
  },
  "docker_env_vars": {"HF_TOKEN": "<gated-repo-token>"}
}
```

**Required top-level fields (live-verified against FTMS 6.26.3, 2026-04-29):**

- `kind` — always `"experiment"` for cosmos-rl jobs.
- `action` — one of `train`, `evaluate`, `quantize`.
- `name` — human-readable, persisted on the FTMS job; the Blueprint emits `vlm-fb-<action>-<project_id_8>-<chain_id_8>-<chain_seq_2>` for traceability.
- `network_arch` — `"cosmos-rl"` for Cosmos Reason2 SFT chains.
- `workspace` — the workspace UUID hosting both the dataset export S3 objects and the produced artifacts.
- `base_experiment_ids` — list with one entry: the registered Cosmos Reason2 base experiment UUID returned by `:list_base_experiments` (or `:load_airgapped`).
- `timeout_minutes` — required stale-heartbeat ceiling on every suite job. Default `1440`; operator-configurable via `TAO_JOB_TIMEOUT_MINUTES`. This is a dead-job reaper, not an ETA. Training preflight MUST fail closed unless the TAO v2 `ExperimentJobReq` OpenAPI schema declares this field; apply the v8 install guide §13.15 patch on FTMS 6.26.3.
- `docker_env_vars.HF_TOKEN` — gated-repo authentication token. cosmos-rl's `transformers.from_pretrained()` call requires HF auth even when the base experiment is registered via airgapped load. See §9.7.3.4.

**Top-level fields that MUST be omitted:**

- `encryption_key`, `train_datasets`, `eval_dataset` — legacy TAO 5.x fields not consumed by the cosmos-rl action handler. Dataset binding is action-specific (§9.3.3).

**Spec-level fields that MUST be omitted:**

- `policy.parallelism` — TAO's docker_handler auto-injects `tp_size=1, dp_shard_size=NUM_GPU_PER_NODE`; specifying explicit values caps the job to fewer GPUs than the rental provides. The `parallelism_config` value in `job_config` is for audit/reproducibility only; it MUST NOT be propagated into `tao_create_job_request.specs.policy.parallelism`.
- `num_gpu` (top-level) and `specs.train.num_gpus` — same rationale; the docker_handler default `num_gpu=-1` allocates the full visible GPU pool, which is what the chain submission wants. `num_gpus_per_node` in `job_config` is audit metadata only.

#### 9.7.3.2 LoRA Configuration

Cosmos-RL supports LoRA-based fine-tuning and requires LoRA configuration to be persisted when used. Every TAOJob MUST persist a `lora_config` object in `job_config` with at minimum:

- `enable_lora: boolean` (required)
- `lora_rank: int`
- `lora_alpha: int`
- `lora_dropout: float`
- `lora_target_modules: string[]` (e.g., attention projection modules)
- `modules_to_save: string[] | null` (e.g., `["visual"]` for full vision-side fine-tuning)

**F-B5 amendment (2026-07-14) — the persisted `lora_config` MUST reach the cosmos-rl wire.** Live evidence from the July 2026 live-validation window: every training job persisted `lora_config` (enable_lora=true, r16) while the cosmos-rl spec never carried a LoRA block — `policy.lora` defaulted to `None`, the trainer's parameter table logged every module TRAINABLE, and an 8B "LoRA" train OOM'd an A100-80GB on full-model optimizer state. Trainings were silently full fine-tunes while records (and this section) claimed LoRA. The contract is now: when `enable_lora=true`, the train action's spec MUST emit `specs.policy.lora` mapped onto cosmos-rl's `LoraConfig` field names — `lora_rank`→`r`, `lora_alpha`→`lora_alpha`, `lora_dropout`→`lora_dropout`, `lora_target_modules`→`target_modules`, `modules_to_save`→`modules_to_save` (nested `policy.*` dicts pass through TAO's spec mapper; `policy.parallelism` uses the same path). When `enable_lora=false` (the training-suite request's explicit opt-out, `POST …/training_suites` body field `enable_lora`), no `policy.lora` key is emitted — the legacy full-weight wire shape — and every chain job's persisted `lora_config` records the opt-out. Canonical implementation: `services.training_suite_service._build_train_payload`.

**F-B12 amendment (2026-08-04) — training-mode compatibility MUST fail before TAO submission.** Live qualification of Cosmos 3 Super against `6.26.3-cosmos-rl` proved that the model's mandatory TP=8 path cannot partition the LoRA-wrapped `q_proj`/`v_proj` modules: PyTorch rejects the wrapper's dotted parameter names before the first training step. `POST …/training_preflight` and the suite's repeated server-side validation MUST therefore return a failed per-model `training_mode_compatible` check when Cosmos 3 Super is selected with `enable_lora=true` on that runtime. This restriction is version-scoped; a later Cosmos-RL runtime may restore LoRA only after independent qualification. Other compatible bases remain LoRA-first.

**F-B14 amendment (2026-08-05) — the Training UI exposes only the qualified LoRA matrix.** Full-weight has not been qualified across the Student bases offered by the Training UI. The UI MUST therefore keep `enable_lora=true`, MUST NOT render the Full-weight selector, and MUST filter Cosmos 3 Super from its base-model choices. The backend `enable_lora=false` contract and Super compatibility checks remain intact for API compatibility, historical evidence, and future qualification. Restoring either UI choice requires an explicit live train→quantize(optional)→package→NIM qualification for the intended model matrix.

Implementation notes:

- `modules_to_save: ["visual"]` fully fine-tunes the vision encoder. There are compatibility constraints between some attention target modules and saving `visual`; implementations validate the combination before job submission.
- If LoRA is used and quantization is requested post-training, the adapter MUST be merged before quantization (TAO requires `base_model_path` when `enable_lora=true` during the `quantize` action; see §9.8).
- The chosen LoRA config MUST be persisted as part of student lineage for reproducibility.
- **NIM serving contract:** VLM NIM serves merged checkpoints only; runtime LoRA adapter loading is documented for LLM NIM but NOT for VLM NIM. Ensure adapter merge before any NIM deployment path (§9.5.1).

#### 9.7.3.3 Cosmos-RL SFT Spec Overrides (Required for train + evaluate; F11 amendment for quantize)

Every cosmos-rl `train` and `evaluate` action MUST emit a fixed set of spec overrides on top of the preset's hyperparameter patch. These are not preset-tunable; they are mandatory regardless of preset and base model. Each override prevents a specific failure mode that the cosmos-rl 6.26.3 worker exhibits with its stock defaults (live-verified 2026-04-28/29 against the rental's `/opt/cosmos_rl/tao_sft_example.py`):

- **`train.compile = false`** — `torch.compile` is unsupported for `HFModel` in cosmos-rl 6.26.3. Leaving the default true crashes the worker with an `AssertionError` before training starts.
- **`train.train_policy.type = "sft"`** — cosmos-rl defaults to `grpo` (Group Relative Policy Optimization); the SFT helper (`cosmos-rl-train`) exits silently if this key is missing.
- **`train.train_policy.dataloader_drop_last = false`** — with the default `true`, small datasets yield zero full batches and the trainer treats step 0 as `is_last_step`, which crashes checkpoint saves on `NoneType.state_dict` because the optimizer/scheduler aren't built yet. False keeps every example in training.
- **`train.ckpt.save_mode = "sync"`** — async checkpoint save races with the step-0 path on small datasets; sync mode serializes the writer.
- **`validation.enable = false`** — cosmos-rl asserts on a missing `val_dataset` when validation is enabled. The Blueprint's chain emits a separate evaluate job (§9.7.6) for validation; in-trainer validation is redundant.

**F11 (Phase 12 amendment 2026-05-05): the `quantize` action MUST NOT emit these SFT overrides.** Live-verified that cosmos-rl-quantize's argparse CLI rejects each of `--compile`, `--type`, `--dataloader_drop_last`, `--save_mode`, `--enable` as `unrecognized arguments`. The earlier guidance grouped quantize with train + evaluate; that was incorrect. Quantize's spec is minimal: `specs.quantization_scheme` (the F11-renamed key — see below) plus the action-specific dataset binding (§9.3.3) plus the F11-required omission of `policy.model_name_or_path` (cosmos-rl-quantize uses `--model_path` injected from `parent_tao_job_id`, not the base model URL).

**F11 spec-key rename (`quantization_method` → `quantization_scheme` in `tao_create_job_request.specs`)**: cosmos-rl-quantize's CLI flag is `--quantization_scheme`. The earlier `specs.quantization_method` was rejected by the argparse layer with `unrecognized arguments: --quantization_method`. Implementations MUST emit `specs.quantization_scheme = <FP8_DYNAMIC|W8A8|W8A16|W4A16>`. The persisted `job_config.quantization_method` field is unchanged — that is the user-facing concept and aligns with `StudentModel.quantization_method`; only the wire-level spec-key name changes.

**F11 — known TAO-FTMS-side residual gap (operator-side fix, not a Blueprint or upstream cosmos-rl issue):** the cosmos-rl-quantize CLI is single-process and accepts no parallelism arguments. Its argparse signature is `--model_path / --annotation_path / --media_dir / --quantization_scheme / --enable_lora / --base_model_path / --kv_precision / --smoothing_strength / --skip_test_generation / --num_calibration_samples / --max_sequence_length / --results_dir / --dataset_id / --dataset_split` — no `--tp_size` / `--cp_size` / `--n_init_replicas` / `--dp_shard_size` / `--dp_replicate_size` / `--pp_size`. Parallelism injection is a deployment-time custom patch (`patches/cosmosrl_parallelism_defaults.py`, applied to TAO-FTMS to address Issue #8 — giving `train` explicit parallelism defaults without operator configuration). The patch is not action-aware: it injects parallelism defaults for every cosmos-rl action including quantize and evaluate, and cosmos-rl-quantize argparse rejects the resulting CLI with `unrecognized arguments`. With the Blueprint-side F11 fixes above, the spec sent to TAO is correct (live-verified 2026-05-05: pre-fix had ~13 rejected args, post-fix had exactly the 6 from the parallelism patch). The remediation is operator-side: patch `cosmosrl_parallelism_defaults.py` to be action-aware (inject only for `train`). Documented in the 2026-05-05 full-stack validation findings (finding F11), retained in the project's internal engineering archive.

Implementations MUST apply the train+evaluate overrides via a single helper after the preset patch is computed and BEFORE the request is checksummed. See `services.training_suite_service._build_train_payload` for the canonical implementation. The quantize path is handled by `_build_quantize_payload` (no SFT overrides emitted).

#### 9.7.3.4 Environment Variables (`docker_env_vars`)

The Cosmos Reason2 family is gated on Hugging Face. Even when the base experiment is registered into the workspace via `:load_airgapped`, the cosmos-rl worker container's `transformers.from_pretrained()` call still authenticates against `huggingface.co` to verify gated-repo access — without an `HF_TOKEN` it gets HTTP 401 and the job retry-loops on "config.json not found" until the polling deadline fires.

Implementations MUST inject the operator's `HF_TOKEN` into the TAO request body's `docker_env_vars` field on every chain submission. Persistence:

- The Blueprint persists the operator's token as the `HF_TOKEN` setting (`~/.vlm_feedback_loop/.env`).
- `services.tao_job_service._submit_to_tao` reads the setting and injects `docker_env_vars.HF_TOKEN` on every TAO POST when the value is non-empty. The injection is uniform across train, evaluate, and quantize.

**Whitelisted env-var name:** TAO FTMS 6.26.3 only accepts `HF_TOKEN` in `docker_env_vars`. The aliases `HF_HOME`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`, `HF_HUB_TOKEN`, and `HF_ACCESS_TOKEN` are rejected as `Invalid enum member`.

The injected payload is logged at INFO level with `HF_TOKEN` redacted to `***` (see `tao_job_service._submit_to_tao` diagnostic log) so operators can verify the FTMS-required top-level fields without leaking the secret.

#### 9.7.4 Polling Contract

Polling responsibilities:

- Backend MUST poll TAO for TAOJob records in non-terminal statuses.
- MUST update:
  - `status` (canonical)
  - `tao_status_raw`
  - `progress` (if present)
  - `started_at` / `completed_at` when known
  - `last_polled_at`
  - `error` fields (if present)
  - `outputs` references as available (especially at terminal)

Polling API used (TAO API v2; normative; live-verified against FTMS 6.26.3, 2026-04-29):

- Job metadata/status: `GET /api/v2/orgs/{org_name}/jobs/{job_id}`
- Logs: `GET /api/v2/orgs/{org_name}/jobs/{job_id}:logs`
- List files: `GET /api/v2/orgs/{org_name}/jobs/{job_id}:list_files` (returns a JSON array of workspace-relative keys)
- Artifact retrieval: see §9.7.5. The Blueprint reads artifact bytes directly from the workspace S3 bucket (boto3 `GetObject`) using the keys enumerated by `:list_files`; FTMS 6.26.3's `:download_selective_files` endpoint is unsuitable for cosmos-rl outputs (POST returns 405 Method Not Allowed; GET requires `best_model`/`latest_model` aliases that cosmos-rl does not produce).

Polling frequency (recommended; deterministic policy):

- `submitted/queued`: every 30–60s
- `running/paused`: every 60–180s
- terminal: poll once to finalize outputs then stop
Implementation-defined exact timings MUST be deterministic and rate-limited per job.

On-demand refresh:

- `GET /v1/projects/{project_id}/tao_jobs/{tao_job_id}` accepts `refresh=true` to force poll-update before responding (rate-limited).

#### 9.7.5 Required Outputs to Track

TAOJob MUST track, by reference:

1. **Artifacts (required).** Cosmos-RL writes its outputs directly into the workspace S3 bucket; the Blueprint reads them from there (live-verified 2026-04-29 against TAO FTMS 6.26.3). The artifact layout depends on the action:

   - **`train`** — the merged Hugging Face checkpoint lives under `results/<tao_external_job_id>/<timestamp>/safetensors/epoch_<N>/`. Files include `config.json`, one or more `*.safetensors` shards (or `model.safetensors.index.json` + sharded `*.safetensors` files for larger models), and tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, etc.). The Blueprint selects the **latest epoch directory** (largest `epoch_<N>`) as the canonical merged-HF slot.
   - **`quantize`** — **F31 amendment 2026-05-05**: cosmos-rl-quantize emits the merged HF checkpoint **flat at the job root** under `results/<tao_external_job_id>/`, NOT nested under `<timestamp>/safetensors/epoch_<N>/`. Files include `config.json`, one or more `model-XXXXX-of-NNNNN.safetensors` shards, `model.safetensors.index.json`, tokenizer files, `recipe.yaml` (the quantization recipe used), and `status.json`. cosmos-rl-quantize is single-shot (no epochs) so it skips the per-epoch directory tree. Live-surfaced on a 8B FP8_DYNAMIC quantize (`72db4400-...`): all 4 safetensors files were present at the job root from the moment TAO marked Done, but the original `_select_hf_checkpoint_keys` glob only matched the train shape (`safetensors/epoch_<N>/`), silently returning `([], None)` and triggering a full F25 retry-budget exhaustion thinking the upload was racing — the artifacts were always there.
   - **`evaluate`** — a single tarball at `results/<tao_external_job_id>/evaluate_results.tar.gz` carrying per-sample predictions + aggregate metrics. Extracted layout under the cache dir is described in §9.7.5.1.

   **Retrieval mechanism (TAO API v2; live-verified):**
   - **Enumerate** workspace keys via `GET /api/v2/orgs/{org_name}/jobs/{job_id}:list_files` (returns a JSON array of relative keys).
   - **Read** each artifact directly from the workspace S3 bucket via boto3 `GetObject` (using the workspace's `tao_workspace_s3_endpoint_url_external` + access/secret keys; SeaweedFS- or AWS-compatible per the workspace's `cloud_type`).
   - The selected artifacts are mirrored to the project's local artifact store at `{project_dir}/artifacts/tao_jobs/{tao_job_id}/`. For train/quantize the contents are flattened into a NIM-loadable HF directory shape (config.json + shards + tokenizer at the root) so `_package_checkpoint` (§9.5.1) can validate without a LoRA-merge step.

   **Why workspace-S3 not `:download_selective_files`:** FTMS 6.26.3's `POST .../jobs/{job_id}:download_selective_files` returns `405 Method Not Allowed`. The endpoint's `GET` variant exists but expects `best_model` / `latest_model` aliases that cosmos-rl jobs do not produce — `GET ?best_model=true` returns 500 because no such alias is registered for cosmos-rl outputs. Direct workspace-S3 reads bypass the aliasing entirely and are the durable retrieval path.

   **F25 (Phase 12 amendment 2026-05-05) — TAO ``Done``-before-upload race retry.** TAO FTMS 6.26.3 can mark a cosmos-rl job ``Done`` before SeaweedFS finalizes the upload of that job's safetensors / `evaluate_results.tar.gz`. The first ``:list_files`` after the status-flip can return a workspace tree that has not yet grown the action-appropriate artifact slot, even though the upload finalizes seconds later. Live-surfaced 2026-05-05 on a 2B FP8_DYNAMIC quantize: post-success fired 7 seconds after `Done`, `:list_files` returned 0 matching `safetensors/epoch_<N>/` keys, and the StudentModel landed `checkpoint_packaging_status="failed"` even though the upload finalized minutes later. ``services/tao_polling_service._fetch_tao_artifacts`` MUST retry the listing-and-action-aware-selection step with bounded backoff when (a) ``:list_files`` succeeds but the action-aware selector returns zero matching artifacts AND (b) the caller passed a non-null ``local_cache_dir`` (caller wants real bytes, not a metadata snapshot). Metadata-only callers (``local_cache_dir is None``) MUST NOT retry — they explicitly want a snapshot of the current workspace tree, including empty. After the retry budget is exhausted the existing `"no merged-HF checkpoint slot"` / `"no evaluate_results.tar.gz"` error MUST surface so callers can fall through to the packaging=failed branch.

   **F28 (Phase 12 amendment 2026-05-05) — widened F25 backoff budget.** The original F25 backoff schedule `[10, 20, 40, 60, 90]`s (220s total) was consistently insufficient for fresh quantize uploads in live runs: an OLD W8A16 cleanup (ce4d95da) AND a NEW FP8_DYNAMIC quantize (8a379810) both exhausted all 5 retries with empty listings, leaving the paired StudentModel `checkpoint_packaging_status="failed"` despite the SeaweedFS upload eventually finalizing within the next ~5–10 minutes. The widened schedule is `[10, 20, 40, 60, 90, 150, 240, 360]`s (~970s ≈ 16 min total) — enough headroom for any observed quantize-upload race window without making genuinely-failed uploads burn unbounded background-task time. The schedule is a private constant in ``_fetch_tao_artifacts``; future widenings should publish it as a deployment-tunable setting if operators report longer race windows on slower TAO storage backends.

   **F32 (Phase 12 amendment 2026-05-05) — narrowed F25 retry to genuinely-empty listings.** The F25 retry trigger was originally formulated as "the action-aware selector returns zero matching artifacts," which fired the retry loop on TWO distinct listing shapes: (a) genuinely-empty listings (`len(keys) == 0` — the SeaweedFS finalization race signature), and (b) non-empty listings whose keys are completion metadata only (`status.json`, `microservices_log.txt`, …) without any checkpoint shards. SeaweedFS finalization is per-object: once a listing returns ANY key, finalization is past the race window for that object set — retrying will not conjure shards that cosmos-rl never produced. Shape (b) is a TERMINAL state where cosmos-rl finished and emitted only completion metadata; retrying burns the full ~970s F28 budget for no benefit and, in unit-test contexts, hangs the test runner. The retry trigger MUST narrow to shape (a) only: ``_fetch_tao_artifacts`` retries when ``len(listing["keys"]) == 0`` AND the caller passed a non-null ``local_cache_dir``. A non-empty listing with no matching artifacts MUST fall through immediately to the existing `"no merged-HF checkpoint slot"` / `"no evaluate_results.tar.gz"` error response. Live cases F28 was widened for (ce4d95da, 8a379810) were both genuinely-empty listings, so this narrowing preserves F28's intent on the real-world race signature while eliminating the up-to-970s hang on terminal non-checkpoint listings (live-surfaced as a unit-test hang in `test_no_checkpoint_dir_in_listing_returns_failure`).

   Additionally persist:
   - Exact training configuration used (at minimum `tao_create_job_request` plus resolved specs).
   - The TAOJob record MAY persist `tao_file_path` references for individual artifacts for traceability back to the workspace.

2. **Logs (required).** Track at least one durable log reference, fetched via `GET .../jobs/{job_id}:logs`:
   - `logs_ref` (preferred; stored in system), or
   - `tao_logs_ref` (snapshot of logs endpoint output)

3. **Metrics / Progress (required).** Track metrics/progress if available, at minimum:
   - `epoch_current` (nullable)
   - `epoch_total` (nullable)
   - `eta_seconds` (nullable)
   - `metrics_latest` (nullable object; e.g., loss/accuracy)
   - `metrics_history_ref`

#### 9.7.5.1 Evaluate Results Translation (Required)

For `evaluate` jobs, the Blueprint downloads `evaluate_results.tar.gz` from the workspace S3 bucket and extracts it (using `tarfile.extractall(filter='data')` for safe extraction) under `{project_dir}/artifacts/tao_jobs/{tao_job_id}/`. Cosmos-RL's evaluate output layout inside the tarball is:

- `<prefix>/freeform/<eval_set_name>/images/<image_basename>.json` — per-sample prediction file. The `<prefix>` depends on whether the evaluated checkpoint is a fresh training output or a post-training quantized variant:
  - **Train-checkpoint evaluate**: `<prefix>` is `epoch_<N>` — the chosen training epoch (typically the latest).
  - **Quantize-checkpoint evaluate** (F29 amendment 2026-05-05): `<prefix>` is the parent quantize TAO job's `tao_external_job_id`. cosmos-rl-evaluate uses `parent_job_id`'s server-side resolution to find the quantized checkpoint, and the resulting per-sample tree carries that parent's external id rather than `epoch_<N>`. Live-surfaced on a 2B FP8_DYNAMIC eval whose parent quantize ext was `8a379810-...`: per-sample JSONs landed at `8a379810-.../freeform/eval_set_name/images/<key>.json`.

  Each per-sample JSON file is a list of dicts; each dict carries `video_id` (the path-style image reference, e.g. `images/smoke_1.jpg`), `correct_answer` (the ground-truth JSON string), `answer` (the model's free-form response — i.e. the prediction), `reasoning`, and `full_response`.
- `<prefix>/freeform/<metric_name>_score.json` — aggregate metrics (per-class accuracy / confusion matrix).
- `status.json`, `microservices_log.txt` — log/state artifacts.

The rescoring service (§9.7.6) consumes a single canonical predictions file at `{cache_dir}/per_sample_predictions` containing a JSON list of `{id, prediction, ...}` records. The Blueprint synthesizes that file from the cosmos-rl per-sample tree:

- For each `<prefix>/freeform/<eval_set_name>/images/<key>.json` in the extracted tree (matching BOTH the `epoch_<N>` train-checkpoint shape AND the `<parent_external_id>` quantize-checkpoint shape per F29), parse the list-of-dicts payload.
- Translate each entry's `video_id` to the bare `example_key` by stripping the `images/` directory prefix and the file extension (e.g. `images/smoke_1.jpg` → `smoke_1`). This is the form `dataset_export_service` emits as the ground-truth `id` in `annotations.json`; without normalization the rescoring service finds zero (prediction, ground_truth) pairs.
- Map `answer` → `prediction`; preserve `video_id`, `datasource`, `correct_answer`, `reasoning`, `full_response` as auxiliary fields for debugging.
- Aggregate all entries into a single list and write to `{cache_dir}/per_sample_predictions`.

See `services.tao_polling_service._materialize_evaluate_predictions` for the canonical implementation.

**F10 (Phase 12 amendment 2026-05-05) — `prediction` field MAY carry a markdown-fenced JSON wrapper.** cosmos-rl 6.26.3's vLLM-served `cosmos-rl-evaluate` does NOT enable structured generation; the model emits its instruction-tuned default output, which for Cosmos-Reason2 (a Qwen3-VL fine-tune) wraps JSON in a markdown code fence:

```
```json
{"gesture": "rock"}
```
```

Implementations of the rescoring service (§9.7.6) MUST strip a leading/trailing markdown code-fence wrapper before `json.loads`. The canonical implementation in `services.tao_rescoring_service._strip_markdown_fences` accepts both `\`\`\`json\\n...\\n\`\`\`` and `\`\`\`\\n...\\n\`\`\`` (no language hint) shapes; idempotent on already-bare JSON; idempotent on blank input; tolerant of leading/trailing whitespace. Without this stripping, every prediction parses as schema-invalid and the rescoring service marks the run C2 (zero parseable predictions) regardless of the model's actual accuracy. Live-validated 2026-05-05 against the closing-smoke 2B baseline eval: pre-fix, 84/84 predictions classified schema-invalid; post-fix, 84/84 parsed cleanly with 28/84 (33%) exact-match.

#### 9.7.6 Automatic Post-Training and Post-Quantization Evaluation

When a TAO training job reaches `succeeded`, the system MUST automatically submit a TAO `evaluate` job against the Test Pool export to establish baseline quality for that checkpoint. When a TAO `quantize` job reaches `succeeded`, the system MUST automatically submit a TAO `evaluate` job against the quantized checkpoint. No SME interaction is required for these submissions.

**Test Pool export for TAO evaluation:**

The system MUST export the reserved Test Pool as a distinct Cosmos-RL evaluation dataset (§9.3) and MUST exclude those examples from all training exports. TAO evaluation operates only on the explicitly provided `annotation_path` and `media_dir`; it does not infer pool membership from system records.

Export rules:

- Verified non-pool examples → training export
- Reserved Test Pool examples → evaluation export only
- These MUST never be mixed

The evaluation dataset export MUST use the same Cosmos-RL LLaVA format as training exports (§9.3): `annotations.json` (top-level JSON array) + image files. The `conversations` human turn contains the rendered serving prompt, **byte-identical to the training export's human turn** (§9.3.2 F-W7 amendment — an evaluation export carrying a different prompt scores the Student out of distribution on the evaluate leg); the gpt turn contains the ground-truth label JSON string. TAO reads the gpt turn as the expected answer. The evaluation export MUST use the same `export_field_mode` as the training exports in the same chain — scoring a `core_only`-trained Student against `all`-field ground truth would produce systematic mismatch.

**TAO evaluate configuration (live-verified against FTMS 6.26.3 + cosmos-rl 6.26.3, 2026-04-29):**

- `action: "evaluate"`
- Top-level `parent_job_id` (REQUIRED for chain advancement) → the predecessor's `tao_external_job_id` (the train job's external id for a baseline evaluate; the quantize job's external id for a quantized evaluate). FTMS's `infer_parent_model_folder` helper resolves the trained-checkpoint folder onto the worker's `model.model_name` from this field. The field MUST sit at the top of the POST body, NOT nested under `specs`. Without it, the evaluate worker crashes in `cosmos_rl/evaluation/base.py` with `TypeError: expected str, bytes or os.PathLike object, not NoneType` because `model.model_name` is None.
- `specs.dataset.annotation_path` → URL of the Test Pool evaluation export's sidecar `annotations.json` (see §9.3.3 — evaluate uses the top-level `dataset` shape, not `custom.val_dataset`).
- `specs.dataset.media_dir` → URL of the exported image directory.
- `specs.policy.model_name_or_path` MUST be OMITTED. cosmos-rl's evaluate worker reads `policy.model_name_or_path` first and only falls back to FTMS-injected `model.model_name` when the key is absent. Sending an explicit value (e.g. the base experiment's `hf_model://` URL) overrides FTMS's `parent_job_id` resolution and forces the worker to load the BASE model instead of the trained student — observed live as a near-zero accuracy delta even after substantial fine-tuning.
- The cosmos-rl SFT spec overrides from §9.7.3.3 still apply on evaluate (`train.compile=false`, `train.train_policy.type="sft"`, etc.).
- TAO uploads results as a single tarball at `results/<eval_external_job_id>/evaluate_results.tar.gz`; see §9.7.5.1 for the Blueprint's tarball-extract + per-sample translation contract.

**Re-scoring with canonical evaluator:**

TAO's native aggregate metrics (accuracy, F1, precision, recall) may not match the system's exact Core-field normalization semantics (trim/lowercase enums, set comparison, strict JSON boolean typing, integer range validation). The system MUST:

1. Retrieve TAO's per-sample prediction outputs after the evaluate job succeeds.
2. Parse each prediction as a JSON label.
3. Re-score every prediction against the Test Pool ground truth using the canonical Exact Match evaluator (Appendix A.2).
4. Produce authoritative metrics: overall Exact Match rate, per-core-field match rate, per-value precision/recall/F1.
5. Persist both the TAO-native metrics (informational) and the re-scored metrics (authoritative) on the evaluation Run Record.

The frozen evaluation export defines the scoring population. Before creating
the Run Record or setting `quality_status="validated"`, the rescoring service
MUST verify the archive bytes against the SHA-256 recorded on the
`DatasetExport`, verify that `annotations.json` contains exactly
`DatasetExport.example_count` valid ground-truth rows with each key appearing
once, and verify that every ground-truth key has exactly one materialized
prediction. A missing or malformed checksum, unreadable archive, count
mismatch, duplicate ground-truth key, missing prediction, or duplicate
prediction for a frozen key takes the existing C2 quality-failure path; it
MUST NOT shrink the denominator or select a winner by artifact order.
Predictions for keys outside the frozen export are ignored. A prediction that
is present but schema-invalid remains in the denominator as a zero-match
example (subject to the existing all-schema-invalid C2 rule).

TAO's aggregate metrics are useful for quick directional checks; the system's re-scored metrics are the authoritative source for quality comparison and gate decisions.

**F33 (Phase 12 amendment 2026-05-05) — operator-driven re-rescore for since-fixed amendments.** When the rescoring service is amended (e.g., F10 markdown-fence stripping for cosmos-rl 6.26.3 vLLM-served evaluate output, or F29 quantize-parent prefix glob in `_materialize_evaluate_predictions`), StudentModels whose original rescore returned a C2 path under the old code remain at `quality_status="failed"` even though the per-sample predictions persisted on disk would now parse cleanly. The system MUST expose an operator-callable path to replay the rescore against an existing evaluate TAOJob without reissuing the TAOJob itself. The endpoint is `POST /v1/projects/{project_id}/student_models/{student_model_id}:rerescore` (§10.2.20). The handler:

1. Loads the StudentModel; rejects with HTTP 404 when not found.
2. Refuses to operate (HTTP 409) when `quality_status` is anything other than `"failed"`. F33 MUST NOT overwrite a `"validated"` Student or interfere with a `"pending"` Student that is still mid-pipeline.
3. Locates the paired evaluate TAOJob by traversing `parent_tao_job_id` from the Student's `quantize_tao_job_id` (for quantized variants) or `tao_job_id` (for baseline variants), filtering for `action="evaluate" AND status="succeeded"`. Rejects with HTTP 400 when no paired evaluate TAOJob is found.
4. Invokes the canonical `rescore_evaluate_job(project_id, evaluate_tao_job_id, settings=settings)` against the existing evaluate TAOJob. The rescore reads the on-disk per-sample predictions under current `_strip_markdown_fences` (F10) and `_materialize_evaluate_predictions` (F29) logic, re-scores via the canonical Exact Match evaluator (Appendix A.2), creates a fresh RunRecord with `evaluation_source="tao"`, and atomically updates `StudentModel.quality_status` and `StudentModel.quality_evaluation_run_id`.
5. Returns 200 with `{"run_id": "<new RunRecord>" | null, "quality_status": "<post-rescore status>"}`. A null `run_id` indicates the rescore returned a C2 path even under current code (zero parseable predictions, ground truth missing, or other terminal failure); the Student remains `quality_status="failed"` in that case.

The rescore is IDEMPOTENT against the underlying TAO predictions — F33 does NOT re-fetch artifacts from TAO workspace S3. If the on-disk predictions were materialized under an obsolete `_materialize_evaluate_predictions` glob (F29 case) and the artifact directory has been pruned since, F33 returns the C2 outcome with no flip. Re-fetching artifacts after a Blueprint amendment to the materialization side is a separate concern (out of scope for F33; would require an upstream change to `outputs_fetch_status` lifecycle from terminal `"completed"` back to `"pending"`).

F33's safety guard (the `quality_status="failed"` precondition) makes the endpoint safe to call repeatedly: each call against a Student that is already `"validated"` returns 409 without side effects. Operator-driven invocation is the only entry point — the polling service's automatic post-success rescore is unchanged.

**F33 + F35 interaction.** `partial` Students cannot use `:rerescore`. F33's purpose is to replay a TAO rescore against on-disk per-sample predictions; `partial` is set by NIM-eval (§9.5 F35) and there is no on-disk TAO rescore to replay. The remediation path for a `partial` Student is to re-run NIM eval (the underlying model output may improve on a fresh inference pass) until the run lands `completed` and promotes to `validated`. The endpoint's safety guard refuses `partial` with the same 409 `student_not_failed` body the existing path uses for `validated` and `pending`; the docstring at `routers/student_models.py::rerescore_endpoint` calls this out explicitly so operators understand the flow.

#### 9.7.7 Job Chaining

The post-training workflow is a deterministic chain of TAO jobs. The chain is defined at "Start Training" time based on the selected base models and quantization schemes. All jobs in a chain share a `chain_id` and are sequenced by `chain_sequence`.

**Chain structure per base model:**

```text
1. train (action: "train")
2. evaluate baseline (action: "evaluate", parent: train job)
3. quantize scheme A (action: "quantize", parent: train job)
4. evaluate scheme A (action: "evaluate", parent: quantize job)
5. quantize scheme B (action: "quantize", parent: train job)  [if selected]
6. evaluate scheme B (action: "evaluate", parent: quantize job)  [if selected]
...
```

**Chaining rules:**

- Each job in the chain is a distinct TAOJob record with its own `tao_job_id`.
- `chain_id` groups all jobs for one base model into a single chain. `chain_sequence` (integer, 1-based) orders them for display purposes.
- At "Start Training" time, the system pre-creates TAOJob records for every job in every chain with `status=not_started` (§9.7.2). Only the first job in each chain is immediately submitted (`not_started` → `submitting` → `submitted`). The full chain definition is persisted so the UI can show expected vs completed jobs.
- **Eligibility (chain isolation, amended 2026-05-04):** the chain advances by **dependency satisfaction**, not by `chain_sequence` ordering. A `not_started` job is eligible when its `parent_tao_job_id` is `null` (chain root) OR when the predecessor referenced by `parent_tao_job_id` has reached `status=succeeded`. Among eligible jobs, the system submits the lowest `chain_sequence` first.
- **Halt scope on failure (chain isolation):** when a job reaches `failed` / `canceled`, the system marks as halted **only the transitive dependents** of the failed job — i.e., the `not_started` jobs whose `parent_tao_job_id` chain reaches the failed `tao_job_id`. Independent siblings whose parent is a different (still-`succeeded`) job are NOT halted; they remain `not_started` and become eligible for submission via the eligibility rule above. Concretely: for the chain layout shown above, a failed `evaluate baseline` (seq=2) does NOT halt `quantize scheme A/B` (seq=3, seq=5) because those jobs parent on the still-succeeded `train` (seq=1). A failed `train` (seq=1) DOES halt every job, because every other job depends transitively on the train output.
- **Why chain isolation:** the previous "halt every chain_sequence > N" semantics treated the chain as a strict sequence even though the actual data dependencies form a DAG (quantize jobs parent on train, not on evaluate). When TAO's evaluate fails for an upstream-loader reason that doesn't affect the trained checkpoint itself (see §9.5 NIM-eval-as-quality-fallback; the 2026-05-04 loader-gap finding is retained in the project's internal engineering archive), the quantize jobs SHOULD still proceed — their input is the trained checkpoint, which is fine. The Blueprint's downstream NIM-eval-as-quality-fallback validates the resulting Student variants. Chain isolation makes this DAG-correct rather than sequence-conservative.
- Chains for different base models run sequentially (one model's full chain completes before the next begins).
- On completion of all chains, the system transitions to the Compare & Deploy view with quality results available.
- System-generated chained TAO jobs MUST omit `force_create` (TAO defaults to `false`, which prevents duplicate-artifact creation).

**Submission protocol:**

1. Transition the TAOJob to `submitting`.
2. Send the TAO create-job POST.
3. On success: persist `tao_external_job_id` from the response and transition to `submitted`.
4. On failure: transition to `failed` with `error_ref` describing the submission error.

**Submission interrupted recovery:** if the backend crashes between steps 1 and 3, the TAOJob remains in `submitting` with `tao_external_job_id = null`. On restart, these jobs transition to `failed` with `status_reason="submission_interrupted"`. The UI MUST allow the SME to retry the failed step or restart the chain from the failed step.

**Chain advancement from persisted state:** the system MUST be able to derive the next required submission from persisted chain state after restart (§1.3). On scan, for each chain the system identifies the lowest-`chain_sequence` `not_started` job whose dependency is satisfied (parent is `null` or `succeeded`) and submits it. The chain-isolation eligibility rule above replaces the prior "next sequence after the highest succeeded" rule — independent siblings remain eligible even when an earlier-sequence sibling failed, as long as their own parent is `succeeded`.

#### 9.7.8 TAO Provisioning Model (Workspace, Base Experiments, Dataset Upload)

Stock TAO FTMS deployments require three resources to exist before any training suite can execute: a **workspace** (S3-backed storage owner for all datasets and job outputs), one **base experiment per trainable student base** (e.g. Cosmos Reason2 8B, Cosmos Reason2 2B — registered in the workspace), and the **training / Test Pool dataset archives uploaded** into the workspace's S3 bucket at paths that TAO's execution environment can read. A fresh FTMS instance has none of these — `/orgs/{org}/workspaces`, `:list_base_experiments`, and `/datasets` all return empty.

**FTMS 6.26.3 workspace-create wire contract.** `POST /orgs/{org}/workspaces` MUST send `cloud_type` at both the top level and inside `cloud_specific_details`; the nested value is the OpenAPI discriminator and omitting it produces `HTTP 400 Missing data for required field`. For an S3-compatible workspace, the nested object uses `cloud_bucket_name` (not `cloud_bucket`). Its `endpoint_url` is `tao_workspace_s3_endpoint_url_internal`, the URL visible to TAO's own containers; `tao_workspace_s3_endpoint_url_external` is retained only for Blueprint-side uploads/downloads and MUST NOT be substituted into the FTMS payload. SeaweedFS creation additionally sends the conventional S3-compatible `cloud_region: "us-east-1"`. This contract matches the v8 server bootstrap helper and the committed FTMS 6.26.3 OpenAPI fixture.

The Blueprint supports **two provisioning paths**, with Blueprint self-service as the default:

**Self-service provisioning (DEFAULT, Blueprint-driven).** When the Blueprint host has outbound network access to Hugging Face (or an equivalent mirror) and the deployment's S3 bucket accepts writes from the Blueprint's configured credentials, **Start Training** provisions only the selected missing bases (§9.7.8.1a). It creates a provisional project-scoped TrainingSuite plus one durable deployment-scoped provisioning run, returns immediately, and performs the existing idempotent packaging/upload/registration flow in the background. Training Jobs polls the suite. After provisioning succeeds, the backend materializes dataset exports and TAO chains on that same suite and starts training automatically. Multiple missing selections, including Cosmos 3 Nano and Super, share one provisioning run. `vlm-feedback-loop tao-pull-base-experiments` remains the eager/operator entry point. NVIDIA's packaging tool runs out-of-process in an isolated `uv` environment; the FastAPI process never imports its dependency stack.

**Admin handoff (FALLBACK, for air-gapped or policy-separated deployments).** When Hugging Face is not reachable from the Blueprint host, or when organizational policy mandates that infrastructure ownership (rather than the Blueprint) writes base experiments into the TAO workspace, the admin performs the equivalent steps off-Blueprint (§9.7.8.1b) using either NVIDIA's `nvidia-tao-core` CLI with NGC-TAO registry credentials, or a hand-carried airgapped bundle. The resulting base-experiment UUIDs are communicated back to the Blueprint operator, who patches them into `ModelConfig.tao_base_experiment_id` (via `vlm-feedback-loop tao-bootstrap --admin-managed --base-experiment-id-2b=<UUID> …`).

Both paths produce the same end state: a populated workspace with registered base experiments keyed by `tao_base_experiment_id` on each `student_base` ModelConfig. The suite launcher checks that end state to decide whether Training Jobs needs a provisioning stage.

**Per-training operational flow (every training suite, Blueprint-automated — unchanged across both paths).** Before export work, the launcher creates or adopts a durable TrainingSuite owner in `preparing` and links each completed training + Test Pool DatasetExport before workspace transfer. The Blueprint's dataset upload service (§9.7.8.2) validates and uploads each archive/sidecar pair into the workspace's S3 bucket before submitting any TAOJob. The TAOJob's `tao_create_job_request.specs` reference the uploaded S3 paths (training dataset) or dataset IDs (evaluation dataset), NOT the Blueprint host's local filesystem. Uploads are idempotent per `(workspace_id, dataset_export_id)` only when both objects match. A failed pre-chain transfer can therefore resume the same frozen export IDs and repair an incomplete pair without creating new object keys. Completed uploads persist the resulting S3 URI on the `DatasetExport` record for auditability and reuse. A synchronous ready-base creation failure returns `409 tao_dataset_upload_failed` with remediation. When first-use base provisioning has already returned a provisional suite, a later upload failure instead terminalizes that suite as `failed`; its GET response exposes the same error through `setup_error_ref` and the backend-derived recovery decision through `setup_retryable`.

##### 9.7.8.1a Self-service provisioning (DEFAULT)

Self-service is a UI + service contract with an optional eager CLI. The Blueprint provides:

- `POST /v1/projects/{project_id}/training_suites` performs server-side readiness validation. If every selected base is ready, it creates the ordinary suite directly. Otherwise it returns a provisional suite with `status="provisioning"`, `provisioning_run_id`, `provisioning_model_names[]`, null dataset-export ids, and no chains. The Training Job Monitor renders one NVIDIA-green **Provision Student Bases** step only when `provisioning_run_id` is non-null.
- `POST /v1/projects/{project_id}/tao_base_experiment_provisioning` accepts the selected project-local Student-base ids, filters out ready bases, creates a durable `TAOBaseExperimentProvisioningRun`, and returns 202. `GET .../tao_base_experiment_provisioning/{run_id}` exposes `queued | running | succeeded | failed`, per-target results, and a redacted error. The suite launcher invokes this endpoint's service with the complete selected id list, so all missing targets are batched. Only one deployment-scoped run may be active. Backend restart marks an active run and its provisional suite failed and changes incomplete ModelConfig pull states to `failed`; retry is idempotent.
- The shared service (1) resolves the selected `student_base` entries and Hugging Face paths, (2) writes a temporary `hf_model://` CSV, (3) invokes the packaged `tao_base_experiment_pull/pull_base_experiments.py` helper through `uv run --isolated --no-project --with-requirements ...`, (4) recursively uploads the tree to `s3://{bucket}/shared-storage/models/`, (5) calls `POST /orgs/{org}/jobs:load_airgapped`, (6) resolves the registered UUIDs, and (7) patches `tao_base_experiment_id` + `tao_base_experiment_pull_status="pull_complete"` across every project database. Step 4 opens each regular file beneath the disposable stage root once and uses that descriptor for its hash, size decision, and upload; a changed pathname cannot redirect staged checkpoint bytes after validation, and an in-place change fails the exact-stream hash guard.
- `vlm-feedback-loop tao-bootstrap --self-service` (default) creates/adopts the workspace and defers bases to first use. `--eager-bases`, or the standalone `tao-pull-base-experiments` command, provisions all supported bases ahead of time.
- Inputs: NGC Personal API Key (provided as `TAO_API_KEY` through the process environment or canonical `.env`), a reachable Hugging Face endpoint, and S3 credentials for the workspace backing store (`TAO_WORKSPACE_S3_*`, §1.6) from the same approved secret sources. No TAO-registry-scoped NGC credentials required because the HF path sidesteps NGC entirely. Credential-valued CLI flags are not supported.
- Idempotency: re-running is a no-op on already-registered base experiments (the `load_airgapped` step is skipped when `find_base_experiment_by_arch` finds an existing registration; the CLI reports current state).
- The subprocess driver's dependencies (`nvidia-tao-core>=6.25.0`, `huggingface_hub>=0.23`, `requests>=2.31`) live beside the packaged helper in `tao_base_experiment_pull/requirements.txt` and are resolved in an isolated, cached `uv` environment in local-source, containerized, and installed-wheel modes. `tao-pull-base-experiments --skip-install` uses the current interpreter for operator-managed air-gapped environments.

This provisioning path is **live-verified** against FTMS 6.25.11/6.26.3 as of 2026-04-18 (evidence: 1789-byte `ptm_metadatas.json`, `load_airgapped` returned `{"success":true,"experiments_loaded":1,"experiments_failed":0}`, `POST /jobs` accepted with the registered UUID as `base_experiment_ids[0]`). The full design record is retained in the project's internal engineering archive.

##### 9.7.8.1b Admin handoff (FALLBACK)

When self-service is not usable — air-gapped sites, Hugging Face unreachable, policy separation between infrastructure and application ownership — the admin performs the equivalent steps off-Blueprint. The procedure is documented in `docs/tao-ftms-install.md` (its §10 covers base-experiment registration). High-level contract:

- Admin runs `pip install nvidia-tao-core` on a build host with network access to NGC-TAO (or a known-good pre-packaged bundle), produces `ptm_metadatas.json` + checkpoint dirs for each `student_base` variant, uploads to `s3://{workspace_bucket}/shared-storage/models/`, and calls `POST /orgs/{org}/jobs:load_airgapped`.
- Admin reports the registered `tao_base_experiment_id` UUIDs back to the Blueprint operator.
- Operator runs `vlm-feedback-loop tao-bootstrap --admin-managed --base-experiment-id-2b=<UUID> --base-experiment-id-8b=<UUID>` which writes the UUIDs into `ModelConfig.tao_base_experiment_id` across all project DBs and persists `TAODeploymentConfig.bootstrap_status="bootstrapped"`. The Blueprint does NOT attempt an NGC/HF download in this mode.

The `--admin-managed` flag and `docs/tao-ftms-install.md` remain the operator escape hatch when automatic provisioning reports missing network access or credentials.

##### 9.7.8.2 Dataset upload to workspace S3

When a training suite is created (§10.2 training suite endpoint), the Blueprint:

1. Creates or adopts the durable TrainingSuite in `preparing` before filesystem or network work.
2. Creates each DatasetExport archive on the Blueprint host (§9.3) — training split + Test Pool evaluation split — and commits its completed row plus suite link in one transaction before upload. Both export selection snapshots retain the exact suite-request checksum.
3. Requires each DatasetExport row to be `completed`, obtains both paths exclusively from its `artifact_refs`, and opens both as regular files beneath `{WORKSPACE_ROOT}/projects/{project_id}/exports/`. Non-regular files and paths resolving outside that root are rejected; an in-root symlink is resolved once, then the canonical target is opened through a no-follow directory walk so retargeting cannot change the authorized inode.
4. Through those same open descriptors, verifies the archive against the row's recorded SHA-256, reads the root regular-file member `annotations.json`, hashes the standalone sidecar, and parses it. The two JSON values MUST be recursively type-sensitive and equal: object key order and source formatting are irrelevant; array order and JSON types remain significant (`true` is not `1`, and `1` is not `1.0`). Both representations MUST validate before any S3 request.
5. Uploads the archive and sidecar to the configured workspace's S3 bucket using the workspace's access credentials (retrieved at suite-creation time from TAO via `GET /orgs/{org}/workspaces/{workspace_id}` or configured deployment-level). Hashing, parsing, size selection, and upload all consume duplicates of the original open descriptors, so replacing a stored pathname after authorization cannot change the uploaded inode. The exact byte stream is hashed again: a changed single-PUT body is rejected before dispatch, and a changed multipart stream is aborted before completion. Upload path convention: `s3://{bucket}/vlm-feedback-loop/projects/{project_id}/dataset_exports/{dataset_export_id}/{archive_name}`.
6. Treats the archive and sidecar as one idempotent pair with distinct object keys. A remote object is reused only when its SHA-256 metadata matches; a missing or mismatched member is uploaded. `already_uploaded` is true only when both matched, and local upload lineage is persisted only after both objects succeed. A sidecar failure can leave an unreferenced archive object; a safe retry repairs the pair.
7. Persists the resulting S3 URI and per-file paths on the DatasetExport record (`dataset_upload_ref`, `dataset_upload_uri`).
8. Constructs TAOJob `tao_create_job_request.specs` referencing the uploaded paths via the action-specific dataset-binding fields (§9.3.3): `custom.train_dataset.{media_path, annotation_path}` for `train`; top-level `dataset.{media_dir, annotation_path}` for `evaluate` and `quantize` (F11 amendment 2026-05-05 — quantize migrated from the train-side binding because cosmos-rl-quantize accepts `--media_dir` and rejects `--media_path`). The sidecar `annotations.json` is uploaded as a separate JSON object alongside the tarball so `annotation_path` is a JSON URL (not a path inside the tar.gz). The job chains and the suite transition from `preparing` to `initialized` commit atomically.

Files above 8 MiB are streamed from the authorized descriptor in multipart chunks; smaller files use one bounded in-memory PUT. Transient upload failures follow the standard retry policy (§11). Permanent failures (auth rejected, bucket missing, quota exceeded) surface a clear error referencing the workspace configuration. For ready-base synchronous creation this is HTTP 409 with code `tao_dataset_upload_failed`. If first-use provisioning already returned a provisional suite, the asynchronous setup task records the error in `setup_error_ref` and makes the suite `failed`.

A `failed` suite with both linked exports, no chain rows, and a transfer or
workspace-configuration error MAY be atomically reclaimed only by the same
idempotency key and exact request body. The retry MUST reuse the linked export
IDs, validate their frozen intent, tier, Guidance, field mode, and request
checksum, and rely on pair-wide remote hash checks to skip or repair objects.
A backend restart during this state retains the same retry contract. Integrity
failures MUST NOT reclaim the compromised exports: after correcting the source,
the caller creates a new suite with a new idempotency key. Once chain rows
exist, normal idempotent replay returns the existing suite and never rebuilds
its setup.

##### 9.7.8.3 Server-side launch validation

The training-suite launch path MUST verify:

- TAO bootstrap is complete and the recorded workspace is accessible (`GET /orgs/{org}/workspaces/{workspace_id}` returns 200).
- TAO's v2 request schema accepts the configured safe `timeout_minutes`.
- Each selected model has the `student_base` role.
- The training export has at least one eligible Verified example.
- The active-Guidance Test Pool contains at least
  `max(1, project.scaleup_min_test_pool_size)` Verified examples. This reuses
  the project's configured statistical floor without importing the other four
  Teacher-quality gate criteria into Student Training.
- For each selected `student_base_model_config_id`, determine either (a) the base is ready (`tao_base_experiment_id` non-null and pull status `pull_complete`) or (b) automatic first-use provisioning is required.
- When `enable_lora=true`, the Blueprint host has `HF_TOKEN` and a usable
  isolated merge interpreter containing torch, transformers, peft, accelerate,
  and safetensors. This is required even when the TAO copy of the base is
  already provisioned because baseline packaging loads the gated base locally.
- The exact `quantization_schemes` selection is compatible with every selected
  base. Cosmos 3 Super accepts only an empty selection (baseline-only) in the
  qualified release; readiness and final suite materialization both enforce
  this before export or TAO submission.

These checks are rendered from the same `training_preflight` service on the
Scale-Up Hub and Student Training screen; they are not reimplemented in
TypeScript. The suite materialization path MUST repeat the two dataset checks
after any long-running base provisioning and MUST reject a final training or
evaluation export that no longer meets them. A valid missing base is
non-blocking and produces the conditional Training Jobs provisioning stage.
Workspace absence and unsafe timeout capability remain blocking and return an
actionable submission error before transfer or TAO-job work begins. Dataset
shortfalls are data-readiness failures and MUST NOT generate a TAO setup Action
Request.

##### 9.7.8.4 Rationale

Self-service as the default reflects three live-verified capabilities: (1) `POST /jobs:load_airgapped` is client-callable with the Blueprint's NGC-exchanged JWT — no admin-only scope; (2) NVIDIA's `nvidia-tao-core.microservices.pretrained_models` tool supports `AIRGAPPED_MODE=true` which writes `ptm_metadatas.json` directly without the MongoDB connection the non-airgapped path requires; (3) the Cosmos Reason2 2B/8B models are published on Hugging Face under the `nvidia/Cosmos-Reason2-{2B,8B}` paths and are fetchable unauthenticated, so TAO-registry NGC scope is not required. Together these make the provisioning step a Blueprint-ownable operation for any deployment with HF reachability.

The admin-handoff path is preserved — and MUST be preserved — for three kinds of deployment: (a) air-gapped sites where the Blueprint host has no outbound network; (b) sites whose policy separates infrastructure ownership from application ownership and explicitly forbids the Blueprint from writing shared TAO state; (c) sites that want NGC-TAO packaged variants (with `experiment.yaml` and other TAO-native artifacts) rather than raw Hugging Face checkpoints. In each case the admin path is the correct path, and the Blueprint's preflight failure message directs the operator to it.

Attempting to force the admin path as the only option would break the Blueprint's out-of-the-box experience for the common case (developer installing on a workstation with HF reachability). Attempting to force self-service as the only option would break deployments in regulated or air-gapped environments. Offering both paths — with self-service as the default — matches the realities of TAO deployments.

##### 9.7.8.5 Residual training-time compatibility note

Self-service registers Hugging Face-format checkpoints (`config.json` + `*.safetensors` shards + tokenizer files) as base experiments. The admin path typically registers NGC-TAO packaged variants (which may additionally include `experiment.yaml` and TAO-native artifacts). For the Cosmos Reason2 family, TAO's `cosmos-rl` training engine is expected to accept the raw Hugging Face format (the upstream `cosmos-rl` project uses Hugging Face-native loading, and Cosmos Reason2 is published as Hugging Face checkpoints). This expectation is empirically validated by Step 11.6's live smoke. If a future base model family requires an NGC-TAO packaged variant for training compatibility, the admin-handoff path remains available for that family without changing the Blueprint's default behavior for the Cosmos Reason2 family.

### 9.8 Quantization Workflow (Post-Training, Two Lanes)

Quantization is an explicit post-training stage, not an implicit attribute of the Student. Quantization uses the TAO Cosmos-RL `quantize` action exclusively. The full Student lifecycle is:

`fine_tune → merge_and_evaluate_baseline_via_student_nim → quantize(optional) → evaluate_quantized_via_tao → package_for_nim → deploy`

**LoRA baseline evaluation amendment (2026-07-30).** TAO FTMS 6.26.3's v2
`evaluate` action does not expose `enable_lora` or `base_model_path`; submitting
the adapter-only training output directly causes vLLM to reject PEFT-prefixed
weights. For the post-`train` baseline row of a LoRA chain, the Blueprint MUST
therefore: fetch the adapter, merge it with the persisted
`resolved_training_fields.policy.model_name_or_path`, validate the merged
Hugging Face checkpoint, deploy it temporarily through the normal local
Student NIM lifecycle, evaluate the frozen Test Pool, and persist that Run ID
on both the StudentModel and the existing baseline `TAOJob` row. That row uses
`training_backend="student_nim_local"` and
`outputs.evaluation_source="student_nim_local"`; it reaches `succeeded` only
when serving validates and quality is `validated` or `partial`. A failure is a
real evaluate failure and follows normal chain-isolation semantics.

This redirect applies only when the evaluate row parents directly on `train`.
An evaluate row parented on `quantize` remains TAO-native because quantize
already merged the LoRA adapter while producing the quantized checkpoint.

**Default quantization:** `FP8_DYNAMIC` alone is pre-selected for **Validate
training setup**, producing a full-precision baseline and FP8 variant.
`W4A16`, `W8A8`, and `W8A16` are explicit additional comparison choices.
Each scheme is a separate TAO `quantize` job; TAO runs one
`quantization_scheme` per job.

**F-B13 amendment (2026-08-04) — Cosmos 3 Super is baseline-only until its
quantization runtime is qualified.** A full-weight Super checkpoint completed
training, packaging, local Student NIM evaluation, benchmarking, and handoff.
The same checkpoint's `FP8_DYNAMIC` TAO action completed activation
calibration across all 65 layers, then Accelerate reported that parameters had
been CPU-offloaded on the meta device. During weight calibration,
llmcompressor attempted to calculate a scale from one of those unmaterialized
weights and failed with `Tensor.item() cannot be called on meta tensors`.
No quantized checkpoint was produced. The Blueprint MUST therefore keep Super
available for baseline training/deployment but reject any non-empty
`quantization_schemes` selection containing Super. The backend-owned
`quantization_compatible` readiness check is authoritative, suite launch and
final materialization repeat the rule before exports or TAO work, and the
Training screen renders a baseline-only notice in its Quantization section
whenever Super is selected. Clearing every quantization checkbox makes the
Super configuration valid. A later stack may remove this restriction only
after an independently qualified Super quantize→package→NIM live run.

**Automatic evaluation:** TAO's `skip_test_generation` (default: `false`) runs a post-quantization smoke check to verify the quantized model can generate. This is a correctness sanity check, not a full accuracy study. The system's automatic TAO `evaluate` job (§9.7.6) runs separately after each quantize job to produce full accuracy metrics against the Test Pool.

#### 9.8.1 TAO / Cosmos-RL `quantize` Action

Documented quantization methods: **FP8_DYNAMIC** (default), **W8A8, W8A16, W4A16** (per current Cosmos-RL docs). Operates within the TAO execution substrate (`tao-client cosmos-rl quantize`).

TAO `quantize` spec structure:

- `model` (path to trained checkpoint or merged LoRA checkpoint)
- `calibration_dataset` (dataset for calibration)
- `quantization_method` (e.g., FP8_DYNAMIC, W8A8, W8A16, W4A16)

Rules:

- If `enable_lora=true` on the training job, `base_model_path` is required so TAO can merge the adapter before quantization.

**F-B11 amendment (2026-07-15) — the LoRA merge flags MUST reach the quantize wire.** Live evidence from the July 2026 live-validation window: every quantize on an adapter-only parent died in the container with `ValueError: base_model_path is required when enable_lora is True` (cosmos-rl `lora_utils.load_model_and_processor` raises before its own adapter-config inference fallback can run), because the Blueprint never emitted the flags. The contract is now: when the chain's training mode is LoRA (`enable_lora=true`), the quantize spec MUST carry `specs.enable_lora=true` and `specs.base_model_path=<bare HuggingFace id>` — the same identifier persisted in the train job's `resolved_training_fields.policy.model_name_or_path`. The bare id, NOT the `hf_model://` scheme: quantize spec keys map 1:1 onto cosmos-rl-quantize CLI flags (the `max_sequence_length` mechanism), and the container feeds `base_model_path` straight to `from_pretrained()` (gated-repo auth via the uniform `docker_env_vars.HF_TOKEN` injection). Full-weight chains (`enable_lora=false`) MUST omit both keys — the legacy wire shape. Live-verified end-to-end on FTMS 6.26.3 (2026-07-15): GT-nano LoRA parent quantized FP8_DYNAMIC clean in ~10 min, in-container merge → calibration → quantize → test generation. Canonical implementation: `services.training_suite_service._build_quantize_payload`.
- MUST be represented as a separate TAOJob record with `action: "quantize"` and linkage to the parent training TAOJob via `parent_tao_job_id`.
- Accuracy verification is automatic: the system submits a TAO `evaluate` job after each quantize job succeeds (§9.7.6, §9.7.7). The SME does not need to trigger evaluation manually.
- TAO quantize produces a quantized checkpoint artifact, not a packaged NIM profile. The output MUST still pass the §9.5.1 packaging step before it can be registered and deployed through NIM.
- Persist `quantization_method` on the Student variant record (e.g., `FP8_DYNAMIC`, `W4A16`, or `none` for full-precision baseline).

---

## 10. Architecture & Interfaces

### 10.1 Components (Conceptual)

- **Frontend** (React + TypeScript SPA) ↔ **Backend** (Python + FastAPI; REST + SSE) ↔ NIM endpoints (Teacher, Student, embedding NIM)
- **Domain store** (SQLite per project): projects, examples, labels, guidance, pools, lineage, operation records, run records, model catalog, dataset exports, TAO job records, audit events
- **Background task executor** (in-process asyncio): evaluations, CLIP embedding computation, Batch Labeling
- Optional adjacent systems: Cosmos-RL / TAO VLM (TAO Toolkit)

### 10.2 Key Interface Requirements

**Project CRUD (required)**

- `POST /v1/projects` create
- `GET /v1/projects/{project_id}` read
- `PATCH /v1/projects/{project_id}` update (selections + `feature_flags`)

**Deployment-scoped endpoints (not project-scoped)**

- environment assessment (Section 10.2.25)
- filesystem browse (Section 10.2.10)

**Project-scoped routing (required)**
All non-project resources MUST be accessed in a project scope (recommended pattern):

- NIM endpoints (Section 10.2.26)
- image ingestion (Section 10.2.1)
- review selector (Section 10.2.4)
- proposals (Section 10.2.3)
- guidance ICL count (Section 10.2.5)
- batch labeling runs (Section 10.2.6)
- evaluation runs/suites (Section 7 + runs record)
- dataset exports (Section 13.5)
- TAO jobs (Section 10.2.7)
- examples query (Section 10.2.8)
- image serving (Section 10.2.9)

Implementations MAY accept `project_id` in request bodies, but MUST enforce partitioning and MUST NOT allow cross-project IDs.

**Background task SSE stream (required)**

- `GET /v1/projects/{project_id}/events`: SSE stream scoped to a single project.
- Event types: `evaluation_started`, `evaluation_progress`, `evaluation_completed`, `embedding_progress`, `embedding_completed`, `batch_label_progress`, `batch_label_completed`, `ingest_progress`, `ingest_completed`, `tao_job_progress`, `tao_job_completed`, `nim_benchmark_progress`, `nim_benchmark_completed`, `run_failed`. (`project_archived` / `project_unarchived` also travel on this stream; their payloads are specified in §10.2.13.1.)
- Each run-scoped event payload MUST include `run_id` (or an equivalent task identifier, e.g. `student_model_id` on `nim_benchmark_*`) and `timestamp`; the ingest sweep is a per-project singleton, so `ingest_*` events carry no run identifier.
- Progress events for count-based work (evaluation, embedding, batch label, ingest) MUST include `processed: int` and `total: int`. `tao_job_progress` instead relays TAO's `status`/`progress`; `nim_benchmark_progress` carries the benchmark `stage` and `elapsed_ms`.
- Completion events MUST include final status and summary metrics.
- `run_failed` MUST include `run_id`, `run_type`, and `error_summary`.
- All state conveyed via SSE MUST also be queryable via the corresponding REST status endpoint.

Model invocation via OpenAI-compatible NIM:

- `POST /v1/chat/completions` (multimodal)
- `GET /v1/models` (when supported)

#### 10.2.1 Image ingestion endpoints (required)

Recommended endpoint:

- `POST /v1/projects/{project_id}/examples:ingest`

Request (single or batch):

```json
{
  "examples": [
    {
      "example_key": "string",
      "storage_ref": "string",
      "source_metadata": {},
      "metadata": {},
      "state": "Unlabeled"
    }
  ]
}
```

Rules:

- `example_key` MUST be unique within project; treated as idempotency key.
- If `{project_id, example_key}` exists and `storage_ref` matches the existing record: server MUST return existing Example record with `status="exists"`; MUST NOT create duplicate. This is true idempotent re-ingest.
- If `{project_id, example_key}` exists but `storage_ref` differs: server MUST reject that item with `status="error"`, `error_code="example_key_collision"`, and a message identifying both paths: *"Generated example_key collided with an existing example from a different path. Existing: {old_storage_ref}. New: {new_storage_ref}."* This distinguishes accidental key collision from true re-ingest.
- If `{project_id, storage_ref}` already exists under a different `example_key`, server MUST reject the new item with `status="error"` and `error_code="storage_ref_already_ingested"`. A single source image cannot enter one project twice merely because a caller supplied or an older scan root produced a different key.
- If `state` omitted, default `Unlabeled`.
- `ingested_at` MUST be set by server.
- `source_metadata` and `metadata` MUST be stored (may be `{}`).
- **Partial success:** batch ingestion MUST process items individually. A failure on one item (conflict, validation error) MUST NOT prevent other items in the same request from being ingested. Each item receives its own result status.

File discovery and storage references:

- The ingestion API operates on `storage_ref` values (absolute filesystem paths), not raw file bytes. Images are not copied; the system records paths to their original locations.
- **Path resolution is backend-driven.** The frontend uses the filesystem browse endpoint (§10.2.10) to let the user navigate the backend host's filesystem visually. The backend resolves all selections to absolute paths and passes them to the ingestion API. A direct path entry field is also available. Standard browser APIs do not expose absolute filesystem paths, so the backend MUST be the source of path resolution.
- **Folder ingestion helper:** when the user checks a directory entry in the file browser tree and clicks [Ingest Selected], the frontend calls `POST /v1/filesystem/scan` (§10.2.10) which scans the directory recursively, discovers all supported image files, generates `example_key` values using the deterministic slug+hash scheme, and returns `(storage_ref, suggested_example_key)` rows that the frontend then submits to the ingestion pipeline (§10.2.1). The scheme is deterministic so that re-ingesting the same directory is idempotent and collision-resistant. The scan endpoint is no longer a user-facing preview step (F47, 2026-05-15) — direct path entry above the file browser only navigates the tree.

Image validation at ingest:

- **File existence:** the system MUST verify that the file at `storage_ref` exists and is readable before accepting the item. If the file is missing or unreadable, fail that item with a clear error (e.g., `"File not found: {storage_ref}"`). This does not block other items in the batch (partial success rule above).
- **Format validation:** the system MUST validate that the file is a supported image format by attempting to open it with an image library (e.g., Pillow). Supported formats: **JPEG, PNG, WebP, BMP, TIFF (single-page only)**. Animated GIFs and multi-page TIFFs MUST be rejected. Unsupported or unreadable files fail that item with a clear error.
- **Size warnings (non-blocking):** if an image exceeds 20 MB file size or 8192 px on its longest edge, the system MUST return a per-item warning in the ingestion response. The image is still ingested successfully; Visual Budget Controls (§6.9) handle preprocessing and NIM resizes as needed.

Response:

```json
{
  "results": [
    {
      "example_key": "string",
      "status": "created | exists | error",
      "error": "string | null",
      "warnings": ["string"],
      "example": { "... Example record or null on error ..." }
    }
  ]
}
```

#### 10.2.2 Guidance endpoints (required)

Recommended endpoints:

- `POST /v1/projects/{project_id}/guidance` create new Guidance version
- `GET /v1/projects/{project_id}/guidance/{guidance_id}` retrieve specific Guidance
- `GET /v1/projects/{project_id}/guidance` list Guidance versions (newest-first recommended)

Draft validation (required; §6.6.6):

- `POST /v1/projects/{project_id}/guidance:validate_draft`

Request:

```json
{
  "description": "string",
  "schema": { "... SchemaCore field records per §4.4 ..." },
  "rules": "string"
}
```

`description` and `rules` MAY be empty strings; the only structural save requirement is at least one valid Core field (`NO_CORE_FIELDS` otherwise).

Response:

```json
{
  "issues": [
    {
      "severity": "error | warning",
      "code": "string",
      "message": "string",
      "fieldPath": "string | null"
    }
  ],
  "derived_json_schema": { "... canonical JSON Schema per §6.2 ..." },
  "schema_hash": "string",
  "save_allowed": true
}
```

Rules:

- The backend MUST use the same validation and derivation function for this endpoint and for the Guidance save endpoint (`POST /v1/projects/{project_id}/guidance`). One canonical derivation implementation, used by both preview and save.
- `derived_json_schema`: the canonical JSON Schema derived from SchemaCore (same derivation as §6.2). The frontend displays this in the Derived JSON Schema preview (§6.6.5).
- `schema_hash`: deterministic hash of the derived schema; the frontend uses this to detect whether the draft has changed since the last validation.
- `save_allowed`: `true` when zero errors; `false` when any error exists.
- `issues[]` is the complete issue set the builder renders (§6.6.6); each issue's `field_path` uses per-section indices (`core[i]` / `aux[j]`, in submission order, with `.name`/`.role`/`.type` suffixes for field-level issues) so the client can address the offending row.

Activate the first Guidance via Project update:

- `PATCH /v1/projects/{project_id}` with `{ "active_guidance_id": "..." }`
- This PATCH performs first activation only (FTUE, before any labels or
  runs exist). Once a Guidance is active, the pointer can be neither
  switched nor cleared through PATCH (400): the active version only moves
  through the guidance edit endpoint (§10.2.2 `:edit`), which cancels
  in-flight runs and re-points existing labels so the corpus cannot
  orphan from the active version.

#### 10.2.3 Interactive proposal endpoint (required)

Recommended endpoint:

- `POST /v1/projects/{project_id}/proposals`

Request a Teacher proposal for a specific example. Used for initial proposals and Retry. For Auto-Labeled examples, the system surfaces the existing Auto-Labeled label as the proposal (§4.7); set `use_existing_label=true` to skip a fresh Teacher call.

Request:

```json
{
  "example_key": "string",
  "teacher_model_config_id_override": "string | null",
  "guidance_id_override": "string | null",
  "generation_preset_key_override": "string | null",
  "thinking_mode_override": "on | off | null",
  "visual_budget_preset_key_override": "string | null",
  "retry_of_inference_invocation_id": "string | null",
  "use_existing_label": "boolean"
}
```

Response:

```json
{
  "inference_invocation_id": "string",
  "example_key": "string",
  "proposal_json": { "... normalized label fields ..." },
  "schema_valid_core": "boolean",
  "validation_errors_core": ["string"],
  "validation_errors_aux": ["string"],
  "invocation_status": "success | schema_invalid | timeout | endpoint_error | rate_limited",
  "latency_ms_end_to_end": "int | null",
  "icl_images_attached_count": "int",
  "icl_example_keys_used": ["string"],
  "used_existing_label": "boolean"
}
```

Rules:

- MUST persist an Operation Record before invoking the model (§13.1).
- When `use_existing_label=true` and a Label record with `label_status=auto_labeled` exists for the example, return the stored label without a Teacher call. `invocation_status="success"`, `used_existing_label=true`, and `inference_invocation_id` references the original batch labeling invocation.
- When `retry_of_inference_invocation_id` is set, the new Operation Record MUST link to the prior attempt.
- Response MUST include structured validation fields separating Core vs Aux errors (§6.3.1).
- `invocation_status="rate_limited"` distinguishes hosted-NIM 429 responses from generic endpoint errors so the UI can render wait-and-retry copy instead of the generic failure banner.
- `icl_images_attached_count` and ordered `icl_example_keys_used[]` expose the rendered ICL set (the §3.3 invariant-2 lineage and the enforcement observable for budget pruning, §6.2); the UI's ICL chip reads them.

#### 10.2.4 Review selector endpoint (required)

Recommended endpoint:

- `GET /v1/projects/{project_id}/review_selector/next`

Request the next example from the diversity-driven review selector. After a semantic Core change, prioritizes Unlabeled examples with `prior_verified_label_ref` (prior Edits first; §4.4.1, §6.5).

Response:

```json
{
  "example_key": "string | null",
  "example_state": "Unlabeled | Auto-Labeled",
  "has_existing_label": "boolean",
  "selection_mode": "clip_diverse | phash_diverse",
  "queue_empty": "boolean",
  "storage_ref": "string | null",
  "prior_verified_label_ref": "string | null"
}
```

Rules:

- When `queue_empty=true`, `example_key` is null. The UI transitions to the empty-queue state (the `queue_empty` phase of `LabelingPage.tsx`).
- `storage_ref` and `prior_verified_label_ref` carry the selected Example's image path and prior-label snapshot ref so the client needs no examples-list lookup; both null when `queue_empty=true`.
- `has_existing_label=true` when the example has a Label record with `label_status=auto_labeled`. The frontend uses this to set `use_existing_label=true` on the subsequent proposal request.
- MUST persist selector state for reproducibility (§13.3).

#### 10.2.5 Guidance rewrite endpoint (removed)

> **Removed 2026-07-21** with the AI Guidance Rewrite feature (§6.4). The pre-run ICL count is served by `GET /v1/projects/{project_id}/guidance:icl_count` → `{"eligible_count": int}` (non-pool Verified Edits under the active Guidance, §6.2).

#### 10.2.6 Batch Labeling run endpoints (required)

Recommended endpoints:

- `POST /v1/projects/{project_id}/batch_label_runs` — start run
- `GET /v1/projects/{project_id}/batch_label_runs/{run_id}` — get run status + counters
- `GET /v1/projects/{project_id}/batch_label_runs` — list runs
- `POST /v1/projects/{project_id}/batch_label_runs/{run_id}:resume` — resume paused run
- `POST /v1/projects/{project_id}/batch_label_runs/{run_id}:cancel` — cancel run

**Start run:**

Request:

```json
{
  "include_auto_labeled": "boolean",
  "run_limit": "int | null",
  "structured_generation_mode": "auto | prompt_only | null",
  "concurrency": "int | null",
  "icl_mode": "enabled | disabled",
  "ingested_after": "timestamp | null",
  "ingested_before": "timestamp | null"
}
```

The request body is flat and rejects unknown keys (`extra="forbid"`): the ingest-date bounds are top-level `ingested_after` / `ingested_before` fields, not a nested `filters` object.

`include_auto_labeled` defaults to `false`. When `true`, previously Auto-Labeled examples are included in the input pool alongside Unlabeled examples; their existing Label records are replaced with new Auto-Labeled output from the current configuration (§8.2 step 5). `structured_generation_mode` defaults to `"auto"`. `concurrency` (1–64) overrides the provider-aware dispatch width (§8.2 step 6); default `null` resolves `BATCH_LABEL_CONCURRENCY_HOSTED` / `BATCH_LABEL_CONCURRENCY_SELF_HOSTED` by the Teacher endpoint's mode. The override is persisted on the Run Record, so restart recovery resumes at the same width. `icl_mode` defaults to `"enabled"`; `"disabled"` skips ICL selection for the whole run (§8.3 F-S9 amendment — the supported zero-shot path for ICL-negative teachers), is persisted on the Run Record, and is echoed on create/detail responses.

Response:

```json
{
  "run_id": "string",
  "run_type": "batch_label_run",
  "status": "queued",
  "guidance_id": "string",
  "model_config_id": "string",
  "generation_preset_key": "string",
  "thinking_mode_effective": "on | off",
  "visual_budget_preset_key": "string | null",
  "structured_generation_mode_effective": "auto | prompt_only",
  "examples_total": "int",
  "created_at": "timestamp"
}
```

Rules:

- MUST verify Scale-Up Readiness Gate is `ready` (§8.2 step 1). Returns `409` with gate status if not ready.
- MUST snapshot configuration on Run Record (§8.2 step 3).

**Get run status:**

Response:

```json
{
  "run_id": "string",
  "run_type": "batch_label_run",
  "status": "queued | running | paused | canceling | completed | canceled | failed",
  "status_reason": "string | null",
  "paused_reason": "string | null",
  "circuit_breaker_threshold": "int | null",
  "progress": { "processed": "int", "total": "int" },
  "examples_succeeded": "int",
  "examples_schema_invalid": "int",
  "examples_timeout": "int",
  "examples_endpoint_error": "int",
  "examples_total": "int",
  "guidance_id": "string",
  "model_config_id": "string",
  "generation_preset_key": "string",
  "visual_budget_preset_key": "string | null",
  "created_at": "timestamp",
  "started_at": "timestamp | null",
  "completed_at": "timestamp | null"
}
```

**List runs:** cursor pagination (`limit`, `cursor`). Newest-first.

**Resume:** transitions `paused` → `running`. Returns `409` if not in `paused` state. Resets the consecutive failure counter (§8.2 step 8).

**Cancel:** transitions `running` or `paused` → `canceling` → `canceled` (§13.2.2). Returns `409` if in a terminal state.

#### 10.2.7 Student Training execution interface (Cosmos-RL / TAO VLM) (required when Student Training implemented)

Endpoints:

- Create/trigger TAO job:
  - `POST /v1/projects/{project_id}/tao_jobs`
- Get TAO job record (pollable):
  - `GET /v1/projects/{project_id}/tao_jobs/{tao_job_id}`
- List TAO jobs:
  - `GET /v1/projects/{project_id}/tao_jobs?limit=&cursor=&status=`
- Cancel a TAOJob and halt downstream chain siblings:
  - `POST /v1/projects/{project_id}/tao_jobs/{tao_job_id}:cancel?force_local=<bool>` (default `force_local=false`). Transitions the job to `canceled`, halts every downstream `not_started` sibling in the same chain with `chain_halted_reason` populated, and emits one SSE `run_failed` event per affected job. Returns 404 if the job does not exist, 409 if the job is already in a terminal status, and 502 `tao_error` if the TAO POST `:cancel` call fails.
  - **F17 (Phase 12 amendment 2026-05-05) — `force_local=true` skips the TAO POST entirely** and transitions the local row to `canceled` regardless of whether TAO is reachable or whether the `tao_external_job_id` is still resolvable. Use ONLY when the external TAO has been rebuilt or is permanently unreachable so its `tao_external_job_id` no longer exists. The canceled row's `poll_error_ref` is stamped `"forced_local_cancel: external TAO unreachable or external_id orphaned"` so the audit trail records why the bypass was used. The service emits a `warning`-level log line on every force-local cancel so operators can spot drift between local and external state. There is no auth gate on this flag (v1 single-user); operators are expected to use it sparingly and document use in the run record.
- Cancel a complete Training Suite:
  - `POST /v1/projects/{project_id}/training_suites/{training_suite_id}:cancel`. The service MUST set the parent suite to terminal `canceled` before external calls, cancel its in-process setup/provisioning tasks, request cancellation for every known non-terminal external TAO job, and locally cancel every remaining non-terminal TAOJob while preserving completed/otherwise-terminal work. Cancellation is best effort: remote failures are returned in `remote_cancel_failures[]` and persisted on the affected job's diagnostic reference, but MUST NOT retain the project re-entry redirect. Late submission, poll, or chain-advance results MUST NOT restore the suite to an active state or submit another chain job. Returns 409 for a `completed` or `failed` suite.

Create request body (minimum required fields; normative):

```json
{
  "student_base_model_config_id": "string",
  "dataset_export_ids": ["string"],
  "job_config": {
    "training_backend": "cosmos_rl_tao_vlm",
    "training_preset": "standard",
    "training_policy_type": "sft",
    "lora_config": {
      "enable_lora": true,
      "lora_rank": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.05,
      "lora_target_modules": ["q_proj", "v_proj"],
      "modules_to_save": null
    },
    "hyperparameters": {
      "train": {
        "epoch": 3,
        "resume": false,
        "ckpt": { "enable_checkpoint": true, "save_freq_in_epoch": 1, "max_keep": 1, "export_safetensors": true }
      }
    },
    "parallelism_config": { "tensor_parallel": 1, "pipeline_parallel": 1, "data_parallel": 8 },
    "num_nodes": 1,
    "num_gpus_per_node": 8,
    "redis_config": null,
    "tao_release_version": "6.26.3",
    "cosmos_rl_container_tag": "6.26.3-cosmos-rl",
    "dataset_refs": {},
    "intended_outputs": {
      "track_logs": true,
      "track_metrics": true,
      "track_artifacts": ["best_model", "latest_model", "training_config"]
    }
  },
  "tao_create_job_request": {
    "kind": "experiment",
    "action": "train",
    "specs": {}
  }
}
```

Rules:

- Server MUST validate `dataset_export_ids[]` and `student_base_model_config_id` belong to same `project_id`.
- For `action="train"`: server MUST load every referenced `DatasetExport` record and verify: (1) all have `dataset_intent="training"`, and (2) all share the same `export_field_mode`. Mixed modes are invalid because a Student training run MUST learn a single output contract (§9.3.5). On violation, return `400 Bad Request` with error code `MIXED_EXPORT_FIELD_MODE` and a payload identifying the conflicting exports and modes. The effective field mode is always derived from the DatasetExport records (single source of truth); no separate field mode is stored on the TAOJob or StudentModel records.
- Server MUST persist TAOJob record before triggering external TAO job.
- Server MUST persist `tao_create_job_request` exactly as submitted (or exactly as generated) and store a checksum/hash for integrity.
- If external TAO submission fails, TAOJob MUST transition to `failed` with error payload reference.

Create response:

```json
{
  "tao_job": {
    "project_id": "string",
    "tao_job_id": "string",
    "status": "submitted",
    "tao_status_raw": "string-or-null",
    "job_config": { "... as stored ..." },
    "tao_create_job_request": { "... as stored ..." },
    "outputs": { "artifacts": [], "logs_ref": null, "metrics_ref": null },
    "created_at": "timestamp",
    "started_at": null,
    "completed_at": null,
    "last_polled_at": null
  }
}
```

Get response:

```json
{
  "tao_job": {
    "project_id": "string",
    "tao_job_id": "string",
    "status": "queued | running | paused | succeeded | failed | canceled | deleted",
    "tao_status_raw": "string-or-null",
    "progress": {
      "epoch_current": 1,
      "epoch_total": 3,
      "eta_seconds": 3600,
      "metrics_latest": { "loss": 1.23 }
    },
    "outputs": {
      "artifacts": [
        { "name": "best_model", "artifact_ref": "opaque", "tao_file_path": "/workspace/models/best.pth" }
      ],
      "logs_ref": "opaque-or-null",
      "metrics_ref": "opaque-or-null",
      "tao_job_metadata_ref": "opaque-or-null"
    },
    "created_at": "timestamp",
    "started_at": "timestamp-or-null",
    "completed_at": "timestamp-or-null",
    "last_polled_at": "timestamp-or-null"
  }
}
```

#### 10.2.8 Example query endpoints

Recommended endpoint:

- `GET /v1/projects/{project_id}/examples`

Minimum query behavior:

- Cursor pagination: `limit`, `cursor`; response includes `next_cursor` when more results exist.
- Filtering:
  - `state` filter (at minimum supports `state=Verified`)
  - Verified filters (when `state=Verified`):
    - `verified_after`, `verified_before` (timestamps)
    - `verified_outcome` (`Accept`/`Edit`)
    - `guidance_id`
    - `schema_id_or_hash`
  - Pool membership filter: `pool_membership` ∈ {`test_pool`, `none`, `any`}
- Stable ordering: `verified_at desc, example_key asc`
- Include controls: `include=verified_label`

Response (illustrative):

```json
{
  "items": [
    {
      "example": { "... Example record ..." },
      "verified_label": { "... Label record ..." }
    }
  ],
  "next_cursor": "opaque-string-or-null"
}
```

#### 10.2.9 Image serving endpoint (required)

Browsers cannot load arbitrary filesystem paths. The backend MUST serve images from persisted `storage_ref` values (§1.7).

Recommended endpoint:

- `GET /v1/projects/{project_id}/examples/{example_key}/image`

Behavior:

- Look up the Example record by `{project_id, example_key}`.
- Read the file from that record's `storage_ref`.
- Stream the image bytes with the correct `Content-Type` (e.g., `image/jpeg`, `image/png`). Use standard file streaming (`FileResponse` or equivalent), not base64-in-JSON.
- If `{project_id, example_key}` does not exist, return `404`.
- If the Example exists but the file at `storage_ref` is missing or unreadable, return an appropriate error (e.g., `404` with a body indicating the file is not found at the stored path). The UI shows a broken-image placeholder and offers **Skip** (§4.5).

Security constraints:

- The endpoint MUST NOT accept arbitrary filesystem paths from request parameters.
- The only image source is the `storage_ref` already persisted on the project-scoped Example record.
- Project scoping is enforced: the Example MUST belong to the requested `project_id`.
- Authorization and streaming MUST bind to the same regular-file inode. The
  endpoint MUST re-apply the current `IMAGE_ROOT` policy and MUST NOT reopen the
  mutable `storage_ref` pathname after authorization. Normal and byte-range
  responses own and close the descriptor on success or failure.

#### 10.2.10 Filesystem browse endpoint (required)

The ingestion API (§10.2.1) requires absolute `storage_ref` paths and the system stores images by reference (§1.7). The backend MUST provide a filesystem browse endpoint so the frontend can navigate the backend host's filesystem and select image directories or files for ingestion.

Recommended endpoint:

- `GET /v1/filesystem/browse`

Query parameters:

- `path: string` (optional; absolute directory path to list; when omitted, the backend opens `IMAGE_ROOT`, or `/` in unrestricted loopback development)
- `show_files: boolean` (default `true`; when `false`, return directories only for faster navigation)
- `image_formats_only: boolean` (default `true`; when `true`, filter files to supported image formats: JPEG, PNG, WebP, BMP, single-page TIFF)

Response:

```json
{
  "path": "/data/images/project_alpha",
  "parent": "/data/images",
  "entries": [
    {
      "name": "batch_01",
      "type": "directory",
      "path": "/data/images/project_alpha/batch_01"
    },
    {
      "name": "sample_001.jpg",
      "type": "file",
      "path": "/data/images/project_alpha/sample_001.jpg",
      "size_bytes": 2048576
    }
  ]
}
```

Rules:

- The endpoint is **deployment-scoped** (not project-scoped). It browses the backend host's filesystem, not a project directory.
- When supplied, `path` MUST be absolute. Relative paths, `..` traversal, and symlinks pointing outside `IMAGE_ROOT` MUST be rejected.
- The response for `IMAGE_ROOT` itself MUST set `parent: null`, so the picker cannot navigate above its boundary.
- If `path` does not exist or is not a directory, return `404` with a clear error: `"Directory not found: {path}"`.
- If `path` is not readable, return `403`.
- Entries MUST be sorted: directories first (alphabetical), then files (alphabetical).
- Hidden files/directories (names starting with `.`) MUST be excluded by default.

**Folder ingest helper endpoint (required):**

- `POST /v1/filesystem/scan`

Request:

```json
{
  "path": "/data/images/project_alpha/batch_01",
  "recursive": true,
  "project_id": "string | null"
}
```

`project_id` is optional. When provided, the backend checks both the canonical suggested key and the source path against existing Example records in that project and reports collision status per image. When omitted, no project collision checking is performed.

Response:

```json
{
  "path": "/data/images/project_alpha/batch_01",
  "images": [
    {
      "storage_ref": "/data/images/project_alpha/batch_01/img_001.jpg",
      "suggested_example_key": "project_alpha_batch_01_img_001--9528ef04c649",
      "size_bytes": 2048576,
      "key_status": "available | already_exists_same_path | collision_different_path",
      "existing_storage_ref": "string | null"
    }
  ],
  "skipped": [
    {
      "path": "/data/images/project_alpha/batch_01/notes.txt",
      "reason": "unsupported_format"
    }
  ],
  "total_images": 150,
  "total_skipped": 3,
  "total_collisions": 0
}
```

Rules:

- Scan the directory (recursively when `recursive=true`) and return all supported image files with suggested `example_key` values.
- `suggested_example_key` MUST be deterministic, collision-resistant, and independent of which ancestor directory the SME selected for scanning. Generation rule: (1) resolve the image path and normalize it to POSIX form; (2) when `IMAGE_ROOT` is configured, use the path relative to that stable deployment root; otherwise use the normalized absolute path; (3) build a readable slug from that canonical path without extension, replacing path separators with `_`; (4) compute `hash12 = first_12_hex(sha256(canonical_path_with_extension))`; (5) set `suggested_example_key = "{slug}--{hash12}"`. The 12 hex character suffix (48 bits) provides strong collision resistance for project sizes up to hundreds of thousands of images. The extension participates in the hash input so that `foo.jpg` and `foo.png` in the same folder produce distinct keys. With `IMAGE_ROOT=/data/images`, `batch_01/sub/img_001.jpg` yields the same key whether `/data/images`, `/data/images/batch_01`, or `/data/images/batch_01/sub` is scanned.
- **Collision checking** (when `project_id` is provided): source-path identity is checked first. If `storage_ref` is already present, `key_status=already_exists_same_path` and the response reuses the persisted `example_key`, including for projects whose keys were generated by an older release. Otherwise the backend checks the canonical suggested key: `available` means unused and `collision_different_path` means that key belongs to another `storage_ref` and ingestion would reject it. When `collision_different_path`, `existing_storage_ref` is populated so the SME can see what file already owns that key. `total_collisions` counts `collision_different_path` items. When `project_id` is omitted, `key_status` defaults to `available` for all items and `total_collisions` is `0`.
- Skipped files (unsupported formats, unreadable) are reported in the `skipped` array but do not block the scan.
- **F47 amendment (2026-05-15).** The standalone "Scan + preview" SME workflow was removed. The scan endpoint is no longer surfaced as a user-visible step; it is an **internal helper** the frontend calls when the SME checks a directory entry in the file browser tree and clicks [Ingest Selected]. The frontend consumes only the `images[]` portion of the response (to extract `storage_ref` + `suggested_example_key` pairs for the ingestion endpoint, §10.2.1); the `skipped[]` array is not rendered to the SME because the wall-of-skipped-files UI failure mode it produced on wide paths (e.g., a home directory) added no SME value once §6.1 ingestion became fast and asynchronous (F46, 2026-05-14). `total_skipped` and `total_collisions` remain in the response for diagnostic purposes. Direct path entry is preserved as a text input above the file browser tree: pressing Enter navigates the tree via `GET /v1/filesystem/browse`, not via this scan endpoint.

**Security constraints (filesystem browse):**

- **Image root (`IMAGE_ROOT`):** one optional absolute directory path that bounds filesystem browsing, scanning, ingestion, image serving, and path remapping.
  - When the backend binds to a loopback address (`127.0.0.1`, `::1`) and `IMAGE_ROOT` is unset, the effective root is `/` (the entire filesystem is browsable). This is the expected local-development mode.
  - When the backend binds to a non-loopback address (`0.0.0.0`, a LAN IP, etc.), `IMAGE_ROOT` MUST be explicitly configured. If it is unset, browse, scan, image serving, and path remapping MUST return `403` with: *"Filesystem browsing is disabled. Configure IMAGE_ROOT to allow browsing when the backend is network-accessible."* Batch ingestion preserves the §10.2.1 partial-success contract: the request returns `202`, and every disallowed item returns `status="error"`, `error_code="path_not_allowed"`, and the same guidance without persisting an Example.
- Any `path` that resolves outside `IMAGE_ROOT` MUST be rejected with `403`, except a disallowed item in the batch-ingestion API, which uses the same per-item `path_not_allowed` result described above.
- The browse endpoint MUST NOT follow symlinks that escape `IMAGE_ROOT`.
- The browse endpoint MUST NOT expose file contents, only names, types, paths, and sizes.
- The same root policy applies to every later read of a persisted
  `storage_ref`, including inference, embeddings, pHash, image serving,
  benchmark workloads, and dataset export. Persisting an allowed path does not
  preserve access after `IMAGE_ROOT` changes.

**Frontend integration:**

The frontend renders the browse endpoint's responses as an in-app file/folder browser (tree or column view). Its initial request MUST omit `path`, allowing the backend to return the deployment's effective image root without requiring the SME to know a server path. The user navigates directories, checks one or more folders or files, and clicks [Ingest Selected] to trigger ingestion. A text field for direct path entry is always available above the tree: pressing Enter navigates the tree via `GET /v1/filesystem/browse` to the typed path (no preview step). When the SME's [Ingest Selected] payload contains a directory entry, the frontend calls `POST /v1/filesystem/scan` internally to expand it into the per-image `(storage_ref, suggested_example_key)` rows that the ingestion endpoint (§10.2.1) requires; the scan response's `skipped[]` array is not rendered to the SME (F47, 2026-05-15). The file browser MUST persist recently used directory paths per project and show them as shortcuts for quick navigation on return visits. The file browser is browsing the backend host's filesystem; when the backend runs on a remote server, the browser navigates that server's directories. This distinction MUST be communicated in the UI (e.g., helper text: *"Browsing files on the server at {backend_host}."*).

#### 10.2.11 Storage path remapping endpoint (required)

When image files are moved (drive reorganization, NFS remount, container bind-mount change), persisted `storage_ref` values become invalid. This endpoint performs a bulk prefix replacement across all Example records in a project, with a mandatory dry-run preview to prevent destructive mistakes.

Recommended endpoint:

- `POST /v1/projects/{project_id}/examples:remap_paths`

Request:

```json
{
  "old_prefix": "/mnt/old_nas/images",
  "new_prefix": "/mnt/new_nas/images",
  "dry_run": true
}
```

Response (dry-run):

```json
{
  "dry_run": true,
  "matched_count": 1042,
  "sample_remappings": [
    {
      "example_key": "batch_01_img_001",
      "old_storage_ref": "/mnt/old_nas/images/batch_01/img_001.jpg",
      "new_storage_ref": "/mnt/new_nas/images/batch_01/img_001.jpg"
    }
  ],
  "unmatched_count": 58,
  "validation": {
    "sample_checked": 10,
    "sample_resolved": 8,
    "sample_missing": 2,
    "missing_examples": ["batch_01_img_404", "batch_01_img_405"]
  }
}
```

Response (commit):

```json
{
  "dry_run": false,
  "remapped_count": 1042,
  "audit_event_id": "string"
}
```

Rules:

- `old_prefix` and `new_prefix` MUST be non-empty absolute paths.
- The operation replaces `old_prefix` at the start of each `storage_ref` that begins with it. Examples whose `storage_ref` does not start with `old_prefix` are unaffected.
- **Dry-run is the default.** When `dry_run=true` (or omitted), the endpoint MUST NOT modify any records. It returns the count of matched examples, a sample of remappings (up to 10), and a validation summary.
- **Validation on dry-run:** the backend spot-checks a sample of remapped paths (up to 10) by verifying that the file at `new_storage_ref` exists and is readable. The response reports how many resolved and which were missing. This gives the user confidence before committing.
- **Commit mode (`dry_run=false`):** the backend MUST verify that at least one remapped path resolves to an existing file before applying. If zero sampled paths resolve, the request MUST be rejected with `400` and a message: *"None of the sampled remapped paths resolve to existing files. Verify new_prefix is correct."* On success, all matching `storage_ref` values are updated in a single transaction.
- The endpoint MUST create an AuditEvent with `event_type="storage_ref_remap"` and `event_data` containing `old_prefix`, `new_prefix`, `remapped_count`, and `timestamp`. This is persisted regardless of dry-run (for commit only).
- The operation does not re-validate image format or recompute pHash/CLIP embeddings. It is a path-only update. Downstream operations (image serving, labeling) will validate readability when they access the file.
- Project scoping is enforced: only Example records belonging to the requested `project_id` are affected.

#### 10.2.12 Model config re-probe endpoint (required)

When a NIM endpoint is updated (new model version, configuration change), persisted capability check results may be stale. This endpoint re-runs all capability checks for a model config entry.

Recommended endpoint:

- `POST /v1/projects/{project_id}/model_configs/{model_config_id}:reprobe`

Request: empty body.

Response:

```json
{
  "model_config_id": "string",
  "structured_generation_support": "supported | unsupported | unknown",
  "thinking_toggle_support": "supported | unsupported | unknown",
  "visual_budget_support": "supported | unsupported | unknown"
}
```

Rules:

- The endpoint MUST reject the request with `409 Conflict` if the model config is currently referenced by a `queued`, `running`, or `canceling` evaluation/Batch run, or by an active training job. A paused Batch run uses its persisted runtime snapshot and does not block re-probe.
- On invocation, reset `structured_generation_support`, `thinking_toggle_support`, and `visual_budget_support` to `unknown`, then re-run: the structured-generation probe (§6.2), the thinking override acceptance check (§6.7.4, only when `thinking_toggle.mode` is request-based), and the visual-budget probe (§6.9.2).
- Each check runs independently; a failure in one MUST NOT prevent the others from executing.
- Probe results are persisted on the ModelConfig as usual. The response returns the updated values.
- The operation is bounded by the enforced deadline per probe. If a probe times out, the corresponding field remains `unknown`.

#### 10.2.13 Project list endpoint (required)

Recommended endpoint:

- `GET /v1/projects`

List all projects with summary counts for the project list home screen.

Request: cursor pagination (`limit`, `cursor`) plus optional `include_archived: bool = false`. When `include_archived=false` (default) the list excludes any project whose `archived_at` is non-null. When `true`, archived projects are returned alongside active ones with `archived_at` set in the response item.

For performance, the backend MUST short-circuit the per-project DB open when a sentinel marker file `{project_dir}/.archived` is present and `include_archived=false` — the marker is the lazy index over the `archived_at` column. The DB column remains the source of truth; on any drift the column wins, a warning is logged, and the marker is rewritten (or removed) to match the column so subsequent scans stop paying the DB open.

The response also carries a workspace-global `has_archived: bool` — true when at least one project directory carries the `.archived` marker, regardless of the `include_archived` filter or the current page. It is computed from the marker files alone (no DB opens), and it exists so the Project List screen can decide whether to render the "Show archived" affordance without issuing a second, archived-inclusive fetch that would open every project DB in the workspace.

Response:

```json
{
  "items": [
    {
      "project_id": "string",
      "name": "string",
      "description": "string | null",
      "created_at": "timestamp",
      "updated_at": "timestamp",
      "archived_at": "timestamp | null",
      "counts": {
        "verified": "int",
        "unlabeled": "int",
        "auto_labeled": "int",
        "omitted": "int",
        "pending_relabel": "int"
      }
    }
  ],
  "next_cursor": "opaque-string-or-null",
  "has_archived": "bool"
}
```

`pending_relabel`: count of examples awaiting re-labeling after a semantic Core schema change. Query: `COUNT(Example WHERE project_id=:pid AND state='Unlabeled' AND prior_verified_label_ref IS NOT NULL)`. Zero until a schema change returns previously Verified examples to Unlabeled with prior-label references. Decreases as examples are re-labeled.

##### 10.2.13.1 Project archive / unarchive endpoints (required)

Soft-archive a project so it is hidden from the default list (the Project List screen's archived-card behavior, `ProjectListPage.tsx`). Endpoints follow the verb-suffix pattern used by `tao_jobs:cancel`:

- `POST /v1/projects/{project_id}:archive` — sets `archived_at = utc_now()`, writes the `.archived` marker, persists an `AuditEvent(event_type="project_archived")`, and emits an SSE `project_archived` event. Returns the updated `ProjectResponse`.
- `POST /v1/projects/{project_id}:unarchive` — clears `archived_at`, removes the marker, persists an `AuditEvent(event_type="project_unarchived")`, and emits an SSE `project_unarchived` event. Returns the updated `ProjectResponse`.

**Busy gate (mandatory before archive).** The reentrant in-process file lock cannot detect "this project is currently busy", so an application-level check is required. The backend MUST reject `:archive` with **409 `project_busy`** when ANY of the following holds:

- `RunRecord` exists with `status NOT IN {succeeded, completed, incomplete, failed, canceled}`
- `TAOJob` exists with `status NOT IN {succeeded, failed, canceled, deleted}`
- `LocalNimDeployment` exists with `status IN {starting, running}`

The 409 body MUST carry a structured `reasons[]` array with diagnostic strings (e.g. `"2 evaluation/batch run(s) still in progress"`, `"1 TAO job(s) still in progress"`) so the UI surfaces the cause inline.

Other 409 codes:

- `already_archived` — `:archive` called on a project whose `archived_at` is already set.
- `not_archived` — `:unarchive` called on a project whose `archived_at` is null.
- `project_in_use` — another **process** holds the file lock (existing `ProjectLockedError` mapping).

**Cross-mutation guard.** Mutating endpoints MUST refuse work on archived projects with **409 `project_archived`**. v1 applies the `require_not_archived` dependency to:

- `PATCH /v1/projects/{project_id}`
- `POST /v1/projects/{id}/batch_label_runs`
- `POST /v1/projects/{id}/evaluation_runs`
- `POST /v1/projects/{id}/training_suites`

Other mutating endpoints (label save/skip, rationale regeneration, etc.) are reachable only from a labeling session that the operator cannot enter on an archived card (archived cards are non-clickable on the Project List screen, `ProjectListPage.tsx`); the four entry points above are the pragmatic v1 minimum.

**Background workers MUST skip archived projects.** Each project-directory enumeration loop checks `(entry / ".archived").exists()` before opening/locking the DB:

- `main._recover_interrupted_runs()` (startup recovery)
- `tao_polling_service.tick()` (periodic chain advancement)
- `clip_embedding_service.recover_embedding_tasks()` (startup)
- `local_nim_service.recover_local_deployments()` (startup) and the periodic GPU enumerator

This keeps recovery and polling cost O(active projects).

**SSE behavior.** A `project_archived` SSE event is emitted to subscribers of the project's stream after the DB commit. v1 does not auto-disconnect the subscribers; the typical archive flow is from the project list itself, where a query-cache invalidation refreshes the UI naturally. Cross-tab archive while another tab has the project open is acceptable v1 behavior.

#### 10.2.14 Label save and skip endpoints (required)

**Save label (Accept/Edit):**

- `POST /v1/projects/{project_id}/labels`

Create or promote a Label record with verification metadata. The backend computes the deterministic diff between the proposal and the submitted label to determine `verified_outcome` (Accept vs Edit) and populate `edited_core_fields[]` and `edited_aux_fields[]`. If the example was Auto-Labeled, the existing Label record is promoted from `auto_labeled` to `verified` (§4.7).

Request:

```json
{
  "example_key": "string",
  "inference_invocation_id": "string",
  "label_json": { "... field values matching SchemaCore ..." }
}
```

When rationale notes are enabled, the request additionally includes
`rationale_source` (`teacher_proposal`, `sme_edited`, or
`teacher_regenerated_approved`) and, for an approved regeneration,
`rationale_regeneration_invocation_id`. Both fields are optional in the API
shape because the default Guidance disables rationale notes.

Response:

```json
{
  "example_key": "string",
  "label_status": "verified",
  "verified_outcome": "Accept | Edit",
  "verified_at": "timestamp",
  "edited_core_fields": ["string"],
  "edited_aux_fields": ["string"],
  "pool_assignment": "test_pool | null"
}
```

Rules:

- Backend MUST compute the diff; the frontend sends only the final label.
- MUST trigger pool routing and rebalancing (§4.3).
- When the active Guidance enables rationale notes, MUST require and validate `rationale_source`; when `verified_outcome=Edit`, it MUST NOT be `teacher_proposal` (§4.4).
- When the active Guidance disables rationale notes, MUST strip any client-supplied `rationale_note`, ignore rationale provenance metadata, and persist null rationale provenance.
- When promoting an Auto-Labeled label, `batch_label_run_id` is retained on the Label record for lineage.
- **Save invocation validation:** the backend MUST validate that the submitted `inference_invocation_id`: (1) exists, (2) belongs to the same `project_id`, (3) pertains to the same `example_key`, and (4) is an allowed proposal source — `purpose=interactive_proposal` or the original `purpose=batch_label` invocation when the SME saves the surfaced Auto-Labeled proposal without Retry. If the invocation has been superseded by a later Retry proposal for the same example in the same review session, the backend MUST reject the save with `409 Conflict` and a stale-proposal error. This prevents the frontend from accidentally saving against an older proposal after the SME already reviewed a newer one.

**Skip (Omit):**

- `POST /v1/projects/{project_id}/examples/{example_key}:skip`

Transition the example to Omitted state. Sets `omitted_source="sme_skip"` and `omitted_at` (§4.5). If the Example is Auto-Labeled, the same transaction deletes its machine Label while preserving the batch-label Operation Record.

Request: empty body (optional `{ "reason": "string | null" }`).

Response:

```json
{
  "example_key": "string",
  "state": "Omitted",
  "omitted_at": "timestamp"
}
```

**Restore Omitted:**

- `POST /v1/projects/{project_id}/examples:restore_omitted`

Bulk-restore all Omitted examples to `state="Unlabeled"`, clearing `omitted_source` and `omitted_at` (§4.5). A discarded Auto-Labeled proposal is not restored.

Request: empty body.

Response:

```json
{
  "restored_count": "int"
}
```

Rules:

- Available when the review selector queue is empty and Omitted examples exist.
- Restored examples re-enter the review selector.

#### 10.2.15 Rationale regeneration endpoint (required)

Recommended endpoint:

- `POST /v1/projects/{project_id}/examples/{example_key}:regenerate_rationale`

Call the Teacher with the image and active task context to produce a fresh, independently observed rationale (Appendix D.3). The request carries no original, corrected, or reviewed field values. Creates an Operation Record with `purpose="rationale_regeneration"`.

Request:

```json
{
  "teacher_model_config_id": "string | null"
}
```

Response:

```json
{
  "inference_invocation_id": "string",
  "rationale_note": "string",
  "invocation_status": "success | timeout | endpoint_error"
}
```

Rules:

- Available only while the active Guidance enables rationale notes; otherwise returns `409 Conflict` without dispatching a Teacher request.
- When `teacher_model_config_id` is null, uses `project.teacher_model_config_id`.
- On failure, the SME writes the rationale directly (§6.3.2).

#### 10.2.16 Evaluation run endpoints (required)

Recommended endpoints:

- `POST /v1/projects/{project_id}/evaluation_runs` — trigger evaluation
- `GET /v1/projects/{project_id}/evaluation_runs/{run_id}` — get run status + metrics
- `GET /v1/projects/{project_id}/evaluation_runs` — list runs
- `POST /v1/projects/{project_id}/evaluation_runs/{run_id}:cancel` — cancel run

**Trigger evaluation:**

Request:

```json
{
  "icl_mode": "enabled | disabled",
  "structured_generation_mode": "auto | prompt_only | null"
}
```

`icl_mode` defaults to `"enabled"`. `structured_generation_mode` defaults to `"auto"`. All other configuration is read from current project settings and snapshotted on the Run Record (§7.1).

Response:

```json
{
  "run_id": "string",
  "run_type": "evaluation_run",
  "status": "queued",
  "pool_version": "int",
  "guidance_id": "string",
  "model_config_id": "string",
  "generation_preset_key": "string",
  "thinking_mode_effective": "on | off",
  "visual_budget_preset_key": "string | null",
  "structured_generation_mode_effective": "auto | prompt_only",
  "created_at": "timestamp"
}
```

Rules:

- If a gate-basis evaluation is already running and the new run is also gate-basis, the in-progress run MUST transition to `canceling` → `canceled` with `status_reason="superseded_by_newer_evaluation"`. Student benchmark runs (`student_model_config_id` set) neither fire this supersede nor receive it (§7.1).
- If the Test Pool has no members, MUST reject with `400` and a message (§4.3.3).

**Get run status + metrics:**

Response:

```json
{
  "run_id": "string",
  "run_type": "evaluation_run",
  "status": "queued | running | canceling | completed | incomplete | canceled | failed",
  "status_reason": "string | null",
  "pool_version": "int",
  "guidance_id": "string",
  "model_config_id": "string",
  "icl_mode": "enabled | disabled",
  "evaluation_source": "tao | nim",
  "progress": { "processed": "int", "total": "int" },
  "metrics": {
    "overall": {
      "exact_match_rate": "float",
      "per_field_match_rates": { "field_name": "float" },
      "per_value_metrics": {
        "field_name": { "value": { "precision": "float", "recall": "float", "f1": "float" } }
      },
      "example_count": "int"
    },
    "returning": "object | null",
    "new": "object | null"
  },
  "previous_overall_exact_match": "float | null",
  "coverage_gaps": [
    { "field_name": "string", "field_type": "string", "missing_values": ["string"] }
  ],
  "icl_eligible_count_at_completion": "int | null",
  "created_at": "timestamp",
  "started_at": "timestamp | null",
  "completed_at": "timestamp | null"
}
```

`metrics.returning` and `metrics.new` share the same shape as `metrics.overall`; null on the first evaluation (no previous snapshot to compare against).

**List runs:** cursor pagination (`limit`, `cursor`). Filter by `status`, and by provenance with `basis` (`gate` = gate-basis Teacher runs only — the evaluation strip's view; `benchmark` = Student benchmark runs only; omitted = all). Newest-first.

**Cancel:** transitions `running` → `canceling` → `canceled` (§13.2.1). Returns `409` if in a terminal state.

Request: empty body.

Response:

```json
{
  "run_id": "string",
  "status": "canceling",
  "cancel_requested_at": "timestamp"
}
```

#### 10.2.17 Scale-Up Readiness Gate endpoint (required)

Recommended endpoint:

- `GET /v1/projects/{project_id}/scaleup_gate`

Evaluate all five gate criteria (§7.3) and return structured status. Lightweight — queries persisted metrics and counts only, no model invocation.

Response:

```json
{
  "gate_status": "not_ready | ready",
  "criteria": [
    {
      "criterion_name": "string",
      "passed": "boolean",
      "current_value": "float | int",
      "threshold": "float | int",
      "message": "string",
      "details": "object | null"
    }
  ],
  "evaluated_at": "timestamp"
}
```

Rules:

- `criterion_name` values: `overall_exact_match`, `per_field_match`, `min_per_value_f1`, `accept_rate`, `min_test_pool_size`.
- `details` for `overall_exact_match`: `{ "no_completed_run": true }` when no completed NIM-sourced evaluation exists — the structural discriminator UIs MUST key "no evaluation yet" on (`current_value` is `0.0` in both the no-eval and genuine-0% cases, and `message` is display copy, not a wire contract). Omitted/`null` once a completed evaluation exists.
- `details` for `per_field_match`: `{ "failing_fields": [{ "field_name": "string", "current_rate": "float" }] }`.
- `details` for `min_per_value_f1`: `{ "failing_values": [{ "field_name": "string", "value": "string", "f1": "float", "precision": "float", "recall": "float" }] }`.
- `message` MUST be plain language; no MLOps jargon (§7.3.4).

#### 10.2.18 Dataset export endpoints (required when exports used)

Recommended endpoints:

- `POST /v1/projects/{project_id}/dataset_exports` — create export
- `GET /v1/projects/{project_id}/dataset_exports/{dataset_export_id}` — get export record
- `GET /v1/projects/{project_id}/dataset_exports` — list exports

**Create export:**

Request:

```json
{
  "dataset_intent": "training | evaluation | testing",
  "label_tier_filter": "verified_only | auto_labeled_only | combined",
  "export_field_mode": "all | aux_and_core | core_only",
  "batch_label_run_id": "string | null",
  "selection_filters": {
    "guidance_id": "string | null"
  }
}
```

Response (201 — the full export record, the same document `GET .../dataset_exports/{id}` returns; the archive builds in a background task):

```json
{
  "dataset_export_id": "string",
  "project_id": "string",
  "dataset_intent": "string",
  "export_field_mode": "string",
  "label_tier_filter": "string",
  "guidance_id": "string",
  "selection_definition_snapshot": "object (§13.5)",
  "example_count": "int",
  "status": "running | completed | failed",
  "status_reason": "string | null",
  "progress": { "images_written": "int", "images_total": "int" },
  "started_at": "timestamp | null",
  "completed_at": "timestamp | null",
  "artifact_refs": { "archive_path": "string", "annotations_path": "string", "checksum_sha256": "string" },
  "manifest_ref": "string | null",
  "created_at": "timestamp"
}
```

Rules:

- Export MUST create a DatasetExport record (§13.5).
- **Background build:** selection and validation run synchronously (validation failures return 4xx with no record); the create response then carries `status="running"` with `artifact_refs`/`manifest_ref` `null`. The multi-GB archive build runs as an in-process background task — the request MUST NOT block on it. `artifact_refs` and `manifest_ref` populate when the record reaches `completed`; consumers poll `GET .../dataset_exports/{id}` or follow the `export_progress` / `export_completed` / `export_failed` SSE events (§1.8). On restart, `running` exports are marked `failed` (`backend_restart_interrupted`) and partial artifact files deleted — exports are not resumable; the SME retries. At most one export builds per project at a time: a create request while another is `running` returns `409` (each build is a multi-GB job; an accidental re-request would silently double it).
- The training-suite creation path (§9.7.8) is exempt from the background build: it first persists a `preparing` TrainingSuite, builds exports in worker threads, and atomically commits each DatasetExport row with its suite link in a short-lived session before workspace upload (SQLite write-discipline, §1.8). Its records are created directly in `status="completed"`. TAOJob chain creation and the suite transition to `initialized` remain one transaction; a retryable pre-chain transfer failure retains the frozen linked exports for an exact idempotent retry.
- Evaluation and testing intents require `label_tier_filter="verified_only"`; requests for `auto_labeled_only` or `combined` are rejected as validation errors.
- After a semantic Core change, old labels are deleted (§4.4.1); only examples re-labeled under the current Guidance are exportable.
- Auto-Labeled exports include only `schema_valid_core=true` outputs (enforced at Label creation — schema-invalid outputs never produce Label records).
- Auto-Labeled and combined exports include a machine Label only when its exact owning Example remains `state="Auto-Labeled"`.
- Archive format: Cosmos-RL TAO-native (§9.3).

**List exports:** cursor pagination (`limit`, `cursor`). Filter by `dataset_intent`. Newest-first.

#### 10.2.19 Model catalog endpoints (required)

Recommended endpoints:

- `GET /v1/projects/{project_id}/model_configs` — list model configs
- `POST /v1/projects/{project_id}/model_configs` — create model config
- `PATCH /v1/projects/{project_id}/model_configs/{model_config_id}` — update model config

New evaluation and Batch Run Records persist a credential-free
`runtime_config_snapshot` containing every result-shaping model, endpoint, and
process-config value their inference path consumes. PATCH, capability re-probe,
and semantic process-config changes affect future runs; delayed execution,
restart recovery, and explicit Batch Resume continue from the run-owned
snapshot.

**List model configs:**

Request: `GET /v1/projects/{project_id}/model_configs?eligible_role=teacher&limit=50&cursor=<opaque>`

`eligible_role` filter is optional; when provided, returns only entries containing that role in `eligible_roles[]`. When filtering for `teacher`, results MUST be further restricted to `supports_image_input=true` (§4.8).

Response:

```json
{
  "items": [
    {
      "model_config_id": "string",
      "model_name": "string",
      "endpoint_id": "string",
      "context_window_tokens": "int",
      "eligible_roles": ["string"],
      "supports_image_input": "boolean",
      "structured_generation_support": "unknown | supported | unsupported",
      "thinking_toggle": { "mode": "none | qwen_enable_thinking | kimi_thinking" },
      "thinking_toggle_support": "unknown | supported | unsupported",
      "visual_budget_mode": "none | mm_processor_size | mm_processor_pixels | mm_processor_tiles",
      "visual_budget_support": "unknown | supported | unsupported",
      "default_icl_max_examples": "int | null",
      "tao_base_experiment_id": "string | null",
      "tao_base_experiment_pull_status": "unknown | starting | in_progress | pulling | pull_complete | invalid_pull | failed | null",
      "created_at": "timestamp"
    }
  ],
  "next_cursor": "opaque-string-or-null"
}
```

**F22 (Phase 12 amendment 2026-05-05):** `tao_base_experiment_id` and `tao_base_experiment_pull_status` are exposed on the GET response so operators can verify TAO base-experiment provisioning state (§9.7.8) without direct sqlite3 access to `project.db`. Both fields are read-only via this endpoint — `PATCH /model_configs` MUST continue to reject them as `extra_forbidden`. They are set by first-use automatic provisioning, the equivalent eager CLI, or the admin-managed bootstrap path. Both fields are `null` on non-`student_base` ModelConfigs.

**Create model config:**

Request:

```json
{
  "endpoint_id": "string",
  "model_name": "string",
  "context_window_tokens": "int",
  "eligible_roles": ["string"],
  "supports_image_input": "boolean",
  "thinking_toggle": { "mode": "none | qwen_enable_thinking | kimi_thinking" },
  "visual_budget_mode": "none | mm_processor_size | mm_processor_pixels | mm_processor_tiles"
}
```

Rules:

- `model_config_id` MUST be backend-generated (§4.8).
- `eligible_roles` MUST be non-empty. Only `nvidia/cosmos-reason2-8b` and `nvidia/cosmos-reason2-2b` may have `student_base` role.
- Capability fields (`structured_generation_support`, `thinking_toggle_support`, `visual_budget_support`) initialize to `unknown`; updated by probes.

**Update model config:** partial update. `model_config_id` and `model_name` are immutable. MUST validate role and endpoint constraints (§4.8).

#### 10.2.20 Student model endpoints (required when Student Training enabled)

Recommended endpoints:

- `GET /v1/projects/{project_id}/student_models` — list student models
- `GET /v1/projects/{project_id}/student_models/{student_model_id}` — get student model
- `POST /v1/projects/{project_id}/student_models/{student_model_id}:deploy_nim` — trigger NIM deployment for evaluation
- `POST /v1/projects/{project_id}/student_models/{student_model_id}:repackage` — replay checkpoint packaging after an environment fix (F-B10 amendment 2026-07-15; mirrors the F33 `:rerescore` rationale for the packaging stage — canonical case: adapter-only LoRA output with the merge interpreter unprovisioned. 409 unless `checkpoint_packaging_status="failed"`; variant-aware: quantized students replay from their quantize TAOJob)

**List student models:** cursor pagination (`limit`, `cursor`). Response returns StudentModel records per §13.13, including nullable `training_suite_id` for immutable run grouping, plus derived `serving_benchmark_current: bool` and nullable `serving_benchmark_blocker` fields computed from the referenced serving run and current configured concurrency contract.

**Get student model:** full StudentModel record including checkpoint status, quality/serving readiness, current-benchmark assessment, NIM deployment state, and evaluation run links.

**Deploy NIM:**

Request:

```json
{
  "nim_endpoint_url": "string | null"
}
```

When `nim_endpoint_url` is provided (Tier 2 — external deployment), the system skips local container orchestration and runs NIM evaluation against that endpoint. When null, the system attempts local orchestration (Tier 1; §9.5.2).

Response:

```json
{
  "student_model_id": "string",
  "nim_preflight_status": "passed | failed | null",
  "nim_preflight_details": "object | null",
  "nim_deployment_mode": "local | external | null",
  "status": "deploying | preflight_failed | evaluating"
}
```

Rules:

- When `nim_endpoint_url` is null and preflight fails, returns `nim_preflight_status="failed"` with `nim_preflight_details` so the frontend can generate an Action Request (§10.3).
- Progress is reported via SSE events. Container lifecycle events are logged on the StudentModel record (§9.5.2).
- Variants are deployed sequentially; each container is stopped before the next starts.

#### 10.2.21 Evaluation suite endpoints (Removed in v1.0)

**Removed in v1.0** with the evaluation-suite subsystem (§7.2 note). The Compare & Benchmark screen reads `GET .../evaluation_runs` for the Teacher baseline and `GET .../evaluation_runs/{run_id}` for each Student's quality and serving runs. Section number retained so existing cross-references stay resolvable.

#### 10.2.22 Training diagnostics and preset-resolution endpoints

Authoritative readiness endpoint used by the Scale-Up Hub, Student Training
screen, operators, and API clients:

- `POST /v1/projects/{project_id}/training_preflight`

This endpoint exposes the same TAO/workspace/timeout/role/data checks enforced
inside `POST .../training_suites`.

Request:

```json
{
  "student_base_model_config_ids": ["string"],
  "include_auto_labeled": true,
  "enable_lora": true,
  "quantization_schemes": ["FP8_DYNAMIC"]
}
```

The read-only Advanced disclosure uses:

- `POST /v1/projects/{project_id}/training_presets:resolve`

It accepts the same `student_base_model_config_ids[]` request shape and returns
only `resolved_presets`. It performs no TAO or workspace check.

Response:

```json
{
  "status": "passed | failed",
  "checks": [
    {
      "check_name": "tao_reachable | tao_job_timeout_supported | tao_workspace_reachable | tao_base_experiment_ready | hf_token_configured | lora_merge_runtime | student_base_role | verified_train_examples | min_test_pool_size",
      "passed": "boolean",
      "message": "string",
      "model_config_id": "string | null",
      "provisioning_required": "boolean"
    }
  ],
  "data_summary": {
    "verified_training_count": "int",
    "test_pool_count": "int",
    "required_test_pool_count": "int",
    "auto_labeled_eligible_count": "int",
    "auto_labeled_included_count": "int",
    "excluded_test_pool_count": "int",
    "excluded_auto_labeled_count": "int",
    "usable_training_count": "int"
  },
  "resolved_presets": {
    "<model_config_id>": {
      "quick | standard | high_quality | max_quality": { "train": { "epoch": "int", "resume": "boolean", "ckpt": { "...": "..." } } }
    }
  }
}
```

`resolved_presets` carries the server-resolved §9.7.3.1 hyperparameter patches for each requested base model. The Training screen's Advanced expander MUST render these values verbatim — the backend resolver (`services/training_preset.py`) is the single source of truth; the frontend MUST NOT recompute them (a former frontend mirror drifted on `max_keep` and the cosmos3-super epoch schedule).

Check semantics (§9.7.8.3):

- `tao_reachable`: TAO API connection probe succeeds.
- `tao_workspace_reachable`: `TAODeploymentConfig.bootstrap_status == "bootstrapped"` AND `GET /api/v2/orgs/{org}/workspaces/{workspace_id}` returns 200. If the deployment has not been bootstrapped, this check fails with a plain-language next step referencing `vlm-feedback-loop tao-bootstrap` (§9.7.8.1a).
- `student_base_role`: the ModelConfig has `student_base` in `eligible_roles[]`.
- `tao_base_experiment_ready`: a ready ModelConfig has non-null `tao_base_experiment_id` + `pull_complete` and returns `provisioning_required=false`. A valid missing or incomplete base is non-blocking and returns `passed=true`, `provisioning_required=true`; Start Training must complete the tracked provisioning endpoint before suite creation.
- `verified_train_examples`: the §9.3 training-export selection is non-empty — at least one Verified label under the active Guidance with no Test Pool assignment (Test Pool members are evaluation-only per §4.3). Global check (`model_config_id: null`). On failure the message tells the SME to continue labeling; the Scale-Up screen renders this failure as "No Verified training examples yet. Continue labeling." instead of a TAO-setup call to action.
- `min_test_pool_size`: the active-Guidance Test Pool count is at least
  `max(1, project.scaleup_min_test_pool_size)`. The default is 60. This is a
  Student dataset-readiness check for the automatic baseline and quantized
  evaluation jobs; it does not import the Scale-Up gate's Teacher accuracy,
  per-field, per-value F1, or Accept-rate criteria. On failure, both UI
  consumers direct the SME to continue labeling and MUST NOT offer a TAO setup
  Action Request.
- `data_summary`: authoritative distinct-example counts under the active
  Guidance and request's Auto-Labeled policy. Test Pool members are always
  excluded from training and reported separately.
  `required_test_pool_count` is the backend-resolved effective minimum;
  `usable_training_count` plus both dataset checks control whether Start can
  be enabled.

Checks run independently — a failure in one does NOT prevent the others from executing, matching the capability-probe pattern in §10.2.12.

#### 10.2.22a TAO base-experiment provisioning endpoints

- `POST /v1/projects/{project_id}/tao_base_experiment_provisioning` — body:
  `{"student_base_model_config_ids": ["string"]}`; returns 202 with a
  `TAOBaseExperimentProvisioningRun`. Ready bases are filtered out and an
  all-ready selection returns an immediately `succeeded` run. The endpoint
  rejects missing workspace/S3/HF credentials before queuing work.
- `GET /v1/projects/{project_id}/tao_base_experiment_provisioning/{run_id}` —
  returns the same durable run with status
  `queued | running | succeeded | failed`.

Only project-local ModelConfig ids carrying `student_base` are accepted. The
underlying TAO resource is deployment-scoped, so only one provisioning run may
be active across projects. A repeated request whose missing targets are a
subset of the active run returns that run; a request requiring additional
targets returns 409 until the active run finishes. Errors and per-target
failures MUST be redacted before persistence or response.

#### 10.2.23 Action request endpoints (required)

Recommended endpoints:

- `POST /v1/projects/{project_id}/action_requests:generate` — generate pre-filled request
- `POST /v1/projects/{project_id}/action_requests:log_copy` — log clipboard copy for audit

**Generate:**

Request:

```json
{
  "request_type": "tao_setup | nim_setup | nim_issue | missing_files | student_nim_deploy | tao_issue | deployment_handoff",
  "context": {
    "student_model_id": "string | null",
    "example_keys": ["string"] | null,
    "error_ref": "string | null",
    "tao_job_id": "string | null"
  }
}
```

Response:

```json
{
  "request_type": "string",
  "generated_at": "timestamp",
  "project_name": "string",
  "technical_requirements": "object",
  "current_environment": "object",
  "rendered_text": "string"
}
```

`rendered_text` is the pre-formatted message ready for clipboard copy. MUST NOT contain secrets (§10.3.2).

Rules for `deployment_handoff`: MUST reject with `409` if the Student does not have `quality_status=validated`, `serving_status=validated`, and a referenced serving run satisfying the current AIPerf workload contract (§9.5.2, §13.13). A historical synthetic result returns `conflict: serving_benchmark_requires_aiperf`. The `technical_requirements` payload includes the deployment-specific fields defined in §10.3.3. The portable deployment bundle applies the identical gate.

**Log copy:**

Request:

```json
{
  "request_type": "string",
  "rendered_text": "string"
}
```

Response:

```json
{
  "audit_event_id": "string"
}
```

Creates an AuditEvent with `event_type="action_request_copied"` (§10.3.4).

#### 10.2.24 Batch labeling manifest download (required)

Recommended endpoint:

- `GET /v1/projects/{project_id}/batch_label_runs/{run_id}/schema_invalid_manifest`

Return a downloadable manifest listing all examples with schema-invalid Core output in this batch labeling run.

Response (JSON or file download with `Content-Disposition: attachment`):

```json
{
  "batch_label_run_id": "string",
  "schema_invalid_examples": [
    {
      "example_key": "string",
      "validation_errors_core": ["string"],
      "inference_invocation_id": "string"
    }
  ],
  "total_count": "int"
}
```

#### 10.2.25 Environment assessment endpoint (required)

Deployment-scoped (not project-scoped). Returns the environment assessment (§1.5 Mode C) used by the onboarding recommendation screen.

Recommended endpoint:

- `GET /v1/environment` — reuse the process-local machine snapshot
- `GET /v1/environment?refresh_hardware=true` — explicitly rebuild the Docker/toolkit/GPU snapshot before responding

Response:

```json
{
  "hosted_nim_available": "boolean",
  "local_deploy_available": "boolean",
  "docker_available": "boolean",
  "nvidia_toolkit_available": "boolean",
  "nvidia_api_key_configured": "boolean",
  "ngc_api_key_configured": "boolean",
  "gpus": [
    {
      "name": "string",
      "memory_total_gb": "float",
      "compute_capability": "float | null"
    }
  ],
  "local_deployable_models": [
    {
      "model_name": "string",
      "model_config_id": "string",
      "nim_container_image": "string",
      "gpu_memory_minimum_gb": "int",
      "compute_capability_minimum": "float | null",
      "fits": "boolean"
    }
  ],
  "embedding_deployment": {
    "model_name": "string",
    "nim_container_image": "string",
    "gpu_memory_minimum_gb": "int",
    "fits": "boolean",
    "provider": "string"
  },
  "missing_prerequisites": [
    { "check": "string", "install_hint": "string" }
  ],
  "recommended_teacher_mode": "hosted | local | none",
  "recommended_embedding_mode": "hosted | local | none",
  "recommended_local_teacher_model_name": "string | null",
  "recommended_local_teacher_image": "string | null",
  "recommended_local_teacher_gpu_memory_minimum_gb": "int | null",
  "active_local_nim_residents": [
    {
      "project_id": "string",
      "project_name": "string",
      "local_nim_deployment_id": "string",
      "role": "teacher | embedding | student",
      "model_name": "string | null",
      "nim_container_image": "string",
      "gpu_assignment": "device=N",
      "status": "starting | running"
    }
  ]
}
```

Rules:

- MUST NOT return secret values (API keys). Only reports whether each key is configured.
- GPU detection uses `nvidia-smi`; unavailable or failing `nvidia-smi` produces an empty `gpus` array.
- `local_deployable_models` includes only seeded catalog entries with `local_deploy_metadata` (Teacher/Student VLM models from the model catalog).
- `embedding_deployment` is sourced from `EmbeddingDeploymentConfig` (§13.17), not from the model catalog. Its `fits` flag additionally requires a currently claimable GPU whose detected name exactly matches the pinned support matrix. The embedding NIM is infrastructure for background embedding computation and is configured separately from user-selectable inference models.
- An exact compatible `running` Teacher yields `recommended_teacher_mode="local"` when it matches the current quality recommendation, even when `NVIDIA_API_KEY` is configured. A different resident remains in `active_local_nim_residents` for the explicit keep/replace choice. Without a preferred resident, an API key remains the hosted-primary recommendation.
- Docker, NVIDIA Container Toolkit, and GPU inventory are cached for the backend process lifetime. Credentials, embedding configuration, and active NIM residents are recomposed from current state on every call.
- `refresh_hardware=true` invalidates and replaces that machine snapshot. It is intended for operator-driven host changes; deployment preflight independently performs fresh live checks.

#### 10.2.26 NIM endpoint endpoints (required)

Recommended endpoints:

- `GET /v1/projects/{project_id}/nim_endpoints` — list endpoints
- `POST /v1/projects/{project_id}/nim_endpoints` — create endpoint
- `GET /v1/projects/{project_id}/nim_endpoints/{endpoint_id}` — get endpoint
- `PATCH /v1/projects/{project_id}/nim_endpoints/{endpoint_id}` — update endpoint
- `POST /v1/projects/{project_id}/nim_endpoints:configure_self_hosted_teacher` — verify, bind, and select an exact cataloged self-hosted Teacher

There is no standalone probe verb: endpoint health is auto-probed on create and update.

Endpoint updates affect future evaluation and Batch runs. Existing runs use
their credential-free `runtime_config_snapshot`, including endpoint URL, mode,
authentication mode, and resolved image cap; credentials themselves remain at
the deployment boundary and are never copied into a Run Record.

**Create endpoint:**

Request:

```json
{
  "display_name": "string",
  "endpoint_mode": "hosted | self_hosted | local_system_managed",
  "base_url": "string",
  "auth_mode": "bearer | none"
}
```

Response: full NimEndpoint record per §13.18.

Rules:

- `endpoint_id` MUST be backend-generated UUID4.
- On create, the system MUST run a connection probe automatically (`GET {base_url}/v1/models`) and persist `last_probe_at`, `last_probe_status`, `last_probe_error_ref`.
- `auth_mode="bearer"` (hosted) resolves the deployment-scoped `NVIDIA_API_KEY` at probe time (§12.1); `auth_mode="none"` (self-hosted / local) sends no credential — self-hosted NIMs are expected to run on a trusted private network or behind an external gateway.

**Update endpoint:**

Partial update. `endpoint_id` is immutable. Accepts `display_name`, `base_url`, `auth_mode`, `is_enabled`. On update, the system re-probes automatically.

**Configure a self-hosted Teacher:**

Request:

```json
{
  "base_url": "http://nim.internal:8000/v1",
  "model_config_id": "uuid"
}
```

The backend normalizes a credential-free HTTP(S) URL, rejects embedded
credentials/query/fragment data, confirms `GET {base_url}/models` reports the
exact selected vision-Teacher model, then creates or idempotently reuses a
`self_hosted` endpoint. It binds that endpoint to the selected ModelConfig,
selects it on the Project, and re-probes generation capabilities. A connection
or model mismatch MUST leave the existing project binding unchanged. Repeating
the same successful request MUST reuse the endpoint rather than duplicate it.

**Auto-probe semantics** (runs on create and update; there is no callable probe route):

- The probe tests `GET {base_url}{models_path}` (default `/models`, because
  `base_url` already includes `/v1`). For `local_system_managed` endpoints, it
  also tests `GET {base_url}{health_ready_path}` (default `/health/ready`).
- Successful `GET /v1/models` → `healthy`; 401/403 → `auth_failed`; connection failure → `unreachable`; other errors → `unhealthy`.
- Probe results are persisted on the NimEndpoint record (`last_probe_at`, `last_probe_status`, `last_probe_error_ref`) and returned as part of the create/update response.
- Bounded by `HTTP_DEADLINE_INTERACTIVE_S`.

**Transient connection test (deployment-scoped):**

- `POST /v1/nim/test_connection`

A deployment-scoped proxy for browser-safe endpoint testing. The frontend cannot call NIM directly due to CORS and secret exposure.

Request:

```json
{
  "base_url": "string",
  "auth_mode": "bearer | none",
  "credential_transient": "string | null",
  "probe_kind": "models | embeddings"
}
```

Response:

```json
{
  "success": "boolean",
  "models": ["string"] | null,
  "error": "string | null"
}
```

Rules:

- `probe_kind=models` tests `GET {base_url}/v1/models`; `probe_kind=embeddings` tests a lightweight `/v1/embeddings` call.
- `credential_transient` is held in request memory only, used for this single probe, then discarded. It MUST NOT be written to `.env`, YAML config, SQLite, browser storage, or any project-scoped record (§12.1).
- Bounded by `HTTP_DEADLINE_INTERACTIVE_S`.

### 10.3 Action Requests (Structured Handoff)

#### 10.3.1 Purpose

When the SME is blocked by an infrastructure dependency (self-hosted NIM needed, TAO endpoint not configured), the system generates a pre-filled, read-only message containing the technical requirements and current diagnostic context. The SME copies it to clipboard and pastes into their organization's preferred channel.

Action Requests are NOT a ticketing system. The system does not track request state, integrate with external issue trackers, or notify recipients. It produces a message; where it goes is the organization's choice.

#### 10.3.2 Request Content

Every Action Request is a single read-only block containing:

- `request_type`: identifies the infrastructure need (see §10.3.3)
- `project_id` and `project_name`
- `generated_at`: timestamp
- `technical_requirements`: type-specific structured data (see §10.3.3)
- `current_environment`: relevant detected environment state (e.g., TAO connection status, endpoint probe results)

The rendered output MUST NOT contain secrets (API keys, credentials).

#### 10.3.3 Request Types and Pre-filled Content

**`tao_setup`: TAO Endpoint Configuration**

Triggered from: Scale-Up hub or Student Training when TAO infrastructure is
not configured/reachable, and from operator/API workflows handling a
suite-launch or first-use provisioning prerequisite failure (§9.7.8).

Pre-filled technical requirements:

- Required settings: `TAO_API_BASE_URL`, `TAO_API_KEY`, and `TAO_ORG_NAME` (§1.6). The deployment must also be bootstrapped so workspace identity is present in `deployment.db`.
- Selected Student base model(s), current `tao_base_experiment_id` values, and whether automatic provisioning was attempted (§4.8, §9.7.8).
- Connection test endpoint: `GET /api/v2/orgs/{org_name}/jobs?limit=1`.
- Workspace check: `GET /api/v2/orgs/{org_name}/workspaces/{workspace_id}` MUST return 200.
- Base-experiment check: suite creation requires each selected base to exist on TAO with local status `pull_complete`; ordinary missing bases are handled automatically before this point.
- Bootstrap pointer: reference `vlm-feedback-loop tao-bootstrap` for the workspace, the automatic **Start Training** path for ordinary missing bases, `tao-pull-base-experiments` / `--eager-bases` for operator warming, and `docs/tao-ftms-install.md` for air-gapped registration.
- Current probe result (if attempted): error details (sanitized).

**`nim_setup`: Self-Hosted NIM Endpoint**

Triggered from: Onboarding step 2 (self-hosted NIM toggle).

Pre-filled technical requirements:

- Required NIM endpoint config: base URL (including `/v1`), reachable from the backend on a trusted private network
- Current catalog model options, with explicit guidance that only one is needed
- Hosting-team NGC Personal API Key requirement for pulling the NIM container and model artifacts; credentials are never requested by email
- Known GPU memory minimums from local deployment metadata, with support-matrix sizing guidance for other catalog models
- Verification endpoints: `GET /v1/models` and a vision request to `POST /v1/chat/completions`

**`nim_issue`: NIM Endpoint Unreachable**

Triggered from: labeling screen proposal failure (endpoint error).

Pre-filled technical requirements:

- NIM endpoint base URL and model name
- Sanitized error details from the failed invocation
- Timestamp of the failure
- Suggested diagnostic: `GET {base_url}/v1/models`

**`missing_files`: Image Files Not Found**

Triggered from: labeling screen missing image (§7.5).

Pre-filled technical requirements:

- Expected `storage_ref` path(s) that failed to resolve
- Project name
- Suggested fix: restore files to the expected paths, or run bulk path remapping (`POST /v1/projects/{project_id}/examples:remap_paths`, §10.2.11) if files were moved to a new location

**`student_nim_deploy`: Student NIM Deployment for Evaluation**

Triggered from: Compare & Deploy screen when NIM deployment preflight fails (§9.5.2).

Pre-filled technical requirements:

- Exact `docker run` command with all non-secret flags pre-filled: NIM VLM container image and pinned release, `NIM_MODEL_NAME` (checkpoint path), `NIM_SERVED_MODEL_NAME`, name-only `-e NGC_API_KEY` forwarding, cache mount (`~/.cache/nim:/opt/nim/.cache`), checkpoint mount, `--shm-size=32GB`, `-u $(id -u)`, `-p 8000:8000`, `--runtime=nvidia --gpus all`
- Host prerequisites: Linux OS (Ubuntu 22.04 LTS), NVIDIA driver 580.65.06+, Docker 29.4.0+, NVIDIA Container Toolkit 1.19.0+, no vGPU
- GPU memory requirements per variant (from support matrix)
- Checkpoint path and expected directory structure
- Health check: `GET {base_url}/v1/health/ready`, `GET {base_url}/v1/models`
- Smoke test: minimal `POST /v1/chat/completions` request
- NIM startup note: *"Startup may take several minutes while NIM builds runtime artifacts. Use a persistent cache mount to avoid rebuilding on restart."*
- Temporary infrastructure note: *"This endpoint is for evaluation only. It can be stopped after results are collected."*
- Preflight diagnostic details (which checks failed and why)

**`tao_issue`: TAO Job Failure**

Triggered from: Training Job Monitor when a remote TAO job fails
(`TrainingJobMonitorPage.tsx`). A Blueprint-local `student_nim_local`
evaluation failure is not a TAO issue: its job card links to Compare &
Benchmark, where the persisted Student exposes the serving-validation retry
and preflight Action Request paths.

Pre-filled technical requirements:

- TAO endpoint URL and organization name
- Job ID, action (`train`, `evaluate`, `quantize`), and base model
- Sanitized error details from the failed job
- Job configuration summary (preset, epoch, dataset size)
- Suggested diagnostic: check TAO logs, verify GPU memory/disk, confirm driver compatibility

**`deployment_handoff`: Student Production Deployment**

Triggered from: Compare & Benchmark screen per-variant [Request Production Deployment] button (`CompareBenchmarkPage.tsx`). Requires `quality_status=validated`, `serving_status=validated`, and current AIPerf evidence on the referenced serving RunRecord. A pre-AIPerf historical result instead renders [Revalidate with AIPerf]. The label's "Production" qualifier disambiguates this from the same screen's [Deploy for serving validation] fallback affordance, which dispatches the gate-less `student_nim_deploy` AR for temporary evaluation infrastructure.

Pre-filled technical requirements:

Checkpoint and NIM configuration:

- `nim_checkpoint_ref` (path/URI to validated, NIM-loadable checkpoint)
- `nim_model_name_recommended` (recommended `NIM_MODEL_NAME` value)
- `nim_model_profile_recommended` (recommended `NIM_MODEL_PROFILE`; nullable for custom-checkpoint deployments)
- `nim_backend` (e.g., `"vllm"`)
- `nim_vlm_release_version` (pinned NIM release)
- `nim_env_vars_recommended` (any additional recommended NIM environment variables)

Model metadata:

- `student_base_model_config_id` (base model identity)
- `quantization_method` (e.g., `"FP8_DYNAMIC"`, `"W4A16"`, or `null` for full-precision)
- `tensor_parallelism`
- `gpu_requirements` (from support matrix, e.g., `"8× A100 80 GB"`)

Evaluation snapshot:

- `evaluation_summary` (at minimum: overall Exact Match rate, per-core-field match rates, per-value precision/recall/F1 for categorical Core fields, latency p50/p90/p99, pool version used, ICL mode, Guidance version)
- Test Pool dataset SHA-256 from the held-out export paired with the
  artifact-producing train/quantize job (never inferred from the Student's
  training-only `dataset_export_ids[]`)

Training lineage:

- `tao_job_id` (training job that produced the checkpoint)
- `quantize_tao_job_id` (quantization job, if quantized)
- `dataset_export_ids[]` (training data used)
- `training_preset`
- `lora_config` (from TAOJob)

Multiple `deployment_handoff` Action Requests MAY exist per project (different variants, redeployments).

After the Action Request is generated successfully, the inline panel MUST
offer **Download portable NIM deployment bundle** as its primary action and retain
**Copy to Clipboard** as the secondary operator-handoff action. The download
uses the streamed §9.5.1 endpoint so multi-gigabyte checkpoints are never
buffered in browser JavaScript memory.

#### 10.3.4 UX Contract

- Action Request CTAs appear inline on the screen where the blocker occurs, not in a separate settings area.
- The CTA is contextual: "Request TAO Setup", "Request NIM Setup", "Report NIM Issue", "Report Missing Files", "Deploy for serving validation", "Report TAO Issue", or "Request Production Deployment" — not a generic link. The two "deployment" labels intentionally differ: "Deploy for serving validation" is the imperative-action affordance for the temporary-evaluation `student_nim_deploy` AR (Compare & Benchmark preflight fallback); "Request Production Deployment" is the request-to-ship affordance for the gated `deployment_handoff` AR (Compare & Benchmark dual-validated variants). The "Production" qualifier on the latter is what separates the two flows for screenshots, audit logs, and tooltips that lift the label out of card-level context.
- Clicking the CTA shows the pre-filled content (inline expansion or popover) with a **Copy to Clipboard** button. No form fields; the SME adds their own context in the message they send.
- After copy, the system logs an AuditEvent (`event_type="action_request_copied"`) with the pre-filled content in `event_data` for audit trail.

### 10.4 Prompt Package Export (Removed in v1.0)

**Absent from v1.0.** Prompt Package export was a proposed direct "deploy Teacher as-is" path. A static `.zip` could not reproduce the live loop's per-query ICL example selection (§6.2), which is central to labeling quality, and the one-NIM-per-GPU rule (§1.5) makes a standing deployed Teacher expensive. The public schema has no `prompt_packages` table. Section number retained so existing cross-references stay resolvable.

## 11. Error Handling, Retries, and Idempotency

**Deadlines**

- Every external invocation (Teacher/Student model calls; TAO API calls) MUST enforce a finite deadline.
- Default deadlines (configurable): interactive proposals `HTTP_DEADLINE_INTERACTIVE_S` (default: 180s), evaluation / batch labeling / embedding `HTTP_DEADLINE_BACKGROUND_S` (default: 300s). Longer background deadlines accommodate hosted NIM endpoints that may exhibit higher latency under load. The 180s interactive deadline accommodates thinking-mode models whose reasoning phase can add ~60–90s to a single proposal.

**Retry classes (distinct semantics)**

1. **Automatic retry (system-initiated):**
  - MUST be bounded: at most `HTTP_MAX_RETRIES` attempts (default: 3).
  - MUST occur within a single `inference_invocation_id` attempt.
  - MUST NOT create additional Operation Records.
  - MAY include prompt budget fallback retries (e.g., diversity-based ICL pruning on context overflow or increasing `max_output_tokens` on truncation), but remains bounded and within the same invocation.
  - **Retryable errors:** `429`, `502`, `503`, `504`, and transient connection failures (connection reset, DNS timeout). Use exponential backoff with jitter (base intervals: 1s, 2s, 4s).
  - **Non-retryable errors:** `400`, `401`, `403`, `404`, `422`, schema-invalid outputs, and deterministic validation failures. These fail immediately.
  - **Rate limiting:** on `429` from hosted NIM, backoff is the primary mitigation. Existing concurrency controls (`EVAL_CONCURRENCY_HOSTED` / `EVAL_CONCURRENCY_SELF_HOSTED` for evaluations, bounded background tasks for embeddings and batch labeling) limit concurrent requests. For sustained high-volume workloads, self-hosted NIM is recommended over aggressive retry loops.
2. **Retry (UI action):**
  - MUST create a new proposal attempt (new `inference_invocation_id`) for the same `example_key`.
  - MUST set `retry_of_inference_invocation_id` to link the new attempt to the prior attempt.
  - MUST NOT advance the review selector and MUST NOT create a Verified label (only Accept/Edit do).

**Idempotency / dedupe keys**

- **Batch Labeling per-example operation ID MUST be:**

```text
operation_id = "{batch_label_run_id}:{example_key}"
```

- Re-execution of a Batch Labeling run MUST NOT duplicate persisted outcomes for the same `operation_id`.

**Failure handling**

- **Interactive proposals:** for `schema_invalid` / `timeout` / `endpoint_error`, UI MUST surface the failure state and allow **Edit** (manual label) and **Retry**.
- **Skip/Omission:** per §4.5.
- **Missing image files:** if `storage_ref` is not found or unreadable at runtime (image moved/deleted after ingest), the image serving endpoint (§10.2.9) returns an error. The labeling screen shows a broken-image placeholder and offers **Skip**. The system MUST NOT crash or block the labeling session.
- **Evaluation:** after the concurrent burst, any failed examples are retried sequentially (concurrency=1; §7.1). If any example still fails after the sequential retry pass, the run MUST be marked **incomplete** (no pass/gate). There is no configurable failure tolerance; one persistently failed example means the run cannot satisfy the Scale-Up Readiness Gate. Partial diagnostic metrics MAY be displayed but MUST be labeled as diagnostic only.
- **Batch Labeling:** per-example failures do not make the run incomplete; the run completes when every example reaches a terminal state (success or failure). A circuit breaker pauses the run after consecutive endpoint failures (§8.2 step 8). Failures MUST persist failure metadata + best-available artifacts/refs; exports MUST include only `schema_valid_core=true` outputs (schema-invalid outputs are never exported).
- **Review selector:** if CLIP embeddings are missing/unavailable for the active selection mode, selection MUST use available pHash values (§5.6), with deterministic `example_key` fallback while every candidate hash is pending or failed.
- **Embedding computation:** probe failures (403/no model access, network errors) on every arm of the §5.5.1 cascade MUST set `embedding_provider=none`; the review selector continues with pHash-diverse mode. Per-image CLIP embedding failures (payload too large, rate limits, transient errors) MUST be handled gracefully: log the failure, skip the embedding for that image, and continue.
- **TAO jobs:** submission failures MUST set TAOJob `status="failed"` with an error ref; polling failures MUST NOT corrupt last-known-good status and MUST record/update `poll_error_ref` for diagnostics.

**Operational Logging**

Operational logging captures process-level reasoning and decisions that are not part of the durable record model (§13) but are essential for debugging, development, and understanding system behavior. Operation Records capture *what happened*; logs capture *why it happened*, the reasoning at each decision point.

This system is a Blueprint. Implementations MUST produce structured operational logs at key decision points so that developers and operators can trace behavior, diagnose failures, and validate correctness. Each log entry MUST be a self-contained, parseable unit with enough context to understand the decision without reading surrounding entries.

*Log structure (required):*

Every log entry MUST be structured (JSON-compatible key-value) and MUST include at minimum:

- `timestamp` (ISO 8601)
- `level` ∈ {`debug`, `info`, `warn`, `error`}
- `component`: identifies the subsystem (e.g., `icl_selection`, `token_budget`, `schema_validation`, `review_selector`, `model_invocation`, `capability_probe`, `pool_routing`, `gate_evaluation`, `visual_budget`)
- `project_id`
- `correlation_id`: the most specific scope ID for the current operation: `inference_invocation_id` for proposals/evaluation/batch, `batch_label_run_id` for batch runs, `evaluation_run_id` for evaluation, `tao_job_id` for training
- `message`: human-readable summary of the decision or event
- `details`: structured payload specific to the component (the reasoning data)

*Correlation:*

A single user action (e.g., requesting a proposal) triggers a chain of operations: ICL selection → token budget → prompt rendering → model invocation → schema validation. All log entries in that chain MUST share the same `inference_invocation_id` so the full decision trace is recoverable with a single query.

*Required log points:*

All 8 log points MUST use the same structured JSON format defined above. Each log point MUST emit at least one structured event at its default level. Additional detail MAY be emitted only when `LOG_LEVEL=debug`. Debug-default log points MAY escalate to `info` when something materially changed (e.g., ICL pruning occurred, CLIP→pHash fallback, capability probe flipped status, pool membership changed).

**Default level: `info`** (always visible at default config):

1. **Model invocation**: model name, endpoint, sampling params sent, visual budget params sent (if any), `finish_reason` received, response latency ms, input/output token counts (when reported by provider). On automatic retry: log trigger (`finish_reason=length`, context overflow) and what changed (budget increase, ICL drop).

2. **Schema validation**: validity classification (`schema_valid_core`), Core error count and specific errors, Aux error count, normalization steps applied. On schema-invalid results, log enough detail to diagnose why the output failed.

3. **Gate evaluation**: each criterion's current value, threshold, and pass/fail. Overall `gate_status`. Log on every re-evaluation (not just when the status changes).

**Default level: `debug`** (visible when `LOG_LEVEL=debug`; escalate to `info` on material changes):

4. **ICL selection**: candidate pool size, selected `example_key[]`, total ICL count, pruned count/keys. On cold start (zero candidates), log explicitly. Escalate to `info` when ICL pruning drops examples.

5. *(Removed in v1.0 — token-budget utilization reporting retired with the Context Budget Indicator, §6.10. Enforcement observability: log point 4 carries pruned keys/count, and image-cap pruning emits an `info` line at the dispatch boundary. Number retained so log points 6–8 keep their identities.)*

6. **Capability probes**: probe type (structured generation, thinking toggle, visual budget), request sent (summarized), response status code, resulting classification (`supported`/`unsupported`/`unknown`), and the ModelConfig field updated. For multi-stage probes (visual budget, §6.9.2), log each stage separately. Escalate to `info` when a probe flips status.

7. **Review selector**: selection mode (CLIP-diverse vs pHash-diverse), candidate set size, selected `example_key`, diversity score. On mode selection, log which diversity signal was used and why (CLIP available vs pHash fallback). Escalate to `info` on CLIP↔pHash mode switch.

8. **Pool routing**: routing decision for each newly Verified example: destination (`test_pool` or non-pool), `verified_outcome`, current pool count, target count, whether rebalancing was triggered and what was promoted. Escalate to `info` when rebalancing actually changes pool membership.

*Security constraints apply:* operational logs MUST NOT contain API keys, raw image bytes, full prompt text, or user-identifiable information beyond `project_id` and `example_key`. Prompt content is referenced by `prompt_hash` (§2.1); image content is referenced by `storage_ref`. Log entries MUST use an approved-fields approach: only emit the structured fields defined above. A basic regex redaction filter for API key patterns and bearer tokens MUST be applied as a safety net.

*Development mode:*

Implementations MUST support a `LOG_LEVEL` configuration (default: `info`). When set to `debug`, the debug-default log points (4 and 6–8 above) become visible, and all seven log points emit richer detail: full candidate lists, complete scoring vectors, rendered prompt token breakdowns, raw provider response metadata, and step-by-step normalization traces. `debug` output is verbose and intended only for active development and troubleshooting.

---

## 12. Security and Privacy (Blueprint-level, Basic)

### 12.1 Secrets Storage

Secrets (API keys, authentication credentials) MUST be stored in environment variables or the canonical `.env` file (§1.9), following NVIDIA Blueprint conventions. Required secrets: `NVIDIA_API_KEY` (hosted NIM), `NGC_API_KEY` (local NIM container pulls), `TAO_API_KEY` (Student Training). Not all are required for every deployment — only the secrets needed by the active NIM modes and features.

The canonical `.env` file location is `~/.vlm_feedback_loop/.env` (§1.9), colocated with `config.yaml`. The file MUST be:

- Created by the bootstrap command (`vlm-feedback-loop init`) as a commented skeleton documenting required variables without values
- Readable only by the local user where practical (file permissions `600`); the bootstrap command MUST create `~/.vlm_feedback_loop/` with user-only permissions (`700`) where supported by the host OS
- Outside the repository — the canonical location is not within any repo working directory, so `.gitignore` is not the primary safeguard. If developers use the explicit override (`--env-file` or `VLM_FEEDBACK_LOOP_ENV_FILE`) to point at a repo-local `.env` for development, that repo-local file MUST be gitignored.

The `.env` file contains secrets in plain text. For production or managed deployments, operators MAY bypass the file entirely and provide secrets through process environment variables or a secrets manager.

API keys MUST NOT be stored in the YAML config file (§1.9).

Credential values MUST NOT be accepted through command-line arguments or embedded in subprocess argument vectors. A parent launching a credential-consuming child MUST pass the value through a private child environment (or stdin where the child protocol requires it), keep the parent environment unchanged, and redact supplied private values from spawn errors and captured child output before logging, returning, or persisting diagnostics. Operator-visible Docker commands use name-only environment forwarding such as `-e NGC_API_KEY`, never `-e NGC_API_KEY=<value>` or a shell placeholder containing the value.

**Deployment-scoped secret persistence:** `NVIDIA_API_KEY`, `NGC_API_KEY`, and `TAO_API_KEY` are deployment-scoped secrets. The system supports three persistence shapes — selected per call by the SME or operator (FTU Phase G, 2026-05-12 amendment):

  - (a) **One-time connection test** — the web UI accepts a pasted key for a single connection probe (Screen 2.3 / Screen 2.4 / Screen 2.8). The value MUST be held only in request memory, used for the probe, and discarded after the response. Never written to any persistent store.
  - (b) **Runtime override (in-memory)** — `POST /v1/secrets:set` with `persist=false` installs the key into the deployment-scoped runtime-override layer (§12.1.2). The value applies to the next NIM call (embedding worker spawn, Teacher proposal, local NIM image pull) without a backend restart. Held only in process memory; lost on process exit.
  - (c) **Persisted to `.env`** — `POST /v1/secrets:set` with `persist=true` AND deployment-level `ALLOW_UI_SECRET_PERSIST=true` atomically upserts the line in `~/.vlm_feedback_loop/.env` with `0600` file / `0700` parent-dir permissions and reloads `Settings`. The same value remains in the runtime override for the rest of the process so already-queued work holding an older `Settings` snapshot still sees the new credential; the disk value is canonical after restart. A write or permission failure MUST leave the prior file intact and returns an explicit persistence error; the new value remains a session-only runtime override. Container / production deployments where `.env` is managed externally SHOULD set `ALLOW_UI_SECRET_PERSIST=false` to disable this path; the endpoint then returns 403 `ui_secret_persist_disabled` and the UI hides the persist checkbox.

The web UI MUST NOT write secrets to YAML config, SQLite, browser `localStorage`, or any project-scoped record. The three permitted persistence shapes above are exhaustive.

### 12.1.1 Audit Trail

The Secrets API emits structured `info`-level logs (`secret_runtime_set` for path (b); `secret_persisted` for path (c)) into the project-relative logging surface defined in §11. Key VALUES MUST NOT appear in audit output; only the secret name + value length is recorded.

### 12.1.2 Runtime Override Layer

The runtime-override layer lives in `services/runtime_secrets.py`. Services MUST read secret values via `get_effective_secret(name, settings)` rather than `settings.NAME` directly so the override path is honored. The override wins over the `Settings`-loaded value; when no override is set the function falls through to `Settings` (the operator's `.env`-loaded value). The override layer is reset to empty on process restart.

### 12.2 General Requirements

- MUST NOT log API keys (NIM or TAO).
- MUST NOT embed API keys in persisted prompts/invocations/UI payloads.
- MUST NOT log raw image bytes in operational logs.
- Filesystem operations (§10.2.10) MUST enforce `IMAGE_ROOT`. When the backend is network-accessible (non-loopback bind), one absolute image root MUST be configured. Symlinks escaping it MUST be rejected.

---

## 13. Required Records

Twenty numbered sections describe 19 active persisted record families;
§13.14 documents the removed `PromptPackage` and is not an active family.
Implementations MAY store these in one table/collection per family (or
normalized equivalents).

### 13.1 Operation Record (Required)

Single per-invocation record used across interactive proposals, evaluation, and batch labeling.

Minimum fields:

- `project_id` (required)
- `inference_invocation_id`
- `purpose` ∈ {`interactive_proposal`, `evaluation`, `batch_label`, `rationale_regeneration`} (historical rows may carry `purpose="guidance_rewrite"`, retained from the removed rewrite feature, §6.4)
- `example_key` (required when invocation pertains to one example)
- `guidance_id`
- `model_config_id`
- `endpoint_id` (MUST reference the `NimEndpoint` used for this invocation, §13.18)
- `model_name`
- `icl_example_keys_used[]` (empty for purposes that do not use ICL)
- `invocation_status` (`pending` → final)
- provider/latency/status metadata (include `latency_ms_end_to_end` when attempted)

Generation Controls fields (required; §6.7):

- `generation_preset_key: string` (the preset key used, e.g., `"precise"`, `"explore"`, `"creative"`)
- `sampling_params_effective`:
  - `temperature: float`
  - `top_p: float`
  - `top_k: int | null` (optional; if sent)
  - `seed: int | null` (null for interactive; set for evaluation/batch)
- `thinking_mode_effective: "on" | "off"`
- `thinking_request_fields_effective: object | null` (e.g., `{"chat_template_kwargs": {"enable_thinking": false}}`; null when Thinking=ON or model does not support toggle)
- `max_tokens_effective: int`
- `reasoning_headroom_tokens_effective: int | null` (the reasoning headroom added to `base_output_tokens` for this invocation; null when `thinking_mode_effective="off"`; equals `MODEL_REASONING_HEADROOM_TOKENS` unless overridden by `RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE`)

Visual Budget fields (required; §6.9):

- `visual_budget_preset_key: string | null` (the preset key used, e.g., `"fast"`, `"balanced"`, `"high_detail"`; null when model does not support visual budget controls)
- `visual_budget_params_effective: object | null` (the exact `mm_processor_kwargs` object sent in the request; null when not sent)

Image transport fields (required when invocation includes images; §6.3):

- `image_transport_mode` ∈ {`base64_inline`, `direct_url`} | null (how the query image was sent; null for non-image invocations)
- `image_format_transmitted: string | null` (MIME type of the image as sent, e.g., `"image/png"`, `"image/jpeg"`; null for non-image invocations)

> **Absent from v1.0:** the `asset_upload_ref` and `asset_upload_error_ref` fields and the `asset_upload` transport mode are not part of the public schema. Hosted endpoints accept inline base64, and NVIDIA retired the `/v2/nvcf/assets` endpoints.

Label tier (recommended; required when a label is produced):

- `label_tier` ∈ {`proposal`, `auto_labeled`} (nullable for non-label purposes)
  - `interactive_proposal` → `proposal`
  - `batch_label` → `auto_labeled`

Attempt artifact references (required for model-invoking purposes; stored as files under `{project_dir}/artifacts/`, §1.7):

- `raw_model_response_ref: string | null`
- `normalized_json_ref: string | null`
- `validation_report_ref: string | null`

Validation report fields (required):

- `schema_valid_core` (boolean; nullable when not applicable)
- `validation_errors_core[]` (nullable when not applicable)
- `validation_errors_aux[]` (nullable when not applicable)
- `structured_generation_fallback_used: boolean` (default: `false`; set `true` when the invocation fell back to prompt-only JSON generation after a runtime `json_schema` rejection, §6.2)
- `structured_generation_mode_effective` ∈ {`auto`, `prompt_only`} | null (the run-level mode governing this invocation; null for interactive proposals which use per-invocation fallback)
- `structured_generation_attempted: boolean` (whether `response_format` with `json_schema` was included in the request)

Provider token usage, completion, and truncation fields (required when model invocation attempted):

- `finish_reason: string | null` (the `finish_reason` value from the model response, e.g., `"stop"`, `"length"`, `"content_filter"`; null when no response received)
- `prompt_tokens: int | null` (token count from provider `usage.prompt_tokens`; null when not reported)
- `completion_tokens: int | null` (token count from provider `usage.completion_tokens`; null when not reported)
- `total_tokens: int | null` (token count from provider `usage.total_tokens`; null when not reported; the backend does not invent this value from partial usage)
- `truncation_attributed_schema_invalid: boolean` (default: `false`; set `true` when `finish_reason="length"` and the output is `schema_valid_core=false`, indicating truncation caused the schema failure)

Error payload references:

- `provider_error_ref: string | null` (sanitized when present)

Linkage fields (nullable):

- `retry_of_inference_invocation_id`
- `evaluation_run_id`
- `batch_label_run_id`
- `ignored_due_to_run_cancellation: boolean` (default: `false`; set `true` when the Operation Record is persisted after the parent run entered `canceling`, §13.2.1; such records are retained for audit but MUST NOT contribute to authoritative metrics)

Purpose-dependent outcome fields:

- `exact_match_pass` (evaluation only)
- failure class (e.g., `schema_invalid` / `timeout` / `endpoint_error`)

Notes:

- Verification metadata for Accept/Edit is stored with the Label record (§4.6, §13.7).
- Per-example evaluation results are Operation Records with `purpose=evaluation` linked by `evaluation_run_id`.
- Batch labeling results are Operation Records with `purpose=batch_label` linked by `batch_label_run_id`.

### 13.2 Run Record (Required if Enabled)

Single run-level schema covering: evaluation runs and batch labeling runs.

Minimum fields (common to all run types):

- `project_id` (required)
- `run_id` (required; unique within project)
- `run_type` ∈ {`evaluation_run`, `batch_label_run`} (required)
- `status` (required; run-type-specific enum; see state machines below)
- `status_reason: string | null` (human-readable reason for the current status, e.g., `"backend_restart_interrupted"`, `"circuit_breaker_threshold_reached"`, `"superseded_by_newer_evaluation"`)
- `cancel_requested_at: timestamp | null` (set when cancellation is requested; null otherwise)
- `recovered_from_restart: boolean` (default: `false`; set `true` when a run is recovered after backend restart)
- `created_at: timestamp` (required)
- `started_at: timestamp | null` (when worker execution began)
- `completed_at: timestamp | null` (when run reached a terminal state)

#### 13.2.1 Evaluation Run State Machine

Statuses:

```text
queued | running | canceling | completed | incomplete | canceled | failed
```

- `queued`: Run record exists, configuration snapshot and pool version snapshot taken, worker not yet started.
- `running`: Actively issuing evaluation inferences and/or performing the sequential retry pass.
- `canceling`: Cancellation requested (supersession or manual cancel). Two-phase: (1) transition to `canceling` immediately, stop dispatching new per-example inferences immediately; (2) wait for all in-flight tasks to settle before transitioning to `canceled`. Each outcome transaction MUST classify itself at the durable run-status boundary: only an outcome committed after the parent entered `canceling` is marked `ignored_due_to_run_cancellation=true`; earlier committed outcomes MUST NOT be bulk-reclassified by finalization. Ignored records remain linked to the canceled run for diagnostics and MUST NOT contribute to authoritative metrics, summaries, comparisons, or the Scale-Up Readiness Gate. Because HTTP calls are bounded by deadlines (`HTTP_DEADLINE_BACKGROUND_S`), the wait for in-flight tasks is finite. The run MUST NOT transition to `canceled` until all in-flight tasks have reached a terminal state and all final writes have committed.
- `completed`: Run finished cleanly; all pool examples reached a terminal per-example outcome with no persistent failures. Results are authoritative and satisfy the Scale-Up Readiness Gate.
- `incomplete`: Run finished, but at least one example still failed after the sequential retry pass. Metrics are diagnostic only and MUST NOT satisfy the Scale-Up Readiness Gate (§7.1).
- `canceled`: Run was intentionally stopped, either manually by the SME or because a newer evaluation superseded it. All in-flight tasks have settled. Operation Records persisted during the `canceling` window are retained for audit but are non-authoritative. Canceled runs MUST NEVER produce authoritative aggregate metrics, even if some child Operation Records finished successfully. Evaluation metrics, run summaries, and gate checks MUST be computed only from Operation Records whose `evaluation_run_id` matches a `completed` run.
- `failed`: Unrecoverable execution failure (backend restart interruption, database/persistence failure, snapshot creation failure, uncaught runtime exception).

Transitions:

```text
queued     -> running | canceling | failed
running    -> completed | incomplete | canceling | failed
canceling  -> canceled | failed
```

Terminal states: `completed`, `incomplete`, `canceled`, `failed`.

Restart recovery: evaluation runs in `queued`, `running`, or `canceling` state MUST transition to `failed` with `status_reason="backend_restart_interrupted"`. The user re-triggers evaluation.

#### 13.2.2 Batch Labeling Run State Machine

Statuses:

```text
queued | running | paused | canceling | completed | canceled | failed
```

- `queued`: Run record exists, input set and configuration snapshot taken, worker not yet started. Also used after restart recovery (with `recovered_from_restart=true`) before auto-resume.
- `running`: Actively processing examples.
- `paused`: Processing intentionally halted because the circuit breaker fired (§8.2 step 8). Resumable from the next unprocessed example. `paused_reason` records the cause (e.g., `"circuit_breaker_threshold_reached"`).
- `canceling`: Cancellation requested; no new examples may start, and in-flight work is unwinding. Each outcome transaction uses the durable run status to mark only post-request outcomes `ignored_due_to_run_cancellation=true`; finalization MUST preserve earlier outcomes. A schema-valid ignored outcome MUST NOT create an Auto-Labeled Label.
- `completed`: All selected examples reached a terminal per-example outcome (success, schema-invalid, timeout, or endpoint_error). This does NOT mean every example succeeded — use counters to show the breakdown. The run itself finished.
- `canceled`: User stopped the run (from paused or running state); already-persisted results are retained.
- `failed`: Unrecoverable run-level failure (database/persistence failure, corrupted run state, uncaught runtime exception that prevents safe resume).

Transitions:

```text
queued     -> running | canceling | failed
running    -> paused | completed | canceling | failed
paused     -> queued | canceled | failed
canceling  -> canceled | failed
```

Terminal states: `completed`, `canceled`, `failed`.

Restart recovery: batch labeling runs in `queued` or `running` state MUST transition to `queued` with `recovered_from_restart=true`, then auto-resume to `running` from the next unprocessed example using idempotent per-example persistence and the durable circuit-breaker streak/latch (§8.2 steps 7–8). A set latch pauses before any new invocation even when an already-in-flight success reset the stored streak before interruption. Runs in `paused` state remain `paused` (do not auto-resume). Runs in `canceling` state transition to `canceled` if `cancel_requested_at` was persisted and their exact item lineage can be reconciled; recovery rebuilds counters from non-ignored authoritative items. Missing cancellation intent produces `failed` with `status_reason="backend_restart_interrupted"`; malformed snapshots or invalid durable lineage fail only the affected run with `status_reason="batch_recovery_state_invalid"` and do not abort recovery of other runs.

Additional batch labeling run fields:

- `paused_reason: string | null` (e.g., `"circuit_breaker_threshold_reached"`; null when not paused)
- `circuit_breaker_threshold: int | null` in the detail response (the
  run-start snapshot used to explain a circuit-breaker pause truthfully;
  legacy run records may return null)
- `circuit_breaker_consecutive` in durable run metadata (updated atomically
  with each authoritative item outcome; reset by success or explicit Resume)
- `circuit_breaker_tripped` in durable run metadata (latched atomically with
  the item that reaches the threshold; cleared only by explicit Resume)
- `examples_succeeded: int` (default: 0; schema-valid Core outputs with Label records created)
- `examples_schema_invalid: int` (default: 0; Core-invalid outputs)
- `examples_timeout: int` (default: 0)
- `examples_endpoint_error: int` (default: 0)
- `examples_total: int` (total selected for this run)

**Design note: why `incomplete` exists for evaluation but not batch labeling.** Evaluation has a special "finished but not trustworthy" outcome because the spec requires that one persistently failed example makes results unsuitable for gating (§7.1). Batch labeling is different: per-example failures are expected, the run can finish successfully as a whole, and exports filter to `schema_valid_core=true` outputs. So batch labeling uses `completed` plus counters; evaluation uses `completed` vs `incomplete`.

**Design note: why `paused` exists for batch labeling but not evaluation.** The spec defines a circuit breaker with Resume/Cancel for batch labeling (§8.2 step 8). Evaluation has cancel/restart behavior but no pause concept.

#### 13.2.3 Required Fields by Run Type

`evaluation_run`:

- pool version ID
- `guidance_id`
- `model_config_id`
- `icl_mode` (`enabled`/`disabled`)
- `evaluation_source` ∈ {`tao`, `nim`} (required; §7.2. `tao` for TAO-backed quality evaluation against checkpoint; `nim` for NIM-backed evaluation against deployed endpoint)
- `generation_preset_key` (required; the Output Stability preset used; §6.7)
- `thinking_mode_effective` ∈ {`on`, `off`} (required; §6.7)
- `visual_budget_preset_key: string | null` (required; §6.9; null when model does not support visual budget controls or when `evaluation_source=tao`)
- `structured_generation_mode_effective` ∈ {`auto`, `prompt_only`} (required; the run-level structured generation mode; §6.2)
- `inference_contract: object` (required; the effective Inference Contract for this run, §6.11: `output_field_mode`, `icl_field_mode`, `icl_max_examples`; legacy snapshots may additionally carry the retired `icl_pinned_edits_k` key, which comparisons ignore)
- `runtime_config_snapshot: object | null` (required for new NIM-backed evaluation and Batch runs; null for TAO runs and terminal pre-v1_0003 history). The v1_0003 migration backfills valid non-terminal legacy NIM runs, including paused Batch runs, with version 1: model/endpoint lineage, URL/mode/auth mode, context and capability controls, resolved image cap, and model-default ICL cap. New runs use version 2, which adds a nested `inference_settings` object containing the selected sampling values, effective visual processor arguments, token-budget primitives, effective ICL cap/candidate limit/adaptive thresholds, and image transport edge. Startup upgrades non-terminal or paused Batch version-1 snapshots exactly once before recovery or Resume; terminal version-1 history remains valid. Snapshots MUST NOT contain credentials, filesystem authorization, timeouts, retries, concurrency, rate-limit policy, or emergency capability kill switches. `services.run_config.RuntimeConfigSnapshot` is the executable version-2 schema and MUST reject unknown or malformed persisted shapes.
- `icl_eligible_count_at_start: int` (required; number of ICL-eligible Edits when the run started)
- `icl_eligible_count_at_completion: int | null` (required; number of ICL-eligible Edits when the run completed; null while running; used for ICL growth recommendation threshold, §7.1)
- `tao_job_id: string | null` (required when `evaluation_source=tao`; links to the TAO evaluate job)
- `tao_native_metrics: object | null` (TAO's own aggregate metrics; informational when `evaluation_source=tao`)
- `rescored_metrics: object | null` (system's canonical evaluator metrics; authoritative when `evaluation_source=tao`)
- `examples_total: int` (default 0; pool member count when the run started)
- `examples_succeeded: int` (default 0; OperationRecords with `invocation_status="success"`)
- `examples_schema_invalid: int` (default 0; OperationRecords with `invocation_status="schema_invalid"`)
- `examples_timeout: int` (default 0; OperationRecords with `invocation_status="timeout"`)
- `examples_endpoint_error: int` (default 0; OperationRecords with `invocation_status="endpoint_error"`)

**F45 (Phase 12 closeout amendment 2026-05-13) — per-status counter aggregation on evaluation_run finalize.** The five `examples_*` counters above MUST be aggregated from the run's OperationRecord rows during the evaluation Phase G finalize, NOT left at their `0` defaults. Pre-F45 the evaluation_service Phase G wrote `metrics` + `coverage_gaps` + `icl_eligible_count_at_completion` but never aggregated the per-status counts; this silently broke F35's `_compute_parseable_rate` (which divides `examples_succeeded / examples_total` and refused the `failed → partial` promotion when the numerator stayed at 0). F45 added a single `GROUP BY invocation_status` query against `OperationRecord` (scoped to the run, excluding `ignored_due_to_run_cancellation` rows per §13.2.1) to the finalize block. Matches the existing pattern in `batch_label_service` finalize. Implementation: `services/evaluation_service._execute_evaluation` Phase G. Live evidence motivating F45: the 2026-05-13 rental's 8B NIM eval landed 83 success / 1 schema-invalid (parseable=0.988, well above F35's 0.90 threshold) but `quality_status` stayed `failed` because the counters never made it to the run record.

`batch_label_run`:

- `batch_label_run_id` (alias of `run_id`)
- `guidance_id`
- Teacher `model_config_id`
- ICL settings snapshot
- input selector definition
- `structured_generation_mode_effective` ∈ {`auto`, `prompt_only`} (required; the run-level structured generation mode; §6.2)

Note: Batch Labeling always uses the Teacher, whose Inference Contract is fixed at `output_field_mode=all, icl_field_mode=core_only` (§6.11). No per-run Inference Contract snapshot is needed because the contract cannot vary.

### 13.3 Review Selector Scheduler State (Required for Reproducibility)

System MUST persist enough state to reproduce review selector decisions (e.g., selection mode and recent-window list of presented `example_key`s).

- Stored on Project record.
- **Schema change reset:** when a semantic Core change occurs (§4.4.1 step 4f), the recent window MUST be cleared to empty and any persisted selector-history state MUST be reset. The selector starts with no history, as if the project were new.

### 13.4 Project Record and Project Scoping (Required)

A Project is top-level workspace boundary and MUST be partition key for all persisted entities.

#### 13.4.1 Project Record Schema (minimum)

- `project_id: string` (required)
- `name: string` (required)
- `description: string | null` (optional)
- `project_dir: string` (required; absolute path to `{workspace_root}/projects/{project_id}/`; §1.7)
- `created_at: timestamp` (required)
- `updated_at: timestamp` (required)

Derived response field (not persisted):

- `usage_policy` ∈ {`evaluation_only`, `operator_managed`} (required on API
  responses). A hosted endpoint whose `base_url` host is
  `integrate.api.nvidia.com` is `evaluation_only`; every other endpoint is
  `operator_managed`. The latter
  means only that the operator supplied or manages the endpoint. It MUST NOT be
  presented as proof of a commercial subscription or entitlement.
- `archived_at: timestamp | null` (default: `null`; set by `POST /v1/projects/{id}:archive`, cleared by `:unarchive`. Source of truth for soft-archive state; the `{project_dir}/.archived` sentinel marker is a lazy index. See §10.2.13.1 for the busy gate, cross-mutation guard, and worker-filter contract.)
- `setup_completed_at: timestamp | null` (default: `null`; stamped on the first transition through onboarding via `POST /v1/projects/{id}:mark_setup_completed`. ``ProjectIndexRedirect`` routes to `/setup` when null; afterward it resumes non-terminal training, opens the state-aware Project Overview when training history or Student models exist, or enters labeling for an early project. Stamping is idempotent at the service layer — redundant calls collapse to a single AuditEvent and stable timestamp.)

Default selections:

- `teacher_model_config_id: string`
- `active_guidance_id: string`
- `active_student_model_config_id: string | null`

Generation Controls defaults (§6.7):

- `labeling_generation_preset_key: "precise" | "explore"` (default: `"precise"`)
- `thinking_default_on: boolean` (default: `true`)

Visual Budget Controls defaults (§6.9):

- `visual_budget_preset_key: "fast" | "balanced" | "high_detail"` (default: `"high_detail"`)

Structured generation default (§6.2):

- `structured_generation_mode_default: "auto" | "prompt_only"` (default: `"auto"`; applies to evaluation and batch labeling runs when no explicit mode is provided)

Rationale display (§6.3):

- `rationale_anti_anchoring: boolean` (default: `true`; when true, rationale is hidden until Edit)

Evaluation state (§7.1):

- `auto_evaluate_enabled: boolean` (default: `false`)
- `icl_recommendation_dismissed_at_count: int` (default: `0`; updated when the SME dismisses an ICL growth recommendation; reset to `0` on semantic Core change, §4.4.1)

Export field mode defaults:

- `export_field_mode: "all" | "aux_and_core" | "core_only"` (default: `"all"`)

Embedding configuration:

- `embedding_provider: "hosted_nvclip" | "self_hosted_nvclip" | "none"` (resolved by the §5.5.1 cascade: `self_hosted_nvclip` when a healthy local embedding NIM deployment exists, else `hosted_nvclip` when the hosted NIM API key is configured, else `none`. Enum value names retained for backwards compatibility with existing project DBs; the actual model is governed by `embedding_model_id`.)
- `embedding_model_id: string | null` (default: `nvidia/llama-nemotron-embed-vl-1b-v2` when provider is `hosted_nvclip`)
- `embedding_dim: int | null` (default: `2048` when provider is `hosted_nvclip`)
- `embedding_endpoint_id: string | null` (references the NIM endpoint used for embedding; nullable when disabled)

pHash configuration:

- `phash_algorithm: string` (default: `"dct_phash_64"`; recorded at project creation; used to detect algorithm changes across versions, §5.6.1)

Feature flags:

- `feature_flags: object` (must support flags referenced in spec)

Schema refinement reminder state (§6.8):

- `schema_refinement_reminders_dismissed: int` (default: `0`; tracks how many reminders have been dismissed; reset to `0` on semantic Core change)
- `schema_change_context_example_key: string | null` (default: `null`; set when a semantic Core change occurs while the SME is reviewing an example, §4.4.1; cleared after the selector presents it)

Test Pool configuration (§4.3):

- `test_pool_fraction: float` (default: `0.40`)

Scale-Up Readiness Gate configuration (§7.3):

- `scaleup_exact_match_threshold: float` (default: `0.80`)
- `scaleup_per_field_match_threshold: float` (default: `0.80`)
- `scaleup_min_per_value_f1_threshold: float` (default: `0.60`)
- `scaleup_accept_rate_threshold: float` (default: `0.80`)
- `scaleup_accept_rate_window: int` (default: `50`)
- `scaleup_min_test_pool_size: int` (default: `60`)

Scale-Up Readiness Gate state (§7.3):


Review selector scheduler state pointer:

- `review_selector_scheduler_state_ref: string | null` OR `review_selector_scheduler_state: object | null`
If inline: MUST satisfy Section 13.3 reproducibility requirements.

#### 13.4.2 Project CRUD APIs (required)

- `POST /v1/projects` create
  - Request: `name` (required), `description` (optional).
  - On create, server MUST: create the project directory under `{workspace_root}/projects/{project_id}/` (§1.7), initialize the project database, seed the model catalog with the entries listed in §4.8, set `teacher_model_config_id` per §4.8, and set Generation Controls and embedding defaults per §13.4.1.
  - Response: the full Project record including all seeded defaults.
- `GET /v1/projects/{project_id}` read
- `PATCH /v1/projects/{project_id}` update allowed fields (name, description, selections, feature_flags, scheduler pointer)
  - MUST validate references are within same `project_id`
  - MUST validate `teacher_model_config_id` eligibility per §4.8 on every update: the referenced ModelConfig MUST have `teacher ∈ eligible_roles` AND `supports_image_input=true`. Role/vision violations MUST return HTTP 422 with a descriptive detail message. This contract is required by the top-bar Teacher picker (`TeacherModelPicker.tsx`), which writes to this endpoint on every change.
  - MUST validate `active_student_model_config_id` eligibility: when set, the referenced ModelConfig MUST have `student_base ∈ eligible_roles`.
  - MUST NOT affect in-flight evaluation or Batch Labeling runs. Both workflows snapshot `teacher_model_config_id` at run start (§7.1, §8.2); a PATCH during an active run updates only the project-wide default and takes effect on the next run or next interactive proposal.

#### 13.4.3 Partitioning Rules (required)

Every record family MUST include `project_id` including at minimum:

- Operation Record (§13.1)
- Run Record (§13.2)
- Review Selector Scheduler State (§13.3; embedded on its Project record)
- Project (§13.4)
- DatasetExport (§13.5)
- AuditEvent (§13.6)
- Label (§13.7)
- Example (§13.8)
- Guidance (§13.9)
- ModelConfig (§13.10)
- Pool (§13.11)
- TAOJob (§13.12)
- StudentModel (§13.13)
- LocalNimDeployment (§13.15)
- ClipEmbedding (§13.16)
- NimEndpoint (§13.18)

All APIs MUST enforce project scoping: no cross-project reads/writes; reject cross-project IDs even if syntactically valid.

**Deployment-scoped exceptions:** `EmbeddingDeploymentConfig` (§13.17),
`TAODeploymentConfig` (§13.19), and
`TAOBaseExperimentProvisioningRun` (§13.20) are shared across projects and do
not carry `project_id`. They store deployment infrastructure state, not
per-project domain data. TAO endpoint configuration (§1.6) is likewise
deployment-scoped but stored in environment variables rather than a record.

#### 13.4.4 Active Student resolution rules (required)

For workflows requiring Student selection:

1. If `project.active_student_model_config_id` non-null, use it.
2. Else if exactly one Student model config available, use it.
3. Else require explicit `student_model_config_id` in request; MUST NOT guess.

#### 13.4.5 Feature flag scoping

All workflow flags are project-scoped and stored in Project `feature_flags`:

- `REVIEW_SELECTION_MODE`: `auto`, `phash_diverse`, or `clip_diverse`
- `EMBEDDINGS_AUTO_COMPUTE`: when `true` (default when hosted NIM API key is configured), the system automatically computes CLIP embeddings for ingested images in the background. When `false`, CLIP embedding computation is skipped (review selector uses pHash-diverse mode). The independent background pHash sweep always runs regardless of this flag.
- `CLIP_SWITCHOVER_MIN_COUNT`: minimum number of eligible examples with CLIP embeddings before the review selector upgrades from pHash-diverse to CLIP-diverse (default: `50`; §5.5.2).

Deployment capabilities (e.g., TAO installed) are not user-editable flags and are exposed read-only if surfaced (Section 10.2.8).

### 13.5 DatasetExport Record (Required When Exports Used)

Persists versioned dataset export artifacts with manifests and lineage for training, evaluation, and testing intents (§8.4, §9.3).

Minimum fields:

- `project_id` (required)
- `dataset_export_id` (required; unique within project)
- `dataset_intent` ∈ {`training`, `evaluation`, `testing`} (required)
- `export_field_mode` ∈ {`all`, `aux_and_core`, `core_only`} (required; the field scope used for this export; §9.3)
- `guidance_id` (required; Guidance version governing the export)
- `label_tier_filter` ∈ {`verified_only`, `auto_labeled_only`, `combined`} (required)
- `selection_definition_snapshot: object` (required; filters, pool exclusions, Core-valid requirement for Auto-Labeled, etc.)
- `status` ∈ {`running`, `completed`, `failed`} (required; the standalone API endpoint creates the record `running` and a background task builds the archive — §10.2.18; training-suite exports are created directly `completed`)
- `status_reason: string | null` (failure detail; `backend_restart_interrupted` when startup recovery failed an export orphaned by a restart)
- `progress: object | null` (`{images_written, images_total}` while the archive builds)
- `started_at: timestamp | null` / `completed_at: timestamp | null` (lifecycle timestamps)
- `artifact_refs: object | null` (archive paths/URIs + SHA-256 checksums; null until `completed`)
- `manifest_ref: string | null` (enumerates included example_keys + filters used; null until `completed`)
- `example_count: int` (required)
- `created_at: timestamp` (required)

Notes:

- Exports include only Verified examples under the current Guidance. After a semantic Core change, old labels are deleted (§4.4.1); only examples re-labeled under the new Guidance are exportable.
- **DatasetExport container format:** DatasetExport artifacts MUST use a TAO/Cosmos-RL-compatible dataset layout. The canonical export layout is `annotations.json` plus an `images/` directory. When a single-file portable bundle is required, the export artifact MUST use `.tar.gz`, not `.zip`. This aligns with TAO/Cosmos-RL dataset packaging expectations (`images.tar.gz` per §9.3.1). Manifests within exports MUST be JSON.

### 13.6 AuditEvent Record (Optional)

Persists notable system and user actions for audit trails. AuditEvents are optional, persisted for observability but not required for correctness.

Minimum fields:

- `project_id` (required)
- `audit_event_id` (required; unique within project)
- `event_type: string` (required; e.g., `skip`, `guidance_save`, `guidance_edit`, `semantic_core_change`, `pool_create`, `batch_label_start`, `action_request_copied`, `storage_ref_remap`)
- `example_key: string | null` (when event pertains to a specific example)
- `event_data: object` (required; type-specific payload; may be `{}`)
- `created_at: timestamp` (required)

Notes:

- Skip actions create an AuditEvent with `event_type="skip"` and the `example_key` (§4.5). The Skip's omission fields on the Example record are the normative persistence; the AuditEvent is supplementary.

### 13.7 Label Record (Required)

Persists all labels — both Auto-Labeled (from Batch Labeling) and Verified (from SME Accept/Edit). A single record family with `label_status` as the discriminator. See §4.5 for action semantics, §4.6 for verification metadata rules, and §4.7 for unified storage design.

Common fields (required for all labels):

- `project_id` (required)
- `example_key` (required)
- `label_status` ∈ {`verified`, `auto_labeled`} (required)
- `guidance_id` (required; Guidance version under which the label was produced)
- `inference_invocation_id` (required; the invocation that produced the label — interactive proposal for verified, batch labeling invocation for auto_labeled)
- `label_json: object` (required; the label matching SchemaCore)
- `labeled_at: timestamp` (required; when the label was created or last promoted)

Verification fields (required when `label_status=verified`; null when `auto_labeled`):

- `verified_outcome` ∈ {`Accept`, `Edit`} | null (required when verified)
- `verified_at: timestamp | null` (required when verified; the moment the SME confirmed the label)
- `edited_core_fields[]: string[]` (required when verified; empty for Accept; lists field_names changed by SME)
- `edited_aux_fields[]: string[]` (required when verified; empty for Accept; persisted for audit and provenance)
- `rationale_source` ∈ {`teacher_proposal`, `sme_edited`, `teacher_regenerated_approved`} | null (required only for Verified labels whose Guidance enables rationale notes; null when disabled; §4.4)
- `rationale_regeneration_invocation_id: string | null` (required when `rationale_source="teacher_regenerated_approved"`; links to the regeneration Operation Record)

Batch Labeling lineage field:

- `batch_label_run_id: string | null` — identifies the Batch Labeling run that produced the label when the label originated from Batch Labeling. This field MUST be retained if the label is later promoted from `auto_labeled` to `verified`. It is null only when the label was never Auto-Labeled.

Pool assignment (§4.3; applies only to `label_status=verified`):

- `pool_assignment` ∈ {`test_pool`} | null (null = non-pool; set automatically by pool routing, §4.3.1; updated by rebalancing, §4.3.2; always null when `label_status=auto_labeled`)

**Promotion (Auto-Labeled → Verified):** when an SME reviews an Auto-Labeled example and clicks Save, the Label record's `label_status` transitions from `auto_labeled` to `verified`. The verification fields are populated, `labeled_at` is updated, and `inference_invocation_id` is updated to reference the proposal the SME reviewed. `batch_label_run_id` is retained for lineage.

**Deletion on semantic Core change:** when a semantic Core change occurs (§4.4.1), all Label records (both `verified` and `auto_labeled`) are deleted. Prior Verified label data is preserved as reference on the Example record (`prior_verified_label_ref`). Operation Records are preserved for audit.

Notes:

- Skip does not create a Label. If an Auto-Labeled Label already exists, Skip deletes that machine proposal in the same transaction as the Example's omission fields; its Operation Record remains for audit (§4.2, §4.5).
- Only `label_status=verified` labels are ground truth for evaluation, ICL selection, pool candidacy, and training export. `label_status=auto_labeled` labels are used for training scale only (§9.2).

### 13.8 Example Record (Required)

Persists each ingested image with its current state, provenance, and optional metadata. See §4.1 for state semantics, §4.2 for field rules, §4.4.1 for schema evolution workflow.

Minimum fields:

- `project_id` (required)
- `example_key` (required; unique within project)
- `storage_ref` (required; absolute filesystem path to the original image; §1.7)
- `ingested_at: timestamp` (required; server timestamp)
- `source_metadata: object` (required; JSON; may be `{}`)
- `state` ∈ {`Unlabeled`, `Auto-Labeled`, `Verified`, `Omitted`} (required)

pHash (§5.6; populated asynchronously after ingest):

- `phash: string | null` (required; hex-encoded perceptual hash; null while the sweep is pending or after a computation failure)

Omission provenance (required when `state="Omitted"`; null otherwise):

- `omitted_source` ∈ {`sme_skip`} | null
- `omitted_at: timestamp | null`

Embedding fields (§5.5.5):

- `clip_embedding_present: boolean` (default: `false`; set `true` when embedding is stored)
- `clip_embedding_dim: int | null`
- `clip_embedding_model_id: string | null`
- `embedding_provider` ∈ {`hosted_nvclip`, `self_hosted_nvclip`} | null

Prior-label reference fields (§4.4.1; set during semantic Core schema change; null otherwise):

- `prior_verified_label_ref: string | null` (JSON snapshot of prior label data; null for examples never verified under a prior schema)
- `prior_verified_outcome: "Accept" | "Edit" | null` (prior `verified_outcome`; used by review selector priority; null when no prior label)

Notes:

- pHash values are unaffected by schema changes (the image does not change; §5.6.3).

### 13.9 Guidance Record (Required)

Persists an immutable Guidance version: Description, SchemaCore, and Rules. See §4.4 for edit policy, SchemaCore field record structure, and type system.

Minimum fields:

- `project_id` (required)
- `guidance_id` (required; unique within project)
- `version_number: int` (required; 1-based, monotonically increasing within the project, assigned by the backend at creation time, immutable once assigned. This is the user-visible version number displayed as `v{version_number}` in the UI. `guidance_id` is the internal unique identifier.)
- `description: string` (required; may be empty string)
- `schema: object` (required; SchemaCore definition with field records per §4.4, each with `field_id`)
- `rules: string` (required; may be empty string)
- `created_at: timestamp` (required)
- `semantic_core_change_from_guidance_id: string | null` (set when this Guidance version was created by a semantic Core change from the referenced prior version; null for first version and non-semantic edits. This is the audit trail for schema evolution — it replaces the need for a separate ReVerificationRun record.)
- `schema_change_summary: object | null` (structured description of what changed in SchemaCore; set when `semantic_core_change_from_guidance_id` is set; null otherwise)

Notes:

- Guidance records are immutable once persisted. Editing creates a new record with a new `guidance_id` and the next `version_number`.
- Projects select active version via `project.active_guidance_id`.
- When a Guidance version is created via semantic Core change, `semantic_core_change_from_guidance_id` and `schema_change_summary` provide full audit lineage without requiring a separate run record.

### 13.10 ModelConfig Record (Required)

Persists a model catalog entry binding an endpoint, model name, operational metadata, eligible roles, and capability probe results. See §4.8 for catalog rules and role filtering.

Minimum fields:

- `project_id` (required)
- `model_config_id` (required; unique within project; backend-generated)
- `endpoint_id: string` (required; MUST reference `NimEndpoint.endpoint_id` within the same project, §13.18)
- `model_name: string` (required; NIM `model` parameter)
- `context_window_tokens: int` (required)
- `eligible_roles[]: string[]` (required; non-empty; values from {`teacher`, `student_base`})
- `supports_image_input: boolean` (required; whether the model accepts image content in messages; distinct from `visual_budget_mode` which tracks visual token control support; §4.8)
- `structured_generation_support` ∈ {`unknown`, `supported`, `unsupported`} (required; updated via probe §6.2)
- `thinking_toggle.mode` ∈ {`none`, `qwen_enable_thinking`, `kimi_thinking`} (required)
- `thinking_toggle_support` ∈ {`unknown`, `supported`, `unsupported`} (required)
- `visual_budget_mode` ∈ {`none`, `mm_processor_size`, `mm_processor_pixels`, `mm_processor_tiles`} (required; §6.9.1)
- `visual_budget_support` ∈ {`unknown`, `supported`, `unsupported`} (required; §6.9.2)
- `default_icl_max_examples: int | null` (optional; per-model default ICL depth cap consumed by §6.2 selection when no explicit `icl_max_examples` override is present. Seeded from the July 2026 cross-model depth studies; null = no default. Settable at create and PATCHable for operator re-tuning.)
- `model_quantization: string | null` (optional deployment metadata)
- `nim_model_profile: string | null` (optional deployment metadata)
- `nim_profile_metadata: object | null` (optional; backend/precision/TP per §2.3)
- `local_deploy_metadata: object | null` (optional; when present, enables system-managed local NIM deployment §1.5 Mode C. Fields: `nim_container_image` (pinned image ref, e.g., `nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0`), `nim_gpu_memory_minimum_gb` (support-matrix minimum), `preferred_host_port` (default host port for this model type), `extra_container_env: object | null` (optional operator-supplied env vars appended to the system-managed `docker run` as sorted `-e KEY=VALUE` pairs — the in-product vehicle for live-validated NIM remediations such as `NIM_DISABLE_CUDA_GRAPH=1` or `NIM_MAX_MODEL_LEN` clamps. Keys MUST be `UPPER_SNAKE_CASE` and MUST NOT shadow builder-owned env (`NGC_API_KEY`, `NIM_MODEL_NAME`, `NIM_MODEL_SIZE`, `NIM_MODEL_PROFILE`, `NIM_SERVED_MODEL_NAME`, `NIM_MAX_IMAGES_PER_PROMPT`, `NIM_MAX_VIDEOS_PER_PROMPT`); offending keys are skipped with a logged warning. Values render verbatim in the operator-visible command displays — never secrets. Student deploys reference the base ModelConfig and therefore inherit its env). Null for models without known local deployment metadata.)
- `tao_base_experiment_id: string | null` (nullable; UUID of the base experiment registered in the bootstrapped workspace. Null means "not yet provisioned"; suite launch creates the conditional Training Jobs provisioning stage, while TAO chain materialization still rejects a null value as its final invariant.)
- `tao_base_experiment_pull_status` ∈ {`unknown`, `starting`, `in_progress`, `pulling`, `pull_complete`, `invalid_pull`, `failed`} | null (nullable; cached registration lifecycle. The tracked first-use worker sets `pulling`, then `pull_complete` or `failed`; admin/eager paths may also write terminal state.)
- `created_at: timestamp` (required)

### 13.11 Pool Record (Required)

Persists frozen Test Pool version snapshots. Pool Records are auto-created when evaluation runs start (§4.3.3), capturing the current live Test Pool membership for reproducibility. Live pool assignment is tracked on Label records via `pool_assignment` (§13.7).

Minimum fields:

- `project_id` (required)
- `pool_id` (required; unique within project)
- `pool_type: "test_pool"` (required)
- `pool_version: int` (required; monotonically increasing within project)
- `member_example_keys[]: string[]` (required; frozen at creation; immutable)
- `member_count: int` (required)
- `guidance_id` (required; Guidance version active at snapshot creation)
- `created_at: timestamp` (required)

Notes:

- Pool version snapshots are immutable once created.
- Test Pool members are excluded from ICL selection (§6.2).
- Live pool state is the authoritative source; snapshots are point-in-time captures for evaluation reproducibility.

### 13.12 TAOJob Record (Required When Student Training Enabled)

Persists Cosmos-RL / TAO VLM job state, configuration, and tracked outputs. See §9.7.2 for state machine, §9.7.3 for configuration payload, §9.7.4 for polling contract, §9.7.5 for required outputs.

Minimum fields:

- `project_id` (required)
- `tao_job_id` (required; unique within project)
- `student_base_model_config_id` (required)
- `dataset_export_ids[]: string[]` (required; the `export_field_mode` is derived from the referenced DatasetExport records — all must share the same mode, validated at creation per §10.2.7)
- `action` ∈ {`train`, `evaluate`, `inference`, `quantize`} (required)
- `status` ∈ {`not_started`, `submitting`, `submitted`, `queued`, `running`, `paused`, `succeeded`, `failed`, `canceled`, `deleted`} (required; `not_started` for pre-created chain jobs awaiting predecessor completion; `submitting` for jobs with persisted submission intent awaiting TAO confirmation; §9.7.2)
- `tao_status_raw: string | null` (raw status from TAO API)
- `training_backend: "cosmos_rl_tao_vlm"` (required)
- `training_policy_type: "sft"` (required when `action=train`)
- `job_config: object` (required; structure per §9.7.3)
- `tao_create_job_request: object` (required; exact payload submitted to TAO)
- `tao_external_job_id: string | null` (job ID returned by TAO API)

Progress and outputs:

- `progress: object | null` (`epoch_current`, `epoch_total`, `eta_seconds`, `metrics_latest`, `metrics_history_ref`)
- `outputs: object` (`artifacts[]`, `logs_ref`, `metrics_ref`, `tao_job_metadata_ref`)

Linkage and lifecycle:

- `parent_tao_job_id: string | null` (for `quantize` and `evaluate` actions; links to parent training or quantize job)
- `preflight_result: object | null` (TAO reachable + model role check)
- `created_at: timestamp` (required)
- `started_at: timestamp | null`
- `completed_at: timestamp | null`
- `last_polled_at: timestamp | null`
- `error_ref: string | null` (sanitized error payload)
- `poll_error_ref: string | null`

Job chaining fields (§9.7.7):

- `chain_id: string | null` (groups all jobs for one base model into a chain; null for standalone jobs)
- `chain_sequence: int | null` (1-based position within the chain; null for standalone jobs)
- `chain_halted_reason: string | null` (set on remaining jobs when a chain halts due to a failed predecessor)

Outputs-fetch lifecycle (§1.3 background-task recovery + §9.7.5 required outputs; Phase 12 amendment 2026-05-05):

- `outputs_fetch_status` ∈ {`pending`, `in_progress`, `completed`, `failed`} (required; default `pending`). Tracks the post-`succeeded` "fetch artifacts + merge resolved fields + emit SSE + run downstream actions" flow that the polling service performs inline once a TAO job reaches `succeeded`. Lifecycle: at handler entry the marker flips `pending → in_progress` in a short transaction so a crash from there on leaves a recovery marker; at successful exit the handler flips it to `completed`; on exception the handler flips it to `failed` and persists the sanitized exception text on `outputs_fetch_error_ref` (the failure is logged but not re-raised so the polling loop continues). Status-machine invariant: `outputs_fetch_status` is INDEPENDENT of `status` — `status="succeeded"` says TAO finished; `outputs_fetch_status="completed"` says the Blueprint finished consuming the result.
- `outputs_fetch_error_ref: string | null` (sanitized exception text when `outputs_fetch_status="failed"`).

The TAO job detail response and each TrainingSuite chain-job summary MUST expose
both fields. The Training Job Monitor MUST continue polling a `succeeded` job
while `outputs_fetch_status` is `pending` or `in_progress`, render that state as
**Finalizing**, and explain that the downstream `not_started` job is waiting for
artifact processing. `outputs_fetch_status="failed"` MUST render as the distinct
**Artifact Failed** state with the sanitized diagnostic; it MUST NOT appear as a
completed job with a downstream sibling indefinitely "waiting for a predecessor."

Restart-recovery contract (Phase 12 amendment 2026-05-05): the polling service tick MUST scan, per project, for `status="succeeded" AND outputs_fetch_status IN (pending, in_progress)` and re-fire the success handler + chain-advance for each match. This closes the silent-halt failure mode where a backend restart mid-multi-GB-artifact-download leaves the chain frozen indefinitely (the legacy implementation only fired the handler once, on the tick that observed the status flip to `succeeded`). The recovery is idempotent: artifact downloads use atomic `.part` files truncated on retry; `register_from_tao_terminal` and `rescore_evaluate_job` tolerate re-execution; `_advance_after_terminal` only submits jobs still in `not_started`. `outputs_fetch_status="failed"` is terminal — the operator must manually flip back to `pending` to retry, otherwise auto-retry would mask real outages (S3 ACL bug, network partition, etc.). The live failure that motivated the contract (2026-05-04 full-stack validation, finding F7) is documented in the project's internal engineering archive.

**F8 (Phase 12 amendment 2026-05-05) — non-blocking post-success dispatch.** The post-`succeeded` flow (artifact fetch + Student registration / re-scoring + chain advance) MUST run as a non-blocking in-process background task, NOT inline in the polling tick. The tick — which scans every project sequentially per `tao_polling_service.tick()` — would otherwise stall for the duration of a multi-GB safetensors download (typically 5–15 min on the workspace S3), preventing every other project's chain from advancing. Implementation: `services/tao_polling_service._dispatch_post_success_flow` registers `_post_success_flow(...)` (the wrapper around `_handle_succeeded` + `_advance_after_terminal`) via `background_manager.register("post-success-{tao_job_id}", ...)`. The task_id is unique per `tao_job_id` and dedup'd at the registry level: a recovery scan running in the same tick that already saw a fresh `succeeded` transition MUST NOT register a second task for the same job. Both the regular terminal-handling path (`_poll_single_job` line 1294 onward) and the `_recover_stuck_outputs_fetch_in_project` recovery scan use the same dispatch helper. The live head-of-line-blocking failure that motivated the change (2026-05-05 full-stack validation, finding F8) is documented in the project's internal engineering archive.

Notes:

- State machine transitions per §9.7.2; terminal states: `succeeded`, `failed`, `canceled`, `deleted`.

### 13.13 StudentModel Record (Required When Students Registered)

Persists a trained or deployed Student model with full training lineage, checkpoint status, and deployment metadata. See §9.5 for registry and deployment rules.

Minimum fields:

- `project_id` (required)
- `student_model_id` (required; unique within project)
- `training_suite_id: string | null` (immutable parent Start Training action; required for suite-created Students, null only for legacy/ad-hoc TAO jobs outside a TrainingSuite)
- `student_base_model_config_id` (required; catalog entry with `student_base` role)
- `tao_job_id` (required; training TAOJob that produced this model)
- `guidance_id` (required; Guidance version used for training data)
- `dataset_export_ids[]: string[]` (required; training data used; the Student's effective field mode for its Inference Contract (§6.11) is derived from the `export_field_mode` on these DatasetExport records — all must share the same mode)
- `training_preset: string` (required)
- `lora_config: object` (required; from TAOJob per §9.7.3.2)
- `created_at: timestamp` (required)

Checkpoint status:

- `checkpoint_packaging_status` ∈ {`pending`, `validated`, `failed`} (required; §9.5.1)
- `nim_checkpoint_ref: string | null` (path/URI to NIM-loadable checkpoint directory)

Two-part readiness (§9.5):

- `quality_status` ∈ {`pending`, `validated`, `partial`, `failed`} (required; set to `validated` when TAO evaluate succeeds and canonical re-scoring produces acceptable metrics; `failed` when TAO evaluate fails; `partial` (F35) when a NIM-source eval finishes `incomplete` but produces parseable output on at least `STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD` of examples — informational, not gate-passing)
- `quality_evaluation_run_id: string | null` (links to the evaluation Run Record from TAO evaluate)
- `serving_status` ∈ {`pending`, `validated`, `failed`, `not_attempted`} (required; `validated` when NIM deployment + evaluation + benchmark succeed; `failed` when NIM deployment or evaluation fails; `not_attempted` when NIM evaluation has not been run)
- `serving_evaluation_run_id: string | null` (links to the evaluation Run Record from NIM evaluation)

The Student list/get API projection also derives
`serving_benchmark_current: bool` and `serving_benchmark_blocker: string |
null` from this referenced RunRecord. These are not additional persisted
columns. They fail closed for missing, legacy synthetic, incomplete, failed,
wrong-version, provenance-drifted, or incomplete-concurrency AIPerf evidence.

NIM deployment state (§9.5.2):

- `nim_preflight_status` ∈ {`passed`, `failed`} | null (result of local NIM preflight; null when not attempted)
- `nim_preflight_details: object | null` (per-check pass/fail with diagnostic messages)
- `nim_preflight_at: timestamp | null`
- `nim_deployment_mode` ∈ {`local`, `external`} | null (how NIM was deployed for evaluation; null when not deployed)
- `nim_container_id: string | null` (Docker container ID when locally deployed; null for external)
- `nim_endpoint_url: string | null` (NIM endpoint URL used for serving evaluation)

Deployment metadata (required on deployed Students):

- `nim_vlm_release_version: string | null` (required on deployed Students; pin NIM release)
- `nim_model_profile_requested: string | null`
- `nim_model_profile_selected: string | null`
- `nim_profile_metadata: object | null` (backend/precision/TP/optimization target)
- `gpu_type: string | null`
- `gpu_count: int | null`

Quantization provenance:

- `quantization_method: string | null` (required; e.g., `"FP8_DYNAMIC"`, `"W4A16"`, or `null` for full-precision baseline)
- `quantize_tao_job_id: string | null` (links to the TAO quantize job; null for full-precision)

Notes:

- Each precision variant is a distinct StudentModel record (§2.3).
- Quality validation (`quality_status="validated"`) requires TAO evaluate to succeed.
- Serving validation (`serving_status="validated"`) requires NIM deployment, evaluation, and the current AIPerf workload to succeed. Persisted legacy values remain auditable but project to `serving_benchmark_current=false`.
- The `deployment_handoff` Action Request (§10.3) and portable bundle require `quality_status="validated"`, `serving_status="validated"`, AND `serving_benchmark_current=true`.

### 13.14 PromptPackage Record (Removed in v1.0)

**Absent from v1.0** together with Prompt Package export (§10.4). The public schema has no `prompt_packages` table. Section number retained so existing cross-references stay resolvable.

### 13.15 LocalNimDeployment Record (Required When Local NIM Deployed)

Tracks system-managed local NIM containers (§1.5 Mode C). One record per active or recently-stopped local deployment.

Minimum fields:

- `project_id` (required)
- `local_nim_deployment_id` (required; unique within project)
- `model_config_id` (required; the catalog entry being deployed)
- `role` ∈ {`teacher`, `embedding`} (required; what this deployment serves)
- `nim_container_image: string` (required; exact image ref used, e.g., `nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0`)
- `container_name: string` (required; Docker container name assigned by the system)
- `container_id: string | null` (Docker container ID; null before container starts)
- `host_port: int` (required; resolved host port)
- `endpoint_url: string` (required; e.g., `http://localhost:8000/v1`)
- `gpu_assignment: string` (required; GPU device specification used, e.g., `device=0`)
- `status` ∈ {`starting`, `running`, `stopped`, `failed`} (required)
- `status_reason: string | null` (e.g., `"preflight_failed"`, `"startup_timeout"`, `"user_stopped"`, `"backend_shutdown"`, `"displaced_by_replace"`)
- `activate_on_success: boolean` (required; default false. Teacher-only activation intent used by post-onboarding NIM Configuration. When true, set the project's `teacher_model_config_id` only after verified healthy adoption; persisted so restart recovery preserves the intent.)
- `deployed_at: timestamp | null` (when the container became healthy)
- `stopped_at: timestamp | null`
- `created_at: timestamp` (required)
- `displaced_by_deployment_id: string | null` (F49 amendment, 2026-05-19; the `local_nim_deployment_id` whose acquire-GPU call stopped this deployment as part of one-NIM-per-GPU replace semantics; null otherwise)
- `displaced_at: timestamp | null` (F49 amendment, 2026-05-19; when the displacement occurred; null otherwise)
- `matches_active_role_config: boolean` (F-D4 amendment, 2026-07-15; **response-computed, not persisted** — whether the project's active config for this deployment's role still references the deployment's `model_config_id`. False on a `teacher` deploy whose Teacher the SME has since switched away from; the UI suppresses the stale failure banner on that signal. Config identity — not endpoint mode — because the ModelConfig only repoints to the `local_system_managed` endpoint on deploy *success*, so a fresh FTUE failure must stay visible. Always true for non-teacher roles.)

Notes:

- On backend startup, the system inspects containers by `container_name`. Running + healthy → rebind. Otherwise → mark `stopped`.
- The system MUST NOT discover or manage containers it did not create.
- `displaced_by_deployment_id` and `displaced_at` retain the replacement audit and identify residents eligible for best-effort restoration when their replacement fails. They also make "who stopped my Teacher" answerable from a single SQL query (§1.5 one-NIM-per-GPU invariant, §9.5.2 step 0).

### 13.16 ClipEmbedding Record (Required When Embeddings Computed)

Persists CLIP embedding vectors in a dedicated table, separate from the Example record. One-to-one with Example within a project. See §5.5.5 for the storage contract, cache lifecycle, and design rationale.

Minimum fields:

- `project_id` (required)
- `example_key` (required; unique within project; foreign key to Example)
- `embedding_provider` ∈ {`hosted_nvclip`, `self_hosted_nvclip`} (required)
- `clip_embedding_model_id: string` (required; `nvidia/llama-nemotron-embed-vl-1b-v2` for the supported NeMo Retriever VL embedding NIM)
- `clip_embedding_dim: int` (required; `2048` for NeMo Retriever VL)
- `vector_blob_f32: bytes` (required; float32 binary vector; length = `clip_embedding_dim × 4` bytes)
- `created_at: timestamp` (required)
- `updated_at: timestamp` (required)

Constraints:

- `PRIMARY KEY (project_id, example_key)`
- Foreign key to `Example(project_id, example_key)` with `ON DELETE CASCADE`

Notes:

- The `Example` record retains lightweight presence/summary fields (`clip_embedding_present`, `clip_embedding_dim`, `clip_embedding_model_id`, `embedding_provider`) for fast filtering and selector-mode checks without joining or reading the vector payload. The `Example` record MUST NOT store the full vector.
- On provider/model invalidation (§5.5.3), rows in this table are deleted or replaced. The in-memory cache rebuilds from this table.
- The in-memory embedding cache (§5.5.5) is loaded from this table with a single bulk `SELECT` on project open.

### 13.17 EmbeddingDeploymentConfig Record (Deployment-Scoped)

Persists desired-state configuration for the NeMo Retriever VL 1B v2 embedding service NIM. This record is **deployment-scoped** (shared across all projects), not project-scoped — an explicit exception to the project-scoping rule in §13.4.3. Recent NVIDIA Blueprints treat embedding models as dedicated service configuration, not as entries in the user-facing model catalog alongside LLM/VLM inference roles. The model catalog (§13.10) is reserved for interactive inference roles (Teacher, Student base); the embedding NIM is background infrastructure for embedding computation.

Runtime container state for a locally deployed embedding NIM instance is tracked in `LocalNimDeployment` (§13.15) with `role="embedding"`. The Project record (§13.4.1) stores only the effective embedding provider, model identity, and resolved endpoint reference for reproducibility — it does not store deployment metadata.

Minimum fields:

- `embedding_deployment_config_id: string` (required; singleton in v1 — one config per deployment)
- `provider` ∈ {`hosted_nvclip`, `self_hosted_nvclip`, `local_nvclip`, `none`} (required; the desired deployment mode. Enum value names retained for backwards compatibility with existing project DBs; the actual model is governed by `model_name`.)
- `model_name: string` (required; `nvidia/llama-nemotron-embed-vl-1b-v2`)
- `embedding_dim: int` (required; `2048`)
- `endpoint_url: string | null` (resolved endpoint URL; set for self-hosted or after local deployment succeeds; null for hosted when using the shared `NVIDIA_API_KEY` base URL)
- `nim_container_image: string` (required; pinned image ref, default `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0`)
- `preferred_host_port: int` (required; default `8001`; §1.5 Mode C)
- `gpu_memory_minimum_gb: int` (required; seeded default: 24 GB because the smallest GPU SKUs listed for NeMo Retriever VL 1B v2 NIM 2.0.0, L4 and A10G, have 24 GB. This is an eligibility floor, not a generic compatibility claim.)
- `gpu_assignment: string | null` (e.g., `device=1`; null when not locally deployed)
- `created_at: timestamp` (required)
- `updated_at: timestamp` (required)

Notes:

- The environment assessment endpoint (`GET /v1/environment`, §10.2.25) sources embedding-NIM deployment metadata from this record, not from the model catalog. The `local_deployable_models[]` response field covers catalog models; embedding NIM availability is reported via `recommended_embedding_mode` and the embedding-NIM-specific fields derived from this config.
- On local embedding NIM deployment, the system creates a `LocalNimDeployment` record (§13.15) with `role="embedding"` for runtime container lifecycle tracking, and updates the Project's `embedding_provider` and `embedding_endpoint_id` to reference the resolved local endpoint.
- This record is managed via the split NIM setup chain and the post-onboarding
  NIM Configuration screen (`NIMConnectionPage.tsx`), not via project CRUD.

### 13.18 NimEndpoint Record (Required)

Persists a configured NIM-compatible inference endpoint. NVIDIA NIM defines a concrete endpoint surface with standard OpenAI-compatible paths (`/v1/models`, `/v1/health/ready`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/metrics`). This record captures the transport, auth, and health configuration for that surface. Model identity (`model_name`, context window, roles, capabilities) remains on `ModelConfig` (§13.10); `ModelConfig.endpoint_id` references the `NimEndpoint` the model is served from.

`NimEndpoint` and `LocalNimDeployment` (§13.15) are separate concerns. `NimEndpoint` is the logical endpoint configuration the system points at. `LocalNimDeployment` is the runtime container state for system-managed local NIMs. A local deployment creates or updates a `NimEndpoint` with `endpoint_mode="local_system_managed"` and links via `local_nim_deployment_id`; the endpoint identity remains stable if the container is stopped and restarted.

Minimum fields:

- `project_id` (required)
- `endpoint_id` (required; unique within project; backend-generated UUID4)
- `display_name: string` (required; e.g., `"NVIDIA Hosted NIM"`, `"Self-hosted LAN"`, `"Local Teacher"`)
- `endpoint_mode` ∈ {`hosted`, `self_hosted`, `local_system_managed`} (required; maps to §1.5 Modes A/B/C)
- `base_url: string` (required; e.g., `https://integrate.api.nvidia.com/v1`, `http://10.0.1.50:8000/v1`, `http://localhost:8000/v1`)
- `api_format: "openai_compatible"` (required; all NIM endpoints expose the OpenAI-compatible surface)
- `auth_mode` ∈ {`bearer`, `none`} (required; `bearer` for hosted with `NVIDIA_API_KEY`; `none` for self-hosted and local endpoints, which are expected to run on a trusted private network or behind an external gateway — the app attaches no credential)
- `models_path: string` (default `/models`; appended to a base URL that already includes `/v1`)
- `health_ready_path: string` (default `/health/ready`; used for local NIM startup polling)
- `health_live_path: string | null` (default `/health/live`)
- `metrics_path: string | null` (default `/metrics`; Prometheus metrics, available on self-hosted/local NIM only)
- `is_enabled: boolean` (default `true`; allows disabling without deletion)
- `last_probe_at: timestamp | null`
- `last_probe_status` ∈ {`unknown`, `healthy`, `unhealthy`, `auth_failed`, `unreachable`} (default `unknown`; updated by probe)
- `last_probe_error_ref: string | null` (sanitized error from last failed probe)
- `source_kind` ∈ {`seeded_hosted`, `user_configured`, `auto_registered_local`} (required; how the endpoint was created: seeded at project creation, configured by the SME, or auto-registered by local NIM deployment)
- `local_nim_deployment_id: string | null` (references `LocalNimDeployment` when `endpoint_mode=local_system_managed`; null otherwise)
- `created_at: timestamp` (required)
- `updated_at: timestamp` (required)

Foreign key relationships:

- `ModelConfig.endpoint_id` MUST reference `NimEndpoint.endpoint_id` within the
  same project. This binding is explicit: generic endpoint create/update never
  changes it; `PATCH .../model_configs/{id}` or the dedicated verified
  self-hosted-Teacher configuration operation changes it intentionally.
- `OperationRecord.endpoint_id` MUST reference the exact `NimEndpoint` used for that invocation.

Notes:

- On project creation, the system seeds a default `NimEndpoint` with `endpoint_mode="hosted"`, `base_url` from `HOSTED_NIM_BASE_URL` (Appendix A.4), `auth_mode="bearer"`, and `source_kind="seeded_hosted"`. All 8 seeded ModelConfig entries (§4.8) reference this endpoint.
- When local NIM deployment succeeds (§1.5 Mode C), the system creates a `NimEndpoint` with `endpoint_mode="local_system_managed"`, the resolved local URL, and `source_kind="auto_registered_local"`, linked to the `LocalNimDeployment` record.
- `NimEndpoint` records do NOT store API key values. The only credential is the deployment-level `NVIDIA_API_KEY` for hosted (`auth_mode="bearer"`) endpoints, resolved at runtime from secret storage (`.env` / environment variables, §12.1). Self-hosted and local endpoints carry no credential.
- Post-onboarding NIM Configuration (`NIMConnectionPage.tsx`) durably creates
  or reuses and binds a verified self-hosted endpoint. The seeded hosted
  endpoint remains the hosted path; local deployment auto-registers its own
  endpoint only after verification.

### 13.19 TAODeploymentConfig Record (Deployment-Scoped; Required When Student Training Enabled)

Persists deployment-level TAO workspace configuration — the identity of the single TAO workspace this Blueprint deployment uses and the S3 credentials needed to upload dataset archives into that workspace's backing store (§9.7.8). Like :class:`EmbeddingDeploymentConfig` (§13.17), this record is deployment-scoped (shared across all projects), NOT project-scoped — an explicit exception to the project-scoping rule in §13.4.3. The TAO workspace is a shared organizational resource; cloning it per-project would proliferate storage buckets and conflict with TAO's workspace ownership model.

Minimum fields:

- `tao_deployment_config_id: string` (required; singleton in v1 — one record per deployment)
- `tao_workspace_id: string | null` (the UUID of the workspace as returned by `POST /orgs/{org}/workspaces`. Null indicates "bootstrap not yet run"; the training preflight endpoint §10.2.22 rejects training until this is set.)
- `tao_workspace_name: string | null` (human-readable workspace name; set by the bootstrap CLI)
- `tao_workspace_cloud_type: "seaweedfs" | "aws" | "azure" | "self_hosted" | "huggingface" | "lepton" | "slurm" | null` (the backing store type)
- `tao_workspace_bucket: string | null` (the S3 bucket the workspace owns; set by bootstrap)
- `tao_workspace_s3_endpoint_url_internal: string | null` (endpoint URL as TAO's own containers see it, e.g., `http://seaweedfs-s3:8333` on a compose-bundled deployment. Used in TAO's job specs for dataset paths that TAO resolves.)
- `tao_workspace_s3_endpoint_url_external: string | null` (endpoint URL the Blueprint uses to upload archives — may be a tunneled or LAN-routable URL, e.g., `http://127.0.0.1:8333` when the Blueprint reaches SeaweedFS via an SSH tunnel.)
- `tao_workspace_s3_access_key_ref: string | null` (reference label for the S3 access key; the credential itself lives in `.env` or a secret store, per §12.1)
- `tao_workspace_s3_secret_key_ref: string | null` (reference label for the S3 secret key)
- `bootstrap_status: "not_bootstrapped" | "in_progress" | "bootstrapped" | "failed"` (required; default `"not_bootstrapped"`)
- `bootstrap_last_run_at: timestamp | null`
- `bootstrap_error_ref: string | null` (sanitized error from the last failed bootstrap attempt)
- `created_at: timestamp` (required)
- `updated_at: timestamp` (required)

Notes:

- The training preflight endpoint (§10.2.22) reads this record to check workspace reachability (`bootstrap_status == "bootstrapped"` AND the workspace UUID resolves via `GET /orgs/{org}/workspaces/{workspace_id}`). A failing check returns a plain-language next step referencing the bootstrap CLI.
- The per-training dataset upload service (§9.7.8.2) reads the S3 endpoint + credentials from this record to upload dataset archives into the workspace bucket before submitting TAOJobs.
- Actual S3 credential values (access_key, secret_key) are NOT stored in this record; only their environment-variable reference names are stored. Live secrets flow through the process environment or `~/.vlm_feedback_loop/.env`.
- This record is managed by the bootstrap CLIs and read by preflight, dataset upload, and first-use base provisioning. The project-scoped provisioning endpoint creates a deployment-scoped run but does not mutate workspace identity.

### 13.20 TAOBaseExperimentProvisioningRun Record (Deployment-Scoped)

Tracks one first-use or eager base-registration attempt against the shared TAO
workspace.

Minimum fields:

- `provisioning_run_id: string`
- `project_id: string` (request origin and endpoint access scope)
- `requested_model_config_ids: string[]` (missing project-local selections)
- `requested_model_names: string[]` (canonical deployment-wide targets)
- `status: "queued" | "running" | "succeeded" | "failed"`
- `registered: string[]`
- `already_registered: string[]`
- `failures: {target: string, error: string}[]`
- `error_ref: string | null` (redacted)
- `started_at`, `completed_at`, `created_at`, `updated_at`

An interrupted `queued` or `running` row becomes `failed` during startup
recovery. The associated incomplete ModelConfig rows become
`tao_base_experiment_pull_status="failed"` while any already
`pull_complete` row is preserved.

---

## Appendix A: Algorithms and Contracts

### A.1 Markers and IDs

1. Backend MUST NOT include `example_key` in model-visible prompt text.
2. ICL examples in prompt MUST be labeled `E01`, `E02`, …
3. Backend MUST persist `icl_example_keys_used[]` for every call (interactive, evaluation, batch labeling).

### A.2 Exact Match (Core Fields) (Normative Summary)

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
- **Per-value precision, recall, and F1** (categorical Core fields only - enum, enum set, boolean): for each allowed value, compute precision (true positives / predicted positives), recall (true positives / actual positives), and F1 (harmonic mean). For enum set fields, membership of each value is evaluated independently.

All derived metrics MUST be reported alongside the overall Exact Match rate in evaluation results (§7) and Student comparison results (§5.4).

### A.3 Diversity-Driven Review Selector Scheduling

The review selector uses a greedy max-min diversity algorithm with two tiers of similarity signal. The algorithm structure is identical for both; only the similarity function differs.

**Mode selection** (when `REVIEW_SELECTION_MODE=auto`):

1. If the number of eligible examples with CLIP embeddings ≥ `CLIP_SWITCHOVER_MIN_COUNT` (default: 50) → use CLIP-diverse mode.
2. Otherwise → use pHash-diverse mode for candidates with hashes; if every candidate hash is null, use deterministic `example_key` order (§5.6).

Forced modes: `REVIEW_SELECTION_MODE=clip_diverse` forces CLIP (falls back to pHash if fewer than `CLIP_SWITCHOVER_MIN_COUNT` CLIP embeddings exist); `phash_diverse` forces pHash.

**Similarity functions:**

CLIP-diverse:

```text
sim_clip(a, b) = dot(emb(a), emb(b)) / (||emb(a)|| * ||emb(b)||)
```

pHash-diverse:

```text
sim_phash(a, b) = 1 - hamming_distance(phash(a), phash(b)) / hash_bits
```

**Inputs:**

- Eligible examples `U = { e | state(e)=Unlabeled or state(e)=Auto-Labeled }`
- Per-example similarity data: CLIP embedding (when `clip_embedding_present(e)=true`; loaded from in-memory cache, §5.5.5) and/or `phash(e)`
- Recent history list `H` of last `REVIEW_RECENT_WINDOW_K` presented examples
- Active similarity function `sim` (selected by mode above)

**Prior-label priority rule (§4.4.1, §6.5):**

Before diversity selection, the selector partitions `U` into two tiers:

- **Tier 1 (prior-label):** `U_prior = { e ∈ U | prior_verified_label_ref(e) is not null }` — examples that were previously Verified under a prior schema and need re-labeling. Within this tier, ordered by: (1) `schema_change_context_example_key` first (if set on Project), then (2) examples with `prior_verified_outcome=Edit`, then (3) examples with `prior_verified_outcome=Accept`.
- **Tier 2 (standard):** `U_standard = U \ U_prior` — examples with no prior label.

The selector exhausts Tier 1 before presenting Tier 2 examples. Within each tier, diversity selection applies normally (steps 1–5 below). Once `schema_change_context_example_key` has been presented, it is cleared from the Project record.

**Deterministic greedy diversity selection:**

1. Candidate set with the active signal:

```text
C = { e ∈ U_active_tier | signal(e) is not null }
```

(where `signal` = `clip_embedding_present` in CLIP-diverse mode, `phash` in pHash-diverse mode; `U_active_tier` is the current tier being served)

2. Recent window with the active signal:

```text
H_sig = [ h ∈ H | signal(h) is not null ]
```

3. If `H_sig` empty: select first candidate in deterministic stable order (ascending `example_key`).
4. Else for each `e ∈ C` compute:

```text
score(e) = max_{h in H_sig} sim(e, h)
```

Select:

- `e* = argmin_e score(e)` over `C`
- tie-break by stable ordering (ascending `example_key`)

5. If `C` empty (no candidates have the active signal): select first eligible example by `example_key` ascending. This is expected while the post-ingest pHash/CLIP workers are still populating signals and remains the safe fallback after per-image signal failures.

Omit-on-Skip: per §4.5. Omitted examples excluded from review selector and Batch Labeling inputs by default.

Reproducibility requirements:

- Candidate sets computed from persisted state.
- Similarity data retrieval stable for a given ref/hash.
- Similarity scoring deterministic.
- Tie-breakers explicit and stable.

### A.4 Reference Configuration Defaults (Canonical List)

Note: Settings in this section may be treated as optional overrides. When a value is `null`, the system MUST derive an effective value automatically from the selected model’s `context_window_tokens`, invocation purpose, and runtime feedback, and MUST enforce budgets by relevance-tail pruning of ICL examples (§6.2) rather than blocking progress.

Embedding computation:

```yaml
EMBEDDING_PROVIDER: auto
EMBEDDING_MODEL_ID: nvidia/llama-nemotron-embed-vl-1b-v2
EMBEDDING_DIM: 2048
EMBEDDING_INPUT_TYPE: passage
EMBEDDINGS_AUTO_COMPUTE: true
EMBEDDING_CONCURRENCY_HOSTED: 1
EMBEDDING_BATCH_SIZE_HOSTED: 8
EMBEDDING_CONCURRENCY_SELF_HOSTED: 4
EMBEDDING_BATCH_SIZE_SELF_HOSTED: 1
```

Note: `EMBEDDING_PROVIDER=auto` means the §5.5.1 cascade: prefer a healthy local embedding NIM deployment (live-verified), else the hosted embedding NIM when the hosted NIM API key is configured, else `none`. `EMBEDDING_MODEL_ID` targets NVIDIA NeMo Retriever VL 1B v2 (2048-dim; requires `input_type` in the request body, default `"passage"`). The `*_nvclip` provider values and `LOCAL_NIM_NVCLIP_PORT` setting name are retained only for compatibility with existing databases and deployments.

Generation Controls (§6.7):

```yaml
THINKING_DEFAULT_ON: true
```

Per-project (Project record §13.4, adjustable via the project API — not process config): `labeling_generation_preset_key: precise`.

Labeling preset definitions:

```yaml
precise:  { temperature: 0.0, top_p: 1.0 }
explore:  { temperature: 0.3, top_p: 0.9 }
```

Note: these are policy defaults; actual effective values are always persisted in Operation Records (§13.1).

Visual Budget Controls (§6.9):

Per-project (Project record §13.4): `visual_budget_preset_key: high_detail`.

Visual budget preset definitions (resolve per `visual_budget_mode`):

```yaml
fast:
  mm_processor_size:   { size: { shortest_edge: 1568, longest_edge: 65536 } }
  mm_processor_pixels: { images_kwargs: { min_pixels: 1568, max_pixels: 65536 } }
  mm_processor_tiles:  { max_num_tiles: 8 }
balanced:
  mm_processor_size:   { size: { shortest_edge: 1568, longest_edge: 131072 } }
  mm_processor_pixels: { images_kwargs: { min_pixels: 1568, max_pixels: 131072 } }
  mm_processor_tiles:  { max_num_tiles: 16 }
high_detail:
  mm_processor_size:   { size: { shortest_edge: 1568, longest_edge: 262144 } }
  mm_processor_pixels: { images_kwargs: { min_pixels: 1568, max_pixels: 262144 } }
  mm_processor_tiles:  { max_num_tiles: 32 }
```

Note: preset parameter values are internal policy defaults (not NVIDIA-specified); adjust per deployment based on model documentation and observed visual token consumption. Cosmos Reason2 is documented to perform best with ≤16k multimodal tokens per image.

Prompt budgets:

```yaml
RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE: null
RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN: 0.85
BASE_OUTPUT_TOKENS_FLOOR: 256
JSON_STRUCTURAL_OVERHEAD_TOKENS: 48
MAX_OUTPUT_FRACTION: 0.25
RATIONALE_NOTE_ESTIMATE_TOKENS: 160
DEFAULT_UNBOUNDED_STRING_BUDGET: 200
MODEL_REASONING_HEADROOM_TOKENS: 16384
```

Note: `RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE` replaces the derived output budget when set. Under normal operation it is `null` and the system derives `max_output_tokens` per invocation from SchemaCore (§6.2). `RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN` defaults to `0.85` (15% reserved). None of the seeded catalog models have tiktoken encoder mappings; all use the `cl100k_base` fallback, so token counts are approximate. The 0.85 margin absorbs tokenizer mismatch while preserving reasonable context utilization (~217k effective input on a 256k window). Diversity-based ICL pruning (§6.2) handles overflow gracefully: the consequence of a too-tight margin is dropping one extra ICL example, not a system failure. Provider-reported calibration (tracking the ratio of NIM-reported `usage.prompt_tokens` to tiktoken estimates and dynamically adjusting the margin) is a candidate v1.1 enhancement requiring no architectural changes.

Configuration validation requires positive token budgets, fractions and safety
margins in `(0, 1]`, non-negative structural overhead, and valid sampling
pairs (`temperature >= 0`, `0 < top_p <= 1`). Invalid values stop startup
with the affected key instead of reaching an inference run.

`MODEL_REASONING_HEADROOM_TOKENS` defaults to `16384`. This is the fixed headroom added to the schema-derived output budget when `thinking_mode_effective="on"`. Reasoning tokens consume the normal generation budget on vLLM-backed reasoning models (bounded by `max_tokens` unless `thinking_token_budget` is set separately); they are not free even when separated as `reasoning_content` in the response. NVIDIA's Cosmos Reason2 model guidance recommends `max_tokens=4096` or more to avoid truncating chain-of-thought responses. The `16384` default provides generous margin above the `4096` minimum floor, covering multi-field schemas and complex reasoning without requiring adaptive adjustment. Minimum acceptable value: `4096`. The automatic retry on truncation (§6.7.6 Step 6) handles individual invocations that exceed the budget; `RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE` is the deployment-level escape hatch.

ICL selection:

```yaml
ICL_MAX_EXAMPLES: null
```

Note: `ICL_MAX_EXAMPLES: null` (the default) defers the selection cap to the Teacher model's `default_icl_max_examples` (§6.2); setting it to a positive integer is an explicit deployment-wide override that replaces every per-model default. Zero and negative values are invalid; use the supported per-run `icl_mode="disabled"` where zero-shot execution is required (§7.1, §8.3). Note: ICL field rendering is governed exclusively by the effective Inference Contract (`icl_field_mode`, §6.11). There is no separate config key for ICL rendering mode. The Teacher always uses `icl_field_mode=core_only`; a Student uses the field mode derived from its training/export lineage. Valid field modes are `all`, `aux_and_core`, and `core_only`.

Export field mode:

Per-project (Project record §13.4): `export_field_mode: all`.

Note: `all` remains the default to preserve available auxiliary/audit data in training exports. Set to `aux_and_core` to omit explicit rationale text, or `core_only` for labels-only training (smallest output, lowest inference cost). This export choice is independent of the production Teacher's Core-only ICL contract.

Test Pool:

Per-project (Project record §13.4): `test_pool_fraction: 0.40`.

Note: 40% of Verified examples are reserved for evaluation. Both Accepts and Edits are candidates for the Test Pool, producing a representative evaluation set that reflects actual task difficulty. ICL draws exclusively from Edits (the corrective signal); Accepts are not selected for ICL.

Scale-Up Readiness Gate (§7.3):

Per-project (Project record §13.4): `scaleup_exact_match_threshold: 0.80`, `scaleup_per_field_match_threshold: 0.80`, `scaleup_min_per_value_f1_threshold: 0.60`, `scaleup_accept_rate_threshold: 0.80`, `scaleup_accept_rate_window: 50`, `scaleup_min_test_pool_size: 60`.

Note: All thresholds are configurable per project (project API §10.2.13 — not process config). The defaults are 80% overall accuracy, 80% per-field, 60% minimum per-value F1, and 80% Accept rate. The lower per-value F1 floor tolerates sparse-category variance while still catching severe systematic failures. The 60-member Test Pool floor gives the common six-class datasets about ten held-out examples per class when balanced while remaining modest for three-class smoke datasets. Lower thresholds accept more risk in Auto-Labeled data quality; higher thresholds delay Batch Labeling but improve data quality.

Batch labeling defaults:

```yaml
BATCH_LABEL_RUN_LIMIT: null
BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD: 10
BATCH_LABEL_CONCURRENCY_HOSTED: 1
BATCH_LABEL_CONCURRENCY_SELF_HOSTED: 8
```

Note: `BATCH_LABEL_RUN_LIMIT` caps the total number of examples processed in a single batch labeling run and MUST be a positive integer when set. Default `null` means all eligible examples are processed. `BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD` MUST also be positive. These values are validated at startup. The run limit is a cost-control and incremental-batching knob, distinct from `ICL_MAX_EXAMPLES` which governs per-invocation ICL selection for all purposes including batch labeling. `BATCH_LABEL_CONCURRENCY_HOSTED` / `BATCH_LABEL_CONCURRENCY_SELF_HOSTED` set the provider-aware dispatch width (§8.2 step 6), overridable per run via the start request's `concurrency` field.

TAO endpoint (deployment-level; §1.6):

```yaml
TAO_API_BASE_URL: unset
TAO_API_KEY: unset
TAO_ORG_NAME: unset
TAO_WORKSPACE_S3_ACCESS_KEY: unset
TAO_WORKSPACE_S3_SECRET_KEY: unset
```

Note: `TAO_API_BASE_URL`, `TAO_API_KEY`, and `TAO_ORG_NAME` are required when Student Training is enabled; workspace identity, bucket, and endpoint URLs are read from `deployment.db.tao_deployment_configs` (populated by `tao-bootstrap`). No defaults; must be explicitly configured. The two workspace S3 credential settings are required by the dataset-upload service (§9.7.8.2) and by the self-service base-experiment upload (§9.7.8.1a). `TAO_API_KEY`, `TAO_WORKSPACE_S3_ACCESS_KEY`, and `TAO_WORKSPACE_S3_SECRET_KEY` are supplied as secrets through the process environment or canonical `.env` (§12) before bootstrap; the CLI never accepts them as arguments or persists process-only values. On a fresh FTMS instance the operator configures these credentials, then bootstrap records only the resulting non-secret workspace state in `deployment.db`.

Student training defaults:

```yaml
TAO_RELEASE_VERSION: "6.26.3"
COSMOS_RL_CONTAINER_TAG: "6.26.3-cosmos-rl"
STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD: 0.90
```

Note: `TAO_RELEASE_VERSION` and `COSMOS_RL_CONTAINER_TAG` are the pinned versions this Blueprint is built and tested against; they are persisted on every TAOJob record for reproducibility. `STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD` (F35, 2026-05-06) is the parseable-rate threshold a NIM-source eval must clear to promote a paired Student to `quality_status="partial"` when the run finishes `incomplete` (§9.5). Default 0.90 matches the Phase 12 Final Report's near-miss case (135/148 = 91% parseable). Operator-overridable in `config.yaml`; values must lie in `(0.0, 1.0]`.

NIM serving benchmark:

```yaml
NIM_STARTUP_TIMEOUT_S: 1200
NIM_BENCHMARK_TIMEOUT_S: 1200
STUDENT_LATENCY_TEST_CONCURRENCIES: [1,8,24]
```

Note: `NIM_STARTUP_TIMEOUT_S` bounds container startup including runtime artifact build. `NIM_BENCHMARK_TIMEOUT_S` bounds the evaluation + concurrency sweep per variant. On timeout, the variant is marked and the queue continues. No global workflow timeout in v1.

Review selector selection defaults:

```yaml
REVIEW_SELECTION_MODE: auto
REVIEW_RECENT_WINDOW_K: 20
CLIP_SWITCHOVER_MIN_COUNT: 50
```

Note: `REVIEW_SELECTION_MODE=auto` means: CLIP-diverse when ≥ `CLIP_SWITCHOVER_MIN_COUNT` eligible examples have CLIP embeddings, pHash-diverse otherwise. Other values: `clip_diverse` (force CLIP, pHash fallback below threshold), `phash_diverse` (force pHash). There is no random selection mode.

Schema refinement reminders (§6.8):

```yaml
SCHEMA_REFINEMENT_REMINDER_THRESHOLD_1: 10
SCHEMA_REFINEMENT_REMINDER_THRESHOLD_2: 35
```

Note: Set either to `0` to disable that reminder. Both reset on semantic Core change so the SME gets fresh reminders after a schema change. With default settings, the first schema reminder fires at 10 Verified labels; the first evaluation recommendation fires at ~13 (when the Test Pool reaches 5 members at 40% fraction). The schema reminder intentionally precedes evaluation so the SME can adjust while re-labeling cost is low.

Rationale display (§6.3):

Per-project (Project record §13.4): `rationale_anti_anchoring: true`.

Evaluation concurrency (§7.1):

```yaml
EVAL_CONCURRENCY_HOSTED: 1
EVAL_CONCURRENCY_SELF_HOSTED: 8
EVAL_FIRST_POOL_SIZE: 5
```

HTTP client (§11):

```yaml
HTTP_DEADLINE_INTERACTIVE_S: 180
HTTP_DEADLINE_BACKGROUND_S: 300
HTTP_MAX_RETRIES: 3
```

Note: Background deadline covers evaluation, batch labeling, and embedding calls. Interactive deadline covers proposals. Retries use exponential backoff with jitter (1s, 2s, 4s). Only transient errors (429/502/503/504/connection failures) are retried.


Operational logging (§11):

```yaml
LOG_LEVEL: info
```

Note: Set to `debug` during development for verbose decision traces (candidate lists, scoring vectors, token breakdowns). `info` is the production default.

Workspace, server, and NIM endpoint quickstart:

```yaml
WORKSPACE_ROOT: unset
BIND_HOST: 127.0.0.1
BIND_PORT: 8000
IMAGE_ROOT: unset
HOSTED_NIM_BASE_URL: https://integrate.api.nvidia.com/v1
NVIDIA_API_KEY: unset
NGC_API_KEY: unset
LOCAL_NIM_TEACHER_PORT: 8000
LOCAL_NIM_NVCLIP_PORT: 8001
```

Note: `LOCAL_NIM_TEACHER_PORT` and `LOCAL_NIM_NVCLIP_PORT` are preferred host ports for local NIM deployment (§1.5 Mode C). The container always listens on port 8000 internally. If the preferred host port is occupied, the system uses the next available port and persists the resolved endpoint URL.

`IMAGE_ROOT` is the single absolute directory tree exposed to filesystem browsing, scanning, ingestion, image serving, and path remapping (§10.2.10). When unset and `BIND_HOST` is loopback (`127.0.0.1`, `::1`), its effective value is `/` (local-only development mode); when the bundled sample is available, the UI's initial view starts at its parent so the sample directory itself is selectable without narrowing that boundary. When unset and `BIND_HOST` is non-loopback, filesystem access is disabled until it is configured (§12.2). The shipped Compose stack sets `/data/images`.

### A.5 Pool Diversity Selection (Rebalancing)

When rebalancing promotes non-pool Verified examples into the Test Pool (§4.3.2), the system uses the review selector's switchover policy (§4.3.2, Appendix A.3) to select a single diversity signal per rebalancing pass — no mixed-mode scoring.

Inputs:

- Candidate set `P_cand`: non-pool Verified examples (both Accept and Edit)
- Current pool members `P_curr` (may be empty on first fill)
- Number of slots to fill: `n_needed`
- Per-example: CLIP embedding (optional) and pHash (nullable while pending or failed, §5.6)

Selection procedure:

1. Determine active similarity function using the same switchover rule as the review selector (Appendix A.3): if the number of eligible examples with CLIP embeddings ≥ `CLIP_SWITCHOVER_MIN_COUNT` (default: 50), use `sim_clip` (cosine similarity). Otherwise, use `sim_phash` (hamming similarity per §5.6.2). Do not mix CLIP and pHash within a single rebalancing pass.
2. Let `C = { e ∈ P_cand | signal(e) is not null }` (where `signal` = `clip_embedding_present` or `phash` depending on active function).
3. Let `P_ref = P_curr` (existing pool members with the active signal).
4. For `i = 1..n_needed`:
  a. If `P_ref` is empty: select the first candidate in `C` by `example_key` ascending.
  b. Else: for each `e ∈ C`, compute `score(e) = max_{r ∈ P_ref} sim(e, r)`.
  c. Select `e* = argmin_e score(e)` (most dissimilar from current pool). Tie-break by `example_key` ascending.
  d. Add `e*` to `P_ref`. Remove `e*` from `C`.
5. If `n_needed` exceeds `|C|`, fill remaining slots from `P_cand` members without the active signal by `example_key` ascending.

This is the same greedy max-min diversity approach used by the review selector (Appendix A.3), applied to pool construction instead of review ordering.

---

## Appendix B: Acceptance Tests / Validation Checklist

**Project CRUD, workspace, and storage (§1.7, §13.4):**
- Verify: `POST /v1/projects` with `name` creates project directory at `{workspace_root}/projects/{project_id}/` containing subdirectories `exports/`, `artifacts/`, `logs/`.
- Verify: `POST /v1/projects` initializes `project.db` inside the project directory with all record table schemas from §13 present (empty).
- Verify: `POST /v1/projects` returns the full Project record with all defaults from §13.4.1: `labeling_generation_preset_key="precise"`, `thinking_default_on=true`, `visual_budget_preset_key="high_detail"`, `structured_generation_mode_default="auto"`, `rationale_anti_anchoring=true`, `auto_evaluate_enabled=false`, `export_field_mode="all"`, `test_pool_fraction=0.40`, `phash_algorithm="dct_phash_64"`, and all Scale-Up gate thresholds at defaults.
- Verify: `POST /v1/projects` with missing `name` returns 400/422.
- Verify: `GET /v1/projects/{project_id}` returns 404 for non-existent project ID.
- Verify: `GET /v1/projects` returns project list with `counts` object (verified, unlabeled, auto_labeled, omitted, pending_relabel) and `next_cursor` pagination. A newly created project shows all counts at zero.
- Verify: `PATCH /v1/projects/{project_id}` accepts `name`, `description`, and selection field updates; rejects unknown fields.
- Verify: `PATCH /v1/projects/{project_id}` rejects cross-project ID references (e.g., `teacher_model_config_id` referencing an ID from another project).
- Verify: `PATCH /v1/projects/{project_id}` with `teacher_model_config_id` pointing at a catalog entry that does NOT have `teacher ∈ eligible_roles` returns HTTP 422 with a message naming the missing role (top-bar Teacher picker contract, `TeacherModelPicker.tsx`; Eng Spec §4.8 invariant).
- Verify: `PATCH /v1/projects/{project_id}` with `teacher_model_config_id` pointing at a catalog entry with `supports_image_input=false` returns HTTP 422 with an image-input-support message (§4.8 invariant).
- Verify: `PATCH /v1/projects/{project_id}` with `teacher_model_config_id` pointing at a different teacher-role catalog entry in the same project returns HTTP 200 and persists the new value (happy path for the top-bar Teacher picker).
- Verify: if `WORKSPACE_ROOT` directory does not exist, the system creates it (and the `projects/` subdirectory) on first project creation.
- Verify: all 16 record families listed in §13.4.3 include `project_id`; a query for records in project A never returns records from project B.
- Verify: large artifacts (raw model responses, validation reports) are stored as files under `{project_dir}/artifacts/`, not as database blobs. Database records store only `_ref` file-path references.
- Verify: `feature_flags` on Project record is an object that accepts the flags read by the implementation: `REVIEW_SELECTION_MODE`, `CLIP_SWITCHOVER_MIN_COUNT`.

**Database and migrations (§1.8):**
- Verify: SQLite WAL mode is enabled (`PRAGMA journal_mode` returns `wal`) for every project database.
- Verify: SQLite `busy_timeout` is configured to at least 5000 ms (`PRAGMA busy_timeout` returns ≥ 5000).
- Verify: on startup, `PRAGMA quick_check` runs on each opened project database. On check failure, the backend surfaces a clear error with the database path and does not proceed.
- Verify: Alembic manages all schema migrations. On startup, pending migrations are detected and applied before the project is usable.
- Verify: before applying Alembic migrations, the backend copies `project.db` to `project.db.backup.{ISO 8601 timestamp}`. Confirm backup file exists after a migration.
- Verify: if a migration fails, the backend surfaces the error and the backup file path, does not proceed with the partially migrated database, and the original backup is still intact.
- Verify: schema version is tracked in the database (Alembic version table present and populated).
- Verify: write transactions do not span outbound HTTP calls or long-running operations. A synthetic test holding a write lock for >5s triggers `busy_timeout` on a concurrent writer, confirming the timeout is enforced.
- Verify: on fresh system initialization, exactly one `EmbeddingDeploymentConfig` singleton record exists with all required fields populated with seeded defaults: `model_name="nvidia/llama-nemotron-embed-vl-1b-v2"`, `embedding_dim=2048`, `nim_container_image="nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0"`, `preferred_host_port=8001`, `gpu_memory_minimum_gb=24`, `provider="none"`. Existing deployment rows carrying the shipped 1.x default or former 10 GB floor are upgraded idempotently to the 2.0.0 image and supported-SKU floor. The Phase 3 environment-assessment endpoint (`GET /v1/environment`) returns embedding-NIM metadata from this record without requiring prior user configuration.

**Configuration and bootstrap (§1.9):**
- Verify: if `~/.vlm_feedback_loop/config.yaml` does not exist, the backend fails fast with a message directing the user to run the bootstrap command (e.g., `vlm-feedback-loop init`).
- Verify: the bootstrap command prompts for `WORKSPACE_ROOT`, writes a commented `config.yaml` documenting the Appendix A.4 defaults with `WORKSPACE_ROOT` as the only active key (a snapshotted default would pin its install-date value and silently shadow later default changes), generates a commented skeleton `.env` at `~/.vlm_feedback_loop/.env` documenting required variables without values, and exits cleanly.
- Verify: bootstrap creates `~/.vlm_feedback_loop/` with user-only permissions (`700`) and `.env` with permissions `600` where supported by the host OS.
- Verify: configuration precedence (highest wins): explicit process environment variable > explicit env file from `--env-file` or `VLM_FEEDBACK_LOOP_ENV_FILE` > default `.env` at `~/.vlm_feedback_loop/.env` > config file value > built-in default. Set the same key at all five levels; confirm the explicit process environment variable value is used. Remove levels progressively and confirm the next-highest source takes effect.
- Verify: `.env` file values at the canonical location `~/.vlm_feedback_loop/.env` are loaded at startup without requiring `--env-file` or `VLM_FEEDBACK_LOOP_ENV_FILE`.
- Verify: `--env-file /path/to/custom.env` overrides the canonical location. A key in the custom file is loaded; the same key in the canonical file is not.
- Verify: the backend does NOT search for `.env` in the current working directory, `WORKSPACE_ROOT`, or project directories. Place a `.env` with a test key in each of CWD, `WORKSPACE_ROOT`, and a project directory; confirm none are loaded (only the canonical or explicit-override file).
- Verify: if no `.env` file exists at the selected location, startup continues. Missing secrets surface only when a workflow requiring them is used (e.g., hosted NIM inference fails locally, before any outbound request, with a clear "NVIDIA_API_KEY not configured" error rather than a startup crash or an unauthenticated probe).
- Verify: all configurable defaults from Appendix A.4 are present and match their documented values when no overrides are set.
- Verify: Appendix A.4 defaults are type-validated at load time. Arrays (e.g., `STUDENT_LATENCY_TEST_CONCURRENCIES`) parse as arrays, nested structures (e.g., visual budget preset definitions) are preserved, integers are integers. Setting an array-typed default to a plain string → startup fails with a clear error naming the misconfigured key.

**Entity identifiers and timestamps (§2.2, §2.4):**
- Verify: all backend-generated entity identifiers (`project_id`, `guidance_id`, `model_config_id`, `run_id`, `tao_job_id`, `dataset_export_id`, `pool_id`, `audit_event_id`, `student_model_id`, `local_nim_deployment_id`, `inference_invocation_id`) are UUID4 in canonical lowercase hyphenated format. Validate with regex `^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`.
- Verify: all persisted timestamps use UTC ISO 8601 with `Z` suffix (e.g., `2026-03-30T14:22:07Z`). No timezone offsets, no naive timestamps.
- Verify: `created_at` and `updated_at` on Project are server-set; client-provided values are ignored.

**SSE infrastructure (§1.3):**
- Verify: `GET /v1/projects/{project_id}/events` returns a stream with `Content-Type: text/event-stream`.
- Verify: SSE events are project-scoped. Connecting to project A's stream and emitting a test event for project B: project A's stream does not receive it.
- Verify: each SSE event payload includes at minimum `run_id` (or equivalent task identifier) and `timestamp`.
- Verify: progress events include `processed: int` and `total: int`.
- Verify: completion events (`*_completed`) include final status and summary metrics.
- Verify: `run_failed` events include `run_id`, `run_type`, and `error_summary`.
- Verify: SSE does not support `Last-Event-ID` replay in v1. Reconnecting after disconnect does not replay missed events.
- Verify: the SSE event dispatcher accepts arbitrary event type strings without validation. Emitting a custom event type (e.g., `test_custom_event`) succeeds and is received by connected clients. The §1.3 event types are the required minimum, not a closed enum.
- Verify: the SSE endpoint accepts connections for valid project IDs and returns 404 for non-existent projects.

**Background task framework (§1.3, §1.8):**
- Verify: background tasks (evaluation, embedding, batch labeling) run in-process via asyncio. No external task queue (Celery, Redis, etc.) is required or used.
- Verify: a run record is persisted in the project database before background work begins. Killing the process immediately after task creation and restarting: the run record exists and is detectable.
- Verify: on backend restart, incomplete background tasks in non-terminal states are detected. Evaluation runs transition to `failed` with `status_reason="backend_restart_interrupted"`. Batch Labeling runs in `queued` or `running` transition to `queued` with `recovered_from_restart=true`.
- Verify: the restart recovery framework supports resumable background tasks without durable run records (e.g., CLIP embedding computation resumes from the first example without an embedding). The framework provides a hook for continuous-process recovery, not only bounded-run recovery.
- Verify: on graceful shutdown (SIGTERM), in-flight background tasks are canceled after a short grace period. Run records are updated to reflect interruption before exit.
- Verify: foreground priority dispatch: while a foreground request (`interactive_proposal`, `retry`, `rationale_regeneration`) is in flight, new background HTTP requests are held. Once the foreground request completes, background dispatch resumes. Already-in-flight background requests complete normally (not canceled).
- Verify: CPU-bound work (pHash computation, file I/O) runs in a thread pool and does not block the asyncio event loop. Confirm the event loop remains responsive during a CPU-intensive operation.

**Single-process concurrency (§1.3):**
- Verify: when a second backend process attempts to open the same project, it receives a hard error: "This project is already open in another process."
- Verify: no override or force-open path exists for the project lock in v1.

**Error handling and retry framework (§11):**
- Verify: every outbound HTTP invocation enforces a finite deadline. A mocked endpoint that never responds triggers a timeout at the configured deadline.
- Verify: default deadlines: `HTTP_DEADLINE_INTERACTIVE_S` = 180s, `HTTP_DEADLINE_BACKGROUND_S` = 300s. Both are configurable.
- Verify: automatic retries are bounded at `HTTP_MAX_RETRIES` (default: 3) attempts total per invocation.
- Verify: automatic retries do not create additional Operation Records; all attempts are within a single `inference_invocation_id`.
- Verify: retryable HTTP status codes (429, 502, 503, 504) and transient connection failures trigger retry with exponential backoff. Confirm backoff intervals increase (base: 1s, 2s, 4s) with jitter applied.
- Verify: non-retryable HTTP status codes (400, 401, 403, 404, 422) fail immediately without retry. Confirm only 1 attempt is made.
- Verify: after exhausting retries, the system persists the failure with the best-available error information rather than crashing or retrying indefinitely.
- Verify: failure semantics distinguish three classes on Operation Records (§3.4): `schema_invalid` (model responded but output does not match Core schema), `timeout` (model did not respond within deadline), `endpoint_error` (transport or HTTP error prevented a response). Each class is persisted distinctly and surfaced differently in the UI.

**Logging infrastructure (§11):**
- Verify: every log entry is structured JSON with required base fields: `timestamp` (ISO 8601), `level` ∈ {`debug`, `info`, `warn`, `error`}, `component`, `project_id`, `correlation_id`, `message`, `details`.
- Verify: `LOG_LEVEL` defaults to `info`. Setting `LOG_LEVEL=debug` produces additional detail from debug-default log points.
- Verify: logs write to both stdout and `{project_dir}/logs/*.jsonl`.
- Verify: a regex redaction filter catches API key patterns (e.g., `nvapi-...`, `Bearer ...`) and replaces them with `[REDACTED]`. Inject a mock API key into a log `details` field and confirm it is redacted in the output.
- Verify: log entries do not contain raw image bytes, full prompt text, or user-identifiable information beyond `project_id` and `example_key`.

**Action Request generic mechanism (§10.3):**
- Verify: `POST /v1/projects/{project_id}/action_requests:generate` with a valid `request_type` returns `rendered_text` (non-empty string), `generated_at` (timestamp), and `project_name`.
- Verify: `rendered_text` does not contain secrets (scan for API key patterns).
- Verify: `POST /v1/projects/{project_id}/action_requests:log_copy` creates an AuditEvent with `event_type="action_request_copied"` and the rendered text in `event_data`.
- Verify: unknown `request_type` values return an appropriate error.
- Verify: the generate endpoint accepts an optional `context` object with nullable type-specific fields (`student_model_id`, `example_keys`, `error_ref`, `tao_job_id`). When `context` is omitted or null, the endpoint still returns a valid response. The `context` fields are consumed by type-specific Action Request implementations in later phases.

**Frontend application shell (§1.8):**
- Verify: React + TypeScript + Vite app builds without errors and serves static assets.
- Verify: KUI Foundations component primitives (buttons, inputs, dialogs, cards, badges, progress indicators) render with NVIDIA design language. Tailwind utility classes are available for layout.
- Verify: a single command starts both frontend dev server and backend for local development.
- Verify: Vite dev server proxies `/v1/` API routes and the SSE events path to the FastAPI backend origin. API calls from the browser reach the backend without CORS errors.
- Verify: the dev proxy supports SSE passthrough — an `EventSource` connection to the proxied SSE endpoint stays open and receives events without premature closure or buffering.
- Verify: in production (same-origin or reverse proxy), no dev proxy or CORS configuration is required; the app functions without either.

**Frontend SSE recovery contract (§1.3):**
- Verify: on page load, the frontend fetches state from REST endpoints before establishing an SSE connection. The UI renders correctly from the REST snapshot alone (SSE not yet connected).
- Verify: while SSE is healthy, incoming SSE events update the UI without redundant REST calls for the same data.
- Verify: on SSE disconnect or error with active background work, the frontend begins short-interval REST polling (~5s) until SSE reconnects.
- Verify: on SSE reconnect/open, the frontend immediately refreshes relevant REST state (not waits for the next SSE event).
- Verify: on terminal SSE events (`*_completed`, `run_failed`), the frontend immediately refreshes the corresponding REST state.
- Verify: browser tab closure does not affect backend background tasks; they continue running.

**Project List screen (`ProjectListPage.tsx`):**
- Verify: when no projects exist, the empty state (Screen 1.1) renders: centered message "No projects yet." with subtitle and a single [+ Create Project] button. No project cards, no header Create button.
- Verify: when projects exist (Screen 1.2), [+ Create Project] appears in the header. Each project renders as a card showing: name, description (when set; omitted when null), summary counts (Verified, Unlabeled, Auto-Labeled, Omitted), created date, and last updated date. The card displays 4 counts; `pending_relabel` is returned in the API response (§10.2.13) for programmatic consumers but is not shown in the card UI.
- Verify: a project with no activity (just created) shows all counts at zero (Screen 1.2, "Weld Quality" example).
- Verify: projects without a description show no description line (not an empty line or placeholder).
- Verify: clicking a project card resolves a state-aware destination. A new project with no activity navigates to the NIM setup choice (Screen 2A). Non-terminal Training Suites resume in Training Jobs; an active Student serving validation resumes Models & Results; any terminal Training Suite history or registered Student opens Project Overview; a labeling-ready project without Student-training history opens Labeling.
- Verify: [+ Create Project] button (in empty state or header) opens the Create Project dialog (Screen 1.3) as a modal overlay.
- Verify: Create Project dialog requires Name (marked with *); submitting with an empty name shows a validation error and does not call the backend.
- Verify: Description field in the dialog is optional; submitting with name only succeeds.
- Verify: clicking [Create Project] in the dialog calls `POST /v1/projects` with the entered name and description, and on success navigates to Screen 2A (NIM setup choice), unless an already-complete setup is auto-skipped.
- Verify: clicking [Cancel] in the dialog closes it and returns to the project list without creating a project. No backend call is made.
- Verify: when the backend returns a project lock error (§1.3), the Project Lock Error dialog (Screen 1.4) renders with title "Project In Use", message "This project is already open in another process. Close the other process and try again.", and a single [OK] button. No override, force-open, or retry action exists.
- Verify: timestamps are displayed in the user's local time (backend stores UTC; frontend converts per §2.4).

**Project Overview (`ProjectOverviewPage.tsx`):**
- Verify: a mature project (any Training Suite history or registered Student) renders three distinct destinations: Interactive Loop, Models & Results, and Training Runs. Interactive Loop offers [Continue labeling], [Add images], and [Scale-Up]. Models & Results reports Student quality/serving counts and is disabled when no Student exists. Training Runs remains available whenever suite history exists, including failed and canceled runs.
- Verify: the project header keeps [Overview], [Models & Results], and [NIM Configuration] reachable from every post-setup project screen. The overview is a re-entry choice surface, not a blocking modal.

**Training Runs (`TrainingRunsPage.tsx`):**
- Verify: Training Runs lists suites newest-first with model identity, preset, completed-job count, local-time start, and user-facing status. Active runs offer [Resume]; terminal runs offer [View details].
- Verify: Training Runs and Training Job details expose status, metrics, lineage, and logs but no standalone checkpoint or TAO-output download. The portable NIM deployment bundle on Models & Results is the only trained-model delivery download.

**Environment assessment (§1.5, §10.2.25):**
- Verify: `GET /v1/environment` returns all required fields: `hosted_nim_available`, `local_deploy_available`, `docker_available`, `nvidia_toolkit_available`, `nvidia_api_key_configured`, `ngc_api_key_configured`, `gpus[]`, `local_deployable_models[]`, `embedding_deployment`, `missing_prerequisites[]`, `recommended_teacher_mode`, `recommended_embedding_mode`, and non-secret `active_local_nim_residents[]`.
- Verify: `hosted_nim_available` = `true` when `NVIDIA_API_KEY` present; `false` otherwise.
- Verify: `local_deploy_available` = `true` only when Docker + NVIDIA toolkit + at least one GPU all detected; `false` if any is missing.
- Verify: `gpus[]` populated from `nvidia-smi` output, including compute capability; empty array when `nvidia-smi` unavailable or fails.
- Verify: `local_deployable_models[]` includes only seeded catalog entries with `local_deploy_metadata` (Teacher/Student VLM models). Each entry is annotated with `fits: boolean` comparing one physical GPU against both `nim_gpu_memory_minimum_gb` and optional `nim_compute_capability_minimum`. The embedding NIM is NOT in this array (reported separately via `embedding_deployment`).
- Verify: `embedding_deployment` is sourced from `EmbeddingDeploymentConfig` (§13.17), not from the model catalog. Reports the embedding NIM model name, container image, GPU minimum, `fits` status, and provider. `fits=true` only when a placement-available GPU both meets the configured floor and has an exact detected-name match in the pinned NIM support matrix. This is the single source for embedding-NIM deployment metadata used by the environment assessment and NIM setup/configuration surfaces.
- Verify: `missing_prerequisites[]` lists each missing component with an `install_hint`.
- Verify: quality-first recommendation logic: Omni wins on a GPU meeting ≥80 GB and compute capability ≥9.0; CR3-Nano wins when ≥56 GB but Omni is ineligible; Cosmos Reason2 2B wins on 36–55 GB. Super and Cosmos Reason2 8B remain selectable but do not outrank these defaults by memory size. An exact compatible `running` Blueprint Teacher yields `recommended_teacher_mode="local"` only when it matches that recommendation; a different resident is reported for explicit keep/replace. Without a preferred resident, the API-key + GPU-fit case recommends the fitting local Teacher for latency while using the hosted default as an immediate bridge during download; the local Teacher activates only after verification, and hosted-only remains available. No key + fitting local Teacher recommends `"local"`. A supported 24–35 GB GPU with an exact pinned-matrix name recommends hosted Teacher plus the local embedding NIM; a GPU below 24 GB or with an unrecognized name does not recommend local embeddings.
- Verify: the endpoint does not return secret values (API keys). Only reports whether each key is configured (`nvidia_api_key_configured`, `ngc_api_key_configured`).
- Verify: ordinary endpoint calls reuse one process-local Docker/toolkit/GPU snapshot while recomposing credentials, embedding configuration, and active NIM residents from current state.
- Verify: `refresh_hardware=true` forces a new machine probe and replaces the cached snapshot; local NIM preflight remains a fresh live check regardless of the cached recommendation.
- Verify: the endpoint is deployment-scoped (no `project_id` in path).

**Model catalog and seeded defaults (§4.8, §10.2.19):**
- Verify: on project creation, 8 seeded entries are present with correct values per §4.8: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (128000, teacher, vision: true, thinking: qwen, media: none, default ICL cap 4, local image `:1.7.0-variant`, 80 GB / cc9.0 floors), `nvidia/cosmos3-nano-reasoner` (131072, roles: teacher/student_base, vision: true, thinking: qwen, media: mm_processor_size), `nvidia/cosmos3-super-reasoner` (131072, same roles, same capabilities), `nvidia/cosmos-reason2-8b` (256000 tokens, roles: teacher/student_base, vision: true, thinking: qwen, media: mm_processor_size), `nvidia/cosmos-reason2-2b` (256000, same roles, same capabilities), `nvidia/nemotron-nano-12b-v2-vl` (128000, teacher, vision: true, thinking: none, media: mm_processor_tiles), `stepfun-ai/step-3.7-flash` (262144, teacher, vision: true, thinking: always_on_reasoning, media: none), and `mistralai/mistral-medium-3.5-128b` (262144, teacher, vision: true, thinking: none, media: none). No fresh seed has non-commercial-only or unknown published model terms; MiniMax M3 is absent.
- Verify: `teacher_model_config_id` defaults to the effective `DEFAULT_TEACHER_MODEL` — `stepfun-ai/step-3.7-flash` by default — when no preferred healthy Teacher resident exists. With a compatible resident matching the current quality recommendation, project creation attaches its endpoint and returns the matching local model config as the selected Teacher. An override outside the commercial seed fails with an actionable error.
- Verify: all 8 seeded entries reference a default hosted NIM endpoint. The endpoint is a concrete, usable configuration from project creation — not a placeholder that requires Phase 3 to become functional. Capability probes (`unknown` at creation) are run later to validate what the endpoint supports.
- Verify: `GET .../model_configs?eligible_role=teacher` returns only entries with `teacher ∈ eligible_roles` AND `supports_image_input=true`.
- Verify: `GET .../model_configs?eligible_role=student_base` returns only Cosmos Reason2 8B/2B and Cosmos 3 Nano/Super reasoner.
- Verify: `POST .../model_configs` with `student_base` role on a non-seeded Cosmos Student base is rejected.
- Verify: `model_config_id` is backend-generated UUID4; `eligible_roles[]` is required and non-empty.
- Verify: `PATCH .../model_configs/{id}` allows updating endpoint, context window, roles, capabilities. `model_config_id` and `model_name` are immutable (reject attempts to change them).
- Verify: model configs are project-scoped; seeded entries from project A are not visible in project B.
- Verify: `local_deploy_metadata` is present on Cosmos Reason2 8B (image: `nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0`, GPU min: ≥56 GB), 2B (image: `nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0`, GPU min: ≥36 GB), and absent on other seeded entries.
- Verify: `supports_image_input` is seeded per catalog entry and is distinct from `visual_budget_mode` (§4.8).
- Verify: all three capability fields (`structured_generation_support`, `thinking_toggle_support`, `visual_budget_support`) initialize to `unknown` on creation.

**NIM endpoint modes and connection testing (§1.5):**
- Verify: Mode A (hosted NIM): base URL defaults to `https://integrate.api.nvidia.com/v1` with `Authorization: Bearer $NVIDIA_API_KEY`. Connection test via `GET /v1/models` returns the model list on valid key.
- Verify: Mode B (self-hosted NIM): base URL is user-configured (including `/v1`); no credential is sent (self-hosted NIMs run on a trusted network). Connection test via `GET /v1/models` confirms endpoint reachable.
- Verify: a self-hosted embedding selection is not accepted on model-list reachability alone. Test and Save each obtain one finite 2,048-dimensional vector through the configured NeMo Retriever `/embeddings` operation; Save persists the normalized deployment-scoped URL and re-resolves every non-archived project. A different URL is rejected while a Blueprint-managed embedding resident is active.
- Verify: connection test failure returns a user-facing error message, not an unhandled exception or stack trace.
- Verify: Mode C (local deploy): the system constructs `docker run` commands from `local_deploy_metadata` with: pinned container image, name-only `-e NGC_API_KEY`, NIM cache mount (`~/.cache/nim:/opt/nim/.cache`), loopback-only host-to-container port mapping (`127.0.0.1:{host_port}:8000`), `--runtime=nvidia`, GPU assignment, `--shm-size=32GB`, named container (`--name`). The Docker client receives the value only through a private child environment; neither a literal value nor a `KEY=value` assignment appears in argv, generated handoffs, or persisted validation evidence.
- Verify: the NIM client abstraction supports both `GET /v1/models` (connection test) and `POST /v1/chat/completions` (inference dispatch). Inference dispatch enforces deadline (`HTTP_DEADLINE_INTERACTIVE_S` or `HTTP_DEADLINE_BACKGROUND_S`), applies bounded retries (§11), and parses the OpenAI-compatible response. This abstraction is exercised by capability probes (§6.2, §6.7.4, §6.9.2) and reused for all Teacher/Student invocations.

**Capability probes — initial behavior (§6.2, §6.7.4, §6.9.2):**
- Verify: structured generation probe sends the fixed minimal request from §6.2 (`{"ok": boolean}` schema, `max_tokens: 16`), bounded by `HTTP_DEADLINE_INTERACTIVE_S`. Successful parse → `supported`; 4xx from `response_format` → `unsupported`; timeout/5xx → remains `unknown`.
- Verify: thinking toggle check sends mode-specific `chat_template_kwargs` (Qwen: `{"enable_thinking": false}`, Kimi: `{"thinking": false}`), bounded by `HTTP_DEADLINE_INTERACTIVE_S`. Non-error response → `supported`; 4xx from `chat_template_kwargs` → `unsupported`; timeout/5xx → remains `unknown`. When `thinking_toggle.mode="none"`, auto-set `unsupported` without sending any request.
- Verify: visual budget probe runs two stages with a deterministic 512×512 RGB PNG probe image (§6.9.2). Stage 1 (baseline): request without `mm_processor_kwargs`. Stage 2 (capability): same request with mode-specific `mm_processor_kwargs` values from §6.9.2. Both stages bounded by `HTTP_DEADLINE_INTERACTIVE_S`. Baseline success + capability success → `supported`; baseline success + capability fail → `unsupported`; baseline fail → remains `unknown`. When `visual_budget_mode="none"`, auto-set `unsupported` without probing.
- Verify: all three probe results persisted on ModelConfig; reused for subsequent invocations without re-probing.
- Verify: a failure in one probe does not prevent the other probes from executing.

**Image transport rules (§6.3):**
- Verify: images from `storage_ref` are read, format-normalized if needed (default transcode: PNG), and encoded as base64 data URL (`data:image/<format>;base64,<data>`).
- Verify: images are sent inline as base64 for all endpoint modes (hosted, self-hosted, local) regardless of image size. The NVCF Asset API is never invoked (the path was removed 2026-06-02; see §6.3).
- Verify: filesystem paths and `storage_ref` values are never passed as URL references to NIM endpoints.
- Verify: Operation Record persists `image_transport_mode` and `image_format_transmitted` when the invocation includes images.
- Verify: the `image_transport_mode` enum is {`base64_inline`, `direct_url`}. `direct_url` is used only when the image already exists at a fully qualified, runtime-reachable HTTP(S) URL that the target NIM endpoint can download.
- Verify: the image transport service exposes a batched preparation API that accepts all images for one model invocation in one call (ICL examples + query image), preserves input order, and returns prepared content parts plus per-image status/metadata. Callers do not loop one image at a time. Partial preparation failure (read/normalise error on any image) prevents the model request from being sent.

**Local NIM deployment lifecycle (§1.5 Mode C):**
- Verify: preflight checks run in order: (1) Docker available, (2) NVIDIA Container Toolkit, (3) GPU memory vs `nim_gpu_memory_minimum_gb`, (4) NGC_API_KEY configured, (5) model profile compatible — MUST attempt `list-model-profiles` against the target container image; if no runnable profile exists on the current hardware, deployment fails with a clear error. A single-model image's exact missing-utility result is an explicit inconclusive pass to bounded serve health/served-model verification; shared-image selection remains blocked without a successful size-aware probe. (6) container image pullable. Each check reports pass/fail with diagnostic message.
- Verify: `docker run` command includes `--gpus` with device specification (e.g., `device=0`). The resolved GPU assignment is persisted as `gpu_assignment` on the `LocalNimDeployment` record. Without explicit `--gpus`, the container would claim all GPUs, breaking concurrent local services.
- Verify: GPU placement policy (F49 amendment, 2026-05-19, §1.5 one-NIM-per-GPU invariant) — at most one NIM container is `starting` or `running` on any GPU at any time. On a multi-GPU host, every role takes the lowest compatible free index at dispatch time; embedding placement additionally skips devices below its configured eligibility floor. Roles do not own fixed device numbers. Same-GPU co-location is not supported in v1. When every GPU is occupied, the auto-placer raises `GpuExhaustedError` and the deploy router returns `409 gpu_exhausted`. An exact compatible `running` Teacher is reused across projects without another reservation or container; the response is `disposition="reused"`, `deployment=null`, and names the owner resident. A different resident produces a structured 409 naming its model/project and is stopped only after the caller explicitly retries with `replace_resident=true`; the FTUE presents Keep vs Stop-and-start actions. An exact compatible `starting` Teacher produces `409 resident_starting` and cannot be replaced from that prompt. Each displaced deployment's `displaced_by_deployment_id` and `displaced_at` are persisted on its `LocalNimDeployment` row (§13.15).
- Verify: single-GPU FTUE (Run-locally) skips the embedding deploy when a Teacher is queued (frontend defense-in-depth) and falls back to pHash diversity (§5.6); backend rejects the embedding deploy with `409 gpu_occupied` if the frontend somehow sends it without `replace_resident=true`.
- Verify: Student NIM lifecycle (§9.5.2 step 0) stops every resident deployment on its target GPU before constructing the Student `docker run` and (§9.5.2 step 9) best-effort auto-restores those displaced deployments after the Student stops. Auto-restore failure does NOT fail `serving_status` — surfaces as a warning only.
- Verify: GPU memory fast-fail: if detected GPU memory is below the model's minimum, deployment fails immediately naming the GPU and the requirement (e.g., "24 GB available, need ≥56 GB").
- Verify: on health: container polls `/v1/health/ready` up to `NIM_STARTUP_TIMEOUT_S` (default: 1200s). On healthy, the system auto-registers the endpoint (Teacher: model catalog; embedding NIM: `embedding_provider=self_hosted_nvclip`). The SME does not configure URLs manually.
- Verify: served-model verification (Teacher deployments, §1.5 Mode C step 3) runs before healthy: the system queries `/v1/metadata` for the loaded model slug and confirms the requested model has real weight files in the NIM cache (not a config-only stub). A loaded-model mismatch or zero non-trivial weight files marks the deployment `failed` with an actionable `status_reason` (guards the silent `cosmos3-reasoner` nano→super fallback) rather than registering the wrong model. On the same healthy transition the endpoint's per-prompt image cap is auto-set from the served NIM so ICL is not silently truncated.
- Verify: port allocation uses preferred host port (Teacher: 8000, embedding NIM: 8001). When occupied, next available port is used. Resolved URL is persisted.
- Verify: container named via `--name`. On restart recovery, the system inspects only containers matching persisted names — no generic Docker orphan discovery.
- Verify: restart recovery: a persisted `starting` Teacher/embedding whose named container is still running but not yet ready remains `starting` and resumes the ordinary background health poll for a fresh `NIM_STARTUP_TIMEOUT_S` window; container death and timeout retain the ordinary poller's fail-and-teardown behavior. A `running` + healthy container rebinds and restores registration; a previously `running` but unhealthy container is stopped rather than misclassified as startup. Not running → mark `stopped` and surface "Redeploy locally" action.
- Verify: stopping, displacing, or failing a shared Teacher resident disables every project-local endpoint attached to that deployment, while a backend restart preserves and re-adopts the owner record and healthy Docker container.
- Verify: `LocalNimDeployment` record created with: `container_name`, `nim_container_image`, `role`, `host_port`, `endpoint_url`, `gpu_assignment`, `status`.
- Verify: if preflight fails, the system generates an Action Request with the exact `docker run` command and all prerequisites pre-filled (§10.3 pattern).

**TAO endpoint configuration (§1.6):**
- Verify: TAO configuration (`TAO_API_BASE_URL`, `TAO_API_KEY`, `TAO_ORG_NAME`) is deployment-level (shared across all projects), not per-project.
- Verify: connection probe (`GET /api/v2/orgs/{org_name}/jobs?limit=1`) runs on configuration. Auth failure surfaces: "Could not connect to TAO. Verify the API URL, key, and organization name."
- Verify: `TAO_API_KEY` never appears in logs, API responses, or UI payloads.

**NIM setup chain (`NIMNvidiaKeyPage.tsx`, `NIMNgcKeyPage.tsx`, `NIMSetupGatePage.tsx`) and post-onboarding NIM Configuration (`NIMConnectionPage.tsx`):**
- Verify: when project creation reused an exact running Teacher, onboarding confirms that the project-selected model matches the recommendation, uses `activePath="local"`, and queues no Teacher deployment. It fully auto-skips when embeddings need no action; a separately recommended local embedding NIM still follows the NGC/setup gate.
- Verify: auto-configured skip (2.1): when both Teacher and embeddings can be configured without user input, this screen is skipped entirely, except that a third-party hosted default with point-of-selection usage terms remains visible until the SME continues. A non-blocking dismissable banner appears on the next screen showing the auto-selected configuration. Three banner variants tested: hosted Teacher + local embedding NIM, hosted-only, all-local.
- Verify: recommendation screen (2.2): two independent service rows (Teacher, Embeddings) each showing recommended mode and a [Configure] override. Only missing credentials prompted — keys already configured show "✓ configured." Detected hardware shown at top. [Continue] navigates forward.
- Verify: the post-onboarding configuration screen proactively validates each credential required by its currently effective service modes exactly once, including under React Strict Mode. An all-local selection may validate the NGC key but MUST NOT contact a hosted NVIDIA endpoint merely because an unused `NVIDIA_API_KEY` remains configured; an all-hosted selection has the inverse behavior. `EMBEDDING_PROVIDER=none` produces `recommended_embedding_mode="none"`, visibly names pHash diversity, and does not prompt or probe for a hosted embedding key. Changing an override may validate the newly relevant credential.
- Verify: [Configure] → hosted (2.3): shows API key field when not configured, [Get NVIDIA API Key ->] link, [Test Connection] action. Success shows "✓ Connected to NVIDIA hosted NIM." Failure shows "✗ This key was rejected..." with recovery guidance.
- Verify: post-onboarding [Configure] → self-hosted (2.4): base URL required (with a helper noting to include `/v1` and that the endpoint must be reachable from the backend) and a trusted-network disclaimer in place of any credential field, [Test Connection]. A green test lists only exact vision-Teacher models reported by the endpoint and present in the project catalog. [Save] re-verifies the selection, durably binds/selects it, re-probes capabilities, and returns to the project; a failed save remains on screen with an actionable error. Repeating the same save reuses one endpoint. When no endpoint is available, [Request NIM Setup] generates an Action Request (2.7) inline.
- Verify: the Embeddings self-hosted override stages no URL until a real NeMo Retriever embedding request returns one finite 2,048-dimensional vector. Editing the URL clears that stage. [Save] repeats the proof, durably updates the deployment-scoped provider, re-resolves projects, and leaves an actionable error on screen without mutation when the endpoint fails or conflicts with an active managed embedding resident.
- Verify: [Configure] → deploy locally (2.5): shows preflight results. Passed: Docker ✓, GPU ✓ (with model), NGC key ✓, image ref, profile. Failed: identifies which check failed, suggests alternatives ("Use hosted NIM or a self-hosted endpoint").
- Verify: missing prerequisites (2.6): shown when local deployment recommended but Docker or NVIDIA Container Toolkit is missing. Copy-ready install commands, reference to `setup-local.sh`, [Switch to Hosted] fallback offered.
- Verify: Action Request (2.7): read-only pre-filled block with project name, current catalog model options (explicitly stating only one is needed), hosting-team NGC key guidance, known GPU minimums, verification steps, and the endpoint/network/contact details to return. [Copy to Clipboard] action. Content does not contain secrets.
- Verify: in the split setup chain, NGC [Back] returns to the setup choice and successful choices proceed through the setup gate to Confirm Defaults (or skip Confirm Defaults when valid). Post-onboarding NIM Configuration uses [Cancel] to return without applying staged changes and [Save] to apply verified changes before returning to the project.

**Confirm Model Defaults screen (`ConfirmDefaultsPage.tsx`):**
- Verify: skipped (3.1) when the selected Teacher is valid and its endpoint is connected. The one-shot banner names the actual selected Teacher, including a local resident adopted during project creation.
- Verify: confirm required (3.2): shown when defaults missing, invalid, or custom endpoint where seeded models may not exist. One dropdown: Teacher (filtered to `teacher` role + `supports_image_input=true`). Seeded default preselected. The dropdown shows context window and capability summary.
- Verify: [Continue ->] proceeds to Screen 4 (Image Ingestion). [Back] returns to Screen 2.

**Guidance CRUD and versioning (§4.4, §10.2.2, §13.9):**
- Verify: `POST /v1/projects/{project_id}/guidance` creates a new immutable Guidance version. Subsequent calls create new versions, not updates to existing records.
- Verify: `version_number` is 1-based, monotonically increasing within the project, backend-assigned, and immutable. First version = `v1`, second = `v2`, etc. Client-provided `version_number` is ignored.
- Verify: `GET /v1/projects/{project_id}/guidance/{guidance_id}` returns the version with all required fields: `guidance_id`, `version_number`, `description`, `schema` (with `field_id` per field), `rules`, `created_at`.
- Verify: `GET /v1/projects/{project_id}/guidance` lists all versions for the project, newest-first. Supports cursor pagination.
- Verify: setting `project.active_guidance_id` via `PATCH /v1/projects/{project_id}` activates the first Guidance and does not modify any Guidance record; once a Guidance is active, PATCHing a different id (or null) returns 400 — the active version moves only through the guidance edit endpoint.
- Verify: Guidance records are project-scoped; versions from project A are not visible in project B.
- Verify: creating a Guidance with empty `description` is accepted (stored as an empty string). Empty `rules` is allowed. A Guidance with no Core fields is rejected (`NO_CORE_FIELDS`); an invalid Core field is rejected by its own field-level error.

**SchemaCore type system and field validation (§4.4):**
- Verify: exactly five field types supported: `enum`, `enum_set`, `boolean`, `integer`, `string`. A field with any other type is rejected.
- Verify: Enum requires `allowed_values[]` with ≥2 values, no empty strings, unique after trim. One value → `ENUM_TOO_FEW_VALUES`. Empty string → `ENUM_EMPTY_VALUE`. Duplicates after trim → `ENUM_DUPLICATE_VALUE`.
- Verify: Enum Set has the same `allowed_values[]` constraints as Enum.
- Verify: Integer `minimum` and `maximum` are optional. When both set, `minimum` ≤ `maximum` enforced; violation → `MIN_EXCEEDS_MAX`.
- Verify: String `minLength` and `maxLength` are optional. When both set, `minLength` ≤ `maxLength` enforced; violation → `MINLENGTH_EXCEEDS_MAXLENGTH`.
- Verify: Boolean has no additional constraints beyond the type.
- Verify: `field_name` must match `^[a-zA-Z_][a-zA-Z0-9_]*$`, max 64 characters. Names like `type`, `id`, `class` are valid. Names starting with a digit → `INVALID_FIELD_NAME`. Names >64 chars → `FIELD_NAME_TOO_LONG`. Empty name → `MISSING_FIELD_NAME`.
- Verify: `field_name` unique across Core + Aux within the schema. Duplicate → `DUPLICATE_FIELD_NAME`.
- Verify: omission of `rationale_note` is valid and is the default disabled state. If present, it is accepted only as the reserved `role="aux"`, `type="string"` field; invalid role/type is rejected.
- Verify: `field_id` is system-generated at field creation, immutable, unique across Core + Aux. Client-provided values are ignored.
- Verify: at least one Core field is required. Zero Core fields → `NO_CORE_FIELDS`.
- Verify: `display_order` controls within-group ordering. Aux-before-Core group ordering is system-enforced.
- Verify: when enabled, `rationale_note` has the lowest `display_order` (system-enforced).

**Draft validation and JSON Schema derivation (§10.2.2, §6.2, §6.6.6):**
- Verify: `POST /v1/projects/{project_id}/guidance:validate_draft` accepts `{description, schema, rules}` and returns `{issues[], derived_json_schema, schema_hash, save_allowed}`.
- Verify: `save_allowed` = `true` when zero issues with `severity="error"`; `false` when any error exists.
- Verify: `derived_json_schema` properties are ordered by `generation_order`: optional `rationale_note` first when enabled, then remaining Aux by `display_order`, then Core by `display_order`.
- Verify: `required` in derived schema includes only `role="core"` fields; Aux fields are not in `required`.
- Verify: `additionalProperties: false` set on the top-level derived schema.
- Verify: type mapping in derived schema: enum → `{"type": "string", "enum": [...]}`, enum_set → `{"type": "array", "items": {"type": "string", "enum": [...]}, "uniqueItems": true}`, boolean → `{"type": "boolean"}`, integer → `{"type": "integer"}` with optional `minimum`/`maximum`, string → `{"type": "string"}` with optional `minLength`/`maxLength`.
- Verify: `schema_hash` is deterministic — same draft produces same hash on repeated calls.
- Verify: the same `validate_and_derive` function is used by both draft validation and the Guidance save endpoint. A draft that passes validation → saved Guidance has the identical `derived_json_schema`.
- Verify: `SCHEMA_COMPILE_FAILURE` is detected when field constraints create an internally inconsistent schema.
- Verify: the derived schema includes `x-generation-order` as a top-level extension property containing the field names in `generation_order` (optional rationale first when enabled, then Aux, then Core). This array matches the `properties` key ordering.
- Verify: `derived_json_schema`, when persisted on a Guidance record and retrieved via `GET .../guidance/{guidance_id}`, produces byte-identical JSON output. Comparison MUST use raw string equality, not parsed-object equality. This catches property reordering introduced by ORM serialization, JSON Schema libraries, or database round-tripping (§6.2).

**Schema evolution resets (§4.4.1 — detail supplement to "SchemaCore and Core edit policy" tests):**
- Verify: on semantic Core change, `icl_recommendation_dismissed_at_count` resets to `0` on the Project record.
- Verify: on semantic Core change, any in-progress evaluation is canceled.
- Verify: on semantic Core change, Auto-Evaluate trigger counters (first pool threshold, config change, ICL growth) rebuild from zero under the new Guidance.
- Verify: on semantic Core change, review selector scheduler state is reinitialized: recent window cleared, selector-history state reset.
- Verify: on semantic Core change, `schema_refinement_reminders_dismissed` resets to `0`.
- Verify: on semantic Core change, if the SME was reviewing an example, `schema_change_context_example_key` is recorded on the Project record.
- Verify: prior evaluation Run Records remain in history for audit but are not used as the "previous" baseline for evaluations under the new Guidance.
- Verify: examples with `state="Omitted"` remain Omitted — unaffected by schema evolution.

**Create Guidance screen — detailed UI (`CreateGuidancePage.tsx`, §6.6):**
- Verify: sticky header shows "Create Guidance", project name, and live-updating status badge: "Valid" when 0 errors, "{N} errors" when errors exist. Badge updates as the SME edits.
- Verify: sticky footer shows [Cancel] (secondary) and [Save Guidance] (primary). [Save Guidance] disabled when errors exist; enabled when 0 errors.
- Verify: "Start from:" dropdown above the cards defaults to Blank. Options: Blank, Classification, Rock, paper, scissors, Multi-label classification, Presence and count, Packaging information audit, Industrial anomaly inspection. Selecting a template pre-fills Description, Schema, and Rules. After the SME edits any content, changing the selection asks before replacing all three sections; cancel preserves the draft and restores the applied selection.
- Verify: each template pre-fills the correct fields and Rules per §6.6.2: Classification (1 enum Core with conspicuous replacement values), Rock-paper-scissors (1 enum Core matching the bundled sample), Multi-label (1 enum_set Core with conspicuous replacement values), Presence and count (1 boolean Core + 1 non-negative integer Core), Packaging information audit (1 enum Core + 1 boolean Core), and Industrial anomaly inspection (2 enum Core matching the reduced, evidence-backed VisA contract). All templates omit `rationale_note` so the default is disabled.
- Verify: one concise explanation appears below the applied selection. Rock-paper-scissors identifies the bundled 15-image walkthrough and CC BY 2.0 license. Packaging information audit identifies Open Food Facts image provenance and CC BY-SA terms. Industrial anomaly inspection identifies VisA, its 10,821-image/12-category scope, CC BY 4.0 license, and official source link. Generic option labels have no Starter/Demo prefix. No Open Food Facts or VisA images are bundled.
- Verify: Description card shows helper text ("Describe the task..."), placeholder text, soft character counter (awareness only, no hard limit). An empty description does not block save and shows no inline error. No on-blur error.
- Verify: SchemaCore card shows the explainer strip: "Core fields = required + evaluated · Aux fields = optional + not evaluated" with info tooltip. On create flow, the Core edit policy info banner is visible.
- Verify: Core Fields section has primary styling; Aux Fields section has muted/secondary styling. The physical separation encodes the role at field creation time.
- Verify: each field row contains: drag handle (reorder within section), field name input, type dropdown (Enum / Enum Set / Boolean / Integer / String), contextual constraints, role badge, and row actions (Delete, Move to Aux / Move to Core).
- Verify: Enum/Enum Set constraint control is a tag/chip input. Clicking [+] opens a text input on its own line. Enter commits a chip; Escape cancels. Each chip shows [x] on hover to remove. At least 2 values required.
- Verify: Integer shows Min and Max inputs always visible. String shows `minLength`/`maxLength` behind a collapsed "Advanced constraints" expander. Boolean shows no constraints.
- Verify: Move to Aux / Move to Core moves the field row to the other section (pre-save only; post-save role change is a semantic Core edit).
- Verify: the Aux section has a dedicated Rationale notes toggle, off by default. Enabling adds the reserved field; disabling removes it; it never appears as a normal editable row.
- Verify: toggling rationale notes in either direction is an in-place Aux edit with no label invalidation, including after Verified labels exist.
- Verify: Rules card labeled "Rules & Edge Cases" with helper text and placeholder. Optional — empty rules accepted on save.
- Verify: Derived JSON Schema preview collapsed by default. When expanded, shows the canonical schema from the backend validation response (read-only). Updates on each backend validation cycle.
- Verify: Example label output preview collapsed by default. When expanded, shows a JSON object with Core fields populated by placeholder values (first enum value, mid-range integer) and Aux fields optional.
- Verify: on Save with errors, page scrolls to the first error and focuses the offending control. Errors appear inline below offending fields with `x` icon. Fix-it buttons appear where applicable: `ENUM_DUPLICATE_VALUE` → "Remove duplicates", `MIN_EXCEEDS_MAX` → "Swap min/max".
- Verify: successful save creates an immutable Guidance version, sets it as `project.active_guidance_id`, shows toast "Guidance v1 saved.", navigates to the labeling screen.
- Verify: [Cancel] returns to the previous onboarding step without saving. No backend call.

**Edit Guidance screen (`EditGuidancePage.tsx`, §6.6.8):**
- Verify: same layout as Create Guidance but without the template selector (create flow only). The create-flow info banner is replaced by the post-save banner: "Renames apply directly. Changing what a correct answer looks like invalidates labels and returns them to Unlabeled. The model re-proposes labels under the new schema and shows your prior labels as reference. Look for the label-invalidation icon on controls that trigger it."
- Verify: controls that trigger label invalidation carry a `~` marker (icon or visual indicator) with tooltip: "Changing this invalidates your {N} verified labels. They return to Unlabeled for re-labeling." `~`-marked controls: type dropdown, constraint inputs (min/max, minLength/maxLength), allowed value [+] add and [x] remove, [+ Add Core Field], [del] delete field, [>Aux] / [>Core] role change.
- Verify: controls without `~` marker (in-place edits, no invalidation): field name text input, existing value chip rename (exact 1:1), drag handle reorder.
- Verify: Description and Rules textareas are freely editable; saving creates a new Guidance version without label invalidation (Description/Rules are not Core).
- Verify: [+ Add Aux Field] is available and does not trigger label invalidation.
- Verify: modifying a `~`-marked control triggers the schema change confirmation dialog.
- Verify: confirmation dialog — Verified only (6.2): title "Update schema and re-label?", body references the specific change (e.g., "the allowed values for damage_type"), states the Verified count, lists what happens next (prior labels preserved, model re-proposes, prior edits first). Actions: [Cancel] and [Update and Re-label]. No "warning", "caution", or "are you sure" language.
- Verify: confirmation dialog — Verified + Auto-Labeled (6.3): same as 6.2 with an additional paragraph: "This will also revert {M} Auto-Labeled examples to Unlabeled. You can re-run Batch Labeling when ready. Your improved Guidance and ICL examples carry over to the new run."
- Verify: [Update and Re-label] triggers the schema evolution flow (Phase 4 backend); [Cancel] closes dialog without changes.

**Filesystem browse and scan endpoints (§10.2.10):**
- Verify: `GET /v1/filesystem/browse?path=/some/dir` returns `{path, parent, entries[]}` where each entry has `name`, `type` (`directory` or `file`), `path`, and `size_bytes` (files only).
- Verify: entries sorted: directories first (alphabetical), then files (alphabetical). Hidden files (names starting with `.`) excluded.
- Verify: `image_formats_only=true` (default) filters files to JPEG, PNG, WebP, BMP, single-page TIFF.
- Verify: path not found → 404 with "Directory not found: {path}". Path not readable → 403.
- Verify: omitting `path` opens `IMAGE_ROOT` and returns `parent: null`; on loopback with `IMAGE_ROOT` unset, it opens `/`.
- Verify: `IMAGE_ROOT` is enforced. Any path resolving outside it (after symlink resolution) → 403. Symlinks escaping it → rejected.
- Verify: when backend binds to a non-loopback address and `IMAGE_ROOT` is unset, browse, scan, serve, and remap return `403` with "Filesystem browsing is disabled. Configure IMAGE_ROOT..." Batch ingest returns `202` under its partial-success contract, with every disallowed row reported as `status="error"`, `error_code="path_not_allowed"`, and the same guidance; no Example is persisted.
- Verify: when backend binds to loopback (`127.0.0.1`, `::1`) and `IMAGE_ROOT` is unset, filesystem access defaults to unrestricted.
- Verify: the browse endpoint is deployment-scoped (no `project_id` in path).
- Verify: `POST /v1/filesystem/scan` with `path` and `recursive=true` returns `{images[], skipped[], total_images, total_skipped}`. Each image has `storage_ref`, `suggested_example_key`, `size_bytes`.
- Verify: `suggested_example_key` is deterministic: slug from the path relative to configured `IMAGE_ROOT` (or the normalized absolute path when the root is unset) + `--` + the first 12 hex chars of SHA-256 of that canonical path with extension. The same file produces the same key when scanned from the dataset root or any nested directory.
- Verify: when `project_id` is provided, scan reports per-image `key_status`: `available` (key and path unused), `already_exists_same_path` (source path already present, including under a legacy key), `collision_different_path` (key belongs to another path and would be rejected). `total_collisions` counts `collision_different_path` items.
- Verify: skipped files (unsupported formats, unreadable) appear in `skipped[]` with reason.

**Image ingestion — detailed behavior (§10.2.1):**
- Verify: idempotent re-ingest: `{project_id, example_key}` exists with same `storage_ref` → returns existing record with `status="exists"`. No duplicate created.
- Verify: key collision: `{project_id, example_key}` exists with different `storage_ref` → rejected with `status="error"`, `error_code="example_key_collision"`, message identifying both paths.
- Verify: source-path duplicate: `{project_id, storage_ref}` exists under a different `example_key` → rejected with `status="error"`, `error_code="storage_ref_already_ingested"`. No duplicate Example is created.
- Verify: `ingested_at` is server-set; client-provided value is ignored.
- Verify: ingest returns 202 with skeleton Example rows whose `phash` may be null; the restartable background sweep populates hashes in bounded batches, emits progress/completion SSE, and resumes pending rows after backend restart without duplicating Examples.
- Verify: after ingestion completes, CLIP embedding background computation is triggered for newly ingested examples.
- Verify: Example record has `state="Unlabeled"` after successful ingestion.

**CLIP embedding — storage, cache, and switchover (§5.5.2, §5.5.5):**
- Verify: embeddings stored in dedicated `ClipEmbedding` table (float32 binary vector), not on the Example record. Example record stores only `clip_embedding_present`, `clip_embedding_dim`, `clip_embedding_model_id`, `embedding_provider`.
- Verify: in-memory cache loaded with a single bulk `SELECT` on project open. New embeddings update the cache incrementally (not full reload).
- Verify: CLIP embeddings computed in pHash-diverse order (same order the review selector would present under pHash-diverse mode, §5.5.2).
- Verify: on restart, computation resumes from the first example without an embedding. Already-computed embeddings are not recomputed.
- Verify: incremental ingestion: new images appended to the end of the computation queue.
- Verify: CLIP switchover: the review selector upgrades from pHash-diverse to CLIP-diverse when the number of eligible examples with CLIP embeddings reaches `CLIP_SWITCHOVER_MIN_COUNT` (default: 50). Below this count, pHash-diverse mode is used.
- Verify: CLIP embeddings computed regardless of example state — an example that becomes Verified before background computation reaches it still gets its embedding computed (§5.5.2).
- Verify: on provider/model change (§5.5.3), `ClipEmbedding` rows are deleted or replaced and the in-memory cache invalidated and rebuilt as recomputation proceeds.

**Storage path remapping (§10.2.11):**
- Verify: `POST .../examples:remap_paths` with `dry_run=true` (default) returns `matched_count`, `sample_remappings[]` (up to 10), `unmatched_count`, and a `validation` summary (sample checked, resolved, missing).
- Verify: dry-run does not modify any records.
- Verify: commit (`dry_run=false`) replaces `old_prefix` at the start of each matching `storage_ref` in a single transaction. Non-matching records unaffected.
- Verify: commit rejected if zero sampled remapped paths resolve to existing files ("None of the sampled remapped paths resolve to existing files.").
- Verify: commit creates an AuditEvent with `event_type="storage_ref_remap"` containing `old_prefix`, `new_prefix`, `remapped_count`.
- Verify: `old_prefix` and `new_prefix` are required non-empty absolute paths.

**Interactive proposal endpoint (§10.2.3):**
- Verify: `POST .../proposals` with `example_key` returns `inference_invocation_id`, `proposal_json`, `schema_valid_core`, `validation_errors_core[]`, `validation_errors_aux[]`, `invocation_status`, `latency_ms_end_to_end`, `icl_images_attached_count`, `icl_example_keys_used[]`.
- Verify: override fields work independently: `teacher_model_config_id_override`, `guidance_id_override`, `generation_preset_key_override`, `thinking_mode_override`, `visual_budget_preset_key_override`. Each override replaces the project default for that invocation only; project defaults unchanged.
- Verify: `retry_of_inference_invocation_id` links the new Operation Record to the prior attempt. The prior attempt's record remains unchanged.
- Verify: `use_existing_label=true` with an Auto-Labeled example returns the stored label without a Teacher call. `used_existing_label=true` in response.
- Verify: Operation Record is persisted before the model call begins (§13.1). If the process crashes mid-call, the record exists with `invocation_status="pending"`.

**Label save, skip, and restore endpoints (§10.2.14):**
- Verify: `POST .../labels` with `example_key`, `inference_invocation_id`, and `label_json` creates or promotes a Label record with `label_status=verified`. `rationale_source` is additionally required only when the active Guidance enables rationale notes. Response includes `verified_outcome`, `edited_core_fields[]`, `edited_aux_fields[]`, `pool_assignment`.
- Verify: backend computes the deterministic diff between proposal `normalized_json` and submitted `label_json`. If no fields differ → `verified_outcome=Accept`; any field differs → `verified_outcome=Edit`.
- Verify: when rationale notes are enabled and `verified_outcome=Edit`, `rationale_source` MUST NOT be `teacher_proposal`. Backend rejects with 400 if this constraint is violated.
- Verify: when rationale notes are disabled, Save strips a stale client-supplied `rationale_note`, ignores rationale provenance, persists null rationale metadata, and does not require `rationale_source`.
- Verify: the pool routing call site is invoked on every Save. In isolation (before pool routing is implemented), `pool_assignment` returns null. Once pool routing is implemented (§4.3), the new Verified example is assigned to Test Pool or non-pool per §4.3.1.
- Verify: stale-proposal rejection: if the submitted `inference_invocation_id` has been superseded by a later Retry proposal for the same example, the backend returns `409 Conflict` with a stale-proposal error.
- Verify: save invocation validation: `inference_invocation_id` must exist, belong to the same project, pertain to the same `example_key`, and have an allowed purpose (`interactive_proposal` or original `batch_label` when saving the surfaced Auto-Labeled proposal).
- Verify: Auto-Labeled promotion: when saving an Auto-Labeled example, `label_status` transitions from `auto_labeled` to `verified`, verification fields populated, `batch_label_run_id` retained for lineage, `inference_invocation_id` updated to the reviewed proposal.
- Verify: `POST .../examples/{key}:skip` sets `state="Omitted"`, `omitted_source="sme_skip"`, `omitted_at` (server timestamp). No Label record is created; an existing `auto_labeled` Label for that Example is deleted atomically while its Operation Record is preserved.
- Verify: `POST .../examples:restore_omitted` transitions all Omitted examples to `state="Unlabeled"`, clearing `omitted_source` and `omitted_at`, without restoring a discarded Auto-Labeled proposal. Returns `restored_count`. Available when queue empty and Omitted examples exist.

**Rationale regeneration endpoint (§10.2.15):**
- Verify: when the active Guidance disables rationale notes, the endpoint returns 409 without image preparation, Teacher dispatch, or Operation Record creation.
- Verify: `POST .../examples/{key}:regenerate_rationale` calls the Teacher with the image and task context (Appendix D.3), accepts no proposed or corrected label values, and returns `inference_invocation_id`, `rationale_note`, `invocation_status`.
- Verify: creates an Operation Record with `purpose="rationale_regeneration"`.
- Verify: when `teacher_model_config_id` is null in request, uses `project.teacher_model_config_id`.
- Verify: on failure (timeout/endpoint error), returns the error; the SME writes the rationale directly.

**Token budget derivation (§6.2):**
- Verify: `max_output_tokens` derived per invocation from SchemaCore. `schema_output_estimate` = `JSON_STRUCTURAL_OVERHEAD_TOKENS` (48) + sum of per-field worst-case estimates. `base_output_tokens` = `max(BASE_OUTPUT_TOKENS_FLOOR (256), 2 × schema_output_estimate)`.
- Verify: per-field worst-case estimates: `rationale_note` = `RATIONALE_NOTE_ESTIMATE_TOKENS` (160); enum = key overhead + max tokenized value + 6; enum_set = key overhead + sum all tokenized values + array syntax; boolean = 6; integer = 8; string with maxLength = key + ceil(maxLength/4); string without maxLength = `DEFAULT_UNBOUNDED_STRING_BUDGET` (200).
- Verify: when Thinking=ON, `reasoning_headroom_tokens` = `MODEL_REASONING_HEADROOM_TOKENS` (16384) is added. When OFF, 0.
- Verify: `max_output_tokens` is capped by `floor(context_window_tokens × MAX_OUTPUT_FRACTION (0.25))`.
- Verify: `effective_max_input_tokens` = `floor((context_window_tokens - max_output_tokens) × RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN (0.85))`.
- Verify: when `RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE` is set, it replaces the derived `max_output_tokens`.
- Verify: when Thinking=ON, `effective_max_input_tokens` is smaller (reasoning headroom reduces input budget). ICL capacity shrinks; relevance-tail pruning handles overflow.
- Verify: fixed prompt cost is tokenized from the actual rendered zero-ICL message envelope, including the compact field schema; raw Guidance length plus a fixed constant is not used.
- Verify: `truncation_attributed_schema_invalid` is `true` on the Operation Record when `finish_reason="length"` and `schema_valid_core=false`.

**Review selector endpoint (§10.2.4):**
- Verify: `GET .../review_selector/next` returns `example_key`, `example_state`, `has_existing_label`, `selection_mode` (`clip_diverse` or `phash_diverse`), `queue_empty`, `storage_ref`, `prior_verified_label_ref`.
- Verify: when `queue_empty=true`, `example_key` is null.
- Verify: `has_existing_label=true` when the example has `label_status=auto_labeled`.
- Verify: selector state persisted for reproducibility (§13.3).

**Structured generation runtime rejection (§6.2):**
- Verify: interactive invocations (`interactive_proposal`, `retry`, `rationale_regeneration`): on `json_schema` rejection after prior `supported` probe, retry once with prompt-only (omit `response_format`). Surface warning. `structured_generation_fallback_used=true` on Operation Record. Model config remains `supported`.
- Verify: evaluation and batch labeling: on `json_schema` rejection mid-run with `structured_generation_mode=auto`, run transitions to `failed` with `status_reason="structured_generation_rejected"`. No silent mixing of structured and prompt-only modes.
- Verify: `structured_generation_mode=prompt_only` omits `response_format` from all invocations in the run from the first item.
- Verify: `structured_generation_attempted` is `true` when `response_format` was included; `false` when omitted.
- Verify: prompt-only requests contain the actual compact field schema, including Core names/types and enum values, and never contain a dangling reference to an absent schema.

**Operation Record completeness (§13.1):**
- Verify: Operation Record persisted before model invocation begins. `inference_invocation_id` generated before the call.
- Verify: records persisted for all outcomes: success, schema_invalid, timeout, endpoint_error.
- Verify: all Generation Controls fields populated: `generation_preset_key`, `sampling_params_effective` (temperature, top_p, seed), `thinking_mode_effective`, `thinking_request_fields_effective`, `max_tokens_effective`, `reasoning_headroom_tokens_effective`.
- Verify: Visual Budget fields populated: `visual_budget_preset_key`, `visual_budget_params_effective`. Both null when model does not support visual budget.
- Verify: image transport fields populated: `image_transport_mode`, `image_format_transmitted`.
- Verify: provider usage and completion fields populated when reported: `finish_reason`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `truncation_attributed_schema_invalid`.
- Verify: `purpose` correctly set per invocation type: `interactive_proposal`, `evaluation`, `batch_label`, `rationale_regeneration`.
- Verify: persisted Operation Records, rendered prompt artifacts under `{project_dir}/artifacts/`, and API responses contain no API keys. Scan for patterns matching `nvapi-`, `Bearer `, and known key prefixes (§12.2).

**Labeling screen — proposal states (`LabelingPage.tsx`):**
- Verify: proposal loading (7.1): on entering labeling view or after Save advances, both image and proposal areas show loading state. Top bar and persistent elements are fully rendered and interactive during loading.
- Verify: proposal displayed, Core valid (7.2): image left, editable proposal form right. Core fields first as form-style editor, then Aux fields with secondary styling. If enabled, rationale is hidden by the anti-anchoring default. Three actions: [Save] (one-click when unmodified), [Skip], [Retry]. Aux warnings shown inline below Aux fields.
- Verify: proposal displayed, rationale visible variant (7.3): when rationale notes are enabled and `rationale_anti_anchoring=false`, rationale panel is visible below Aux fields from the start. All other elements are identical to 7.2.
- Verify: proposal failed — schema-invalid (7.4): failure message replaces proposal form. Distinguishes failure type: "Schema-invalid: the model returned JSON that does not match your Core schema." Lists Core errors. [Save] available for manual labeling (SME fills fields from scratch). [Skip] and [Retry] available. Image still displayed.
- Verify: proposal failed — timeout (7.4): "Timeout: the model did not respond within the deadline (180s)." Includes helper copy: "If this keeps happening, check with your administrator." Same actions available.
- Verify: proposal failed — endpoint error (7.4): "Endpoint error: could not reach the NIM endpoint." [Report NIM Issue] generates Action Request with endpoint URL, model name, and error details. Same actions available.
- Verify: missing image (7.5): broken-image placeholder on left. "Image not found at original location." with expected path shown. [Report Missing Files] generates Action Request. Only [Skip] available (no Save/Retry). Proposal not attempted.
- Verify: after [Save], the system immediately advances to the next image and requests a new proposal. No manual navigation required.
- Verify: persistent elements on the labeling screen: Verified/Unlabeled/Omitted counts in the status bar; [Add Images] action always available. When at least one Student model is registered, [Models & Results] appears beside the Scale-Up readiness action and navigates directly to Compare & Benchmark; it is absent for projects that have not produced a Student.

**Labeling screen — edit flow and rationale (`LabelingPage.tsx`):**
- Verify: when rationale notes are disabled, modifying a field never renders a rationale display/panel or calls regeneration; [Save] remains available and sends no rationale key or provenance metadata.
- Verify: when rationale notes are enabled and the SME modifies any field value, the rationale panel expands below the proposal fields and [Save] becomes disabled. [Reset] appears.
- Verify: rationale needs review (7.6): editable textarea pre-populated with model's original rationale. Two actions: edit the text directly, or click [Generate AI Rationale]. Save disabled; helper text: "Update the rationale before saving."
- Verify: rationale regeneration loading (7.7): inline loading replaces textarea content while Teacher generates. No rationale actions available.
- Verify: AI-regenerated, review required (7.8): regenerated text in editable textarea. Save still disabled. SME either edits the text (→ "Edited", Save enabled) or clicks [Approve AI Rationale] (→ "Approved", Save enabled).
- Verify: rationale edited or approved (7.9): Save enabled. [Reset] remains available. Whitespace-only changes do not count as meaningful (do not transition from "Needs review" to "Edited").
- Verify: [Reset] restores all fields (including rationale) to the VLM's original proposal, collapses the rationale panel, returns to unmodified state (Save becomes one-click again).

**Labeling screen — retry and top bar controls (`LabelingPage.tsx`):**
- Verify: retry (7.10): clicking [Retry] shows inline parameter controls pre-populated with current project settings. Five controls: Teacher dropdown (from catalog), Guidance version dropdown, Output Stability (Precise/Explore), Thinking (ON/OFF, hidden when model's `thinking_toggle.mode="none"`), Visual Budget (Fast/Balanced/High Detail, hidden when `visual_budget_support=unsupported`). [Cancel] returns to current proposal; [Retry] re-runs and returns to loading state (7.1).
- Verify: top bar (7.14): Teacher model selector dropdown + Output Stability segmented control (Precise/Explore) + Thinking toggle (ON/OFF) always on first line. Thinking hidden for `mode="none"` models (e.g., Mistral Large 3).
- Verify: top bar (7.15): Visual Budget segmented control (Fast/Balanced/High Detail) adjacent to Output Stability. Hidden when active Teacher's `visual_budget_support=unsupported`.
- Verify: top bar all-controls visible variant (Cosmos Reason2: thinking + visual budget) vs minimal variant (Mistral Large 3: no thinking, no visual budget) — both render correctly.

**Labeling screen — notices and terminal states (`LabelingPage.tsx`):**
- Verify: cold start notice (7.11): shown on first proposal when Verified=0. Spec-mandated copy: "This is your first label. The model has no examples to learn from yet. Accuracy improves immediately with every Edit." Dismissable, fires once per project.
- Verify: schema refinement reminder first (7.12): after 10th Verified label. "Need to adjust your schema? Fewer labels to re-do now." [Review Schema] navigates to Edit Guidance. [Dismiss]. Does not fire if SME already edited Guidance post-save.
- Verify: schema refinement reminder second (7.13): after 35th Verified label. "You have 35 labels. Schema changes mean more images to re-label." Same pattern. After this, no more reminders.
- Verify: prior-label hints (7.26): after semantic Core change, labeling screen is standard (no special mode). Examples with `prior_verified_label_ref` show annotated prior-label reference per field: "you edited" badge on previously-edited fields, VLM agree/disagree indication, schema-invalid prior values flagged with `!`, [Adopt prior] on schema-valid prior values. Progress indicator: "Prior labels: N of M re-labeled (prior edits first)." Adopting a prior value triggers Edit semantics and, only when enabled, rationale review. Once all prior-label examples re-labeled, message: "All prior labels re-labeled. Continuing with remaining Unlabeled."
- Verify: queue empty (7.27): "All images have been reviewed." with Verified/Omitted/Unlabeled counts. Actions: [Add More Images], [Restore N Omitted] (shown when Omitted exist), [Go to Scale-Up].

**Image Ingestion screen (`ImageIngestPage.tsx`):**
- Verify: file browser (4.1): path field serves as address bar (updates on browse) and direct entry (type + [Scan]). Tree shows directories first, then files, with checkboxes for multi-select. Recently used directories shown as shortcuts. When backend is remote, notice: "Browsing files on the server at {backend_host}."
- Verify: scan preview (4.2): after typing a path and clicking [Scan], results replace the tree area showing image count, skipped count, and skipped reasons. [Ingest All N] to proceed; [x Clear] to dismiss and return to tree.
- Verify: ingestion progress (4.3): progress bar with running counts (accepted, skipped, errors) updating as each item is processed. Skipped and errored items listed inline as they occur. Batch continues through failures (partial success).
- Verify: completion summary (4.4): total processed, accepted (now Unlabeled), skipped with reasons, errors with reasons. Per-item size warnings for images >20 MB or >8192 px longest edge (non-blocking — images still ingested). Total Unlabeled count. "CLIP embeddings computing in background..." note. Footer: [Add More Images] and [Continue ->].
- Verify: low image count warning (4.5): non-blocking info banner when fewer than 150 images are ingested. Spec-mandated copy explains that 150 is the mathematical minimum for a 40% allocation to fill the default 60-image Test Pool, assumes every image becomes Verified, and that real model-quality needs may be higher. The bundled 15-image sample is explicitly called a walkthrough that cannot pass the default Scale-Up gate. [Add More Images] and [Continue] remain available; this notice is not itself a gate.
- Verify: path not found error (4.6): inline error below path field: "Directory not found: {path}". Path field remains editable.
- Verify: permission denied error (4.7): inline error: "Permission denied: {path}".
- Verify: browse disabled error (4.8): when backend non-loopback + `IMAGE_ROOT` unconfigured, file browser and path entry entirely replaced by a single message with spec-mandated copy and [Back] only.

**Evaluation run endpoints and state machine (§10.2.16, §13.2.1):**
- Verify: `POST .../evaluation_runs` accepts `icl_mode` (`enabled`/`disabled`, default `enabled`) and `structured_generation_mode` (`auto`/`prompt_only`, default `auto`). Returns `run_id`, `pool_version`, snapshotted config fields, `created_at`.
- Verify: trigger rejects with 400 if Test Pool has no members.
- Verify: if a gate-basis evaluation is already running when a new gate-basis run starts, the in-progress run transitions `running` → `canceling` → `canceled` with `status_reason="superseded_by_newer_evaluation"`, and a new run starts. A Student benchmark run neither cancels a running gate-basis evaluation nor is canceled by one (§7.1).
- Verify: `GET .../evaluation_runs/{run_id}` returns full status, progress (`processed`/`total`), per-bucket metrics (`overall`, `returning`, `new`), `previous_overall_exact_match`, `coverage_gaps[]`, `icl_eligible_count_at_completion`.
- Verify: `GET .../evaluation_runs` lists runs newest-first with cursor pagination. Supports `status` filter.
- Verify: `POST .../evaluation_runs/{run_id}:cancel` transitions `running` → `canceling` → `canceled`. Returns 409 if already in a terminal state.
- Verify: Evaluation Run state machine enforces only allowed transitions: `queued → running | canceling | failed`, `running → completed | incomplete | canceling | failed`, `canceling → canceled | failed`. Terminal states (`completed`, `incomplete`, `canceled`, `failed`) never transition out.
- Verify: canceling is two-phase: stop dispatching new inferences immediately, wait for in-flight tasks to settle, then `canceled`. Only outcomes committed after the durable `canceling` transition are marked `ignored_due_to_run_cancellation=true`; finalization preserves earlier outcome authority, and ignored records never contribute to metrics.
- Verify: canceled runs never produce authoritative aggregate metrics, even if some child Operation Records finished successfully.
- Verify: restart recovery: evaluation runs in `queued`, `running`, or `canceling` → `failed` with `status_reason="backend_restart_interrupted"`.

**Evaluation configuration snapshot and reproducibility (§7.1):**
- Verify: on evaluation start, all semantic project configuration is read once and persisted on the Run Record: model, Guidance, preset keys, thinking mode, effective Inference Contract, and the credential-free version-2 `runtime_config_snapshot` containing model/endpoint values plus concrete sampling, visual, token-budget, ICL, and image-downscale inputs.
- Verify: `icl_eligible_count_at_start` recorded at run creation.
- Verify: configuration changes made after run creation do not affect the in-flight run. Change Teacher mid-evaluation → run continues with snapshotted Teacher.
- Verify: two runs with different Inference Contracts are flagged as not directly comparable.

**Returning vs New metric detail (§7.1.2):**
- Verify: Returning = examples present in both current and previous pool version snapshots. New = present in current but not previous. Overall = Returning + New.
- Verify: first evaluation (no previous snapshot) reports Overall only; `returning` and `new` are null.
- Verify: after a semantic Core change, the first evaluation under the new Guidance is treated as a fresh first evaluation (Overall only, no Returning/New split). Prior evaluation Run Records preserved for audit but not used as previous baseline.
- Verify: each bucket (Returning, New, Overall) reports full metrics: Exact Match rate, per-core-field match rates, per-value precision/recall/F1.
- Verify: `previous_pool_version`, `returning_example_keys[]`, `new_example_keys[]`, `previous_overall_exact_match` persisted on the Run Record.

**Scale-Up Readiness Gate endpoint (§10.2.17):**
- Verify: `GET .../scaleup_gate` returns `gate_status` ∈ {`not_ready`, `ready`} and `criteria[]` with per-criterion `{criterion_name, passed, current_value, threshold, message}`.
- Verify: five criterion names: `overall_exact_match`, `per_field_match`, `min_per_value_f1`, `accept_rate`, `min_test_pool_size`.
- Verify: `details` for `overall_exact_match` identifies the qualifying
  evaluation run and its snapshotted Teacher model/config, plus whether the
  current project configuration differs and which tracked fields changed. The
  Scale-Up Teacher-readiness card names the current Teacher separately from
  the evaluated Teacher; a historical score is never presented as if the
  mutable current Teacher produced it.
- Verify: `details` for `per_field_match` includes `failing_fields[{field_name, current_rate}]`. `details` for `min_per_value_f1` includes `failing_values[{field_name, value, f1, precision, recall}]`.
- Verify: `message` for each criterion is plain language. No MLOps jargon.
- Verify: endpoint is lightweight (no model invocation; queries persisted metrics and counts only).

**Evaluation strip on labeling screen (`LabelingPage.tsx`):**
- Verify: evaluation strip not rendered until Test Pool has members. Once the pool forms, strip appears with Test Pool counter (left), Auto toggle + [Evaluate] button (right).
- Verify: first recommendation (7.18): banner when pool reaches 5 images. Spec-mandated copy. Banner carries [Evaluate] and [Dismiss]; clicking the banner's [Evaluate] triggers an evaluation run in the background. Message MUST NOT contain "(threshold: N)" or similar jargon (§7.3.4). When a schema refinement reminder (§6.8, 7.12/7.13) is visible, the first-pool banner MUST be suppressed until the reminder is dismissed or acted on.
- Verify: running (7.19): "Evaluating N of M..." shown. [Evaluate] disabled. If settings changed since run start, non-blocking note: "Running with settings from {time}. Current settings differ." with [Cancel Evaluation].
- Verify: complete notification — first run (7.20): "Evaluation complete: 80% accuracy (5 images)." [View Results] / [Dismiss].
- Verify: complete notification — subsequent run (7.20): "Evaluation complete: 85% accuracy (+3 vs previous on same images)." [View Results] / [Dismiss].
- Verify: incomplete (7.21): "Evaluation incomplete: 2 examples failed. Results are diagnostic only." [View Results] / [Dismiss].
- Verify: structured generation rejected (7.21a): "Evaluation failed: structured generation rejected." [Restart with prompt-only] / [Dismiss].
- Verify: config change nudge (7.22): "Settings changed since last evaluation..." [Dismiss]. Dismissing acknowledges; next banner on next change.
- Verify: ICL growth nudge (7.23): "{N} new edits since last evaluation..." [Dismiss].
- Verify: Auto-Evaluate toggle (7.24): OFF default shows banners on triggers; ON silently runs evaluations instead.
- Verify: coverage warning (7.25): "Test pool has no examples with {field_name}='{value}' [...]" Non-blocking. Grouped per field.

**Evaluation Results detail panel (`LabelingPage.tsx`):**
- Verify: subsequent run (7.28): "Previous: 82% (15)" → "Same images now: 85% (15)" with green/red delta. New and Overall secondary. Per-field match rate bars. Coverage gaps shown at bottom.
- Verify: first run (7.29): Overall only (no Returning/New). Per-field match rates.
- Verify: per-value breakdown expanded (7.30): F1/precision/recall per value of categorical Core fields. Values below 60% flagged "below 60%". Missing test pool values show "— (no examples)". Integer/string show match rate only.
- Verify: incomplete run (7.31): warning "Incomplete: N of M examples failed. Results are diagnostic only." Metrics labeled as diagnostic.
- Verify: panel shows the run's config snapshot (Teacher, Guidance version, Generation Controls, pool version).

**Batch Labeling run endpoints and state machine (§10.2.6, §13.2.2):**
- Verify: `POST .../batch_label_runs` accepts `include_auto_labeled` (default false), `run_limit`, `structured_generation_mode` (`auto`/`prompt_only`), `icl_mode` (`enabled`/`disabled`, default `enabled`; §8.3), `filters` (ingested_after/before). Returns `run_id`, `status="queued"`, snapshotted config (guidance_id, model_config_id, preset, thinking, visual budget, icl_mode), `examples_total`.
- Verify: start rejects with 409 when Scale-Up Readiness Gate is not `ready`, referencing which criteria failed.
- Verify: `GET .../batch_label_runs/{run_id}` returns `status`, `progress` (processed/total), per-outcome counters (`examples_succeeded`, `examples_schema_invalid`, `examples_timeout`, `examples_endpoint_error`), `paused_reason` (when paused), snapshotted config, timestamps.
- Verify: `GET .../batch_label_runs` lists runs newest-first with cursor pagination.
- Verify: `POST .../batch_label_runs/{run_id}:resume` transitions `paused` → `running`, resets the consecutive failure counter. Returns 409 if not in `paused` state.
- Verify: `POST .../batch_label_runs/{run_id}:cancel` transitions `running` or `paused` → `canceling` → `canceled`. Returns 409 if in a terminal state.
- Verify: Batch Labeling Run state machine enforces: `queued → running | canceling | failed`, `running → paused | completed | canceling | failed`, `paused → queued | canceled | failed`, `canceling → canceled | failed`. Terminal: `completed`, `canceled`, `failed`.
- Verify: `completed` means all selected examples reached a terminal per-example outcome (success, schema-invalid, timeout, or endpoint_error) — not that all succeeded. Per-outcome counters show the breakdown.
- Verify: `paused` is reserved for circuit breaker behavior only (§8.2 step 8), not for foreground-priority holds.
- Verify: restart recovery: `queued`/`running` → `queued` with `recovered_from_restart=true` and auto-resume from the next reconciled item. `paused` stays `paused`. `canceling` → `canceled` only with persisted cancellation intent and valid item lineage; otherwise the affected run fails closed.

**Batch Labeling execution detail (§8.2):**
- Verify: configuration snapshot persisted on Run Record before processing begins: model, Guidance, preset keys, thinking mode, and credential-free version-2 `runtime_config_snapshot`. Model/endpoint and semantic process-config changes after run creation affect future runs only, including after restart or explicit Resume; live operational boundaries remain excluded.
- Verify: input selection defaults to `state="Unlabeled"` excluding `state="Omitted"`. When `include_auto_labeled=true`, previously Auto-Labeled examples are also included; their existing Label records are replaced with new output.
- Verify: when `BATCH_LABEL_RUN_LIMIT` is set, the run caps at that many examples (selected in ingestion order). Default null = all eligible.
- Verify: `structured_generation_mode` is snapshotted as `structured_generation_mode_effective` on the Run Record.
- Verify: visual budget uses the snapshotted preset for all examples — no per-example variation.
- Verify: seed injection: `seed_effective = derive_seed(batch_label_run_id, example_key)` per §2.1.
- Verify: per-example results keyed by `{batch_label_run_id}:{example_key}`. Re-execution does not duplicate persisted outcomes for the same key.
- Verify: only `schema_valid_core=true` outputs produce Label records (`label_status=auto_labeled`). Schema-invalid outputs are recorded on Operation Records only.
- Verify: Example transitions to `state="Auto-Labeled"` on successful per-example processing.

**Circuit breaker (§8.2 step 8):**
- Verify: consecutive failure counter rules: `timeout` → increment; `endpoint_error` → increment; `schema_invalid` → ignore (does not increment, does not reset); successful example → reset to 0.
- Verify: counter reaching `BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD` (default: 10) transitions the run to `paused` with `paused_reason="circuit_breaker_threshold_reached"`.
- Verify: user notification on pause: "Endpoint appears unreachable." with [Resume] and [Cancel].
- Verify: [Resume] resets counter to 0 and continues from the next unprocessed example.
- Verify: [Cancel] transitions to `canceled`; already-persisted results are retained.

**Schema-invalid manifest and dataset export endpoints (§10.2.24, §10.2.18):**
- Verify: `GET .../batch_label_runs/{run_id}/schema_invalid_manifest` returns list of schema-invalid examples with `example_key`, `validation_errors_core[]`, `inference_invocation_id`, and `total_count`.
- Verify: `POST .../dataset_exports` accepts `dataset_intent` (training/evaluation/testing), `label_tier_filter` (verified_only/auto_labeled_only/combined), `export_field_mode` (all/aux_and_core/core_only), optional `batch_label_run_id`, optional `selection_filters` (guidance_id). Returns 201 with `dataset_export_id`, `example_count`, and `status="running"` (`artifact_refs`/`manifest_ref` null) while the archive builds in the background; the record transitions to `completed` with `artifact_refs` (archive path + SHA-256 checksum) and `manifest_ref` populated, or `failed` with `status_reason`.
- Verify: exports include only Verified examples under the current Guidance. After a semantic Core change, old labels are deleted — only re-labeled examples are exportable.
- Verify: Auto-Labeled exports include only `schema_valid_core=true` outputs.
- Verify: DatasetExport record persisted with `dataset_intent`, `export_field_mode`, `guidance_id`, `selection_definition_snapshot`, `example_count`.
- Verify: `GET .../dataset_exports/{id}` retrieves the full record. `GET .../dataset_exports` lists exports with pagination, filterable by `dataset_intent`.
- Verify: `GET .../dataset_exports/{id}/archive` streams only a completed
  project-contained `.tar.gz` with `application/gzip`, an attachment filename,
  and `X-Checksum-SHA256`; it works through both the Vite and nginx same-origin
  paths without requiring client access to the backend workspace filesystem.

**Scale-Up Hub (`ScaleUpHubPage.tsx`):**
- Verify: gate not ready (10.1): "Not ready for Batch Labeling." with plain-language next steps (1–2 items). Examples: "Continue labeling — the test pool needs 17 more images." / "The model struggles with 'scratch' in damage_type — 67% accuracy (need 80%)." [Go to Labeling] and [Details] shown. [Details] opens full 5-criterion breakdown.
- Verify: gate ready (10.2): "Ready for Batch Labeling." [Details] available. [Run Batch Labeling] enabled — clicking it IS the SME's confirmation.
- Verify: primary CTAs (10.3): [Run Batch Labeling] disabled when gate not ready (reason shown below); [Train a Student] always enabled as navigation. The suite-launch endpoint owns authoritative TAO/workspace/timeout/data/role validation after [Start Training], including at least one non-pool Verified training example and the configured Test Pool minimum. Both hub CTAs are always visible.
- Verify: TAO setup Action Request (10.4): [Request TAO Setup] expands read-only pre-filled block with required config fields, target models, verification endpoint. [Copy to Clipboard]. Content does not contain secrets.

**Batch Labeling pre-run screen (`BatchPreRunPage.tsx`):**
- Verify: configuration review (11.1): shows Teacher, Guidance version, ICL edit count, Visual Budget (when supported), input count (Unlabeled excluding Omitted). Auto-Labeled notice always present: "Batch Labeling generates Auto-Labeled outputs. These are not ground truth until reviewed."
- Verify: the selected Teacher endpoint's derived `usage_policy` resolves before launch. A seeded hosted API Catalog endpoint displays `Endpoint Use: NVIDIA API Catalog · evaluation only`; Run opens a confirmation naming the planned image count, trial-credit limitation, production-endpoint path, and NVIDIA API Trial Terms. Cancel and Configure do not create a run; Continue evaluation creates exactly one. Operator-managed endpoints launch without the confirmation and are not described as entitlement-verified.
- Verify: advanced filters collapsed by default. Expanded: ingestion date range, "Include previously Auto-Labeled images" checkbox (off by default). When checked, input count updates to show both pools.
- Verify: no-unlabeled state (11.2): input count 0 for Unlabeled. If Auto-Labeled exist, SME can check the include toggle. Otherwise [Add Images] / [Cancel].
- Verify: [Run Batch Labeling] launches the run and navigates to Run Status screen (12).

**Batch Labeling Run Status screen (`BatchRunStatusPage.tsx`):**
- Verify: running (12.1): progress bar with processed/total. Live schema-valid Core rate vs schema-invalid Core rate with common error types. [Download manifest] for schema-invalid examples. Snapshotted config shown.
- Verify: paused/circuit breaker (12.2): "Endpoint appears unreachable. 10 consecutive failures." Already-persisted results visible. [Resume] and [Cancel].
- Verify: completed (12.3): final counts and rates. Schema-invalid and errored excluded from export. [Export Dataset] CTA. [Back to Scale-Up].
- Verify: failed (12.4): error details. Already-persisted results retained and exportable if any exist. [Export Partial] / [Back to Scale-Up].
- Verify: structured generation rejected (12.4a): "Run failed: structured generation rejected." [Restart with prompt-only] restarts with same config, `prompt_only` mode. [Export Partial] / [Back to Scale-Up].
- Verify: canceled (12.5): "Canceled by user after circuit breaker pause." Partial results retained. [Export Partial] / [Back to Scale-Up].

**TAO job endpoints (§10.2.7):**
- Verify: `POST .../tao_jobs` accepts `student_base_model_config_id`, `dataset_export_ids[]`, `job_config` (with `training_backend`, `training_preset`, `training_policy_type`, `lora_config`, `hyperparameters`, `parallelism_config`, `tao_release_version`, `cosmos_rl_container_tag`), `tao_create_job_request`. Returns the full TAOJob record with `status="submitted"`.
- Verify: server validates `dataset_export_ids[]` and `student_base_model_config_id` belong to the same `project_id`.
- Verify: for `action="train"`, server validates all referenced DatasetExport records share the same `export_field_mode`. Mixed modes → `400` with `MIXED_EXPORT_FIELD_MODE`.
- Verify: `tao_create_job_request` is persisted exactly as submitted with a checksum/hash for integrity.
- Verify: TAOJob record persisted before triggering the external TAO job. If external submission fails, TAOJob transitions to `failed` with error payload reference.
- Verify: `GET .../tao_jobs/{tao_job_id}` returns status, `tao_status_raw`, progress (epoch_current/total, eta_seconds, metrics_latest), outputs (artifacts, logs_ref, metrics_ref), timestamps.
- Verify: `GET .../tao_jobs` supports cursor pagination and `status` filter.
- Verify: `refresh=true` query parameter on get forces a poll-update before responding (rate-limited).

**TAO job state machine detail (§9.7.2):**
- Verify: ten canonical statuses: `not_started`, `submitting`, `submitted`, `queued`, `running`, `paused`, `succeeded`, `failed`, `canceled`, `deleted`.
- Verify: `not_started` used for pre-created chain jobs awaiting predecessor. `submitting` is local only (POST to TAO in progress, no `tao_external_job_id` yet).
- Verify: allowed transitions enforced: `not_started → submitting | failed | canceled`; `submitting → submitted | failed`; `submitted → queued | running | failed | canceled | succeeded`; `queued → running | failed | canceled | succeeded`; `running → succeeded | failed | paused | canceled`; `paused → running | canceled | succeeded`; any → `deleted`. No other transitions permitted. F9 amendment 2026-05-05: ``submitted/queued/paused → succeeded`` direct transitions skip ``running`` to handle fast-TAO races where the poll cadence misses an intermediate status.
- Verify: terminal states (`succeeded`, `failed`, `canceled`, `deleted`) never transition to non-terminal.
- Verify: TAO raw status mapping is case-insensitive: `Done → succeeded`, `Failed → failed`, `Error`/`Errored → failed` (FTMS 6.26.3 surfaces container-level worker failures as raw status `"Error"` rather than `"Failed"`; without this mapping the polling loop never detects the terminal state), `Running → running`, `Queued`/`Pending → queued`, `Paused → paused`, `Canceled`/`Cancelled → canceled`. Unknown raw status → conservative non-terminal default (`running` if active progress; `queued` if not started; else `running`).
- Verify: `tao_status_raw` is persisted exactly as returned from TAO API alongside the canonical mapped status.

**Training preset system (§9.7.3.1):**
- Verify: four presets resolve to deterministic epoch patches: Quick=1, Standard=3, High Quality=9, Max Quality=18. All presets include `resume=false`, `ckpt.enable_checkpoint=true`, `ckpt.save_freq_in_epoch=1`, `ckpt.max_keep=8`, `ckpt.export_safetensors=true`.
- Verify: `training_preset` (user-facing label) and `hyperparameters` (resolved patch) both persisted in `job_config`.
- Verify: preset resolution is deterministic: the same preset + base model always produces the same `hyperparameters` patch.
- Verify: the resolved patch is applied to the TAO default `specs` to produce `tao_create_job_request.specs`.

**TAO polling contract (§9.7.4):**
- Verify: backend polls TAO for TAOJob records in non-terminal statuses. Terminal jobs are polled once to finalize outputs then polling stops.
- Verify: polling updates `status`, `tao_status_raw`, `progress` (when present), `started_at`/`completed_at` (when known), `last_polled_at`.
- Verify: poll errors do not corrupt last-known-good status. `poll_error_ref` is updated with the error; canonical `status` remains unchanged.

**TAO required outputs (§9.7.5):**
- Verify: on job success, the system retrieves artifacts via the workspace-S3 path described in §9.7.5: `GET .../jobs/{job_id}:list_files` enumerates workspace-relative keys, then boto3 `GetObject` pulls the bytes directly from the workspace S3 bucket. For `train`/`quantize` the latest `safetensors/epoch_<N>/` directory is mirrored under `{project_dir}/artifacts/tao_jobs/{tao_job_id}/` (NIM-loadable HF flat layout). For `evaluate` the single `evaluate_results.tar.gz` is downloaded + extracted, and per-sample predictions are translated into `{cache_dir}/per_sample_predictions` per §9.7.5.1. Artifact references are persisted on the TAOJob record's `outputs.artifact_cache_dir`.
- Verify: logs are durably tracked — at minimum `logs_ref` (stored in system) or `tao_logs_ref` (snapshot from `GET .../jobs/{job_id}:logs`).
- Verify: metrics/progress fields tracked on the TAOJob record: `epoch_current`, `epoch_total`, `eta_seconds`, `metrics_latest` (object), `metrics_history_ref`. All nullable; populated as TAO reports them.
- Verify: exact training configuration persisted (`tao_create_job_request` + resolved specs) for reproducibility.

**TAO submission protocol and restart recovery (§9.7.7):**
- Verify: submission protocol: (1) transition to `submitting`, (2) send TAO create-job POST, (3) on success persist `tao_external_job_id` and transition to `submitted`, (4) on failure transition to `failed` with `error_ref`.
- Verify: restart recovery — `submitting` with `tao_external_job_id = null` → `failed` with `status_reason="submission_interrupted"`.
- Verify: restart recovery — non-terminal with known `tao_external_job_id` (`submitted`, `queued`, `running`, `paused`) → resume polling (TAO jobs execute externally and continue regardless of backend).
- Verify: chain advancement from persisted state — `chain_sequence = N` has `succeeded` and `N+1` has `not_started` → system submits N+1. If any prior job is `failed` or `canceled` → chain remains halted.
- Verify: system-generated chained TAO jobs omit `force_create` (TAO defaults to `false`).

**Training diagnostics and student model endpoints (§10.2.22, §10.2.20):**
- Verify: `POST .../training_preflight` returns the same checks enforced by
  suite launch plus authoritative Verified Training Pool, Test Pool,
  required Test Pool minimum, Auto-Labeled eligible/included, excluded, and
  usable totals.
- Verify: selecting Cosmos 3 Super with any quantization scheme returns a
  failed `quantization_compatible` check and starts no export, provisioning,
  or TAO job; the same full-weight selection with an empty scheme list passes
  that check as baseline-only.
- Verify: `min_test_pool_size` fails when the active-Guidance Test Pool is
  below `max(1, project.scaleup_min_test_pool_size)`; the failure blocks suite
  launch and base provisioning without creating a TAO setup Action Request.
- Verify: suite materialization rechecks the data requirements after base
  provisioning and rejects a final evaluation export below the snapshotted
  required minimum before uploads or TAO job creation.
- Verify: `POST .../training_presets:resolve` returns deterministic per-model patches without probing TAO.
- Verify: `GET .../student_models` lists StudentModel records with cursor pagination.
- Verify: `GET .../student_models/{id}` returns full record: `student_base_model_config_id`, `tao_job_id`, `dataset_export_ids[]`, `training_preset`, `lora_config`, `checkpoint_packaging_status`, `quality_status`, `serving_status`, derived `serving_benchmark_current` / `serving_benchmark_blocker`, and `quantization_method`.
- Verify: `quality_status` set to `validated` when TAO evaluate succeeds and canonical re-scoring produces metrics. Set to `failed` when evaluate fails. Does not require NIM.

**TAO re-scoring detail (§9.7.6):**
- Verify: after TAO evaluate succeeds, the system retrieves per-sample predictions, parses each as JSON, and re-scores against Test Pool ground truth using the canonical Exact Match evaluator (Appendix A.2).
- Verify: re-scored metrics include overall Exact Match rate, per-core-field match rate, per-value precision/recall/F1 for categorical Core fields.
- Verify: both TAO-native aggregate metrics (informational) and system re-scored metrics (authoritative) are persisted on the evaluation Run Record. The re-scored metrics are used for quality comparison and gate decisions.
- Verify: the evaluation export uses the same `export_field_mode` as the training exports in the same chain — scoring a `core_only`-trained Student against `all`-field ground truth is prevented.

**Student Training screen (`StudentTrainingPage.tsx`):**
- Verify: **Validate training setup** defaults to one recommended small base,
  Quick, and FP8_DYNAMIC, displaying baseline + FP8 and an exact four-job
  workload. Unprovisioned seeded entries remain visible and selectable.
- Verify: **Compare candidate variants** is the explicit multi-base,
  Standard-preset, multi-quantization advanced intent.
- Verify: the base selector excludes Cosmos 3 Super even when the backend
  catalog response includes it, and both validation and comparison submissions
  keep `enable_lora=true`. No Full-weight selector is rendered.
- Verify: training data displays backend-authoritative Verified Training Pool,
  Test Pool, Auto-Labeled eligible/included, exclusion reasons, and final
  usable count. Auto-Labeled remains deselectable; Test Pool never trains.
- Verify: the visible preflight shows checking and each backend result;
  TAO-infrastructure failure includes **Request TAO setup**; transient failure
  includes Retry. Raw status/body JSON is never shown as the SME message.
- Verify: Start is fail-closed on form or preflight/data failure.
- Verify: Start first opens a mandatory confirmation with selected models,
  resolved preset, variants, exact train/evaluate/quantize/benchmark job count,
  usable/Test Pool counts, and long-running remote-infrastructure warning. The
  selected-model names preserve the exact request/execution order instead of
  being re-sorted into catalog order.
- Verify: explicit confirmation creates the suite and navigates to the
  Training Job Monitor. Missing bases are provisioned together under one
  conditional setup step before dataset exports and TAO chains.

**Training Job Monitor (`TrainingJobMonitorPage.tsx`):**
- Verify: `provisioning_run_id != null` renders one compact **Provision Student Bases** step with the selected missing model names and an NVIDIA-green running/completed status. No blue info treatment is used. If `provisioning_run_id == null`, the entire setup section is absent.
- Verify: a failed pre-chain workspace transfer with both frozen export IDs linked returns backend-derived `setup_retryable=true` and renders its actionable `setup_error_ref` plus **Retry Dataset Upload**. The action resubmits every persisted suite-request field with the same idempotency key. Integrity failures return `setup_retryable=false` and render the diagnostic without a retry action because they require fresh exports and a new key.
- Verify: project re-entry while a Training Suite is non-terminal (`provisioning`, `preparing`, `initialized`, or `running`) resumes the newest active suite in the Training Job Monitor. The monitor does not show a **Back to Scale-Up** action; the global **Projects** link is the deliberate exit, and reopening that project resumes its active Training Jobs.
- Verify: a secondary **Cancel Jobs** action occupies the former **Back to Scale-Up** position for a non-terminal suite. A confirmation explains that completed work is preserved and remote cancellation is best effort. On confirmation, the UI calls the suite-level cancel endpoint, hides the action after the suite becomes `canceled`, and reports any unconfirmed TAO cancellations as a warning. The terminal banner exposes a right-aligned, bold white **Back to Projects** action. Reopening the project after local cancellation follows the normal project entry route instead of resuming Training Jobs.
- Verify: the Training Job Monitor uses a display-label layer: canonical TAOJob statuses map to user-facing labels per the mapping in `src/ui/src/lib/training/statusDisplay.ts`. Every canonical status has a documented display label. Raw backend enum values never appear as badge text.
- Verify: after TAO reports `succeeded`, the monitor shows **Finalizing** and
  continues polling until `outputs_fetch_status` reaches `completed` or
  `failed`; the downstream card identifies artifact handoff rather than
  incorrectly claiming the TAO predecessor is unfinished.
- Verify: not started (14.0): `not_started` renders as **Not Started** (neutral badge). Visible in the chain display for jobs awaiting predecessor completion.
- Verify: submitting (14.0): `submitting` renders as **Submitting** (info badge + spinner). Brief transitional state while the system sends the job to TAO.
- Verify: submitted (14.1): `submitted` renders as **Submitted** (info badge). Queued (14.1): `queued` renders as **Queued** (info/neutral badge). Both shown on the same card layout; badge updates as status transitions. Card shows model, preset, policy, dataset info. "Waiting for TAO to start..."
- Verify: running (14.2): `running` renders as **Running** (success/info badge + spinner). Epoch progress, ETA, latest metrics, and [View Logs] render independently only when TAO supplies each value; absent telemetry produces no label, dash placeholder, empty body, or reserved action row.
- Verify: paused (14.3): `paused` renders as **Paused** (warning badge). Progress frozen. "Job paused by TAO." [View Logs] / [Cancel Job].
- Verify: completed (14.4): `succeeded` renders as **Completed** (success badge), not "Succeeded." Final metrics and duration remain visible with [View Logs]. Backend artifact references and workspace paths are not rendered, and no per-file download action exists; retrieved outputs remain internal inputs to evaluation, recovery, and portable NIM bundle construction.
- Verify: failed (14.5): `failed` (without `chain_halted_reason`) renders as **Failed** (error badge). Sanitized error details. Remote TAO failures expose [View Logs] / [Report TAO Issue]. A `student_nim_local` evaluation failure instead identifies Student NIM serving validation and links to Compare & Benchmark; it never reports the local failure as a TAO issue.
- Verify: canceled (14.6): an operator/upstream `canceled` job renders as **Canceled** (neutral badge). Cancellation timestamp and complete current/total progress render only when supplied; partial or absent telemetry is omitted. A durable `auto-skip:` evaluation instead renders as **Not Required** and explains that local Student NIM validation continues in Compare; it is success-equivalent chain policy, not a cancellation.
- Verify: deleted (14.7): `deleted` renders as **Deleted** (subdued/neutral badge). "Job removed from TAO. Record preserved locally for audit."
- Verify: full chain display (14.8): all jobs for each base model grouped under model header. Chain progress line uses compact family labels (for example, "8B: done  2B: 5 of 6" and "Nano: done  Super: 2 of 4"). Job headings humanize canonical quantization enums (for example, `FP8_DYNAMIC` is displayed as **FP8 Dynamic**) while API and persistence values remain unchanged. Jobs show display labels inline (Not Started, Submitting, Submitted, Queued, Running, Completed, Failed, Halted, Canceled, Deleted). Once every job is terminal, [Compare Students] is enabled when at least one finalized training artifact exists, even if an independent model chain failed; Compare remains responsible for each Student's quality and serving eligibility.
- Verify: chain halted (14.9): failed job shown with **Failed** badge and error. Subsequent jobs marked **Halted** (warning badge) with a concise outcome such as "Chain halted: {failed_job} failed." or "Chain halted: {canceled_job} canceled by SME." Durable sequence numbers and job IDs remain available through the API but do not appear in SME-facing copy. **Halted** is used for `failed` + `chain_halted_reason != null` — distinct from **Failed** (job ran and errored). "Skipped" MUST NOT be used.

**Student NIM deploy endpoint (§10.2.20):**
- Verify: `POST .../student_models/{id}:deploy_nim` with `nim_endpoint_url=null` triggers local orchestration (Tier 1). Returns `nim_preflight_status`, `nim_preflight_details`, `nim_deployment_mode="local"`, `status="deploying"`.
- Verify: with `nim_endpoint_url` provided, system skips local orchestration and runs evaluation against that endpoint. Returns `nim_deployment_mode="external"`.
- Verify: when local preflight fails, returns `nim_preflight_status="failed"` with per-check diagnostics so the frontend can generate an Action Request.
- Verify: progress reported via SSE events. Container lifecycle events logged on StudentModel record.
- Verify: variants deployed sequentially; each container stopped before the next starts.

**Benchmarking detail (§9.5.2, Appendix E):**
- Verify: latency captured at configured concurrency levels: `STUDENT_LATENCY_TEST_CONCURRENCIES` (default: [1, 8, 24]).
- Verify: per-variant results include p50/p90/p99 end-to-end latency at each concurrency level (client boundary).
- Verify: TTFT (`time_to_first_token_seconds`) and ITL (`time_per_output_token_seconds`) histogram summaries (p50/p90/p99) collected when available from NIM `/v1/metrics`.
- Verify: NIM Prometheus metrics collected when endpoint is self-hosted and exposes `/v1/metrics`: `request_failure_total`, `request_success_total`, `gpu_cache_usage_perc`.
- Verify: per-variant benchmark bounded by `NIM_BENCHMARK_TIMEOUT_S` (default: 1200s). On timeout, the variant is marked `timed_out`, diagnostics preserved, and the queue continues with the next variant.
- Verify: startup bounded by `NIM_STARTUP_TIMEOUT_S` (default: 1200s). On startup timeout, the variant is marked `failed` and the queue continues.
- Verify: results shown progressively — each variant's card updates immediately when its benchmark completes. The SME does not wait for all benchmarks.
- Verify: failed or timed-out variants do not invalidate completed results from other variants.

**Inference Contract for Students (§6.11):**
- Verify: the Inference Contract is a concrete Pydantic model with 3 fields: `output_field_mode` ∈ {`all`, `aux_and_core`, `core_only`}, `icl_field_mode` ∈ {`all`, `aux_and_core`, `core_only`}, `icl_max_examples: int | None` (legacy snapshots may carry the retired `icl_pinned_edits_k` key; dict-level comparisons ignore it). The Teacher contract is a fixed instance: `output_field_mode=all`, `icl_field_mode=core_only`. ICL field rendering and prompt assembly consume the Inference Contract, not hardcoded field-mode assumptions.
- Verify: a Student's `output_field_mode` and `icl_field_mode` are derived from the `export_field_mode` of the training DatasetExport records. No separate field mode stored on TAOJob or StudentModel.
- Verify: ICL demonstrations for a Student use `icl_field_mode` from the Inference Contract. The Teacher itself uses the fixed `core_only` ICL mode.
- Verify: the effective Inference Contract is snapshotted on evaluation Run Records and persisted as part of Student training lineage.
- Verify: two evaluation runs are only directly comparable if their effective Inference Contracts match. Mismatched contracts are flagged.
- Verify: no config key outside the Inference Contract can alter ICL field rendering for Students.
- Verify: Student runs are evaluated without ICL — TAO evaluation does not support ICL injection (§9.6), and Students deploy and serve bare in v1.0.
- Verify: release gating matches the Inference Contract — a Student MUST be evaluated under the same field-mode contract it will use in production (§9.6). Deploying a Student evaluated under a different contract than its training is flagged as invalid.

**NIM benchmark provenance (Appendix E):**
- Verify: every benchmark run captures required provenance: `student_model_config_id`, `nim_model_profile_requested` (nullable), `nim_model_profile_selected` + `nim_profile_metadata` (backend/precision/TP/optimization target), `quantization_method`, GPU model/count, dataset manifest ref + SHA-256, code commit SHA.
- Verify: decoding parameters are identical across all variants compared.
- Verify: `visual_budget_preset_key` is the same across all variants.
- Verify: NIM logs profile selection at startup; the system records it on the StudentModel.

**Compare & Benchmark screen — default and fallback views (`CompareBenchmarkPage.tsx`):**
- Verify: default view (15.1): TAO quality results available immediately. Teacher shown as accuracy baseline with Exact Match and per-field rates, sourced from the most recent completed Teacher-contract evaluation run (empty state when none exists). The view is project-wide: every retained Student remains visible and actionable, grouped under newest-first Training Run headings that show local-time start, preset, Guidance version, selected bases, and training/Test Pool counts. Each Student variant card shows Exact Match and per-field rates; the delta vs Teacher renders only when Guidance, frozen Test Pool, and effective Inference Contract match. Historical cards use their own immutable Guidance schema. "Not benchmarked" appears for variants without NIM results. Scope controls: [Benchmark All (N)], [Benchmark Selected] (checkboxes on cards), per-card [Benchmark]. Metric selector dropdown controls per-field sections across all cards. [Chart] toggle renders a grouped bar chart whose Student series labels include the Training Run timestamp; duplicate candidates remain uniquely identified.
- Verify: cross-run provenance (15.1b): when an older run differs from the latest run's Guidance, export field mode, or evaluation dataset checksum—or the evaluation set cannot be verified—the run heading states that cross-run score differences are directional. When a Student's quality run and the Teacher baseline differ in Guidance, Pool version, or effective Inference Contract, its numeric Teacher delta is hidden and the run heading explains why. The chart repeats a mixed-context warning whenever any visible run carries one of these warnings; it never silently presents mixed evidence as directly comparable.
- Verify: NIM preflight fallback (15.1a): when preflight failed, per-card [Benchmark] replaced by [Deploy for serving validation] (per §10.3.4 label list). Clicking expands `student_nim_deploy` Action Request with `docker run` command. Once deployed externally and endpoint registered, [Benchmark] becomes available.
- Verify: a packaged quality-failed Student remains visible and can run [Deploy for serving validation] through the normal NIM lifecycle while serving is not yet validated. The card shows live stages and states that only a backend-classified upstream loader gap may let the resulting clean NIM evaluation recover quality; arbitrary TAO failures remain quality-failed.
- Verify: benchmark in progress (15.2): card shows sequential stages: Starting container → Health check (with elapsed time) → Smoke inference → Running evaluation (N/M) → Running latency benchmarks (c=1, c=8, c=24) → Stopping container. Only one variant benchmarks at a time.

**Compare & Benchmark screen — metric drill-down and chart (`CompareBenchmarkPage.tsx`):**
- Verify: per-value expanded (15.3a): selecting "Per-value F1" from metric dropdown expands all categorical Core fields (enum, enum_set, boolean) to show each value's F1. Field-level match rate remains as anchor line; per-value numbers indented beneath. Integer/string show match rate only. Switching to "Per-value Precision" or "Per-value Recall" replaces all per-value numbers across all cards simultaneously. Works with both TAO-only and TAO+NIM results.
- Verify: comparative chart (15.3b): [Chart] toggle renders grouped bar chart below the cards. Chart uses the metric selected in dropdown (match rate → per-field bars; per-value F1 → per-value bars). All variants as adjacent color-coded bars. Teacher included as baseline. Hovering a bar shows exact percentage. Cards remain visible above chart. [Chart] toggle hides it.

**Compare & Benchmark screen — serving results and deployment (`CompareBenchmarkPage.tsx`):**
- Verify: serving latency (15.5): a serving-validated variant renders p50/p90/p99 at each configured concurrency level (default c=1, c=8, c=24), read from its serving run's persisted `metrics.benchmarks`.
- Verify: Student identity rows present canonical quantization in readable product
  copy (for example, `FP8 Dynamic`) while APIs and lineage retain the exact wire
  enum (`FP8_DYNAMIC`), and label the persisted GPU type/count as benchmark
  hardware rather than leaving it ambiguous as a deployment requirement.
- Verify: deployment handoff (15.6): [Request Production Deployment] on a variant card with `quality_status=validated`, `serving_status=validated`, and `serving_benchmark_current=true`. A historical synthetic serving result instead shows its evidence plus **AIPerf revalidation required** / [Revalidate with AIPerf], and both handoff and portable bundle reject it. An eligible card renders the `deployment_handoff` Action Request inline with the five headings Checkpoint, NIM Configuration, Model, Evaluation, and Training Lineage; includes checkpoint path, NIM config (NIM_MODEL_NAME, NIM_MODEL_PROFILE, backend, release), model metadata (base model, quantization, TP, GPU), evaluation snapshot (Exact Match, per-field, per-value, latency per concurrency, pool version, ICL mode, Guidance), and training lineage (TAO job, quantize job, datasets, preset, LoRA). [Download portable NIM deployment bundle] [Copy to Clipboard]. This is the sole trained-model delivery download; raw TAO outputs are not downloadable separately.
- Verify: closing validation may re-execute the same deterministic handoff name. It removes an exact exited validation container before launch and removes the container it started after stopping it. A running container with that name is never force-removed or displaced; the rerun fails with an actionable name-conflict result. Failure to stop or remove a container started by the validation is itself a failed handoff result rather than a hidden successful run with a stranded GPU resident.
- Verify: [Request Production Deployment] returns 409 if Student does not have both quality and serving validated.
- Verify: label-disambiguation invariant — [Deploy for serving validation] (15.1a) and [Request Production Deployment] (15.6) are visually + textually distinct. The same Student card MUST NOT render both buttons simultaneously, and neither label MUST ever appear as the bare "Request Deployment" string used by the legacy spec wireframes.

**Data model (§4.1, §4.2, §3.1):**
- Verify: state machine allows exactly the transitions in §4.1; no others.
- Verify: system operates with `Verified=0`; labeling loop functions; ICL selection returns empty.
- Verify: `state=Omitted` examples excluded from review selector and Batch Labeling inputs.

**Review actions (§4.5):**
- Verify: UI presents exactly four actions; Accept/Edit create Verified; Skip → Omitted (§4.5); Retry creates new invocation only.
- Verify: each Retry creates distinct invocation record linking to prior attempt.
- Verify: on Edit, backend persists `edited_core_fields[]` and `edited_aux_fields[]` (§6.3.2).

**SchemaCore and Core edit policy (§4.4, §4.4.1):**
- Verify: Core field rename (via `field_id`) does not trigger label invalidation; propagates to all Verified labels within the project.
- Verify: exact 1:1 enum value rename does not trigger label invalidation; propagates to all Verified labels within the project.
- Verify: presentation metadata edits do not trigger label invalidation.
- Verify: add Core field, remove Core field, type change, constraint change, allowed-value change, and Core ↔ Aux role change each trigger label invalidation (§4.4.1).
- Verify: semantic Core edits atomically: delete all Verified and Auto-Labeled Labels, transition those Examples to `Unlabeled`, preserve prior label data on Example record (`prior_verified_label_ref`, `prior_verified_outcome`), clear pool, reset baselines.
- Verify: all Auto-Labeled examples transition to `Unlabeled` on semantic Core change. Operation Records with `purpose=batch_label` are preserved for audit.
- Verify: system surfaces notice with Auto-Labeled revert count when applicable.
- Verify: confirmation dialog shows Auto-Labeled count when Auto-Labeled examples exist, with positive framing about improved Guidance carrying over.
- Verify: `semantic_core_change_from_guidance_id` and `schema_change_summary` persisted on the new Guidance record.
- Verify: prior label data preserved as `prior_verified_label_ref` on each formerly-Verified Example. The snapshot MUST be self-contained JSON including: the final verified label JSON, the VLM proposal JSON (from the Operation Record's `normalized_json_ref`), `edited_core_fields[]`, `edited_aux_fields[]`, and `verified_outcome`; `rationale_note` is included only when it exists. The VLM proposal is required for "You changed this from {original_VLM_proposal} to {your_correction}" badges (§4.4.1).
- Verify: Test Pool clears on semantic Core change (Label records deleted → pool assignments gone).
- Verify: after semantic Core change, old labels are deleted — no stale labels in ICL, evaluation, export, or training.
- Verify: review selector presents examples with `prior_verified_label_ref` before standard Unlabeled (prior Edits first).
- Verify: `schema_change_context_example_key` is recorded and presented first by the selector, then cleared.
- Verify: VLM proposal is the editable primary; prior label is annotated reference (shown when `prior_verified_label_ref` populated).
- Verify: "Adopt prior" per-field action available for schema-valid prior values; triggers Edit semantics.
- Verify: rationale review required when adopting prior values only while rationale notes are enabled (§6.3.2).
- Verify: ICL builds naturally from zero after semantic Core change (Edit-first selector ordering seeds ICL).
- Verify: no separate Re-verify state exists; only four example states.
- Verify: each SchemaCore field has an immutable `field_id` that persists across renames.
- Verify: Aux-only additive extensions still work in place without label invalidation.
- Verify: Core fields required for validity + scoring; Aux fields never block validity nor affect scoring.
- Verify: validation report separates Core vs Aux errors (§6.3.1).

**Automatic pool management (§4.3):**
- Verify: pool assignment is outcome-agnostic: both Accepts and Edits are candidates for the Test Pool.
- Verify: newly Verified examples are assigned to the Test Pool when it is below its target fraction; otherwise non-pool. Initial routing (§4.3.1): when CLIP embeddings are available and multiple candidates could fill the next slot, select the most dissimilar from current pool members; when embeddings are unavailable, assign the next Verified example directly (no diversity selection for initial routing).
- Verify: as total Verified grows, rebalancing (§4.3.2) promotes non-pool Verified examples (both Accept and Edit) to the Test Pool. Rebalancing uses the same switchover policy as the review selector (Appendix A.3): CLIP diversity when available, pHash diversity otherwise. Initial routing and rebalancing use different fallback behavior when CLIP is unavailable.
- Verify: pool assignments are durable: once assigned to the Test Pool, an example is never demoted.
- Verify: `pool_assignment` persisted on Label record (§13.7).
- Verify: Verified examples with `pool_assignment=null` (e.g., created before pool routing was active) are treated identically to non-pool during rebalancing — they are eligible candidates for promotion to the Test Pool.
- Verify: when an evaluation run starts, system auto-creates a frozen pool version snapshot (§4.3.3). Evaluation references the snapshot, not the live pool.
- Verify: if the Test Pool has no members, evaluation does not proceed.
- Verify: evaluation is available during Interactive Labeling (§7.1).
- Verify: on semantic Core change, pool assignments are cleared (Label records deleted, §4.4.1); the pool rebuilds as examples are re-labeled.

**Scale-Up Readiness Gate (§7.3):**
- Verify: gate criteria are system-evaluated automatically; no manual metric interpretation required.
- Verify: gate MUST NOT block Interactive Labeling, evaluation runs, or any other workflow outside Batch Labeling.
- Verify: Batch Labeling CTA is disabled when gate status is `not_ready`; enabled when `ready`.
- Verify: overall Exact Match criterion uses the most recent completed evaluation run; fails when no evaluation exists.
- Verify: per-core-field match rate criterion checks every Core field individually; reports which field(s) failed.
- Verify: minimum per-value F1 criterion checks every value of every categorical Core field; reports which value(s) failed with F1, precision, and recall.
- Verify: per-value F1 threshold is configurable per project (default 60%).
- Verify: Accept rate computed as rolling window over most recent `SCALEUP_ACCEPT_RATE_WINDOW` Verified labels; uses actual count as denominator when fewer exist.
- Verify: minimum Test Pool size criterion checks live Test Pool member count.


- Verify: gate re-evaluates after each new Verified label, after each completed evaluation run, and on Scale-Up hub page load.
- Verify: gate evaluation is lightweight (no model invocation; queries persisted metrics and counts only).
- Verify: all gate thresholds configurable per project; defaults match Appendix A.4 values.
- Verify: gate status is a structured object with `gate_status`, per-criterion details, timestamps.
- Verify: UI presents gate results in plain language; no MLOps jargon exposed to SME.
- Verify: Batch Labeling run execution rejects if `gate_status` is not `ready` (§8.2).

**ICL selection and prompt (§6.2, §3.3):**
- Verify: ICL eligibility = Edits only, excluding pool members (§6.2). Accepted examples are never selected for ICL.
- Verify: if no Edits exist, ICL selection returns empty list and the system proceeds with zero ICL examples.
- Verify: prompt contains `E01..` markers; never contains `example_key`.
- Verify: `icl_example_keys_used[]` persisted for every call.
- Verify: budget enforcement drops from the END of the selection-ordered list (relevance-tail pruning: the least query-relevant exemplars go first); can run with zero ICL.

- Verify: ICL selection deterministic for evaluation and batch labeling (§6.2).

**Rationale note (§4.4, §6.2, §6.3.2):**
- Verify: new Guidance and every starter template omit `rationale_note`; the Guidance toggle adds/removes the reserved `role="aux"`, `type="string"` field.
- Verify: when disabled, production prompts and output schemas do not mention `rationale_note`, models need not return it, Labeling has no rationale workflow, and new Verified labels contain no rationale data or provenance.
- Verify: when enabled, production Teacher requests `rationale_note` last and Core validity does not depend on it.
- Verify: the backend derives a canonical `generation_order` list: optional `rationale_note` when enabled, then remaining Aux fields by `display_order`, then Core fields by `display_order` (§4.4).
- Verify: the same `generation_order` is used consistently across all order-bearing outputs: prompt-rendered example JSON, ICL example JSON, derived JSON Schema `properties`, and exported training JSON string. A single test helper MUST compare all five outputs; if one drifts, the test fails.
- Verify: when enabled, `rationale_note` is first in the base derived JSON schema `properties`, followed by other Aux fields, then Core fields; it is absent when disabled.
- Verify: when enabled, `rationale_note` has the lowest `display_order`; Aux-before-Core group ordering is system-enforced.
- Verify: when enabled and `rationale_anti_anchoring=true` (default), UI does not show `rationale_note` alongside the initial proposal; hidden until SME begins modifying a field. When `false`, rationale is visible from the start. Rationale review on Edit is mandatory only while the feature is enabled.
- Verify: on Accept, rationale is never shown to the SME; Teacher's version is retained silently with `rationale_source="teacher_proposal"`.
- Verify: on Edit, Save button is disabled until rationale is in an approved state (Edited or AI-regenerated-and-approved).
- Verify: on Edit, SME can write rationale directly → `rationale_source="sme_edited"` → Save enabled.
- Verify: on Edit, SME can click Generate AI Rationale → waits for Teacher → reviews result → either edits further or clicks Approve AI Rationale → Save enabled.
- Verify: Approve AI Rationale stores with `rationale_source="teacher_regenerated_approved"`.
- Verify: whitespace-only modifications do not enable Save (do not count as meaningful edits).
- Verify: Undo changes resets rationale to Teacher's original and returns to "Needs review" state.
- Verify: every newly Verified example has a non-null, SME-reviewed `rationale_note` only when its active Guidance enables the feature; otherwise rationale provenance is null.
- Verify: production Teacher ICL renders Core fields only and never includes rationale or other Aux values.
- Verify: after relevance-tail token/image pruning, retained examples render in bookend order: top-1 first, top-2 last, with the middle examples stable.
- Verify: no config key outside the Inference Contract (§6.11) can alter ICL field rendering. All runtime ICL rendering decisions are attributable to the effective Inference Contract only.
- Verify: when enabled, no unreviewed rationale enters the system; all Edit rationale is either SME-written or AI-regenerated-and-explicitly-approved.

**Dataset export wire format (§9.3):**
- Verify: `annotations.json` is a top-level JSON array of sample objects, not a wrapper object.
- Verify: each sample has `id` (maps to `example_key`), `images` (array with exactly one relative path), and `conversations` (array of two turns).
- Verify: conversation turns use `from` / `value` fields, not `role` / `content`.
- Verify: human turn `value` is the rendered D.1 serving prompt (§9.3.2 F-W7) containing the literal `<image>` token exactly once at the query-image position and a compact prompt-visible label schema; verbose Guidance Description and Rules prose is omitted exactly as at serve time.
- Verify: assistant (gpt) turn `value` is a JSON **string** containing the serialized label, not a nested JSON object.
- Verify: label field ordering within the serialized string matches the production serving prompt, with `rationale_note` last when included.
- Verify: every `images[*]` path in `annotations.json` resolves to an existing file in the exported media payload.
- Verify: export archive layout (`images.tar.gz` or `images/` + `annotations.json`) matches §9.3.1.
- Verify: export validation passes against pinned `cosmos_rl_container_tag` per §9.3.4.

**Export field mode (§9.3.5):**
- Verify: `export_field_mode=all` includes rationale_note + other Aux fields + Core fields in the gpt turn, with rationale last to match serving.
- Verify: `export_field_mode=aux_and_core` includes Aux fields (excluding rationale_note) + Core fields in gpt turn.
- Verify: `export_field_mode=core_only` exports Core fields only in gpt turn (no Aux, no rationale).
- Verify: export field mode is persisted on DatasetExport record and applied consistently within a single export artifact.
- Verify: default `export_field_mode` is `all`.
- Verify: `dataset_intent=evaluation` export includes only Test Pool members with Verified labels as ground-truth gpt turns. Test Pool examples are excluded from training exports (§9.7.6).
- Verify: `dataset_intent=testing` export is structurally valid and uses the same Cosmos-RL format.
- Verify: all three intents (`training`, `evaluation`, `testing`) produce archives passing §9.3.4 validation (top-level JSON array, every image path resolves, conversation turns use `from`/`value`).

**Test pool coverage warnings (§7.1.3):**
- Verify: after evaluation completes, system checks enum, boolean, integer, and enum_set Core fields for missing values in test pool ground truth.
- Verify: coverage warning is non-blocking and does not affect metrics or gate.
- Verify: `coverage_gaps[]` persisted on Run Record.

**Structured generation (§6.2):**
- Verify: JSON schema derived deterministically from SchemaCore.
- Verify: runtime capability probe performed once per model; result persisted.
- Verify: if unsupported, system proceeds with prompt-only JSON.

**Evaluation (§7):**
- Verify: Exact Match on Core fields (Appendix A.2).
- Verify: after the concurrent evaluation burst, any failed examples are retried sequentially (concurrency=1) before marking the run incomplete (§7.1).
- Verify: incomplete evaluation runs (failures persist after sequential retry) do not satisfy the Scale-Up Readiness Gate.
- Verify: `evaluation_run` Run Record persists `generation_preset_key`, `thinking_mode_effective`, and `visual_budget_preset_key`.

**Evaluation runs (§7.1, §7.1.1, §7.1.2):**
- Verify: first evaluation recommended when Test Pool reaches `EVAL_FIRST_POOL_SIZE` (default 5). Auto-runs when Auto-Evaluate is ON.
- Verify: configuration change shows non-blocking recommendation banner. Auto-runs when Auto-Evaluate is ON.
- Verify: Auto-Evaluate toggle defaults to OFF.
- Verify: the `POST .../evaluation_runs` endpoint accepts manual invocation at any time without requiring a trigger condition (first pool, config change, or ICL growth). The endpoint's only precondition is pool membership (§7.1).
- Verify: ICL growth recommendation fires when ICL-eligible Edit count doubles since last baseline (last evaluation or last dismiss); non-blocking.
- Verify: both recommendation banners offer **[Dismiss]** (the persistent [Evaluate] button in the evaluation strip is the action).
- Verify: dismiss on ICL growth updates `icl_recommendation_dismissed_at_count`; next recommendation fires on next doubling.
- Verify: evaluation runs in background, does not block labeling.
- Verify: summary updates when evaluation completes.
- Verify: no summary line when no evaluation has run yet (handled by the first evaluation hint).
- Verify: if an evaluation is already running when a new one is triggered, the in-progress run is canceled and a new run starts.
- Verify: evaluation concurrency is provider-aware and configurable: `EVAL_CONCURRENCY_HOSTED` (default 1) for hosted endpoints, `EVAL_CONCURRENCY_SELF_HOSTED` (default 8) for self-hosted/local NIMs.
- Verify: Returning vs New metric split reported per §7.1.2; Previous Overall shown alongside current Returning.
- Verify: the gate always uses the most recent completed evaluation's overall metrics.
- Verify: configuration change detection is lightweight (compares persisted values only; no model invocation).

**pHash computation (§5.6):**
- Verify: pHash is computed by the restartable background ingest sweep after the 202 skeleton response; pending rows carry `phash=null`.
- Verify: pHash is CPU-only with no external dependency.
- Verify: pHash stored as hex-encoded string on Example record (`phash` field).
- Verify: pHash computation failure for an individual image does not fail the ingestion.
- Verify: pHash is unaffected by schema changes (§5.6.3).
- Verify: pHash is always computed regardless of `EMBEDDINGS_AUTO_COMPUTE` flag.
- Verify: on project open, the system checks that `project.phash_algorithm` is a supported value (`dct_phash_64` in v1). An unsupported value surfaces a clear error and prevents the project from proceeding until migration or recomputation. No per-computation runtime algorithm dispatch exists in v1.

**CLIP embedding computation (§5.5):**
- Verify: default-on via hosted embedding NIM when API key is configured.
- Verify: when `EMBEDDINGS_AUTO_COMPUTE=false` (feature flag), CLIP-style background computation is entirely skipped after ingestion. No embedding-NIM calls are made. The independent pHash background sweep still runs, and the review selector uses available pHash values with deterministic no-signal fallback while hashes are pending.
- Verify: the embedding-NIM probe runs at project creation when `embedding_provider` is configured. If the probe fails at project creation, the system re-attempts at first ingest. A single early failure does NOT permanently disable embeddings for the project. If the re-attempt at first ingest also fails, `embedding_provider` is set to `none`.
- Verify: after both probe attempts fail and `embedding_provider=none`, the review selector continues with pHash-diverse mode. The SME can trigger a re-probe by reconfiguring the embedding provider.
- Verify: background computation; labeling not blocked (§5.5.2).
- Verify: selector upgrades from pHash-diverse to CLIP-diverse as CLIP embeddings become available.
- Verify: embedding model consistency enforced within project (§5.5.3).

**Diversity-driven review selector (Appendix A.3):**
- Verify: reproducible given persisted scheduler state (§13.3).
- Verify: CLIP-diverse mode uses cosine similarity; pHash-diverse mode uses hamming similarity.
- Verify: each selection minimizes max similarity to recent window; stable tie-break.
- Verify: no random selection mode exists; pHash-diverse is the baseline.
- Verify: `REVIEW_SELECTION_MODE=auto` uses CLIP-diverse when available, pHash-diverse otherwise.

**Create Guidance screen (§6.6):**
- Verify: three stacked cards (Description, SchemaCore, Rules); two separated field sections (Core / Aux).
- Verify: all three cards always visible. On first load, Description is focused.
- Verify: Rationale notes toggle is shown in Aux, defaults off, and is freely reversible without an invalidation marker.
- Verify: backend-driven validation (§6.6.6): the builder's inline errors, badge, and save gating render the draft-validation response (debounced while editing, fresh on save); backend canonical validation (§10.2.2) provides authoritative schema derivation and `SCHEMA_COMPILE_FAILURE` detection.
- Verify: the draft validation endpoint and the Guidance save endpoint use the same backend derivation function.
- Verify: the Derived JSON Schema preview displays the canonical schema from the backend validation response, not a frontend-derived schema.
- Verify: fix-it affordances deterministic and re-trigger validation.
- Verify: save persists Guidance version; post-save Core fields support in-place edits (rename, exact enum rename, presentation); semantic Core changes trigger label invalidation with confirmation dialog.
- Verify: post-save edit UX shows per-control visual indicators: in-place controls have normal styling; label-invalidating controls (type dropdown, constraints, add/remove values, add/remove field, role change) show a warning icon/marker with tooltip.
- Verify: schema change confirmation dialog uses positive framing ("Update schema and re-label?"), not warning language; explains what happens next; primary action is "Update and Re-label".
- Verify: accessibility requirements (§6.6.9).

**Schema refinement reminders (§6.8):**
- Verify: first reminder fires once per project after Verified count crosses threshold 1 (default 10).
- Verify: second reminder fires once per project after Verified count crosses threshold 2 (default 35).
- Verify: each reminder is dismissable and never reappears in the same project after dismissal.
- Verify: neither reminder fires if user has already edited Guidance post-save in this project.
- Verify: if Verified count crosses both thresholds before either fires, only the higher applicable reminder fires.
- Verify: `schema_refinement_reminders_dismissed` resets to `0` on semantic Core change.
- Verify: both reminders link to Edit Guidance screen.
- Verify: setting a threshold to `0` disables that reminder; both `0` disables reminders entirely.

**Generation Controls (§6.7):**
- Verify: preset selection changes only `temperature` and `top_p`; effective values persisted in `sampling_params_effective` on Operation Record.
- Verify: Thinking toggle default is ON; OFF sends correct model-specific `chat_template_kwargs` for Qwen (`enable_thinking: false`) and Kimi (`thinking: false`) only when `thinking_toggle_support="supported"`.
- Verify: request-based modes with support `unknown` or `unsupported` hide the UI toggle, omit `chat_template_kwargs`, persist effective Thinking ON, and allocate reasoning headroom in proposal, evaluation, Batch Labeling, and rationale regeneration paths.
- Verify: Thinking toggle hidden/disabled for models where `thinking_toggle.mode="none"` (e.g., Mistral VLMs).
- Verify: `max_tokens` always explicitly set (never relies on NIM default 16).
- Verify: structured generation uses `json_schema` when supported; truncation tracked.
- Verify: evaluation and batch label calls set `seed_effective = derive_seed(scope_id, example_key)` and persist it.
- Verify: interactive proposals do not inject `seed`.
- Verify: thinking toggle rejection triggers `thinking_toggle_support="unsupported"`; interactive auto-retries once without override; evaluation/batch fails the run.
- Verify: automatic retry on truncation changes only output budget (preset and thinking unchanged).
- Verify: `generation_preset_key`, `sampling_params_effective`, `thinking_mode_effective`, `thinking_request_fields_effective`, and `max_tokens_effective` persisted on every Operation Record.

**Visual Budget Controls (§6.9):**
- Verify: `visual_budget_mode` seeded per model in the catalog: the Cosmos family (Reason2 8B/2B, Cosmos 3 Nano/Super) → `mm_processor_size`; Nemotron Nano VL → `mm_processor_tiles`; all other seeded teachers (Mistral Medium 3.5, Nemotron 3 Nano Omni, Step 3.7 Flash) → `none`.
- Verify: runtime probe uses a two-stage flow (baseline without `mm_processor_kwargs`, then capability with mode-specific kwargs) using a 512×512 RGB PNG probe image (§6.9.2). Baseline success + capability success → `supported`; baseline success + capability failure → `unsupported`; baseline failure → `unknown`. `visual_budget_mode=none` auto-sets `unsupported` without probing.
- Verify: preset selection resolves to model-specific `mm_processor_kwargs` based on `visual_budget_mode`.
- Verify: when `visual_budget_support=unsupported`, no `mm_processor_kwargs` sent and UI hides Visual Budget controls.
- Verify: `visual_budget_preset_key` and `visual_budget_params_effective` persisted on every Operation Record.
- Verify: evaluation and batch labeling use the project's `visual_budget_preset_key` (no per-example variation).
- Verify: Retry allows per-attempt `visual_budget_preset_key_override`.
- Verify: project-level default is `"high_detail"`.

**Capability re-probe (§4.8, §10.2.12):**
- Verify: re-probe resets all three capability fields to `unknown` and re-runs: the structured-generation probe (§6.2), the thinking override acceptance check (§6.7.4, only when `thinking_toggle.mode` is request-based), and the visual-budget probe (§6.9.2).
- Verify: re-probe returns `409 Conflict` if the model config is referenced by a running evaluation, batch labeling run, or training job.
- Verify: a failure in one capability check does not prevent the others from executing.
- Verify: a check timeout leaves the corresponding field as `unknown`.

**Operational logging — per-log-point (§11):**

Each log point is tested as part of its owning phase. The owning step verifies that the log point emits structured JSON with the 7 required base fields and the correct `component` and `details` per §11.

- Verify: log point 6 (capability probes, `component=capability_probe`) emits on each probe with type, request summary, response status, resulting classification. Escalates to `info` when a probe flips status. (Phase 3 Step 3.2)
- Verify: log point 7 (review selector, `component=review_selector`) emits on each selection with mode, candidate set size, selected key, diversity score. Escalates to `info` on CLIP↔pHash mode switch. (Phase 7 Step 7.1)
- Verify: log point 4 (ICL selection, `component=icl_selection`) emits with candidate pool size, selected `example_key[]`, total ICL count, and pruned count/keys; logs explicitly on cold start (zero candidates). Escalates to `info` when pruning drops examples. (Phase 7 Step 7.2)
- Verify: log point 1 (model invocation, `component=model_invocation`) emits at `info` with model name, endpoint, sampling params, visual budget params, finish_reason, latency, token counts. On automatic retry: logs trigger and what changed. (Phase 7 Step 7.3)
- Verify: log point 2 (schema validation, `component=schema_validation`) emits at `info` with validity classification, Core/Aux error counts and details, normalization steps. (Phase 7 Step 7.4)
- Verify: log point 8 (pool routing, `component=pool_routing`) emits on each newly Verified example with destination, outcome, pool count, target count, rebalancing triggered. Escalates to `info` when rebalancing changes membership. (Phase 9 Step 9.1)
- Verify: log point 3 (gate evaluation, `component=gate_evaluation`) emits at `info` with each criterion's current value, threshold, pass/fail, and overall gate_status. (Phase 9 Step 9.3)

**Operational logging — integration (§11, cross-phase checkpoint after Phase 9):**
- Verify: all seven log points fire in a single proposal → save → pool route → evaluate flow. Each entry shares the correct `correlation_id` for its scope (`inference_invocation_id` for proposal chain, `evaluation_run_id` for evaluation).
- Verify: `debug` level emits additional detail from all seven log points (candidate lists, scoring vectors, token breakdowns, normalization traces).
- Verify: logs do not contain API keys, raw image bytes, full prompt text, or user-identifiable information beyond `project_id` and `example_key`.

**Ingestion and querying (§10.2.1, §10.2.8):**
- Verify: ingestion API supports batch and idempotency; required fields persisted.
- Verify: batch ingestion processes items individually (partial success); per-item `status` includes `created`, `exists`, and `error`.
- Verify: examples query supports cursor pagination, `state` filter, Verified filters (`verified_after`/`verified_before`, `verified_outcome`, `guidance_id`, `schema_id_or_hash`), `pool_membership` filter (`test_pool`/`none`/`any`), stable ordering (`verified_at desc, example_key asc`), and `include=verified_label`.

**Image handling (§10.2.1, §10.2.9):**
- Verify: ingest validates file existence; missing files fail that item with a clear error.
- Verify: ingest validates image format via image library; only JPEG, PNG, WebP, BMP, single-page TIFF accepted; animated GIFs and multi-page TIFFs rejected.
- Verify: ingest warns (non-blocking) when file size exceeds 20 MB or longest edge exceeds 8192 px; image is still ingested.
- Verify: image serving endpoint streams from persisted `storage_ref`; never accepts arbitrary paths from the client.
- Verify: image serving endpoint returns 404 when example does not exist.
- Verify: if image file is missing at labeling time, UI shows broken-image placeholder and offers Skip.

**Operation records and lineage (§13.1, §13.5):**
- Verify: Operation Records persist for all outcomes with artifacts + validation report.
- Verify: Auto-Labeled uses `purpose=batch_label`; never overwrites Verified.
- Verify: dataset exports are first-class records with manifest refs.

**Guidance versioning (§4.4):**
- Verify: Guidance records immutable once persisted; active set via Project update.

**Batch labeling (§8):**
- Verify: Batch Labeling always uses `project.teacher_model_config_id` and `project.active_guidance_id`; no per-run model/guidance override.
- Verify: ICL is enabled for Batch Labeling by default; a per-run `icl_mode="disabled"` on the start request skips ICL selection, is snapshotted on the Run Record, and survives restart recovery (§8.3 F-S9 amendment).
- Verify: per-example operation IDs keyed `{batch_label_run_id}:{example_key}`; idempotent.
- Verify: exports include only `schema_valid_core=true` outputs; schema-invalid outputs are never exported.

**Student Training and deployment (§9, §5.5, §1.6):**
- Verify: TAO endpoint configuration is deployment-level (`TAO_API_BASE_URL`, `TAO_API_KEY`, `TAO_ORG_NAME`); all three required when Student Training is enabled.
- Verify: TAO connection probe runs on configuration; failure disables Student Training with a clear message.
- Verify: `TAO_API_KEY` stored as a secret; never logged.
- Verify: all TAO API calls use configured base URL and org name.
- Verify: preflight validates TAO reachability and base model `student_base` role (§4.8). Hardware constraints are not enforced by preflight; they surface through TAO job failures.
- Verify: `training_backend="cosmos_rl_tao_vlm"` persisted on every TAOJob; `training_policy_type="sft"` persisted.
- Verify: TAOJob state machine enforced (§9.7.2); poll-tracked; payload persisted; artifacts/logs/metrics tracked (§9.7.5).
- Verify: `lora_config` persisted on every TAOJob (§9.7.3.2); LoRA merge enforced before any NIM deployment path (§9.5.1).
- Verify: NIM VLM serves merged checkpoints only; no runtime LoRA adapter serving specified for VLM NIM (§9.5.1, §9.7.3.2).
- Verify: `resolved_training_fields` persisted with Cosmos-RL-specific fields (§9.7.3).
- Verify: `tao_release_version` and `cosmos_rl_container_tag` persisted on TAOJob; `nim_vlm_release_version` persisted on deployed Student.
- Verify: checkpoint handoff (§9.5.1) validates NIM-loadable directory structure before Student registration.
- Verify: quantization uses TAO `quantize` action exclusively (§9.8); supported methods: FP8_DYNAMIC, W8A8, W8A16, W4A16.
- Verify: TAO quantization represented as TAOJob with `action: "quantize"` and parent linkage.
- Verify: quantized checkpoints feed into §9.5.1 packaging step.
- Verify: each Student variant persists `quantization_method` (Appendix E).
- Verify: each Student variant is a distinct `model_config_id` with NIM profile metadata (requested + observed); suites compare latency + accuracy (Appendix E).

**Automatic TAO evaluation and job chaining (§9.7.6, §9.7.7):**
- Verify: when a TAO training job succeeds, the system automatically submits a TAO `evaluate` job against the Test Pool export.
- Verify: when a TAO quantize job succeeds, the system automatically submits a TAO `evaluate` job against the quantized checkpoint.
- Verify: the full chain (train → evaluate → quantize → evaluate) runs without SME intervention after "Start Training."
- Verify: the Test Pool is exported as a separate Cosmos-RL evaluation dataset; Test Pool examples are excluded from training exports.
- Verify: TAO per-sample predictions are re-scored by the system's canonical Core-field evaluator (Appendix A.2); re-scored metrics are authoritative.
- Verify: both TAO-native metrics (informational) and re-scored metrics (authoritative) are persisted on the evaluation Run Record.
- Verify: Validate training setup pre-selects FP8_DYNAMIC only and produces
  baseline + FP8 variants for one recommended small base; broader comparison
  is an explicit advanced intent.
- Verify: job chains are linked by `chain_id` and ordered by `chain_sequence`.
- Verify: if any job in a chain fails, remaining jobs are marked with `chain_halted_reason` and are not submitted.
- Verify: chains for different base models run sequentially.
- Verify: `evaluation_source` ∈ {`tao`, `nim`} is persisted on every evaluation Run Record.

**NIM local deployment orchestration (§9.5.2):**
- Verify: NIM deployment preflight checks Docker, NVIDIA Container Toolkit (GPU passthrough), GPU memory, NGC API key, NIM image accessibility, and checkpoint validation.
- Verify: preflight result (`nim_preflight_status`, `nim_preflight_details`) persisted on StudentModel.
- Verify: if preflight passes, system orchestrates local NIM container lifecycle (start → health poll → evaluate → benchmark → stop) per variant, sequentially.
- Verify: NIM startup timeout (`NIM_STARTUP_TIMEOUT_S`, default 1200s) is enforced; health polling uses `/v1/health/ready`.
- Verify: smoke inference runs after health check passes and before evaluation begins.
- Verify: NIM containers use `--shm-size=32GB`, persistent cache mount, and `-u $(id -u)`.
- Verify: each variant's container is stopped before the next starts (GPU resource sharing).
- Verify: if preflight fails, system generates a `student_nim_deploy` Action Request with exact `docker run` command and prerequisites.
- Verify: external NIM endpoints can be manually registered for evaluation when local deployment is not possible.

**Two-part Student readiness (§9.5, §13.13):**
- Verify: `quality_status` on StudentModel is set to `validated` after TAO evaluate succeeds; does not require NIM.
- Verify: `serving_status` on StudentModel is set to `validated` after NIM deployment + evaluation + every current AIPerf concurrency cell succeeds; legacy synthetic evidence remains historical and derives `serving_benchmark_current=false`.
- Verify: `deployment_handoff` Action Request and portable deployment bundle require `quality_status="validated"`, `serving_status="validated"`, AND a current AIPerf serving run; legacy/missing/drifted evidence fails closed.
- Verify: `deployment_handoff` evaluation snapshot includes per-value precision/recall/F1 for categorical Core fields (not just per-field match rates). §10.3.3 requires "at minimum: overall Exact Match rate, per-core-field match rates, per-value precision/recall/F1 for categorical Core fields, latency p50/p90/p99."
- Verify: Compare & Deploy screen shows TAO quality results immediately; NIM serving results shown when available.
- Verify: Teacher appears as accuracy baseline only; Teacher latency is not included in Student comparison (may be hosted externally).

**Partial quality_status (F35, §9.5):**
- Verify: when a NIM-source eval finishes `incomplete` with parseable rate ≥ `STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD` and prior `quality_status` is `pending` or `failed`, the paired StudentModel transitions to `quality_status="partial"` with `quality_evaluation_run_id` pointing at the NIM RunRecord.
- Verify: when prior `quality_status="validated"`, the F35 promotion helper is a no-op (audit invariant — `validated` is never demoted to `partial`); the existing `quality_evaluation_run_id` is preserved.
- Verify: when run finishes `incomplete` but parseable rate < threshold, `quality_status` is unchanged (stays at prior value); `_emit_failed` fires with `failure_stage="eval_failed"`.
- Verify: F35 logic is applied consistently across BOTH the local-mode and external-mode lifecycle paths in `student_nim_lifecycle.py` (the F14 strict gate at the external-mode call site is unified with the local-mode lenient gate via the F35 amendment).
- Verify: `deployment_handoff` Action Request returns 409 with body containing `conflict: quality_status_partial` when the paired Student has `quality_status="partial"` (distinct from `conflict: quality_status_not_validated` for `pending`/`failed`).
- Verify: F33 `:rerescore` returns 409 `student_not_failed` when invoked against a Student with `quality_status="partial"` (regression test — F33 guard semantics unchanged by F35).
- Verify: `STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD` default is 0.90 in `_defaults.py`; the value flows through `Settings` and is read by the lifecycle.
- Verify: Compare & Benchmark `<StudentVariantCard>` renders a yellow `<Badge>` reading "Quality: Partial" with the F35 helper line ("Re-run NIM evaluation to promote to validated …") only when `quality_status === "partial"`.
- Verify: `<ActionRequestPanel>` renders the F35-specific message when the conflict body contains `quality_status_partial`; the lookup precedes `quality_status_not_validated` so the partial substring matches first defensively.

**Action Requests (§10.3):**
- Verify: Action Request CTAs appear at blocker screens: Scale-Up hub (TAO not configured), onboarding step 2 (self-hosted NIM), and the Compare & Benchmark screen ([Deploy for serving validation] when NIM preflight failed; [Request Production Deployment] on dual-validated variants). Student Training surfaces suite-launch errors inline and has no separate preflight Action Request panel.
- Verify: each request type (`tao_setup`, `nim_setup`, `student_nim_deploy`, `deployment_handoff`) pre-fills the correct technical requirements and diagnostic context per §10.3.3.
- Verify: `student_nim_deploy` Action Request includes exact `docker run` command, checkpoint path, GPU requirements, NIM release, environment variables, health check commands, and temporary infrastructure note.
- Verify: `deployment_handoff` Action Request includes checkpoint ref, NIM config, model metadata, evaluation snapshot, and training lineage.
- Verify: `deployment_handoff` references only checkpoints with `checkpoint_packaging_status="validated"`.
- Verify: the portable deployment bundle applies every `deployment_handoff`
  gate, streams the complete project-contained NIM-loadable checkpoint payload
  (excluding TAO completion logs/state) plus manifest,
  checksums, README, launch, schema-aware serving request, health, and
  real-image structured-label verification helpers, and rejects symlinks/path
  escape without including a NIM image or credential.
- Verify: `deployment_handoff` `technical_requirements` populates the four §10.3.3 customer-facing keys: `nim_model_profile_recommended` (sourced from `student.nim_model_profile_selected ?? nim_model_profile_requested`), `gpu_requirements` (formatted `f"{gpu_count}× {gpu_type} (≥{memory_gb} GB)"` with the per-base-model memory minimum), `tensor_parallelism` (integer; sourced from `nim_profile_metadata.tp` with default 1), and `nim_env_vars_recommended` (dict with at minimum `NGC_API_KEY` placeholder + `NIM_MODEL_NAME`, plus `NIM_SERVED_MODEL_NAME` and `NIM_MODEL_PROFILE` when set).
- Verify: `deployment_handoff` `technical_requirements.docker_run_command` and `docker_run_args` are sourced via the canonical `local_nim_service.build_student_docker_run_*` wrappers — byte-equivalent (modulo the literal `NGC_API_KEY` value) to what `:deploy_nim` actually executed for that Student. The rendered command MUST include `-u $(id -u)`, the read-only `:ro` flag on the checkpoint mount, and the canonical `--gpus "device={n}"` form.
- Verify: Action Request content is read-only (no SME form fields); copy-to-clipboard is the only action.
- Verify: Action Requests do not contain secrets (API keys, credentials).
- Verify: AuditEvent with `event_type="action_request_copied"` logged on copy.

**Full-stack validation (Phase 12.4):**

These verify items are exercised by `scripts/full_stack_validation.py` —
an operator-driven closing smoke on a full-stack GPU host (1×A100 80 GB,
~1.5–2h). Each item carries two evidence states: **mock-validated** (CI,
no hardware, exercised via `tests/unit/test_full_stack_validation_mock.py`)
and **live-validated** (GPU validation window; evidence retained in the
project's internal engineering archive). Phase 12 formally
closes only when live-validated evidence lands.

- C20 (handoff `docker run` re-execution proof): For both Cosmos Reason2
  2B and 8B: parse `technical_requirements.docker_run_args` from
  `:deployment_handoff`, subprocess-exec it (live mode) or send the
  request to the mock NIM (mock mode), poll `/v1/health/ready`, send a
  `POST /v1/chat/completions` round-trip, assert the response contains
  schema-valid label JSON parseable as an object. When `--rps-root` is
  supplied, use the unquantized baseline Student as each base's stable
  representative and require one schema-parseable prediction from an image in
  each `rock`, `paper`, and `scissors` directory. Accept both raw JSON and a single
  surrounding Markdown JSON fence through the product's canonical fence
  stripper; HTTP success with missing or malformed category JSON is not a pass.
  Record the directory-ground-truth match independently; a wrong category is
  truthful model-quality evidence governed by the held-out evaluation and does
  not turn a successfully deployed NIM into an infrastructure failure.
  Live execution requires
  the operator's configured `NGC_API_KEY` in the validation process
  environment; the parsed argv stays unchanged and contains only
  name-only `-e NGC_API_KEY` forwarding. Both 2B and 8B MUST round-trip
  successfully and, when the RPS proof is requested, all six per-class
  predictions MUST be schema-parseable.
- C21 (2B vs 8B handoff content differentiation): The
  `:deployment_handoff` payload's `technical_requirements` MUST
  differentiate between 2B and 8B representative variants across:
  `nim_container_image` (different per-base NIM image),
  `nim_model_profile_recommended` (different pinned profiles when present, or
  both `null` when custom-checkpoint NIM images perform compatible automatic
  selection),
  `gpu_requirements` (different memory hint — 2B `≥36 GB`, 8B `≥56 GB`),
  `nim_env_vars_recommended` (different `NIM_SERVED_MODEL_NAME` and
  `NIM_MODEL_PROFILE` only when a profile is pinned), and
  `tensor_parallelism` (present + integer;
  may equal across single-GPU deploys — population is what matters).
- Verify: `closing_acceptance.json` schema includes `execution_mode ∈
  {"mock", "live"}`, `phase_a_complete: bool`, `phase_b_validated_count:
  int`, `phase_b_target_count: int`, `c21_differentiation: dict`,
  `c20_handoff_rerun: {two_b: bool, eight_b: bool, detail_2b: str,
  detail_8b: str, predictions_2b: list, predictions_8b: list}`. Each
  prediction carries `image_class`, `image_path`, parseability `ok`,
  `predicted_category`, nullable `matches_ground_truth`, bounded `raw_content`,
  and `detail`. The remaining top-level fields are
  `final_integration_checkpoint: bool`,
  `final_integration_skipped: bool`, `error: str`.
- Verify: Final integration checkpoint (Phase E) covers the full pipeline
  via `scripts/full_pipeline_smoke.py`: create project → configure NIM
  → ingest → Guidance → label 10 → evaluate → check_gate → batch_label →
  dataset_export → validate Cosmos-RL format per §9.3.2. Library
  function `run_full_pipeline_smoke()` is invoked from Phase E. Skips
  cleanly when `NVIDIA_API_KEY` is unset (final_integration_skipped=true).
- Verify: Cosmos-RL format validator rejects archives that are not a
  top-level JSON array, missing `id` / length-1 `images` / length-2
  `conversations`, missing the literal `<image>` token in the human
  turn, or whose gpt turn `value` is not a JSON-string-encoded object
  (a common drift point — nesting a dict instead of stringifying).
- Verify: `LOCAL_NIM_MOCK_ENDPOINT_URL` env-var hook in
  `services/local_nim_service.py` is the test seam ONLY — production
  code path runs verbatim when the var is unset. The hook covers
  `deploy_local_nim`, `stop_local_nim`, and `run_preflight_checks` for
  `role="student"`; Teacher and embedding deployments are not affected.

**CI / pre-commit pipeline (Phase 13 Step 13.1):**
- Verify: `.github/workflows/ci.yml` exists at the repo root, triggers on
  `push` and `pull_request` to `main` plus `workflow_dispatch`, and declares
  the eight release jobs: `backend-lint`, `backend-tests`,
  `backend-typecheck`, `frontend-lint`, `frontend-tests`, `frontend-build`,
  `dependency-audit`, and `compose-smoke`.
- Verify: all jobs run on `ubuntu-latest` without job-level
  `continue-on-error: true`. The seven non-Compose jobs use a 10-minute
  timeout; `compose-smoke` uses 20 minutes to accommodate cold image builds.
- Verify: `backend-tests` runs the unit suite with the configured coverage gate
  on Python 3.11, 3.12, and 3.13. `backend-lint` runs `ruff check`,
  `ruff format --check`, and the `AGENTS.md` / `CLAUDE.md` twin check.
  `backend-typecheck` runs strict pyright on `src/backend/`.
- Verify: `frontend-lint` runs ESLint, Prettier, and `pnpm typecheck` inside
  `src/ui`; `frontend-tests` runs `pnpm test`; and `frontend-build` runs
  `pnpm build` and produces the production JS/CSS bundles and `index.html`.
- Verify: `dependency-audit` checks the complete frozen backend lock with
  `pip-audit --no-deps --disable-pip` (uv, not pip, remains the resolver)
  and production frontend dependencies with `pnpm audit --prod --audit-level
  high`. The only package-native audit exception is
  `GHSA-qwww-vcr4-c8h2`: the affected unstable React Server Components APIs
  are absent from this client-only Vite SPA, and the upstream patched major
  requires a Node/React baseline beyond the Blueprint's current support
  contract. Remove that exception when React Router 8 is adopted.
  The default `aiperf` operational group overrides AIPerf 0.10.0's stale
  aiohttp/Pillow caps to the audited aiohttp 3.14.3+ and Pillow 12.3+
  releases; the real raw-payload integration test MUST pass with that locked
  environment.
  `compose-smoke` builds the Compose images, starts the stack, requires a
  healthy edge at `http://localhost:3000/health`, prints diagnostics on
  failure, and always tears the stack down.
- Verify: integration tests remain outside the default CI workflow because
  they start live services on fixed ports and may require credentials or
  Docker. The operator validation command is `uv run pytest
  tests/integration/ -q -n 0`; the suite rejects xdist execution.
- Verify: `.pre-commit-config.yaml` declares the seven hooks in execution
  order: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `ruff`,
  `ruff-format`, `gitleaks`, and `agents-claude-twins`. Formatting exclusions
  are hook-specific so gitleaks still scans every shipping text file. The
  pre-commit ruff revision tracks the resolved project minor.
- Verify: `scripts/ci-local.sh` reproduces the six deterministic core jobs
  (backend lint/tests/typecheck and frontend lint/tests/build). Network-bound
  dependency audits and the Docker Compose smoke remain explicit release
  commands rather than hidden local-script side effects.

**Secret scanning + organization-only SonarQube delegate (Phase 13 Step 13.1):**
- Verify: the private source profile keeps `.github/workflows/sonarqube.yml`
  with a manual trigger and uses
  `NVIDIA-AI-Blueprints/sonarqube-workflows/.github/workflows/sonarqube-reusable-template.yml@main`
  — the same ref pinned by Retail-Agentic-Commerce and
  Retail-Catalog-Enrichment. The `with:` block populates
  `organization`, `team`, `product`, `scmRepoName`, `projectTags`,
  and `language: python`. `secrets: inherit` is set so the org
  SonarQube token flows through.
- Verify: the curated independent public export omits that workflow file.
  GitHub resolves a private reusable-workflow reference before evaluating a
  job-level repository-owner condition, so an inaccessible org-private call
  cannot be made to skip cleanly in a public mirror. When the repository is
  installed under `NVIDIA-AI-Blueprints`, enable its push and pull-request
  triggers as part of the organization publication profile.
- Verify: the `gitleaks` pre-commit hook is configured with
  `repo: https://github.com/gitleaks/gitleaks` and a pinned
  `rev:` tag (not `main`, not unpinned). The repo includes a
  `.gitleaks.toml` at the repo root that `extend.useDefault = true`
  AND adds an `id = "nvidia-api-key"` rule with regex matching
  `nvapi-` followed by 40–80 token characters (covers
  `NVIDIA_API_KEY` and `NGC_API_KEY` formats — the documented
  minimal-setup credentials).
- Verify: the `nvidia-api-key` rule fires on a planted credential.
  Stage a file containing `NVIDIA_API_KEY = "nvapi-<60 token chars>"`
  and run `pre-commit run gitleaks` — the hook MUST exit non-zero
  with `RuleID: nvidia-api-key` reported. Short-form test fixtures
  matching `nvapi-(test|fake|stub|placeholder|example|dummy)…` MUST
  NOT trigger the rule (the `[allowlist]` block in `.gitleaks.toml`
  exempts them).
- Verify: the independent-public CVE backstop is the locked-set-only
  `pip-audit` plus `pnpm audit` CI job. SAST remains delegated to the NVIDIA
  organization SonarQube workflow when that publication profile is enabled;
  the repo does not add a parallel `bandit`,
  `eslint-plugin-security`, or custom suppression-ledger program.

**Backend type checking (Phase 13 Step 13.2 — COMPLETE):**
- Verify: `pyproject.toml` declares a `[tool.pyright]` block with
  `typeCheckingMode = "strict"`, `pythonVersion = "3.11"`,
  `include = ["src/backend"]`, and excludes (at minimum) `**/__pycache__`,
  `**/node_modules/**`, and `src/backend/vlm_feedback_loop/migrations/**`.
  The version matches the package's minimum supported interpreter.
- Verify: `pyright>=1.1.408` is declared in `[dependency-groups.dev]`
  and resolves to a compatible release in `uv.lock`; no second Python type
  checker is configured.
- Verify: `uv run pyright src/backend/` exits 0 with zero errors and zero
  warnings. Every suppression includes its bracketed diagnostic code and a
  defensible local reason; the checklist does not pin a historical suppression
  count that would become false as code evolves.
- Verify: `.github/workflows/ci.yml` declares a `backend-typecheck` job
  that runs `uv run pyright src/backend/` as a required gate (no
  `continue-on-error: true`), with `timeout-minutes: 10`, in parallel with the
  other CI jobs.

---

## Appendix C: Minimal Configuration (Interactive Labeling Only)

### C.1 Fastest Path: NVIDIA-hosted NIM APIs

1. Set a workspace root and get an API key:

```bash
export WORKSPACE_ROOT=”/path/to/workspace”
export NVIDIA_API_KEY=”nvapi-...”
```

1. Defaults:

- Hosted base URL: `https://integrate.api.nvidia.com/v1`
- Model catalog seeded with default entries.
- `teacher_model_config_id` seeded to the effective `DEFAULT_TEACHER_MODEL` — `stepfun-ai/step-3.7-flash` by default.
- Ingest returns 202 after skeleton persistence; CPU-only pHash and optional CLIP-style embeddings then compute in background. The review selector uses deterministic no-signal fallback immediately, pHash diversity as hashes arrive, and CLIP diversity once embedding coverage reaches the configured threshold.
- Point the system at image directories via the in-app file browser or direct path entry. Images are stored by reference (not copied). Begin labeling; ICL becomes effective as Verified accumulates.
- SMEs can Retry to compare Teachers/Guidance versions and re-run proposals on same example.
- SMEs can Skip to omit images so they do not reappear.

When the Scale-Up Readiness Gate passes and the SME approves (§7.3), run Batch Labeling to generate Auto-Labeled fully synthetic labels, often immediately before Student training via Cosmos-RL / TAO VLM.

---

## Appendix D: Default Prompts

### D.1 Teacher Interactive Proposal Prompt

This prompt MUST:

- Prepend a compact prompt-visible contract derived from SchemaCore. It MUST list every output field with required/optional status, type, categorical values, and applicable bounds so prompt-only fallback is self-contained.
- Omit verbatim Guidance Description and Rules prose. Guidance remains versioned task provenance and the source of SchemaCore; the compact field contract and demonstrations carry the model-facing task definition.
- Provide ICL examples labeled `E01..` without revealing `example_key`.
- Render only Core fields in each ICL label and use bookend order after pruning: strongest match first, second-strongest last.
- **When ICL examples are present, include the concise §6.2 ICL-use directive in two places:** a header scoping every label to its paired example image and a pre-query instruction to apply demonstrated field meanings and decision boundaries while deriving every value independently from the query image. Never instruct the model to prefer or copy a retrieved example's label based on visual proximity. Cold-start prompts omit both.
- Require output to be JSON that matches the SchemaCore-derived JSON schema.
- When the active Guidance enables `rationale_note`, request it last after label fields and keep it to one or two short image-specific sentences. When disabled, omit all rationale instructions.
- When enabled, describe the subject the active task asks about. Distinguish a physical carrier from content shown on it only when that distinction matters to the task; do not impose a universal “physical foreground item” assumption on OCR, scene, or anomaly tasks.
- When enabled, require honesty when evidence is weak or conflicts with a generated field value; the rationale MUST describe the pixels rather than invent support.

Template (illustrative; the backend prepends `{COMPACT_SCHEMA_CONTRACT}` and inserts image/ICL sentinels):

```text
SYSTEM:
You are a vision labeling assistant. Output valid JSON only.

USER:
{COMPACT_SCHEMA_CONTRACT}

ICL Examples (each block is the IMAGE for that position followed by its JSON label):
Each JSON label applies only to its paired example image. Use the examples to infer field meanings and decision boundaries. Determine the QUERY values independently from the QUERY image.

{ICL_EXAMPLES_RENDERED}

Now label the QUERY image below.
Apply the demonstrated field meanings and decision boundaries, then determine every value independently from the QUERY image.
{QUERY_IMAGE}

Return one JSON object matching the label schema above.
{% if RATIONALE_NOTE_ENABLED %}
Add rationale_note last. Keep it to one or two short, image-specific sentences.
Describe the subject the task asks about. Distinguish a physical carrier from content shown on it only when that distinction matters to the task.
State uncertainty when needed. Do not invent evidence or merely repeat the label.
{% endif %}

Return JSON only.
```

The ICL block and both ICL-use directives MUST be omitted at cold start. The compact schema contract is present in both structured and prompt-only modes. Reference rendering: `src/backend/vlm_feedback_loop/prompts/teacher_interactive_proposal.txt`.

### D.2 Guidance Rewrite Prompt (removed)

> **Removed 2026-07-21:** the D.2 prompt was removed with the AI Guidance Rewrite feature (§6.4); D.1 and D.3 numbering is unchanged.

### D.3 Rationale Regeneration Prompt

This prompt is available only when the active Guidance enables rationale notes. It is used when the SME edits a label and requests AI-generated rationale (§6.3.2); a disabled Guidance returns 409 without dispatch. The regenerated `rationale_note` is written back as auxiliary Label data and may flow to Student training when the export mode includes it (§9.3); production Teacher ICL excludes it (§6.2). It MUST read as direct observational prose describing what is visible, not post-hoc validation of a field value.

This prompt MUST:

- Provide the image, the active Guidance Description and Rules, and a SchemaCore summary.
- Exclude every original, corrected, or reviewed field value from both the request contract and the writer prompt.
- Ask the Teacher to produce only a new `rationale_note` string in the direct observational voice the D.1 pipeline produces.
- Require an independent inspection of the subject the task asks about. Distinguish a physical carrier from content shown on it only when that distinction matters to the active task.
- If the image is ambiguous, describe what is visible and state the uncertainty rather than inventing support. This makes uncertainty loud to the SME before approval instead of silently poisoning ICL and exports.
- Be domain-agnostic: no task-specific vocabulary, prohibited-word list, fixed visual-feature palette, or examples in the prompt body. Natural domain vocabulary, quantities, identities, comparisons, and locations are allowed when visibly supported and relevant. All task policy enters via the rendered variables.
- Produce output matching the same `rationale_note` contract (30–60 words target, 80-word hard cap, observable evidence only, no speculation).

This prompt SHOULD additionally provide task-awareness signals so the rationale is discriminator-focused rather than generic vision commentary:

- The active Guidance Description and Rules, so the rationale reflects the labeling task's policy.
- A compact Core-field schema summary enumerating allowed values for every categorical Core field (enum, enum_set, boolean) and range bounds for integer fields. This gives the model the field semantics without requiring it to echo or defend the values.

These additions are compatible with the MUST clauses above: they provide task context without disclosing either the Teacher's original proposal or the SME's correction. The canonical implementation lives at `services.prompt_service.render_rationale_regeneration_prompt`.

Template (illustrative; placeholders are provided by the backend; Task/Rules/schema blocks are omitted when their source content is empty):

```text
SYSTEM:
You write rationale_note text for vision-labeling records. Inspect the image independently and describe concrete, image-specific evidence relevant to the active task. Describe the subject the task asks about. Distinguish a physical carrier from content shown on it only when that distinction matters to the task. Do not infer observations from a proposed or reviewed label. If the subject or relevant visual evidence is ambiguous, state the uncertainty. Return only the new rationale text.

USER:
Task:
{GUIDANCE_DESCRIPTION}

Rules:
{GUIDANCE_RULES}

Label schema (for context):
  {CORE_FIELD_1_NAME}: {type} — {allowed values | range}
  {CORE_FIELD_2_NAME}: {type} — {allowed values | range}
  ...

Write one new rationale_note from your own inspection of the image. Focus on visible properties of the subject the task asks about that help distinguish the task's possible outcomes. Use natural vocabulary appropriate to this domain. Quantities, identities, comparisons, and locations may be described when they are visibly supported and useful for this task.
Do not copy or paraphrase a previous rationale. Do not merely name an outcome, recite a generic visual-feature checklist, or invent supporting details. Use only what is visible; do not speculate about causes or events outside the image. State uncertainty when the subject or relevant evidence is ambiguous.
Target 30–60 words. Never exceed 80 words.
Prefer 2–4 short sentences or one compact paragraph.

Return only the rationale text, no JSON.
```

The regenerated rationale is presented to the SME for review. If the SME approves (via **Approve AI Rationale** or further editing), it is stored on the Label with `rationale_source="teacher_regenerated_approved"` (or `"sme_edited"` if further modified). The regeneration invocation is persisted as an Operation Record with `purpose="rationale_regeneration"`.

---

## Appendix E: Student Precision/Quantization Evaluation Suite

Purpose: Provide a minimal, repeatable suite to compare Student variants across quality, latency, and robustness.

**Evaluation source distinction:** Quality comparison (Experiment 1) can use either TAO-backed evaluation (`evaluation_source=tao`, available immediately after training/quantization) or NIM-backed evaluation (`evaluation_source=nim`, available after deployment). TAO-backed quality results use the system's canonical Core-field evaluator for authoritative scoring (§9.7.6). Experiments 2–5 (latency, concurrency, efficiency, robustness) require NIM deployment and are NIM-sourced only. The Teacher appears as an accuracy baseline in quality comparisons; Teacher latency is not included because the Teacher may be hosted externally.

Definitions:

- "Variant" = a deployed Student endpoint + its selected NIM profile OR custom quantized checkpoint (backend/precision/TP/quantization lane).
- "Min latency" is measured at concurrency=1 (single stream).
- "Max throughput" is measured at the maximum concurrency that saturates throughput.

(These terms match NVIDIA NIM benchmarking conventions.)

Required provenance captured for every run:

- `student_model_config_id` (variant identity)
- `nim_model_profile_requested` (nullable)
- `nim_model_profile_selected` + `nim_profile_metadata` (precision/backend/TP)
- `quantization_method`: TAO quantization method used (e.g., `"FP8_DYNAMIC"`, `"W4A16"`, or `"none"` for full-precision)
- GPU model/count
- dataset manifest ref + dataset SHA-256
- code commit SHA
- decoding parameters (MUST be identical across variants)

NOTE: NIM logs profile selection and metadata at startup; record it.

Required driver: pinned `aiperf==0.10.0`, installed by the default `uv`
operational group in source and container modes. Its upstream dependency caps
are overridden to the audited aiohttp 3.14.3+ and Pillow 12.3+ releases; the
real raw-payload integration test is the compatibility gate. The standalone
backend wheel intentionally omits the external load-driver executable. The
adapter uses AIPerf `raw_payload` JSONL so the complete multimodal OpenAI
request body is replayed verbatim. There is no synthetic-text or `httpx`
fallback.

Additionally supported instrumentation: sample NIM Prometheus metrics at `/metrics` before and after each cell. Counter series are aggregated and stored as deltas; cache gauges use the observed value. Unavailable metrics remain `null`. TTFT/ITL remain `null` for this non-streaming production contract.

### E.1 Experiment 1 - Quality Delta vs 16-bit Baseline

Goal: quantify accuracy/quality loss vs baseline.

Baseline: Use the highest-precision deployed profile available for the same Student weights (BF16 for Cosmos Reason2). Precision availability is model/hardware dependent.

Slice (internal default; adjust): `heldout_200.jsonl` (200 examples; balanced).

Procedure:

1. Evaluate baseline variant (full-precision BF16) on heldout_200.
2. Evaluate each TAO-quantized variant: FP8_DYNAMIC, and any additional selected schemes (W8A8, W8A16, W4A16).
3. Report delta-metric relative to baseline. Record `quantization_method` per variant.

Output: `variant_id`, `profile_selected`, `quantization_method`, `primary_metric`, `delta_vs_baseline`, plus per-core-field match rates and per-value precision/recall/F1 for categorical fields (per Appendix A.2)

Decision heuristic (internal; not NVIDIA-specified): If |delta| <= 0.5-1.0 pts, proceed to performance tests; else investigate or stop.

### E.2 Experiment 2 - Min Latency + Peak Memory

Goal: measure raw speed/memory gain.

Method:

- Min latency is captured at concurrency=1 (single stream) per NVIDIA benchmarking convention.
- Collect P50/P90/P99 end-to-end latency at the client boundary using the real image workload. Report integer milliseconds in the UI so sub-second results are not collapsed by decimal-second rounding.

Output: `variant_id`, `p50_e2e_ms`, `p90_e2e_ms`, `p99_e2e_ms`, `ttft_p50_ms` (if collected), `itl_p50_ms` (if collected), `peak_gpu_mem_MB`, `model_resident_MB`

### E.3 Experiment 3 - Concurrency Scaling and Reliability

Goal: quantify behavior under load.

Method:

- Drive the same deterministic, real-image AIPerf workload at every configured concurrency.
- Record achieved request throughput, latency distribution, exact attempted/successful/failed counts, and failure percentage.
- A cell passes only when every selected image completes exactly once with no transport, HTTP, or timeout failure.

Output: `variant_id`, `concurrency`, `rps`, `p50_e2e_ms`, `p90_e2e_ms`, `p99_e2e_ms`, `attempted_requests`, `successful_requests`, `failed_requests`, `failure_pct`

No SLO-miss metric is reported in v1 because the project defines no latency SLO target.

NOTE: NVIDIA defines "Max Throughput" as throughput at the maximum concurrency that saturates throughput.

### E.4 Experiment 4 - Efficiency Normalization

Goal: compare work per unit resource.

Default normalization: `throughput_per_GB = (req/s) / peak_mem_GB`

Optional: Record GPU cache usage percentage if available (`gpu_cache_usage_perc`).

### E.5 Experiment 5 - Robustness (Long Input / Long Output)

Goal: find where low-precision variants fail (timeouts/OOM/truncation).

Method:

- Use two micro-suites (internal default: 30 each): Long-Input, Long-Output.
- Enforce timeouts; count error rate and truncation.
- Serving requests are uncapped: omit both output-limit spellings and let EOS/server policy terminate generation. Timeouts and malformed outputs remain explicit failures/evidence.

Output: `variant_id`, `suite`, `error_pct`, `trunc_pct`, `avg_len_tokens`, `quality_proxy_metric`

### E.6 Apples-to-Apples Rules (mandatory)

- Same prompts and decoding params across variants.
- Same input distributions (image sizes, token lengths).
- Same `visual_budget_preset_key` across variants; effective `mm_processor_kwargs` recorded per invocation (§6.9.5).
- KV cache reuse disabled (`NIM_ENABLE_KV_CACHE_REUSE=0`) across variants and external endpoints confirmed equivalent.
- Same pinned/selected NIM profile metadata recorded for reproducibility.
- Same `quantization_method` compared across variants of the same base model for meaningful comparison.

---

## End of Engineering Specification (v1.10.2)

*v1.10.2 (2026-08-04): Student serving benchmarks now replay deterministic real Test Pool images through the active Guidance-derived production prompt using pinned AIPerf, uncapped output, disabled KV reuse, strict all-concurrency reliability gating, durable workload provenance, RPS/failure metrics, and honest nulls for unavailable telemetry. External endpoints run the same benchmark; quality promotion remains independent.*

*v1.10.1 (2026-08-04): Models & Results remains a single project-wide comparison while grouping retained Students by immutable Training Suite lineage. Run headings expose provenance, mixed evaluation contexts are labeled directional, incompatible Teacher deltas fail closed, and chart labels remain unique across repeated candidates.*

*v1.10.0 (2026-05-19): F49 amendment — one-NIM-per-GPU invariant codified in §1.5 (replaces the prior "same-GPU co-location is never automatic" prose with explicit replace semantics); §9.5.2 step 0 + step 9 added (Student lifecycle acquires the GPU before docker_run and auto-restores the displaced Teacher after stop); §13.15 LocalNimDeployment gains `displaced_by_deployment_id` and `displaced_at` diagnostic fields. Implementation: `services/local_nim_service.acquire_gpu()`. Empirical motivation: README "One-NIM-per-GPU policy".*

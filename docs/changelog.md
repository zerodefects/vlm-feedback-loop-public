# Changelog

- 2026-08-07: The independent public snapshot temporarily omits GitHub Actions
  CI while the canonical source moves to NVRetail GitLab. Source CI remains
  active during cutover, and the exporter now prevents the removed public
  workflow from returning on the next snapshot refresh.

- 2026-08-07: README clone commands now resolve to the canonical repository for
  each distribution (NVRetail GitLab in source and the anonymous mirror in the
  public export). The data-handling, cost, and limitations section has also
  been condensed while retaining its operational warnings.

- 2026-08-07: The README's opening project-loop image is now cropped to
  the focused first-run card, renamed `project-loop.png`, and the superseded
  first-run guidance screenshot has been removed.

- 2026-08-07: NVIDIA API-key setup now recognizes a hosted `429 Too Many
  Requests` response as proof that the bearer cleared authentication instead
  of falsely rejecting a fresh key. Invalid keys still fail on `401`/`403`;
  ordinary inference continues to report and retry rate limiting separately.

- 2026-08-07: The README now opens with the first-run Projects screen and its
  Propose, Review, Refine, Optimize, and Deploy loop, giving new users the
  Blueprint's workflow before setup details.

- 2026-08-07: The curated independent public export now omits the
  NVIDIA-organization-only SonarQube delegate. GitHub resolves a private
  reusable workflow before evaluating its job-level repository-owner guard,
  which otherwise made every independent-mirror push appear failed with zero
  jobs. The public CI's locked `pip-audit` and `pnpm audit` backstop remains
  active; the source-only delegate is manually dispatchable until installed in
  an NVIDIA-owned repository with push and pull-request triggers enabled.

- 2026-08-07: Fixed Scale Up retaining a stale Unlabeled count when an SME
  leaves Image Ingestion after the first accepted batch while later batches
  continue in the background. Student Training remains available for setup,
  but its action now uses NVIDIA green only after the authoritative TAO and
  data preflight reports readiness. Fresh-install release checks now follow
  the current direct-open bundled-sample flow and receive explicit runtime
  settings instead of depending on a developer's machine configuration.

- 2026-08-06: Repaired pre-change labeling projects that could still enter the
  workflow with MiniMax M3 selected. Schema revision `v1_0004` moves only a
  zero-label MiniMax project to its existing Step 3.7 Flash configuration;
  projects with labels and all catalog/invocation history stay untouched.
  Interactive Labeling now offers **Retry image** in place after a transient
  image-load failure and restores the proposal controls when the reload
  succeeds.

- 2026-08-06: Simplified Image Ingestion by removing the dedicated bundled-
  sample shortcut from the path row. In local-source mode the initial view now
  starts beside the sample, making its root directory selectable through the
  ordinary browser while retaining `/` as the browse boundary. Compose keeps
  its existing `/data/images` boundary and stored paths.

- 2026-08-06: Simplified hosted onboarding around a commercially usable fresh
  catalog. New projects now seed eight models whose published model terms
  permit commercial use, select Step 3.7 Flash by default, and no longer stop
  on a standalone non-commercial-model warning; MiniMax M3 is excluded from
  new seeds while historical and operator-created records remain inspectable.
  NIM endpoint responses now expose a derived usage policy. Batch Labeling
  shows the selected endpoint's use class and requires a final confirmation
  before a seeded NVIDIA API Catalog endpoint is used, explaining that trial
  credits do not grant production rights and routing production users to a
  subscribed or appropriately entitled endpoint.

- 2026-08-06: Image Ingestion now exposes an explicit **Up one folder**
  control beside the filesystem path, identifies the configured `IMAGE_ROOT`
  boundary when no parent is available, and labels the page-level exit as
  **Previous screen** instead of the ambiguous **Back**.

- 2026-08-06: Fixed new-project setup appearing blank while a saved NVIDIA
  credential check was slow. Automatic setup transitions now show a clear
  progress card, and the best-effort saved-key probe no longer blocks
  onboarding beyond eight seconds.

<!-- public-export:exclude-autorun:start -->
- 2026-08-06: AutoRun can now bound a warmup or resume phase by successful
  saves with `--max-examples`, leaving remaining examples Unlabeled and still
  running the final evaluation at the phase boundary. It can also update the
  project's pool fraction before a phase with `--test-pool-fraction`, enabling
  fixed held-out sets across controlled resumes, and forward an explicit
  evaluation concurrency for long hosted checkpoints. Its report now computes
  the Test Pool target from that current value instead of assuming 40 percent.
<!-- public-export:exclude-autorun:end -->

- 2026-08-05: Credential-free run snapshots now use version 2 and freeze the
  sampling preset values, effective visual processor arguments, token-budget
  inputs, ICL controls, and image downscale edge used by Evaluation and Batch
  inference. Delayed dispatch and Batch restart/Resume no longer re-read
  those semantic values from process configuration. Existing resumable
  version-1 Batch snapshots upgrade once at startup. Credentials, filesystem
  authorization, timeouts, retries, scheduling, and emergency capability
  kill switches intentionally remain live operational policy. Startup now
  validates sampling and token-budget ranges, and malformed persisted enum or
  numeric snapshot values fail recovery closed before dispatch.

- 2026-08-05: `ICL_MAX_EXAMPLES` now rejects zero and negative configuration
  values at startup. A positive integer sets the deployment-wide cap; the
  existing per-run `icl_mode="disabled"` remains the supported zero-shot path.

- 2026-08-05: Batch Labeling run limits now select examples in ingestion
  order, with the example key as a deterministic tie-breaker, as documented.

- 2026-08-05: Non-positive Batch Labeling run limits and circuit-breaker
  thresholds now fail configuration validation at startup instead of creating
  a run that cannot execute.

- 2026-08-05: Documentation now distinguishes immediate Test Pool assignment
  from CLIP/pHash diversity rebalancing and explains the manual versus
  automatic evaluation triggers controlled by Auto-Evaluate. Runtime behavior
  is unchanged.

- 2026-08-05: Public schema revision `v1_0003` adds a credential-free,
  immutable model/endpoint runtime snapshot to new Evaluation and Batch Run
  Records. Delayed dispatch, restart recovery, and explicit Batch Resume now
  retain their model and endpoint inputs while later catalog updates,
  capability re-probes, and local-NIM rebinding affect future runs only. The
  migration freezes valid non-terminal pre-v1_0003 NIM runs at the upgrade
  boundary, including paused Batch runs; startup fails closed if a residual
  legacy row cannot be materialized safely. The version-2 entry above extends
  this boundary to semantic process configuration.

- 2026-08-05: Batch Labeling restart recovery now reuses each unfinished
  item's pending invocation identity and accepts a successful item as complete
  only when its exact Label lineage and Example state agree. Terminal
  Operation outcome, Auto-Labeled Label, and Example transition commit
  atomically; duplicate or foreign lineage fails closed. Cancellation marks
  only outcomes committed after the durable request as ignored, preserving
  earlier evidence, and timestamp-less `canceling` rows recover as interrupted
  failures instead of invented user cancellations. Frozen input totals,
  circuit-breaker thresholds, consecutive failure streaks, and a durable
  tripped latch now remain authoritative across restart and concurrent request
  drain; explicit Resume safely joins the old task, resets breaker state, and
  cannot strand a queued run. Evaluation and Batch cancellation exclude
  ignored outcomes from partial counters, interrupt retry waits promptly, and
  retain the user's terminal cause when a late capability rejection crosses
  the durable cancellation boundary. Recovery isolates malformed canceled-run
  state, and terminal events are emitted only by the transition that wins.

- 2026-08-05: Opening a nonempty SQLite schema without one canonical,
  nonempty Alembic revision now creates a validated recovery backup and stops
  before revision discovery or migration. Empty databases and a sole canonical
  empty version table still initialize normally; existing public revisions
  retain their supported upgrade behavior.

- 2026-08-05: Student dataset staging now accepts only completed export pairs,
  opens the archive and `annotations.json` sidecar beneath the owning
  project's export directory, and uses those same descriptors for validation
  and upload. The recorded archive checksum and type-sensitive JSON equality
  are verified before any S3 request, and the exact bytes sent are rehashed so
  in-place mutation cannot inherit stale hash metadata; pair-wide idempotency
  repairs a missing member and persists lineage only after both objects
  succeed. A durable preparing suite links each frozen export before upload;
  an identical pre-chain transfer retry reuses those export IDs and repairs
  only incomplete objects, while an integrity failure requires fresh exports.
  Each export row and suite link now commit together, suite responses retain
  the selected LoRA mode for exact request replay, and Training Jobs exposes
  the safe same-key dataset-upload retry when the backend-derived
  `setup_retryable` flag allows it. Existing suites are
  upgraded through migration `v1_0002`, deriving their mode from frozen train
  job lineage when available.
  Ready-base failures return `409 tao_dataset_upload_failed`, while a failure
  after provisional base setup makes that suite `failed` with `setup_error_ref`.
  Self-service base provisioning also binds each staged checkpoint's hash,
  size decision, and upload to one descriptor beneath its disposable stage,
  with the same exact-stream hash guard.

- 2026-08-05: Bound every persisted-image read, TAO re-scoring archive read,
  and dataset-export download to the regular-file descriptor authorized under
  the current root policy. Teacher and rationale calls, embeddings, pHash,
  benchmark workloads, image serving, archive assembly, re-scoring, and HTTP
  downloads no longer authorize one pathname and reopen it later;
  policy-denied images fail before any outbound dispatch, partial export
  artifacts are cleaned up, and range/download failures close their owned
  descriptors.

- 2026-08-05: Local NeMo Retriever VL eligibility now starts at 24 GB,
  matching the smallest GPU SKUs in NVIDIA's pinned 2.0.0 support matrix;
  automatic recommendations also require an exact supported GPU-name match,
  and deployment state carrying the former 10 GB floor self-heals on startup.
  Setup guidance now distinguishes free Developer Program development access
  from production terms, Compose no longer advertises or forwards an unusable
  local-NIM NGC credential, and multi-GPU docs match lowest-compatible-device
  placement.

- 2026-08-05: Hosted setup now links the third-party MiniMax M3 default to its
  non-commercial model-card terms at the point of selection. The API
  walkthrough creates and activates its Guidance, uses the shipped `category`
  schema, and truthfully starts with no ICL examples.

- 2026-08-05: The backend test environment now uses Starlette's supported
  `httpx2` TestClient transport instead of its deprecated `httpx` fallback;
  the lockfile and third-party license inventory include the new dev-only
  dependency.

- 2026-08-05: SQLite connection setup now closes rejected database handles
  before surfacing corruption or foreign-key enforcement errors, preventing a
  failed open from leaking a descriptor until garbage collection.

- 2026-08-05: Backend subprocesses now share cancellation-safe lifecycle
  handling. Docker probes, LoRA merge checks and packaging, TAO base pulls,
  and AIPerf are killed and reaped on timeout, shutdown cancellation, or pipe
  failure before the original outcome propagates.

- 2026-08-05: Deployment-scoped SQLite now has one engine per canonical
  workspace and deterministic server/CLI teardown. Repeated environment,
  embedding, and TAO checks no longer accumulate database descriptors, and
  startup failures still release every opened resource.

- 2026-08-05: Navigation, guidance editors, evaluation details, rationale and
  action-request text, model capability pills, and archived-project status now
  consistently use KUI text and status primitives.

- 2026-08-05: Startup now rejects zero concurrency, batch size, request
  attempts, and timeout settings, plus empty or non-positive Student benchmark
  concurrency lists. Invalid operator configuration fails before it can crash
  a worker or leave a semaphore permanently blocked.

- 2026-08-05: Compare and labeling-result quality warnings now consume each
  project's backend-owned per-field and per-value F1 thresholds. Precision and
  recall remain diagnostic instead of being presented as gate failures.

- 2026-08-05: Generic project updates can no longer stamp or clear onboarding
  completion. The dedicated idempotent transition remains the only writer, so
  every first completion retains its required audit context.

- 2026-08-05: Documented the image, label, prompt, and training-data boundaries
  for hosted inference, embeddings, and TAO, including potential compute,
  storage, egress, and API costs and the reference stack's operating limits.
  The Compose API walkthrough now uses one coherent endpoint, image mount, and
  synchronously resolved embedding state. Local-NIM guidance now links its
  pinned support matrices and distinguishes NVIDIA-validated configurations
  from the Blueprint's measured Cosmos 3 Super policy.

- 2026-08-05: Operator and API examples now match the runtime's nominal GPU
  floors, single-GPU Teacher displacement, fresh-project embedding state, and
  initial Test Pool routing. The Compose environment template now exposes its
  supported non-secret overrides, including `IMAGE_ROOT`, and the documented
  Playwright install includes browser system dependencies. UI and generated
  deployment handoffs now describe GPU requirements as inclusive minima.

- 2026-08-05: The optional local-NIM pre-pull now follows the application's
  hardware-gated Teacher quality order: Omni on supported 80 GB / cc 9.0+
  GPUs, CR3-Nano on the 56 GB tier, and CR2-2B on the 36 GB tier. Smaller GPUs
  pre-pull only the eligible embedding NIM. The script pulls only the selected
  Teacher plus embeddings and runs the pinned Nano profile pre-cache only when
  CR3-Nano is selected. Delivery tests now guard setup image pins and generated
  prerequisite floors against their backend and shared-shell owners.

- 2026-08-05: Local-NIM setup now enforces its documented Docker Engine
  29.4.0+ and NVIDIA Container Toolkit 1.19.0+ floors. The strict prerequisite
  script stops with upgrade guidance, while the general developer bootstrap
  completes hosted setup and clearly marks local NIM unavailable when an
  operator-managed runtime is too old, unreadable, or cannot pass a container
  GPU test.

- 2026-08-05: TAO base-experiment provisioning now ships its subprocess
  helper and isolated requirements inside the backend package. Source,
  container, public-export, and installed-wheel executions now resolve one
  runtime layout instead of the wheel exposing a command it could not run.
  CI now validates the actual default public snapshot, builds both declared
  Python distributions, smokes the wheel rebuilt from the source distribution,
  and runs the browser-level first-time labeling workflow.

- 2026-08-05: Docker Compose now forwards the documented
  `TAO_QUANTIZATION_CALIBRATION_SAMPLES` override to the backend instead of
  silently leaving quantization at its built-in default.

- 2026-08-05: Graceful backend shutdown now disposes every cached project
  database engine before releasing its process locks. Test-only cache,
  throttle, selector, authentication, and polling controls no longer expand
  production service interfaces.

- 2026-08-05: Added a learner-oriented code trace for first-run setup and one
  interactive labeling turn, with explicit safe customization points. The
  README hardware table now states the seeded Teacher memory floors as
  inclusive, matching the backend's nominal-capacity tolerance.

- 2026-08-05: Simplified TAO diagnostics around their canonical records.
  Base-experiment provisioning now identifies missing workspace state in
  `deployment.db`, job responses and issue reports share one historical-log
  read repair, and model-name configuration imports the canonical constants.

- 2026-08-05: Made local-NIM runtime identity use the same validated custom
  environment as Docker launch. Reserved, malformed, and unsupported metadata
  entries can no longer prevent reuse of an otherwise exact running Teacher
  or hide it from the first-run recommendation.

- 2026-08-05: Hardened deployment-secret boundaries. UI-persisted keys now
  publish through a flushed, mode-0600 temporary file and atomic replace;
  permission or publication failures preserve the prior file. Opaque child
  credentials now use one exact-value redactor, including LoRA merge output,
  before diagnostics can be returned or logged.

- 2026-08-05: Aligned local-NIM setup with the CUDA 13 / R580 driver floor.
  Both setup entry points now enforce driver 580.65.06+, generated deployment
  handoffs report the same requirement, and the first-run docs now use the
  correct clone directory, Python range, pinned pnpm version, Compose env-file
  location, and current bundled-sample UI flow.

- 2026-08-05: Kept the local-source launcher and its Vite proxy on one
  backend address for configured or command-line port overrides. The launcher
  now fails as soon as either child exits, Vite refuses to hide a busy port by
  moving silently, CI runs the safe local integration suite serially, and the
  Compose smoke verifies the UI and API proxy as well as backend health.

- 2026-08-05: Consolidated the unreleased Student Training Suite lineage
  schema into the public `v1_0001` Alembic baseline. Fresh v1 projects now
  receive the complete StudentModel schema from the single initial revision;
  the transitional `v1_0002` candidate migration was removed before release.

- 2026-08-05: Restricted new Student Training runs to the qualified LoRA
  matrix. The Training base selector no longer exposes Cosmos 3 Super, and the
  Full-weight toggle is withheld until that method is qualified across the
  UI-supported Student bases. Backend/API compatibility and historical Super
  results remain intact.

- 2026-08-05: Simplified project navigation by removing the persistent
  Models & Results header link. Trained Student comparison and deployment
  remain reachable from the project Overview and contextual workflow actions,
  without advertising an unavailable destination before a Student exists.

- 2026-08-05: Kept shared local Teachers consistent across restore cycles.
  When an exact Teacher runtime returns under a new deployment identity, former
  consumers that still select its disabled managed attachment now reconnect to
  the healthy resident. Interactive proposals also fail closed on disabled or
  hard-unhealthy endpoints instead of following a reused host port behind the
  dropdown's unavailable state.

- 2026-08-05: Made environment assessment deployment-scoped in execution as
  well as API shape. The app now warms one shared assessment without blocking
  the Project List, ordinary project screens no longer wait for hardware data
  they do not use, and the backend reuses its expensive Docker/toolkit/GPU
  snapshot across projects while recomposing current credentials and NIM
  residents. An explicit `refresh_hardware=true` read handles rare host changes;
  launch preflight remains a fresh live safety check. Project-card lock probes
  now also seed the detail cache instead of forcing a duplicate tunneled read.

- 2026-08-05: Made every eligible **Request Production Deployment** action on
  Compare & Benchmark green. The UI no longer visually ranks production-ready
  Students by accuracy alone.

- 2026-08-05: Simplified the mature-project Overview navigation. The
  Interactive Loop card no longer repeats a Scale-Up shortcut, and its three
  peer destinations now use consistent green actions instead of promoting
  Models & Results solely because a Student exists.

- 2026-08-05: Kept the labeling header truthful when the selected Teacher is
  temporarily unavailable or no longer cataloged. The native selector now
  retains the backend-authoritative model as an explicit unavailable choice
  with a configuration recovery link instead of visually substituting the
  first reachable hosted model while project state still points elsewhere.

- 2026-08-05: Made completed-project onboarding deep links truthful. Copied
  setup, NGC, setup-summary, and model-default URLs now return to the
  authoritative Project Overview instead of reconstructing a stale hosted
  model summary from ephemeral defaults. Real in-progress onboarding keeps
  its explicit model-path state through confirmation and Back navigation.

- 2026-08-05: Made the public Blueprint's agent instructions self-contained.
  Public export now removes source-host sudo authority, private developer
  datasets, and companion-checkout assumptions from the otherwise identical
  `AGENTS.md` / `CLAUDE.md` twins while retaining portable product,
  architecture, test, UI, and contribution guidance. Both the exporter and
  offline public validator fail closed on malformed markers, twin drift, or
  leaked source-only instructions.

- 2026-08-05: Made current AIPerf evidence a backend-authoritative production
  deployment gate. Upgraded workspaces may retain earlier synthetic `httpx`
  serving sweeps for historical comparison, but those runs no longer unlock a
  deployment handoff or portable NIM bundle. Models & Results now identifies
  the affected Student and offers **Revalidate with AIPerf**; successful real
  Test Pool revalidation restores deployment eligibility.

- 2026-08-05: Reduced the published backend container's operating-system
  attack surface by moving it from Debian slim to the official Python 3.12
  Alpine 3.24 image. Native compilation for AIPerf's locked psutil and
  sentencepiece dependencies remains confined to the discarded builder stage;
  the non-root runtime adds only `libstdc++` and retains its existing health,
  signal, configuration, and uid-1001 contracts.

- 2026-08-05: Corrected custom Cosmos 3 Nano Student deployment so the
  shared-image size selector remains `nano` while the base model's pinned
  profile is omitted. NIM can now select a profile compatible with the
  mounted Student checkpoint instead of exiting while trying to combine the
  checkpoint with bundled base weights. Preflight, live launch, generated
  handoffs, and portable bundle commands now use the same effective selectors;
  Teacher profile pinning remains unchanged.

- 2026-08-05: Kept all four workflow counts visible on every populated
  project card, including zero Auto-Labeled and Omitted values. Stable card
  metrics make project state directly comparable and satisfy the Project List
  screen contract without making counts appear only after a transition.

- 2026-08-05: Made the public candidate's full-tree secret scan clean by
  shortening non-secret example/idempotency fixtures and narrowly marking the
  two deliberately key-shaped logging-redaction probes. This preserves real
  credential detection without a broad test-directory exemption.

- 2026-08-05: Kept the pinned AIPerf 0.10.0 serving contract while removing
  its stale vulnerable dependency caps. Source and container installs now use
  a default operational dependency group with verified overrides for aiohttp
  3.14.3+ and Pillow 12.3+, while the standalone backend wheel remains a
  clean, resolvable package without an external load-driver executable. The
  dependency audit consumes uv's complete pinned set directly instead of
  asking pip to reject the intentional override before scanning it.

- 2026-08-05: Kept the default public candidate focused on the Blueprint by
  excluding the private live-release run ledger and its one-off project-clone
  helper. The public README now leads readers only to product, deployment,
  API, contributor, and normative engineering documentation; maintainers keep
  the acceptance materials in the source repository where live identities and
  evidence belong.

- 2026-08-05: Made post-onboarding self-hosted embedding configuration real
  and durable. The connection test now exercises the configured NeMo Retriever
  model rather than accepting a generic model-list response, requires one
  finite 2,048-dimensional vector, and Save repeats that proof before switching
  the deployment-scoped provider and re-resolving every project. A different
  external URL cannot silently orphan a running Blueprint-managed embedding
  container. Local deployment and restart recovery now retain its exact GPU
  assignment, and the NIM Configuration summary names the one-NIM-per-GPU
  invariant instead of claiming embeddings never compete with a local Teacher.
  When placement policy reserves a physically capable free GPU for that
  Teacher, the embedding panel now names the reservation instead of falsely
  reporting that no compatible GPU is free.

- 2026-08-05: Made post-onboarding self-hosted Teacher configuration durable.
  A successful URL test now exposes only exact vision Teachers present in the
  project catalog; Save re-verifies that model, idempotently creates or reuses
  the credential-free endpoint, selects it, refreshes capability controls, and
  keeps actionable failures on screen. This closes a misleading path where a
  green connection test was shown but Save discarded the endpoint.

- 2026-08-04: Made Auto-Evaluate operational rather than display-only. When
  enabled, the backend now evaluates the persisted first-pool,
  configuration-change, and ICL-growth triggers after label, project-setting,
  and Guidance mutations, durably queues one gate-basis evaluation, and
  suppresses duplicate automatic starts while one for the same configuration
  is active. A newly changed configuration still supersedes a stale run under
  the newest-config-wins contract. Teacher inference remains in the normal
  background task, so saving a label never waits on the model.

- 2026-08-04: Made Student serving-benchmark code provenance complete in
  local source mode by resolving the current Git revision automatically, and
  exposed `VLM_FEEDBACK_LOOP_BUILD_SHA` for packaged/container builds whose
  runtime image intentionally omits `.git`.

- 2026-08-04: Corrected queued local-NIM deploy responses so
  `preflight.resolved_port` reports the host port actually reserved by the
  allocator and agrees with `deployment.host_port`, including when the
  preferred port is occupied.

- 2026-08-04: Preserved NVIDIA's `AIPerf` product spelling when Models &
  Results renders the lowercase machine driver identifier in live Student
  workload provenance.

- 2026-08-04: Completed the closing keyboard and assistive-technology pass.
  The ingest path, Guidance free-form fields, Guidance/Teacher/training
  selectors, and compact benchmark choices now expose stable names or a
  visible NVIDIA-green focus indicator as applicable. Dismissing Create
  Project with Escape restores focus to its trigger in both empty and
  populated project-list states.

- 2026-08-04: Replaced the synthetic Student latency probe with a
  production-representative VLM serving comparison. Each variant now replays
  the same deterministic sample of up to 200 real frozen Test Pool images
  through its active Guidance-derived production prompt and Inference
  Contract, with no output-token cap and KV-cache reuse disabled. Pinned
  AIPerf records integer-millisecond p50/p90/p99 latency, achieved RPS, exact
  failure rates, token means when available, and honest optional NIM metrics;
  workload provenance and failed cells remain visible on Models & Results.
  Local and external endpoints share the same strict all-concurrency gate.

- 2026-08-04: Made the live integration server credential-safe by default.
  Disposable backends now inherit hosted NIM, NGC, Hugging Face, and TAO
  credentials only when the operator explicitly exports them for that test
  process; an absent variable can no longer fall through to the operator's
  canonical `.env`. The no-key pipeline path also asserts the backend is truly
  local-only before exercising failure behavior.

- 2026-08-04: Repaired local-NIM adoption after a backend restart. A surviving
  healthy Teacher now restores and re-enables its project endpoint without
  creating one duplicate per restart, while a surviving embedding NIM restores
  the embedding provider and pending-work sweep. Stopped or failed containers
  remain protected from late health responses that could otherwise resurrect
  them.

- 2026-08-04: Reworked Guidance starters into a focused seven-choice catalog:
  Blank plus six useful task patterns—Classification, Rock-paper-scissors,
  Multi-label classification, Presence and count, Packaging information audit,
  and a tested VisA industrial-anomaly inspection. Every non-blank choice now
  supplies practical Rules in addition to Description and Schema; generic
  values are conspicuous replacement markers, while dataset-referenced choices
  show their scope, license, and source context. VisA now keeps only its two
  reliable evaluated fields rather than adding a redundant free-text defect
  observation. Removed the placeholder attribute-extraction, inconsistent
  damage-severity, TrashNet recycling, and Freiburg grocery choices from the
  user-facing selector without changing internal validation datasets or
  bundling additional sample data.

- 2026-08-04: Made repeated Student training iterations readable without
  hiding history. Models & Results remains one project-wide comparison but now
  groups retained Students under newest-first Training Run provenance, shows
  dataset counts and selected models, qualifies duplicate chart identities by
  run time, and warns when cross-run evidence is directional. Student-vs-
  Teacher deltas now fail closed on Guidance, Test Pool, or output-contract
  mismatch. The consolidated public-v1 schema persists immutable Training Suite
  lineage for suite-created Students; every retained model remains benchmarkable
  and eligible for the sole portable NIM deployment bundle path.

- 2026-08-04: Kept public-release validation compatible with product upgrades.
  Public snapshots now accept only a contiguous, correctly linked `v1_*`
  Alembic lineage instead of incorrectly rejecting every migration after the
  v1 baseline.

- 2026-08-04: Refreshed the README's RPS Guidance image so the public front
  door shows the shipped starter's 15-image scope and CC BY 2.0 attribution.

- 2026-08-04: Kept project identity visible throughout the mature-project
  workflow. The persistent header now names the active project beside its
  Overview, Models & Results, and NIM Configuration destinations, so labeling
  counts, interrupted evaluations, and deployment actions cannot be mistaken
  for another project's state.

- 2026-08-04: Kept Models & Results unambiguous across populated projects.
  The screen now names its active project in both populated and empty states,
  and its per-field metric selector is programmatically associated with the
  visible label for keyboard and assistive-technology users.

- 2026-08-04: Added a state-aware **Models & Results** action beside Scale-Up
  in the Interactive Labeling footer. It appears only after the project has a
  Student model and returns directly to comparison, serving validation, and
  portable NIM deployment without routing through the Scale-Up workflow.

- 2026-08-04: Made trained Students durably reachable after project re-entry.
  Mature projects now open a state-aware overview with separate Interactive
  Loop, Models & Results, and Training Runs destinations, while active training
  or serving validation still resumes automatically. Production delivery is
  consolidated into one **portable NIM deployment bundle** on Models & Results;
  standalone TAO output/checkpoint downloads and their public endpoint were
  removed, while internal artifacts remain available to evaluation, recovery,
  NIM validation, and bundle construction.

- 2026-08-04: Made the SME rationale editor programmatically identifiable.
  Its visible label, state-specific review guidance, and regeneration failures
  now remain associated with the editor for assistive technology, and a
  regeneration failure is announced as an actionable alert. Retry settings now
  expose matching names for both selects and every segmented choice group.

- 2026-08-04: Made the primary labeling and evaluation-review controls
  screen-reader identifiable. Every schema-driven proposal editor now carries
  its visible field name programmatically, enum-set choices expose a named
  group, proposal failures announce through an alert live region, and the
  icon-only Evaluation Results close action has an explicit accessible name.

- 2026-08-04: Made the qualified Cosmos 3 Super Student matrix truthful.
  Super remains available for its validated full-weight training, packaging,
  local NIM evaluation, benchmark, and deployment-handoff path, while backend
  readiness and final suite materialization now reject its unqualified
  quantization path before export or TAO work. The Training screen shows a
  baseline-only notice in the Quantization section whenever Super is selected.

- 2026-08-04: Made Create Project failures actionable and safe to read. The
  dialog now extracts the backend's structured detail and presents it as an
  accessible alert instead of exposing HTTP status framing and raw JSON; its
  pending state blocks duplicate submission and unsafe dismissal until the
  non-cancelable request resolves. Project-card metric labels and values also
  wrap atomically instead of orphaning a number on the next line.

- 2026-08-04: Made Batch run recovery and export evidence truthful. Circuit
  breaker pauses now report the snapshotted consecutive-failure threshold
  instead of mislabeling aggregate lifetime errors, and the export lifecycle
  exposes its Cosmos-RL format, selected fields, label tier, example count,
  and manifest presence without revealing workspace paths. Export submission
  and background archive generation also have distinct pending/progress states
  instead of presenting unfinished work with completed-success styling.

- 2026-08-04: Completed the Batch Labeling pre-run contract in the product UI.
  SMEs can now cap an individual run, filter by a UTC ingestion-time range,
  select prompt-only generation, and disable ICL for a known ICL-negative
  Teacher without using raw REST calls;
  unsupported structured generation is pinned safely to prompt-only. The run
  monitor now displays the complete snapshotted generation contract and turns
  unexpected internal status codes into actionable operator guidance. Launch
  conflicts now retain the backend's current gate/configuration reason instead
  of collapsing every rejection into an unhelpful retry message; a gate-race
  rejection refreshes readiness, disables the stale launch action, and links
  directly back to the current Scale-Up criteria.

- 2026-08-04: Kept Student Training submission visibly in progress. After the
  SME confirms a suite, the locked confirmation now shows **Starting…** until
  creation succeeds; a failure closes it and returns to the actionable inline
  error instead of leaving an unexplained disabled form during a slow request.
  The same confirmation now uses the product-facing **FP8 Dynamic** label while
  retaining `FP8_DYNAMIC` in the request and durable records.

- 2026-08-04: Kept dense Compare charts complete and inspectable. A typical
  25-value field with Teacher plus three Students now fits the desktop canvas,
  dense value labels use readable words and angled spacing instead of colliding,
  and genuinely wider matrices provide visible, keyboard-focusable horizontal
  scroll guidance rather than silently clipping their final categories.

- 2026-08-04: Made the Training Jobs terminal record easier to read across
  Cosmos generations. Overall progress now uses compact Nano/Super labels and
  job headings display FP8 Dynamic as product copy while preserving the
  canonical TAO enum in APIs and durable records. Intentional evaluation
  omissions now say **Not Required** and direct the SME to local Student NIM
  validation instead of looking canceled; completed Students remain reachable
  in Compare after an independent model chain fails. Provider errors and halt
  reasons also shed transport escaping and internal job IDs, including
  SME-canceled predecessor outcomes that are no longer mislabeled as failures.

- 2026-08-04: Clarified the Compare screen's deployment evidence. Historical
  Teacher warnings now name the currently selected replacement and explain how
  to refresh the baseline; Student cards humanize FP8 naming and label the
  recorded GPU as benchmark hardware rather than an unstated requirement.

- 2026-08-04: Correctly attributed Scale-Up quality to the Teacher captured by
  the qualifying evaluation. The readiness card now distinguishes the current
  Teacher from the evaluated Teacher and warns when project settings changed,
  instead of placing a historical score beside a new model without context.

- 2026-08-04: Made evaluation restart recovery explicit on the Labeling
  screen. Interrupted runs now say that the backend restarted, remain clearly
  non-authoritative, and direct the SME to start a complete replacement run
  instead of collapsing the durable recovery reason into a generic failure.

- 2026-08-04: Preserved evaluation retry lineage across the initial sequential
  retry and every bounded rate-limit retry pass. Each new Operation Record now
  points to the exact preceding failed attempt, so operators can reconstruct a
  recovered evaluation without ambiguous duplicate records.

- 2026-08-04: Preserved one meaningful decimal on evaluation and Student
  comparison metrics. Close measured results such as 90.83% and 91.2% no
  longer collapse to the same whole-percent label, while exact whole values
  remain compact. Scale-Up readiness copy retains whole-percent formatting so
  it stays aligned with the backend-owned gate explanations.

- 2026-08-04: Kept the NIM Configuration hosted-key explanation aligned with
  the current seeded Teacher roster. The screen no longer advertises retired
  Mistral Large 3, Qwen, or Cosmos hosted choices after the live catalog moved
  to MiniMax, Mistral Medium, Step, and Nemotron.

- 2026-08-04: Completed the post-onboarding local embedding NIM lifecycle in
  NIM Configuration. Operators can now start NeMo Retriever VL from the normal
  screen, receive a named Keep-versus-replace decision when a GPU resident is
  present, follow durable startup, and stop a project-owned embedding NIM to
  return projects to the backend's hosted-or-pHash fallback. The first deploy
  click remains non-destructive, and an occupied but capable GPU is no longer
  misreported as below the model's hardware floor. A queued deployment disables
  its action immediately, before the first polling response; the same durable
  reconciliation guard now covers Teacher deployment, so rapid repeated clicks
  cannot submit duplicate NIM work. Once a replacement is durably accepted,
  the embedding panel also suppresses the displaced-resident and GPU-floor
  warnings from the pre-request environment snapshot while it reconciles.
  Local-NIM endpoint cleanup now skips archived projects before opening their
  databases, preventing retained pre-v1 workspaces from flooding a successful
  resident stop with irrelevant migration warnings.

- 2026-08-04: Limited automatic credential validation on the post-onboarding
  NIM Configuration screen to the services the SME has actually selected. An
  all-local setup no longer contacts the hosted NVIDIA API merely because an
  unused hosted key remains configured; hosted and local paths validate only
  their own required credentials. An explicit `EMBEDDING_PROVIDER=none` now
  remains a first-class pHash-diversity recommendation instead of being
  relabeled as hosted embeddings and prompting for an unwanted key. The
  relevant credential check is also single-dispatch under React Strict Mode.

- 2026-08-04: Restricted system-managed Teacher, embedding, and Student NIM
  port publication to host loopback. These unauthenticated local implementation
  services are no longer exposed on every network interface by the generated
  `docker run` command.

- 2026-08-04: Clarified the post-onboarding NIM Configuration screen so it
  truthfully tells operators to start each local service with its explicit
  deploy action; Save applies tested credentials and no longer appears to
  promise container startup.

- 2026-08-04: Made the public configuration examples a complete operator map.
  `config.yaml.example` now lists every supported non-secret tuning key at its
  safe shipped default, while `.env.example` remains the complete secret
  scope. A delivery drift guard prevents new Settings fields from becoming
  undiscoverable or credentials from moving into the wrong template.

- 2026-08-04: Clarified the network-accessible filesystem boundary without
  contradicting batch ingestion's partial-success API. Browse, scan, image
  serving, and path remapping return endpoint-level `403` when `IMAGE_ROOT` is
  absent on a non-loopback bind; ingestion still returns `202` and reports
  each rejected item as `path_not_allowed`, with the same actionable guidance
  and no persisted Example.

- 2026-08-04: Made the backend container forward shutdown signals directly to
  Uvicorn. Its JSON-form command still expands the runtime bind host and port,
  then uses `exec` so Uvicorn becomes PID 1 instead of remaining behind an
  intermediate shell during Compose stop/restart.

- 2026-08-04: Kept archived projects paused during backend startup and
  deployment-wide TAO metadata updates. The shared project scan now skips the
  archive sentinel before opening a project database, avoiding migrations,
  recovery writes, and legacy-database warnings for projects the SME has taken
  out of service.

- 2026-08-04: Corrected full-stack 2B/8B handoff validation for portable
  custom checkpoints. The closing harness now requires distinct per-base NIM
  images, GPU floors, and per-Student environments while accepting two
  intentionally unset model profiles; single-GPU tensor parallelism may
  correctly remain 1 for both sizes. Its optional RPS proof now selects the
  baseline Student deterministically, accepts normal Markdown-fenced model
  JSON, compares predictions with directory ground truth, and requires all six
  per-class responses to be schema-parseable. Accuracy remains a separate,
  visible held-out evaluation result rather than an infrastructure verdict.
  Re-execution now removes only an exact stopped validation container before
  launch and removes its own stopped container afterward, so deterministic
  handoff names are safely rerunnable without displacing a live NIM.
  Student Training now also describes an unavailable TAO base experiment as
  future first-use provisioning work instead of implying it is already
  provisioned. Completed Training Job outputs now provide project-scoped,
  containment-checked per-file downloads instead of inert path text.

- 2026-08-04: Removed TAO completion diagnostics from portable Student NIM
  deployment bundles. `status.json` and `microservices_log.txt` remain useful
  to Blueprint-side job finalization but are not NIM model inputs, so public
  bundles now omit that unnecessary infrastructure context while retaining
  every deployable checkpoint file and checksum.

- 2026-08-04: Made Action Request generation single-dispatch under React
  Strict Mode. Opening a production Student deployment handoff (or another
  audited Action Request) now issues exactly one backend POST and creates one
  audit event; the development UI previously replayed the mount effect and
  generated the same handoff twice from one visible click.

- 2026-08-04: Corrected Training Jobs recovery after Blueprint-local Student
  NIM serving validation failures. The failed evaluation card now identifies
  the local serving stage and opens Compare & Benchmark for inspection or
  retry instead of incorrectly generating a TAO issue report. Local NIM
  failures before quality is measured now preserve pending quality, and a
  retry repairs the narrow legacy failed-quality state so a clean deployment
  can still validate the Student. Compare also renders persisted failure
  tokens as SME-facing text instead of raw snake_case, and its single-GPU
  warning now refers to the actual local NIM resident class rather than
  misnaming a selected hosted Teacher as local.
- 2026-08-04: Allowed published single-model NIM images that omit the optional
  `list-model-profiles` utility to proceed from preflight to the authoritative
  bounded serve health and served-model checks. Live Reason2 8B 1.6.0 exposed
  that the former classification incorrectly treated a missing utility as an
  incompatible GPU. Shared Nano/Super images remain strict: their size-aware
  profile probe must succeed so a default sibling model cannot be validated.

- 2026-08-04: Corrected Cosmos 3 Student NIM deployment for the shared
  Nano/Super runtime image. Student preflight, system-managed launch,
  deployment handoff, and the downloadable portable bundle now inherit the
  base model's `NIM_MODEL_SIZE` and pinned profile while preserving the
  Student-specific served name; Super can no longer silently start the image's
  Nano default. The seeded Super deployment also applies the live-validated
  65,536-token context clamp required for its 62.6 GiB BF16 checkpoint on a
  96 GB single-GPU host; without it the image's 262k default demanded a 64 GiB
  KV cache and failed after loading. Handoffs and bundles preserve the clamp,
  and container startup failures now retain the actionable root-cause line
  instead of only generic shutdown messages. Compare also reconciles a durable
  server-side `serving_status=pending` after refresh, keeps polling, and
  suppresses every duplicate deploy action until the lifecycle terminates.
  The project-wide deploy banner now defers to that newer lifecycle during
  preflight instead of showing an older failed Student deployment as current.

- 2026-08-04: Made post-TAO artifact processing visible and truthful in
  Training Jobs. Job and suite responses now expose the durable
  `outputs_fetch_status` lifecycle; a remotely completed job remains
  **Finalizing** while the Blueprint downloads/packages its artifacts, and the
  next job identifies that artifact handoff instead of claiming its TAO
  predecessor is unfinished. Artifact-processing failures now render as a
  distinct **Artifact Failed** state with the sanitized diagnostic, and the UI
  keeps polling until the Blueprint-side lifecycle is terminal.

- 2026-08-04: Corrected Student serving and production-handoff dataset
  provenance. `StudentModel.dataset_export_ids` remains the authoritative
  training-data lineage; the Blueprint now follows the artifact-producing
  train/quantize job to its paired evaluation job and records the held-out
  export checksum from either `evaluation` or `testing` intent. Compare
  handoffs and portable NIM bundle manifests no longer claim the Test Pool was
  unexported when the training suite created a real evaluation export.

- 2026-08-04: Added an explicit LoRA versus Full-weight choice to Student
  Training and made the backend preflight the authority for model/runtime
  compatibility. Live Cosmos 3 Super qualification proved that
  `6.26.3-cosmos-rl` cannot tensor-parallelize its LoRA-wrapped modules, so the
  Blueprint now stops that incompatible selection before TAO spend, explains
  the failure in the UI, and allows the SME to rerun readiness and submit the
  supported Full-weight path. The chosen method is now visible in the final
  workload confirmation and persisted suite request.

- 2026-08-04: Stabilized Compare & Benchmark's first render so it waits for
  model catalog, Guidance, and referenced evaluation details before presenting
  cards. The page no longer briefly exposes raw model UUIDs or changes model
  names after paint, and Cosmos 3 Students use the same friendly identity as
  the rest of the UI. Comparison deltas now retain a meaningful fractional
  percentage point (for example, `-2.5 pts`) rather than rounding a real
  half-point gap down to a misleading whole number.

- 2026-08-04: Restored the UI entry point for a packaged Student whose TAO
  evaluation is deliberately skipped for NIM-based cold-start validation.
  Quality-pending Students now remain visible on Compare with a clear pending
  explanation and **Deploy and benchmark** action; previously the page filtered
  them out, leaving the supported NIM-only quality path reachable only through
  the REST API.

- 2026-08-04: Made local Student/Teacher NIM preflight invoke the image's
  standalone `list-model-profiles` binary directly. Shell-form NIM entrypoints
  previously ignored the appended probe argument and started a full inference
  server, unnecessarily holding most GPU memory until the bounded timeout
  before the real deployment could begin. The probe now remains fast while
  preserving its timeout cleanup and the authoritative serving health gate.
  Project-wide deployment banners now also distinguish temporary Student NIM
  validation from Teacher and embedding deployments: Student startup explains
  that Compare will reconcile the result, and Student failures link to Compare
  instead of incorrectly directing the SME to Teacher NIM configuration.

- 2026-08-04: Corrected portable Student NIM launchers with a pinned model
  profile so the generated Docker command passes the runtime image immediately
  after `NIM_MODEL_PROFILE`. The prior launcher inserted a stray argument in
  this branch, which could make Docker treat that argument as the image name.
  Bundle validation now executes the generated launcher against a recording
  Docker boundary and verifies the exact profile-to-image argument sequence.

- 2026-08-04: Made the Student Training confirmation preserve the exact
  selected-base request order. Multi-model suites now show the SME the same
  Nano/Super (or other base) sequence that the backend receives and executes,
  rather than silently re-sorting the confirmation into model-catalog order.

- 2026-08-04: Restored the visible NIM-validation path for packaged Students
  whose TAO quality evaluation failed. The Compare screen now lets the SME run
  the normal local Student NIM lifecycle from the failed-variant card, shows
  its live stages, and explains that only the backend's narrowly classified
  upstream loader failures may use a clean NIM evaluation to recover quality.
  Other TAO failures remain failed even if their checkpoint serves. Projects
  whose only Students are quality-failed now render those recoverable records
  instead of an empty state, and a reopened Compare screen reconciles a
  server-side pending validation without offering a duplicate deploy action.
  The loader-gap classifier now follows the exact artifact-producing TAO job,
  so quantized Students classify their evaluate child under `quantize` rather
  than incorrectly inspecting the baseline evaluate under `train`. It also
  inspects the complete bounded 64 KB failure log retained by TAO polling;
  the former 32 KB window could discard an otherwise persisted loader marker
  and conservatively—but incorrectly—leave a cleanly NIM-validated Student
  quality-failed.

- 2026-08-03: Added a gated portable Student NIM deployment bundle to the
  Compare screen. A dual-validated Student can now stream its complete
  project-contained checkpoint with SHA-256 checksums, pinned NIM runtime
  configuration, ready-to-run launch helper, the exact schema-aware request
  template proven by serving evaluation, parsed real-image verification,
  evaluation snapshot, and TAO/dataset lineage. The bundle contains no
  credential and does not redistribute the licensed NIM image; operators pull
  the pinned image from NGC with their own entitlement. Production handoffs
  now enforce validated checkpoint packaging, render the intended five
  sections, and recover a missing release value from the executed image tag.

- 2026-08-03: Corrected TAO quantized-checkpoint retrieval when FTMS returns
  both the quantize job's flat merged checkpoint and its nested parent-train
  adapter tree. Quantize jobs now prefer their own NIM-loadable root, so a
  successful FP8 job registers a deployable Student instead of packaging the
  inherited `adapter_config.json` alone and failing deployment.
  Repackage now refreshes a failed quantized artifact from TAO workspace
  storage before replaying validation, so an already-completed expensive job
  can be recovered in place without retraining.

- 2026-08-03: Corrected self-service TAO base provisioning so the disposable
  Hugging Face download cache is never uploaded into the air-gapped model
  registry, and made the 2B/8B qualification harnesses request only the
  Reason2 bases required by the current run. Required validation no longer
  downloads or stages the optional Cosmos 3 roster implicitly. Deployment-wide
  base identity/status patching now isolates an unsupported historical project
  database so preserved pre-v1 data cannot block current TAO qualification.
  The live wiring-smoke verifier now follows the shipped LoRA baseline contract
  by validating merged Student-NIM evaluation lineage instead of requiring the
  retired adapter-only TAO-evaluate record shape.

- 2026-08-03: Reconciled the user-facing ingestion contract with the shipped
  restartable background pHash sweep: the 202 response may contain
  `phash=null`, selection remains immediately available through deterministic
  fallback, and pHash/CLIP diversity activates as signals arrive. Corrected
  persisted-record counts and UUID-shaped API examples at the same time.

- 2026-08-03: Removed the deprecated Mistral Large 3 free endpoint from
  new-project model seeding and the default live-harness rosters after the
  NVIDIA catalog omitted it and repeated calls returned HTTP 410. Existing
  historical and operator-created model configurations remain inspectable.
  Updated the image-cap probe and hosted setup copy to use the complete current
  hosted Teacher roster, preventing a retired alias or partial default sweep
  from silently returning.

- 2026-08-03: Corrected the Profile B live validation harness so truthful
  Core-correct proposals remain Accepts instead of being converted to Edits by
  an Aux rationale rewrite, and extended its bounded hosted-evaluation poll
  window to cover observed provider latency without resubmitting runs. The
  harness now also records its 20-member RPS Test Pool minimum through the
  project API, leaving the 60-member product default unchanged. Container
  healthchecks now reuse the image's Python runtime instead of installing curl
  solely for a loopback probe, reducing the shipped OS package surface. The
  static UI image now drops root privileges before starting nginx on port 3000.

Notable user-facing changes to the Interactive VLM Feedback Loop are recorded
here. The public release begins with v1.0; private development history and
pre-release database revisions are intentionally excluded.

## 2026-08-03 — Student training data readiness

- Kept Student Training independent of the Teacher-quality criteria that gate
  Batch Labeling while adding a fail-closed held-out data requirement: the
  active-Guidance Test Pool must reach the project-configured minimum before a
  TAO suite can start.
- Retained the existing non-pool Verified training-example requirement and
  added an absolute one-example floor for the evaluation split when an operator
  configures the shared Test Pool threshold to zero.
- Recheck both dataset requirements during final suite materialization after
  first-use base provisioning, and reject an undersized frozen evaluation
  export before upload or TAO job creation.
- Present dataset shortfalls as labeling work on Scale-Up and Student Training,
  not as TAO infrastructure failures or setup Action Requests.
- Keep the deliberately tiny live TAO wiring smoke valid without bypassing the
  product check by explicitly setting that disposable project's Test Pool
  minimum to its one held-out fixture example.

## 2026-08-03 — Release-acceptance automation

- Updated the locked frontend build/test toolchain to patched in-major Vite
  and Vitest releases, with scoped transitive resolutions for current
  high-severity development-dependency advisories.
- Made the cold-start live smoke ingest canonical RPS images beneath its
  operator-selected data root instead of generating files outside a
  containment-enabled backend's `IMAGE_ROOT`.
- Restored the shared glass-card border, radius, blur, and transition on all
  elevated onboarding surfaces instead of applying only the elevated colors.
- Made the Edit Guidance empty state actionable by linking directly to the
  Guidance creation flow when a project has no active Guidance.
- Made the queue-empty Labeling top bar report that no proposal is active
  instead of falsely claiming ICL context selection was still running.
- Made Batch Labeling's advanced helper acknowledge a missing active Guidance
  instead of promising to reuse a configuration that does not exist.
- Moved the schema-evolution live smoke from the retired Mistral Large
  endpoint to the current default hosted Teacher.
- Removed persisted field identities from Edit Guidance draft-validation
  requests so the create-shaped validation API accepts a loaded schema while
  the actual edit request still preserves identity for change classification.
- Preserved circuit-breaker context in the Batch Labeling terminal banner when
  an SME cancels a run from its paused recovery state.
- Reconciled the Labeling screen's Test Pool count immediately after a save
  and replaced its false cold-start ICL message with an honest context-selection
  state while the next Teacher proposal is loading.
- Scoped `scripts/dev.sh` stale-process cleanup to the active source checkout
  so starting or stopping local-source mode cannot mistake a concurrently
  running Compose backend for an orphaned dev process.
- Made the vendored Nebula stageability gate construct its own fresh Git
  repository, allowing the deterministic suite to run from a public snapshot
  that correctly contains no private `.git` metadata.
- Strengthened the live release-acceptance prompt with a terminal-condition
  contract for persistent long-job monitoring, restart-safe ID tracking, safe
  concurrency, and completion instead of status-only handoff.
- Made diagnose, fix, test, live-rerun, evidence, and commit a mandatory loop
  for every clearly wrong, safely fixable Blueprint behavior or instruction;
  report-only findings now require an explicit stop condition.
- Added an optional, non-gating Cosmos 3 Nano/Super comparison lane: a logical
  pre-training clone of the selected real-world project in one separate
  UI-visible project, one multi-model training-suite submission, overlapping
  remote TAO work, sequential Student NIM validation, and retained Training
  Jobs/Compare evidence.
- Made the existing 447-Verified TrashNet project the preferred real-world
  acceptance source, with the cleanly evaluated 300-Verified Freiburg project
  as a bounded external-blocker fallback. The runbook now reconciles legacy
  projects' persisted 80% per-value F1 threshold with the current 60% default
  through the normal API and requires a fresh authoritative evaluation.
- Added the source-to-Profile-C workspace/dataset transfer prerequisite and
  corrected a stale acceptance instruction: the Scale-Up gate blocks Batch
  Labeling only; Student Training uses independent TAO/data/base validation.
- Repaired the shipped live-validation harnesses so hosted runs cover every
  current hosted-compatible Teacher, local-only Cosmos models are not sent to
  the hosted API, capability probes are checked against the seed catalog, and
  SQLite audits follow the project directory returned by the API instead of a
  hard-coded workspace.
- Made the full-pipeline smoke discover recursive class-directory datasets and
  choose a deterministic balanced cohort, aligned the Profile B harness with
  its exact 100-image ingestion contract, and made architecture-image drift
  validation reproducible through an inline uv-managed tool environment.
- Kept public-release link validation focused on the candidate by excluding
  generated dependency environments such as `.venv` from Markdown discovery.
- Added a project-scoped dataset-export download endpoint and completed-state
  UI action, so archives stream with filename, MIME type, and checksum through
  both Vite and nginx instead of assuming the backend/container filesystem is
  visible to the browser or an operator harness.
- Restored visible, accessible required-name validation in the Create Project
  dialog by applying the error status to the complete KUI form field.
- Added descriptive accessible names to file-browser directory and image
  selection controls so the ingest journey is operable with assistive technology.
- Made scan-generated image keys independent of the selected scan ancestor and
  rejected duplicate source paths under alternate keys, preventing a nested-folder
  ingest followed by a dataset-root ingest from creating duplicate examples.

## 2026-08-03 — TAO operator-guide hardening

- Expanded the public TAO FTMS 6.26.3 operator guide with GPU/ECC/topology and
  practical storage gates, distinct NGC Personal/legacy/Hugging Face credential
  roles, and version-pinned NVIDIA references.
- Documented verifiable, durable contracts for the 1440-minute Cosmos-RL
  stale-job timeout and bounded high-resolution quantization preprocessing,
  including the need to reapply patches after service or image refreshes.
- Clarified that Ubuntu 22.04 remains NVIDIA's recommended FTMS 6.26.3
  baseline while Ubuntu 24.04 has also been Blueprint-live-validated; every
  fresh host still requires the complete validation chain. The internal
  installer guard now accepts both validated Ubuntu LTS releases.

## 2026-08-02 — Release-gate hardening

- Added a paste-ready live release-acceptance runbook covering both delivery
  modes, hosted and local NIMs, TAO/Student workflows, recovery testing,
  MiniMax-on-TrashNet regression, evidence retention, and individual visual
  inspection of every screen state.
- Aligned the Python package, backend module, UI package, and product overview
  on version `1.0.0`, with an offline public-release validation check that
  prevents those public version declarations from drifting again.
- Deduplicated shared Student quality and serving evaluation-run queries on the
  Compare screen, removing ambiguous duplicate React Query observers.
- Removed the last shipping references to the retired advanced-test document
  and repaired backend/frontend formatting drift caught by the release gates.
- Documented a narrow frontend audit exception for a React Router advisory
  confined to unused unstable server-component APIs; the standard production
  dependency audit remains clean for the Blueprint's client-only Vite build.
- Split UI pages into route-level production chunks, reducing the initial
  minified JavaScript bundle from roughly 903 KB to 314 KB, and migrated pHash
  pixel extraction off Pillow's deprecated API without changing stored hashes.
- Added a public-snapshot gate that rejects minified or bundled first-party UI
  source while retaining explicitly vendored library artifacts, and moved
  shared UI helpers out of component modules so ESLint is warning-free.
- Restricted the Python source distribution to the backend package and public
  metadata/licenses, preventing the private engineering archive and frontend
  dependency tree from leaking into package uploads, and refreshed the
  third-party manifest for the locked Playwright test toolchain.
- Updated both nginx runtime images to patched stable `1.30.4-alpine` and the
  backend image's retained `uv` executable to `0.12.1`, removing the fixable
  high/critical findings reported by the release container scan.

## 2026-07-31 — Blueprint README refresh

- Reworked the repository front page around the conventions used by current
  NVIDIA AI Blueprints: product overview, use case, workflow, capabilities,
  architecture, software components, audience, requirements, deployment
  profiles, quick starts, configuration, development, and support material.
- Shortened the first-run walkthrough while keeping both Docker Compose and
  local-source delivery modes prominent, and added explicit hosted, local NIM,
  hybrid, and TAO requirement guidance.
- Reconciled README cross-references in deployment documentation and removed
  links to retired validation documents.

## 2026-07-30 — High-resolution quantization recovery

- Cosmos-RL quantization now caps calibration at 128 examples by default,
  preventing high-resolution VLM datasets from exceeding PyArrow's 2 GiB
  nested-array offset ceiling while preserving a representative calibration
  set. Operators can tune the cap with
  `TAO_QUANTIZATION_CALIBRATION_SAMPLES`.
- Failed-job cards now pass the exact TAO job identity into Report TAO Issue,
  so reports contain the local/external IDs, action, model, configuration, and
  logs endpoint.
- When FTMS returns only a generic action-failed status, the Blueprint now
  extracts and classifies the actionable worker exception from captured TAO
  logs for both the monitor and issue report.

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
- A five-criterion readiness gate controls Batch Labeling; Student Training
  uses its independent TAO, data, and base-model readiness checks.
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

# Deployment Guide

The Interactive VLM Feedback Loop ships with two first-class delivery modes. Both are maintained together and validated together — pick whichever fits your environment:

| Mode | Command | Best for |
|---|---|---|
| **Containerized** | `docker compose up --build` | Evaluating the Blueprint, demo hosts, anything that shouldn't need a Python/Node toolchain |
| **Local source** | `./scripts/dev.sh` | Development, debugging, running against local checkouts |

Local NIM containers (Teacher, Student, embeddings) are **not** part of either mode's process tree — the backend orchestrates them at runtime via `docker run`. See [GPU & local NIM runtime behavior](#gpu--local-nim-runtime-behavior).

Configuration reference (settings, secrets, precedence) lives in the [README](../README.md#configuration); this guide covers how to stand the system up.

> [!WARNING]
> The Blueprint is a single-user reference application with **no
> authentication, authorization, or multi-user isolation**. The shipped
> launchers bind to `127.0.0.1`. Do not expose the application directly to the
> public internet. A remote deployment requires a trusted network and an
> operator-managed authenticated reverse proxy or equivalent access-control
> layer.

## Containerized (docker compose)

### Prerequisites

- Docker + [Docker Compose v2](https://docs.docker.com/compose/install/)
- (Optional) `NVIDIA_API_KEY` from [build.nvidia.com](https://build.nvidia.com/settings/api-keys) for hosted Teacher inference and embeddings

> **System-managed local NIM deployment requires local-source mode.** The compose backend runs inside a container with no docker CLI or socket — by design — so it cannot orchestrate local NIM containers (Teacher, Student, embeddings), and the in-app environment assessment says so at runtime. On a GPU host, run the backend in [local-source mode](#local-source-scriptsdevsh) for system-managed local NIMs, or keep the compose stack on hosted NIM endpoints.

### Quick start

```bash
# 1. (Optional) Point the stack at a host directory of images to ingest:
export IMAGE_ROOT=/absolute/path/to/your/images

# 2. (Optional) Export NIM credentials so the container can reach hosted NIM:
export NVIDIA_API_KEY=nvapi-...            # hosted Teacher + NeMo Retriever VL embeddings

# 3. Build and launch the three-service stack (nginx + backend + ui):
docker compose up --build

# Then open: http://localhost:3000
```

### Services

| Service | Image | Port | Role |
|---|---|---|---|
| `nginx` | `nginx:1.27-alpine` | host `${EDGE_PORT:-3000}` → container `80` | Edge reverse proxy. Routes `/v1/*` and `/health` to the backend with SSE-safe passthrough (`proxy_buffering off`, 300 s read/send timeouts; the backend adds `X-Accel-Buffering: no` on the SSE response) so the long-lived `/v1/projects/{id}/events` stream is flushed immediately; routes `/` to the ui. `client_max_body_size 50M` for large scan/ingest JSON payloads. |
| `backend` | built from `src/backend/Dockerfile` | internal `8000` | FastAPI via uvicorn, multi-stage `python:3.12-slim` build (`uv sync --frozen`), runs as non-root `appuser` (uid 1001). Seeded with a minimal `config.yaml` (`WORKSPACE_ROOT: /data/workspace`); every setting is overridable via process env. Healthcheck: `GET /health`. |
| `ui` | built from `src/ui/Dockerfile` | internal `3000` | Vite static bundle served by `nginx:alpine` with SPA client-side-routing fallback and long-cache headers for hashed assets. |

### Environment overrides

Read by `docker-compose.yml` from the shell environment or a top-level `.env`;
backend settings in this table are passed through to the container:

| Variable | Default | Purpose |
|---|---|---|
| `EDGE_BIND_HOST` | `127.0.0.1` | Host address for nginx. Keep loopback for workstation use. Set `0.0.0.0` only on a trusted network behind operator-managed access controls |
| `EDGE_PORT` | `3000` | Host port for the edge nginx |
| `IMAGE_ROOT` | `./deploy/example-images` on the host, mounted at `/data/images` | One image directory for browsing and ingestion. When set, it is mounted read-only at the **same absolute path** inside the container and the Ingest screen opens there automatically |
| `NVIDIA_API_KEY` / `NGC_API_KEY` | unset | NIM credentials passed through to the backend |
| `TAO_API_KEY` / `TAO_API_BASE_URL` / `TAO_ORG_NAME` | unset | TAO FTMS credentials for Student training |
| `TAO_JOB_TIMEOUT_MINUTES` | `1440` | Per-job TAO stale-heartbeat ceiling. This is a dead-job reaper, not a runtime estimate; Student Training preflight blocks an unpatched FTMS v2 API that cannot accept it |
| `TAO_WORKSPACE_S3_ACCESS_KEY` / `TAO_WORKSPACE_S3_SECRET_KEY` | unset | TAO workspace S3 credentials for training-dataset uploads (§9.7.8.2) |
| `HF_TOKEN` | unset | HuggingFace token for gated Cosmos base pulls. **Start Training** uses it to provision a selected missing base in both local-source and containerized modes, and injects it into remote TAO jobs. The local LoRA-merge fallback remains local-source-only. |
| `ALLOW_UI_SECRET_PERSIST` | `false` | Whether the UI's "Save to .env" path may persist pasted API keys. Off by default in containers: the write target (`/home/appuser/.vlm_feedback_loop/.env`) sits in the container's ephemeral layer — no volume covers it — so a "saved" key would silently vanish on the next `docker compose up --build`/recreate. Keys pasted in the UI still apply for the process lifetime; persistent keys belong in the shell environment or the repo-root `.env` compose reads. Set `true` only if you mount a persistent `/home/appuser/.vlm_feedback_loop` |
| `LOG_LEVEL` | `info` | `info` or `debug` |

Credential values belong in the process environment or canonical `.env`,
never in command-line arguments. Local-source TAO bootstrap and provisioning
read `TAO_API_KEY`, `TAO_WORKSPACE_S3_ACCESS_KEY`,
`TAO_WORKSPACE_S3_SECRET_KEY`, and optional `HF_TOKEN` from those approved
sources. The system-managed NIM launcher emits name-only
`-e NGC_API_KEY` in Docker argv and supplies the value through the Docker
client's private child environment. The same contract applies to generated
deployment handoffs and live-validation re-execution.

`IMAGE_ROOT` is deliberately one concept in both delivery modes. To expose
several datasets, place or mount them as subdirectories beneath that one root.

To listen beyond the local workstation, use an explicit override:

```bash
EDGE_BIND_HOST=0.0.0.0 docker compose up --build
```

This does not add authentication. Use it only on a trusted private network
behind an authenticated proxy or equivalent operator-managed boundary. Never
forward the resulting port directly from the public internet.

For local-source mode, override `TAO_JOB_TIMEOUT_MINUTES` in
`~/.vlm_feedback_loop/config.yaml`; Compose accepts the same key from the shell
or repository-root `.env`. The shipped 1440-minute value prevents
cosmos-rl's quiet training loop from being mistaken for a stale job at FTMS's
60-minute fallback. Increase it for legitimate runs longer than 24 hours.

### Data persistence

Project data (SQLite DBs, exports, artifacts, logs) lives in the Docker named volume **`vlm-feedback-loop-workspace`**, mounted at `/data/workspace` in the backend container:

```bash
docker volume inspect vlm-feedback-loop-workspace
```

Docker seeds the volume's ownership from the image mount point (uid 1001), so it works with no extra setup. To use a host directory instead, replace the `workspace:/data/workspace` volume line in `docker-compose.yml` with a bind mount (e.g. `${VLM_WORKSPACE_HOST}:/data/workspace`) — the host directory must be **writable by uid 1001**.

### Stop, restart, resume, and reset

Stop the services while retaining projects:

```bash
docker compose down
```

Restart them with `docker compose up --build` (or `docker compose up -d` when
the images are current). Projects resume from the named
`vlm-feedback-loop-workspace` volume. In-process background jobs are
reconciled from persisted records; consult the relevant monitor after restart.

To inspect the persistent volume:

```bash
docker volume inspect vlm-feedback-loop-workspace
```

> [!CAUTION]
> The following reset permanently deletes every Compose-mode project,
> database, export, artifact, and log in the named workspace volume. Stop the
> stack first and verify the exact volume name before proceeding.

```bash
docker compose down
docker volume rm vlm-feedback-loop-workspace
```

This reset does not delete source-mode projects or the referenced source
images.

### Custom `config.yaml`

The backend image ships a minimal default config. To override it wholesale, uncomment the mount line under the `backend` service in `docker-compose.yml`:

```yaml
# - ./my-config.yaml:/home/appuser/.vlm_feedback_loop/config.yaml:ro
```

The file must be readable by uid 1001. Individual settings can also be overridden via process env, which takes precedence over `config.yaml` (Spec §1.9).

### Zero-setup first run: bundled sample images

With no environment overrides at all, the stack mounts the repo's [`deploy/example-images/`](../deploy/example-images/) at `/data/images` — 15 rock/paper/scissors images (see `deploy/example-images/LICENSE.DATA`) — and the Ingest screen opens there automatically. Launch the stack, create a project, and select the images to walk the full labeling loop without staging any data or typing a server path.

## Local source (`scripts/dev.sh`)

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js 20+ and [pnpm](https://pnpm.io/)
- (Optional) `NVIDIA_API_KEY` for hosted NIM inference

### Quick start

```bash
# 1. Install backend dependencies
uv sync

# 2. Install frontend dependencies
cd src/ui && pnpm install && cd ../..

# 3. Initialize configuration (creates ~/.vlm_feedback_loop/config.yaml and .env)
uv run vlm-feedback-loop init

# 4. Set your API key (optional — needed for hosted NIM inference)
# Edit ~/.vlm_feedback_loop/.env and uncomment NVIDIA_API_KEY=nvapi-...

# 5. Start both backend and frontend with a single command
./scripts/dev.sh

# Backend runs on http://localhost:8000
# Frontend runs on http://localhost:5173 (proxies /v1/ to backend)
```

`dev.sh` requires `~/.vlm_feedback_loop/config.yaml` to exist (step 3), installs frontend deps if missing, and reaps any orphaned backend/Vite processes from a previous run before binding ports — a stale stack from an interrupted session won't block startup.

Arguments passed to `dev.sh` are forwarded to the backend module runner. For
example, `./scripts/dev.sh --host 0.0.0.0` makes that address the effective
`BIND_HOST` used by both Uvicorn and filesystem authorization. Configure
`IMAGE_ROOT` before using a non-loopback host; otherwise browse, scan, ingest,
image serving, and path remapping intentionally return `403`. A raw
`uvicorn ... --host` invocation bypasses the module runner, so operators using
that manual launch form must set `BIND_HOST` to the same address.

Stop local-source mode with `Ctrl+C`. Restart with `./scripts/dev.sh`; project
data remains under `{WORKSPACE_ROOT}/projects/`, using the
`WORKSPACE_ROOT` recorded in `~/.vlm_feedback_loop/config.yaml`.

> [!CAUTION]
> To reset one source-mode project, use the application's project archive or
> delete only the exact `{WORKSPACE_ROOT}/projects/{project_id}` directory
> after stopping the backend. That deletion is permanent. Do not remove the
> workspace root to reset one project.

### Ports and proxying

| Process | Port | Notes |
|---|---|---|
| FastAPI backend | `8000` | Loopback bind by default (`BIND_HOST` / `BIND_PORT` in `config.yaml`) |
| Vite dev server | `5173` | Proxies `/v1/` (REST **and** SSE) to the backend — the browser talks to a single origin |

### First-run images

The backend binds to loopback by default, so the Ingest screen opens at `/`
without an image setting and can browse any readable local dataset. To start in
one directory and keep the browser contained there, set:

```yaml
IMAGE_ROOT: /absolute/path/to/vlm-feedback-loop/deploy/example-images
```

## GPU & local NIM runtime behavior

Local NIMs (Teacher, Student, NeMo Retriever VL embeddings) are **runtime-orchestrated by the backend** — launched via `docker run` from `services/local_nim_service`, not defined as compose services. This requires the backend itself to run in **local-source mode** (`./scripts/dev.sh`): the containerized backend has no docker CLI or socket by design, so in compose mode the environment assessment reports system-managed local NIMs as unavailable and hosted NIM endpoints are the path. Requirements (local-source mode): Docker + NVIDIA Container Toolkit, `NGC_API_KEY`, and a suitable GPU.

- **One NIM per GPU (Spec §1.5).** At most one NIM container runs per GPU, regardless of VRAM math. Deploying onto an occupied GPU uses **replace** semantics: opt-in via `replace_resident: true` on `POST /v1/projects/{id}/local_nim/deploy` (default `false` → `409 gpu_occupied`; all GPUs occupied → `409 gpu_exhausted`). NIM Configuration first submits the non-destructive request, names the exact resident returned by the 409, and pins that GPU only after confirmation. Teacher model changes also send `activate_on_success=true`, so the project changes Teachers only after verification; failed preflight/startup/health verification best-effort requeues the displaced resident. The Student NIM benchmark lifecycle passes `replace_resident=true` automatically and best-effort restores a displaced Teacher afterwards. **Why:** the Cosmos NIM's vLLM backend hardcodes `gpu_memory_utilization=0.9` and its profile selector checks *currently-free* GPU memory, so any neighbor process makes it report `Detected 0 compatible profile(s)` and refuse to start — live-verified on an A100-80GB; see the [README's One-NIM-per-GPU policy](../README.md#one-nim-per-gpu-policy) for the scenario table.
- **Multi-GPU placement** is deterministic: lowest free index first (Teacher → `device=0`, embeddings → `device=1`); embedding deploys skip free GPUs below the embedding memory floor (10 GB seeded for NeMo Retriever VL NIM 2.0.0). Single-GPU hosts run the Teacher locally with pHash image diversity, hybrid with hosted embeddings via `NVIDIA_API_KEY`, or — when the only GPU is below every Teacher floor but at/above the embedding floor — the local embedding NIM with a hosted Teacher.
- **Local embedding NIM: consumed, and the default provider when deployed.** A healthy local embedding NIM (live-verified at probe time) is the default embedding provider; the hosted endpoint (`HOSTED_NIM_BASE_URL`, requires `NVIDIA_API_KEY`) is the fallback, and pHash diversity is the last resort. The local NIM is unauthenticated, so a keyless GPU-only host gets semantic embeddings. The first-run setup chain queues this deploy automatically: on GPU hosts where the placement-aware assessment says the embedding NIM fits, the NGC-key screen appends it to the deploy queue and the setup gate dispatches it — no manual API call needed. When the NIM turns healthy, every non-archived project re-resolves its provider and pending embedding work drains; when it stops or is displaced, the config resets and the next probe falls back to hosted. The `EMBEDDING_*_SELF_HOSTED` worker-shape settings apply on the local path. Live-validated on RTX PRO 6000 Blackwell (SM120): NIM 2.0.0 served a finite 2,048-dimensional image vector at 6.3 GiB.
- **Validated per-image launch profiles.** NeMo Retriever VL NIM 2.0.0 is started with `NIM_PRECISION=fp16`; its unset-precision SM120 path exits while looking for a cuDNN plan directory that its own entrypoint skips creating. Its `NIM_MODEL_PATH` points below the host cache so the ~3.2 GiB model survives container replacement. Nemotron 3 Nano Omni keeps the image's declared `nvs` user: forcing the host UID makes Omni fail in `getpwuid()` before model loading. The orchestrator opens only the shared cache parent/scratch directories required by that declared user; it does not recursively change permissions on cached model artifacts.
- **NIM cache.** Model weights persist on the host at `~/.cache/nim`, mounted into each NIM container at `/opt/nim/.cache`; re-deploys reuse cached weights instead of re-downloading. Most NIMs run as the host user's UID. Omni retains its declared `nvs` user because its startup requires a passwd-backed identity; the cache-parent preparation above keeps that exception writable.
- **Registry authentication is automatic.** The NIM container images on `nvcr.io` are private, and a configured `NGC_API_KEY` alone does not authenticate the *image pull* — that key is passed to the running container for weight download, but the docker daemon needs its own registry login. Every local-NIM deploy therefore runs `docker login nvcr.io -u '$oauthtoken' --password-stdin` (key on stdin, never argv) as preflight **check 5 (`registry_auth`)**, before the image pull. Without it, a fresh host with no `~/.docker/config.json` fails the pull with an opaque `Access Denied` despite a green NGC-key credential test. The login persists to `~/.docker/config.json` and covers subsequent pulls; a failed login stops preflight with an actionable "key needs the NGC Catalog and Private Registry scopes" message. You do **not** need to `docker login nvcr.io` manually first.
- **Container secret forwarding is name-only.** Docker commands use `-e NGC_API_KEY`, which asks Docker to copy the variable from its own process environment. The backend supplies that value only to the Docker child process; neither a literal `KEY=value` assignment nor a shell-value placeholder is written to Docker argv or the rendered Docker-command fields in handoffs, logs, or validation evidence.
- **Deploy requests return immediately.** `POST .../local_nim/deploy` first persists a `starting` reservation for the selected GPU and host port, then runs registry login, image pull, the model-profile probe, container startup, and health polling in tracked background tasks. The setup UI can continue while a cold image downloads; poll the deployment record for `running` or `failed`. Queued host ports count as occupied even before Docker binds them, preventing concurrent background deploys from choosing the same port.
- **Shared-image preflight matches the requested model.** For multi-model images such as `cosmos3-reasoner`, the `list-model-profiles` probe receives the same `NIM_MODEL_SIZE`, pinned `NIM_MODEL_PROFILE` (when configured), and `NIM_SERVED_MODEL_NAME` as the real container. A Super request therefore cannot be preflighted accidentally as the image's Nano default.
- **Teacher profile probes reuse the NIM cache.** Some current VLM images perform a full engine initialization for `list-model-profiles` rather than a cheap manifest query. The bounded probe therefore mounts the same host `~/.cache/nim` directory and uses the same writable container identity as the eventual deployment. Any weights downloaded before the probe finishes or times out are reused by the real container instead of being discarded and downloaded again.
- **Recommended local Teacher is quality-first, hardware-gated:** **Nemotron 3 Nano Omni** (`nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant`) on GPUs with **≥80 GB and compute capability ≥9.0**; **CR3-Nano** on **≥56 GB** when Omni is ineligible; **Cosmos Reason2 2B** on **36–55 GB**. Cosmos Reason2 8B and Cosmos 3 Super remain selectable. GPUs in the **10–35 GB** tier are embedding-only: local NeMo Retriever VL NIM 2.0.0 + hosted Teacher. Full table: [README Hardware Requirements](../README.md#hardware-requirements).
- **Served-model verification.** A "healthy" NIM can still serve the wrong weights (e.g. `cosmos3-reasoner` silently falling back to cached Super weights while reporting the nano name). Before registering a Teacher deployment, the backend queries the NIM's `/v1/metadata` for the actually-loaded model and confirms real weight files exist in the cache; a mismatch marks the deploy **failed** with an actionable reason. Details: [README Served-model verification](../README.md#served-model-verification-on-local-deploy).
- **Per-prompt image cap.** VLM NIM defaults for `NIM_MAX_IMAGES_PER_PROMPT` are version-specific (cosmos `:1.7.0` defaults to 5) and exceeding the cap fails hard with HTTP 400. Every app-driven local deploy sets `NIM_MAX_IMAGES_PER_PROMPT` explicitly to match the backend's ICL image-budget pruner; externally managed NIMs must use the same cap as their corresponding model configuration. Details: [README Per-prompt image cap](../README.md#per-prompt-image-cap-nim_max_images_per_prompt).
- **Operator container env (`extra_container_env`).** Some NIM/GPU pairings need extra env vars on the container — e.g. `NIM_DISABLE_CUDA_GRAPH=1` (remediates the Cosmos Reason2 XID-31 crash on Hopper) or a `NIM_MAX_MODEL_LEN` context clamp. Set `local_deploy_metadata.extra_container_env` (a `KEY: value` object) on the model's catalog entry via `PATCH /v1/projects/{id}/model_configs/{model_config_id}`; every subsequent system-managed deploy of that model appends the pairs as sorted `-e KEY=VALUE` flags, so the remediation rides the product's lifecycle management instead of a manual `docker run`. Keys must be `UPPER_SNAKE_CASE`; builder-owned keys (`NGC_API_KEY`, the `NIM_MODEL_*` / `NIM_MAX_*` selectors) are refused with a logged warning. Values appear verbatim in the operator-visible command displays — never put secrets here.

## Deployment profiles

| Profile | Credentials / hardware | What it enables |
|---|---|---|
| **Hosted NIM** (minimal) | `NVIDIA_API_KEY` | Teacher inference + NeMo Retriever VL 1B v2 embeddings via build.nvidia.com — no local GPU |
| **Local GPU** | `NGC_API_KEY` + Docker + NVIDIA Container Toolkit + GPU + **local-source mode** | System-managed local NIM deployment (Teacher, Student serving; embedding NIM on any free GPU ≥10 GB — the default embedding provider when deployed, no `NVIDIA_API_KEY` needed for embeddings). Not available from the compose backend — see [GPU & local NIM runtime behavior](#gpu--local-nim-runtime-behavior) |
| **Hybrid** (recommended for single-GPU hosts with an API key) | `NGC_API_KEY` + GPU **and** `NVIDIA_API_KEY` | Local Teacher on the only GPU, hosted embeddings — keeps semantic image diversity without a second GPU |
| **TAO access** (Student training) | `TAO_API_KEY` / `TAO_API_BASE_URL` / `TAO_ORG_NAME` + `TAO_WORKSPACE_S3_ACCESS_KEY` / `TAO_WORKSPACE_S3_SECRET_KEY` | Cosmos-RL / TAO fine-tuning of Student models. Workspace *identity* is bootstrapped once via `uv run vlm-feedback-loop tao-bootstrap` (stored in `deployment.db`, not `.env`) |

Profiles compose: a full Scale-Up deployment uses Local GPU (or Hybrid) plus
TAO access. See [`.env.example`](../.env.example) for the annotated secrets
template and [`tao-ftms-install.md`](tao-ftms-install.md) for standing up a
TAO FTMS instance.

> **TAO access in containerized mode:** TAO job submission, monitoring, and dataset upload are remote HTTP and work from the container. The post-training packaging step for **adapter-only (full-precision) checkpoints** — `scripts/merge_lora.py` plus its provisioned merge venv — is **local-source-only**: the backend image ships neither the script nor a merge interpreter. Quantized variants are unaffected (TAO merges them remotely), and serving a trained Student behind a local NIM already requires local-source mode.

For local-source Profile C installations, `scripts/setup-dev.sh` provisions
the isolated merge runtime at
`~/.local/share/vlm-feedback-loop/merge-lora-venv`. LoRA Student Training
readiness fails closed if that runtime or `HF_TOKEN` is absent: the token is
needed on the Blueprint host to load the gated base for adapter merge even
when TAO already has its own provisioned copy.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Edge port already in use | `EDGE_PORT=3001 docker compose up` (compose); in local-source mode, `dev.sh` reaps stale backend/Vite processes automatically — for a genuinely busy port 8000, change `BIND_PORT` in `config.yaml` |
| Backend can't write the workspace | The backend runs as uid 1001. Named-volume default needs nothing; a bind-mounted workspace or custom `config.yaml` must be writable/readable by uid 1001 |
| Changes not showing up in containers | `docker compose build --no-cache && docker compose up` |
| File browser shows nothing | Set `IMAGE_ROOT` to an existing readable absolute directory. It is mandatory on non-loopback binds; Compose supplies `/data/images` automatically |
| Where are the logs? | `docker compose logs -f backend` (or `nginx` / `ui`) for process logs; structured per-project operational logs (JSONL) at `{WORKSPACE_ROOT}/projects/{project_id}/logs/` — inside the `vlm-feedback-loop-workspace` volume in containerized mode |

# Local NIM Development Machine Setup

Complete setup guide for a GPU-equipped development machine running the VLM Feedback Loop Blueprint with local NIM deployment. This machine is used for:

- **Local NIM deployment** — system-managed Teacher and embedding NIM containers
- **General development** — running the full Blueprint stack with local GPU acceleration
- **Student NIM benchmarking** — deploying fine-tuned Student checkpoints for evaluation and latency benchmarks

For TAO training server setup (a different machine), see
[`tao-ftms-install.md`](tao-ftms-install.md).

**NIM version note:** This Blueprint pins Cosmos Reason2 to **VLM NIM 1.6.0**, the last release supporting Cosmos Reason2 (now in NVIDIA's [past-releases documentation](https://docs.nvidia.com/nim/vision-language-models/latest/release-notes.html)); the Cosmos 3 reasoner models use the separate `nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0` image. The general VLM NIM line has since moved on (2.0.6 as of 2026-07-06). Pinning to these tags is intentional for reproducibility.

---

## What This Machine Runs

```
GPU Development Machine
├── VLM Feedback Loop Backend (FastAPI + SQLite)
├── VLM Feedback Loop Frontend (React + Vite)
├── Local Teacher NIM container (Omni quality default; CR3-Nano / Cosmos alternates)
├── Local embedding NIM container (NeMo Retriever VL 1B v2)
└── Student NIM containers (evaluation + benchmarking)
```

---

## 1. Hardware Requirements

### GPU — What NVIDIA Documents vs. What This Blueprint Requires

GPU memory claims below are sourced from the current [VLM NIM support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html) and [NeMo Retriever VL Embedding support matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html) (which covers `nvidia/llama-nemotron-embed-vl-1b-v2`). Where this Blueprint makes a policy choice beyond what NVIDIA documents, it is marked explicitly.

| GPU Memory | What Can Run | Source |
|---|---|---|
| ≥ 10 GB | Embedding-only Blueprint path: NeMo Retriever VL 1B v2 NIM 2.0.0 + hosted Teacher | NIM 2.0.0 compatibility floor; optimized profiles use approximately 5.6–9.3 GiB by architecture/precision. |
| > 36 GB | + Cosmos Reason2 2B Teacher (BF16) | VLM support matrix says "> 36 GB" (strict greater-than). |
| > 56 GB | + CR3-Nano recommended Teacher; Cosmos Reason2 8B also selectable | VLM support matrix plus completed Blueprint long-horizon ICL matrix. |
| ≥ 80 GB and compute capability ≥9.0 | Nemotron 3 Nano Omni recommended Teacher | Specialized Omni NIM 1.7.0 support matrix + Blueprint long-horizon ICL matrix. |
| > 88 GB | Cosmos 3 Super selectable | VLM support matrix; Super is no longer selected merely because this larger floor fits. |

**FP8 note:** NVIDIA's validated FP8 configurations for Cosmos Reason2 are specific GPU models (H100 80 GB, H100 NVL 94 GB, H200, L40S 48 GB, Blackwell-class). A generic "24 GB = FP8-capable" claim is not documented. For FP8, use a validated GPU from the [support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html).

**Compute capability:** Cosmos Reason2 requires compute capability ≥ 7.0 (generic), ≥ 8.0 (BF16), ≥ 8.9 (FP8). This matters when choosing rental GPUs — Ampere (A100) is 8.0, Ada Lovelace (L40S) is 8.9, Hopper (H100/H200) is 9.0.

**Recommended for development:** **H100 80 GB or a supported ≥80 GB Blackwell GPU** — these can run the quality-default Omni Teacher. A100 80 GB is compute capability 8.0, so it falls back to CR3-Nano even though its memory is sufficient.

**Multi-GPU note:** Each local NIM gets its own GPU — the one-NIM-per-GPU policy. Teacher on GPU 0, embedding NIM on GPU 1. Same-GPU co-location is not supported; deploying onto an occupied GPU requires `replace_resident=true` on the deploy request, which stops the resident NIM first (replace semantics). Student NIM benchmarking runs one variant at a time.

### Other Hardware

| Component | Requirement |
|---|---|
| **CPU** | x86_64 (this Blueprint is validated on x86_64 only; some NIMs may also support aarch64 — check the NeMo Retriever VL Embedding and VLM NIM support matrices) |
| **RAM** | 32 GB+ recommended |
| **Disk** | 100 GB+ free (NIM images ~15–25 GB each, NIM cache, workspace data) |
| **glibc** | ≥ 2.35 (per [VLM NIM prerequisites](https://docs.nvidia.com/nim/vision-language-models/latest/getting-started.html)) |
| **Network** | Internet access for NGC image pulls and hosted NIM API calls |

**License requirement:** Self-hosting VLM NIM and the NeMo Retriever VL embedding NIM requires an [NVIDIA AI Enterprise (NVAIE)](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/) license or evaluation entitlement. A valid NGC API key alone is not sufficient if the underlying org does not have NVAIE access — image pulls will fail with an auth error even though the key itself is valid. The `setup-dev.sh` script cannot directly verify contract status. Instead, it verifies practical entitlement by authenticating to NGC and checking access to the exact self-hosted NIM image required for this Blueprint. If the entitlement check fails, the script provides specific diagnostic guidance (missing Catalog permission, wrong org context, or no self-hosting entitlement).

---

## 2. Operating System

| Component | Version |
|---|---|
| **OS** | Ubuntu 22.04 LTS |

Ubuntu 22.04 ships with glibc 2.35, satisfying the NIM prerequisite.

---

## 3. NVIDIA Driver

| Component | Version |
|---|---|
| **NVIDIA Driver** | 580.65.06+ (R580 branch) |

**Why R580, not R570:** NIM VLM 1.6.0 containers use CUDA 13.0, which requires R580+. Some NVIDIA prerequisite docs still reference R570 — that is insufficient.

```bash
# Verify
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
```

Driver installation is OS/hardware-specific and outside this guide's scope. See [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx).

---

## 4. Docker Engine

| Component | Version |
|---|---|
| **Docker Engine** | 29.4.0+ (NVIDIA docs minimum: 23.0.1+; this Blueprint uses a stricter floor) |

The `setup-dev.sh` script installs Docker automatically when a GPU is detected. To install manually:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

---

## 5. NVIDIA Container Toolkit

| Component | Version |
|---|---|
| **NVIDIA Container Toolkit** | 1.19.0+ (Blueprint pin; NVIDIA docs do not specify a minimum toolkit version) |

Enables `--runtime=nvidia` and `--gpus` flags for GPU passthrough. Also installed automatically by `setup-dev.sh` on GPU machines.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

---

## 6. NGC Credentials and API Keys

### NGC Personal API Key

One NGC Personal API Key serves two purposes in this Blueprint:
1. **NGC container registry** — pulling NIM images from `nvcr.io`
2. **Hosted NIM API** — Teacher inference and embeddings via the hosted endpoint at `https://integrate.api.nvidia.com` (the API catalog UI is at `build.nvidia.com`; the actual API base is `integrate.api.nvidia.com/v1`)

If your NGC key has both **NGC Catalog** and **Public API Endpoints** scopes enabled for your user/org, you can use the same key for both `NGC_API_KEY` and `NVIDIA_API_KEY`. Two env vars are an application abstraction — they do not require two separate keys. **Caveat:** Public API Endpoints access depends on your NGC org roles and assigned services. If your key authenticates to NGC but API calls to `integrate.api.nvidia.com` fail, verify that Public API Endpoints access is granted to your user in the NGC org settings.

### Get Your Key

1. Go to [https://org.ngc.nvidia.com/setup/api-key](https://org.ngc.nvidia.com/setup/api-key)
2. Generate a Personal API Key (ensure NGC Catalog scope is enabled)

### Configure Docker Login (non-interactive, per NVIDIA docs)

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

### Configure Blueprint

Add to `~/.vlm_feedback_loop/.env`:

```bash
NGC_API_KEY=<your NGC API key>
NVIDIA_API_KEY=<typically the same NGC key>
```

---

## 7. NIM Container Images and Cache

### Pre-Pull Images

NIM images are large (check the [VLM support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html) and [NeMo Retriever VL Embedding support matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html) for current disk space requirements — typically 19–30 GB per model). Pre-pulling avoids waiting during development:

```bash
# Embedding NIM (default: NeMo Retriever VL 1B v2) — 10 GB floor
docker pull nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0

# Quality-default local Teacher on supported ≥80 GB / cc≥9.0 GPUs.
docker pull nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant

# CR3-Nano Teacher (default when Omni is ineligible) — served from the shared
# cosmos3-reasoner image with NIM_MODEL_SIZE=nano. ~8B-class qwen3_vl reasoner.
docker pull nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0

# Cosmos Reason2 2B Teacher (fallback on 36–55 GB GPUs) — requires GPU ≥ 36 GB (BF16)
docker pull nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0

# Cosmos Reason2 8B Teacher — still fully selectable; requires GPU ≥ 56 GB (BF16)
docker pull nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0

```

**CR3-Nano genuine-weights caveat.** The shared `cosmos3-reasoner:1.7.0`
image serves nano vs super by `NIM_MODEL_SIZE`. A plain `docker pull` fetches
the image but NOT the nano weights — those are downloaded on first deploy. If
the nano profile fetch fails or the cache mount isn't writable by the
container, the NIM silently falls back to the cached SUPER weights (a 52 KB
config-only stub is left in the nano cache). The Blueprint deploy path guards
against this by pinning `NIM_MODEL_PROFILE`
(`e2e00f3e555bb4fe0ef011faadd56a37441c7274e149d482cfeb67dbfb75b092`, the nano
fp8 / RTX PRO 6000 / tp1 profile), running the container as the host UID, and
verifying the served model is backed by real weight files. `setup-local.sh`
pre-caches the genuine ~9.9 GB nano weights using the same recipe via the
image's CPU-only `download-to-cache` entrypoint.

The Blueprint pulls images automatically on first local NIM deploy. You can also pre-pull manually to avoid waiting during development. The `setup-dev.sh` script verifies NGC entitlement but does not pull images — that's the application's job.

### NIM Cache Directory

**Stock NIM containers** (Teacher, embedding NIM) download model artifacts and optimized engines from NGC on first start. **Custom fine-tuned checkpoints** (Student NIM) trigger a local engine build from HuggingFace weights, which takes 5–15 minutes. In both cases, a persistent cache avoids repeating this work:

```bash
mkdir -p ~/.cache/nim
```

All NIM containers mount `-v ~/.cache/nim:/opt/nim/.cache`.

---

## 8. Blueprint Application Stack

### Automated Install

The fastest path — installs everything including GPU components:

```bash
git clone https://github.com/zerodefects/vlm-feedback-loop-public.git
cd vlm-feedback-loop
chmod +x scripts/setup-dev.sh
./scripts/setup-dev.sh
```

### Then Bootstrap

```bash
uv sync                                    # Backend dependencies
cd src/ui && pnpm install && cd ../..      # Frontend dependencies
uv run vlm-feedback-loop init              # Create config + .env skeleton
# Edit ~/.vlm_feedback_loop/.env — add NVIDIA_API_KEY and NGC_API_KEY
```

### Manual Install (if not using setup-dev.sh)

| Component | Install |
|---|---|
| Python 3.12+ | `deadsnakes` PPA or system Python |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js 20+ | NodeSource: `curl -fsSL https://deb.nodesource.com/setup_20.x \| sudo -E bash -` |
| pnpm | `sudo corepack enable && corepack prepare pnpm@latest --activate` |
| Pillow deps | `sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev libwebp-dev` |
| sqlite3 | `sudo apt-get install -y sqlite3` |

---

## 9. Verify Everything

```bash
# 1. Driver + compute capability
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

# 2. Docker + GPU passthrough
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi

# 3. glibc version
ldd --version | head -1
# Expect: 2.35+

# 4. NGC images cached
docker images | grep nvcr.io/nim

# 5. NIM cache
ls ~/.cache/nim

# 6. Backend tests
uv run pytest tests/unit/ -q --tb=no

# 7. Frontend build
cd src/ui && pnpm build && cd ../..

# 8. Start servers + environment check
./scripts/dev.sh &
sleep 5
curl -s http://localhost:8000/v1/environment | python3 -m json.tool
# Expect: docker_available=true, nvidia_toolkit_available=true, gpus listed
kill %1
```

---

## Port Allocation

| Service | Default Port | Config Key |
|---|---|---|
| Blueprint Backend | 8000 | `BIND_PORT` |
| Blueprint Frontend (dev) | 5173 | Vite config |
| Local Teacher NIM | 8000 → host dynamic | `LOCAL_NIM_TEACHER_PORT` |
| Local embedding NIM | 8000 → host 8001 | `LOCAL_NIM_NVCLIP_PORT` |
| Student NIM | 8000 → host dynamic | Auto-allocated |

The system auto-allocates the next available port when the preferred port is occupied.

---

## Container Lifecycle and Docker Run Templates

### Lifecycle

All local NIM containers (Teacher, embedding NIM, Student) follow this lifecycle:

```
1. Preflight (6 checks)     → Pass or generate Action Request
2. Start container           → docker run with image + cache mount + GPU
3. Health poll               → GET /v1/health/ready (up to 1200s)
4. Auto-register             → Endpoint URL persisted in project DB
5. [Use the service]         → Inference, embedding, or evaluation
6. Stop container            → docker stop + remove (or persist)
```

### 6-Step Preflight

| # | Check | Failure |
|---|---|---|
| 1 | Docker available (`docker info`) | Suggest install |
| 2 | NVIDIA Container Toolkit (GPU passthrough) | Suggest install |
| 3 | GPU memory sufficient for target model | Name GPU and requirement |
| 4 | NGC API key configured | Explain how to add to .env |
| 5 | Container image pullable | Check NGC credentials |
| 6 | Model profile compatible (via NIM container `list-model-profiles` utility) | Name hardware mismatch |

**Note on `list-model-profiles`:** This is a utility provided by the NIM container, not a host binary. The system runs it inside the container to check hardware compatibility before launching the main model server — which is why the image pull (check 5) must succeed first.

### VLM Docker Run (Teacher / Student)

```bash
docker run --rm -d \
  --name vlm-{service}-{id} \
  --runtime=nvidia --gpus '"device=0"' \
  --shm-size=32GB \
  -u $(id -u) \
  -p {host_port}:8000 \
  -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  {vlm_container_image}
```

For Student NIM, add checkpoint mount:
```bash
  -e NIM_MODEL_NAME=/checkpoints/{dir} \
  -e NIM_SERVED_MODEL_NAME=student-{variant_id} \
  -v /path/to/checkpoint:/checkpoints/{dir}:ro \
```

### Embedding NIM Docker Run (Default: NeMo Retriever VL 1B v2)

The Blueprint applies `-u $(id -u)` to Teacher, Student, and embedding images that support an arbitrary UID so the shared cache stays writable. Nemotron 3 Nano Omni is the supported exception: its `1.7.0-variant` startup calls `getpwuid()` and exits when the host UID has no image-local passwd entry. The orchestrator therefore keeps Omni's declared `nvs` user and makes only the shared cache parent/scratch directories writable to it; existing model artifacts are not recursively permission-changed.

```bash
docker run --rm -d \
  --name vlm-embed-{id} \
  --runtime=nvidia --gpus '"device=1"' \
  --shm-size=32GB \
  -u $(id -u) \
  -p {host_port}:8000 \
  -e NGC_API_KEY \
  -e NIM_PRECISION=fp16 \
  -e NIM_MODEL_PATH=/opt/nim/.cache/models/llama-nemotron-embed-vl-1b-v2 \
  -v ~/.cache/nim:/opt/nim/.cache \
  nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.0.0
```

Both examples use Docker's name-only environment forwarding. Export
`NGC_API_KEY` in the shell before running them; do not place its value in the
command line, where it would be visible through process inspection.

Those two embedding-specific environment values are product behavior, not optional tuning. On SM120, the image's unset-precision path exits while requiring `/opt/cache/sm120/cudnn-sdpa-plan`, although its entrypoint skips creating that directory. The explicit FP16 path selects the working AVO profile. The model path moves the ~3.2 GiB download out of the disposable container layer. A live RTX PRO 6000 Blackwell validation served a finite 2,048-dimensional image embedding at 6.3 GiB VRAM.

---

## Student NIM Benchmarking

This same machine deploys fine-tuned Student checkpoints for evaluation and latency benchmarking (the Compare & Benchmark screen). Everything above is prerequisite.

### Load Driver (genai-perf → AIPerf)

NVIDIA's [VLM benchmarking docs](https://docs.nvidia.com/nim/vision-language-models/latest/benchmarking.html) document **genai-perf** as the current VLM benchmarking tool. The Blueprint isolates the load driver behind a benchmark adapter because NVIDIA is steering newer benchmarking work toward **AIPerf** — but for VLMs today, genai-perf is the documented path.

**Install:** genai-perf's `pip install` path expects **host CUDA 12**. Since this setup relies on NIM containers and does not install a host CUDA toolkit, use the **Triton SDK container approach** documented on the VLM benchmarking page (includes a tested `nvcr.io/nvidia/tritonserver` tag and a tested genai-perf commit for reproducibility):

```bash
# Option 1: pip install (requires host CUDA 12 already installed)
pip install genai-perf

# Option 2: Triton SDK container (no host CUDA needed — recommended)
# The VLM benchmarking page documents a specific tritonserver SDK tag
# and tested perf_analyzer commit. Use those for reproducible results.
```

### Benchmark Configuration

| Setting | Default | Config Key |
|---|---|---|
| Concurrency levels | [1, 8, 24] | `STUDENT_LATENCY_TEST_CONCURRENCIES` |
| Startup timeout | 1200s | `NIM_STARTUP_TIMEOUT_S` |
| Benchmark timeout | 1200s | `NIM_BENCHMARK_TIMEOUT_S` |

### Metrics Collected

From benchmark sweep: p50/p90/p99 end-to-end latency per concurrency level.

From NIM Prometheus (`/v1/metrics`): TTFT, ITL, request success/failure counts, GPU cache utilization. **Caveat:** Vision-specific metrics (encoder timing) are only available on TRT-LLM backends. Cosmos Reason2 custom checkpoints use the **vLLM backend**, which collects **end-to-end metrics only** (see [VLM Observability docs](https://docs.nvidia.com/nim/vision-language-models/latest/observability.html)).

### Student Checkpoint Requirements

TAO-produced checkpoints must be NIM-loadable HuggingFace format:

```
checkpoint_dir/
├── config.json
├── generation_config.json
├── model-*.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── runtime_params.json
```

**LoRA merge:** Full-precision baselines need adapter merged before NIM deployment. Quantized variants produced by TAO include merged weights (per TAO Cosmos-RL quantize workflow — see Spec §9.8). VLM NIM serves merged checkpoints only.

### GPU Sharing with Teacher

Student NIM is temporary — each variant starts, benchmarks, and stops. On a multi-GPU host the Student takes a free GPU (one NIM per GPU, never co-located). On a single-GPU host the Student NIM lifecycle automatically displaces the resident Teacher before the benchmark (the displacement is audited on the deployment record) and best-effort restores the Teacher afterwards; if restoration fails, a warning is surfaced and the Teacher can be re-deployed from the NIM Configuration page.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi` fails | Driver missing/wrong version | Install R580+, reboot |
| `docker: command not found` | Docker not installed | Run `setup-dev.sh` or install manually |
| `--runtime=nvidia` fails | Toolkit missing | Install + `nvidia-ctk runtime configure` |
| NGC pull 401 | Not logged in | `echo "$NGC_API_KEY" \| docker login nvcr.io --username '$oauthtoken' --password-stdin` |
| Health poll >1200s | First start builds engines | Wait; ensure `~/.cache/nim` mount |
| "No runnable profile" | GPU too small or wrong compute capability | Check memory and compute cap vs [support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html) |
| Omni exits with `getpwuid(): uid not found` | A manual command forced an arbitrary `-u` value | Use the Blueprint-managed command, which retains Omni's declared `nvs` user and prepares its cache parents |
| Embedding NIM asks for a missing `sm120/cudnn-sdpa-plan` | NIM 2.0.0 started without the validated precision pin | Use the Blueprint-managed command or add `-e NIM_PRECISION=fp16` |
| Port conflict | Multiple services | System auto-allocates next port |
| genai-perf not found | Not installed | `pip install genai-perf` (Student benchmarking only) |
| Checkpoint rejected | LoRA not merged | Check HuggingFace dir structure (Student benchmarking) |

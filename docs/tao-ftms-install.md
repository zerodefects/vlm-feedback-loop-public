# TAO FTMS Operator Guide

This guide separates the Subject Matter Expert (SME) handoff from the
infrastructure work required to enable Student training. The Blueprint talks
to NVIDIA TAO Fine-Tuning Microservices (FTMS) over its v2 REST API; TAO may run
on a separate GPU cluster and does not share a filesystem with the Blueprint.

The supported and validated release is **TAO FTMS 6.26.3** with the matching
`6.26.3-cosmos-rl` container. Treat a different TAO release as a new
compatibility target and revalidate the preflight and one complete training
suite before making it available to SMEs.

NVIDIA's versioned 6.26.3 FTMS setup guide recommends Ubuntu 22.04. This
Blueprint integration has also been live-validated on Ubuntu 24.04 with an
8× A100 host, but 24.04 remains a Blueprint-validated variant rather than the
vendor-recommended 6.26.3 baseline. On either release, require every host, API,
storage, patch, and end-to-end check in this guide for each fresh deployment.
Do not apply host claims from the newer TAO 7 line to this FTMS 6.26.3
integration without revalidation.

## 1. SME handoff

The SME should not install TAO. When Student Training reports that
infrastructure is required, use **Request TAO setup** and send the generated
Action Request to the infrastructure operator. It supplies non-secret
diagnostics and the contract the deployment must satisfy.

The operator returns:

- a backend-reachable TAO v2 base URL, such as
  `https://<tao-host>/api/v2`;
- the TAO organization name;
- an NGC Personal API Key or pre-exchanged TAO JWT;
- a workspace name, bucket, external S3 URL, and internal S3 URL;
- workspace S3 access and secret keys; and
- confirmation that the health checks in this guide pass.

Do not put credentials in an Action Request, issue, chat transcript, command
argument, Git repository, `config.yaml`, or project database.

## 2. Infrastructure requirements

TAO training infrastructure and the Blueprint host are separate roles:

| Role | Requirement |
|---|---|
| Blueprint backend | HTTP(S) access to TAO and S3; a GPU is not required for remote training |
| TAO API and workers | Linux; Ubuntu 22.04 LTS is NVIDIA's pinned 6.26.3 recommended baseline. Ubuntu 24.04 is Blueprint-live-validated; treat other distributions/releases as new compatibility targets |
| Cosmos training | NVIDIA's documented Cosmos-Reason baseline is 8× A100 80 GB; size the cluster from the current TAO support matrix |
| Driver | R580 or newer for the CUDA 13 runtime in the pinned 6.26.3 containers |
| Host storage | 200 GB is an installation floor; use at least 500 GB for one retained train/evaluate/quantize suite and prefer 1 TB for variants and retries |
| Workspace storage | S3-compatible durable storage reachable by both TAO jobs and the Blueprint backend |
| Network | TLS or a trusted private network; expose only the API and storage endpoints required by the two systems |

The credentials have distinct roles:

| Credential | Purpose |
|---|---|
| NGC Personal API Key (`nvapi-...`) | FTMS v2 login/JWT exchange; create it in the organization used by the deployment |
| NGC legacy API key | `nvcr.io` container-registry login for the pinned TAO images |
| `HF_TOKEN` | Access to selected gated Hugging Face Student bases and LoRA merge/evaluation |
| Workspace S3 key pair | Least-privilege read/write access to the deployment's workspace bucket |

The pinned NVIDIA setup requires both NGC key types. Keep them separate and
pass the legacy registry key through stdin rather than a command argument:

```bash
printf '%s' "$NGC_LEGACY_API_KEY" | \
  docker login nvcr.io --username '$oauthtoken' --password-stdin
```

### Host health and capacity gate

Run this before changing drivers or pulling large images:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total \
  --format=csv
nvidia-smi topo -m
nvidia-smi -q > /tmp/tao-gpu-health.txt
df -h / /data 2>/dev/null || true
if command -v dcgmi >/dev/null; then
  sudo dcgmi diag -r 2
fi
```

For the 8× A100 baseline, all eight devices must be present and healthy. Stop
on uncorrectable ECC, failed row remapping/retired pages, a disappearing GPU,
or an unexpected topology. On HGX/NVSwitch systems, require the matching
NVIDIA Fabric Manager service after the final driver is loaded. Select the
large Docker data root before the first image pull; after Docker starts,
confirm `docker info` reports the intended root and storage driver.

VLM NIM serving is a separate concern. See
[`local_nim_dev_setup.md`](local_nim_dev_setup.md) for local Teacher,
embedding, and Student serving-validation setup.

## 3. Install TAO FTMS

Follow NVIDIA's
[TAO 6.26.3 Microservices Setup](https://docs.nvidia.com/tao/tao-toolkit/6.26.03/text/tao_toolkit_api/api_setup.html)
for host prerequisites. The following reproducible skeleton uses NVIDIA's
public `tao_tutorials` Compose bundle at the exact source revision validated
for this Blueprint:

```bash
export TAO_INSTALL_ROOT=/opt/nvidia/tao-ftms
export TAO_TUTORIALS_COMMIT=3e2b4eb51549ed9aac70637d8c3ee07bc676773f

sudo install -d -m 0755 "$TAO_INSTALL_ROOT"
sudo chown "$(id -u):$(id -g)" "$TAO_INSTALL_ROOT"
git clone https://github.com/NVIDIA/tao_tutorials.git \
  "$TAO_INSTALL_ROOT/tao_tutorials"
git -C "$TAO_INSTALL_ROOT/tao_tutorials" checkout --detach \
  "$TAO_TUTORIALS_COMMIT"
cd "$TAO_INSTALL_ROOT/tao_tutorials/setup/tao-docker-compose"
```

Confirm that `config.env` pins this family before launch:

```dotenv
IMAGE_TAO_API=nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl
IMAGE_TAO_PYTORCH=nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt
IMAGE_TAO_DEPLOY=nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy
IMAGE_TAO_DS=nvcr.io/nvidia/tao/tao-toolkit:6.26.3-data-services
IMAGE_COSMOS_RL=nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl
```

Populate the Compose bundle's `secrets.json` and storage configuration by
following its pinned README. Use deployment-specific values in the real file;
never substitute a credential into this guide or commit that file. For the
bundle-provided SeaweedFS profile:

```bash
./run.sh config
./run.sh up-all
./run.sh status
```

For an organization-managed S3 service, configure that service instead and
ensure:

- TAO job containers use an internal endpoint reachable from their network;
- the Blueprint uses an external endpoint reachable from its backend host;
- both endpoints address the same bucket; and
- credentials have only the required bucket permissions.

### Required compatibility patches

These are deployment-owned compatibility changes for the pinned 6.26.3
stack. Keep their source, checksums, target image IDs, and re-application
procedure in the TAO deployment's change-control system; do not commit a
machine-specific patch or rebuilt image to this Blueprint repository.

#### Long-job timeout schema

The Blueprint sends `timeout_minutes` on each job because Cosmos-RL may not
emit a heartbeat during a healthy long-running training step. The TAO v2
`ExperimentJobReq` schema must accept that field. Student Training preflight
checks this contract and refuses to submit if it is missing.

Apply the deployment-maintained schema patch to both the FTMS API and workflow
services, then restart them and the edge proxy. The patch mirrors the shipping
v1 `timeout_minutes` declaration into the v2 experiment/dataset request
schemas. The Blueprint's default is 1440 minutes: this preserves a 24-hour
dead-job reaper rather than disabling cleanup, while avoiding the unsafe
implicit 60-minute ceiling.

Verify the effective API contract rather than trusting a successful restart:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8090/api/v2/openapi.json \
  -o /tmp/tao-openapi.json
python3 - <<'PY'
import json

with open("/tmp/tao-openapi.json", encoding="utf-8") as handle:
    schemas = json.load(handle)["components"]["schemas"]
properties = schemas["ExperimentJobReq"].get("properties", {})
assert "timeout_minutes" in properties, properties.keys()
print("PASS: ExperimentJobReq accepts timeout_minutes")
PY
```

#### Quantization preprocessing

The operator must also validate Cosmos-RL quantization with representative
image resolution. For the pinned 6.26.3 release, keep the Blueprint's
`max_sequence_length=16384` request and ensure the deployed **job image** sets
`writer_batch_size=4` on the quantization preprocessing `ds.map(...)` call in
`/workspace/cosmos_rl_merged/cosmos_rl/quantization/quantize.py`. This bounds
nested pixel-array accumulation below Arrow's 2 GiB list-offset ceiling.

For a local Compose worker, verify the effective image directly (adapt the
image reference for a registry-published patched image):

```bash
docker run --rm --entrypoint bash \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-cosmos-rl -lc \
  'grep -n "writer_batch_size=4" /workspace/cosmos_rl_merged/cosmos_rl/quantization/quantize.py'
```

The Blueprint also caps calibration at 128 samples and sends the LoRA merge
fields required by quantization. Those client-side safeguards do not replace
the job-image writer bound. A failed representative-resolution quantization
test is an infrastructure blocker; do not bypass it.

Cosmos 3 Super is deliberately baseline-only in this Blueprint release. On the
qualified 6.26.3 stack, Super FP8 activation calibration completed, but
llmcompressor's weight observer failed on an Accelerate CPU-offloaded
meta-device parameter (`Tensor.item() cannot be called on meta tensors`). This
does not invalidate the successfully qualified full-weight Super Student/NIM
path. Do not patch around the failure or submit another Super quantize job from
the Blueprint; clear every quantization scheme for Super. Re-enable that
matrix only after a replacement stack completes Super quantization, packaging,
local NIM serving, evaluation, and benchmark qualification.

In-place API patches may be lost when services are recreated, and a registry
pull may replace a locally re-tagged job image. Re-apply/rebuild and rerun both
checks after any TAO service, image, driver, or patch refresh. A later TAO
release must either support these contracts natively or receive an
independently reviewed compatibility change.

## 4. Validate FTMS health

On the TAO host:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,driver_version,memory.total,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total --format=csv
nvidia-smi topo -m
docker ps --format '{{.Names}}\t{{.Status}}'
docker info --format 'Storage={{.Driver}} Root={{.DockerRootDir}}'
cd "$TAO_INSTALL_ROOT/tao_tutorials/setup/tao-docker-compose"
./run.sh status
```

From the Blueprint backend host, first verify basic reachability without
printing a secret:

```bash
curl --fail --silent --show-error \
  "https://<tao-host>/api/v2/openapi.json" >/dev/null
```

Then configure credentials in the deployment secret store and use the
Blueprint's preflight. A stock FTMS v2 deployment exchanges an NGC Personal API
Key at `POST /api/v2/login` for a JWT; the Blueprint performs this exchange
automatically when `TAO_API_KEY` starts with `nvapi-`.

Health is not complete until all of these pass:

- TAO OpenAPI is reachable from the Blueprint backend.
- Authenticated `GET /orgs/<org>/jobs?limit=1` succeeds.
- `ExperimentJobReq` accepts `timeout_minutes`.
- The effective Cosmos-RL job image contains the bounded quantization writer
  batch.
- GPU count, ECC state, topology, Fabric Manager (when applicable), and Docker
  storage match the deployment record.
- Workspace create/read validation succeeds.
- The external S3 endpoint permits Blueprint upload and download.
- TAO jobs can read the same bucket through the internal S3 endpoint.
- The Quick train/evaluate/FP8 quantize/evaluate suite succeeds, and
  quantization is separately confirmed at the production dataset's
  representative image resolution.

## 5. Connect the Blueprint

### 5.1 Set deployment secrets

For local-source mode, place the values in
`~/.vlm_feedback_loop/.env`. For Compose, place them in the repository-root
`.env` or inject them through the deployment's secret manager:

```dotenv
TAO_API_BASE_URL=https://<tao-host>/api/v2
TAO_ORG_NAME=<organization>
TAO_API_KEY=<ngc-personal-api-key-or-tao-jwt>
TAO_WORKSPACE_S3_ACCESS_KEY=<workspace-access-key>
TAO_WORKSPACE_S3_SECRET_KEY=<workspace-secret-key>
HF_TOKEN=<hugging-face-token-if-required-by-selected-base>
TAO_JOB_TIMEOUT_MINUTES=1440
```

`TAO_JOB_TIMEOUT_MINUTES` is a stale-job ceiling, not an ETA. Keep it above the
longest legitimate job while retaining a finite backstop for a genuinely dead
trainer.

### 5.2 Bootstrap the shared workspace

Run this once per Blueprint deployment, not once per project:

```bash
uv run vlm-feedback-loop tao-bootstrap \
  --workspace-name <workspace-name> \
  --cloud-type seaweedfs \
  --bucket <bucket-name> \
  --s3-endpoint-url-external https://<s3-host-reachable-by-blueprint> \
  --s3-endpoint-url-internal http://<s3-service-reachable-by-tao-jobs> \
  --self-service
```

The command stores only non-secret workspace identity in `deployment.db`.
It is safe to rerun. In Compose mode, stop the long-running backend and run
the same command in the backend service so it uses the named workspace volume:

```bash
docker compose build backend
docker compose stop backend
docker compose run --rm backend \
  vlm-feedback-loop tao-bootstrap \
  --workspace-name <workspace-name> \
  --cloud-type seaweedfs \
  --bucket <bucket-name> \
  --s3-endpoint-url-external https://<s3-host-reachable-by-blueprint> \
  --s3-endpoint-url-internal http://<s3-service-reachable-by-tao-jobs> \
  --self-service
docker compose up -d backend
```

Self-service mode provisions a selected missing Student base when a suite
starts. An operator may warm all supported bases in advance:

```bash
uv run vlm-feedback-loop tao-pull-base-experiments
```

For air-gapped or policy-separated deployments, use `tao-bootstrap
--admin-managed` with administrator-provided base-experiment UUIDs.

### 5.3 Confirm in the application

Restart the backend after changing its environment, open **Scale Up**, and
wait for the Student Training card to finish checking. It is ready only when
TAO, timeout support, workspace, credentials, selected base roles, and usable
training data all pass. Usable data means at least one active-Guidance Verified
example outside the Test Pool plus an active-Guidance Test Pool at the
project-configured minimum (default 60; effective floor one). The SME can then
use **Validate training setup**, which defaults to one small Student base, the
Quick preset, and one FP8 variant.

## 6. Data and artifact contract

The Blueprint uploads versioned Cosmos-RL/LLaVA archives to the workspace:

```text
dataset_export/
  images/
  annotations.json
```

`annotations.json` is a top-level array. Each sample contains an `images`
list and `conversations` turns using `from`/`value`; the human turn contains a
literal `<image>` marker. Training excludes the Test Pool. Evaluation uses a
separate Verified-only frozen Test Pool archive.

TAO exposes job state and file listings over REST. Artifact bytes are read
directly from the workspace S3 service. The Blueprint imports checkpoints,
logs, and per-sample predictions into the project artifact store and records
their lineage.

The first-run validation chain for one base and `FP8_DYNAMIC` is exactly:

1. train;
2. evaluate the full-precision baseline;
3. quantize FP8;
4. evaluate the FP8 checkpoint.

This confirms wiring. A tiny successful run is not evidence of a
production-quality model.

## 7. Troubleshooting

| Symptom | Operator action |
|---|---|
| TAO not reachable | Verify DNS/TLS/firewall and the backend-reachable base URL |
| 401/403 | Verify organization and login/JWT exchange; rotate the credential if exposure is suspected |
| Timeout field rejected | Apply/review the 6.26.3 schema compatibility patch, restart FTMS, rerun preflight |
| Train is terminated near 60 minutes | Confirm the submitted job contains `timeout_minutes`, then repeat the OpenAPI check against the effective API/workflow deployment |
| Workspace check fails | Confirm both S3 endpoints address the same bucket and credentials permit required objects |
| Base provisioning fails | Verify NGC/Hugging Face access or register an administrator-managed base experiment |
| Job remains queued | Check TAO scheduler and GPU availability |
| Train starts but fails | Check workspace `results/<job-id>/microservices_log.txt`, worker logs, driver/runtime compatibility, dataset access, GPU health, and sizing |
| Job remains `Running` after a worker reports `Policy ... is dead` | Preserve logs, cancel the wedged job, correct the worker/GPU failure, and rerun; do not accept partial artifacts |
| Quantization fails | Validate sequence length and preprocessing-batch compatibility on the effective pinned job image; rebuild after an image refresh |
| Artifact import fails | Check `:list_files` output and external S3 read permissions |

## 8. Security and change control

- Bind TAO and its storage to trusted networks; use TLS across network
  boundaries.
- Store secrets only in deployment secret stores or approved `.env` files
  with restrictive permissions.
- Never log, paste, or commit personal API keys, JWTs, SSH keys, host
  identities, or storage secrets.
- Keep `TAO_TUTORIALS_COMMIT`, container tags, compatibility patches, and
  their checksums and effective image IDs immutable for a validated
  deployment.
- Re-run the complete validation chain after changing TAO, Cosmos-RL,
  drivers, storage, auth, or patches.

## 9. References

- [TAO 6.26.3 Microservices Setup](https://docs.nvidia.com/tao/tao-toolkit/6.26.03/text/tao_toolkit_api/api_setup.html)
- [TAO 6.26.3 REST API](https://docs.nvidia.com/tao/tao-toolkit/6.26.03/text/tao_toolkit_api/api_rest_api.html)
- [TAO 6.26.3 release notes](https://docs.nvidia.com/tao/tao-toolkit/latest/text/release_notes.html#tao-6-26-3)
- [TAO 6.26.3 Cosmos-Reason fine-tuning](https://docs.nvidia.com/tao/tao-toolkit/6.26.03/text/vlm_finetuning/cosmos_rl.html)
- [Pinned TAO tutorials source](https://github.com/NVIDIA/tao_tutorials/tree/3e2b4eb51549ed9aac70637d8c3ee07bc676773f/setup/tao-docker-compose)

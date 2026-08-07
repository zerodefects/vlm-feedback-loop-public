# Agent Instructions — Interactive VLM Feedback Loop

> **AGENTS.md and CLAUDE.md are identical twins.** Same content, two filenames, so every agent tool finds it. Edit them together — `diff AGENTS.md CLAUDE.md` must stay empty.

## Project

Build and maintain the **Interactive VLM Feedback Loop** — an NVIDIA Blueprint for interactive VLM labeling with In-Context Learning (ICL), optional fine-tuning via Cosmos-RL / TAO, and Student deployment via NIM.

The product loop: a **Teacher VLM** (served via NVIDIA NIM) proposes a label for an image → the SME **Accepts / Edits / Skips** → Accepted and Edited labels become **Verified** ground truth → Verified Edits feed subsequent proposals via **ICL** → background **evaluation** against a held-out Test Pool measures accuracy → a 5-criteria **Scale-Up Readiness Gate** unlocks **Batch Labeling** (auto-label at scale). **Student training** is independent of the Teacher-quality gate but requires its own server-side data readiness (at least one non-pool Verified training example plus the project-configured minimum Test Pool) and TAO infrastructure readiness → Cosmos-RL / TAO fine-tunes the seeded Cosmos Reason bases → trained Students deploy behind local NIM and are benchmarked against the Teacher on the Compare screen.

Single-repo app: **Python + FastAPI** backend · **React 19 + TypeScript + Vite** frontend · **KUI Foundations + Tailwind 4** · **SQLite per project** (WAL) with Alembic · in-process **asyncio** background jobs — no external task queue.

This is an **open-source NVIDIA Blueprint reference application**. Optimize for correctness AND repo clarity, inspectability, reproducibility, deployment usability, and teaching value for outside developers. Two delivery modes are first-class and must both stay healthy at all times:

1. **Local source** — backend + frontend from source (`./scripts/dev.sh`).
2. **Containerized** — `docker compose up --build` (nginx + backend + ui).

Any change touching startup, networking, ports, env vars, build steps, static serving, proxying, or local-NIM behavior must update BOTH modes and their docs/launch helpers in the same change. Do not silently diverge from recent NVIDIA Blueprint conventions unless the Engineering Spec requires it.


## Documents — authority and when to read what

Authority order when documents overlap:

1. **`docs/Engineering_Spec.md`** — the normative implementation contract (data model, API contracts, algorithms, prompts, state machines, acceptance tests). Section numbers cited anywhere ("§4.3", "Appendix A.2") refer to this file.
2. **The implemented UI is the screen contract.** There is no separate UI spec — the old UI.md was retired as unmaintainable. Screen behavior = the shipped React code; styling authority = the committed Retail Blueprint reference screenshots (see Visual verification).
3. **`docs/Overview.md`** — the human-readable product summary; loses to the Spec on conflict.

Do NOT read the big documents front-to-back at session start. Work from this file and open the right section when the task touches its area:

| Task touches | Read first |
|---|---|
| Product behavior, workflow, onboarding/FTUE | `docs/Overview.md` (short — the only doc worth reading whole) |
| Consumer-facing API doc (what outside users read) | `docs/API.md` (curated; Spec §10 stays normative — update both when endpoints change) |
| Deployment, compose overrides, GPU/NIM ops docs | `docs/deployment.md` |
| Cross-cutting contracts, invariants, map of where detail lives | `docs/Engineering_Spec_Brief.md` (structure mirrors the full Spec) |
| Guidance / SchemaCore / validation / schema evolution | Spec §4 |
| Ingestion, pHash, embeddings, review selector, pools | Spec §5 (+ §4.3 for Test/Train routing) |
| Prompts, ICL, Teacher invocation, generation controls | Spec §6 + Appendix D |
| Evaluation, metrics, Scale-Up gate | Spec §7 + Appendix A.2 |
| Batch labeling, dataset export formats | Spec §8 + §9.3 |
| Student training / TAO / cosmos-rl wire format | Spec §9.7 + `docs/tao-ftms-install.md` |
| Student NIM deploy / benchmark / deployment handoff | Spec §9.5 |
| API shapes and endpoints | Spec §10 |
| Record families / DB columns | Spec §13 |
| Public release history | `docs/changelog.md` |

## Repository

- Public release mirror:
  `https://github.com/zerodefects/vlm-feedback-loop-public.git`, default branch
  `main`. Maintainers may use a separate private source remote locally.

## Repository layout

```
vlm-feedback-loop/
  docs/                    # user-facing product docs, changelog, and runbooks
  deploy/                  # bundled sample images (example-images/ with LICENSE.DATA inside); compose mounts it at /data/images
  scripts/                 # dev.sh + setup scripts, live smokes/probes, experiment tooling
  src/backend/vlm_feedback_loop/
    routers/               # thin /v1 APIRouters
    services/              # domain logic — the real implementation lives here
    db/models/             # SQLAlchemy models
    schemas/               # Pydantic request/response models
    migrations/versions/   # Alembic revisions (numbered; never rewrite committed ones)
    main.py                # FastAPI app entry
    cli.py
  src/ui/
    assets/kui-foundations-react-external-0.504.1.tgz   # vendored KUI package
    src/{pages,components,api,types,lib,stores,hooks}/
  tests/unit/   tests/integration/
  docker-compose.yml  nginx.conf  pyproject.toml  uv.lock
  .env.example  config.yaml.example  AGENTS.md  CLAUDE.md  README.md
```

Keep the top level shallow and Blueprint-shaped; a new top-level directory needs a clear, documented purpose. Backend: route handlers stay thin, domain logic goes in `services/`. Frontend: feature-oriented; frontend concerns stay local to `src/ui/`. Don't build speculative internal taxonomy ahead of need.

## Canonical commands

```bash
uv sync                                  # backend deps (uv only — never pip/poetry)
cd src/ui && pnpm install                # frontend deps (pnpm only)
uv run vlm-feedback-loop init            # create ~/.vlm_feedback_loop/{config.yaml,.env}
./scripts/dev.sh                         # backend :8000 + frontend :5173 (Vite proxies /v1 + SSE)
docker compose up --build                # containerized stack → http://localhost:3000

uv run pytest tests/unit/ -q             # backend unit tests
uv run pytest tests/unit/ -q --cov=src/backend/vlm_feedback_loop --cov-report=term   # + coverage (CI gate: fail_under in pyproject)
uv run pytest tests/integration/ -q -n 0 # integration tests — SERIAL ONLY (they spawn live servers on fixed ports; the conftest refuses xdist)
uv run ruff check . && uv run ruff format --check .
uv run pyright src/backend/              # strict mode; must stay 0 errors / 0 warnings
cd src/ui && pnpm test                   # vitest
cd src/ui && pnpm test:e2e               # Playwright first-time workflow
cd src/ui && pnpm typecheck && pnpm lint && pnpm build

uv run python scripts/generate_third_party_licenses.py   # regenerate LICENSE-3rd-party.txt after dependency changes
uv run scripts/render_architecture_diagram.py            # re-render docs/images/architecture.png after editing the .mmd
```


Migrations apply automatically when a project DB is opened (`db/engine.py`). The public lineage starts at `v1_0001`; pre-v1 development databases are unsupported. For a schema change, add a new numbered revision under `migrations/versions/` — never edit committed public revisions.

## Architecture invariants

- **v1 is a single-user tool.** One active session per project. No auth, no RBAC, no multi-reviewer workflows — do not build them; remote access assumes a trusted network.
- **The backend is the single source of truth** for schema derivation, validation, evaluation scoring, prompt construction, run state machines, trigger detection, and pool routing. Never duplicate any of these in TypeScript.
- **Single-process lock per project** — one backend process per project database; no concurrent access to the same project DB.
- **REST is authoritative; SSE is a hint channel.** The frontend reconciles from REST on page load, reconnect, and terminal events.
- **Foreground priority dispatch** — interactive proposals hold new background HTTP dispatches; already in-flight background requests complete normally.
- **One-NIM-per-GPU (Spec §1.5).** At most one NIM container per GPU, regardless of what the VRAM math suggests. An exact compatible running Teacher is reused across projects without a second container or ownership transfer; fresh-project creation attaches and selects it before returning, even when a hosted API key is configured. A different resident is stopped only after the SME explicitly confirms replacement (`replace_resident=true` on `POST .../local_nim/deploy`); the FTUE names its model/project and offers Keep vs Stop-and-start. The Student NIM lifecycle defaults to replacement (§9.5.2, which auto-restores a displaced Teacher). Displacements are audited (§13.15). Owner state is durable in project SQLite and named Docker containers survive backend restart; recovery health-verifies/re-adopts them, and stop/failure disables all consumer endpoint attachments. Single-GPU hosts: local Teacher + pHash diversity, or Student benchmarking with Teacher displacement, or hybrid (local Teacher + hosted embeddings via `NVIDIA_API_KEY`), or — single GPU below every Teacher floor but ≥ the embedding floor (24 GB seeded) — local embeddings + hosted Teacher. Multi-GPU: deterministic lowest-free-index placement (embedding deploys skip free GPUs below the embedding floor); no co-location. Enforced in `services/local_nim_service.resolve_gpu_placement()` / `stop_gpu_residents()`; the FTUE's skip-embedding-when-a-Teacher-is-queued-on-a-single-GPU stays as defense-in-depth. Empirical motivation (Cosmos NIM's hardcoded `gpu_memory_utilization=0.9` profile floor) is documented in the README — measured on Cosmos Reason2 8B / A100-80GB (2026-05-19), re-confirmed on Cosmos 3 nano / H100 NVL (2026-07-13).

## Backend patterns

- **Config**: single flat Pydantic `Settings` cached with `@lru_cache` (Retail-Agentic-Commerce pattern), extended with a five-level precedence loader: process env > explicit `--env-file` / `VLM_FEEDBACK_LOOP_ENV_FILE` > `~/.vlm_feedback_loop/.env` > `~/.vlm_feedback_loop/config.yaml` > built-in defaults.
- **Startup**: FastAPI `@asynccontextmanager` lifespan (never `@app.on_event`).
- **Routers**: versioned `APIRouter(prefix="/v1")`, thin handlers. Middleware order: CORS → domain middleware → request logging (added bottom-to-top).
- **HTTP out**: `httpx.AsyncClient` for all outbound calls (NIM, TAO, embeddings); `asyncio.to_thread()` for unavoidable sync work. Every outbound NIM request includes the `{"source": "vlm-feedback-loop"}` header (Blueprint usage-tracking convention).
- **Errors**: structured JSON with specific codes — 503 connection errors, 504 timeouts, 400 validation, 409 conflicts.
- **Logging**: named loggers `vlm_feedback_loop.{module}`, structured JSON with contextual fields.
- **Docker**: multi-stage (deps → builder → runtime) from `python:3.12-alpine3.24`, `uv sync --frozen`, non-root user, healthcheck. Native build tools stay in the discarded builder stage; runtime carries only `libstdc++` for the locked AIPerf dependencies. Entry: `uvicorn` on `main:app`, port 8000.
- **Token counting**: `tiktoken` with `cl100k_base` fallback. **pHash**: DCT-based 64-bit.

## Frontend patterns

- **State**: Zustand for scoped client state; React Query (`@tanstack/react-query`) for server state. Typed fetch client functions in `api/`, interfaces in `types/` — no axios. Icons: `lucide-react`.
- **KUI**: vendored tgz referenced in `src/ui/package.json` as `"@kui/react": "./assets/kui-foundations-react-external-0.504.1.tgz"`; `index.css` imports `tailwindcss` then `@kui/react/base.css`; app wrapped in `<ThemeProvider theme="dark" density="standard">` with `nv-dark` on `<html>`.
- **Styling values**: radii 18px cards / 14px inner / 999px pills; rgba-white text at .92/.82/.62/.4; NVIDIA green `#76b900` (+opacity variants); `backdrop-filter: blur(20px) saturate(150%)` glass; 24px card padding; 140/200/300 ms animation tiers.

**KUI-first rules** — every new UI element starts from a KUI component; raw HTML is the exception. The decision framework:

1. **Action buttons** (Save, Retry, Export, …) → KUI `<Button>`: `kind="primary"` + `.nvidia-green-button` for the main CTA, `secondary`/`tertiary` otherwise. Never a raw `<button>` for anything a user would call a button.
2. **All displayed text** → KUI `<Text kind="...">`. No raw `<p>`/`<span>`/`<h*>`/text-only `<div>`.
3. **Form inputs inside glassmorphic cards** → raw `<input>`/`<textarea>`/`<select>` + `.glass-input`. KUI `<TextInput>`/`<TextArea>` render opaque backgrounds that break the glass aesthetic — Retail Agentic Commerce makes the same exception with its custom `StyledInput`.
4. **Toggle pills / segmented controls** (Output Stability, Thinking, Visual Budget) → KUI `<Button kind="tertiary">` with pill styling; reference implementations: `TopBarControls.tsx`, `RetryPanel.tsx`.
5. **Inline text affordances** (fix-it links in errors, "▸ Advanced" disclosures, file-tree controls) → raw `<button>` acceptable; KUI Button chrome would be wrong inside flowing content.
6. **Custom controls styled by `.glass-btn` / `.glass-pill`** (add-field buttons, chip remove `[x]`, tree checkboxes) → raw HTML acceptable; these have their own design language in `index.css`.
7. **Loading** → KUI `<Spinner>` only. **Dialogs** → KUI `<Modal>` with `slotHeading`/`slotFooter` only — the modal surface and its inputs are glass-themed globally in `index.css`, so `<FormField>`+`<TextInput>`+`<TextArea>` are correct *inside modals* (and in simple settings forms; not in dense builders). Dropdowns → KUI `<Select>`; status → `<Badge>`; substantive help → `<Tooltip>` (native `title` for lightweight hints).

Audit: grep `<button|<span|<p |<select|<input` in `.tsx` (excluding `__tests__`) — every hit must match a rule above, else convert to KUI.

## Compose patterns

nginx edge proxy in front of ui + backend (SSE buffering off, upload limits, timeouts); service names must match code references; `${VARIABLE:-default}` env syntax sourced from `.env`. The compose stack has no GPU or NIM services — NIM containers are runtime-orchestrated via `docker run` by `services/local_nim_service` (hardcoded `~/.cache/nim:/opt/nim/.cache` cache mount, `--shm-size=32GB`), and the containerized backend cannot orchestrate them at all (no docker CLI/socket by design): system-managed local NIMs require local-source mode.

## Visual verification — mandatory for UI work

Every UI change must be visually verified before it is reported complete: after CSS/styling changes, new pages, layout/structure changes, and before committing any UI work. Headless capture is installed (Playwright ≥1.58 with headless Chromium; `shot-scraper` for one-offs). No display server needed.



Review screenshots at their original dimensions with the highest-detail visual reviewer available; do not resample evidence before inspection.

| Viewport (dsf=1) | Use for |
|---|---|
| **2560×1440** | **Default** — native desktop full-screen layout review |
| **2560×1600** | Taller folds (three vertical sections in one frame) |
| **2000×1250** | Intentional narrower layout for content-dense forms/builders |

Do not shrink a capture or alter the browser viewport merely to fit a fixed image-token estimate. Keep `device_scale_factor=1` for model-consumed full screens. Crop at capture time when the question is local; this improves focus and lowers native input size without resampling the evidence.

```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    page = p.chromium.launch(headless=True).new_page(viewport={"width": 2560, "height": 1440}, device_scale_factor=1)
    page.goto("http://localhost:5173/...")
    page.wait_for_load_state("networkidle"); page.wait_for_timeout(2000)
    page.screenshot(path="/tmp/screenshots/check.png")
```

**Focused review workflow:**

1. Read the relevant Spec/Overview section as text first — zero image cost.
2. Capture everything to disk (`/tmp/screenshots/vlm/`), not into context.
3. Crop at capture time (`element.screenshot()` / `clip=`), never post-hoc — reading both a full frame and its crop double-bills.
4. Verbatim copy (labels, helper text, errors) is a DOM text-equality check in the capture script, not a pixel check. Vision is for layout, spacing, color, hierarchy.
5. When a configured visual reviewer is available, delegate focused visual
   slices to it with original-detail image loads; otherwise perform the same
   focused review directly.
6. Read ≤3 crops yourself, only those backing the highest-severity findings or a fix you'll apply; name the question each image answers before reading it.
7. Wireframes and specs can't prescribe case/tracking/density — write pros/cons before "fixing" a Blueprint-refined treatment to match ASCII art. If the code is more correct than the doc, fix the doc.
8. After a fix: typecheck, rerun the affected test file, re-capture only the changed state, read that one crop. No full re-sweeps.

Dev servers must run the latest code — kill stale ones before capturing. Screenshots under `/tmp/screenshots/` are ephemeral and never committed.

## Testing

When changing behavior: add/update backend unit tests for logic, API/integration tests for endpoint behavior, UI tests for critical interactions; keep contract tests aligned with backend behavior. Do not claim something works unless the relevant tests pass, and verify both delivery modes plus startup docs when you touch them.

**What NOT to test** — a test should fail when something the user cares about is wrong:

- **Pydantic field defaults.** `assert obj.foo == "bar"` on a fresh model tests your typing, not behavior; if a default has business meaning, test the consumer exercising it.
- **Mock-the-thing-under-test.** Mock collaborators at the boundary (HTTP, DB, time) — never patch the function you're verifying and assert on the mock. The wire-mocked `TestProfileBProductionPipeline` classes in `test_batch_label_service.py` / `test_evaluation_service.py` are the reference pattern.
- **Status-code-only assertions.** Pair `assert resp.status_code == 200` with at least one body, side-effect, or invariant assertion.
- **Milestone artifact tests.** Name tests after behavior (`test_chain_advance_after_train_succeeded`), put them in the steady-state file for the subsystem, and give the docstring a plain-language statement of the rule being pinned — plan/step numbering and spec-section citations rot; the docstring must stand alone.
- **Duplicated business-rule tests.** Parallel coverage of two *implementations* is correct; a byte-identical copy written without checking for the first is not — delete one.

When in doubt ask: "if this fails, what does it tell me about the system?" If the answer is "the setup broke" or "I typed a default wrong" — delete it.

## Configuration and runtime data

Three scopes, all outside the repo:

1. **`~/.vlm_feedback_loop/config.yaml`** — non-secret app settings (WORKSPACE_ROOT, LOG_LEVEL, BIND_HOST/BIND_PORT, IMAGE_ROOT, EMBEDDING_*, local NIM ports, default models). `IMAGE_ROOT` is one optional absolute directory: unset loopback development opens `/`; network-accessible deployments must set it; the Ingest screen obtains it by omitting the initial browse path. `ALLOW_UI_SECRET_PERSIST` (bool) gates whether the UI may persist pasted keys to `.env` — disable where `.env` is managed externally.
2. **`~/.vlm_feedback_loop/.env`** — deployment secrets: `NVIDIA_API_KEY`, `NGC_API_KEY`, `TAO_API_KEY` / `TAO_API_BASE_URL` / `TAO_ORG_NAME`, `TAO_WORKSPACE_S3_ACCESS_KEY` / `TAO_WORKSPACE_S3_SECRET_KEY`. TAO workspace *identity* lives in `deployment.db` (via `tao-bootstrap`), not `.env`.
3. **Compose vars** — `EDGE_PORT`; `IMAGE_ROOT` is shared with backend configuration and also drives the Compose image bind mount.

Runtime data lives under `{WORKSPACE_ROOT}/projects/{project_id}/` — `project.db`, `exports/`, `artifacts/`, `logs/`. Images are stored by reference (filesystem path), never copied in. The repo holds source, docs, fixtures, templates, migrations, and config samples — never commit generated runtime outputs.

## Documentation maintenance

Documentation is part of the Blueprint product surface. When architecture, API shape, repo structure, startup, deployment, env vars, or setup behavior changes: update the relevant `docs/` file, `README.md`, `.env.example` / `config.yaml.example`, and compose docs/launch helpers in the same change. Append a dated entry to `docs/changelog.md` for anything an outside developer would notice — the README carries no changelog. If code and docs disagree, fix the disagreement immediately. Keep the README's quick starts, config tables, and doc links accurate — it is the Blueprint's front door.

## Implementation style

Be practical and direct · simple, testable code · explicit APIs · boring, predictable file names · no speculative abstraction · clear boundaries over cleverness · nothing beyond what the current step requires · optimize for teachability and inspectability.

## Git

Commit when a logical unit is complete and validated; prefer fewer, better commits. Review `git status` and `git diff` first; stage only intended files. Clear, imperative messages; non-trivial commits get a short body: what changed, why, how validated. Never commit broken or unvalidated work; never rewrite committed migrations or push force without instruction.

## Validation

Report exactly what you ran and whether it passed. Never claim a feature works without verifying it; say clearly when something was not validated. When startup, packaging, or deployment behavior changes, state what was validated in local-source mode AND containerized mode.

## Priority

When in doubt, preserve in this order:

1. backend authority over domain logic
2. alignment with `Engineering_Spec.md` and the implemented screen contract
3. Blueprint usability for outside developers
4. repo clarity and predictable structure
5. testability
6. documentation accuracy

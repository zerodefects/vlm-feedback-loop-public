# Advanced Tests — Cross-Cutting Test Infrastructure

This document covers the cross-cutting test infrastructure the Blueprint ships — the CI pipeline and pre-commit hooks, secret scanning and the SonarQube delegate, and strict backend type checking. Feature-level tests live next to the features they prove (`tests/unit/`, `tests/integration/`, the vitest suites under `src/ui/`); this document describes the surfaces that cut across all of them.

Read this alongside:

- `docs/Engineering_Spec.md` — the normative implementation contract. Appendix B carries the acceptance / verify items for the sections here ("CI / pre-commit pipeline", "Secret scanning + SonarQube delegate", "Backend type checking").
- The **implemented React UI is the screen contract** — there is no separate UI specification. UI behavior is verified by the frontend test suite and the visual-verification workflow, not by a document.

The Spec is normative; this document is normative for **how** the cross-cutting tests are structured. If they conflict, the Spec wins.

---

## Methodology

Each section is described in the same shape so a developer can maintain it without inventing structure:

1. **Goal** — what behavior the category proves.
2. **Tooling** — concrete libraries / commands; cite versions where pinning matters.
3. **Layout** — file paths and directory conventions.
4. **Execution model** — local / CI / operator-driven; preconditions and runtime budgets.
5. **Acceptance criteria** — what counts as passing; cross-reference to the normative contract.
6. **Failure handling** — how regressions surface; what a developer does when a test fails.
7. **Maintenance** — how the test stays accurate as the product evolves.

CI and secret scanning are mandatory at every PR. Live validation runs on an operator schedule when its required credentials or infrastructure are available.

---

## 1 — CI Pipeline and Pre-Commit

**Goal.** Every PR gets the same automated lint + tests + typecheck + build that a release engineer would run locally, in under 10 minutes wall-clock for the core jobs. Commits that violate basic hygiene are blocked at the local pre-commit boundary.

**Tooling.**

- **GitHub Actions** for CI (`.github/workflows/ci.yml`). Node 20, Python 3.11/3.12/3.13, `uv`, `pnpm` set up via official actions (`astral-sh/setup-uv@v3`, `pnpm/action-setup@v4` with `version: 10`, `actions/setup-node@v4`).
- **pre-commit** for local hygiene (`.pre-commit-config.yaml`). Hooks in order, cheapest first: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `ruff` (lint with `--fix`) + `ruff-format`, `gitleaks` (secret scan, see section 2), and a local `agents-claude-twins` hook that runs `diff AGENTS.md CLAUDE.md` — the two agent-instruction files are deliberate identical twins, and nothing else enforces the empty-diff contract their headers declare. A top-level `exclude` keeps the hooks away from the internal engineering archive's append-only run-evidence artifacts, the dependency lockfiles (`uv.lock`, `src/ui/pnpm-lock.yaml`), and the vendored KUI tarball.
- **uv 0.5+** invoked as `uv sync --frozen` and `uv run pytest tests/unit/ -q --maxfail=5` (plus coverage flags, below).
- **pnpm 10** invoked as `pnpm install --frozen-lockfile`, `pnpm exec eslint .`, `pnpm run format:check`, `pnpm run typecheck`, `pnpm test`, `pnpm build`.

The pre-commit ruff `rev:` MUST track the project's resolved ruff minor (currently `v0.15.10`, against `pyproject.toml`'s `ruff>=0.9,<1.0` constraint) so pre-commit and CI agree on enabled rules — older ruff revs flag rules (e.g., UP038) that newer revs have demoted, producing fail-locally / pass-CI drift.

**Layout.**

```
.github/workflows/ci.yml
.github/workflows/sonarqube.yml         # SonarQube delegate, see section 2
.pre-commit-config.yaml
.gitleaks.toml                          # nvidia-api-key custom rule, see section 2
sonar-project.properties                # SonarQube project config, see section 2
scripts/ci-local.sh                     # local reproducer for the six core jobs
```

**CI workflow jobs (parallel):**

1. `backend-lint` — `uv run ruff check .`, `uv run ruff format --check .` (whole-repo scope, matching the pre-commit hook scope), plus the `diff AGENTS.md CLAUDE.md` twin check.
2. `backend-tests` — `uv run pytest tests/unit/ -q --maxfail=5 --cov=src/backend/vlm_feedback_loop --cov-report=term --cov-report=xml`, in a matrix over Python 3.11 / 3.12 / 3.13 (exercising the `requires-python >=3.11,<3.14` claim instead of asserting it on one interpreter; `UV_PYTHON` pins each matrix leg). The coverage gate is `[tool.coverage.report] fail_under = 80` in `pyproject.toml`. The `coverage.xml` artifact is CI-local — the SonarQube template runs its own pytest+coverage pass and does not consume it.
3. `frontend-lint` — `pnpm exec eslint .`, `pnpm run format:check` (Prettier), `pnpm run typecheck` (`tsc -b`) inside `src/ui`.
4. `frontend-tests` — `pnpm test` (the `test` script runs `vitest run`) inside `src/ui`.
5. `frontend-build` — `pnpm build` (`tsc -b && vite build`) inside `src/ui`.
6. `backend-typecheck` — `uv run pyright src/backend/` in strict mode (see section 3).
7. `dependency-audit` — advisory known-CVE backstop for public forks and mirrors, where the org-level SonarQube scan cannot run: exports the locked backend set with `uv export --frozen` and audits it with `uvx pip-audit --no-deps`, plus `pnpm audit --prod --audit-level high` in `src/ui`. Audits the LOCKED dependency sets only — no resolution, no installs.
8. `compose-smoke` — the containerized delivery mode is a first-class product surface, but none of the core jobs builds an image (a fully broken `docker compose up --build` once shipped with every check green). This job runs `docker compose build`, `docker compose up -d --no-build`, polls `http://localhost:3000/health` for up to two minutes, dumps `docker compose ps` and the last 100 log lines on failure, and always tears down with `docker compose down -v`.

Per-job `timeout-minutes: 10` caps any runaway; `compose-smoke` gets 20 to cover a cold image cache. Integration tests (`tests/integration/`) are intentionally NOT in CI scope — they require credentials (`NVIDIA_API_KEY` / `NGC_API_KEY` / `RUN_LIVE_TAO_FIXTURE`) or a live Docker daemon. Run them locally with `uv run pytest tests/integration/ -q -n 0` (serial only — they spawn live servers on fixed ports). The live smokes under `scripts/` are operator-run at the same boundary; in particular, run `uv run python scripts/schema_evolution_smoke.py` (live backend + `NVIDIA_API_KEY`) after touching guidance/schema-evolution code — it is the live check of the 8-step semantic-core Guidance transition.

**Execution model.** Every PR; every push to `main`; `workflow_dispatch` for manual reruns. No nightly schedule — the SonarQube workflow handles continuous quality scanning at the org level, and the secret-scan catch boundary is pre-commit.

**Acceptance criteria.** `docs/Engineering_Spec.md` Appendix B "CI / pre-commit pipeline" carries the verify items for the five original core jobs; the Appendix B "Backend type checking" section covers the `backend-typecheck` gate. `dependency-audit` and `compose-smoke` post-date those acceptance items — the workflow file is the source of truth for the current job set.

**Failure handling.** Failed jobs surface a red check on the PR; the merge button blocks until all jobs go green. `compose-smoke` failures include the container state and log tail in the job output.

**Maintenance.** The workflow file is the single source of truth — local dev reproduces it via `scripts/ci-local.sh`, which runs the six core stages in order with bit-for-bit identical commands. When the workflow's `run:` lines change, `ci-local.sh` MUST be updated in the same commit so the two artifacts cannot drift. `dependency-audit` (network-bound advisory lookups) and `compose-smoke` (Docker build) are deliberately not mirrored in the script — run `docker compose up --build` yourself when touching packaging.

---

## 2 — Secret Scanning + SonarQube Delegate

**Goal.** Two complementary surfaces: (a) catch secret leaks at the local pre-commit boundary before they reach origin, and (b) delegate SAST + supply-chain scanning to the org-managed SonarQube reusable workflow that the Retail Blueprints already use.

**Tooling.**

- **gitleaks** (pre-commit) — secret pattern scan run as the last third-party pre-commit hook. Pinned to a tagged release (`gitleaks/gitleaks` at `v8.21.4`). Custom rules live in `.gitleaks.toml` at the repo root.
- **SonarQube reusable workflow** (CI) — `NVIDIA-AI-Blueprints/sonarqube-workflows/.github/workflows/sonarqube-reusable-template.yml@main`, called from `.github/workflows/sonarqube.yml` with `secrets: inherit` and `language: python` (the template auto-detects `uv.lock` and runs its own pytest+coverage pass). Same delegate pattern Retail-Agentic-Commerce and Retail-Catalog-Enrichment use. The job is gated on `github.repository_owner == 'NVIDIA-AI-Blueprints'` because the reusable template is org-private — on forks and mirrors it skips cleanly instead of failing every push, and the `dependency-audit` CI job (section 1) is the known-CVE backstop there.
- **`sonar-project.properties`** — project config the scanner reads: sources rooted at the repo, exclusions for non-product trees (`docs/`, `scripts/`, `deploy/`, the vendored KUI tarball), backend-unit-coverage only (`sonar.python.coverage.reportPaths=coverage.xml`, with tests, migrations, and `src/ui/` excluded from coverage), and `sonar.qualitygate.wait=true` so the delegate job fails when the quality gate fails.

**Why gitleaks at the local boundary, not just in CI.** This is an open-source repo, and the documented minimal-setup credentials (README "Quick Start") are `NVIDIA_API_KEY` / `NGC_API_KEY` in `nvapi-...` format. The blast radius of a key reaching origin is high; pre-commit is the cheapest catch point. None of the three reference Blueprints runs gitleaks — but none of them ships hosted-NIM keys as the minimal-setup credential, which is why this repo takes the deliberate departure.

**Why a custom `nvidia-api-key` gitleaks rule.** Out of the box, gitleaks recognizes AWS / GCP / GitHub / Stripe / etc. credential patterns — but NOT NVIDIA's `nvapi-` format. Without `.gitleaks.toml`, the hook would pass on a leaked NVIDIA key. The custom rule matches `nvapi-` followed by 40–80 token characters (real keys are ~64 chars after the prefix; short test fixtures like `nvapi-test`, `nvapi-fake`, `nvapi-stub` stay under the 40-char floor and are additionally allowlisted in `[allowlist]` for clarity). The `[allowlist]` also exempts a by-name list of docs that legitimately discuss the credential *format* — by name, not the whole `docs/` tree, so new leaks can't auto-bypass the scanner.

**Scope discipline.** Per-PR SAST is delegated to SonarQube inside the org rather than re-implemented with per-tool pipelines (bandit, eslint-plugin-security, a CVE-suppression ledger) — each of those produces a false-positive backlog needing its own documented-suppression process, and the org's SonarQube configuration is the maintained answer. The one deliberate addition is the `dependency-audit` CI job: a narrow, locked-set-only known-CVE check that keeps public forks and mirrors covered where the org-private SonarQube template cannot run.

**Layout.**

```
.github/workflows/sonarqube.yml         # delegate to NVIDIA-AI-Blueprints/sonarqube-workflows
sonar-project.properties                # scanner config: exclusions, coverage, quality gate
.gitleaks.toml                          # extends defaults with nvidia-api-key rule
.pre-commit-config.yaml                 # gitleaks hook
```

**Execution model.**

- **Pre-commit (local):** `gitleaks` runs on every commit attempt. Sub-second on most diffs. The hook scans staged content only (not full history); the cost of a full-history scan vs the marginal value at this gate is unfavorable. Run the repo-wide smoke with `pre-commit run --all-files`.
- **CI:** `sonarqube.yml` triggers on `push` to `main`, `pull_request` to `main`, and `workflow_dispatch` — inside the org only, per the owner gate above.

**Acceptance criteria.** `docs/Engineering_Spec.md` Appendix B "Secret scanning + SonarQube delegate" — including the planted-credential test that the `nvidia-api-key` rule fires on a staged `nvapi-<60 token chars>` value. (Appendix B's negative-coverage item pre-dates the `dependency-audit` backstop job; the deliberate scope is as described above.)

**Failure handling.** A pre-commit gitleaks finding blocks the local commit; the developer either redacts the secret or, if it's a false positive, extends `[allowlist]` in `.gitleaks.toml` with a justifying inline comment. SonarQube findings flow through the org's SonarQube project page — suppressions live in the SonarQube project config, not in this repo.

**Maintenance.** The `nvidia-api-key` rule's regex tracks NVIDIA's published key format. If NVIDIA changes the format (e.g., a different prefix), update `.gitleaks.toml` AND verify the planted-secret check still fires. Keep the `[allowlist]` path list by-name — never widen it to a directory glob.

---

## 3 — Backend Type Checking

**Goal.** Keep the backend at pyright strict-mode parity with the frontend's TypeScript strict mode. Refactor risk grows quadratically without it.

**Tooling.**

- **`pyright`** via `uv run pyright src/backend/`, pinned as `pyright>=1.1.408` in the dev dependency group. Mirrors `Retail-Agentic-Commerce`'s precedent — the only one of the three reference NVIDIA Blueprints that runs strict-mode static type checking in CI as a hard gate. No mypy, no debt ledger, no stub packages.

**Layout.**

```
pyproject.toml                          # [tool.pyright] block
src/backend/vlm_feedback_loop/py.typed  # PEP 561 marker
```

The `[tool.pyright]` block sets `typeCheckingMode = "strict"`, `pythonVersion = "3.12"`, `include = ["src/backend"]`, and excludes `**/__pycache__`, `**/node_modules/**`, and `src/backend/vlm_feedback_loop/migrations/**` (migrations are imperative side-effect modules validated by end-to-end DB behavior, not static types).

**Execution model.** CI (the `backend-typecheck` job, a required gate with no `continue-on-error`) and locally via `uv run pyright src/backend/`. ~30 s on a warm cache. Must stay at 0 errors / 0 warnings.

**Acceptance criteria.** `docs/Engineering_Spec.md` Appendix B "Backend type checking".

**Failure handling.** Type errors block the build. Fix the type, don't relax the rule.

**Maintenance.** Every suppression MUST carry the bracketed rule code (`# pyright: ignore[code]`) so a future re-probe can check whether the underlying issue is fixed upstream. The footprint stays deliberately small — on the order of 20 suppressions across `src/backend/`, mirroring Retail-Agentic-Commerce's — and any PR that grows it needs a reason in review.

---

## Acceptance cross-reference

The acceptance items for these surfaces live in `docs/Engineering_Spec.md`:

- **CI pipeline + pre-commit** — Appendix B, "CI / pre-commit pipeline".
- **Secret scanning + SonarQube delegate** — Appendix B, "Secret scanning + SonarQube delegate".
- **Backend type checking** — Appendix B, "Backend type checking".

The acceptance items are the unit of completion; this document is the methodology. The Spec wins on conflict.

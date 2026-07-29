# Public Release Validation Handoff

**Validation date:** 2026-07-29
**Public mirror:** <https://github.com/zerodefects/vlm-feedback-loop-public>
**Release profile:** public v1 baseline, AutoRun excluded (default)

## Verdict

The public snapshot passed release-readiness, offline structure/link,
secret/private-reference, anonymous-acquisition, local-source runtime, and
containerized runtime validation. The tested public repository is
anonymously readable and its default branch is `main`.

This validates the Blueprint's installation, first-time labeling loop, release
shape, and two delivery modes. It does not claim that a live GPU/NIM topology,
live TAO cluster, or domain-specific production model was validated in this
CPU-hosted release pass.

## Artifact shape

The final snapshot contains:

- 642 total files after adding this validation record;
- 407 files under `src/`;
- 158 backend source-tree files and 249 UI source-tree files;
- 383 Python/TypeScript/TSX/CSS/HTML source files under `src/`;
- exactly one migration, `001_public_v1_schema.py`;
- exactly 15 licensed bundled sample images; and
- no AutoRun CLI module, private evidence, private agent configuration,
  research tree, or private migration history.

## Export and offline validation

Commands:

```bash
uv run python scripts/export_public_snapshot.py \
  ~/release-vlm-feedback-loop --dry-run

uv run python scripts/export_public_snapshot.py \
  ~/release-vlm-feedback-loop --replace-existing

uv run python scripts/validate_public_release.py \
  ~/release-vlm-feedback-loop
```

Results:

- clean dry run: passed with zero readiness issues;
- real export without `--allow-unready`: passed;
- AutoRun: excluded by default;
- public structure and all relative Markdown/image links: passed;
- public v1 migration and bundled-image counts: passed.

The exporter starts from a committed `git archive`; ignored files, working
tree state, runtime databases, virtual environments, and `node_modules` are
not copied.

## Secret and private-reference validation

Command, run in the exported public Git checkout:

```bash
uv run pre-commit run --all-files
```

All hooks passed:

- trailing whitespace;
- end-of-file normalization;
- YAML validation;
- Ruff lint;
- Ruff format;
- pinned gitleaks hardcoded-secret detection; and
- the `AGENTS.md` / `CLAUDE.md` twin check.

A separate case-insensitive scan found no shipping references to the private
GitLab host, internal TAO organization/topology, private evidence tree,
personal Git identity, or internal preservation branch. Test fixtures use the
neutral TAO organization `example-org`.

## Anonymous acquisition

Ambient GitHub credentials were disabled for these commands:

```bash
GIT_TERMINAL_PROMPT=0 GIT_CONFIG_GLOBAL=/dev/null \
  git -c credential.helper= ls-remote \
  https://github.com/zerodefects/vlm-feedback-loop-public.git \
  HEAD refs/heads/main

GIT_TERMINAL_PROMPT=0 GIT_CONFIG_GLOBAL=/dev/null \
  git -c credential.helper= clone \
  https://github.com/zerodefects/vlm-feedback-loop-public.git \
  /tmp/vlm-public-anonymous/vlm-feedback-loop-public

python3 scripts/validate_public_release.py \
  /tmp/vlm-public-anonymous/vlm-feedback-loop-public
```

Results:

- unauthenticated `ls-remote`: passed;
- unauthenticated clone: passed;
- GitHub repository page: HTTP 200 without authentication;
- validator and local-link check against the clone: passed; and
- the three vendored Nebula distribution files were present in the public Git
  tree and survived the clone.

## Automated test and static-analysis evidence

Commands and results:

```bash
uv run pytest tests/unit/ -q
# 2745 passed

uv run pytest tests/integration/ -q -n 0
# 21 passed, 6 skipped (environment-gated live integrations)

uv run ruff check .
# passed

uv run ruff format --check .
# passed

uv run pyright src/backend/
# 0 errors, 0 warnings, 0 informations

cd src/ui
pnpm test
# 68 files passed; 720 tests passed

pnpm typecheck
# passed

pnpm lint
# passed with 0 errors and 5 existing react-refresh warnings

pnpm build
# passed; Vite reported only its non-blocking >500 kB chunk warning

pnpm test:e2e
# 1 Playwright FTUE journey passed
```

The Playwright journey covers Create Project, the bundled-sample shortcut,
ingestion, the RPS Guidance preset, a genuine correction of a deliberately
wrong mocked proposal for a known image, and confirmation that the next
proposal reports one Verified Edit in ICL context. It also asserts that the
15-image sample is not represented as Scale-Up-ready.

## Local-source quick-start validation

The public commit was cloned into a newly created OS user with an empty home
directory. Python 3.12.13, Node 20.20.2, and the repository-pinned pnpm 10.33.0
were used.

Commands:

```bash
uv sync
cd src/ui && pnpm install --frozen-lockfile && cd ../..
uv run vlm-feedback-loop init --workspace-root ~/vlm-workspace
./scripts/dev.sh

curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:5173/v1/projects
```

Results:

- dependencies installed from the public lockfiles;
- configuration, secret skeleton, and `deployment.db` initialized;
- backend reached application-startup completion;
- Vite reached ready state;
- backend health returned `{"status":"ok"}`;
- the proxied empty project list returned successfully;
- the root UI, `main.tsx`, global CSS, Nebula component, and vendored Nebula
  JS/CSS assets all transformed and loaded without errors; and
- `Ctrl+C` stopped both processes cleanly.

## Containerized quick-start validation

The following ran from the authentication-disabled anonymous clone, not from
the source repository:

```bash
docker compose config --quiet
docker compose build
docker compose up -d --wait --wait-timeout 120
docker compose ps
curl --fail http://127.0.0.1:3000/health
curl --fail http://127.0.0.1:3000/
docker compose down
```

Results on Docker Engine 29.6.2 and Compose 5.3.1:

- Compose model: valid;
- backend image: built;
- UI production image: built, including the vendored Nebula distribution;
- backend, UI, and nginx: started healthy;
- nginx published only `127.0.0.1:3000`;
- edge health returned `{"status":"ok"}`;
- edge root returned the production UI HTML; and
- the stack and project network stopped cleanly while preserving the named
  project-data volume.

The host already had a `vlm-feedback-loop-workspace` volume from an earlier
validation, so Compose correctly warned that it belonged to a prior project
name and then reused the explicitly named persistence volume. No volume or
project data was deleted.

## Visual review

The UI inspector captured and reviewed:

- RPS Guidance selection and its `rock`, `paper`, `scissors` Core enum;
- Student Training with a visually distinct default “Validate training setup”
  intent;
- the advanced candidate-comparison intent; and
- the structured TAO Setup Action Request.

The reviewed RPS Guidance capture is committed as
[`images/ftue-rps-guidance.png`](images/ftue-rps-guidance.png) and displayed
near the README opening.

## Explicitly outside this release pass

- live hosted Teacher inference with a real user API key;
- local Teacher, embedding, or Student NIM deployment on a GPU;
- live TAO workspace provisioning or four-job training execution;
- domain-specific model-quality claims; and
- operation of a permanent production service.

Those are environment- or workload-specific validations. The Blueprint's
production boundary remains quality validation, temporary serving validation,
and a generated handoff to the infrastructure team that owns production.

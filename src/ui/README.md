# Frontend (React 19 + TypeScript + Vite)

Orientation for the UI half of the VLM Feedback Loop. Product docs live in `../../docs/`.

## Vendored packages

Two dependencies are not published to public npm and are vendored into this repo:

- **KUI Foundations** — `"@kui/react"` in `package.json` resolves to the local tarball
  `./assets/kui-foundations-react-external-0.504.1.tgz`. To upgrade: drop the new tgz into
  `assets/`, update the path (and version in the filename) in `package.json`, then run
  `pnpm install`.
- **`@kui-contrib/nebula`** (`src/components/nebula/`) — the animated background used by
  `NebulaBg.tsx` and imported as CSS in `index.css`. Vendored as a prebuilt `dist/` for the
  same reason: it is an NVIDIA-internal KUI-contrib package with no public npm release.
  Provenance: `@kui-contrib/nebula` 1.3.9, prebuilt dist copied verbatim; its `package.json`
  ships without a license field.

## Build and dev

Requires Node >= 20 and pnpm >= 9.

```bash
pnpm install
../../scripts/dev.sh        # from repo root: backend :8000 + Vite dev server :5173
pnpm typecheck && pnpm lint && pnpm test && pnpm build
```

The Vite dev server proxies `/v1` (including the SSE events path) and `/health` to the
backend at `http://localhost:8000`; set `VITE_BACKEND_URL` to point elsewhere.

## State management

- **React Query** (`@tanstack/react-query`) owns all server state.
- **Zustand** is used for exactly one store: the SSE connection (`src/stores/sse-store.ts`).
- Everything else is component-local state.
- API access goes through typed fetch clients in `src/api/`, with response interfaces in
  `src/types/` — no axios.

## Component conventions

New UI starts from a KUI component — see the KUI-first rules in the repo-root `AGENTS.md`.

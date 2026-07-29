<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Description

<!-- What does this PR change, and why? Link related issues. -->

## Checklist

- [ ] I have read the [CONTRIBUTING](../CONTRIBUTING.md) guidelines.
- [ ] New or changed behavior is covered by tests, and the relevant suites pass
      (`uv run pytest tests/unit/ -q`, `cd src/ui && pnpm test`).
- [ ] Lint and type checks pass (`uv run ruff check .`, `uv run pyright src/backend/`,
      `cd src/ui && pnpm typecheck && pnpm lint`).
- [ ] User-visible changes have a dated entry in `docs/changelog.md`.
- [ ] Changes touching startup, packaging, networking, or deployment were verified
      in BOTH delivery modes (local source `./scripts/dev.sh` and containerized
      `docker compose up --build`), with docs updated in the same change.

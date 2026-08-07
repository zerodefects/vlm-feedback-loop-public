# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance tests for the frontend application shell.

The shell must satisfy six criteria:
  1. React + TypeScript + Vite app builds without errors and serves static assets.
  2. KUI Foundations component primitives render with NVIDIA design language.
     Tailwind utility classes are available for layout.
  3. A single command starts both frontend dev server and backend.
  4. Vite dev server proxies /v1/ API routes and the SSE events path to the
     FastAPI backend origin. API calls reach the backend without CORS errors.
  5. The dev proxy supports SSE passthrough — an EventSource connection stays
     open and receives events without premature closure or buffering.
  6. In production (same-origin or reverse proxy), no dev proxy or CORS
     configuration is required; the app functions without either.

The proxy/SSE tests start real backend and frontend processes and make
HTTP requests; the build and launch-script tests run `pnpm build` /
inspect committed files without servers. None of them are unit tests.

Server fixtures (backend_server, frontend_server) are provided by
tests/integration/conftest.py.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "src" / "ui"

BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5173"


# ---------------------------------------------------------------------------
# Criterion 1: App builds without errors and serves static assets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built_dist() -> Path:
    """Run ``pnpm build`` once for this module and return the dist/ path.

    Building inside a fixture (instead of relying on the build test having
    run earlier in the same session) means every dist-inspection test
    asserts against a bundle built from the current source — under ``-k``
    selection or after a failed build there is no stale-dist window where
    e.g. the hardcoded-localhost check passes against an old bundle.
    """
    result = subprocess.run(
        ["pnpm", "build"],
        cwd=str(FRONTEND_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"pnpm build failed:\n{result.stdout}\n{result.stderr}")
    return FRONTEND_DIR / "dist"


class TestAppBuilds:
    """Verify: React + TypeScript + Vite app builds without errors."""

    @pytest.mark.timeout(180)
    def test_pnpm_build_succeeds(self, built_dist: Path):
        """pnpm build exits 0 (asserted inside built_dist) and produces dist/."""
        assert built_dist.is_dir()

    def test_dist_contains_index_html(self, built_dist: Path):
        """Build output includes index.html."""
        assert (built_dist / "index.html").is_file(), (
            "dist/index.html not found after build"
        )

    def test_dist_contains_assets(self, built_dist: Path):
        """Build output includes CSS and JS bundles."""
        assets = built_dist / "assets"
        css_files = list(assets.glob("*.css"))
        js_files = list(assets.glob("*.js"))
        assert len(css_files) >= 1, "No CSS bundle in dist/assets"
        assert len(js_files) >= 1, "No JS bundle in dist/assets"


# ---------------------------------------------------------------------------
# Criterion 2: KUI + Tailwind render with NVIDIA design language
# ---------------------------------------------------------------------------


class TestKuiTailwindRender:
    """Verify the two documented KUI-integration invariants (CLAUDE.md):
    the ``nv-dark`` theme class on ``<html>`` and the Tailwind-before-KUI
    import order in ``index.css``. Which token names / utility spellings
    land in the built bundles is an implementation detail of the KUI and
    Tailwind versions — the old substring greps on the bundles and page
    source broke on upgrades without telling us anything was wrong."""

    def test_index_html_has_nv_dark(self):
        """index.html sets nv-dark class on <html> for NVIDIA dark theme."""
        html = (FRONTEND_DIR / "index.html").read_text()
        assert 'class="nv-dark"' in html

    def test_css_imports_kui_base(self):
        """index.css imports @kui/react/base.css after tailwind."""
        css = (FRONTEND_DIR / "src" / "index.css").read_text()
        tw_pos = css.find("tailwindcss")
        kui_pos = css.find("@kui/react/base.css")
        assert tw_pos >= 0, "Tailwind import not found"
        assert kui_pos > tw_pos, "KUI base.css must be imported after Tailwind"

    def test_native_glass_choices_restore_visible_keyboard_focus_after_reset(self):
        """Checkboxes and radios retain a visible focus ring after form resets."""
        css = (FRONTEND_DIR / "src" / "index.css").read_text()
        base_pos = css.find('.glass-input[type="checkbox"],')
        focus_pos = css.find('.glass-input[type="checkbox"]:focus-visible,')
        assert base_pos >= 0, "Native glass choice styling not found"
        assert focus_pos > base_pos, "Choice focus styling must follow the native reset"
        focus_rule_end = css.find("}", focus_pos)
        focus_rule = css[focus_pos:focus_rule_end]
        assert "outline: 2px solid var(--accent-green);" in focus_rule
        assert "outline-offset: 2px;" in focus_rule


# ---------------------------------------------------------------------------
# Criterion 3: Single-command launch
# ---------------------------------------------------------------------------


class TestSingleCommandLaunch:
    """Verify: scripts/dev.sh exists and is executable."""

    def test_dev_script_exists(self):
        script = REPO_ROOT / "scripts" / "dev.sh"
        assert script.is_file(), "scripts/dev.sh not found"

    def test_dev_script_is_executable(self):
        script = REPO_ROOT / "scripts" / "dev.sh"
        assert os.access(script, os.X_OK), "scripts/dev.sh is not executable"

    def test_dev_script_references_both_servers(self):
        """dev.sh starts both backend and frontend."""
        script = (REPO_ROOT / "scripts" / "dev.sh").read_text()
        assert "vlm_feedback_loop" in script, "Script should start the backend"
        assert "pnpm dev" in script, "Script should start the frontend"

    def test_dev_script_propagates_child_failure_and_configured_port(
        self, tmp_path: Path
    ):
        """A failed child stops the launcher and Vite sees the backend port."""
        real_uv = shutil.which("uv")
        assert real_uv is not None

        config_dir = tmp_path / ".vlm_feedback_loop"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text(
            f"WORKSPACE_ROOT: {tmp_path / 'workspace'}\nBIND_PORT: 8123\n"
        )

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_uv = fake_bin / "uv"
        fake_uv.write_text(
            "#!/usr/bin/env bash\n"
            'case " $* " in\n'
            '  *" --print-backend-url "*) exec "$REAL_UV" "$@" ;;\n'
            "  *)\n"
            "    for _ in $(seq 1 100); do\n"
            '      [ -f "$FRONTEND_READY" ] && exit 7\n'
            "      sleep 0.05\n"
            "    done\n"
            "    exit 9\n"
            "    ;;\n"
            "esac\n"
        )
        fake_uv.chmod(0o755)

        proxy_target = tmp_path / "vite-target.txt"
        fake_pnpm = fake_bin / "pnpm"
        fake_pnpm.write_text(
            "#!/usr/bin/env bash\n"
            'printf \'%s\' "${VITE_BACKEND_URL:-missing}" > "$PROXY_TARGET"\n'
            'printf \'%s\' "$$" > "$FRONTEND_READY"\n'
            "exec node -e 'setInterval(() => {}, 1000)' \"$FAKE_VITE_ARG\"\n"
        )
        fake_pnpm.chmod(0o755)

        env = os.environ.copy()
        env.update(
            HOME=str(tmp_path),
            PATH=f"{fake_bin}{os.pathsep}{env['PATH']}",
            REAL_UV=real_uv,
            PROXY_TARGET=str(proxy_target),
            FRONTEND_READY=str(tmp_path / "frontend-ready.txt"),
            FAKE_VITE_ARG=str(FRONTEND_DIR / "node_modules" / ".bin" / "vite"),
        )
        result = subprocess.run(
            [str(REPO_ROOT / "scripts" / "dev.sh")],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode == 7, result.stdout + result.stderr
        assert proxy_target.read_text() == "http://127.0.0.1:8123"
        frontend_pid = (tmp_path / "frontend-ready.txt").read_text()
        assert not Path(f"/proc/{frontend_pid}").exists()

    def test_vite_refuses_to_hide_a_busy_documented_port(self):
        """The dev server fails instead of silently moving away from 5173."""
        config = (FRONTEND_DIR / "vite.config.ts").read_text()
        assert "strictPort: true" in _config_object_block(config, "server")

    def test_backend_cleanup_is_scoped_to_source_checkout(self, tmp_path: Path):
        """A same-named backend outside this checkout is never reaped."""
        external_cwd = tmp_path / "compose-app"
        external_cwd.mkdir()
        repo_process = subprocess.Popen(
            ["vlm_feedback_loop.main", "30"],
            executable="sleep",
            cwd=REPO_ROOT,
        )
        external_process = subprocess.Popen(
            ["vlm_feedback_loop.main", "30"],
            executable="sleep",
            cwd=external_cwd,
        )
        try:
            command = (
                f"REPO_ROOT={REPO_ROOT!s}; "
                f"source {REPO_ROOT / 'scripts' / 'dev-processes.sh'}; "
                "_vlm_dev_backend_pids"
            )
            result = subprocess.run(
                ["bash", "-c", command],
                check=True,
                capture_output=True,
                text=True,
            )
            selected = {int(pid) for pid in result.stdout.split()}
            assert repo_process.pid in selected
            assert external_process.pid not in selected
        finally:
            repo_process.terminate()
            external_process.terminate()
            repo_process.wait(timeout=5)
            external_process.wait(timeout=5)


# ---------------------------------------------------------------------------
# Criterion 4: Dev proxy forwards /v1/ API routes and SSE to backend
# ---------------------------------------------------------------------------


class TestDevProxy:
    """Verify: Vite dev proxy forwards API and SSE to backend."""

    def test_health_via_proxy(self, frontend_server):  # noqa: ARG002
        """GET /health through the Vite proxy reaches the backend."""
        resp = httpx.get(f"{FRONTEND_URL}/health", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_api_v1_projects_via_proxy(self, frontend_server):  # noqa: ARG002
        """GET /v1/projects through the Vite proxy reaches the backend."""
        resp = httpx.get(f"{FRONTEND_URL}/v1/projects", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    def test_api_v1_post_via_proxy(self, frontend_server):  # noqa: ARG002
        """POST /v1/projects through the proxy reaches backend (validation)."""
        # Missing required 'name' field → expect 422 from backend
        resp = httpx.post(
            f"{FRONTEND_URL}/v1/projects",
            json={},
            timeout=5,
        )
        assert resp.status_code == 422, (
            f"Expected 422 (validation), got {resp.status_code}"
        )
        # The validation error is the backend's (proves the proxy reached
        # it), and it names the missing field.
        errors = resp.json()["detail"]
        assert any("name" in err.get("loc", []) for err in errors), errors


# ---------------------------------------------------------------------------
# Criterion 5: SSE passthrough stays open
# ---------------------------------------------------------------------------


class TestSsePassthrough:
    """Verify that the backend exposes the SSE route used by proxy tests."""

    def test_sse_endpoint_exists_on_backend(self, backend_server):  # noqa: ARG002
        """Backend has the SSE events endpoint (returns 404 for non-existent project)."""
        resp = httpx.get(
            f"{BACKEND_URL}/v1/projects/00000000-0000-4000-8000-000000000000/events",
            timeout=5,
        )
        # 404 because the project doesn't exist, but the endpoint is routed
        # (a missing ROUTE would be FastAPI's generic "Not Found" body).
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"


# ---------------------------------------------------------------------------
# Criterion 6: Production same-origin works
# ---------------------------------------------------------------------------


def _config_object_block(source: str, key: str) -> str:
    """Return the balanced-brace object literal assigned to ``key`` in
    a TS config source.

    A plain substring slice up to the next ``}`` silently truncates once
    the block grows past the slice window, so this walks the braces.
    """
    m = re.search(rf"\b{re.escape(key)}:\s*{{", source)
    assert m is not None, f"{key} block not found"
    depth = 0
    for i in range(m.end() - 1, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[m.end() - 1 : i + 1]
    raise AssertionError(f"unbalanced braces in {key} block")


class TestProductionSameOrigin:
    """Verify: built app works without dev proxy in same-origin setup."""

    def test_built_assets_are_static(self, built_dist: Path):
        """Built dist/ is a static SPA — no server-side rendering required."""
        index = (built_dist / "index.html").read_text()
        assert '<div id="root">' in index
        assert "<script" in index

    def test_no_hardcoded_localhost_in_built_js(self, built_dist: Path):
        """Built JS does not hardcode localhost URLs that would break same-origin."""
        # The backend origin must NOT be baked into the built JS in ANY
        # host spelling — `localhost:8000` alone misses the equally
        # broken `127.0.0.1:8000` / `0.0.0.0:8000` variants. API calls
        # should use relative paths (e.g., /v1/projects) which resolve
        # to the same origin in production.
        backend_origin = re.compile(r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0):8000")
        js_files = list((built_dist / "assets").glob("*.js"))
        assert js_files, "No JS bundle in dist/assets"
        for js_file in js_files:
            content = js_file.read_text()
            match = backend_origin.search(content)
            assert match is None, (
                f"Built JS contains hardcoded backend origin "
                f"{match.group(0)!r} in {js_file.name}"
            )

    def test_vite_config_no_proxy_in_preview(self):
        """vite.config.ts preview section has no proxy (same-origin in prod)."""
        config = (FRONTEND_DIR / "vite.config.ts").read_text()
        # Positive control: the dev server block DOES define the proxy,
        # so a broken block extractor cannot pass the preview assertion
        # vacuously.
        assert "proxy" in _config_object_block(config, "server")
        assert "proxy" not in _config_object_block(config, "preview"), (
            "Preview mode should not define proxy rules"
        )

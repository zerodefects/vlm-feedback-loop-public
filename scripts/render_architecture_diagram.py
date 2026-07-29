#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render docs/images/architecture.mmd to docs/images/architecture.png.

Uses the repo's existing headless-Chromium toolchain (Playwright) with a
pinned Mermaid release from the jsdelivr CDN — no extra system packages.
Network access is needed only when regenerating the PNG after editing the
.mmd source; the rendered PNG is committed.

Usage (from the repo root):

    python3 scripts/render_architecture_diagram.py   # system python3 (Playwright is a dev-box tool, not a project dep)
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "docs" / "images" / "architecture.mmd"
OUTPUT = REPO_ROOT / "docs" / "images" / "architecture.png"
MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.12.1/dist/mermaid.min.js"

# NVIDIA-dark styling consistent with the app's design language.
MERMAID_CONFIG = {
    "startOnLoad": False,
    "theme": "base",
    "themeVariables": {
        "background": "#111111",
        "primaryColor": "#1c1c1c",
        "primaryTextColor": "rgba(255,255,255,0.92)",
        "primaryBorderColor": "#76b900",
        "lineColor": "rgba(255,255,255,0.62)",
        "secondaryColor": "#161616",
        "tertiaryColor": "#161616",
        "clusterBkg": "#161616",
        "clusterBorder": "rgba(255,255,255,0.25)",
        "edgeLabelBackground": "#111111",
        "fontFamily": "DejaVu Sans, Helvetica, Arial, sans-serif",
        "fontSize": "15px",
    },
    "flowchart": {
        "curve": "basis",
        "nodeSpacing": 46,
        "rankSpacing": 64,
        "wrappingWidth": 300,
    },
}

PAGE_HTML = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <script src="{MERMAID_URL}"></script>
    <style>
      body {{ margin: 0; background: #111111; }}
      #diagram {{ display: inline-block; padding: 28px; }}
    </style>
  </head>
  <body>
    <div id="diagram"></div>
    <script>
      window.renderDone = false;
      mermaid.initialize({json.dumps(MERMAID_CONFIG)});
      window.render = async (source) => {{
        const {{ svg }} = await mermaid.render("arch", source);
        const holder = document.getElementById("diagram");
        holder.innerHTML = svg;
        // Pin the SVG to its natural viewBox size — mermaid's default
        // max-width/width:100% collapses it inside an inline-block parent.
        const el = holder.querySelector("svg");
        const vb = el.viewBox.baseVal;
        el.style.maxWidth = "none";
        el.setAttribute("width", vb.width + "px");
        el.setAttribute("height", vb.height + "px");
        window.renderDone = true;
      }};
    </script>
  </body>
</html>"""


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": 1600, "height": 1200}, device_scale_factor=1
        )
        page.set_content(PAGE_HTML, wait_until="networkidle")
        page.evaluate("source => window.render(source)", source)
        page.wait_for_function("window.renderDone === true")
        page.locator("#diagram").screenshot(path=str(OUTPUT))
        browser.close()
    from PIL import Image

    size = Image.open(OUTPUT).size
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({size[0]}x{size[1]})")
    if max(size) > 2576:
        print("note: long edge exceeds 2576px — consider tightening the layout")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the root LICENSE-3rd-party.txt from the locked dependency tree.

Aggregates third-party license information for:

- Python packages installed in ``.venv`` (the ``uv sync --frozen`` environment,
  including dev tools — matching the sibling retail Blueprints, which list
  their build/test toolchain too), read from each package's ``*.dist-info``.
- Frontend packages installed in ``src/ui/node_modules``, read via
  ``pnpm licenses list --json --long``.

The vendored ``@kui/*`` design-system tarball is NVIDIA first-party software,
not third-party, and is deliberately excluded (the NVIDIA RAG Blueprint's
manifest makes the same call). Its open-source dependencies are installed
through pnpm and therefore appear in the frontend section on their own.

Output follows the grouped format of the retail reference Blueprints: for each
license type, the packages under it with their copyright holders, then the
full license text once (taken verbatim from the first listed package that
ships one). Output is deterministic — regenerating without dependency changes
must produce a byte-identical file.

Usage (from the repo root, after ``uv sync`` and ``pnpm install`` in src/ui):

    uv run python scripts/generate_third_party_licenses.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "LICENSE-3rd-party.txt"
FIRST_PARTY_DISTS = {"vlm-feedback-loop", "vlm_feedback_loop"}
FIRST_PARTY_JS_PREFIXES = ("@kui/",)

LICENSE_FILE_PATTERNS = ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*")
CLASSIFIER_MAP = {
    "OSI Approved :: MIT License": "MIT",
    "OSI Approved :: Apache Software License": "Apache-2.0",
    "OSI Approved :: BSD License": "BSD",
    "OSI Approved :: ISC License (ISCL)": "ISC",
    "OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
    "OSI Approved :: Zope Public License": "ZPL-2.1",
}


@dataclass
class Package:
    name: str
    version: str
    ecosystem: str  # "python" | "npm"
    license_label: str
    copyright_line: str = ""
    license_text: str = ""


@dataclass
class Group:
    label: str
    packages: list[Package] = field(default_factory=list)


def _first_copyright_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#/*- ").rstrip()
        if re.match(r"(?i)^copyright\b", stripped) and len(stripped) <= 160:
            return stripped
    return ""


def _normalize_text(raw: bytes) -> str:
    return (
        raw.decode("utf-8", errors="replace")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


def _python_license_label(metadata: str) -> str:
    expr = re.search(r"^License-Expression:\s*(.+)$", metadata, re.MULTILINE)
    if expr:
        return expr.group(1).strip()
    classifiers = re.findall(
        r"^Classifier:\s*License\s*::\s*(.+)$", metadata, re.MULTILINE
    )
    for classifier in classifiers:
        mapped = CLASSIFIER_MAP.get(classifier.strip())
        if mapped:
            return mapped
    lic = re.search(r"^License:\s*(.+)$", metadata, re.MULTILINE)
    if lic and len(lic.group(1).strip()) <= 60:
        return lic.group(1).strip()
    if classifiers:
        return classifiers[0].strip()
    return "Unknown"


def _collect_license_files(dist_info: Path) -> str:
    texts: list[str] = []
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in LICENSE_FILE_PATTERNS:
        candidates.extend(sorted(dist_info.glob(pattern)))
    licenses_dir = dist_info / "licenses"
    if licenses_dir.is_dir():
        candidates.extend(sorted(p for p in licenses_dir.rglob("*") if p.is_file()))
    for path in candidates:
        if path in seen or path.is_dir():
            continue
        seen.add(path)
        texts.append(_normalize_text(path.read_bytes()))
    return "\n\n".join(texts)


def collect_python_packages() -> list[Package]:
    site_packages_dirs = sorted(REPO_ROOT.glob(".venv/lib/python3.*/site-packages"))
    if not site_packages_dirs:
        sys.exit("No .venv site-packages found — run `uv sync` first.")
    packages: list[Package] = []
    for dist_info in sorted(site_packages_dirs[0].glob("*.dist-info")):
        metadata_file = dist_info / "METADATA"
        if not metadata_file.exists():
            continue
        metadata = _normalize_text(metadata_file.read_bytes())
        name_m = re.search(r"^Name:\s*(.+)$", metadata, re.MULTILINE)
        version_m = re.search(r"^Version:\s*(.+)$", metadata, re.MULTILINE)
        if not name_m:
            continue
        name = name_m.group(1).strip()
        if name in FIRST_PARTY_DISTS:
            continue
        text = _collect_license_files(dist_info)
        packages.append(
            Package(
                name=name,
                version=version_m.group(1).strip() if version_m else "",
                ecosystem="python",
                license_label=_python_license_label(metadata),
                copyright_line=_first_copyright_line(text),
                license_text=text,
            )
        )
    return packages


def _js_license_text(package_path: Path) -> str:
    texts: list[str] = []
    for pattern in LICENSE_FILE_PATTERNS:
        for path in sorted(package_path.glob(pattern)):
            if path.is_file():
                texts.append(_normalize_text(path.read_bytes()))
    return "\n\n".join(texts)


def collect_js_packages() -> list[Package]:
    result = subprocess.run(
        ["pnpm", "licenses", "list", "--json", "--long"],
        cwd=REPO_ROOT / "src" / "ui",
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    packages: list[Package] = []
    for license_label, entries in data.items():
        for entry in entries:
            name = entry["name"]
            if name.startswith(FIRST_PARTY_JS_PREFIXES):
                continue
            text = ""
            for path_str in entry.get("paths") or []:
                text = _js_license_text(Path(path_str))
                if text:
                    break
            packages.append(
                Package(
                    name=name,
                    version=", ".join(entry.get("versions") or []),
                    ecosystem="npm",
                    license_label=license_label,
                    copyright_line=_first_copyright_line(text),
                    license_text=text,
                )
            )
    return packages


def build_output(packages: list[Package]) -> str:
    groups: dict[str, Group] = {}
    for pkg in packages:
        groups.setdefault(pkg.license_label, Group(pkg.license_label)).packages.append(
            pkg
        )
    for group in groups.values():
        group.packages.sort(key=lambda p: (p.name.lower(), p.ecosystem))

    lines: list[str] = [
        "Interactive VLM Feedback Loop — Third-Party Licenses",
        "=" * 60,
        "",
        "This file lists third-party license information and copyright notices",
        "for the software packages this project installs (Python packages from",
        "uv.lock and frontend packages from src/ui/pnpm-lock.yaml, including",
        "the development toolchain). For each license type, the packages",
        "distributed under it are listed with their copyright holders, followed",
        "by the full license text as shipped by the first listed package that",
        "includes one.",
        "",
        "The vendored @kui/* design-system package is NVIDIA first-party",
        "software (not third-party) and is not listed here; its open-source",
        "dependencies are installed via pnpm and appear below on their own.",
        "",
        "Generated by scripts/generate_third_party_licenses.py — do not edit",
        "by hand; rerun the script after dependency changes.",
        "",
    ]
    for label in sorted(groups, key=str.lower):
        group = groups[label]
        lines.append("-" * 60)
        lines.append(f"License: {label}")
        lines.append("-" * 60)
        lines.append("")
        for pkg in group.packages:
            suffix = " (frontend)" if pkg.ecosystem == "npm" else ""
            lines.append(f"  {pkg.name} {pkg.version}{suffix}".rstrip())
            if pkg.copyright_line:
                lines.append(f"    {pkg.copyright_line}")
        lines.append("")
        representative = next((p for p in group.packages if p.license_text), None)
        if representative:
            lines.append(f"Full license text (as shipped with {representative.name}):")
            lines.append("")
            lines.append(representative.license_text)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    packages = collect_python_packages() + collect_js_packages()
    OUTPUT.write_text(build_output(packages), encoding="utf-8")
    py_count = sum(1 for p in packages if p.ecosystem == "python")
    print(
        f"Wrote {OUTPUT.name}: {py_count} Python + {len(packages) - py_count} "
        f"frontend packages, {OUTPUT.stat().st_size:,} bytes"
    )


if __name__ == "__main__":
    main()

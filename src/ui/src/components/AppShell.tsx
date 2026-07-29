// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { Text } from "@kui/react";
import { ExternalLink } from "lucide-react";

import { NebulaBg } from "@/components/NebulaBg";
import { LocalDeployBanner } from "@/components/LocalDeployBanner";

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const location = useLocation();
  const isHome = location.pathname === "/";

  return (
    <div className="relative min-h-screen">
      {/* Nebula animated background (fixed, z-0) */}
      <NebulaBg />

      {/* Content layer sits above Nebula */}
      <div className="relative flex min-h-screen flex-col" style={{ zIndex: 1 }}>
        <header
          className="sticky top-0 z-50 flex h-14 items-center border-b px-6"
          style={{
            borderColor: "var(--glass-border)",
            backgroundColor: "rgba(12, 12, 12, 0.8)",
            backdropFilter: "blur(20px) saturate(150%)",
            WebkitBackdropFilter: "blur(20px) saturate(150%)",
          }}
        >
          <Link
            to="/"
            style={{ textDecoration: "none" }}
            className="inline-flex items-center gap-3"
          >
            <img
              src="/logo.png"
              alt="NVIDIA"
              width={32}
              height={32}
              className="object-contain"
            />
            <Text
              kind="title/sm"
              style={{ color: "var(--text-primary)", letterSpacing: "-0.01em" }}
            >
              VLM Feedback Loop
            </Text>
          </Link>
          {!isHome && (
            <nav className="ml-6 flex items-center gap-1">
              <Link to="/" className={`nav-link ${isHome ? "nav-link-active" : ""}`}>
                Projects
              </Link>
            </nav>
          )}
          {/* Right cluster. Pages render into the portal slot via
              <HeaderRightPortal> (e.g. the Project List's Create Project CTA).
              The Docs link is a persistent identity affordance — outside
              developers landing on this Blueprint expect a "what is this?"
              exit (mirrors the top-right utility/status rhythm of both
              Retail Blueprints). It sits at the right edge so portal'd CTAs
              appear to its left. */}
          <div className="ml-auto flex items-center gap-3">
            <div id="header-right-slot" className="flex items-center gap-3" />
            <a
              href="https://github.com/zerodefects/vlm-feedback-loop-public#documentation"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-link"
              title="View the public Blueprint documentation"
            >
              Docs
              <ExternalLink size={12} aria-hidden="true" />
            </a>
          </div>
        </header>
        {/* Local-deploy status banner: renders only on
            project-scoped routes when at least one LocalNimDeployment
            is in ``starting`` status. Self-collapses on the Project
            List screen and on the FTUE pages where it would compete
            with their own affordances. */}
        <LocalDeployBanner />
        <main className="flex flex-1 flex-col">{children}</main>
      </div>
    </div>
  );
}

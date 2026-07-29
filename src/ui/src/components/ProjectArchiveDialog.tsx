// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Archive / Restore Project confirmation dialog.
 *
 * One component serves both directions via ``variant`` — the mutation
 * shell, cancel handling, error parsing, and Modal skeleton are
 * identical; only the API call, the copy, and the archive-side
 * busy-gate ``reasons`` list differ. Mirrors the CreateProjectDialog
 * pattern: KUI Modal with two slot buttons and inline error rendering.
 * The mutation lives here rather than on ProjectListPage so the page
 * stays focused on layout.
 */

import { Button, Modal, Text } from "@kui/react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { parseApiErrorDetail, parseApiErrorReasons } from "@/api/client";
import { archiveProject, unarchiveProject } from "@/api/projects";
import { projectKeys } from "@/api/query-keys";

type ArchiveVariant = "archive" | "unarchive";

interface VariantCopy {
  mutate: (projectId: string) => Promise<unknown>;
  heading: string;
  body: (projectName: string) => string;
  ctaLabel: string;
  pendingLabel: string;
  /** Archive failures carry busy-gate ``reasons``; restore's rare
   *  409 ``not_archived`` does not. */
  showReasons: boolean;
}

const VARIANTS: Record<ArchiveVariant, VariantCopy> = {
  archive: {
    // Bound lazily: VARIANTS evaluates at module scope, and a direct
    // reference would dereference the API bindings on import — every
    // test that transitively imports this dialog would then have to
    // stub them.
    mutate: (projectId) => archiveProject(projectId),
    heading: "Archive project?",
    body: (name) => `Archive “${name}”? You can restore it from this list later.`,
    ctaLabel: "Archive",
    pendingLabel: "Archiving…",
    showReasons: true,
  },
  unarchive: {
    mutate: (projectId) => unarchiveProject(projectId),
    heading: "Restore project?",
    body: (name) => `Restore “${name}” to the active list?`,
    ctaLabel: "Restore",
    pendingLabel: "Restoring…",
    showReasons: false,
  },
};

export interface ProjectArchiveDialogProps {
  variant: ArchiveVariant;
  open: boolean;
  projectId: string;
  projectName: string;
  onClose: () => void;
  onDone?: () => void;
}

export function ProjectArchiveDialog({
  variant,
  open,
  projectId,
  projectName,
  onClose,
  onDone,
}: ProjectArchiveDialogProps) {
  const queryClient = useQueryClient();
  const copy = VARIANTS[variant];

  const mutation = useMutation({
    mutationFn: () => copy.mutate(projectId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: projectKeys.all });
      onDone?.();
      onClose();
      mutation.reset();
    },
  });

  function handleCancel() {
    mutation.reset();
    onClose();
  }

  // Surface the failure inline so the SME knows why the action failed
  // without opening the network panel — archive's busy-gate 409 body
  // carries a ``reasons`` list rendered as bullets.
  const errorMessage = mutation.error
    ? (parseApiErrorDetail(mutation.error) ?? mutation.error.message)
    : null;
  const reasons = copy.showReasons ? parseApiErrorReasons(mutation.error) : [];

  return (
    <Modal
      open={open}
      onOpenChange={(isOpen) => {
        if (!isOpen) handleCancel();
      }}
      slotHeading={copy.heading}
      slotFooter={
        <>
          <Button kind="secondary" onClick={handleCancel}>
            Cancel
          </Button>
          <Button
            kind="primary"
            className="nvidia-green-button"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
          >
            {mutation.isPending ? copy.pendingLabel : copy.ctaLabel}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
          {copy.body(projectName)}
        </Text>
        {errorMessage && (
          <div className="rounded-lg px-4 py-3 toast-error" role="alert">
            <Text kind="body/regular/sm">{errorMessage}</Text>
            {reasons.length > 0 && (
              <ul className="ml-5 mt-2 list-disc">
                {reasons.map((reason) => (
                  <li key={reason}>
                    <Text kind="body/regular/sm">{reason}</Text>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </Modal>
  );
}

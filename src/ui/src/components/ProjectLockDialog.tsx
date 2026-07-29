// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Project Lock Error dialog.
 *
 * Shown when the backend returns 409 Conflict because the project
 * is already open in another process.  Single [OK] button; no
 * override or force-open path in v1.
 */

import { Button, Modal, Text } from "@kui/react";

interface ProjectLockDialogProps {
  open: boolean;
  onClose: () => void;
}

export function ProjectLockDialog({ open, onClose }: ProjectLockDialogProps) {
  return (
    <Modal
      open={open}
      onOpenChange={(isOpen) => {
        if (!isOpen) onClose();
      }}
      dismissible={false}
      slotHeading="Project In Use"
      slotFooter={
        <Button kind="primary" className="nvidia-green-button" onClick={onClose}>
          OK
        </Button>
      }
    >
      <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
        This project is already open in another process. Close the other process and try
        again.
      </Text>
    </Modal>
  );
}

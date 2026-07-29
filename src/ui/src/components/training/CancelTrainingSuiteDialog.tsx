// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Button, Modal, Text } from "@kui/react";
import { useMutation } from "@tanstack/react-query";

import { parseApiErrorDetail } from "@/api/client";
import { cancelTrainingSuite } from "@/api/training";
import type { TrainingSuiteCancelResponse } from "@/types/training";

export interface CancelTrainingSuiteDialogProps {
  open: boolean;
  projectId: string;
  trainingSuiteId: string;
  onClose: () => void;
  onCanceled: (result: TrainingSuiteCancelResponse) => void;
}

export function CancelTrainingSuiteDialog({
  open,
  projectId,
  trainingSuiteId,
  onClose,
  onCanceled,
}: CancelTrainingSuiteDialogProps) {
  const mutation = useMutation({
    mutationFn: () => cancelTrainingSuite(projectId, trainingSuiteId),
    onSuccess: (result) => {
      onCanceled(result);
      onClose();
      mutation.reset();
    },
  });

  function handleClose() {
    if (mutation.isPending) return;
    mutation.reset();
    onClose();
  }

  const errorMessage = mutation.error
    ? (parseApiErrorDetail(mutation.error) ?? mutation.error.message)
    : null;

  return (
    <Modal
      open={open}
      onOpenChange={(isOpen) => {
        if (!isOpen) handleClose();
      }}
      slotHeading="Cancel remaining jobs?"
      slotFooter={
        <>
          <Button kind="secondary" onClick={handleClose} disabled={mutation.isPending}>
            Keep Training
          </Button>
          <Button
            kind="primary"
            className="nvidia-green-button"
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            data-testid="confirm-cancel-training-suite"
          >
            {mutation.isPending ? "Canceling…" : "Cancel Jobs"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <Text kind="body/regular/sm" style={{ color: "var(--text-primary)" }}>
          The Blueprint will ask TAO to cancel every remaining job in this training run.
          Completed work is preserved.
        </Text>
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          If TAO cannot confirm a cancellation, the project will still be released
          locally and the failure will be recorded.
        </Text>
        {errorMessage && (
          <div className="rounded-lg px-4 py-3 toast-error" role="alert">
            <Text kind="body/regular/sm">{errorMessage}</Text>
          </div>
        )}
      </div>
    </Modal>
  );
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Create Project dialog.
 *
 * Modal overlay with Name (required) and Description (optional).
 * Validates locally before calling the backend.
 * On success navigates to the new project.
 */

import { useState, type KeyboardEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Modal, TextArea, TextInput, Text, FormField } from "@kui/react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { parseApiErrorDetail } from "@/api/client";
import { createProject } from "@/api/projects";
import { projectKeys } from "@/api/query-keys";

interface CreateProjectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CreateProjectDialog({ open, onOpenChange }: CreateProjectDialogProps) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [nameError, setNameError] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: createProject,
    onSuccess: (project) => {
      void queryClient.invalidateQueries({ queryKey: projectKeys.all });
      onOpenChange(false);
      resetForm();
      navigate(`/projects/${project.project_id}`);
    },
  });

  function resetForm() {
    setName("");
    setDescription("");
    setNameError(null);
    mutation.reset();
  }

  function handleSubmit() {
    if (mutation.isPending) return;

    const trimmed = name.trim();
    if (!trimmed) {
      setNameError("Project name is required.");
      return;
    }
    setNameError(null);
    mutation.mutate({
      name: trimmed,
      description: description.trim() || null,
    });
  }

  function handleFieldKeyDown(
    event: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }
    event.preventDefault();
    handleSubmit();
  }

  function handleCancel() {
    // A dispatched create cannot be canceled safely. Keep the modal open
    // until it resolves so a late success cannot navigate after the SME was
    // told the action had been canceled.
    if (mutation.isPending) return;
    onOpenChange(false);
    resetForm();
  }

  return (
    <Modal
      open={open}
      onOpenChange={(isOpen) => {
        if (!isOpen) handleCancel();
      }}
      dismissible={!mutation.isPending}
      slotHeading="Create Project"
      slotFooter={
        <>
          <Button kind="secondary" onClick={handleCancel} disabled={mutation.isPending}>
            Cancel
          </Button>
          <Button
            kind="primary"
            className="nvidia-green-button"
            onClick={handleSubmit}
            disabled={mutation.isPending}
          >
            {mutation.isPending ? "Creating..." : "Create Project"}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <FormField
          slotLabel="Name"
          required
          slotError={nameError ?? undefined}
          status={nameError ? "error" : undefined}
        >
          <TextInput
            placeholder="e.g., Damage Inspection"
            value={name}
            onValueChange={(v) => {
              setName(v);
              if (nameError) setNameError(null);
            }}
            attributes={{ TextInputValue: { onKeyDown: handleFieldKeyDown } }}
            status={nameError ? "error" : undefined}
          />
        </FormField>

        <FormField slotLabel="Description">
          <TextArea
            placeholder="e.g., Surface damage classification for manufacturing QA"
            value={description}
            onValueChange={setDescription}
            resizeable="auto"
            attributes={{
              TextAreaElement: { rows: 3, onKeyDown: handleFieldKeyDown },
            }}
          />
        </FormField>

        {mutation.isError && (
          <Text
            kind="body/regular/sm"
            className="text-error"
            role="alert"
            data-testid="create-project-error"
          >
            {parseApiErrorDetail(mutation.error) ??
              "Could not create the project. Check that the backend is running and try again."}
          </Text>
        )}
      </div>
    </Modal>
  );
}

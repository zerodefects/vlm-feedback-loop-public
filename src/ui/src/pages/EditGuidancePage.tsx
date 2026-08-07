// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Edit Guidance screen.
 *
 * Same layout as Create minus template selector. Adds label-invalidation
 * markers, post-save banner, confirmation dialog.
 */

import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Button, Modal, Text } from "@kui/react";
import { Info } from "lucide-react";

import { useSetupContext } from "@/pages/setup-context";
import { scrollToFirstError } from "@/lib/scroll-to-first-error";
import type { EditPreviewResponse } from "@/types/guidance";
import {
  fetchGuidance,
  editGuidancePreview,
  editGuidanceExecute,
} from "@/api/guidance";
import { updateProject } from "@/api/model-configs";
import { projectKeys, guidanceKeys } from "@/api/query-keys";
import {
  stampClientIds,
  stripClientIds,
  responseFieldsToInput,
  describeSemanticChanges,
  MarkerIcon,
  DescriptionCard,
  SchemaCard,
  RulesCard,
  Previews,
  ValidationNotices,
  useGuidanceForm,
} from "@/components/guidance";
import { GuidanceEditorLayout } from "@/components/guidance/GuidanceEditorLayout";
import { makeFieldHandlers } from "@/components/guidance/field-handlers";

// ── Main component ──────────────────────────────────────────────────────────

export function EditGuidancePage() {
  const { projectId, project } = useSetupContext();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const scrollBodyRef = useRef<HTMLDivElement>(null);

  // Load existing guidance
  const guidanceId = project.active_guidance_id;
  const { data: guidance, isLoading } = useQuery({
    queryKey: guidanceKeys.detail(projectId, guidanceId ?? ""),
    queryFn: () => fetchGuidance(projectId, guidanceId!),
    enabled: !!guidanceId,
  });

  const [initialized, setInitialized] = useState(false);
  const [saveToast, setSaveToast] = useState<string | null>(null);
  const [confirmDialog, setConfirmDialog] = useState<EditPreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // Shared form hook — initialized with empty state, populated via effect
  const form = useGuidanceForm({
    projectId,
    initialDescription: "",
    initialFields: [],
    initialRules: "",
  });

  // Populate form from loaded guidance
  useEffect(() => {
    if (guidance && !initialized) {
      form.setDescription(guidance.description);
      form.setFields(stampClientIds(responseFieldsToInput(guidance.schema_fields)));
      form.setRules(guidance.rules);
      setInitialized(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [guidance, initialized]);

  // The marker tooltip is "Changing this invalidates your {N}
  // verified labels. They return to Unlabeled for re-labeling." {N} is the
  // verified-label count. The dialog's preview surfaces a fresh count after
  // a save attempt; before that, the project's ambient counts.verified is
  // the right source — no extra fetch required (already on useSetupContext).
  // When there are no verified labels yet, fall back to a forward-looking
  // hint since "0 verified labels" reads awkwardly.
  const verifiedCount = confirmDialog?.verified_count ?? project.counts.verified;
  const markerTooltip =
    verifiedCount > 0
      ? `Changing this invalidates your ${verifiedCount} verified labels. They return to Unlabeled for re-labeling.`
      : "Changing this may invalidate existing labels.";

  // Save flow: preview → confirm → execute
  const executeMutation = useMutation({
    mutationFn: async () => {
      const result = await editGuidanceExecute(projectId, {
        description: form.description,
        schema: stripClientIds(form.fields),
        rules: form.rules,
      });
      if (result.guidance)
        await updateProject(projectId, {
          active_guidance_id: result.guidance.guidance_id,
        });
      return result;
    },
    onSuccess: (result) => {
      void queryClient.invalidateQueries({ queryKey: projectKeys.detail(projectId) });
      void queryClient.invalidateQueries({ queryKey: guidanceKeys.all(projectId) });
      setConfirmDialog(null);
      setSaveToast(`Guidance v${result.guidance?.version_number} saved.`);
      setTimeout(() => navigate(-1), 600);
    },
  });

  async function handleSave() {
    form.markSaveAttempted();
    // The backend is the sole validator: re-validate the exact draft
    // being saved and gate on the fresh verdict, never on possibly-stale
    // debounced state.
    const ok = await form.validateNow();
    if (!ok) {
      scrollToFirstError(scrollBodyRef);
      return;
    }
    setPreviewError(null);
    try {
      const preview = await editGuidancePreview(projectId, {
        description: form.description,
        schema: stripClientIds(form.fields),
        rules: form.rules,
      });
      if (preview.edit_type === "no_change" || preview.edit_type === "in_place") {
        executeMutation.mutate();
      } else {
        setConfirmDialog(preview);
      }
    } catch (err) {
      // The preview is the ONLY gate that classifies a semantic (label-
      // invalidating) edit and raises the confirmation dialog. If it fails,
      // surface the error and STOP — never fall through to execute, which
      // would silently return every Verified label to Unlabeled with no
      // warning.
      setPreviewError(
        err instanceof Error
          ? err.message
          : "Could not check what this edit affects. Try again.",
      );
    }
  }

  const fieldHandlers = makeFieldHandlers(form);

  // Loading states
  if (!guidanceId)
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="glass-card glass-card--elevated flex flex-col items-center gap-4 p-8 text-center">
          <Text kind="title/sm" style={{ color: "var(--text-primary)" }}>
            No active Guidance
          </Text>
          <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
            Create and activate Guidance before editing it.
          </Text>
          <Button
            kind="primary"
            className="nvidia-green-button"
            onClick={() => navigate(`/projects/${projectId}/create-guidance`)}
          >
            Create Guidance
          </Button>
        </div>
      </div>
    );
  if (isLoading || !initialized)
    return (
      <div className="flex flex-1 items-center justify-center">
        <Text kind="body/regular/sm" style={{ color: "var(--text-muted)" }}>
          Loading guidance...
        </Text>
      </div>
    );

  return (
    <GuidanceEditorLayout
      testId="edit-guidance-page"
      title="Edit Guidance"
      subtitle={
        <>
          v{guidance?.version_number} · Project: {project.name}
        </>
      }
      form={form}
      scrollBodyRef={scrollBodyRef}
      badgeSaveAttempted={form.saveAttempted}
      saveToast={saveToast}
      saveErrorMessage={executeMutation.isError ? "Failed to save guidance." : null}
      footerJustifyClass="justify-end gap-3"
      footerSlot={
        <>
          <Button kind="secondary" onClick={() => navigate(-1)}>
            Cancel
          </Button>
          <Button
            kind="primary"
            className="nvidia-green-button"
            disabled={
              (form.saveAttempted && form.totalErrorCount > 0) ||
              executeMutation.isPending
            }
            onClick={handleSave}
            aria-disabled={
              (form.saveAttempted && form.totalErrorCount > 0) ||
              executeMutation.isPending
            }
            data-testid="save-guidance-btn"
          >
            {executeMutation.isPending ? "Saving..." : "Save Guidance"}
          </Button>
        </>
      }
      dialogSlot={
        /* Confirmation dialog for semantic Core changes */
        confirmDialog && (
          <Modal
            open={!!confirmDialog}
            onOpenChange={(open) => {
              if (!open) setConfirmDialog(null);
            }}
            dismissible
            slotHeading={<Text kind="title/sm">Update schema and re-label?</Text>}
            slotFooter={
              <div className="flex justify-end gap-3">
                <Button kind="secondary" onClick={() => setConfirmDialog(null)}>
                  Cancel
                </Button>
                <Button
                  kind="primary"
                  className="nvidia-green-button"
                  onClick={() => executeMutation.mutate()}
                  disabled={executeMutation.isPending}
                  data-testid="confirm-update-btn"
                >
                  {executeMutation.isPending ? "Updating..." : "Update and Re-label"}
                </Button>
              </div>
            }
          >
            <div
              className="flex flex-col gap-3 text-sm"
              style={{ color: "var(--text-secondary)" }}
              data-testid="confirm-dialog-body"
            >
              <Text
                kind="body/regular/sm"
                style={{ color: "var(--text-secondary)", display: "block" }}
              >
                {describeSemanticChanges(confirmDialog.changes) && (
                  <Text
                    kind="body/regular/sm"
                    style={{ color: "inherit" }}
                    data-testid="confirm-change-description"
                  >
                    {describeSemanticChanges(confirmDialog.changes)}{" "}
                  </Text>
                )}
                Your {confirmDialog.verified_count} labeled images will return to
                Unlabeled for re-labeling under the new schema.
              </Text>
              {confirmDialog.auto_labeled_count > 0 && (
                <Text
                  kind="body/regular/sm"
                  style={{ color: "var(--text-secondary)", display: "block" }}
                  data-testid="confirm-auto-labeled-text"
                >
                  This will also revert {confirmDialog.auto_labeled_count} Auto-Labeled
                  examples to Unlabeled. You can re-run Batch Labeling when ready. Your
                  improved Guidance and ICL examples carry over to the new run.
                </Text>
              )}
              <div className="mt-1">
                <Text
                  kind="body/semibold/sm"
                  style={{ color: "var(--text-primary)", display: "block" }}
                >
                  What happens next:
                </Text>
                <ul className="mt-1 list-disc pl-5 space-y-1">
                  <li>
                    <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                      Your prior labels are preserved as read-only reference.
                    </Text>
                  </li>
                  <li>
                    <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                      The model re-proposes labels under the new schema.
                    </Text>
                  </li>
                  <li>
                    <Text kind="body/regular/sm" style={{ color: "inherit" }}>
                      Prior edits are reviewed first to rebuild context quickly.
                    </Text>
                  </li>
                </ul>
              </div>
            </div>
          </Modal>
        )
      }
    >
      {previewError && (
        <div className="rounded-lg px-4 py-3 toast-error" data-testid="preview-error">
          <Text kind="body/regular/sm" style={{ color: "inherit" }}>
            {previewError} Your labels were not changed.
          </Text>
        </div>
      )}

      {/* Post-save banner (edit-only) */}
      <div
        className="rounded-lg px-4 py-3 glass-info flex items-start gap-2.5"
        data-testid="post-save-banner"
      >
        <Info
          size={16}
          style={{ color: "var(--text-muted)", flexShrink: 0, marginTop: 2 }}
          aria-hidden
        />
        <Text kind="body/regular/sm" style={{ color: "inherit" }}>
          Renames apply directly. Changing what a correct answer looks like invalidates
          labels and returns them to Unlabeled. The model re-proposes labels under the
          new schema and shows your prior labels as reference. Look for the{" "}
          <MarkerIcon tooltip={markerTooltip} /> on controls that trigger it.
        </Text>
      </div>

      <DescriptionCard
        description={form.description}
        onChange={form.handleDescriptionChange}
      />

      <SchemaCard
        coreFields={form.coreFields}
        auxFields={form.auxFields}
        allIssues={form.displayIssues}
        pathPrefixes={form.pathPrefixes}
        handlers={fieldHandlers}
        rationaleEnabled={form.rationaleEnabled}
        onRationaleEnabledChange={form.handleRationaleEnabledChange}
        showMarkers={true}
        markerTooltip={markerTooltip}
        backendErrorSlot={
          <ValidationNotices issues={form.issues} backendError={form.backendError} />
        }
      />

      <RulesCard rules={form.rules} onChange={form.handleRulesChange} />
      <Previews
        fields={form.fields}
        backendValidation={form.backendValidation}
        errorCount={form.totalErrorCount}
      />
    </GuidanceEditorLayout>
  );
}

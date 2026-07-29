// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Create Guidance screen.
 */

import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button, Text, Select, Modal } from "@kui/react";
import { Info } from "lucide-react";

import { useSetupContext } from "@/pages/ProjectSetupLayout";
import { scrollToFirstError } from "@/lib/scroll-to-first-error";
import { createGuidance } from "@/api/guidance";
import { updateProject } from "@/api/model-configs";
import { projectKeys } from "@/api/query-keys";
import {
  GUIDANCE_TEMPLATES,
  DEFAULT_TEMPLATE_NAME,
  getTemplateByName,
  type TemplateName,
} from "@/lib/guidance-templates";
import {
  stampClientIds,
  stripClientIds,
  DescriptionCard,
  SchemaCard,
  RulesCard,
  Previews,
  ValidationNotices,
  useGuidanceForm,
} from "@/components/guidance";
import { GuidanceEditorLayout } from "@/components/guidance/GuidanceEditorLayout";
import { makeFieldHandlers } from "@/components/guidance/field-handlers";

export function CreateGuidancePage() {
  const { projectId, project } = useSetupContext();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const descriptionRef = useRef<HTMLTextAreaElement>(null);
  const scrollBodyRef = useRef<HTMLDivElement>(null);

  // Template state (create-only)
  //
  // - selectedTemplate: what the dropdown displays.
  // - appliedTemplate: the template whose description/fields are actually in
  //   the form. These can diverge briefly while the confirmation modal is
  //   open (user picked a new one but hasn't confirmed the replace yet).
  // - pendingTemplate: non-null iff the replace-work confirmation modal is
  //   open. Set when the user picks a different template after making edits.
  const [selectedTemplate, setSelectedTemplate] =
    useState<TemplateName>(DEFAULT_TEMPLATE_NAME);
  const [appliedTemplate, setAppliedTemplate] =
    useState<TemplateName>(DEFAULT_TEMPLATE_NAME);
  const [pendingTemplate, setPendingTemplate] = useState<TemplateName | null>(null);
  const [hasUserEdited, setHasUserEdited] = useState(false);
  const [saveToast, setSaveToast] = useState<string | null>(null);

  // Shared form hook
  const form = useGuidanceForm({
    projectId,
    initialDescription: "",
    initialFields: stampClientIds(getTemplateByName("blank").fields),
    initialRules: "",
    onEdited: () => {
      if (!hasUserEdited) setHasUserEdited(true);
    },
  });

  // Focus description on mount
  useEffect(() => {
    descriptionRef.current?.focus();
  }, []);

  // Apply a template's description + fields to the form. Applying a template
  // is NOT a user edit — the form is effectively reset to a "template-fresh"
  // state, so hasUserEdited is cleared. The state change itself re-arms the
  // debounced backend validation for the new content.
  function applyTemplate(name: TemplateName) {
    const template = getTemplateByName(name);
    form.setDescription(template.description);
    form.setFields(stampClientIds(template.fields));
    setHasUserEdited(false);
    setAppliedTemplate(name);
  }

  // Template handler (create-only). If the SME has made real edits, ask
  // before replacing their work; otherwise apply immediately.
  function handleTemplateChange(name: TemplateName) {
    setSelectedTemplate(name);
    if (name === appliedTemplate) return;
    if (!hasUserEdited) {
      applyTemplate(name);
    } else {
      setPendingTemplate(name);
    }
  }

  function handleConfirmReplace() {
    if (pendingTemplate) {
      applyTemplate(pendingTemplate);
      setPendingTemplate(null);
    }
  }

  function handleCancelReplace() {
    setSelectedTemplate(appliedTemplate);
    setPendingTemplate(null);
  }

  // Save mutation
  const saveMutation = useMutation({
    mutationFn: async () => {
      const result = await createGuidance(projectId, {
        description: form.description,
        schema: stripClientIds(form.fields),
        rules: form.rules,
      });
      await updateProject(projectId, { active_guidance_id: result.guidance_id });
      return result;
    },
    onSuccess: (result) => {
      void queryClient.invalidateQueries({ queryKey: projectKeys.detail(projectId) });
      setSaveToast(`Guidance v${result.version_number} saved.`);
      // Saving Guidance completes the onboarding chain — go straight to
      // Labeling. The Save toast
      // above remains visible through the 600 ms delay so the SME sees
      // "Guidance v1 saved." before the screen transitions.
      setTimeout(() => {
        navigate("../labeling");
      }, 600);
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
    saveMutation.mutate();
  }

  // Field handlers for SchemaCard
  const fieldHandlers = makeFieldHandlers(form);

  return (
    <GuidanceEditorLayout
      testId="create-guidance-page"
      title="Create Guidance"
      subtitle={<>Project: {project.name}</>}
      form={form}
      scrollBodyRef={scrollBodyRef}
      badgeSaveAttempted={form.saveAttempted}
      ariaLiveTestId="aria-live-region"
      saveToast={saveToast}
      saveErrorMessage={
        saveMutation.isError
          ? `Failed to save guidance. ${(saveMutation.error as Error)?.message ?? ""}`
          : null
      }
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
              (form.saveAttempted && form.totalErrorCount > 0) || saveMutation.isPending
            }
            onClick={handleSave}
            aria-disabled={
              (form.saveAttempted && form.totalErrorCount > 0) || saveMutation.isPending
            }
            data-testid="save-guidance-btn"
          >
            {saveMutation.isPending ? "Saving..." : "Save Guidance"}
          </Button>
        </>
      }
      dialogSlot={
        /* Confirm template replace — protects user edits. */
        pendingTemplate && (
          <Modal
            open={!!pendingTemplate}
            onOpenChange={(open) => {
              if (!open) handleCancelReplace();
            }}
            dismissible
            slotHeading={<span>Replace your work?</span>}
            slotFooter={
              <div className="flex justify-end gap-3">
                <Button
                  kind="secondary"
                  onClick={handleCancelReplace}
                  data-testid="cancel-template-replace-btn"
                >
                  Cancel
                </Button>
                <Button
                  kind="primary"
                  className="nvidia-green-button"
                  onClick={handleConfirmReplace}
                  data-testid="confirm-template-replace-btn"
                >
                  Replace
                </Button>
              </div>
            }
          >
            <Text
              kind="body/regular/sm"
              style={{ color: "var(--text-secondary)", display: "block" }}
              data-testid="confirm-template-replace-body"
            >
              Switching to the {getTemplateByName(pendingTemplate).label} template will
              replace your current description and fields.
            </Text>
          </Modal>
        )
      }
    >
      {/* Template selector (create-only) */}
      <div className="glass-card p-6" data-testid="template-selector-card">
        <Text
          kind="label/bold/sm"
          style={{ color: "var(--text-primary)", display: "block" }}
        >
          Start from
        </Text>
        <Text
          kind="body/regular/sm"
          style={{
            color: "var(--text-muted)",
            display: "block",
            marginTop: 2,
            marginBottom: 10,
          }}
        >
          Pick a starter template to pre-fill the description and schema, or stay with
          Blank to build from scratch.
        </Text>
        <Select
          items={GUIDANCE_TEMPLATES.map((t) => ({
            value: t.name,
            children: t.label,
          }))}
          value={selectedTemplate}
          onValueChange={(val) => handleTemplateChange(val as TemplateName)}
          data-testid="template-selector"
        />
      </div>

      <DescriptionCard
        description={form.description}
        onChange={form.handleDescriptionChange}
        descriptionRef={descriptionRef}
      />

      <SchemaCard
        coreFields={form.coreFields}
        auxFields={form.auxFields}
        allIssues={form.displayIssues}
        pathPrefixes={form.pathPrefixes}
        handlers={fieldHandlers}
        rationaleEnabled={form.rationaleEnabled}
        onRationaleEnabledChange={form.handleRationaleEnabledChange}
        bannerSlot={
          <div
            className="mb-4 rounded-lg px-4 py-3 glass-info flex items-start gap-2.5"
            data-testid="edit-policy-banner"
          >
            <Info
              size={16}
              style={{ color: "var(--text-muted)", flexShrink: 0, marginTop: 2 }}
              aria-hidden
            />
            <Text kind="body/regular/sm" style={{ color: "inherit" }}>
              Core field changes (types, values, or adding/removing) require re-labeling
              existing images.
            </Text>
          </div>
        }
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

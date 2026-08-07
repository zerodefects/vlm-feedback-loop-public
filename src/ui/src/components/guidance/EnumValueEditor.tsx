// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Tag/chip editor for enum and enum_set allowed values — existing values render
 * as removable glass pills, plus an inline "add" input that commits on Enter.
 */

import { useState, useRef, useEffect } from "react";
import { Button, Text } from "@kui/react";
import { X, Plus } from "lucide-react";
import { MarkerIcon } from "./MarkerIcon";

interface EnumValueEditorProps {
  values: string[];
  onChange: (values: string[]) => void;
  fieldClientId: string;
  showMarkers?: boolean;
  /** Every `~` marker carries the same dynamic
   *  "Changing this invalidates your {N} verified labels..." tooltip, so
   *  the parent (SchemaCard) threads the same string through. One marker
   *  per editor (on the [+ add] affordance) — a per-chip repeat reads as
   *  noise. */
  markerTooltip?: string;
}

export function EnumValueEditor({
  values,
  onChange,
  fieldClientId,
  showMarkers,
  markerTooltip,
}: EnumValueEditorProps) {
  const [isAdding, setIsAdding] = useState(false);
  const [newValue, setNewValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isAdding) inputRef.current?.focus();
  }, [isAdding]);

  function commitValue() {
    const trimmed = newValue.trim();
    if (trimmed && !values.includes(trimmed)) onChange([...values, trimmed]);
    setNewValue("");
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") {
      e.preventDefault();
      commitValue();
    } else if (e.key === "Escape") {
      setIsAdding(false);
      setNewValue("");
    }
  }

  return (
    <div className="space-y-1.5" data-testid={`enum-editor-${fieldClientId}`}>
      <div className="flex flex-wrap gap-1.5">
        {values.map((v, i) => (
          <span
            key={`${v}-${i}`}
            className="glass-pill group"
            data-testid={`value-chip-${fieldClientId}-${i}`}
          >
            <Text kind="label/regular/xs">{v}</Text>
            <Button
              kind="tertiary"
              onClick={() => onChange(values.filter((_, j) => j !== i))}
              className="rounded-full p-0.5 opacity-60 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
              style={{ minWidth: "auto", padding: "2px" }}
              aria-label={`Remove value ${v}`}
            >
              <X size={10} />
            </Button>
          </span>
        ))}
        {!isAdding && (
          <Button
            kind="tertiary"
            onClick={() => setIsAdding(true)}
            className="glass-btn muted"
            style={{ borderStyle: "dashed" }}
            data-testid={`add-value-btn-${fieldClientId}`}
          >
            {showMarkers && <MarkerIcon tooltip={markerTooltip} />}
            <Plus size={10} /> add
          </Button>
        )}
      </div>
      {isAdding && (
        <input
          ref={inputRef}
          type="text"
          value={newValue}
          onChange={(e) => setNewValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={() => {
            if (newValue.trim()) commitValue();
            setIsAdding(false);
          }}
          placeholder="Type a value and press Enter"
          className="glass-input w-full rounded-lg px-2 py-1 text-xs"
          data-testid={`new-value-input-${fieldClientId}`}
        />
      )}
    </div>
  );
}

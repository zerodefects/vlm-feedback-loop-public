// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Bare credential-paste field shared by the FTU setup screens and the
 * NIM Connection credential rows. A controlled password input wrapping
 * only the shared glass styling and the anti-autofill attribute cluster
 * (these are pasted API keys, not stored logins — password managers
 * must stay out of the way). Label, error line, and Test/Continue
 * chrome stay with the caller.
 */

interface KeyPasteInputProps {
  name: string;
  ariaLabel: string;
  placeholder: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onClick?: (e: React.MouseEvent<HTMLInputElement>) => void;
  inputRef?: React.RefObject<HTMLInputElement | null>;
  testId?: string;
}

export function KeyPasteInput({
  name,
  ariaLabel,
  placeholder,
  value,
  onChange,
  onKeyDown,
  onClick,
  inputRef,
  testId,
}: KeyPasteInputProps): JSX.Element {
  return (
    <input
      ref={inputRef}
      type="password"
      name={name}
      autoComplete="off"
      spellCheck={false}
      aria-label={ariaLabel}
      data-form-type="other"
      data-lpignore="true"
      data-1p-ignore=""
      className="glass-input w-full px-3 py-2 text-sm"
      placeholder={placeholder}
      value={value}
      onChange={onChange}
      onClick={onClick}
      onKeyDown={onKeyDown}
      data-testid={testId}
    />
  );
}

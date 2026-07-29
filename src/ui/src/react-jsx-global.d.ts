// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Global `JSX` namespace shim for @types/react v19.
//
// React 19's type definitions moved the JSX namespace under `React.JSX` and
// removed the ambient global `JSX` namespace that existed in @types/react v18.
// This codebase annotates component return types as `JSX.Element` in ~15 source
// sites (the established convention here), which no longer resolves against the
// bare global name and fails type-checking with TS2503 ("Cannot find namespace
// 'JSX'").
//
// Re-expose a global `JSX` namespace whose members alias `React.JSX` so the
// existing `JSX.Element` annotations keep resolving without rewriting every call
// site. With `jsx: "react-jsx"`, TypeScript resolves intrinsic-element checking
// (e.g. `<div>`) via `react/jsx-runtime`'s JSX namespace, not this global one, so
// this shim only affects explicit `JSX.*` type references in source.
import type * as React from "react";

declare global {
  namespace JSX {
    type ElementType = React.JSX.ElementType;
    type Element = React.JSX.Element;
    type ElementClass = React.JSX.ElementClass;
    type ElementAttributesProperty = React.JSX.ElementAttributesProperty;
    type ElementChildrenAttribute = React.JSX.ElementChildrenAttribute;
    type LibraryManagedAttributes<C, P> = React.JSX.LibraryManagedAttributes<C, P>;
    type IntrinsicAttributes = React.JSX.IntrinsicAttributes;
    type IntrinsicClassAttributes<T> = React.JSX.IntrinsicClassAttributes<T>;
    type IntrinsicElements = React.JSX.IntrinsicElements;
  }
}

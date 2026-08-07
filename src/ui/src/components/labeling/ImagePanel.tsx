// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Left panel of the labeling screen — displays the current image,
 * a loading spinner, or the missing-image state.
 */

import { useState } from "react";
import { Spinner, Text, Button } from "@kui/react";
import { ImageOff, Minus, Plus, RotateCcw } from "lucide-react";
import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";

import { ActionRequestPanel } from "@/components/ActionRequestPanel";
import { imageUrl } from "@/api/labeling";

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;

interface ImagePanelProps {
  projectId: string;
  exampleKey: string | null;
  /** Filesystem path on the backend host where the image is recorded —
     surfaced in the missing-image state so the SME's `[Report Missing
     Files]` Action Request carries an admin-actionable disk path,
     not the API URL. */
  storageRef?: string | null;
  isLoading: boolean;
  onImageMissing: () => void;
  onImageLoaded: () => void;
}

interface ZoomableImageProps {
  src: string;
  alt: string;
  onError: () => void;
  onLoad: () => void;
}

function ZoomableImage({ src, alt, onError, onLoad }: ZoomableImageProps) {
  const [scale, setScale] = useState(MIN_ZOOM);
  const zoomPercentage = Math.round(scale * 100);
  const isAtMinimum = scale <= MIN_ZOOM;
  const isAtMaximum = scale >= MAX_ZOOM;

  return (
    <TransformWrapper
      initialScale={MIN_ZOOM}
      minScale={MIN_ZOOM}
      maxScale={MAX_ZOOM}
      centerOnInit
      centerZoomedOut
      limitToBounds
      // The library's smooth mode multiplies `step` by the raw wheel delta.
      // A standard mouse-wheel event commonly reports 100–120, which turns
      // one notch into a jump from 1× to the 4× limit.
      smooth={false}
      wheel={{ step: 0.15 }}
      panning={{ velocityDisabled: true }}
      doubleClick={{ mode: "toggle", step: 0.75 }}
      onTransform={(_ref, state) => setScale(state.scale)}
    >
      {({ zoomIn, zoomOut, resetTransform }) => (
        <>
          <div
            className="glass-pill image-zoom-controls absolute right-3 top-3 z-10 flex items-center gap-1"
            role="group"
            aria-label="Image zoom controls"
            data-testid="image-zoom-controls"
          >
            <Button
              kind="tertiary"
              size="tiny"
              className="rounded p-1"
              aria-label="Zoom out"
              title="Zoom out"
              disabled={isAtMinimum}
              onClick={() => zoomOut()}
              data-testid="zoom-out-btn"
            >
              <Minus size={16} aria-hidden="true" />
            </Button>
            <Text
              kind="label/regular/xs"
              className="min-w-10 text-center tabular-nums"
              style={{ color: "var(--text-secondary)" }}
              data-testid="zoom-level"
            >
              {zoomPercentage}%
            </Text>
            <Button
              kind="tertiary"
              size="tiny"
              className="rounded p-1"
              aria-label="Zoom in"
              title="Zoom in"
              disabled={isAtMaximum}
              onClick={() => zoomIn()}
              data-testid="zoom-in-btn"
            >
              <Plus size={16} aria-hidden="true" />
            </Button>
            <Button
              kind="tertiary"
              size="tiny"
              className="rounded p-1"
              aria-label="Reset zoom"
              title="Reset zoom"
              disabled={isAtMinimum}
              onClick={() => resetTransform()}
              data-testid="zoom-reset-btn"
            >
              <RotateCcw size={15} aria-hidden="true" />
            </Button>
          </div>

          <TransformComponent
            wrapperClass={scale > MIN_ZOOM ? "cursor-grab active:cursor-grabbing" : ""}
            wrapperStyle={{ width: "100%", height: "100%" }}
            contentStyle={{
              width: "100%",
              height: "100%",
              alignItems: "center",
              justifyContent: "center",
            }}
            wrapperProps={{
              title:
                "Use the mouse wheel or zoom controls to inspect the image. Drag when zoomed.",
              "aria-label": "Zoomable labeling image",
            }}
          >
            <img
              src={src}
              alt={alt}
              // ``h-full w-full`` + ``object-contain`` lets small images
              // scale UP to fill the container while preserving aspect
              // ratio — important on big monitors where a 400×300 source
              // image otherwise floats in a sea of empty card. ``max-h``
              // caps height to the viewport minus chrome so the image
              // never bleeds into the action buttons. ``object-contain``
              // means the actual pixels are letterboxed inside the box,
              // never cropped.
              className="h-full w-full object-contain"
              style={{ maxHeight: "calc(100vh - 260px)" }}
              data-testid="labeling-image"
              // The proposal POST is the blocking call that gates the SME's
              // next action; the image fetch is allowed to arrive whenever it
              // arrives. Hint to the browser so the image doesn't contend
              // with the proposal request over the shared HTTP/2 connection,
              // and decode off the main thread so paint happens without a
              // jank spike when the proposal response lands.
              fetchPriority="low"
              decoding="async"
              draggable={false}
              onError={onError}
              onLoad={onLoad}
            />
          </TransformComponent>
        </>
      )}
    </TransformWrapper>
  );
}

export function ImagePanel({
  projectId,
  exampleKey,
  storageRef,
  isLoading,
  onImageMissing,
  onImageLoaded,
}: ImagePanelProps) {
  const [imgError, setImgError] = useState(false);
  const [showActionRequest, setShowActionRequest] = useState(false);
  const [retryNonce, setRetryNonce] = useState(0);

  // Reset error state when the example changes
  const [prevKey, setPrevKey] = useState<string | null>(null);
  if (exampleKey !== prevKey) {
    setPrevKey(exampleKey);
    setImgError(false);
    setShowActionRequest(false);
    setRetryNonce(0);
  }

  if (isLoading || !exampleKey) {
    // Loading uses the glass-card's lighter rgba-white tint (no dark overlay)
    // so the KUI Spinner — which renders in the NVIDIA-green primary token via
    // `.nv-spinner-arrow` — is visible. Adding `rgba(0,0,0,0.3)` here (as the
    // success state does to bias for the photographic image inside) buries
    // the spinner against the page's already-dark background and the SME
    // sees a black void. The success state retains the dark overlay because
    // the image carries the visual weight on its own.
    return (
      <div
        className="glass-card flex items-center justify-center"
        style={{ minHeight: 400 }}
        data-testid="image-panel-loading"
      >
        <Spinner size="large" aria-label="Loading image" />
      </div>
    );
  }

  if (imgError) {
    return (
      <div className="flex flex-col gap-2" data-testid="missing-image-wrapper">
        {/* Filename eyebrow ("img_047.jpg" above the
           placeholder). Mirrors the success-path
           filename header below so the SME has the same orientation cue
           in either state. */}
        <Text
          kind="label/regular/xs"
          style={{ color: "var(--text-muted)", letterSpacing: "0.04em" }}
          data-testid="missing-image-filename-label"
        >
          {exampleKey}
        </Text>
        <div
          className="glass-card flex flex-col items-center justify-center gap-4 p-6"
          style={{ background: "rgba(0,0,0,0.3)", minHeight: 400 }}
          data-testid="image-panel-missing"
        >
          <ImageOff size={48} style={{ color: "var(--text-faint)" }} />
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-muted)", textAlign: "center" }}
          >
            Image not found at original location.
          </Text>
          <Text
            kind="label/regular/xs"
            style={{
              color: "var(--text-faint)",
              wordBreak: "break-all",
              textAlign: "center",
            }}
          >
            {/* The SME needs the disk path so the
             Action Request handoff is admin-actionable. Fall back to the API
             URL only when storage_ref is unavailable. */}
            Expected: {storageRef ?? imageUrl(projectId, exampleKey)}
          </Text>

          {!showActionRequest ? (
            <div className="flex items-center gap-3">
              <Button
                kind="primary"
                className="nvidia-green-button"
                onClick={() => {
                  setImgError(false);
                  setRetryNonce((value) => value + 1);
                }}
                data-testid="retry-image-btn"
              >
                Retry image
              </Button>
              <Button
                kind="secondary"
                onClick={() => setShowActionRequest(true)}
                data-testid="report-missing-files-btn"
              >
                Report Missing Files
              </Button>
            </div>
          ) : (
            <div className="w-full">
              <ActionRequestPanel
                projectId={projectId}
                requestType="missing_files"
                onClose={() => setShowActionRequest(false)}
              />
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2" data-testid="image-panel-wrapper">
      {/* Image identity label — the
          example key sits above the image card so the SME knows which
          file they are looking at). Tertiary tracked-caps treatment
          reads as an eyebrow, not a heading. */}
      <Text
        kind="label/regular/xs"
        style={{ color: "var(--text-muted)", letterSpacing: "0.04em" }}
        data-testid="image-filename-label"
      >
        {exampleKey}
      </Text>
      <div
        className="glass-card glass-card--static relative flex flex-1 items-center justify-center overflow-hidden"
        style={{ background: "rgba(0,0,0,0.3)", minHeight: 400 }}
        data-testid="image-panel"
      >
        <ZoomableImage
          key={`${exampleKey}-${retryNonce}`}
          src={`${imageUrl(projectId, exampleKey)}${retryNonce > 0 ? `?retry=${retryNonce}` : ""}`}
          alt={`Example ${exampleKey}`}
          onError={() => {
            setImgError(true);
            onImageMissing();
          }}
          onLoad={onImageLoaded}
        />
      </div>
    </div>
  );
}

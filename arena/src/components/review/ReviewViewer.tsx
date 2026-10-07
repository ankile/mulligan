import { useCallback, useEffect, useRef, useState } from "react";
import type { ReactNode, RefObject } from "react";
import { getVideoUrl, type ReviewEpisode } from "../../lib/hf-api";
import { Timeline } from "./Timeline";
import { cameraLabel, clamp, type CropBox } from "./format";

// ---------------------------------------------------------------------------
// Camera grid + frame-exact seeking, shared by the outcome and stage review views:
// seek + requestVideoFrameCallback drift verification, metadataEpoch re-seek, the
// crop-dims guard, stopAndSnap playback snap and stable-DOM CSS-only crops. FPS is a
// prop (OutcomeReview passes the constant; the stage view passes the exported spec's
// fps), and the decision overlays are render props, called only while not playing.
// ---------------------------------------------------------------------------

export interface ViewerControls {
  togglePlay: () => void;
  /** Stop playback and snap the frame counter to the displayed frame.
   *  Returns the snapped frame when playback WAS running, else null — mark
   *  handlers must use it: during playback the parent `frame` state is frozen
   *  at the play-start value and marking there lands frames early. */
  pause: () => number | null;
}

export function ReviewViewer({
  datasetId,
  episode,
  cameraKeys,
  primaryKey,
  fps,
  frame,
  onFrame,
  lastValidFrame,
  controlsRef,
  cropByCameraKey,
  storedFrameHW,
  onDrift,
  onUnverifiable,
  renderVideoOverlay,
  renderTimelineOverlays,
}: {
  datasetId: string;
  episode: ReviewEpisode;
  cameraKeys: string[];
  primaryKey: string;
  /** Native dataset FPS (frame <-> seconds). Outcome review passes hf-api's
   *  station constant; stage review passes the exported spec's fps. */
  fps: number;
  frame: number;
  onFrame: (frame: number) => void;
  /** Drives the Timeline's is_valid padding hatch; null = unknown/none. */
  lastValidFrame: number | null;
  controlsRef: RefObject<ViewerControls | null>;
  /** Station display crops per camera key (stored-frame px), null = no spec. */
  cropByCameraKey: Record<string, CropBox> | null;
  /** Crop reference space [H, W]; present whenever cropByCameraKey is. */
  storedFrameHW: [number, number] | null;
  /** Frame-verification drift, surfaced so the parent can BLOCK confirms —
   *  a drifted display means the counted frame is not the shown frame. */
  onDrift: (drift: string | null) => void;
  /** Surfaced when the browser lacks requestVideoFrameCallback — parents that
   *  mint gold timestamps should block committed verdicts on it. */
  onUnverifiable?: (unverifiable: boolean) => void;
  /** Per-camera decision overlay (tint/border), rendered only while paused. */
  renderVideoOverlay?: (frame: number) => ReactNode;
  /** Domain markers on the timeline strip (receives the pct positioner). */
  renderTimelineOverlays?: (pct: (value: number) => string) => ReactNode;
}) {
  const videoRefs = useRef(new Map<string, HTMLVideoElement>());
  const seekTokenRef = useRef(0);
  const playingRef = useRef(false);
  const [playing, setPlaying] = useState(false);
  const [drift, setDrift] = useState<string | null>(null);
  const [unverifiable, setUnverifiable] = useState(false);
  // Station crop display defaults ON, as in the Python outcome editor's review view.
  const [cropOn, setCropOn] = useState(true);
  const [cropDimsError, setCropDimsError] = useState<string | null>(null);
  const hasCrops =
    cropByCameraKey !== null &&
    storedFrameHW !== null &&
    cameraKeys.some((key) => cropByCameraKey[key] !== undefined);
  const cropsActive = hasCrops && cropOn && cropDimsError === null;
  // Bumped when a video reaches HAVE_METADATA. Seeks issued before that are
  // dropped by the browser, so the seek+verify pass must re-run afterwards or
  // the operator sees a phantom drift warning from the pre-load frame 0.
  const [metadataEpoch, setMetadataEpoch] = useState(0);
  // Decoded [w, h] per camera key (from loadedmetadata) — refs must not be
  // read during render, so the equal-height aspect math uses this state.
  const [videoDims, setVideoDims] = useState<Map<string, [number, number]>>(() => new Map());

  const primaryFrom = episode.perCamera[primaryKey].fromTimestamp;
  const primaryTo = episode.perCamera[primaryKey].toTimestamp;

  const setVideoRef = useCallback(
    (key: string) => (el: HTMLVideoElement | null) => {
      if (el) videoRefs.current.set(key, el);
      else videoRefs.current.delete(key);
    },
    []
  );

  // Playback state is per-episode; a queue advance must never leave the new
  // episode auto-playing from the previous one's position.
  useEffect(() => {
    playingRef.current = false;
    /* eslint-disable react-hooks/set-state-in-effect -- switching episodes is
       an imperative reset of playback state, not derived render state. */
    setPlaying(false);
    setDrift(null);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [episode]);

  // Crop-vs-video dims guard. This must NOT live in onLoadedMetadata: video
  // src is per-FILE (many episodes share one chunk file, so the event fires
  // once per session), and the crop spec can arrive after metadata. Re-check
  // whenever the spec, camera set, or a video's metadata changes; reset when
  // the SPEC changes (not per episode — that wiped a real mismatch forever).
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect -- imperative guard over
       video-element metadata, not derived render state. */
    setCropDimsError(null);
    if (cropByCameraKey === null || storedFrameHW === null) return;
    const [frameH, frameW] = storedFrameHW;
    for (const key of cameraKeys) {
      if (cropByCameraKey[key] === undefined) continue;
      const video = videoRefs.current.get(key);
      // Metadata not loaded yet -> metadataEpoch re-runs this effect later.
      if (!video || video.videoWidth === 0) continue;
      if (video.videoWidth !== frameW || video.videoHeight !== frameH) {
        setCropDimsError(
          `${cameraLabel(key)} is ${video.videoWidth}x${video.videoHeight}, ` +
            `crop boxes expect ${frameW}x${frameH} stored frames`
        );
        return;
      }
    }
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [cropByCameraKey, storedFrameHW, cameraKeys, metadataEpoch]);

  // Frame-exact seek: every camera lands on the midpoint of frame i, offset by
  // that camera's own from_timestamp (0 today, but multi-episode video chunks
  // rely on it). The primary camera's landing is VERIFIED, never assumed.
  useEffect(() => {
    if (playing) return;
    for (const key of cameraKeys) {
      const video = videoRefs.current.get(key);
      if (!video) continue;
      video.currentTime =
        episode.perCamera[key].fromTimestamp + (frame + 0.5) / fps;
    }
    const primary = videoRefs.current.get(primaryKey);
    if (!primary) return;
    // Capability is flagged from the loadedmetadata handler (see below) so the
    // operator sees a banner instead of a silently unverified seek.
    if (typeof primary.requestVideoFrameCallback !== "function") return;
    const token = ++seekTokenRef.current;
    primary.requestVideoFrameCallback((_now, meta) => {
      if (seekTokenRef.current !== token) return;
      const exact = (meta.mediaTime - primaryFrom) * fps;
      const landed = Math.round(exact);
      if (Math.abs(exact - landed) > 0.25 || landed !== frame) {
        setDrift(
          `requested frame ${frame}, video presented frame ${landed} ` +
            `(mediaTime ${meta.mediaTime.toFixed(4)}s, from_timestamp ` +
            `${primaryFrom.toFixed(4)}s, exact ${exact.toFixed(3)})`
        );
      } else {
        setDrift(null);
      }
    });
  }, [frame, playing, episode, cameraKeys, primaryKey, primaryFrom, metadataEpoch, fps]);

  const stopAndSnap = useCallback((): number | null => {
    playingRef.current = false;
    setPlaying(false);
    const primary = videoRefs.current.get(primaryKey);
    if (!primary) return null;
    const snapped = clamp(
      Math.round((primary.currentTime - primaryFrom) * fps - 0.5),
      0,
      episode.rawLength - 1
    );
    onFrame(snapped);
    return snapped;
  }, [episode.rawLength, onFrame, primaryFrom, primaryKey, fps]);

  const togglePlay = useCallback(() => {
    if (playingRef.current) {
      stopAndSnap();
    } else {
      playingRef.current = true;
      setPlaying(true);
    }
  }, [stopAndSnap]);

  const pause = useCallback((): number | null => {
    if (playingRef.current) return stopAndSnap();
    return null;
  }, [stopAndSnap]);

  useEffect(() => {
    controlsRef.current = { togglePlay, pause };
    // Null on unmount: a stale controls closure would otherwise pause/snap the
    // PREVIOUS episode's videos when the parent calls pause() after a camera-
    // less episode replaced the viewer.
    return () => {
      controlsRef.current = null;
    };
  }, [controlsRef, togglePlay, pause]);

  useEffect(() => {
    onDrift(drift);
  }, [drift, onDrift]);

  useEffect(() => {
    onUnverifiable?.(unverifiable);
  }, [unverifiable, onUnverifiable]);

  // Playback: the primary camera drives, the others follow its offset-corrected
  // clock; playback halts at the raw end of the episode segment.
  useEffect(() => {
    if (!playing) return;
    const primary = videoRefs.current.get(primaryKey);
    if (!primary) return;
    const videos = cameraKeys
      .map((key) => videoRefs.current.get(key))
      .filter((v): v is HTMLVideoElement => Boolean(v));
    for (const video of videos) void video.play();

    let raf = 0;
    const tick = () => {
      const elapsed = primary.currentTime - primaryFrom;
      for (const key of cameraKeys) {
        const video = videoRefs.current.get(key);
        if (!video || video === primary) continue;
        const target = episode.perCamera[key].fromTimestamp + elapsed;
        if (Math.abs(video.currentTime - target) > 0.1) video.currentTime = target;
      }
      if (primary.currentTime >= primaryTo) {
        stopAndSnap();
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      for (const video of videos) video.pause();
    };
  }, [playing, cameraKeys, episode, primaryKey, primaryFrom, primaryTo, stopAndSnap]);

  // All cameras in ONE row at EQUAL HEIGHT, like the Python outcome editor's side-by-side
  // composite: each tile's flex-grow is its aspect ratio (basis 0), so widths
  // distribute proportionally and heights come out identical. Aspect source:
  // the active crop box, else the video's decoded dimensions (metadataEpoch
  // re-renders once they load), else the spec's stored-frame shape.
  const tileAspect = (key: string): number => {
    const box = cropByCameraKey?.[key];
    if (cropsActive && box !== undefined) {
      const [x0, y0, x1, y1] = box;
      return (x1 - x0) / (y1 - y0);
    }
    const dims = videoDims.get(key);
    if (dims && dims[1] > 0) return dims[0] / dims[1];
    if (storedFrameHW !== null) return storedFrameHW[1] / storedFrameHW[0];
    return 4 / 3;
  };

  // Decision overlays are domain-specific (outcome tint / stage event border)
  // and injected by the parent; suppressed during playback — the frame counter
  // only tracks while scrubbing.
  const videoOverlay = !playing ? renderVideoOverlay?.(frame) : null;

  return (
    <div>
      {drift && (
        <div className="mb-3 rounded-lg border border-coral bg-coral-light px-4 py-2 text-xs font-mono text-coral">
          ⚠ frame drift on {cameraLabel(primaryKey)}: {drift}. Do not trust the
          displayed frame — re-seek (arrow keys) before marking.
        </div>
      )}
      {unverifiable && (
        <div className="mb-3 rounded-lg border border-gold bg-gold-light px-4 py-2 text-xs font-mono text-gold">
          ⚠ this browser has no requestVideoFrameCallback — seek landings cannot
          be verified. Review in Chrome/Edge before confirming marks.
        </div>
      )}

      {cropDimsError && (
        <div className="mb-3 rounded-lg border border-gold bg-gold-light px-4 py-2 text-xs font-mono text-gold">
          ⚠ station crops disabled: {cropDimsError}
        </div>
      )}

      <div className="flex gap-3">
        {cameraKeys.map((key) => {
          const box = cropByCameraKey?.[key];
          // Stable DOM per camera: cropping is style/class-only so toggling
          // (or the dims guard tripping) never remounts (= reloads) the video.
          const cropped =
            cropsActive && box !== undefined && storedFrameHW !== null;
          const frameW = (storedFrameHW ?? [0, 0])[1];
          let containerStyle: React.CSSProperties | undefined;
          let videoStyle: React.CSSProperties | undefined;
          if (cropped) {
            const [x0, y0, x1, y1] = box;
            containerStyle = { aspectRatio: `${x1 - x0} / ${y1 - y0}` };
            videoStyle = {
              position: "absolute",
              maxWidth: "none",
              width: `${(frameW / (x1 - x0)) * 100}%`,
              left: `${(-x0 / (x1 - x0)) * 100}%`,
              top: `${(-y0 / (y1 - y0)) * 100}%`,
            };
          }
          return (
            <div
              key={key}
              className={`relative min-w-0 ${cropped ? "overflow-hidden rounded-lg bg-warm-100" : ""}`}
              style={{ ...containerStyle, flex: `${tileAspect(key)} 1 0%` }}
            >
              <video
                ref={setVideoRef(key)}
                src={getVideoUrl(key, episode.perCamera[key].fileIndex, datasetId)}
                className={cropped ? "" : "w-full rounded-lg bg-warm-100"}
                style={videoStyle}
                muted
                playsInline
                preload="auto"
                onLoadedMetadata={(e) => {
                  const video = e.target as HTMLVideoElement;
                  video.currentTime =
                    episode.perCamera[key].fromTimestamp + (frame + 0.5) / fps;
                  if (
                    key === primaryKey &&
                    typeof video.requestVideoFrameCallback !== "function"
                  ) {
                    setUnverifiable(true);
                  }
                  // Crop-dims guard lives in the effect above (src is
                  // per-file, this event fires once per session).
                  setVideoDims((prev) =>
                    new Map(prev).set(key, [video.videoWidth, video.videoHeight])
                  );
                  setMetadataEpoch((epoch) => epoch + 1);
                }}
              />
              {videoOverlay}
              <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/60 text-white text-[11px] font-mono">
                {cameraLabel(key)}
                {key === primaryKey ? " ·primary" : ""}
                {cropped ? " ·crop" : ""}
              </span>
            </div>
          );
        })}
      </div>

      <Timeline
        rawLength={episode.rawLength}
        frame={frame}
        lastValidFrame={lastValidFrame}
        onScrub={(next) => {
          if (playingRef.current) stopAndSnap();
          onFrame(next);
        }}
        renderOverlays={renderTimelineOverlays}
      />

      <div className="flex items-center gap-3 mt-3">
        <button
          onClick={togglePlay}
          className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-teal text-white font-body font-medium text-xs hover:bg-teal/90 transition-colors cursor-pointer"
        >
          {playing ? (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="6,4 20,12 6,20" />
            </svg>
          )}
          {playing ? "Pause" : "Play"}
          <span className="font-mono text-[10px] opacity-70">space</span>
        </button>
        <span className="font-mono text-xs text-ink">
          frame {frame} / {episode.rawLength - 1} · {(frame / fps).toFixed(2)}s
        </span>
        {hasCrops && cropDimsError === null && (
          <button
            onClick={() => setCropOn((on) => !on)}
            className={`ml-auto px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all ${
              cropOn
                ? "bg-teal/10 text-teal border border-teal/40"
                : "bg-white border border-warm-200 text-ink-muted hover:border-warm-300"
            }`}
            title="Station display crops from the task registry (same view as the Python outcome editor)"
          >
            station crop {cropOn ? "on" : "off"}
          </button>
        )}
      </div>
    </div>
  );
}

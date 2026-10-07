import { useState, useEffect, useRef, useCallback } from "react";
import { getVideoUrl, type EpisodeMetadata } from "../lib/hf-api";
import type { RoundVideoSpec } from "../lib/roundVideoSpecs";
import { TONE_BADGE, outcomeLabel, outcomeTone } from "../lib/outcomeScore";

type EpisodeWithoutSuccess = Omit<EpisodeMetadata, "success">;

const GRID_COLS: Record<number, string> = {
  1: "grid-cols-1",
  2: "grid-cols-2",
  3: "grid-cols-3",
  4: "grid-cols-4",
  5: "grid-cols-5",
  6: "grid-cols-6",
};

/**
 * Synchronized grid of episode videos. `null` entries are invisible spacers
 * so a caller can pad a row (the joined view keeps one row per session by
 * padding each session's tiles to `columns`).
 */
export function RoundVideos({
  videos,
  columns,
}: {
  videos: (RoundVideoSpec | null)[];
  columns?: number;
}) {
  const cols = columns ?? Math.max(1, Math.min(videos.length, 4));
  const gridClass = GRID_COLS[cols];
  if (!gridClass) {
    throw new Error(`RoundVideos: unsupported column count ${cols}`);
  }
  const videoRefs = useRef<(HTMLVideoElement | null)[]>([]);
  const [playing, setPlaying] = useState(false);
  const [playbackError, setPlaybackError] = useState<string | null>(null);
  const animFrameRef = useRef<number>(0);

  const togglePlay = useCallback(() => {
    setPlaybackError(null);
    setPlaying((p) => !p);
  }, []);

  const resetVideos = useCallback(() => {
    setPlaying(false);
    for (let i = 0; i < videoRefs.current.length; i++) {
      const video = videoRefs.current[i];
      const episode = videos[i]?.episode;
      if (video && episode) {
        video.pause();
        video.currentTime = episode.fromTimestamp;
      }
    }
  }, [videos]);

  useEffect(() => {
    // Index-aligned with `videos`; tiles without metadata render no <video>.
    const active: Array<{ el: HTMLVideoElement; episode: EpisodeWithoutSuccess }> = [];
    for (let i = 0; i < videos.length; i++) {
      const el = videoRefs.current[i];
      const episode = videos[i]?.episode;
      if (el && episode) active.push({ el, episode });
    }

    if (playing) {
      let cancelled = false;
      // Keep finished clips at their last frame until the whole comparison ends.
      // Rewinding each clip immediately makes a shorter clip appear unfinished
      // on the next animation frame and prevents the group from ever stopping.
      const complete = ({ el, episode }: typeof active[number]) =>
        el.ended || el.currentTime >= episode.toTimestamp - 0.05;
      if (active.every(complete)) {
        for (const { el, episode } of active) el.currentTime = episode.fromTimestamp;
      }
      for (const item of active) {
        if (!complete(item)) {
          void item.el.play().catch((error: unknown) => {
            if (cancelled) return;
            setPlaybackError(error instanceof Error ? error.message : String(error));
            setPlaying(false);
          });
        }
      }

      const sync = () => {
        let allDone = true;
        for (const item of active) {
          const { el } = item;
          if (complete(item)) {
            el.pause();
          } else {
            allDone = false;
          }
        }
        if (allDone) {
          setPlaying(false);
          return;
        }
        animFrameRef.current = requestAnimationFrame(sync);
      };
      animFrameRef.current = requestAnimationFrame(sync);
      return () => {
        cancelled = true;
        cancelAnimationFrame(animFrameRef.current);
        for (const { el } of active) el.pause();
      };
    } else {
      for (const { el } of active) el.pause();
    }

    return () => cancelAnimationFrame(animFrameRef.current);
  }, [playing, videos]);

  return (
    <div className="mt-3 mb-1">
      {playbackError && <p role="alert" className="text-sm text-red-700">Video playback failed: {playbackError}</p>}
      <div className={`grid gap-3 ${gridClass}`}>
        {videos.map((spec, i) => {
          if (spec === null) {
            return <div key={i} aria-hidden className="aspect-video" />;
          }
          const episode = spec.episode;
          if (!episode) {
            return (
              <div
                key={i}
                className="rounded-lg bg-warm-100 aspect-video flex items-center justify-center text-ink-muted text-xs"
              >
                No video data
              </div>
            );
          }

          return (
            <div key={i} className="relative">
              <video
                ref={(el) => {
                  videoRefs.current[i] = el;
                }}
                src={spec.videoUrl ?? getVideoUrl(
                  spec.cameraKey,
                  episode.videoFileIndex,
                  spec.datasetRepo
                )}
                className="w-full rounded-lg bg-warm-100"
                muted
                playsInline
                preload="metadata"
                onLoadedMetadata={(e) => {
                  (e.target as HTMLVideoElement).currentTime =
                    episode.fromTimestamp;
                }}
              />
              <div className="absolute bottom-2 left-2 right-2 flex items-center gap-1.5 flex-wrap">
                {spec.badge !== null && (
                  <span className="px-1.5 py-0.5 rounded bg-gold text-white text-[11px] font-mono font-semibold">
                    {spec.badge}
                  </span>
                )}
                <span
                  className="px-2 py-0.5 rounded bg-black/60 text-white text-[11px] font-mono truncate min-w-0 max-w-full"
                  title={spec.policyName}
                >
                  {spec.policyName}
                </span>
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                    TONE_BADGE[outcomeTone(spec.success, spec.numSubtaskMarks)]
                  }`}
                >
                  {outcomeLabel(spec.success, spec.numSubtaskMarks, spec.maxSubtaskMarks)}
                </span>
                <span className="px-1.5 py-0.5 rounded bg-black/40 text-white/80 text-[10px] font-mono">
                  {episode.numFrames} steps
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex items-center justify-center gap-3 mt-3">
        <button
          onClick={togglePlay}
          className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-teal text-white font-body font-medium text-xs hover:bg-teal/90 transition-colors cursor-pointer"
        >
          {playing ? (
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <polygon points="6,4 20,12 6,20" />
            </svg>
          )}
          {playing ? "Pause" : "Play"}
        </button>
        <button
          onClick={resetVideos}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-warm-200 text-ink-muted font-body text-xs hover:bg-warm-50 transition-colors cursor-pointer"
        >
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="1 4 1 10 7 10" />
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
          </svg>
          Reset
        </button>
      </div>
    </div>
  );
}

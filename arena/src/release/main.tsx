import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { validateRelease } from "./types";
import { ReleaseProvider } from "./ReleaseProvider";
import {
  configureReleaseDatasets,
  seedReleaseEpisodes,
  selectPrimaryCameraKey,
} from "../lib/hf-api";
import type { UISnapshot } from "./adapter";
import App from "../App";
import "../index.css";
import { trackOutboundClicks } from "./trackClicks";
trackOutboundClicks();
const root = createRoot(document.getElementById("root")!);
async function load(path: string) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} returned ${r.status}`);
  return r.json();
}
Promise.all([load("/data/release.json"), load("/data/ui.json"), load("/data/sim-statistics.json")])
  .then(([raw, ui, sim]) => {
    const data = validateRelease(raw);
    configureReleaseDatasets(
      new Map(
        data.datasets.map((d) => [d.id, { revision: d.revision, fps: d.fps }]),
      ),
    );
    // Selected eval metadata and corrected outcomes come from the verified export.
    // Do not re-read mutable reviews or expose unselected policies from mixed recordings.
    for (const task of data.tasks)
      for (const repo of new Set(task.blocks.map((b) => b.dataset))) {
        const blocks = task.blocks.filter((b) => b.dataset === repo);
        const cameraKeys = blocks[0].cameras;
        const camera = selectPrimaryCameraKey(cameraKeys);
        const rows = new Map(
          blocks
            .flatMap((b) => b.starts.flatMap((s) => s.results))
            .map((r) => [r.episode, r]),
        );
        const episodes = [...rows.values()]
          .map((r) => {
            const v = r.videos[camera];
            return {
              episodeIndex: r.episode,
              numFrames: r.frames,
              duration: v.to - v.from,
              videoFileIndex: v.file,
              fromTimestamp: v.from,
              toTimestamp: v.to,
              cameraTimings: Object.fromEntries(
                Object.entries(r.videos).map(([k, v]) => [
                  k,
                  {
                    videoFileIndex: v.file,
                    fromTimestamp: v.from,
                    toTimestamp: v.to,
                  },
                ]),
              ),
            };
          })
          .sort((a, b) => a.episodeIndex - b.episodeIndex);
        seedReleaseEpisodes(repo, {
          episodes,
          cameraKeys,
          successMap: new Map(
            [...rows.values()].map((r) => [r.episode, r.success]),
          ),
          complete: true,
        });
      }
    // Old task-card URLs still land on the corresponding original Arena filter.
    const params = new URLSearchParams(location.search);
    if (params.has("task") && !params.has("tab")) {
      params.set("env", params.get("task")!);
      params.delete("task");
      history.replaceState(null, "", `?${params}`);
    }
    if (params.get("tab") === "labeling") {
      params.set("tab", "leaderboard");
      history.replaceState(null, "", `?${params}`);
    }
    root.render(
      <StrictMode>
        <ReleaseProvider data={data} ui={ui as UISnapshot} sim={sim}>
          <App release={data} />
        </ReleaseProvider>
      </StrictMode>,
    );
  })
  .catch((error) =>
    root.render(
      <main>
        <h1>Release unavailable</h1>
        <p>{String(error)}</p>
      </main>,
    ),
  );

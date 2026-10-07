import { afterAll, afterEach, expect, test } from "bun:test";
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { RoundVideos } from "./RoundVideos";
import type { RoundVideoSpec } from "../lib/roundVideoSpecs";

GlobalRegistrator.register();
const { act, cleanup, fireEvent, render } = await import("@testing-library/react");
afterEach(cleanup);
afterAll(() => GlobalRegistrator.unregister());

test("unequal clips stop together and replay from their own starts", async () => {
  const originalRequest = globalThis.requestAnimationFrame;
  const originalCancel = globalThis.cancelAnimationFrame;
  const callbacks = new Map<number, FrameRequestCallback>();
  let next = 0;
  globalThis.requestAnimationFrame = callback => {
    callbacks.set(++next, callback);
    return next;
  };
  globalThis.cancelAnimationFrame = id => { callbacks.delete(id); };
  const specs: RoundVideoSpec[] = [1, 2].map(duration => ({
    policyName: `Policy ${duration}`, success: true, numSubtaskMarks: null,
    maxSubtaskMarks: 0, episodeIndex: 0, datasetRepo: "mulligan/test",
    cameraKey: "camera", videoUrl: "/fixture.mp4", badge: null,
    episode: { episodeIndex: 0, numFrames: duration * 15, duration,
      videoFileIndex: 0, fromTimestamp: 10, toTimestamp: 10 + duration },
  }));
  try {
    const view = render(<RoundVideos videos={specs} />);
    const videos = [...view.container.querySelectorAll("video")];
    for (const video of videos) {
      video.play = async () => {};
      video.pause = () => {};
      video.currentTime = 10;
    }
    const tick = async () => {
      const pending = [...callbacks.values()]; callbacks.clear();
      await act(async () => { for (const callback of pending) callback(0); });
    };
    fireEvent.click(view.getByText("Play"));
    videos[0].currentTime = 11;
    videos[1].currentTime = 11;
    await tick();
    expect(view.getByText("Pause")).toBeTruthy();
    expect(videos[0].currentTime).toBe(11);
    videos[1].currentTime = 12;
    await tick();
    expect(view.getByText("Play")).toBeTruthy();
    expect(callbacks.size).toBe(0);
    fireEvent.click(view.getByText("Play"));
    expect(videos.map(v => v.currentTime)).toEqual([10, 10]);
    cleanup();
  } finally {
    globalThis.requestAnimationFrame = originalRequest;
    globalThis.cancelAnimationFrame = originalCancel;
  }
});

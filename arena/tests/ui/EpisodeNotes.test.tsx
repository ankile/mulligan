import { afterAll, afterEach, expect, mock, test } from "bun:test";
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { getFunctionName } from "convex/server";

GlobalRegistrator.register();
const recent = { notes: "Recent note", count: 1, lastUsed: 20 };
const common = { notes: "Common note", count: 3, lastUsed: 10 };
const saveNotes = mock(async () => {});
// mock.module is process-wide in bun: keep the real exports so test files that load later
// (StageReview imports usePaginatedQuery) still resolve them.
const arenaClient = await import("../../src/lib/arenaClient");
mock.module("../../src/lib/arenaClient", () => ({
  ...arenaClient,
  useQuery: (query: Parameters<typeof getFunctionName>[0]) =>
    getFunctionName(query) === "reviews:episodeNotes"
      ? { notes: "Existing note" }
      : { recent: [recent, common], mostUsed: [common, recent] },
  useMutation: () => saveNotes,
}));
const { cleanup, fireEvent, render, waitFor } = await import("@testing-library/react");
const { EpisodeNotes } = await import("../../src/components/review/EpisodeNotes");
afterEach(() => { cleanup(); saveNotes.mockClear(); });
afterAll(() => { mock.restore(); GlobalRegistrator.unregister(); });

test("clicking a suggestion preserves the draft, focuses notes, and saves on blur", async () => {
  const view = render(<EpisodeNotes repoId="test/current" episodeIndex={4} />);
  const textarea = view.getByRole("textbox") as HTMLTextAreaElement;
  fireEvent.change(textarea, { target: { value: "Unsaved draft" } });
  fireEvent.click(view.getByRole("button", { name: "Recent note", exact: true }));
  expect(textarea.value).toBe("Unsaved draft\nRecent note");
  expect(document.activeElement).toBe(textarea);
  expect(saveNotes).not.toHaveBeenCalled();
  fireEvent.blur(textarea);
  await waitFor(() => expect(saveNotes).toHaveBeenCalledWith({
    dataset_repo: "test/current", episode_index: 4n, notes: "Unsaved draft\nRecent note",
  }));
});

test("most-used toggle reorders the pills", () => {
  const view = render(<EpisodeNotes repoId="test/current" episodeIndex={4} />);
  const pills = () => view.getAllByRole("button").filter((button) => button.title.startsWith("Recent note") || button.title.startsWith("Common note"));
  expect(pills().map((button) => button.textContent)).toEqual(["Recent note", "Common note"]);
  fireEvent.click(view.getByRole("button", { name: "Most used" }));
  expect(pills().map((button) => button.textContent)).toEqual(["Common note", "Recent note"]);
});

import { setSearchParams } from "./useSearchParam";

const TAB_KEYS = [
  "policy", "session", "mode", "round", "source", "task", "dataset", "episode",
  "outcome", "env", "sort", "policyA", "policyB", "pRound", "rollouts", "view",
  "join", "queue", "arm", "status", "sstatus", "sconf", "sflag", "sarm", "schema",
  "prediction", "blind",
];

/** Keep cleanup and tab selection in one guarded URL transition. */
export function navigateToAppTab(tab: string) {
  return setSearchParams({
    ...Object.fromEntries(TAB_KEYS.map((key) => [key, null])),
    tab: tab === "leaderboard" ? null : tab,
  });
}

export function navigateToDataset(repoId: string | null) {
  return setSearchParams({
    ...Object.fromEntries([
      "episode", "outcome", "view", "queue", "arm", "status", "prediction", "schema",
    ].map((key) => [key, null])),
    dataset: repoId,
  });
}

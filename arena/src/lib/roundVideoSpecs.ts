import { episodeForCamera, type EpisodeMetadata } from "./hf-api";

export interface RoundResult {
  policy_id: string;
  policyName: string;
  success: boolean;
  episode_index: number;
  /** Sub-goal marks reached; absent/null on rounds submitted without them. */
  num_subtask_marks?: number | null;
}

type EpisodeWithoutSuccess = Omit<EpisodeMetadata, "success">;

// One video tile in a RoundVideos grid. Each tile carries its OWN dataset +
// camera so a single synchronized grid can mix rollouts from different
// sessions/datasets (the joined-sessions view); single-session callers build
// specs via roundVideoSpecs().
export interface RoundVideoSpec {
  policyName: string;
  success: boolean;
  numSubtaskMarks: number | null;
  maxSubtaskMarks: number; // 0 = binary task
  episodeIndex: number;
  datasetRepo: string;
  cameraKey: string;
  /** Immutable release URL, including the actual camera chunk and file. */
  videoUrl?: string;
  episode: EpisodeWithoutSuccess | null; // null => no metadata for this episode
  /** Short arm label (`"3"`, or `"B3"` in the joined view); see `Arm.label`. */
  badge: string | null;
}

export function roundVideoSpecs(
  results: RoundResult[],
  datasetRepo: string,
  episodeMap: Map<number, EpisodeWithoutSuccess>,
  cameraKey: string,
  /**
   * Arm label per `policy_id` (every result's policy must be present), or
   * null for surfaces without arm labels (the Pairings page).
   */
  labelByPolicy: ReadonlyMap<string, string> | null,
  maxSubtaskMarks = 0,
): RoundVideoSpec[] {
  return results.map((result) => ({
    policyName: result.policyName,
    success: result.success,
    numSubtaskMarks: result.num_subtask_marks ?? null,
    maxSubtaskMarks,
    episodeIndex: result.episode_index,
    datasetRepo,
    cameraKey,
    episode: episodeMap.has(result.episode_index) ? episodeForCamera(episodeMap.get(result.episode_index)!, cameraKey) : null,
    badge: labelByPolicy === null ? null : labelFor(labelByPolicy, result.policy_id),
  }));
}

function labelFor(labelByPolicy: ReadonlyMap<string, string>, policyId: string): string {
  const label = labelByPolicy.get(policyId);
  if (label === undefined) throw new Error(`No arm label for policy ${policyId}`);
  return label;
}

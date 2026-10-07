import { describe, expect, test } from "bun:test";

import type { EntityStatus } from "../../convex/statusShared";
import { orderTaskChips } from "./taskChips";

describe("orderTaskChips", () => {
  const statuses = new Map<string, EntityStatus>([
    ["square_d2", "mainline"],
    ["routing_d2", "mainline"],
    ["pilot_task_v0", "retired"],
    ["franka_pick_cube", "retired"],
    ["marker_d2_ablate", "ablation"],
    ["smoke", "testing"],
  ]);

  test("mainline first, then ablation, testing, retired; alphabetical within", () => {
    const chips = orderTaskChips(
      ["smoke", "pilot_task_v0", "square_d2", "franka_pick_cube", "marker_d2_ablate", "routing_d2"],
      statuses,
    );
    expect(chips.map((c) => c.task)).toEqual([
      "routing_d2",
      "square_d2",
      "marker_d2_ablate",
      "smoke",
      "franka_pick_cube",
      "pilot_task_v0",
    ]);
  });

  test("tasks without a status row are treated as mainline and dedupe", () => {
    const chips = orderTaskChips(["new_line", "franka_pick_cube", "new_line"], statuses);
    expect(chips).toEqual([
      { task: "new_line", status: "mainline" },
      { task: "franka_pick_cube", status: "retired" },
    ]);
  });
});

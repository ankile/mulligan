import { describe, expect, test } from "bun:test";

import { datasetRoleLabel, resolvedDatasetRole } from "./dataset-classification";

describe("dataset classification", () => {
  test("uses the explicit dataset role", () => {
    expect(resolvedDatasetRole("training_view", "eval")).toBe("training_view");
    expect(datasetRoleLabel("training_view", "eval")).toBe("Training view");
  });

  test("classifies old eval and rollout records from their source type", () => {
    expect(resolvedDatasetRole(undefined, "eval")).toBe("eval_session");
    expect(datasetRoleLabel(undefined, "eval")).toBe("Eval session");
    expect(resolvedDatasetRole(undefined, "rollout")).toBe("rollout");
    expect(datasetRoleLabel(undefined, "rollout")).toBe("Rollout");
  });

  test("calls records without enough metadata unclassified", () => {
    expect(resolvedDatasetRole(undefined, "teleop")).toBeUndefined();
    expect(datasetRoleLabel(undefined, "teleop")).toBe("Unclassified");
  });
});

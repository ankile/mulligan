const ROLE_LABELS: Record<string, string> = {
  aggregate_parent: "Parent",
  training_view: "Training view",
  eval_session: "Eval session",
  rollout: "Rollout",
};

export function resolvedDatasetRole(
  datasetRole: string | undefined,
  sourceType: string
): string | undefined {
  if (datasetRole) return datasetRole;
  if (sourceType === "eval") return "eval_session";
  if (sourceType === "rollout") return "rollout";
  return undefined;
}

export function datasetRoleLabel(
  datasetRole: string | undefined,
  sourceType: string
): string {
  const resolvedRole = resolvedDatasetRole(datasetRole, sourceType);
  if (!resolvedRole) return "Unclassified";
  return ROLE_LABELS[resolvedRole] ?? resolvedRole.replaceAll("_", " ");
}

export type PolicyTags = { round?: number; method?: string; tags?: string[] };

export const SUGGESTED_METHODS = [
  "HG-DAgger baseline",
  "HG-DAgger mulligan",
  "HiL-IDQL mulligan",
];

export function normalizePolicyTags(input: {
  round: number | null; method: string | null; tags: string[];
}): PolicyTags {
  if (input.round !== null && (!Number.isSafeInteger(input.round) || input.round < 0)) {
    throw new Error("Round must be a nonnegative integer.");
  }
  const method = input.method?.trim() || undefined;
  if (method && method.length > 100) throw new Error("Method must be at most 100 characters.");
  const tags = [...new Set(input.tags.map((tag) => tag.trim()).filter(Boolean))];
  if (tags.length > 20 || tags.some((tag) => tag.length > 80)) {
    throw new Error("Use at most 20 tags, each at most 80 characters.");
  }
  return { round: input.round ?? undefined, method, tags };
}

export function policyTagOptions(policies: PolicyTags[]) {
  return {
    rounds: [...new Set(policies.flatMap((p) => p.round === undefined ? [] : [p.round]))].sort((a, b) => a - b),
    methods: [...new Set(policies.flatMap((p) => p.method ? [p.method] : []))].sort(),
    tags: [...new Set(policies.flatMap((p) => p.tags ?? []))].sort(),
  };
}

export function matchesPolicyTags(policy: PolicyTags, round: string, method: string, tag: string) {
  return (round === "" || (round === "untagged" ? policy.round === undefined : String(policy.round) === round))
    && (method === "" || (method === "untagged" ? !policy.method : policy.method === method))
    && (tag === "" || (policy.tags ?? []).includes(tag));
}

import { afterEach, beforeEach, expect, test } from "bun:test";
import { convexTest } from "convex-test";
import { api } from "../convex/_generated/api";
import schema from "../convex/schema";
import { validateTrajectoryEventLinks, type TrajectoryEventLink } from "../convex/trajectoryEventLinks";
import { seedStageReview } from "../src/lib/stagePredictionReview";
import type { ExportedStageSpec } from "../convex/stageConsistency";
import fixtures from "./fixtures/trajectory-review-fixtures.json";

const modules = {
  "../convex/_generated/server.ts": () => import("../convex/_generated/server"),
  "../convex/stageReviews.ts": () => import("../convex/stageReviews"),
  "../convex/stageTaskSpecs.ts": () => import("../convex/stageTaskSpecs"),
};
const source = fixtures.synthetic.tasks.find((task) => task.spec.task === "routing_d2")!;
const fixture = source.cases.find((row) => row.name === "valid_success")!;
const service = { serviceToken: "event-links-local-only" };
let t: ReturnType<typeof convexTest<typeof schema>>;
let oldToken: string | undefined;
beforeEach(async () => {
  oldToken = process.env.ARENA_SERVICE_TOKEN;
  process.env.ARENA_SERVICE_TOKEN = service.serviceToken;
  t = convexTest({ schema, modules, transactionLimits: true });
  await t.mutation(api.stageTaskSpecs.upsert, { ...service, task: source.spec.task,
    taxonomy_version: source.spec.taxonomy_version, taxonomy_hash: source.spec.taxonomy_hash,
    live: true, spec: source.spec, source: "test-only" });
});
afterEach(() => {
  if (oldToken === undefined) delete process.env.ARENA_SERVICE_TOKEN;
  else process.env.ARENA_SERVICE_TOKEN = oldToken;
});
function args() {
  const label = structuredClone(fixture.review_label!);
  label.trajectory_identity.sample_id = "test/event-links#episode=0";
  label.key_action_observations[0].first_time_s = 2;
  label.key_action_observations[0].occurrences[0].time_s = 2;
  return { ...service, reviewer_override: "test-reviewer", task: source.spec.task,
    dataset_repo: "test/event-links", taxonomy_version: source.spec.taxonomy_version, episode_index: 0n,
    status: "confirmed", label, episode_duration_s: fixture.duration_s };
}
const shared: TrajectoryEventLink = {
  action_id: "rope_grasped", stage_id: "controlled_rope_not_at_clip", attempt_index: 1, relation: "shared",
};

test("review associations survive reload outside the label without changing historical rows", async () => {
  const input = args();
  const original = await t.mutation(api.stageReviews.save, input);
  const next = await t.mutation(api.stageReviews.save, { ...input, event_links: [shared] });
  const [before, after] = await t.run(async (ctx) => [await ctx.db.get(original), await ctx.db.get(next)]);
  expect(before?.event_links).toBeUndefined();
  expect(after?.event_links).toEqual([shared]);
  expect(after?.label).toEqual(before?.label);
  expect(after?.label).not.toHaveProperty("event_links");
});

test("stale and mismatched associations are draftable but cannot become gold", async () => {
  for (const change of [
    (link: TrajectoryEventLink) => ({ ...link, stage_id: "deleted_stage" }),
    (link: TrajectoryEventLink) => ({ ...link, attempt_index: 2 }),
    (link: TrajectoryEventLink) => ({ ...link, action_id: "first_clip_contact" }),
  ]) {
    const input = { ...args(), event_links: [change(shared)] };
    const draft = await t.mutation(api.stageReviews.save, { ...input, status: "draft" });
    expect((await t.run((ctx) => ctx.db.get(draft)))?.event_links).toEqual(input.event_links);
    await expect(t.mutation(api.stageReviews.save, input)).rejects.toThrow(/association|timestamp/);
  }
});

test("semantic refs survive ordering but reject deleted or repeated targets", () => {
  const label = args().label;
  expect(validateTrajectoryEventLinks(label, [shared])).toEqual([]);
  label.stage_transitions.reverse();
  label.key_action_observations.reverse();
  expect(validateTrajectoryEventLinks(label, [shared])).toEqual([]);
  const action = label.key_action_observations.find((item) => item.action_id === shared.action_id)!;
  action.occurrences.push({ ...action.occurrences[0], time_s: 7 });
  expect(validateTrajectoryEventLinks(label, [shared]).join(" ")).toContain("unique");
  expect(validateTrajectoryEventLinks(args().label, [shared, shared]).join(" ")).toContain("Duplicate");
});

test("distinct associations retain different times and cannot bypass timeline validation", async () => {
  const input = args();
  const link = { ...shared, relation: "distinct" as const };
  input.label.key_action_observations[0].first_time_s = 1;
  input.label.key_action_observations[0].occurrences[0].time_s = 1;
  expect(validateTrajectoryEventLinks(input.label, [link])).toEqual([]);
  await expect(t.mutation(api.stageReviews.save, { ...input, event_links: [link] })).rejects.toThrow("timeline");
});

test("metadata seeds only from an owning human review and is copied independently", () => {
  const input = { spec: source.spec as ExportedStageSpec, outcome: null, legacy: false };
  const own = { label: args().label, attribution: {}, eventLinks: [shared] };
  const seeded = seedStageReview({ ...input, own });
  expect(seeded.eventLinks).toEqual([shared]);
  seeded.eventLinks![0].relation = "distinct";
  expect(own.eventLinks[0].relation).toBe("shared");
  expect(seedStageReview({ ...input, own: { ...own, label: null },
    prediction: { label: args().label, attribution: {} } }).eventLinks).toBeUndefined();
  expect(seedStageReview({ ...input, prediction: { label: args().label, attribution: {} } }).eventLinks).toBeUndefined();
});

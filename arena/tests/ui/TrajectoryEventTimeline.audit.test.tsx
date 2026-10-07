import { afterAll, afterEach, expect, test } from "bun:test";
import { useCallback, useState } from "react";
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { TrajectoryEventTimeline } from "../../src/components/review/TrajectoryEventTimeline";
import type { ExportedStageSpec, StageLabelRow } from "../../convex/stageConsistency";
import type { TrajectoryEventLink } from "../../convex/trajectoryEventLinks";
import fixtures from "../fixtures/trajectory-review-fixtures.json";

GlobalRegistrator.register({ url: "http://localhost/" });
const { cleanup, render, fireEvent } = await import("@testing-library/react");
afterEach(() => cleanup());
afterAll(() => GlobalRegistrator.unregister());
const source = fixtures.synthetic.tasks.find((task) => task.spec.task === "routing_d2")!;
const valid = source.cases.find((item) => item.name === "valid_success")!.review_label!;
function show(row: StageLabelRow, eventLinks: TrajectoryEventLink[] = []) {
  return render(<TrajectoryEventTimeline row={row} spec={source.spec as ExportedStageSpec} eventLinks={eventLinks}
    violations={Array.isArray(row.failure_events) ? [] : [{ code: "trajectory_shape", message: "failure_events must be an array", fields: ["failure_events"] }]} frame={0} markFrame={() => 0} markDisabled={false} onEdit={() => {}} onSeekTime={() => {}} disabled={false} />);
}

test("unfinished candidate time stays inspectable instead of crashing the event timeline", () => {
  const row = structuredClone(valid);
  row.key_action_observations[0].first_time_s = null;
  row.key_action_observations[0].occurrences[0].time_s = null;
  expect(() => show(row)).not.toThrow();
});

test("a separated pair with unfinished action time remains editable", () => {
  const row = structuredClone(valid);
  row.key_action_observations[0].first_time_s = null;
  row.key_action_observations[0].occurrences[0].time_s = null;
  expect(() => show(row, [{ action_id: "rope_grasped", stage_id: "controlled_rope_not_at_clip", attempt_index: 1, relation: "distinct" }])).not.toThrow();
});


test("malformed failure list reports invalid source without throwing during render", () => {
  expect(() => show({ ...structuredClone(valid), failure_events: null })).not.toThrow();
});


function AttemptFixture() {
  const initial = structuredClone(valid);
  initial.key_action_observations[0].first_time_s = 2;
  initial.key_action_observations[0].occurrences[0].time_s = 2;
  const [row, setRow] = useState<StageLabelRow>(initial);
  const [links, setLinks] = useState<TrajectoryEventLink[]>([{ action_id: "rope_grasped", stage_id: "controlled_rope_not_at_clip", attempt_index: 1, relation: "shared" }]);
  const [pending, setPending] = useState<Set<string>>(() => new Set());
  const onPending = useCallback((id: string, active: boolean) => setPending((old) => {
    const next = new Set(old); if (active) next.add(id); else next.delete(id); return next;
  }), []);
  return <><TrajectoryEventTimeline row={row} spec={source.spec as ExportedStageSpec} eventLinks={links} onEventLinksChange={setLinks}
    violations={[]} frame={0} markFrame={() => 0} markDisabled={false} onEdit={setRow} onSeekTime={() => {}} disabled={false}
    hasPendingInput={pending.size > 0} onPendingInputChange={onPending} />
    <output data-testid="audit-state">{JSON.stringify({ row, links, pending: pending.size })}</output></>;
}

test("clearing shared attempt text preserves the event and guards edits until a valid replacement", () => {
  const view = render(<AttemptFixture />);
  const input = view.getByLabelText("Action 1 occurrence 1 attempt") as HTMLInputElement;
  fireEvent.change(input, { target: { value: "" } });
  const empty = JSON.parse(view.getByTestId("audit-state").textContent!);
  expect(empty.pending).toBe(1);
  expect(empty.row.key_action_observations[0].occurrences[0].attempt_index).toBe(1);
  expect(empty.row.stage_transitions[1].attempt_index).toBe(1);
  expect(view.queryByLabelText("Transition 2 time")).toBeNull();
  expect((view.getByRole("button", { name: "Add transition" }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.change(input, { target: { value: "2" } });
  const complete = JSON.parse(view.getByTestId("audit-state").textContent!);
  expect(complete.pending).toBe(0);
  expect(complete.row.key_action_observations[0].occurrences[0].attempt_index).toBe(2);
  expect(complete.row.stage_transitions[1].attempt_index).toBe(2);
  expect(complete.links[0].attempt_index).toBe(2);
});

test("blind event summaries never expose malformed attempt or failure values", () => {
  const row = structuredClone(valid) as StageLabelRow;
  const sentinel = "POLICY_IDENTITY_SENTINEL";
  const transitions = row.stage_transitions as StageLabelRow[];
  transitions[0].attempt_index = sentinel;
  row.failure_mode = sentinel;
  row.primary_failure_time_s = 2;
  row.failure_events = [{ failure_mode_id: sentinel, time_s: 2, attempt_index: sentinel, confidence: "high", evidence: sentinel }];
  const view = render(<TrajectoryEventTimeline row={row} spec={source.spec as ExportedStageSpec} blind
    violations={[]} frame={0} markFrame={() => 0} markDisabled={false} onEdit={() => {}} onSeekTime={() => {}} disabled={false} />);
  expect(view.container.innerHTML).not.toContain(sentinel);
});

test("explicit chronological repair preserves times, event metadata and primary summary", () => {
  const initial = structuredClone(valid) as StageLabelRow;
  (initial.stage_transitions as StageLabelRow[]).reverse();
  let saved: StageLabelRow | null = null;
  const view = render(<TrajectoryEventTimeline row={initial} spec={source.spec as ExportedStageSpec}
    violations={[]} frame={0} markFrame={() => 0} markDisabled={false} onEdit={(row) => { saved = row; }} onSeekTime={() => {}} disabled={false} />);
  fireEvent.click(view.getByRole("button", { name: "Update recorded event order" }));
  expect(saved).not.toBeNull();
  expect(saved!.stage_transitions).toEqual(valid.stage_transitions);
  expect(saved!.failure_events).toEqual(initial.failure_events);
  expect(saved!.failure_mode).toEqual(initial.failure_mode);
  expect(saved!.primary_failure_time_s).toEqual(initial.primary_failure_time_s);
  expect(saved!.key_action_observations).toEqual(initial.key_action_observations);
  expect(initial.stage_transitions).toEqual([...valid.stage_transitions].reverse());
});

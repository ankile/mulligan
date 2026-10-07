import { afterAll, afterEach, describe, expect, test } from "bun:test";
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import StageReview from "../../src/components/StageReview";
import { AppTabNavigation } from "../../src/components/AppTabNavigation";
import { navigateToDataset } from "../../src/lib/appNavigation";
import { useSearchParam, useSearchParamNullable } from "../../src/lib/useSearchParam";
import { createStageReviewFixture, configureTrajectoryFixture } from "../browser/stageReviewFixture";

GlobalRegistrator.register({ url: "http://localhost/" });
const { act, cleanup, fireEvent, render } = await import("@testing-library/react");
afterEach(async () => { await act(async () => cleanup()); });
afterAll(() => GlobalRegistrator.unregister());

function GlobalNavigationFixture({ props }: { props: ReturnType<typeof createStageReviewFixture>["props"] }) {
  const [tab] = useSearchParam("tab", "leaderboard");
  const [repo] = useSearchParamNullable("dataset");
  const [view] = useSearchParam("view", "explorer");
  return <>
    <AppTabNavigation activeTab={tab} />
    <button onClick={() => navigateToDataset("org/other")}>Open another dataset</button>
    {tab === "explorer" && repo === "org/repo" && view === "stage"
      ? <StageReview {...props} onExit={() => { props.onExit(); navigateToDataset(null); }} />
      : <p data-testid="route">Route {tab} / {repo ?? "none"} / {view}</p>}
  </>;
}

function fixture(
  initialUrl = "?episode=0&prediction=A",
  initialReviews: Array<Record<string, unknown>> = [],
  globalNavigation = false,
) {
  window.history.replaceState(null, "", `/${initialUrl}`);
  const { state, props } = createStageReviewFixture(initialReviews);
  const element = () => globalNavigation ? <GlobalNavigationFixture props={props} /> : <StageReview {...props} />;
  const view = render(element());
  const rerender = () => view.rerender(element());
  return { state, view, rerender };
}

async function settle() { await act(async () => {}); }
function key(value: string) { fireEvent.keyDown(window, { key: value }); }

function chooseVersion(view: ReturnType<typeof render>, version: string) {
  fireEvent.change(view.getByRole("combobox", { name: "Prediction version" }), { target: { value: version } });
}

describe("stage prediction version review", () => {
  test("browsing and inspecting canonical predictions never saves or success-overlays them", async () => {
    const { state, view } = fixture();
    await settle();
    expect(state.saves).toHaveLength(0);
    expect(view.container.textContent).not.toContain("inherited from outcome");
    fireEvent.click(view.getByRole("button", { name: /model evidence/ }));
    expect(view.container.textContent).not.toContain("raw-A");
    expect(state.armFetches).toBe(0);
    await act(async () => fireEvent.click(view.getAllByRole("button", { name: "Show provenance and unblind" })[0]));
    expect(view.container.textContent).toContain("raw-A");
    expect(view.container.textContent).toContain("max_stage: S3");
    expect(view.container.textContent).toContain("Review source: prediction A-prediction-0");
    expect(state.saves).toHaveLength(0);
  });

  test("default pins active ID and reactive publication never switches the working version", async () => {
    const { state, view, rerender } = fixture("?episode=0");
    await settle();
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("A");
    state.active = "B";
    await act(async () => rerender());
    expect((view.getByRole("combobox", { name: "Prediction version" }) as HTMLSelectElement).value).toBe("A");
    expect(view.container.textContent).toContain("Inactive prediction version: A");
    expect(state.saves).toHaveLength(0);
  });

  test("invalid version fails closed and cannot create a source-free review", async () => {
    const { state, view } = fixture("?episode=0&prediction=does-not-exist");
    await settle();
    expect(view.getByRole("alert").textContent).toContain("is not published for this dataset");
    key("5"); key("u");
    await settle();
    expect(state.saves).toHaveLength(0);
    expect((view.getByRole("button", { name: /^uncertain/ }) as HTMLButtonElement).disabled).toBe(true);
  });

  test("switch awaits the old draft save and preserves its identity on resumed own labels", async () => {
    const { state, view } = fixture();
    await settle();
    let release: (() => void) | undefined;
    state.save = () => new Promise<void>((resolve) => { release = resolve; });
    key("4");
    await act(async () => chooseVersion(view, "B"));
    expect(state.saves).toHaveLength(1);
    expect(state.saves[0].label?.max_stage).toBe(4);
    expect(state.saves[0].prediction_id).toBe("A-prediction-0");
    expect(state.saves[0].prediction_sha256).toBe("A".repeat(64));
    expect(state.saves[0].episode_duration_s).toBe(12);
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("A");
    expect((view.getByRole("combobox", { name: "Prediction version" }) as HTMLSelectElement).disabled).toBe(true);
    key("8"); key("u");
    expect(state.saves).toHaveLength(1);
    await act(async () => release?.());
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("B");
    expect(view.container.textContent).toContain("Your saved label is in the form");
    expect(view.container.textContent).toContain("Review source: prediction A-prediction-0");
    state.save = async () => {};
    await act(async () => fireEvent.click(view.getByRole("button", { name: /^uncertain/ })));
    expect(state.saves[1].label?.max_stage).toBe(4);
    expect(state.saves[1].prediction_id).toBe("A-prediction-0");
    expect(state.saves[1].episode_duration_s).toBe(12);
  });

  test("save failures keep edits and selection in place for retry", async () => {
    const { state, view } = fixture();
    await settle();
    state.save = async () => { throw new Error("offline"); };
    key("4");
    await act(async () => chooseVersion(view, "B"));
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("A");
    expect(view.container.textContent).toContain("Episode 0: offline");
    state.save = async () => {};
    await act(async () => chooseVersion(view, "B"));
    expect(state.saves[1].label?.max_stage).toBe(4);
    expect(state.saves[1].prediction_id).toBe("A-prediction-0");
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("B");
  });

  test("a partially loaded version hides stale marks and disables editing and saves", async () => {
    const { state, view, rerender } = fixture();
    await settle();
    state.ready = false;
    await act(async () => { chooseVersion(view, "B"); rerender(); });
    expect(view.getByRole("status").textContent).toContain("complete version loads");
    expect(view.container.textContent).not.toContain("Review source: prediction A-prediction-0");
    key("8"); key("u");
    await settle();
    expect(state.saves).toHaveLength(0);
    state.ready = true;
    await act(async () => rerender());
    expect(view.container.textContent).toContain("Review source: prediction B-prediction-0");
  });

  test("explicit historical selection changes selected evidence without writing reviews", async () => {
    const { state, view } = fixture("?episode=0&prediction=B");
    await settle();
    expect(view.container.textContent).toContain("Inactive prediction version: B");
    fireEvent.click(view.getByRole("button", { name: /model evidence/ }));
    await act(async () => fireEvent.click(view.getAllByRole("button", { name: "Show provenance and unblind" })[0]));
    expect(view.container.textContent).toContain("raw-B");
    expect(view.container.textContent).not.toContain("raw-A");
    await act(async () => chooseVersion(view, "none"));
    expect(view.container.textContent).toContain("No prediction source recorded");
    expect(view.container.textContent).toContain("inherited from outcome");
    expect(state.saves).toHaveLength(0);
  });

  test("external URL switches retain unsaved edits under their original prediction", async () => {
    const { state, view } = fixture();
    await settle();
    key("4");
    await act(async () => {
      window.history.replaceState(null, "", "/?episode=0&prediction=B");
      window.dispatchEvent(new Event("popstate"));
    });
    expect(view.container.textContent).toContain("Unsaved edits are retained");
    expect(view.container.textContent).toContain("Review source: prediction B-prediction-0");
    await act(async () => chooseVersion(view, "A"));
    await act(async () => fireEvent.click(view.getByRole("button", { name: /^uncertain/ })));
    expect(state.saves).toHaveLength(1);
    expect(state.saves[0].label?.max_stage).toBe(4);
    expect(state.saves[0].prediction_id).toBe("A-prediction-0");
  });
  test("edits to committed human labels block navigation instead of discarding or demoting them", async () => {
    const { state, view } = fixture("?episode=0&prediction=B&sstatus=all", [{
      _id: "own-review", episode_index: 0n, reviewer: "annotator", reviewer_user_id: "user-1", status: "confirmed", saved_at: 1_700_000_100_000,
      label: { max_stage: 5 }, prediction_id: "A-prediction-0", prediction_sha256: "A".repeat(64),
      episode_duration_s: 12,
    }]);
    await settle();
    key("6");
    await act(async () => chooseVersion(view, "A"));
    expect(view.container.textContent).toContain("unsaved edits to your confirmed review are still in the form");
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("B");
    expect(state.saves).toHaveLength(0);
    await act(async () => fireEvent.click(view.getByRole("button", { name: /^uncertain/ })));
    expect(state.saves[0].label?.max_stage).toBe(6);
    expect(state.saves[0].prediction_id).toBe("A-prediction-0");
  });

  test("completed pagination with a manifest count mismatch fails closed", async () => {
    const { state, view, rerender } = fixture();
    await settle();
    state.runs[0].expected_count = 3;
    await act(async () => rerender());
    expect(view.getByRole("alert").textContent).toContain("its published manifest requires 3");
    key("u");
    await settle();
    expect(state.saves).toHaveLength(0);
  });

  test("leaving cannot discard dirty drafts retained after an external URL change", async () => {
    const { state, view } = fixture();
    await settle();
    key("4");
    await act(async () => {
      window.history.replaceState(null, "", "/?episode=0&prediction=B");
      window.dispatchEvent(new Event("popstate"));
    });
    await act(async () => fireEvent.click(view.getByRole("button", { name: /Back to explorer/ })));
    expect(state.exits).toBe(0);
    expect(view.container.textContent).toContain("Save the retained edits");
    const event = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
  });

  test("copied human labels save the donor review and its exact original prediction source", async () => {
    const { state, view } = fixture("?episode=0&prediction=A&sstatus=adjudicate", [{
      _id: "donor-review", episode_index: 0n, reviewer: "other", reviewer_user_id: "user-2",
      status: "uncertain", saved_at: 1_700_000_100_000, label: { max_stage: 8 },
      prediction_id: "B-prediction-0", prediction_sha256: "B".repeat(64), episode_duration_s: 20,
    }]);
    await settle();
    fireEvent.click(view.getByRole("button", { name: "load their label" }));
    expect(view.container.textContent).toContain("Copied from human review donor-review");
    await act(async () => fireEvent.click(view.getByRole("button", { name: /^uncertain/ })));
    expect(state.saves[0].copied_from_review_id).toBe("donor-review");
    expect(state.saves[0].prediction_id).toBe("B-prediction-0");
    expect(state.saves[0].prediction_sha256).toBe("B".repeat(64));
    expect(state.saves[0].episode_duration_s).toBe(20);
    expect(state.saves[0].label?.max_stage).toBe(8);
  });

  test("stable user ID finds an own review saved under a previous username", async () => {
    const { state, view } = fixture("?episode=0&prediction=A", [{
      _id: "old-username-review", episode_index: 0n, reviewer: "previous-name", reviewer_user_id: "user-1",
      status: "draft", saved_at: 1_700_000_100_000, label: { max_stage: 8 },
      prediction_id: "B-prediction-0", prediction_sha256: "B".repeat(64), episode_duration_s: 20,
    }]);
    await settle();
    expect(view.container.textContent).toContain("Your saved label is in the form");
    await act(async () => fireEvent.click(view.getByRole("button", { name: /^uncertain/ })));
    expect(state.saves[0].prediction_id).toBe("B-prediction-0");
    expect(state.saves[0].label?.max_stage).toBe(8);
  });

  test("a same-name service review without the signed-in user's ID cannot seed their form", async () => {
    const { state, view } = fixture("?episode=0&prediction=A", [{
      _id: "service-review", episode_index: 0n, reviewer: "annotator",
      status: "confirmed", saved_at: 1_700_000_100_000, label: { max_stage: 8 },
      prediction_id: "B-prediction-0", prediction_sha256: "B".repeat(64), episode_duration_s: 20,
    }]);
    await settle();
    expect(view.container.textContent).not.toContain("Your saved label is in the form");
    await act(async () => fireEvent.click(view.getByRole("button", { name: /^uncertain/ })));
    expect(state.saves[0].prediction_id).toBe("A-prediction-0");
    expect(state.saves[0].label?.max_stage).toBe(3);
  });

  test("blind review hides arbitrary provenance and notes, then records explicit unblinding", async () => {
    const { state, view, rerender } = fixture();
    // These values arrive before the asynchronous episode fetch seeds the form.
    state.predictionOverrides = {
      label: { max_stage: 3, notes: "sensitive-policy-notes" },
      canonical_response: { notes: "sensitive-canonical-policy" },
      evidence: { arm: "sensitive-arm" },
      pipeline: { name: "sensitive-pipeline", version: "v1", git_commit: "abc" },
      review_reason: "sensitive-review-reason", vote_summary: { "sensitive-vote": 1 },
    };
    state.runs[0] = { ...state.runs[0], run_key: "sensitive-run-key" };
    await act(async () => rerender());
    fireEvent.click(view.getByRole("button", { name: /model evidence/ }));
    expect(view.container.innerHTML).not.toContain("sensitive-");
    expect(state.armFetches).toBe(0);
    await act(async () => fireEvent.click(view.getAllByRole("button", { name: "Show provenance and unblind" })[0]));
    expect(view.container.innerHTML).toContain("sensitive-policy-notes");
    expect(view.container.innerHTML).toContain("sensitive-canonical-policy");
    expect(view.container.innerHTML).toContain("sensitive-arm");
    expect(view.container.innerHTML).toContain("sensitive-run-key");
    // Re-hiding identity does not erase the fact it was shown in this session.
    fireEvent.click(view.getByRole("button", { name: /unblinded/ }));
    await act(async () => fireEvent.click(view.getByRole("button", { name: /^uncertain/ })));
    expect(state.saves[0].blind).toBe(false);
    expect(state.saves[0].label?.notes).toBe("sensitive-policy-notes");
  });

  test("invalid known-field values cannot reveal policy identity anywhere in blind DOM", async () => {
    const { state, view, rerender } = fixture();
    state.predictionOverrides = {
      label: { max_stage: "hidden-policy-stage", failure_mode: "hidden-policy-failure", final_state: "hidden-policy-final", notes: "hidden-policy-notes" },
      confidence: "hidden-policy-confidence", canonical_response: { notes: "hidden-policy-raw" },
    };
    await act(async () => rerender());
    fireEvent.click(view.getByRole("button", { name: /model evidence/ }));
    expect(view.container.innerHTML).not.toContain("hidden-policy-");
    expect(view.container.textContent).toContain("invalid stage");
    expect(view.container.textContent).toContain("invalid confidence");
    expect(view.container.textContent).toContain("unparseable_stage");
    await act(async () => fireEvent.click(view.getAllByRole("button", { name: "Show provenance and unblind" })[0]));
    expect(view.container.innerHTML).toContain("hidden-policy-stage");
    expect(view.container.innerHTML).toContain("hidden-policy-failure");
    expect(view.container.innerHTML).toContain("hidden-policy-final");
  });

  test("actual App tab navigation cannot unmount a dirty review or partially clear its URL", async () => {
    const { state, view } = fixture("?tab=explorer&dataset=org%2Frepo&view=stage&episode=0&prediction=A", [], true);
    await settle();
    key("4");
    const sourceUrl = window.location.href;
    fireEvent.click(view.getByRole("button", { name: "Eval Sessions" }));
    expect(window.location.href).toBe(sourceUrl);
    expect(view.queryByTestId("route")).toBeNull();
    expect(view.container.textContent).toContain("Save them before changing tabs or datasets");
    expect(state.saves).toHaveLength(0);
    await act(async () => fireEvent.click(view.getByRole("button", { name: /^uncertain/ })));
    fireEvent.click(view.getByRole("button", { name: "Eval Sessions" }));
    expect(view.getByTestId("route").textContent).toContain("Route sessions / none / explorer");
    expect(new URLSearchParams(window.location.search).has("prediction")).toBe(false);
    expect(state.saves[0].prediction_id).toBe("A-prediction-0");
    expect(state.saves[0].label?.max_stage).toBe(4);
  });

  test("DataExplorer dataset navigation blocks dirty edits; the review's awaited exit can leave after saving", async () => {
    const { state, view } = fixture("?tab=explorer&dataset=org%2Frepo&view=stage&episode=0&prediction=A", [], true);
    await settle();
    key("4");
    fireEvent.click(view.getByRole("button", { name: "Open another dataset" }));
    expect(new URLSearchParams(window.location.search).get("dataset")).toBe("org/repo");
    expect(view.queryByTestId("route")).toBeNull();
    await act(async () => fireEvent.click(view.getByRole("button", { name: /Back to explorer/ })));
    expect(state.saves[0].status).toBe("draft");
    expect(state.saves[0].prediction_id).toBe("A-prediction-0");
    expect(view.getByTestId("route").textContent).toContain("Route explorer / none / explorer");
  });

  test("global navigation is blocked during a save and after its failure", async () => {
    const { state, view } = fixture("?tab=explorer&dataset=org%2Frepo&view=stage&episode=0&prediction=A", [], true);
    await settle();
    let rejectSave: ((error: Error) => void) | undefined;
    state.save = () => new Promise<void>((_resolve, reject) => { rejectSave = reject; });
    key("4");
    await act(async () => chooseVersion(view, "B"));
    fireEvent.click(view.getByRole("button", { name: "Eval Sessions" }));
    expect(view.container.textContent).toContain("Stay on this page until the save finishes");
    expect(view.queryByTestId("route")).toBeNull();
    await act(async () => rejectSave?.(new Error("offline")));
    fireEvent.click(view.getByRole("button", { name: "Eval Sessions" }));
    expect(view.queryByTestId("route")).toBeNull();
    expect(state.saves).toHaveLength(1);
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("A");
  });

  test("Back/Forward route changes are stopped before parent subscribers unmount dirty review", async () => {
    const { state, view } = fixture("?tab=explorer&dataset=org%2Frepo&view=stage&episode=0&prediction=A", [], true);
    await settle();
    key("4");
    const sourceUrl = window.location.href;
    await act(async () => {
      // Browser traversal changes the URL before dispatching popstate.
      window.history.pushState(null, "", "/?tab=sessions");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(window.location.href).toBe(sourceUrl);
    expect(view.queryByTestId("route")).toBeNull();
    expect(state.saves).toHaveLength(0);
    await act(async () => fireEvent.click(view.getByRole("button", { name: /^uncertain/ })));
    await act(async () => {
      window.history.pushState(null, "", "/?tab=sessions");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(view.getByTestId("route").textContent).toContain("Route sessions");
  });

  test("global navigation also protects a dirty draft retained under another prediction version", async () => {
    const { state, view } = fixture("?tab=explorer&dataset=org%2Frepo&view=stage&episode=0&prediction=A", [], true);
    await settle();
    key("4");
    await act(async () => {
      const url = new URL(window.location.href);
      url.searchParams.set("prediction", "B");
      window.history.replaceState(null, "", url);
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(view.container.textContent).toContain("Unsaved edits are retained");
    fireEvent.click(view.getByRole("button", { name: "Eval Sessions" }));
    expect(view.queryByTestId("route")).toBeNull();
    expect(state.saves).toHaveLength(0);
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("B");
  });

});


describe("generic trajectory review", () => {
  function genericFixture() {
    window.history.replaceState(null, "", "/?episode=0&prediction=A");
    const fixture = createStageReviewFixture();
    const contract = configureTrajectoryFixture(fixture);
    const view = render(<StageReview {...fixture.props} />);
    return { ...fixture, ...contract, view };
  }
  test("preserves historical events and final failure while inspecting full generic predictions", async () => {
    const { state, view, selected } = genericFixture();
    await settle();
    expect(view.getByTestId("trajectory-form")).toBeTruthy();
    expect((view.getByRole("combobox", { name: "Task success" }) as HTMLSelectElement).value).toBe("false");
    expect(view.getByRole("group", { name: "Maximum stage" }).querySelector('[aria-pressed="true"]')?.textContent).toBe(`S${selected.review_label!.max_stage}`);
    expect(state.saves).toHaveLength(0);
    fireEvent.click(view.getByRole("button", { name: /model evidence/ }));
    expect(view.container.textContent).toContain(`Prediction source revision: ${"f".repeat(40)}`);
    expect(view.container.textContent).toContain("Prediction duration: 30s");
    expect(state.saves).toHaveLength(0);
  });
  test("shared occurrence attempt edits update both references and retain exact source attribution", async () => {
    const { state, view, selected } = genericFixture();
    await settle();
    const original = structuredClone(selected.review_label!);
    const attemptInput = view.getByRole("textbox", { name: "Action 1 occurrence 1 attempt" });
    fireEvent.change(attemptInput, { target: { value: "" } });
    await act(async () => chooseVersion(view, "B"));
    expect(state.saves).toHaveLength(0);
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("A");
    expect(view.getByRole("textbox", { name: "Action 1 occurrence 1 attempt" })).toBe(attemptInput);
    fireEvent.change(attemptInput, { target: { value: "2" } });
    fireEvent.blur(attemptInput);
    await act(async () => chooseVersion(view, "B"));
    expect(state.saves).toHaveLength(1);
    const saved = state.saves[0];
    expect(saved.prediction_id).toBe("A-prediction-0");
    expect(saved.taxonomy_version).toBe("trajectory-review/v1/routing_d2/v1");
    const expectedTransitions = structuredClone(original.stage_transitions);
    expectedTransitions.find((event) => event.to_stage_id === "controlled_rope_not_at_clip")!.attempt_index = 2;
    expect(saved.label!.stage_transitions).toEqual(expectedTransitions);
    expect(saved.label!.failure_events).toEqual(original.failure_events);
    const expected = structuredClone(original.key_action_observations);
    expected[0].occurrences[0].attempt_index = 2;
    expect(saved.label!.key_action_observations).toEqual(expected);
    expect(saved.label!.trajectory_identity).toEqual(original.trajectory_identity);
    expect(saved.status).toBe("draft");
  });
  test("event edits and unknown scalar values remain blind, then explicit unblind reveals evidence", async () => {
    window.history.replaceState(null, "", "/?episode=0&prediction=A");
    const fixture = createStageReviewFixture();
    const { selected } = configureTrajectoryFixture(fixture);
    const label = structuredClone(selected.review_label!);
    const sentinel = "POLICY_IDENTITY_SENTINEL";
    label.confidence = sentinel;
    label.notes = sentinel;
    label.stage_transitions[0].confidence = sentinel;
    label.stage_transitions[0].evidence = sentinel;
    fixture.state.predictionOverrides.label = label;
    const view = render(<StageReview {...fixture.props} />);
    await settle();
    expect(view.container.textContent).not.toContain(sentinel);
    expect(view.container.innerHTML).not.toContain(sentinel);
    await act(async () => fireEvent.click(view.getAllByRole("button", { name: "Show provenance and unblind" })[0]));
    expect(view.getByText(`Retained source evidence: ${sentinel}`)).toBeTruthy();
    fireEvent.change(view.getByRole("textbox", { name: "Your review notes" }), { target: { value: "reviewed evidence" } });
    await act(async () => key("u"));
    expect(fixture.state.saves[0].blind).toBe(false);
    expect(fixture.state.saves[0].label!.notes).toBe(sentinel);
    expect(fixture.state.saves[0].notes).toBe("reviewed evidence");
    expect(fixture.state.saves[0].review_protocol).toBe("structured-v1");
    expect((view.queryByRole("textbox", { name: "Transition 1 evidence" }) as HTMLTextAreaElement | null)?.readOnly ?? true).toBe(true);
  });
  test("other-schema availability waits for saving and never translates the old label", async () => {
    const { state, props, view } = genericFixture();
    await settle();
    state.otherSchemas = [{ taxonomy_version: "trajectory-review/v1/routing_d2/v2", run_id: "C", expected_count: 26, published_at: 2 }];
    await act(async () => view.rerender(<StageReview {...props} />));
    let release: (() => void) | undefined;
    state.save = () => new Promise<void>((resolve) => { release = resolve; });
    fireEvent.change(view.getByRole("spinbutton", { name: "Attempt count" }), { target: { value: "3" } });
    await act(async () => fireEvent.click(view.getByRole("button", { name: /Open trajectory-review.*26 episodes/ })));
    expect(state.saves[0].taxonomy_version).toBe("trajectory-review/v1/routing_d2/v1");
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("A");
    await act(async () => release?.());
    expect(new URLSearchParams(window.location.search).get("schema")).toBe("trajectory-review/v1/routing_d2/v2");
    expect(new URLSearchParams(window.location.search).get("prediction")).toBe("C");
  });
});


test("generic timestamps keep exact source precision through untouched blur and explicit edits", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture);
  const label = structuredClone(selected.review_label!);
  label.stage_transitions[0].time_s = 1.234567;
  fixture.state.predictionOverrides.label = label;
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  const timeInput = view.getByRole("group", { name: "Transition 1 time" }).querySelector("input")!;
  fireEvent.focus(timeInput); fireEvent.blur(timeInput);
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves).toHaveLength(0);
  const nextInput = view.getByRole("group", { name: "Transition 1 time" }).querySelector("input")!;
  fireEvent.change(nextInput, { target: { value: "1.234568" } }); fireEvent.blur(nextInput);
  await act(async () => chooseVersion(view, "A"));
  expect(fixture.state.saves).toHaveLength(1);
  expect((fixture.state.saves[0].label!.stage_transitions as Array<{ time_s: number }>)[0].time_s).toBe(1.234568);
  expect(fixture.state.saves[0].prediction_id).toBe("B-prediction-0");
});


test("focused timestamp edits participate in actual global navigation and beforeunload guards before blur", async () => {
  const { state, view } = fixture("?tab=explorer&dataset=org%2Frepo&view=stage&episode=0&prediction=A", [], true);
  await settle();
  const input = view.container.querySelector('input[inputmode="decimal"]')!;
  fireEvent.focus(input); fireEvent.change(input, { target: { value: "12.345" } });
  const unload = new Event("beforeunload", { cancelable: true });
  window.dispatchEvent(unload);
  expect(unload.defaultPrevented).toBe(true);
  fireEvent.click(view.getByRole("button", { name: "Leaderboard" }));
  expect(view.queryByTestId("route")).toBeNull();
  expect((input as HTMLInputElement).value).toBe("12.345");
  expect(state.saves).toHaveLength(0);
});

test("invalid timestamp text stays visible and blocks saves and every source navigation until repaired", async () => {
  const { state, view } = fixture("?tab=explorer&dataset=org%2Frepo&view=stage&episode=0&prediction=A", [], true);
  await settle();
  const input = view.container.querySelector('input[inputmode="decimal"]')!;
  fireEvent.change(input, { target: { value: "unfinished" } }); fireEvent.blur(input);
  expect((input as HTMLInputElement).value).toBe("unfinished");
  expect(input.getAttribute("aria-invalid")).toBe("true");
  const unload = new Event("beforeunload", { cancelable: true }); window.dispatchEvent(unload);
  expect(unload.defaultPrevented).toBe(true);
  await act(async () => chooseVersion(view, "B"));
  fireEvent.click(view.getByRole("button", { name: "Leaderboard" }));
  await act(async () => key("u"));
  expect(new URLSearchParams(window.location.search).get("prediction")).toBe("A");
  expect(view.queryByTestId("route")).toBeNull();
  expect(state.saves).toHaveLength(0);
  fireEvent.change(input, { target: { value: "2.5" } });
  await act(async () => chooseVersion(view, "B"));
  expect(state.saves).toHaveLength(1);
  expect(state.saves[0].label!.rope_grasped_time_s).toBe(2.5);
  expect(new URLSearchParams(window.location.search).get("prediction")).toBe("B");
});

test("fractional episode URLs cannot throw during the cross-schema availability query", async () => {
  const { state, view } = fixture("?episode=1.2&prediction=A");
  await settle();
  expect(view.container.textContent).toContain("Stage review");
  expect(state.saves).toHaveLength(0);
});

for (const control of ["clear", "mark"] as const) {
  test(`generic ${control} resolves invalid local timestamp text even when the stored value stays the same`, async () => {
    window.history.replaceState(null, "", "/?episode=0&prediction=A");
    const fixture = createStageReviewFixture(); const { selected } = configureTrajectoryFixture(fixture);
    const label = structuredClone(selected.review_label!);
    label.primary_failure_time_s = null; label.failure_events[0].time_s = null; label.stage_transitions[0].time_s = 0;
    fixture.state.predictionOverrides.label = label;
    const view = render(<StageReview {...fixture.props} />); await settle();
    const groupName = control === "clear" ? "Failure 1 time" : "Transition 1 time";
    let group = view.getByRole("group", { name: groupName });
    fireEvent.change(group.querySelector("input")!, { target: { value: "-" } });
    const button = control === "clear" ? group.querySelector('[title="Clear Failure 1 time"]')!
      : Array.from(group.querySelectorAll("button")).find((item) => item.textContent!.includes("mark"))!;
    fireEvent.click(button); await settle();
    group = view.getByRole("group", { name: groupName });
    expect((group.querySelector("input") as HTMLInputElement).value).toBe(control === "clear" ? "" : "0");
    expect(view.container.textContent).not.toContain("unfinished or invalid text");
    await act(async () => key("u"));
    expect(fixture.state.saves).toHaveLength(1);
  });
}

test("source-free generic annotation waits for policy signals and pins the valid prefix instead of the reset tail", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture(); configureTrajectoryFixture(fixture);
  fixture.state.missingPredictionEpisodes.add(0);
  fixture.state.runs = fixture.state.runs.map((run) => ({ ...run, expected_count: 1 }));
  let release: (() => void) | undefined;
  fixture.state.fetchSignals = () => new Promise((resolve) => { release = () => resolve({ detectedOutcome: "failure",
    validLength: 120, lastValidFrame: 119, doneOnsetFrame: null, rewardSpikeFrames: [] }); });
  const view = render(<StageReview {...fixture.props} />); await settle();
  expect(view.container.textContent).toContain("Loading the validated policy-phase duration");
  expect(view.queryByTestId("trajectory-form")).toBeNull();
  await act(async () => key("u")); expect(fixture.state.saves).toHaveLength(0);
  await act(async () => release?.());
  expect(view.container.textContent).toContain("policy 8.0s / raw 450f");
  expect((view.getByRole("combobox", { name: "Task success" }) as HTMLSelectElement).value).toBe("__invalid__");
  fireEvent.click(view.getByRole("button", { name: "Add failure event" }));
  const input = view.getByRole("group", { name: "Failure 1 time" }).querySelector("input")!;
  fireEvent.change(input, { target: { value: "9" } });
  expect(view.container.textContent).toContain("within the episode duration (8.000000 seconds)");
  await act(async () => key("u"));
  expect(fixture.state.saves[0].episode_duration_s).toBe(8);
  expect(fixture.state.saves[0].prediction_id).toBeUndefined();
  expect(fixture.state.saves[0].label!.task_success).toBeNull();
});

test("source-free policy-signal errors fail closed without silently using raw length", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture(); configureTrajectoryFixture(fixture);
  fixture.state.missingPredictionEpisodes.add(0);
  fixture.state.runs = fixture.state.runs.map((run) => ({ ...run, expected_count: 1 }));
  fixture.state.fetchSignals = async () => { throw new Error("invalid policy prefix"); };
  const view = render(<StageReview {...fixture.props} />); await settle();
  expect(view.container.textContent).toContain("Cannot determine policy-phase duration: invalid policy prefix");
  expect(view.queryByTestId("trajectory-form")).toBeNull();
  await act(async () => key("u")); expect(fixture.state.saves).toHaveLength(0);
});

for (const task of ["marker_d2", "square_d2", "routing_d2"]) {
  test(`${task}: stage buttons edit only the human stage judgment and preserve full event history`, async () => {
    window.history.replaceState(null, "", "/?episode=0&prediction=A");
    const fixture = createStageReviewFixture();
    const { selected } = configureTrajectoryFixture(fixture, `${task}_${task === "routing_d2" ? "v1" : "v3"}`, `real_${task}_valid`);
    const original = structuredClone(selected.review_label!);
    const view = render(<StageReview {...fixture.props} />);
    await settle();
    const form = view.getByTestId("trajectory-form");
    expect(form.className).not.toContain("overflow-y-auto");
    expect(view.getByRole("region", { name: "Episode events" })).toBeTruthy();
    expect(form.querySelectorAll("article").length).toBeGreaterThan(0);
    expect(fixture.state.saves).toHaveLength(0);
    const stage = view.getByRole("button", { name: /^S1:/ });
    fireEvent.click(stage);
    expect(stage.getAttribute("aria-pressed")).toBe("true");
    await act(async () => key("u"));
    expect(fixture.state.saves).toHaveLength(1);
    const saved = fixture.state.saves[0];
    expect(saved.prediction_id).toBe("A-prediction-0");
    expect(saved.label!.max_stage).toBe(1);
    for (const field of ["stage_transitions", "failure_events", "key_action_observations", "task_success", "final_state", "trajectory_identity"]) {
      expect(saved.label![field]).toEqual(original[field]);
    }
  });
}

test("collapsing event details preserves unfinished input and blocks event remapping until repaired", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  configureTrajectoryFixture(fixture);
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  const input = view.getByRole("group", { name: "Transition 1 time" }).querySelector("input")!;
  const section = input.closest("article")!.querySelector("details")!;
  section.open = true;
  fireEvent.change(input, { target: { value: "unfinished" } });
  section.open = false;
  await settle();
  expect(view.getByRole("combobox", { name: "Event display order" }).hasAttribute("disabled")).toBe(true);
  expect(view.getByRole("button", { name: "Remove Transition 1" }).hasAttribute("disabled")).toBe(true);
  expect(view.getByRole("button", { name: "Add transition" }).hasAttribute("disabled")).toBe(true);
  await act(async () => key("u"));
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves).toHaveLength(0);
  expect(new URLSearchParams(window.location.search).get("prediction")).toBe("A");
  section.open = true;
  expect(view.getByRole("group", { name: "Transition 1 time" }).querySelector("input")).toBe(input);
  expect(input.value).toBe("unfinished");
  fireEvent.change(input, { target: { value: "1.234567" } });
  await settle();
  expect(view.getByRole("button", { name: "Remove Transition 1" }).hasAttribute("disabled")).toBe(false);
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves).toHaveLength(1);
  expect(fixture.state.saves[0].prediction_id).toBe("A-prediction-0");
  expect(fixture.state.saves[0].label!.stage_transitions[0].time_s).toBe(1.234567);
});

test("blind form explains correctable event conflicts without exposing arbitrary shape keys", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture);
  const label = structuredClone(selected.review_label!);
  label.stage_transitions[0].POLICY_IDENTITY_SENTINEL = "hidden";
  label.key_action_observations[0].first_time_s = 999;
  fixture.state.predictionOverrides.label = label;
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  expect(view.container.textContent).toContain("Action 1 first time must match its earliest occurrence");
  expect(view.container.textContent).toContain("Transition 1 has an invalid structure");
  expect(view.container.innerHTML).not.toContain("POLICY_IDENTITY_SENTINEL");
  expect(fixture.state.armFetches).toBe(0);
  expect(fixture.state.saves).toHaveLength(0);
});

test("Space retains native activation on focused stage buttons and event summaries", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  configureTrajectoryFixture(fixture);
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  for (const control of [view.getByRole("button", { name: /^S1:/ }), view.getByTestId("trajectory-form").querySelector("summary")!]) {
    control.focus();
    const event = new KeyboardEvent("keydown", { key: " ", bubbles: true, cancelable: true });
    control.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  }
  const pageSpace = new KeyboardEvent("keydown", { key: " ", bubbles: true, cancelable: true });
  window.dispatchEvent(pageSpace);
  expect(pageSpace.defaultPrevented).toBe(true);
  expect(fixture.state.saves).toHaveLength(0);
});

test("action times have one editor and editing them synchronizes the saved pipeline summaries", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture, "square_d2_v3", "real_square_d2_valid");
  const original = structuredClone(selected.review_label!);
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  expect(view.queryByRole("group", { name: "Action 1 first time" })).toBeNull();
  expect(view.queryByRole("combobox", { name: "Action 1 ID" })).toBeNull();
  expect(view.queryByRole("button", { name: "Remove Action 1" })).toBeNull();
  for (const [index, time] of [[1, "2.125"], [2, "3.125"]]) {
    const input = view.getByRole("group", { name: `Action ${index} occurrence 1 time` }).querySelector("input")!;
    fireEvent.change(input, { target: { value: time } });
  }
  fireEvent.click(view.getByRole("button", { name: "Remove Action 3 occurrence 1" }));
  expect(view.queryByRole("group", { name: "Action 3 occurrence 1 time" })).toBeNull();
  expect(view.container.textContent).not.toContain("first time must match its earliest occurrence");
  expect(view.container.textContent).not.toContain("is marked No but still has");
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves).toHaveLength(1);
  const saved = fixture.state.saves[0];
  expect(saved.prediction_id).toBe("A-prediction-0");
  const actions = saved.label!.key_action_observations;
  expect(actions[0].first_time_s).toBe(2.125);
  expect(actions[0].occurrences[0].time_s).toBe(2.125);
  expect(actions[1].first_time_s).toBe(3.125);
  expect(actions[1].occurrences[0].time_s).toBe(3.125);
  expect(actions[2]).toEqual({ ...original.key_action_observations[2], occurred: false, first_time_s: null, occurrences: [] });
  expect(actions.slice(3)).toEqual(original.key_action_observations.slice(3));
  expect(saved.label!.stage_transitions).toEqual(original.stage_transitions);
  expect(saved.label!.failure_events).toEqual(original.failure_events);
  expect(selected.review_label).toEqual(original);
});

test("removing a repeated occurrence and Undo preserves all other events and metadata", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture);
  const label = structuredClone(selected.review_label!);
  label.key_action_observations[0].occurrences.push({ ...label.key_action_observations[0].occurrences[0], time_s: 9.876543, attempt_index: 2 });
  fixture.state.predictionOverrides.label = label;
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  fireEvent.click(view.getByRole("button", { name: "Remove Action 1 occurrence 2" }));
  expect(view.getByRole("group", { name: "Action 1 occurrence 1 time" })).toBeTruthy();
  expect(view.queryByRole("group", { name: "Action 1 occurrence 2 time" })).toBeNull();
  fireEvent.click(view.getByRole("button", { name: "Undo last event edit" }));
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves[0].label).toEqual(label);
});

test("conflicting old drafts remain untouched until the reviewer chooses which time to keep", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture);
  const label = structuredClone(selected.review_label!);
  label.key_action_observations[0].first_time_s = 2.123456;
  fixture.state.predictionOverrides.label = label;
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  expect(view.container.textContent).toContain("The saved action summary disagrees with its occurrences");
  expect(fixture.state.saves).toHaveLength(0);
  fireEvent.click(view.getByRole("button", { name: "Use summary time 2.123456 s" }));
  await act(async () => chooseVersion(view, "B"));
  const action = fixture.state.saves[0].label!.key_action_observations[0];
  expect(action.first_time_s).toBe(2.123456);
  expect(action.occurrences[0].time_s).toBe(2.123456);
  expect(action.occurrences[0].evidence).toBe(label.key_action_observations[0].occurrences[0].evidence);
});

test("unfinished action time blocks removal, sorting, and navigation without discarding input", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  configureTrajectoryFixture(fixture);
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  const input = view.getByRole("group", { name: "Action 1 occurrence 1 time" }).querySelector("input")!;
  fireEvent.change(input, { target: { value: "unfinished" } });
  expect(view.getByRole("combobox", { name: "Event display order" }).hasAttribute("disabled")).toBe(true);
  expect(view.getByRole("button", { name: "Remove Action 1 occurrence 1" }).hasAttribute("disabled")).toBe(true);
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves).toHaveLength(0);
  expect(input.value).toBe("unfinished");
  fireEvent.change(input, { target: { value: "2.5" } });
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves[0].label!.key_action_observations[0].first_time_s).toBe(2.5);
});

test("valid timestamp tolerance does not create a false conflict or normalize an untouched source", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture);
  const label = structuredClone(selected.review_label!);
  label.key_action_observations[0].first_time_s += 0.0000005;
  fixture.state.predictionOverrides.label = label;
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  expect(view.container.textContent).not.toContain("The saved action summary disagrees with its occurrences");
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves).toHaveLength(0);
});

test("a previously saved Yes without occurrences can directly add an unfinished time", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture);
  const label = structuredClone(selected.review_label!);
  label.key_action_observations[0].occurrences = [];
  label.key_action_observations[0].first_time_s = null;
  fixture.state.predictionOverrides.label = label;
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  const checklist = view.getByText("Add an action or another occurrence").closest("details")!;
  checklist.open = true;
  fireEvent.click(view.getByRole("button", { name: "Add action 1 time" }));
  const input = view.getByRole("group", { name: "Action 1 occurrence 1 time" }).querySelector("input")!;
  expect(input.value).toBe("");
  expect(input.closest("[role=group]")!.textContent).toContain("Time");
  fireEvent.change(input, { target: { value: "1.25" } });
  await act(async () => chooseVersion(view, "B"));
  const action = fixture.state.saves[0].label!.key_action_observations[0];
  expect(action.occurred).toBe(true);
  expect(action.first_time_s).toBe(1.25);
  expect(action.occurrences[0].time_s).toBe(1.25);
});

test("human notes remain separate, editable while blind, and survive draft navigation", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture);
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  const note = view.getByRole("textbox", { name: "Your review notes" }) as HTMLTextAreaElement;
  expect(note.value).toBe("");
  fireEvent.change(note, { target: { value: "No jaw closure; the source explanation is wrong." } });
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves).toHaveLength(1);
  expect(fixture.state.saves[0].blind).toBe(true);
  expect(fixture.state.armFetches).toBe(0);
  expect(fixture.state.saves[0].review_protocol).toBe("structured-v1");
  expect(fixture.state.saves[0].notes).toBe("No jaw closure; the source explanation is wrong.");
  expect(fixture.state.saves[0].label).toEqual(selected.review_label);
  expect((view.getByRole("textbox", { name: "Your review notes" }) as HTMLTextAreaElement).value).toBe("No jaw closure; the source explanation is wrong.");
});

test("timeline conflict offers explicit repair and Undo without changing source or saving on load", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  const label = structuredClone(selected.review_label!);
  label.key_action_observations[1].first_time_s = 7.266667;
  label.key_action_observations[1].occurrences[0].time_s = 7.266667;
  fixture.state.predictionOverrides.label = label;
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  expect(view.getByRole("region", { name: "Episode events" }).textContent).toContain("Two estimates for a related event");
  expect(fixture.state.saves).toHaveLength(0);
  fireEvent.click(view.getByRole("button", { name: "Use 7.27 s for this event" }));
  const transitionTime = () => (view.getByRole("group", { name: "Action 2 occurrence 1 time" }).querySelector("input") as HTMLInputElement).value;
  expect(transitionTime()).toBe("7.266667");
  expect(label.stage_transitions[1].time_s).toBe(6.5);
  fireEvent.click(view.getByRole("button", { name: "Undo last event edit" }));
  expect(view.getByRole("button", { name: "Use 6.50 s for this event" })).toBeTruthy();
  fireEvent.click(view.getByRole("button", { name: "Use 7.27 s for this event" }));
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves[0].label!.stage_transitions[1].time_s).toBe(7.266667);
  expect(fixture.state.saves[0].label!.stage_transitions[1].evidence).toBe(label.stage_transitions[1].evidence);
});

test("equivalent matching grasp and stage timestamps follow an explicit edit with Undo", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  const actionTime = view.getByRole("group", { name: "Action 1 occurrence 1 time" }).querySelector("input")!;
  fireEvent.change(actionTime, { target: { value: "4.6" } });
  fireEvent.blur(actionTime);
  const transitionTime = () => (view.getByRole("group", { name: "Action 1 occurrence 1 time" }).querySelector("input") as HTMLInputElement).value;
  expect(view.queryByRole("group", { name: "Transition 1 time" })).toBeNull();
  expect(transitionTime()).toBe("4.6");
  fireEvent.click(view.getByRole("button", { name: "Undo last event edit" }));
  expect(transitionTime()).toBe("3.75");
  expect(actionTime.value).toBe("3.75");
  expect(fixture.state.saves).toHaveLength(0);
});

test("unfinished timestamp text disables timeline repairs and human notes guard navigation", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  const label = structuredClone(selected.review_label!);
  label.key_action_observations[1].first_time_s = 7.266667;
  label.key_action_observations[1].occurrences[0].time_s = 7.266667;
  fixture.state.predictionOverrides.label = label;
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  const input = view.getByRole("group", { name: "Action 2 occurrence 1 time" }).querySelector("input")!;
  fireEvent.change(input, { target: { value: "unfinished" } });
  expect(view.getByRole("button", { name: "Use 7.27 s for this event" }).hasAttribute("disabled")).toBe(true);
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves).toHaveLength(0);
  expect(input.value).toBe("unfinished");
});

test("Undo of an equivalent-event repair restores the exact original disagreement", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  const label = structuredClone(selected.review_label!);
  label.key_action_observations[0].first_time_s = 4.6;
  label.key_action_observations[0].occurrences[0].time_s = 4.6;
  fixture.state.predictionOverrides.label = label;
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  fireEvent.click(view.getByRole("button", { name: "Use 4.60 s for this event" }));
  fireEvent.click(view.getByRole("button", { name: "Undo last event edit" }));
  expect((view.getByRole("group", { name: "Action 1 occurrence 1 time" }).querySelector("input") as HTMLInputElement).value).toBe("4.6");
  expect(view.getByRole("button", { name: "Use 3.75 s for this event" })).toBeTruthy();
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves[0].label).toEqual(label);
});

test("shared editing cannot overwrite unfinished text in another event editor", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  const view = render(<StageReview {...fixture.props} />);
  await settle();
  const transition = view.getByRole("group", { name: "Action 2 occurrence 1 time" }).querySelector("input")!;
  const action = view.getByRole("group", { name: "Action 1 occurrence 1 time" }).querySelector("input")!;
  fireEvent.change(transition, { target: { value: "unfinished" } });
  fireEvent.change(action, { target: { value: "4.6" } });
  fireEvent.blur(action);
  expect(transition.value).toBe("unfinished");
  await act(async () => chooseVersion(view, "B"));
  expect(fixture.state.saves).toHaveLength(0);
  expect(transition.value).toBe("unfinished");
});

test("adjudication distinguishes reviewer notes from retained source notes", async () => {
  const { state, view } = fixture("?episode=0&prediction=A&sstatus=adjudicate&blind=0", [{
    _id: "donor-review", episode_index: 0n, reviewer: "other", reviewer_user_id: "user-2",
    status: "uncertain", saved_at: 1_700_000_100_000,
    label: { max_stage: 8, notes: "Source claims a grasp." }, notes: "Reviewer saw no grasp.",
    prediction_id: "B-prediction-0", prediction_sha256: "B".repeat(64), episode_duration_s: 20,
  }]);
  await settle();
  expect(view.getByText("Reviewer notes: Reviewer saw no grasp.")).toBeTruthy();
  expect(view.getByText("Retained source notes · not reviewer notes")).toBeTruthy();
  expect(state.saves).toHaveLength(0);
});

test("Routing episode 62 three disagreements become one reviewed time per physical event", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected, task } = configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  const label = structuredClone(selected.review_label!);
  const sourceEvent = label.key_action_observations[0].occurrences[0];
  for (const [id, time] of [["first_clip_contact", 6.53], ["first_clip_seated", 10.60], ["second_clip_contact", 21.47]] as const) {
    const action = label.key_action_observations.find((item) => item.action_id === id)!;
    Object.assign(action, { occurred: true, first_time_s: time, occurrences: [{ ...sourceEvent, time_s: time }] });
  }
  label.stage_transitions[1].time_s = 5.75;
  for (const [id, time] of [["one_clip_seated", 9.50], ["remaining_clip_reached_off_axis", 18.50]] as const) {
    const stage = task.spec.trajectory.task_definition.stages.find((item) => item.id === id)!;
    label.stage_transitions.push({ ...label.stage_transitions[1], to_stage_id: id, to_stage_index: stage.index, time_s: time });
  }
  fixture.state.predictionOverrides.label = label;
  const original = structuredClone(label);
  const view = render(<StageReview {...fixture.props} />); await settle();
  for (const candidate of ["6.53", "10.60", "21.47"]) expect(view.getByRole("button", { name: `Use ${candidate} s for this event` })).toBeTruthy();
  expect(fixture.state.saves).toHaveLength(0);
  for (const candidate of ["6.53", "10.60", "21.47"]) fireEvent.click(view.getByRole("button", { name: `Use ${candidate} s for this event` }));
  expect(view.queryByRole("group", { name: "Transition 2 time" })).toBeNull();
  expect(view.queryByRole("group", { name: "Transition 5 time" })).toBeNull();
  expect(view.queryByRole("group", { name: "Transition 6 time" })).toBeNull();
  await act(async () => chooseVersion(view, "B"));
  const saved = fixture.state.saves[0];
  expect(saved.event_links).toHaveLength(3);
  for (const [id, time] of [["first_clip_reached_off_axis", 6.53], ["one_clip_seated", 10.60], ["remaining_clip_reached_off_axis", 21.47]] as const) {
    const event = saved.label!.stage_transitions.find((item) => item.to_stage_id === id)!;
    expect(event.time_s).toBe(time);
    expect(event.evidence).toBe(original.stage_transitions.find((item) => item.to_stage_id === id)!.evidence);
  }
  expect(saved.label!.failure_events).toEqual(original.failure_events);
  expect(label).toEqual(original);
});

test("a separate-event decision alone is dirty, saved outside labels, and restored on reload", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  let view = render(<StageReview {...fixture.props} />); await settle();
  fireEvent.click(view.getByRole("button", { name: "Keep as separate events" }));
  expect(view.getByRole("group", { name: "Transition 2 time" })).toBeTruthy();
  const unload = new Event("beforeunload", { cancelable: true }); window.dispatchEvent(unload);
  expect(unload.defaultPrevented).toBe(true);
  await act(async () => key("u"));
  const saved = fixture.state.saves[0];
  expect(saved.label).toEqual(selected.review_label);
  expect(saved.event_links).toEqual([{ action_id: "first_clip_contact", stage_id: "first_clip_reached_off_axis", attempt_index: 1, relation: "distinct" }]);
  view.unmount();
  window.history.replaceState(null, "", "/?episode=0&prediction=A&sstatus=all");
  view = render(<StageReview {...fixture.props} />); await settle();
  expect(view.getByRole("group", { name: "Transition 2 time" })).toBeTruthy();
  expect(view.queryByRole("button", { name: "Keep as separate events" })).toBeNull();
  const stageInput = view.getByRole("group", { name: "Transition 2 time" }).querySelector("input")!;
  fireEvent.change(stageInput, { target: { value: "7.123456" } });
  await act(async () => key("u"));
  expect(fixture.state.saves[1].label!.stage_transitions[1].time_s).toBe(7.123456);
  expect(fixture.state.saves[1].label!.key_action_observations[1].first_time_s).toBe(6.5);
});

test("primary failure has one editor and its summary follows mode and precise time edits", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  const view = render(<StageReview {...fixture.props} />); await settle();
  expect(view.queryByRole("combobox", { name: "Primary failure" })).toBeNull();
  expect(view.queryByRole("group", { name: "Primary failure time" })).toBeNull();
  fireEvent.change(view.getByRole("combobox", { name: "Failure 1 mode" }), { target: { value: "off_axis" } });
  fireEvent.change(view.getByRole("group", { name: "Failure 1 time" }).querySelector("input")!, { target: { value: "20.123456" } });
  await act(async () => key("u"));
  expect(fixture.state.saves[0].label).toMatchObject({ failure_mode: "off_axis", primary_failure_time_s: 20.123456 });
  expect(fixture.state.saves[0].label!.failure_events[0]).toMatchObject({ failure_mode_id: "off_axis", time_s: 20.123456, evidence: selected.review_label!.failure_events[0].evidence });
});

test("removing a shared event removes its two wire records and Undo restores both exactly", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  const view = render(<StageReview {...fixture.props} />); await settle();
  fireEvent.click(view.getByRole("button", { name: "Remove Action 1 occurrence 1" }));
  expect(view.queryByRole("group", { name: "Action 1 occurrence 1 time" })).toBeNull();
  expect(view.queryByRole("group", { name: "Transition 1 time" })).toBeNull();
  fireEvent.click(view.getByRole("button", { name: "Undo last event edit" }));
  await act(async () => key("u"));
  expect(fixture.state.saves[0].label).toEqual(selected.review_label);
});

test("chronological visual reordering preserves a focused timestamp and never sorts source arrays", async () => {
  window.history.replaceState(null, "", "/?episode=0&prediction=A");
  const fixture = createStageReviewFixture();
  const { selected } = configureTrajectoryFixture(fixture, "routing_d2_v1", "real_routing_d2_valid");
  const view = render(<StageReview {...fixture.props} />); await settle();
  const input = view.getByRole("group", { name: "Action 1 occurrence 1 time" }).querySelector("input")!;
  input.focus();
  fireEvent.change(input, { target: { value: "25.123456" } });
  expect(view.getByRole("group", { name: "Action 1 occurrence 1 time" }).querySelector("input")).toBe(input);
  expect(document.activeElement).toBe(input);
  expect(input.value).toBe("25.123456");
  fireEvent.blur(input);
  await act(async () => key("u"));
  expect(fixture.state.saves[0].label!.stage_transitions.map((event) => event.to_stage_id)).toEqual(selected.review_label!.stage_transitions.map((event) => event.to_stage_id));
  expect(fixture.state.saves[0].label!.key_action_observations[0].first_time_s).toBe(25.123456);
});

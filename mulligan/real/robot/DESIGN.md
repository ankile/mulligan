# Real Robot Pipeline Design Notes

How-to guides: [station setup](../../../docs/station.md) and
[blind evaluation](../../../docs/real_robot_eval.md). This file records why the code
is shaped the way it is.

## Episode Reset Timing: End-of-Episode (not Start)

The robot resets at the **end** of each episode, not the beginning of the next one.

**Why**: While the robot is resetting (~2-5s motion), the human operator resets the scene
(repositioning objects, etc.). This overlaps robot and human work, saving 2-5s per episode.
With start-of-episode resets, the operator would have to wait for the robot to finish before
knowing the scene is ready, then reset the scene themselves -- sequential instead of parallel.

At the start of each episode, `rollout_episode()` calls `env.get_observation()` instead of
`env.reset()` to get fresh sensor data without triggering another physical reset.

## Verified Reset: Handling Silent gRPC Failures

DROID's `FrankaRobot.update_joints()` (in `droid/franka/robot.py`) catches `grpc.RpcError`
with a bare `pass`, silently dropping `move_to_joint_positions()` commands. Transient gRPC
errors between the client and the Polymetis server cause ~10% of resets to fail silently --
`env.reset()` returns successfully but the robot never moves.

`verified_reset()` in `mulligan/real/collect/rollout.py` works around this by:
1. Calling `env.reset()` as normal
2. Comparing post-reset joint positions against `env.reset_joints`
3. If the max joint error exceeds a threshold (default 0.15 rad), retrying with
   bounded backoff (default 6 attempts, 1.0s initial delay, 1.5x backoff, 5.0s cap)
4. Logging warnings on retries and errors if all attempts fail

For data-saving rollouts, `rollout_episode()` also refuses to save a frame if
the post-step robot-state timestamp does not advance, because Cartesian velocity
features would be invalid. It polls for a fresh timestamp for up to 1.5s by
default before raising `RecoverableRolloutError`. Manifest eval treats that as
a discarded rollout for the current anonymous slot, resets loudly, and retries
with bounded backoff (default 6 discarded rollouts, 2s initial delay, 2x
backoff, 30s cap).

---

## Blind Evaluation

```
mulligan/real/eval/
  manifest_eval.py       # the blind eval entry point
  blind_eval_helpers.py  # manifest loading, round-plan loading, saved initial-state fields
  common.py              # rollout records, results.json, eval dataset create/append/checkpoint
  inference_server.py    # optional remote policy inference over an SSH tunnel
  eval_manifest.py       # builder for fresh held-out eval manifests (Sobol registry)
  split_policies.py      # per-policy views of a blind eval dataset
  outcome_results.py     # outcome-edit reconciliation of results.json
```

`manifest_eval` rolls every `--fixed-policy` out once per round from the start the
locked manifest assigns to that round. Before the first rollout, every round plan (a
per-round shuffle of the arms under anonymous labels A, B, C, ...) is written to
`results.json`, so an interrupted run resumes with the same hidden assignment. It loads
ALL scheduled policies up front and keeps them resident: a manifest reuses each policy
across many starts, so per-rollout reloads would re-download the same checkpoint dozens
of times. A mixed DP/IDQL manifest can exhaust GPU memory; `_load_scheduled_policies`
logs a loud warning when post-load free VRAM drops below a safety margin.

**Phased evals.** A manifest eval can start with every arm of its final `--fixed-policy`
list (so every round plan is pre-baked as a full blind shuffle over all arms) while
retiring not-yet-available arms with `--drop-fixed-policy`. Later, the same DATASET_NAME
resumes with the drops removed and `--rerun-incomplete-rounds`: `_pending_manifest_rounds`
marks the earlier rounds pending, `_remaining_from_plan` schedules exactly their missing
slots in the pre-baked slot order (labels unchanged), and the later rounds run every arm.
Every rollout record carries the launching invocation's `visit_id` (per-seal snapshots, so
a hard death still leaves per-record provenance) and every graceful shutdown appends a
`phase_stops` entry; a round whose records share one visit_id was collected in one physical
visit. `--stop-after-round N` is an optional outcome-independent cap for where the earlier
phase stops: rounds above N are excluded BEFORE policy loading / robot init (a restart after
the cap collects nothing), and round N ends through the graceful-quit path (seal drain,
finalize, push, results.json). The IQL sample-count override logs at DEBUG only: it runs
right before each anonymous rollout header and a console line there would un-blind the
DP+IQL arm. `tests/real/test_manifest_eval_phased_simulation.py` replays the three-phase
protocol of the Cable 15-arm lineage eval against the real scheduling code.

**Durability.** Each finished episode's seal chain (save -> parquet footer -> record ->
results.json) runs on one serial background worker, overlapping the next rollout. A
record enters `results.json` only after its episode's footer succeeded, so the results
file never references an episode that is not durable; a crash loses at most the
in-flight episode, whose round then shows incomplete on resume.

Released results files may also carry Policy Arena keys (`arena_session_id`,
`arena_submitted_*`, `args.random_arena_slots`), which the readers ignore.

---

## Operator UI (`mulligan/real/operator_ui/`)

Everything the operator sees and presses while resetting the scene between episodes
lives in one package, used identically by collection, teleop, DAgger, policy rollouts,
and the blind eval:

- `display.py`: display detection, the one-time X11 probe, HighGUI prewarm, and window
  creation / tiling / repaint / close. **Import order invariant:** every entrypoint
  imports this module and runs `prewarm_highgui()` (under `if __name__ == "__main__"`)
  BEFORE lerobot / torch are imported. OpenCV's Qt HighGUI deadlocks in the first
  `namedWindow` if `av` (pulled in by lerobot) has already loaded its bundled `libxcb`.
  `collect.blind_dagger` and `eval.manifest_eval` parse their arguments on light imports
  before the prewarm, so `--help` and argument errors exit in under a second.
  `tests/real/test_operator_ui_structure.py` pins the order.
- `keys.py`: the cbreak terminal listener, the numpad aliases (`key_label` for prompts),
  the OpenCV key poll (which is also what repaints the windows), and
  `drain_operator_keys`. Letters are lowercased at the source.
- `cards.py`: the per-task initial-state target card (matplotlib only, no torch), one
  `CardStyle` for collection and eval. Filenames are `target_<manifest_idx>.png` on every
  task. `python -m mulligan.real.operator_ui.preview --manifest M [--show]` renders a
  manifest's cards on any machine.
- `monitor.py`: live cropped camera monitor windows showing the loaded policy's actual
  crop (`policy_role_crop_boxes`), tiled to the right of the card.
- `gates.py`: `operator_gate`, the single "set up the scene, then press a key" decision.
  It always drains buffered keys first, repaints the monitor while waiting, and offers
  `r` (re-home), `k` (skip, when allowed), and `q` everywhere; the key legend in the
  prompt is generated from the enabled options. `operator_choice` is the sibling for
  the collector's between-episode decisions.
- `session.py`: `OperatorUI`, the object an entrypoint holds: `from_args` (flags from
  `cli.add_operator_ui_args`, display requirements checked loudly at startup),
  `show_card`, `render_monitor`, `gate`, `choose`, `read_key` / `drain_keys`, `close`.
- `progress.py` and `panel.py`: session timing, the anonymous eval context
  (`EvalScene`), the collector's context and quota status (`CollectionScene`,
  `CollectionStatus`), and the live panel surrounding the cached placement diagram.

The manifest model and loader (`InitialStateTarget`, `load_manifest_targets`, ...) live in
`mulligan/real/collect/initial_states.py`, also torch-free.

### Eval display and reset order

A manifest eval shows round, anonymous policy, slot within the round, finished/remaining
rollouts, elapsed time, estimated time left and the current phase in a compact top bar.
Running episodes also show steps and sub-goal marks. `configure_progress()` restores
finished counts from the accepted resume records; retired manifest arms do not count
toward the total. ETA is remaining rollouts times this launch's elapsed time per finished
rollout, including placement, reset, saving waits and discarded retries.

Every eval passes `pre_reset_callback` into `rollout_episode()`:

1. Capture and finalize the terminal frame.
2. Update finished progress and paint the upcoming helper, pumping HighGUI so it is
   visible before motion. Another policy uses the same target; the final slot selects
   the next *pending* round. A restart retains the current target and anonymous slot.
3. Run `verified_reset()` on the main thread, then return to the eval's save and
   placement flow. A helper error still runs the reset through `finally`, then
   propagates instead of continuing with stale instructions.

The panel updates at most four times per second during key polling and redraws raster
text around a cached diagram, with no matplotlib rendering or file reads in the control
loop. Camera monitoring is opt-in with `--monitor-cameras`; `--no-status-window` leaves a
plain target card.

### Collection display

The blind DAgger collector uses the same panel: phase (`Set up the target`, `Policy
running`, `Human correction`, `Saving episode`, `Choose what happens next`, `Resetting
robot`, `Finishing session`), a context line, and per-mode quota cells. The context
line never names the arm.

**Sub-goal marks.** Collection and eval take the same live sub-goal key (`g` / numpad
`3`, active when `RealTaskSpec.num_subtask_marks > 0`). The mark lands on the most recent
frame, is stored as a single-frame `reward=1.0` spike (the same spike the outcome editor
writes), and is listed as `subtask_frames` in the ledger / results row.

Render the operational layout without hardware:

```bash
uv run python -m mulligan.real.operator_ui.preview \
  --manifest data/real/manifests/routing_d2/r08/routing_d2_r8_eval_heldout_independent_sobol.json \
  --idx 12 --dashboard --out-dir /tmp/operator-preview
```

`--collection-style` renders the collector's panels (`setup` / `policy` / `human` /
`saving` / `choose`) the same way.

## Camera roles

Datasets and policies name cameras by ROLE (`wrist_left`, `wrist_right`, `side_1`,
`side_2`); the station config (`station.example.yaml` in this package, copied at `configs/real/`; overridable with
`MULLIGAN_STATION_CONFIG`) maps each role to the ZED `<serial>_<eye>` key the DROID
observation uses. The wrist is a ZED stereo camera recorded as both eyes; the side cameras
are recorded left eye only. Datasets record the mapping in `meta/camera_role_serials.json`.

## Per-Round Lifecycle Analysis (`mulligan/real/lifecycle/`)

Every real task line (marker_d2, square_d2, routing_d2) runs the same round lifecycle:
collection manifest -> blind collection -> split -> matched training -> held-out eval
ingest -> evidence-based start design for the next round. The analysis-side math and
ingest/plot structure are task-invariant and live in `mulligan/real/lifecycle/`; only the
`RealTaskSpec` differs per task.

- `tasks.py` — the task registry (`get_task_spec("square_d2")`): state keys, physical bounds,
  sampled placements, coverage grid dims.
- `geometry.py` — normalized periodic geometry, fill quantiles, cell entropy,
  Sobol/uniform samplers, weighted FPS. `TaskGeometry(spec)` binds bounds/dims.
- `stats.py` — Wilson CI, exact McNemar, sign-flip permutation test, bootstrap paired delta.
- `heldout_eval.py` — N-arm paired held-out eval ingest: HF results.json ->
  paired-round CSVs, policy/pairwise summaries, 4-panel plot. Library-first:
  a per-round driver builds a `HeldoutEvalConfig` and calls `run()`; the CLI builds
  a binary-success config for one session of a released or own eval repo.
  Released datasets are read at their release pin.

Behavior of the shared math is pinned by golden outputs in
`tests/real/test_real_lifecycle_golden.py`. Stage labeling has its own notes in
[`mulligan/real/stage_labeling/DESIGN.md`](../stage_labeling/DESIGN.md).

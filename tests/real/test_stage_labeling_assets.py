"""Characterization guards for mulligan.real.stage_labeling.assets.

Pins the media/dataset layer: the proprioception helpers and the ffmpeg command
construction (captured by stubbing subprocess, so ffmpeg/HF are never invoked).
Also guards the load-bearing spec-wiring — durations/thresholds must come from the
spec, not a hardcoded 15.0.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import types as _pytypes
from pathlib import Path

import pandas as pd
import pytest

import mulligan.real.stage_specs as sl
from mulligan.real.stage_labeling import assets

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = "marker_d2"


def _marker_spec_with_events():
    """marker_d2 with an events CSV path (read_csv is stubbed in the build_items tests)."""
    return dataclasses.replace(sl.get_label_task_spec(MARKER), events_csv=Path("events.csv"))


def _install_genai_stub() -> None:
    if "google.genai" in sys.modules:
        return
    genai = _pytypes.ModuleType("google.genai")
    errors = _pytypes.ModuleType("google.genai.errors")
    types_mod = _pytypes.ModuleType("google.genai.types")

    class APIError(Exception):
        def __init__(self, code: int = 0, *a: object) -> None:
            super().__init__(*a)
            self.code = code

    class _Type:
        INTEGER = "INTEGER"
        NUMBER = "NUMBER"
        BOOLEAN = "BOOLEAN"
        STRING = "STRING"
        OBJECT = "OBJECT"

    class _Stored:
        def __init__(self, **kw: object) -> None:
            self.__dict__.update(kw)

    errors.APIError = APIError
    types_mod.Type = _Type
    types_mod.Schema = _Stored
    for name in ("Part", "Blob", "FileData", "VideoMetadata", "Content", "GenerateContentConfig"):
        setattr(types_mod, name, _Stored)
    genai.Client = _Stored
    genai.errors = errors
    genai.types = types_mod
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.errors"] = errors
    sys.modules["google.genai.types"] = types_mod


@pytest.fixture
def captured_argv(monkeypatch):
    """Capture subprocess.run argv (ffmpeg never runs)."""
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


# --------------------------------------------------------------------------- #
# Pure proprioception helpers.
# --------------------------------------------------------------------------- #


def test_sensor_trace_duration_uses_spec_fps():
    """Load-bearing: episode_duration_s must scale with spec.fps, not a fixed 15."""
    spec15 = sl.get_label_task_spec(MARKER)
    spec30 = dataclasses.replace(spec15, name="fps_probe", fps=30.0)
    row = pd.Series(
        {"gripper_hold_time_s": 2.0, "gripper_release_time_s": None,
         "gripper_reopened_at_end": False, "episode_length": 300}
    )  # fmt: skip
    assert assets.sensor_trace(spec15, row)["episode_duration_s"] == 20.0
    assert assets.sensor_trace(spec30, row)["episode_duration_s"] == 10.0


# --------------------------------------------------------------------------- #
# ffmpeg command construction (argv captured, ffmpeg never runs).
# --------------------------------------------------------------------------- #


def test_clip_and_combo_cmd_golden(tmp_path):
    """clip_video / combo command construction; pin via
    golden argv transcribed from those scripts."""
    src, out = tmp_path / "src.mp4", tmp_path / "clip.mp4"
    assert assets.clip_cmd(src, 1.5, 3.0, out) == [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", "1.500000", "-i", str(src), "-t", "3.000000",
        "-an", "-vf", "scale=640:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    side, wrist, combo = tmp_path / "s.mp4", tmp_path / "w.mp4", tmp_path / "c.mp4"
    cmd = assets.combo_cmd(side, wrist, combo)
    assert cmd[:2] == ["ffmpeg", "-hide_banner"]
    assert "hstack=inputs=2[v]" in cmd[cmd.index("-filter_complex") + 1]
    assert "text=SIDE" in cmd[cmd.index("-filter_complex") + 1]
    assert cmd[-1] == str(combo) and "24" in cmd  # crf 24 for combo


# --------------------------------------------------------------------------- #
# Grasp-window montage (the new S0/S1 evidence crop).
# --------------------------------------------------------------------------- #


def test_montage_crop_cmd_structure(tmp_path):
    """One -ss/-i input per time, hstacked left->right in time order, same bottom-
    aligned crop + 3x upscale as the single-frame moment crop, one output frame."""
    video, out = tmp_path / "w.mp4", tmp_path / "m.png"
    times = [3.0, 4.0, 5.0]
    cmd = assets.montage_crop_cmd(video, times, out)
    # one seek+input per requested time, in time order
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "-ss"] == ["3.00", "4.00", "5.00"]
    assert cmd.count("-i") == 3
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "hstack=inputs=3[v]" in fc
    # same per-frame crop geometry as the single-frame moment crop
    assert "crop=iw*0.7:ih*0.7:(iw-ow)/2:ih-oh,scale=iw*3:ih*3:flags=lanczos" in fc
    assert cmd[-1] == str(out)
    assert "-frames:v" in cmd and cmd[cmd.index("-frames:v") + 1] == "1"


def test_wrist_span_montage_cmd_structure(tmp_path):
    """Full-frame wrist strip: one -ss/-i per time, each scaled to a common height
    with NO crop, hstacked left->right in time order, one output frame."""
    video, out = tmp_path / "w.mp4", tmp_path / "m.png"
    times = [1.0, 5.0, 9.0]
    cmd = assets.wrist_span_montage_cmd(video, times, out)
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "-ss"] == ["1.00", "5.00", "9.00"]
    assert cmd.count("-i") == 3
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "hstack=inputs=3[v]" in fc
    # full frame, common height, explicitly NOT the zoomed bottom-center crop
    assert "scale=-1:480,setsar=1" in fc
    assert "crop=" not in fc
    assert cmd[-1] == str(out)
    assert "-frames:v" in cmd and cmd[cmd.index("-frames:v") + 1] == "1"


def test_grasp_window_crop_time_selection(monkeypatch, tmp_path):
    """grasp_window_crop spans the WHOLE episode at evenly-spaced moments (full frames,
    the decisive grasp instant always in view). The frame extraction is stubbed."""
    captured: list[list[float]] = []

    def fake_run(cmd):
        captured.append(cmd)

    monkeypatch.setattr(assets, "_run", fake_run)
    video, out = tmp_path / "w.mp4", tmp_path / "g.png"

    # whole-episode span: hi = duration - 0.2 = 11.8; 3 frames at 0.08..1.0 of hi,
    # emitted to 2dp in the ffmpeg -ss args (0.944, 6.372, 11.8 -> 0.94, 6.37, 11.80)
    out.unlink(missing_ok=True)
    assets.grasp_window_crop(video, duration_s=12.0, out=out, frames=3)
    times = [float(captured[-1][i + 1]) for i, a in enumerate(captured[-1]) if a == "-ss"]
    assert times == pytest.approx([0.94, 6.37, 11.8])


# --------------------------------------------------------------------------- #
# build_items orchestration (I/O stubbed) — keys + spec field threading.
# --------------------------------------------------------------------------- #


def test_build_items_threads_spec_and_keys(monkeypatch, tmp_path):
    spec = _marker_spec_with_events()
    used_camera_keys: list[str] = []

    # Stub every I/O boundary.
    events = pd.DataFrame(
        {
            "episode_index": [0, 1],
            "episode_length": [150, 90],
            "gripper_hold_time_s": [3.0, None],
            "gripper_release_time_s": [7.0, None],
            "gripper_reopened_at_end": [True, False],
        }
    )
    monkeypatch.setattr(assets.pd, "read_csv", lambda *a, **k: events)

    meta = pd.DataFrame({"episode_index": [0, 1]})  # build_items sets the index itself
    # add the per-camera columns episode_video_clip reads
    for key in (spec.side_camera_key, spec.wrist_camera_key):
        meta[f"videos/{key}/file_index"] = 0
        meta[f"videos/{key}/from_timestamp"] = 0.0
    monkeypatch.setattr(assets.pd, "read_parquet", lambda *a, **k: meta)

    monkeypatch.setattr(assets, "repo_file", lambda spec, path: tmp_path / "fake.bin")
    monkeypatch.setattr(
        assets, "gripper_series_1hz", lambda spec, build_dir: {0: [0.0, 0.0, 0.8, 0.8], 1: [0.0]}
    )
    monkeypatch.setattr(assets, "valid_frame_counts", lambda spec, build_dir: {0: 150, 1: 90})

    def fake_clip(spec, meta_row, camera_key, duration_s, out):
        used_camera_keys.append(camera_key)
        return out

    def fake_window(video, duration_s, out, **k):
        return out

    monkeypatch.setattr(assets, "episode_video_clip", fake_clip)
    monkeypatch.setattr(assets, "final_frame_still", lambda video, out: out)
    monkeypatch.setattr(assets, "final_frame_crop", lambda still, out: out)
    monkeypatch.setattr(assets, "grasp_moment_crop", lambda *a, **k: a[3])
    monkeypatch.setattr(assets, "grasp_window_crop", fake_window)
    monkeypatch.setattr(assets, "combo_video", lambda side, wrist, out: out)

    items = assets.build_items(spec, tmp_path)
    assert [it["episode_index"] for it in items] == [0, 1]
    # spec camera keys were used (not hardcoded), both cameras per episode
    assert set(used_camera_keys) == {spec.side_camera_key, spec.wrist_camera_key}
    expected_keys = {
        "episode_index", "duration_s", "side_video", "wrist_video", "side_path", "wrist_path",
        "side_final_frame", "wrist_final_frame", "side_final_crop", "wrist_final_crop",
        "sensor_trace", "gripper_series_1hz", "grasp_moment_crop", "grasp_early_crop",
        "grasp_window_crop", "release_moment_crop", "combo_path",
    }  # fmt: skip
    assert set(items[0]) == expected_keys
    # episode 0: duration uses spec.fps; reopened -> release crop present
    assert items[0]["duration_s"] == 150 / spec.fps
    assert items[0]["release_moment_crop"] is not None
    # episode 1: never reopened -> no release crop; never closed -> no grasp crops
    assert items[1]["release_moment_crop"] is None
    assert items[1]["grasp_moment_crop"] is None
    # grasp-window crop ALWAYS present (the failed-grasp S0/S1 cases need it most).
    assert items[0]["grasp_window_crop"] is not None
    assert items[1]["grasp_window_crop"] is not None


def test_episodes_filter(monkeypatch, tmp_path):
    spec = _marker_spec_with_events()
    events = pd.DataFrame(
        {
            "episode_index": [0, 1, 2],
            "episode_length": [30, 30, 30],
            "gripper_hold_time_s": [None, None, None],
            "gripper_release_time_s": [None, None, None],
            "gripper_reopened_at_end": [False, False, False],
        }
    )
    monkeypatch.setattr(assets.pd, "read_csv", lambda *a, **k: events)
    meta = pd.DataFrame({"episode_index": [0, 1, 2]})  # build_items sets the index itself
    for key in (spec.side_camera_key, spec.wrist_camera_key):
        meta[f"videos/{key}/file_index"] = 0
        meta[f"videos/{key}/from_timestamp"] = 0.0
    monkeypatch.setattr(assets.pd, "read_parquet", lambda *a, **k: meta)
    monkeypatch.setattr(assets, "repo_file", lambda spec, path: tmp_path / "f.bin")
    monkeypatch.setattr(
        assets, "gripper_series_1hz", lambda spec, build_dir: {0: [0.0], 1: [0.0], 2: [0.0]}
    )
    monkeypatch.setattr(assets, "valid_frame_counts", lambda spec, build_dir: {0: 30, 1: 30, 2: 30})
    monkeypatch.setattr(assets, "episode_video_clip", lambda *a, **k: a[-1])
    monkeypatch.setattr(assets, "final_frame_still", lambda video, out: out)
    monkeypatch.setattr(assets, "final_frame_crop", lambda still, out: out)
    monkeypatch.setattr(assets, "grasp_window_crop", lambda *a, **k: a[2])
    monkeypatch.setattr(assets, "combo_video", lambda side, wrist, out: out)

    items = assets.build_items(spec, tmp_path, episodes={1})
    assert [it["episode_index"] for it in items] == [1]


# --------------------------------------------------------------------------- #
# Bounded parquet catch and the spec-scoped cache manifest.
# --------------------------------------------------------------------------- #


def test_gripper_series_reads_enumerated_shards(monkeypatch, tmp_path):
    spec = _marker_spec_with_events()
    col = spec.gripper_state_column
    frame0 = pd.DataFrame({"episode_index": [0, 0], "frame_index": [0, 1], col: [0.0, 0.8]})
    monkeypatch.setattr(
        assets, "frame_parquet_rel_paths", lambda spec: ["data/chunk-000/file-000.parquet"]
    )
    monkeypatch.setattr(assets, "repo_file", lambda spec, path: tmp_path / "shard0.parquet")
    monkeypatch.setattr(assets.pd, "read_parquet", lambda *a, **k: frame0)
    series = assets.gripper_series_1hz(spec, tmp_path)
    assert 0 in series and len(series[0]) >= 1


def test_gripper_series_uses_is_valid_prefix(monkeypatch, tmp_path):
    spec = dataclasses.replace(sl.get_label_task_spec(MARKER), fps=2.0)
    col = spec.gripper_state_column
    frame0 = pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3],
            col: [0.0, 0.8, 1.0, 1.0],
            "is_valid": [1, 1, 0, 0],
        }
    )

    monkeypatch.setattr(
        assets, "frame_parquet_rel_paths", lambda spec: ["data/chunk-000/file-000.parquet"]
    )
    monkeypatch.setattr(assets, "repo_file", lambda spec, path: tmp_path / "shard0.parquet")
    monkeypatch.setattr(assets.pd, "read_parquet", lambda *a, **k: frame0)
    series = assets.gripper_series_1hz(spec, tmp_path)
    assert series[0] == [0.4]


def test_gripper_series_reraises_non_eof_error(monkeypatch, tmp_path):
    spec = _marker_spec_with_events()
    col = spec.gripper_state_column

    def fake_repo_file(spec, path):
        raise RuntimeError("transient network blip")

    monkeypatch.setattr(
        assets, "frame_parquet_rel_paths", lambda spec: ["data/chunk-000/file-000.parquet"]
    )
    monkeypatch.setattr(assets, "repo_file", fake_repo_file)
    monkeypatch.setattr(
        assets.pd,
        "read_parquet",
        lambda *a, **k: pd.DataFrame({"episode_index": [0], "frame_index": [0], col: [0.0]}),
    )
    with pytest.raises(RuntimeError, match="network blip"):
        assets.gripper_series_1hz(spec, tmp_path)


def _stub_build_items_io(monkeypatch, tmp_path, spec):
    events = pd.DataFrame(
        {
            "episode_index": [0],
            "episode_length": [30],
            "gripper_hold_time_s": [None],
            "gripper_release_time_s": [None],
            "gripper_reopened_at_end": [False],
        }
    )
    monkeypatch.setattr(assets.pd, "read_csv", lambda *a, **k: events)
    meta = pd.DataFrame({"episode_index": [0]})
    for key in (spec.side_camera_key, spec.wrist_camera_key):
        meta[f"videos/{key}/file_index"] = 0
        meta[f"videos/{key}/from_timestamp"] = 0.0
    monkeypatch.setattr(assets.pd, "read_parquet", lambda *a, **k: meta)
    monkeypatch.setattr(assets, "repo_file", lambda spec, path: tmp_path / "f.bin")
    monkeypatch.setattr(assets, "gripper_series_1hz", lambda spec, build_dir: {0: [0.0]})
    monkeypatch.setattr(assets, "valid_frame_counts", lambda spec, build_dir: {0: 30})
    monkeypatch.setattr(assets, "episode_video_clip", lambda *a, **k: a[-1])
    monkeypatch.setattr(assets, "final_frame_still", lambda video, out: out)
    monkeypatch.setattr(assets, "final_frame_crop", lambda still, out: out)
    monkeypatch.setattr(assets, "grasp_window_crop", lambda *a, **k: a[2])
    monkeypatch.setattr(assets, "combo_video", lambda side, wrist, out: out)


def test_build_dir_scoped_per_task():
    spec = _marker_spec_with_events()
    sq = sl.get_label_task_spec("square_d2")
    base = Path("/tmp/shared")
    assert assets.task_build_dir(spec, base) != assets.task_build_dir(sq, base)
    assert assets.task_build_dir(spec, base) == base / spec.name


def test_build_items_manifest_fails_loud_on_config_change(monkeypatch, tmp_path):
    spec = _marker_spec_with_events()
    _stub_build_items_io(monkeypatch, tmp_path, spec)
    assets.build_items(spec, tmp_path)  # writes the manifest
    # Re-point the dataset under the same task name -> stale assets dir -> raise.
    repointed = dataclasses.replace(spec, dataset_repo_id="org/some-other-dataset")
    with pytest.raises(RuntimeError, match="different config"):
        assets.build_items(repointed, tmp_path)


def test_build_items_rejects_stale_events_length(monkeypatch, tmp_path):
    spec = _marker_spec_with_events()
    _stub_build_items_io(monkeypatch, tmp_path, spec)
    monkeypatch.setattr(assets, "valid_frame_counts", lambda spec, build_dir: {0: 12})
    with pytest.raises(RuntimeError, match="Regenerate the events CSV"):
        assets.build_items(spec, tmp_path)


def test_build_items_reference_media_allows_invalid_tail_within_physical_length(
    monkeypatch, tmp_path
):
    spec = _marker_spec_with_events()
    _stub_build_items_io(monkeypatch, tmp_path, spec)
    meta = pd.DataFrame({"episode_index": [0], "length": [30]})
    for key in (spec.side_camera_key, spec.wrist_camera_key):
        meta[f"videos/{key}/file_index"] = 0
        meta[f"videos/{key}/from_timestamp"] = 0.0
    monkeypatch.setattr(assets.pd, "read_parquet", lambda *a, **k: meta)
    monkeypatch.setattr(assets, "valid_frame_counts", lambda spec, build_dir: {0: 12})

    items = assets.build_items(spec, tmp_path, allow_invalid_tail_for_reference_media=True)

    assert items[0]["duration_s"] == 30 / spec.fps


def test_build_items_reference_media_rejects_beyond_physical_length(monkeypatch, tmp_path):
    spec = _marker_spec_with_events()
    _stub_build_items_io(monkeypatch, tmp_path, spec)
    meta = pd.DataFrame({"episode_index": [0], "length": [29]})
    for key in (spec.side_camera_key, spec.wrist_camera_key):
        meta[f"videos/{key}/file_index"] = 0
        meta[f"videos/{key}/from_timestamp"] = 0.0
    monkeypatch.setattr(assets.pd, "read_parquet", lambda *a, **k: meta)
    monkeypatch.setattr(assets, "valid_frame_counts", lambda spec, build_dir: {0: 12})

    with pytest.raises(RuntimeError, match="exceeds physical dataset length 29"):
        assets.build_items(spec, tmp_path, allow_invalid_tail_for_reference_media=True)


def test_build_items_accepts_shorter_outcome_event_length_and_clears_stale_assets(
    monkeypatch, tmp_path
):
    spec = _marker_spec_with_events()
    _stub_build_items_io(monkeypatch, tmp_path, spec)
    events = pd.DataFrame(
        {
            "episode_index": [0],
            "episode_length": [12],
            "gripper_hold_time_s": [None],
            "gripper_release_time_s": [None],
            "gripper_reopened_at_end": [False],
        }
    )
    monkeypatch.setattr(assets.pd, "read_csv", lambda *a, **k: events)
    monkeypatch.setattr(assets, "valid_frame_counts", lambda spec, build_dir: {0: 30})
    assets_dir = tmp_path / spec.name / "assets"
    assets_dir.mkdir(parents=True)
    stale = assets_dir / "episode_000_combo.mp4"
    stale.write_text("old full-length clip")

    items = assets.build_items(spec, tmp_path)

    assert items[0]["duration_s"] == 12 / spec.fps
    assert not stale.exists()
    lengths = json.loads((tmp_path / spec.name / "asset_episode_lengths.json").read_text())
    assert lengths == {"0": 12}

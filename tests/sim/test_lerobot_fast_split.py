"""Unit tests for the file-level fast-split helpers.

Covers the pure logic that guards partially-null columns, plus an end-to-end split of
a tiny synthetic video dataset that pins the codec contract: the output is written with
LeRobot's own RGB defaults, ``meta/info.json`` says so, the post-write decode gate
passes through the training decode path, and the gate actually fires on a broken file.
"""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from mulligan.tools.lerobot_fast_split import (
    _drop_episode_stat_columns,
    is_null_cell,
    last_frame_success_by_episode,
    split_video_encoder,
)


def test_is_null_cell_detects_scalar_and_element_none():
    assert is_null_cell(None) is True
    assert is_null_cell([None, None, None]) is True  # mid-collection [None]*shape form
    assert is_null_cell([1.0, None, 3.0]) is True  # any element None
    assert is_null_cell([0.1, 0.2, 0.3]) is False
    assert is_null_cell(np.array([0.1, 0.2])) is False
    assert is_null_cell(5.0) is False  # scalar non-None, not iterable


class _FakeHF:
    def __init__(self, success):
        self._success = success

    def with_format(self, _fmt):
        return self

    def __getitem__(self, key):
        assert key == "success"
        return self._success


class _FakeEpisodes:
    def __init__(self, bounds):
        self._bounds = bounds

    def __getitem__(self, idx):
        lo, hi = self._bounds[idx]
        return {"dataset_from_index": lo, "dataset_to_index": hi}


class _FakeMeta:
    def __init__(self, bounds):
        self.episodes = _FakeEpisodes(bounds)
        self.total_episodes = len(bounds)


class _FakeDataset:
    def __init__(self, success, bounds):
        self.hf_dataset = _FakeHF(success)
        self.meta = _FakeMeta(bounds)


def test_last_frame_success_by_episode_scalar_and_listwrapped():
    # ep0 frames [0,2) terminal success=1; ep1 frames [2,4) terminal success=0.
    ds = _FakeDataset(success=[0, 1, 0, 0], bounds=[(0, 2), (2, 4)])
    assert last_frame_success_by_episode(ds) == [True, False]
    # list-wrapped success values (shape (1,)) are unwrapped.
    ds2 = _FakeDataset(success=[[0], [1], [0], [1]], bounds=[(0, 2), (2, 4)])
    assert last_frame_success_by_episode(ds2) == [True, True]


def test_drop_episode_stat_columns(tmp_path):
    ep_dir = tmp_path / "meta" / "episodes" / "chunk-000"
    ep_dir.mkdir(parents=True)
    col = "telemetry.franka.motor_torques_external"
    df = pd.DataFrame(
        {
            "episode_index": [0, 1],
            "length": [10, 12],
            f"stats/{col}/min": [None, None],
            f"stats/{col}/max": [None, None],
            "stats/action/min": [0.0, 0.1],
        }
    )
    path = ep_dir / "file-000.parquet"
    df.to_parquet(path, index=False)

    _drop_episode_stat_columns(tmp_path, [col])

    out = pd.read_parquet(path)
    assert f"stats/{col}/min" not in out.columns
    assert f"stats/{col}/max" not in out.columns
    assert "stats/action/min" in out.columns  # other features untouched
    assert "episode_index" in out.columns and "length" in out.columns


def test_split_encoder_is_lerobot_rgb_default():
    """The split encoder is LeRobot's object, not a constant re-pinned in this repo."""
    from lerobot.configs.video import rgb_encoder_defaults

    encoder = split_video_encoder()
    assert encoder == rgb_encoder_defaults()
    # Pin what "the LeRobot default" resolves to today so a silent upstream flip is visible.
    assert encoder.vcodec == "libsvtav1"
    assert encoder.pix_fmt == "yuv420p"
    assert encoder.g == 2


def test_fast_split_takes_no_codec_knobs():
    """The codec is not a parameter: passing one is a TypeError, not a silent override."""
    from mulligan.tools.lerobot_fast_split import fast_split_dataset

    group = {"a": {"repo_id": "x", "root": "/tmp/x", "episodes": [0]}}
    for kwarg in ("vcodec", "preset", "crf", "pix_fmt"):
        with pytest.raises(TypeError, match=kwarg):
            fast_split_dataset(object(), group, **{kwarg: "h264"})
    with pytest.raises(ValueError, match="No groups"):
        fast_split_dataset(object(), {})


_RETIRED_CODEC_FLAGS = (("--vcodec", "h264"), ("--preset", "veryfast"), ("--crf", "23"))

_CLI_BASE_ARGS = {
    "mulligan.data.split_protocol_quota": [
        "--source-repo",
        "owner/parent",
        "--manifest",
        "m.json",
        "--ledger",
        "l.jsonl",
        "--target",
        "no_cf.baseline_uniform=owner/child",
        "--output-root",
        "/tmp/out",
    ],
    "mulligan.real.data.split": [
        "--source-repo",
        "owner/parent",
        "--ledger",
        "l.jsonl",
        "--expected-success-per-arm",
        "baseline_uniform=50",
        "--target",
        "baseline_uniform=owner/child",
        "--output-root",
        "/tmp/out",
    ],
    "mulligan.real.eval.split_policies": [
        "--source-repo",
        "owner/parent",
        "--target",
        "0=owner/child",
        "--output-root",
        "/tmp/out",
    ],
}


@pytest.mark.parametrize("module_name", sorted(_CLI_BASE_ARGS))
def test_split_clis_reject_retired_codec_flags(module_name):
    """The retired --vcodec/--preset/--crf flags must be unrecognised, not ignored."""
    import importlib

    build_parser = importlib.import_module(module_name).build_parser
    base = _CLI_BASE_ARGS[module_name]
    # The baseline argument set parses, so a SystemExit below is about the codec flag and
    # not about a missing required argument (argparse reports those first).
    build_parser().parse_args(base)
    for flag, value in _RETIRED_CODEC_FLAGS:
        with pytest.raises(SystemExit):
            build_parser().parse_args([*base, flag, value])


@pytest.mark.parametrize("module_name", sorted(_CLI_BASE_ARGS))
def test_split_clis_expose_replace_remote_codec_off_by_default(module_name):
    """The codec guard's only override is this flag, and it is opt-in on every split CLI."""
    import importlib

    build_parser = importlib.import_module(module_name).build_parser
    base = _CLI_BASE_ARGS[module_name]
    assert build_parser().parse_args(base).replace_remote_codec is False
    assert build_parser().parse_args([*base, "--replace-remote-codec"]).replace_remote_codec is True


# --- remote codec guard -------------------------------------------------------------


def _write_local_info(root, codec):
    """Minimal LeRobot ``meta/info.json`` carrying one video feature with ``codec``."""
    info = {
        "features": {
            "observation.images.cam": {"dtype": "video", "info": {"video.codec": codec}},
            "action": {"dtype": "float32"},
        }
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


def test_codec_guard_refuses_codec_changing_overwrite(tmp_path, monkeypatch):
    """An h264 repo on the Hub must not be silently replaced by AV1 pixels."""
    from mulligan.tools import lerobot_hub

    root = _write_local_info(tmp_path / "local", "av1")
    monkeypatch.setattr(lerobot_hub, "_remote_video_codecs", lambda repo_id: ["h264"])

    with pytest.raises(RuntimeError, match="CODEC GUARD: owner/child") as excinfo:
        lerobot_hub.assert_remote_codec_matches("owner/child", root, replace_remote_codec=False)
    message = str(excinfo.value)
    assert "'h264'" in message and "'av1'" in message
    assert "--replace-remote-codec" in message

    # The flag is the only override, and it lets the same call through.
    lerobot_hub.assert_remote_codec_matches("owner/child", root, replace_remote_codec=True)


def test_codec_guard_passes_on_same_codec_and_when_nothing_to_compare(tmp_path, monkeypatch):
    """Same-codec re-pushes, new repos and state-only datasets keep today's behavior."""
    from mulligan.tools import lerobot_hub

    root = _write_local_info(tmp_path / "local", "av1")
    # "av1" (stream name) and "libsvtav1" (encoder name) are the same codec.
    monkeypatch.setattr(lerobot_hub, "_remote_video_codecs", lambda repo_id: ["libsvtav1"])
    lerobot_hub.assert_remote_codec_matches("owner/child", root, replace_remote_codec=False)

    # Repo does not exist yet / has no meta/info.json.
    monkeypatch.setattr(lerobot_hub, "_remote_video_codecs", lambda repo_id: None)
    lerobot_hub.assert_remote_codec_matches("owner/child", root, replace_remote_codec=False)

    # Local dataset carries no video features: nothing to compare, and the guard must not
    # be asked to compare against a remote codec set it has no counterpart for.
    state_only = tmp_path / "state_only"
    (state_only / "meta").mkdir(parents=True)
    (state_only / "meta" / "info.json").write_text(json.dumps({"features": {"action": {}}}))
    monkeypatch.setattr(lerobot_hub, "_remote_video_codecs", lambda repo_id: ["h264"])
    lerobot_hub.assert_remote_codec_matches("owner/child", state_only, replace_remote_codec=False)


def test_push_helper_runs_the_codec_guard_before_uploading(tmp_path, monkeypatch):
    """The guard sits on the push path itself, not only in a CLI that remembers to call it."""
    from mulligan.tools import lerobot_hub

    class _FakeApi:
        def list_repo_files(self, **_kwargs):
            return []

        def create_commit(self, **_kwargs):  # pragma: no cover - no stale files here
            raise AssertionError("no stale shards to delete in this fixture")

    pushed = []

    class _FakeDataset:
        repo_id = "owner/child"

        def __init__(self, root):
            self.root = root

        def push_to_hub(self, **kwargs):
            pushed.append(kwargs)

    monkeypatch.setattr(lerobot_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(lerobot_hub, "advance_lerobot_version_tag", lambda repo_id: "sha")
    monkeypatch.setattr(lerobot_hub, "_remote_video_codecs", lambda repo_id: ["h264"])
    dataset = _FakeDataset(_write_local_info(tmp_path / "local", "av1"))

    with pytest.raises(RuntimeError, match="CODEC GUARD"):
        lerobot_hub.push_lerobot_dataset_replacing_remote(dataset)
    assert pushed == [], "guard must run BEFORE any upload"

    lerobot_hub.push_lerobot_dataset_replacing_remote(dataset, replace_remote_codec=True)
    assert pushed == [{"license": "mit"}]


# --- end-to-end split of a tiny synthetic video dataset -----------------------------

_FPS = 10
_HW = 64
_EPISODE_LENGTHS = (7, 5, 6)


def _make_source_dataset(root):
    """Write a tiny multi-episode LeRobotDataset with one video camera."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.cam": {
            "dtype": "video",
            "shape": (_HW, _HW, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "success": {"dtype": "int64", "shape": (1,), "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id="test/fast-split-source",
        fps=_FPS,
        root=str(root),
        robot_type="panda",
        features=features,
        use_videos=True,
    )
    rng = np.random.default_rng(0)
    for ep_idx, length in enumerate(_EPISODE_LENGTHS):
        for frame_idx in range(length):
            ds.add_frame(
                {
                    # Structured noise: a flat frame compresses to nothing and would make
                    # a truncation test meaningless.
                    "observation.images.cam": rng.integers(0, 255, (_HW, _HW, 3), dtype=np.uint8),
                    "action": np.full(2, float(ep_idx), dtype=np.float32),
                    "success": np.array([1], dtype=np.int64),
                    "task": "fast split test",
                }
            )
        ds.save_episode()
    ds.finalize()
    return LeRobotDataset("test/fast-split-source", root=str(root))


@pytest.fixture(scope="module")
def synthetic_split(tmp_path_factory):
    """Split a synthetic AV1 parent; yields (source, split dataset)."""
    from mulligan.tools.lerobot_fast_split import fast_split_dataset

    base = tmp_path_factory.mktemp("fast_split")
    source = _make_source_dataset(base / "source")
    out = fast_split_dataset(
        source,
        {"grp": {"repo_id": "test/fast-split-out", "root": base / "out", "episodes": [0, 2]}},
    )["grp"]
    return source, out


def test_split_output_uses_lerobot_default_codec(synthetic_split):
    """info.json must describe what was actually encoded, with LeRobot's defaults."""
    from lerobot.configs.video import VIDEO_CODECS_ALIASES

    source, out = synthetic_split
    encoder = split_video_encoder()
    on_disk = json.loads((out.root / "meta" / "info.json").read_text())
    for video_key in out.meta.video_keys:
        info = on_disk["features"][video_key]["info"]
        codec = info["video.codec"]
        assert VIDEO_CODECS_ALIASES.get(codec, codec) == encoder.vcodec
        assert info["video.pix_fmt"] == encoder.pix_fmt
        assert info["video.g"] == encoder.g
        assert info["video.crf"] == encoder.crf
        assert info["video.preset"] == encoder.preset
        # Same codec in as out: the synthetic parent is written by LeRobot's own encoder.
        parent = source.meta.info.features[video_key]["info"]
        assert parent["video.codec"] == codec
    assert out.meta.total_episodes == 2
    assert out.meta.total_frames == _EPISODE_LENGTHS[0] + _EPISODE_LENGTHS[2]


def test_split_output_decodes_at_first_middle_last_frame(synthetic_split):
    """The split's own decode gate ran; re-run it explicitly through the training path."""
    from lerobot.datasets.video_utils import decode_video_frames
    from lerobot.utils.import_utils import get_safe_default_video_backend

    from mulligan.tools.lerobot_fast_split import (
        DEFAULT_TOLERANCE_S,
        _verify_split_videos_decode,
    )

    _, out = synthetic_split
    _verify_split_videos_decode(out.meta)

    backend = get_safe_default_video_backend()
    for ep_idx in range(out.meta.total_episodes):
        ep = out.meta.episodes[ep_idx]
        length = int(ep["length"])
        for video_key in out.meta.video_keys:
            from_ts = float(ep[f"videos/{video_key}/from_timestamp"])
            path = out.root / out.meta.get_video_file_path(ep_idx, video_key)
            offsets = [0, length // 2, length - 1]
            frames = decode_video_frames(
                path,
                [from_ts + o / _FPS for o in offsets],
                DEFAULT_TOLERANCE_S,
                backend,
                return_uint8=True,
            )
            assert frames.shape == (len(offsets), 3, _HW, _HW)


def test_decode_gate_fails_on_a_truncated_video(synthetic_split, tmp_path):
    """A written video the trainer cannot open must fail the split, not ship.

    Truncation removes the trailing moov atom, so this pins the unopenable-file case.
    It deliberately does not claim more: a mid-file corruption that libdav1d conceals
    is NOT caught by either gate (see _verify_video_frame_bounds' docstring).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from mulligan.tools.lerobot_fast_split import _verify_split_videos_decode

    _, out = synthetic_split
    broken_root = tmp_path / "broken"
    shutil.copytree(out.root, broken_root)
    broken = LeRobotDataset("test/fast-split-out", root=str(broken_root))

    video_key = broken.meta.video_keys[0]
    path = broken_root / broken.meta.get_video_file_path(0, video_key)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])

    with pytest.raises(RuntimeError, match="not decodable by the training path"):
        _verify_split_videos_decode(broken.meta)


def test_frame_bounds_gate_actually_runs_and_catches_frame_loss(synthetic_split, tmp_path):
    """Regression: the frame-bounds sweep must run on freshly-created split metadata.

    ``LeRobotDatasetMetadata.create`` leaves ``meta.episodes`` as ``None``, so an early
    ``if episodes is None: return`` would make this verifier a no-op for every freshly
    written split. Re-encode one video shorter than its metadata requires and assert it fires.
    """
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from mulligan.tools.lerobot_fast_split import (
        _verify_video_frame_bounds,
        split_video_encoder,
    )

    _, out = synthetic_split
    short_root = tmp_path / "short"
    shutil.copytree(out.root, short_root)
    short = LeRobotDataset("test/fast-split-out", root=str(short_root))

    video_key = short.meta.video_keys[0]
    path = short_root / short.meta.get_video_file_path(0, video_key)
    # Passes unmodified (proves the gate runs rather than vacuously succeeding below).
    _verify_video_frame_bounds(short.meta)

    encoder = split_video_encoder()
    with av.open(str(path)) as src:
        frames = [f for packet in src.demux(src.streams.video[0]) for f in packet.decode()]
    keep = frames[:-2]
    assert keep, "fixture episode is too short to drop frames from"
    with av.open(str(path), mode="w") as dst:
        stream = dst.add_stream(
            encoder.vcodec, rate=_FPS, options=encoder.get_codec_options(as_strings=True)
        )
        stream.width, stream.height, stream.pix_fmt = _HW, _HW, encoder.pix_fmt
        for i, frame in enumerate(keep):
            frame.pts = i
            for pkt in stream.encode(frame):
                dst.mux(pkt)
        for pkt in stream.encode():
            dst.mux(pkt)

    reloaded = LeRobotDataset("test/fast-split-out", root=str(short_root))
    with pytest.raises(RuntimeError, match="decoded frames but metadata requires"):
        _verify_video_frame_bounds(reloaded.meta)


def test_info_codec_gate_fails_when_info_json_lies(synthetic_split, tmp_path):
    """info.json claiming a codec we did not encode must fail loudly."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from mulligan.tools.lerobot_fast_split import _verify_written_info_codec

    _, out = synthetic_split
    lying_root = tmp_path / "lying"
    shutil.copytree(out.root, lying_root)
    info_path = lying_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    video_key = out.meta.video_keys[0]
    info["features"][video_key]["info"]["video.codec"] = "h264"
    info_path.write_text(json.dumps(info))
    lying = LeRobotDataset("test/fast-split-out", root=str(lying_root))

    with pytest.raises(RuntimeError, match="meta/info.json says"):
        _verify_written_info_codec(lying.meta, split_video_encoder())

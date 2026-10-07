"""Fast, file-level LeRobot dataset splitting.

The per-frame copy used by the split scripts (``add_frame`` / ``save_episode``)
decodes and re-encodes *every* frame through torch + PIL. For a multi-camera real
dataset that is extremely slow, and it also crashes on partially-null columns
(``torch`` cannot infer the dtype of ``None``).

This module performs the same split via LeRobot's *file-level* internals — pandas
parquet copy + a single-pass PyAV segment re-encode — which is ~order-of-magnitude
faster and tolerant of null cells.

**There is exactly one encoder here: LeRobot's own RGB default**
(``lerobot.configs.video.rgb_encoder_defaults()`` — libsvtav1 / yuv420p / GOP g=2 /
CRF 30 / preset 12, the same object the collectors encode parents with). It is not
configurable, and no codec constant is re-pinned in this repo: a split output is
therefore in the same format as its AV1 parent, and a caller cannot ask for a codec
the process can already infer.

On top of the stock ``dataset_tools`` we add four things the public ``split_dataset``
does not expose but that we need:

1. **Per-group ``repo_id`` + ``root``.** ``split_dataset`` forces ``{repo_id}_{name}``;
   our datasets have explicit, unrelated names (e.g. ``...-baseline-uniform-r0``).
2. **Length-safe segment re-encode.** A few real parents have a per-camera timestamp
   span that rounds one frame short of the episode's parquet row count; the stock
   helper asserts. See ``_keep_episodes_from_video_with_av_len_safe``.
3. **Honest, gated ``info.json``.** ``split_dataset`` copies the source feature info
   verbatim, so the written files' real codec could differ from what ``info.json``
   claims. We re-probe every written video with ``get_video_info(path, encoder)``,
   raise if any file's stream codec disagrees with the encoder we just ran, write the
   merged probe+encoder info, and re-read it from disk to confirm.
4. **A post-write decode gate.** Every written video is random-accessed at the first,
   middle and last frame of every episode segment through the *training* decode path
   (``lerobot.datasets.video_utils.decode_video_frames`` on the backend the real-world
   readers auto-select). A split the trainer cannot read is a failed split, and it
   fails here rather than hours later inside a dataloader worker.

Partially-null columns (e.g. a ``telemetry.*`` feature added mid-collection, all-None
for the early episodes) otherwise crash ``aggregate_stats`` during the split. They are
handled by one pre-pass on the source before the stock split:
``fill_null_columns`` (preferred) fills the null cells with NaN and keeps the column;
``drop_columns`` removes it entirely.
"""

from __future__ import annotations

import inspect
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from lerobot.configs.video import (
    VIDEO_CODECS_ALIASES,
    RGBEncoderConfig,
    rgb_encoder_defaults,
)
from lerobot.datasets.dataset_tools import (
    _copy_and_reindex_data,
    _copy_and_reindex_episodes_metadata,
    add_features,
    remove_feature,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from lerobot.datasets.io_utils import load_episodes, write_info
from lerobot.datasets.video_utils import decode_video_frames, get_video_info
from tqdm import tqdm

from mulligan.real.robot.cameras import STATION_VIDEO_BACKEND

# LeRobot's own reader default, taken from its signature so it cannot drift here.
DEFAULT_TOLERANCE_S: float = (
    inspect.signature(LeRobotDataset.__init__).parameters["tolerance_s"].default
)


def split_video_encoder() -> RGBEncoderConfig:
    """The one and only encoder used for split outputs: LeRobot's RGB defaults.

    Single source of truth for codec / pixel format / GOP / CRF / preset. Callers do
    not get to override it — the split writes what LeRobot writes.
    """
    return rgb_encoder_defaults()


def is_null_cell(value) -> bool:
    """True if a per-frame cell is null: a literal ``None`` or a sequence of element-``None``.

    Mid-collection-added columns store missing frames as ``[None]*shape`` (a list whose
    elements are ``None``), which the torch transform cannot turn into a tensor, rather
    than a single ``None``. Both forms count as null here.
    """
    if value is None:
        return True
    try:
        return any(x is None for x in value)
    except TypeError:
        return False


def _require_episodes(meta: LeRobotDatasetMetadata):
    """Return the split's episode metadata, loading it from disk if not yet in memory.

    ``LeRobotDatasetMetadata.create`` leaves ``meta.episodes`` as ``None``; it is only
    populated lazily. Verifiers that read it must load it explicitly; reading the
    ``None`` and returning early would silently verify nothing.
    """
    if meta.episodes is None:
        meta.episodes = load_episodes(meta.root)
    if meta.episodes is None or len(meta.episodes) == 0:
        raise RuntimeError(f"{meta.repo_id}: split wrote no episode metadata to verify")
    if len(meta.episodes) != meta.total_episodes:
        raise RuntimeError(
            f"{meta.repo_id}: episode metadata has {len(meta.episodes)} rows but info.json "
            f"says {meta.total_episodes} episodes"
        )
    return meta.episodes


def _refresh_and_verify_video_info(meta: LeRobotDatasetMetadata, encoder: RGBEncoderConfig) -> None:
    """Write the *true* codec info into info.json and fail loudly on a mismatch.

    Every written video must have been produced by ``encoder``; a file whose stream
    codec disagrees means some path copied a whole file instead of re-encoding it ->
    a mixed-codec dataset, which we refuse to produce silently. The persisted info is
    ``get_video_info(path, video_encoder=encoder)`` — the same probe+encoder merge
    LeRobot itself writes at collection time, so a split output's ``info.json``
    round-trips through ``VideoEncoderConfig.from_video_info`` exactly like a parent's.
    """
    for video_key in meta.video_keys:
        files = sorted((meta.root / "videos" / video_key).glob("**/*.mp4"))
        if not files:
            raise RuntimeError(f"{meta.repo_id}: no written video files for {video_key}")
        info = None
        for path in files:
            probed = get_video_info(path, video_encoder=encoder)
            # get_video_info must populate "video.codec" for any real video stream; a missing
            # key (e.g. an empty/streamless file -> {}) is the root cause and should name itself,
            # not read None and produce a confusing "is 'None'" codec mismatch downstream.
            actual = probed["video.codec"]
            # The stream reports the canonical codec name ("av1"); the encoder is named by
            # its ffmpeg encoder ("libsvtav1"). LeRobot's own alias table maps between them.
            if VIDEO_CODECS_ALIASES.get(actual, actual) != encoder.vcodec:
                raise RuntimeError(
                    f"{meta.repo_id}: {path.name} for {video_key} is '{actual}', but the "
                    f"split encoder is '{encoder.vcodec}'. A whole-file copy preserved the "
                    f"source codec instead of re-encoding; the dataset would be mixed-codec."
                )
            if probed["video.pix_fmt"] != encoder.pix_fmt:
                raise RuntimeError(
                    f"{meta.repo_id}: {path.name} for {video_key} has pix_fmt "
                    f"'{probed['video.pix_fmt']}', expected '{encoder.pix_fmt}'."
                )
            if info is None:
                info = probed
        meta.info.features[video_key]["info"] = info
    write_info(meta.info, meta.root)
    _verify_written_info_codec(meta, encoder)


def _verify_written_info_codec(meta: LeRobotDatasetMetadata, encoder: RGBEncoderConfig) -> None:
    """Re-read ``meta/info.json`` from disk and confirm it names what we encoded.

    ``write_info`` succeeding is not evidence that the file on disk says the right
    thing (serializers drop keys, a stale in-memory ``info`` can be re-written). The
    split's contract is that a downstream reader opening the *file* sees the codec it
    will actually decode, so read the file back.
    """
    on_disk = json.loads((meta.root / "meta" / "info.json").read_text())
    for video_key in meta.video_keys:
        info = on_disk["features"][video_key]["info"]
        codec = info["video.codec"]
        if VIDEO_CODECS_ALIASES.get(codec, codec) != encoder.vcodec:
            raise RuntimeError(
                f"{meta.repo_id}: meta/info.json says {video_key} is '{codec}' but the "
                f"split encoder was '{encoder.vcodec}'"
            )
        if info["video.pix_fmt"] != encoder.pix_fmt:
            raise RuntimeError(
                f"{meta.repo_id}: meta/info.json says {video_key} pix_fmt is "
                f"'{info['video.pix_fmt']}', expected '{encoder.pix_fmt}'"
            )
        for field_name in ("g", "crf", "preset"):
            expected = getattr(encoder, field_name)
            actual = info[f"video.{field_name}"]
            if actual != expected:
                raise RuntimeError(
                    f"{meta.repo_id}: meta/info.json says {video_key} video.{field_name}="
                    f"{actual!r}, expected {expected!r} from the split encoder"
                )


def _verify_split_videos_decode(meta: LeRobotDatasetMetadata) -> None:
    """Decode-gate every written video through the TRAINING decode path.

    Random-accesses the first, middle and last frame of every episode segment of every
    video key with ``lerobot.datasets.video_utils.decode_video_frames`` on the backend
    the real-world readers auto-select (``get_safe_default_video_backend()``, i.e.
    torchcodec when installed) — the same call ``mulligan.data.fast_lerobot_reader`` and the
    ``mulligan.data.frame_cache`` self-check make at train time.

    The last frame of a segment is probed deliberately: a segment whose final GOP did
    not flush shows up there and nowhere else. Coverage is 3 frames per episode per
    camera, so this is a liveness gate on seeking, not a content-integrity check — see
    ``_verify_video_frame_bounds`` for what the full sweep does and does not catch.
    """
    # Pinned, not auto-detected: training resolves --video-backend from
    # camera_utils.STATION_VIDEO_BACKEND, and lerobot's get_safe_default_video_backend()
    # silently returns "pyav" when torchcodec is missing. A gate that passed on a decoder
    # training never uses would be worse than no gate.
    backend = STATION_VIDEO_BACKEND
    episodes = _require_episodes(meta)
    fps = meta.fps
    n_probes = 0
    for ep_idx in range(meta.total_episodes):
        ep = episodes[ep_idx]
        length = int(ep["length"])
        if length <= 0:
            raise RuntimeError(f"{meta.repo_id}: episode {ep_idx} has length {length}")
        frame_offsets = sorted({0, length // 2, length - 1})
        for video_key in meta.video_keys:
            from_ts = float(ep[f"videos/{video_key}/from_timestamp"])
            path = meta.root / meta.get_video_file_path(ep_idx, video_key)
            timestamps = [from_ts + offset / fps for offset in frame_offsets]
            try:
                frames = decode_video_frames(
                    path, timestamps, DEFAULT_TOLERANCE_S, backend, return_uint8=True
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{meta.repo_id}: split output {path} is not decodable by the training "
                    f"path (backend={backend!r}, episode {ep_idx} {video_key}, frame offsets "
                    f"{frame_offsets}, timestamps {timestamps}): {exc!r}"
                ) from exc
            if len(frames) != len(timestamps):
                raise RuntimeError(
                    f"{meta.repo_id}: decoding {path} at {len(timestamps)} timestamps for "
                    f"episode {ep_idx} {video_key} returned {len(frames)} frame(s)"
                )
            n_probes += len(timestamps)
    print(f"  decode gate OK: {n_probes} random-access frame reads via backend={backend}")


def _count_video_frames(path: Path) -> int:
    import av

    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]
        return sum(1 for packet in container.demux(stream) for _ in packet.decode())


def _encode_frame(out, v_out, frame, frame_count: int, time_base: Fraction) -> None:
    new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
    new_frame.pts = frame_count
    new_frame.time_base = time_base
    for pkt in v_out.encode(new_frame):
        out.mux(pkt)


def _keep_episodes_from_video_with_av_len_safe(
    input_path: Path,
    output_path: Path,
    episodes_to_keep: list[tuple[int, int, int]],
    fps: float,
    encoder: RGBEncoderConfig,
    *,
    allow_padding: bool = True,
) -> None:
    """Re-encode episode segments and preserve the data-row length per segment.

    Some real parent datasets have a per-camera ``to_timestamp - from_timestamp``
    range that rounds one frame shorter than the episode's parquet row count. LeRobot
    0.5.2 asserts on that mismatch before re-encoding. For training splits, the data
    rows are the contract: keep the timestamp-selected frames, then duplicate the
    segment's terminal decoded frame if needed so the output video has exactly
    ``episode.length`` frames for each copied episode. Padding happens per segment, not
    only at file end, so downstream episode boundaries remain aligned.
    """
    import av

    if not episodes_to_keep:
        raise ValueError("No episodes to keep")

    in_container = av.open(str(input_path))
    if not in_container.streams.video:
        raise ValueError(f"No video streams found in {input_path}")
    v_in = in_container.streams.video[0]

    out = av.open(str(output_path), mode="w")
    fps_fraction = Fraction(fps).limit_denominator(1000)
    v_out = out.add_stream(
        encoder.vcodec,
        rate=fps_fraction,
        options=encoder.get_codec_options(as_strings=True),
    )
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = encoder.pix_fmt
    frame_time_base = Fraction(1, int(fps))
    v_out.time_base = frame_time_base
    out.start_encoding()

    ranges = sorted(episodes_to_keep)
    range_idx = 0
    src_frame_count = 0
    dst_frame_count = 0
    written_in_segment = 0
    last_segment_frame = None
    padded_frames = 0

    def finish_segment() -> None:
        nonlocal dst_frame_count, written_in_segment, last_segment_frame, padded_frames
        if range_idx >= len(ranges):
            return
        expected_len = ranges[range_idx][2]
        if written_in_segment > expected_len:
            raise RuntimeError(
                f"{input_path}: segment {range_idx} wrote {written_in_segment} frames, "
                f"expected {expected_len}"
            )
        if written_in_segment < expected_len and last_segment_frame is None:
            raise RuntimeError(
                f"{input_path}: segment {range_idx} decoded no frames but expected {expected_len}"
            )
        if written_in_segment < expected_len and not allow_padding:
            raise RuntimeError(
                f"{input_path}: segment {range_idx} decoded {written_in_segment} frame(s), "
                f"expected {expected_len}; video padding is disabled"
            )
        while written_in_segment < expected_len:
            _encode_frame(out, v_out, last_segment_frame, dst_frame_count, frame_time_base)
            dst_frame_count += 1
            written_in_segment += 1
            padded_frames += 1

    try:
        for packet in in_container.demux(v_in):
            for frame in packet.decode():
                while range_idx < len(ranges) and src_frame_count >= ranges[range_idx][1]:
                    finish_segment()
                    range_idx += 1
                    written_in_segment = 0
                    last_segment_frame = None
                if range_idx >= len(ranges):
                    break

                start_frame, _end_frame, expected_len = ranges[range_idx]
                if src_frame_count < start_frame:
                    src_frame_count += 1
                    continue

                if written_in_segment < expected_len:
                    _encode_frame(out, v_out, frame, dst_frame_count, frame_time_base)
                    dst_frame_count += 1
                    written_in_segment += 1
                    last_segment_frame = frame
                src_frame_count += 1
            if range_idx >= len(ranges):
                break

        while range_idx < len(ranges):
            finish_segment()
            range_idx += 1
            written_in_segment = 0
            last_segment_frame = None

        for pkt in v_out.encode():
            out.mux(pkt)
    finally:
        out.close()
        in_container.close()

    if padded_frames:
        logging.info(
            "Padded %s terminal frame(s) while splitting %s -> %s",
            padded_frames,
            input_path,
            output_path,
        )


def _copy_and_reindex_videos_len_safe(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
    *,
    encoder: RGBEncoderConfig,
    allow_padding: bool = True,
) -> dict[int, dict]:
    episodes_video_metadata: dict[int, dict] = {new_idx: {} for new_idx in episode_mapping.values()}

    for video_key in src_dataset.meta.video_keys:
        if dst_meta.video_path is None:
            raise ValueError("Destination metadata has no video_path defined")

        file_to_episodes: dict[tuple[int, int], list[int]] = {}
        for old_idx in episode_mapping:
            src_ep = src_dataset.meta.episodes[old_idx]
            file_key = (
                int(src_ep[f"videos/{video_key}/chunk_index"]),
                int(src_ep[f"videos/{video_key}/file_index"]),
            )
            file_to_episodes.setdefault(file_key, []).append(old_idx)

        for (src_chunk_idx, src_file_idx), episodes_in_file in tqdm(
            sorted(file_to_episodes.items()), desc=f"Processing {video_key} video files"
        ):
            sorted_keep_episodes = sorted(episodes_in_file, key=lambda x: episode_mapping[x])
            ranges: list[tuple[int, int, int]] = []
            for old_idx in sorted_keep_episodes:
                src_ep = src_dataset.meta.episodes[old_idx]
                from_frame = round(
                    float(src_ep[f"videos/{video_key}/from_timestamp"]) * src_dataset.meta.fps
                )
                to_frame = round(
                    float(src_ep[f"videos/{video_key}/to_timestamp"]) * src_dataset.meta.fps
                )
                expected_len = int(src_ep["length"])
                if to_frame <= from_frame:
                    raise RuntimeError(
                        f"{src_dataset.repo_id} {video_key} ep {old_idx}: empty video range "
                        f"{from_frame}:{to_frame}"
                    )
                if to_frame - from_frame != expected_len:
                    message = (
                        f"{src_dataset.repo_id} {video_key} ep {old_idx}: timestamp range has "
                        f"{to_frame - from_frame} frame(s), data length is {expected_len}"
                    )
                    if not allow_padding:
                        raise RuntimeError(f"{message}; video padding/truncation is disabled")
                    logging.warning(
                        "%s; splitter will pad/truncate this segment to the data length",
                        message,
                    )
                ranges.append((from_frame, to_frame, expected_len))

            assert src_dataset.meta.video_path is not None
            src_video_path = src_dataset.root / src_dataset.meta.video_path.format(
                video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
            )
            dst_video_path = dst_meta.root / dst_meta.video_path.format(
                video_key=video_key, chunk_index=src_chunk_idx, file_index=src_file_idx
            )
            dst_video_path.parent.mkdir(parents=True, exist_ok=True)
            _keep_episodes_from_video_with_av_len_safe(
                src_video_path,
                dst_video_path,
                ranges,
                src_dataset.meta.fps,
                encoder,
                allow_padding=allow_padding,
            )

            cumulative_ts = 0.0
            for old_idx in sorted_keep_episodes:
                new_idx = episode_mapping[old_idx]
                ep_length = int(src_dataset.meta.episodes[old_idx]["length"])
                ep_duration = ep_length / src_dataset.meta.fps
                episodes_video_metadata[new_idx][f"videos/{video_key}/chunk_index"] = src_chunk_idx
                episodes_video_metadata[new_idx][f"videos/{video_key}/file_index"] = src_file_idx
                episodes_video_metadata[new_idx][f"videos/{video_key}/from_timestamp"] = (
                    cumulative_ts
                )
                episodes_video_metadata[new_idx][f"videos/{video_key}/to_timestamp"] = (
                    cumulative_ts + ep_duration
                )
                cumulative_ts += ep_duration

    return episodes_video_metadata


def _verify_video_frame_bounds(meta: LeRobotDatasetMetadata) -> None:
    """Decode every written video end to end and check it yields enough frames.

    ``_count_video_frames`` demuxes and decodes the whole file via PyAV, so this is the
    full-sweep pass — but its signal is the resulting frame COUNT, not an exception.
    On a corrupted AV1 file libdav1d logs the bad OBU and drops the affected frames,
    returning a short count rather than raising. So this catches frame
    LOSS (truncation, a dropped segment, a short final GOP); corruption that the decoder
    conceals while preserving the frame count passes, here and in the random-access gate.
    ``_verify_split_videos_decode`` covers the complementary axis — seek-based random
    access through the backend training actually decodes with.
    """
    episodes = _require_episodes(meta)
    for video_key in meta.video_keys:
        video_files = sorted(
            {
                (
                    int(ep[f"videos/{video_key}/chunk_index"]),
                    int(ep[f"videos/{video_key}/file_index"]),
                )
                for ep in episodes
            }
        )
        for chunk_idx, file_idx in video_files:
            rows = [
                ep
                for ep in episodes
                if int(ep[f"videos/{video_key}/chunk_index"]) == chunk_idx
                and int(ep[f"videos/{video_key}/file_index"]) == file_idx
            ]
            required = max(
                round(float(ep[f"videos/{video_key}/to_timestamp"]) * meta.fps) for ep in rows
            )
            assert meta.video_path is not None
            path = meta.root / meta.video_path.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            actual = _count_video_frames(path)
            if actual < required:
                raise RuntimeError(
                    f"{meta.repo_id}: {video_key} file-{file_idx:03d} has {actual} decoded "
                    f"frames but metadata requires {required}"
                )


def nan_fill_columns(
    source_ds: LeRobotDataset,
    columns: Sequence[str],
    output_dir: str | Path,
) -> LeRobotDataset:
    """Return a copy of ``source_ds`` with null cells in ``columns`` filled with NaN.

    Why: a column that is all-null for some episodes (e.g. a telemetry feature added
    mid-collection) carries ``None`` cells, which (a) make ``torch`` raise
    ``Could not infer dtype of NoneType`` at train time and (b) leave per-episode stats
    as ``None``, which crashes ``aggregate_stats`` during a split/finalize.

    Filling those cells with a real ``float32`` NaN array keeps the column (real values
    where present, NaN where missing), fixes both crashes, and is semantically honest
    (NaN == missing). The column is fixed at two levels, because the nulls live at both:

    - **Per-frame data**: early-episode cells are a *list of element ``None``s* (e.g.
      ``[None]*7``), which the ``torch`` transform cannot turn into a tensor. We rebuild
      the column (``remove_feature`` + ``add_features``) with NaN arrays for those cells.
    - **Per-episode stats**: those episodes carry ``stats/<col>/min = None`` etc. in
      ``meta/episodes``; ``remove_feature``/``add_features`` copy them verbatim, so a later
      ``aggregate_stats`` still chokes. We drop the stale ``stats/<col>/*`` columns from
      the rebuilt dataset's episode metadata (the refilled column then carries no stats —
      harmless, since training does not normalize these columns).

    Load-bearing contract: the per-frame fill (the ``_fill`` closure below) indexes a lookup
    built exhaustively over every ``(episode_index, frame_index)`` of the source, and relies on
    ``LeRobotDataset.add_features`` replaying *exactly those* rows in any order. If a future
    ``add_features`` refactor skips, reorders, or batches rows differently, ``_fill`` raises
    ``KeyError`` naming the drifted ``(ep, frame)`` (it indexes directly, never defaults to NaN).
    """
    columns = list(columns)
    missing = [c for c in columns if c not in source_ds.meta.features]
    if missing:
        raise KeyError(f"nan_fill_columns: columns not present in source: {missing}")

    hf = source_ds.hf_dataset.with_format(None)
    ep_col = list(hf["episode_index"])
    fr_col = list(hf["frame_index"])

    feature_specs: dict[str, tuple] = {}
    for col in columns:
        info = dict(source_ds.meta.features[col])
        if "dtype" not in info or "shape" not in info:
            raise ValueError(f"nan_fill_columns: feature {col!r} missing dtype/shape: {info}")
        shape = tuple(int(s) for s in info["shape"])
        raw = list(hf[col])
        nan_value = np.full(shape, np.nan, dtype=np.float32)
        lookup: dict[tuple[int, int], np.ndarray] = {}
        n_null = 0
        for ep, fr, value in zip(ep_col, fr_col, raw):
            if is_null_cell(value):
                lookup[(int(ep), int(fr))] = nan_value
                n_null += 1
            else:
                lookup[(int(ep), int(fr))] = np.asarray(value, dtype=np.float32)
        print(f"  nan_fill {col}: {n_null}/{len(raw)} null cells -> NaN")

        def _fill(row, _ep_idx, _frame_in_ep, _lookup=lookup):
            # _lookup is built exhaustively over every (episode_index, frame_index) of the
            # source dataset above, and add_features replays the same rows. A missing key is
            # therefore a logic bug (row set drift), not a valid null frame — index directly so
            # KeyError names the offending (ep, frame) instead of silently writing NaN.
            return _lookup[(int(row["episode_index"]), int(row["frame_index"]))]

        feature_specs[col] = (_fill, info)

    base = Path(output_dir)
    strip_root = base / "_strip"
    fill_root = base / "_fill"
    for path in (strip_root, fill_root):
        if path.exists():
            shutil.rmtree(path)
    stripped = remove_feature(
        source_ds, columns, output_dir=strip_root, repo_id=f"{source_ds.repo_id}__strip"
    )
    filled = add_features(
        stripped, feature_specs, output_dir=fill_root, repo_id=f"{source_ds.repo_id}__filled"
    )
    _drop_episode_stat_columns(Path(filled.root), columns)
    shutil.rmtree(strip_root, ignore_errors=True)
    # Reload so meta reflects the dropped per-episode stat columns.
    return LeRobotDataset(filled.repo_id, root=filled.root)


def _drop_episode_stat_columns(ds_root: Path, columns: Sequence[str]) -> None:
    """Remove stale ``stats/<col>/*`` columns from a dataset's ``meta/episodes`` parquet.

    ``remove_feature`` / ``add_features`` copy episode metadata verbatim, so per-episode
    stat columns for a column with null episodes survive as ``None`` and later crash
    ``aggregate_stats``. Dropping them leaves the (refilled) column statless, which is
    fine for columns training does not normalize.
    """
    import pandas as pd

    prefixes = tuple(f"stats/{col}/" for col in columns)
    for parquet in sorted((ds_root / "meta" / "episodes").glob("**/*.parquet")):
        df = pd.read_parquet(parquet)
        drop = [c for c in df.columns if c.startswith(prefixes)]
        if drop:
            df.drop(columns=drop).to_parquet(parquet, index=False)


def fast_split_dataset(
    source_ds: LeRobotDataset,
    groups: Mapping[str, Mapping[str, object]],
    *,
    drop_columns: Sequence[str] | None = None,
    fill_null_columns: Sequence[str] | None = None,
    intermediate_root: str | Path | None = None,
    allow_video_padding: bool = True,
) -> dict[str, LeRobotDataset]:
    """Split ``source_ds`` into per-group datasets via file-level copy.

    Videos are re-encoded with LeRobot's RGB defaults (``split_video_encoder()``) and
    nothing else; there is no codec/preset/CRF parameter to pass. Every written video
    is codec-verified and decode-gated before this returns.

    Args:
        source_ds: The (full) source LeRobotDataset.
        groups: ``{name: {"repo_id": str, "root": path, "episodes": [src_ep_idx, ...]}}``.
            ``episodes`` are *source* episode indices; they are re-indexed to ``0..n-1``
            within each output in ascending source order.
        drop_columns: Columns removed (via one ``remove_feature`` pass) before the split.
            Use for analysis-only / partially-null columns that training does not consume.
        fill_null_columns: Columns whose null cells are filled with NaN before the split
            (see ``nan_fill_columns``). Preferred over ``drop_columns`` when the
            partially-null data is worth keeping. Applied before ``drop_columns``.
        intermediate_root: Where the temporary clean copies are written (removed
            afterwards). Defaults to ``<first group root>.parent/_fast_split_clean``.

    Returns:
        ``{name: LeRobotDataset}`` for the written splits.
    """
    if not groups:
        raise ValueError("No groups provided")
    encoder = split_video_encoder()
    depth_keys = [key for key in source_ds.meta.depth_keys if key in source_ds.meta.video_keys]
    if depth_keys:
        # The one encoder here is LeRobot's RGB default; depth streams need a
        # DepthEncoderConfig (hevc / gray12le / lossless) plus a dequantization contract.
        # Refuse rather than silently transcode depth through an RGB encoder.
        raise NotImplementedError(
            f"{source_ds.repo_id}: fast_split_dataset only encodes RGB video; depth video "
            f"key(s) {depth_keys} need a depth encoder"
        )

    roots = {name: Path(spec["root"]) for name, spec in groups.items()}  # type: ignore[arg-type]
    for name, root in roots.items():
        if root.exists():
            raise FileExistsError(
                f"Output root for group {name!r} already exists: {root}. Remove it first "
                f"(the splitter's --overwrite does this)."
            )

    base = (
        Path(intermediate_root)
        if intermediate_root
        else next(iter(roots.values())).parent / "_fast_split_clean"
    )
    cleanup_root: Path | None = None
    ds = source_ds
    if fill_null_columns:
        fill_list = list(fill_null_columns)
        missing = [c for c in fill_list if c not in source_ds.meta.features]
        if missing:
            raise KeyError(f"--fill-null-columns not present in source: {missing}")
        if base.exists():
            shutil.rmtree(base)
        print(f"  filling null cells with NaN for {fill_list} -> {base}")
        ds = nan_fill_columns(ds, fill_list, output_dir=base)
        cleanup_root = base
    if drop_columns:
        drop_list = list(drop_columns)
        present = [c for c in drop_list if c in ds.meta.features]
        missing = [c for c in drop_list if c not in ds.meta.features]
        if missing:
            raise KeyError(f"--drop-columns not present in source: {missing}")
        if present:
            if cleanup_root is None and base.exists():
                shutil.rmtree(base)
            drop_root = base / "_nocols"
            print(f"  dropping columns {present} via remove_feature -> {drop_root}")
            ds = remove_feature(ds, present, output_dir=drop_root, repo_id=f"{ds.repo_id}__nocols")
            # remove_feature leaves stale stats/<col>/* columns in episode metadata, which
            # would still crash aggregate_stats during the split; drop them too.
            _drop_episode_stat_columns(Path(ds.root), present)
            ds = LeRobotDataset(ds.repo_id, root=ds.root)
            cleanup_root = base

    try:
        results: dict[str, LeRobotDataset] = {}
        use_videos = len(ds.meta.video_keys) > 0
        for name, spec in groups.items():
            eps = sorted(int(e) for e in spec["episodes"])  # type: ignore[index]
            if not eps:
                raise ValueError(f"Group {name!r} has no episodes")
            if eps[0] < 0 or eps[-1] >= ds.meta.total_episodes:
                raise ValueError(
                    f"Group {name!r} episode indices out of range: {eps[0]}..{eps[-1]}"
                )
            episode_mapping = {old: new for new, old in enumerate(eps)}
            repo_id = str(spec["repo_id"])  # type: ignore[index]
            root = roots[name]
            meta = LeRobotDatasetMetadata.create(
                repo_id=repo_id,
                fps=ds.meta.fps,
                features=ds.meta.features,
                robot_type=ds.meta.robot_type,
                root=root,
                use_videos=use_videos,
                chunks_size=ds.meta.chunks_size,
                data_files_size_in_mb=ds.meta.data_files_size_in_mb,
                video_files_size_in_mb=ds.meta.video_files_size_in_mb,
            )
            if use_videos:
                # The upstream helper asserts that every source video's timestamp span
                # rounds to the episode length. A few real runs violate that by one
                # frame while still carrying valid data rows; use the length-safe
                # variant so split outputs preserve the parquet-row contract.
                video_md = _copy_and_reindex_videos_len_safe(
                    ds,
                    meta,
                    episode_mapping,
                    encoder=encoder,
                    allow_padding=allow_video_padding,
                )
            else:
                video_md = None
            data_md = _copy_and_reindex_data(ds, meta, episode_mapping)
            _copy_and_reindex_episodes_metadata(ds, meta, episode_mapping, data_md, video_md)
            if use_videos:
                _refresh_and_verify_video_info(meta, encoder)
                _verify_video_frame_bounds(meta)
                _verify_split_videos_decode(meta)
            results[name] = LeRobotDataset(repo_id=repo_id, root=root)
            print(
                f"  [{name}] {results[name].meta.total_episodes} eps, "
                f"{results[name].meta.total_frames} frames -> {root}"
            )
        return results
    finally:
        if cleanup_root is not None:
            shutil.rmtree(cleanup_root, ignore_errors=True)


def last_frame_success_by_episode(source_ds: LeRobotDataset) -> list[bool]:
    """Null-safe per-episode terminal ``success`` flags.

    Reads the ``success`` column with ``with_format(None)`` (no torch transform) so it
    tolerates datasets that carry partially-null columns — unlike ``source_ds[idx]``,
    which raises ``Could not infer dtype of NoneType``.
    """
    hf = source_ds.hf_dataset.with_format(None)
    success = list(hf["success"])
    out: list[bool] = []
    for ep_idx in range(source_ds.meta.total_episodes):
        end = int(source_ds.meta.episodes[ep_idx]["dataset_to_index"])
        value = success[end - 1]
        out.append(int(value[0] if isinstance(value, (list, tuple)) else value) == 1)
    return out

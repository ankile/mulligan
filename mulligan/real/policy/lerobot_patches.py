# _mulligan_lossless_feed_frame is adapted from LeRobot (https://github.com/huggingface/lerobot,
# commit 0530dd9b), StreamingVideoEncoder.feed_frame in src/lerobot/datasets/video_utils.py.
# Modified by the Mulligan authors: a full encoder queue blocks the caller instead of dropping
# the frame, and an unknown video key raises. The rest of the file is by the Mulligan authors
# (MIT).
#
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
One-time monkey-patches for LeRobot on the real robot.

Import this module before importing LeRobotDataset to apply patches:

    import mulligan.real.policy.lerobot_patches  # noqa: F401
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

Patches applied:
  - Default to streaming (real-time) video encoding for collection AND eval saving
    (LeRobotDataset.create / .resume streaming_encoding=True + camera_encoder pinned to
    the Mulligan codec), so teleop / blind_dagger / blind_eval / manifest_eval stream AV1
    frames straight to the encoder instead of buffering PNGs. Toggle with
    MULLIGAN_STREAMING_ENCODING=0. Optional resource knobs:
    MULLIGAN_ENCODER_QUEUE_MAXSIZE=<positive int> and MULLIGAN_ENCODER_THREADS=<positive int>
    (default: 4 per camera, matching 4 cameras on a 16-CPU robot workstation).
  - FF write-API compatibility shims: lerobot 0.5.x's DatasetReader/DatasetWriter split
    moved ``start_image_writer`` / ``stop_image_writer`` off ``LeRobotDataset`` onto the
    internal ``DatasetWriter`` and made the bare ``LeRobotDataset(...)`` constructor
    read-only. The real-robot save sites (collection + eval) still call those methods on
    the dataset, so add thin delegating shims (fail loud on a read-only dataset).
  - Mulligan real-data video contract: streaming video encoding blocks instead of dropping
    frames, and episode video timestamp spans are written from the saved data row count
    instead of the container-reported duration. A camera video row must stay one-to-one
    with every saved LeRobot frame; if the encoder cannot keep up, collection should slow
    down or fail, not create a dataset that later needs splitter-side repair.
  - Mulligan LeRobot runtime patches: diffusion-policy fixes and Mulligan extensions
    used by training/eval entry points.

Video codec (libsvtav1/AV1: ~2x smaller than h264, decodes faster under torchcodec, at
comparable encode time — lerobot defaults libsvtav1 to a fast preset). The codec is set by Patch
1's per-camera encoder
config injection (``rgb_encoder=RGBEncoderConfig(vcodec=...)`` on ``create``/``resume``),
which runs unconditionally and therefore covers BOTH the streaming and the batch encode
paths — ``DatasetWriter`` resolves the codec from that config, not from an
``encode_video_frames`` default. Override with the env var MULLIGAN_VIDEO_CODEC
(h264 | hevc | libsvtav1) to select another codec.

The streaming-encoding + write-API shims are guarded separately from the Mulligan runtime
patches: a lerobot that predates the FF streaming/VideoEncoderConfig API (e.g. 0.4.x)
gets no Mulligan codec/streaming default at all (there is no encoder config to inject), but
must still receive the Mulligan runtime patches.

Application is idempotent: a marker on the patched lerobot module makes a second import
(module reload, or an import under a second name) a no-op. Re-running the body would
otherwise re-introspect the ALREADY-PATCHED ``LeRobotDataset.create`` signature
(``cls/args/kwargs``) and raise the "encoder-config API moved again" RuntimeError, and
would re-wrap each already-wrapped function.
"""

import logging as _logging
import os as _os
import queue as _queue
import time as _time

_MULLIGAN_DEFAULT_VCODEC = _os.environ.get("MULLIGAN_VIDEO_CODEC", "libsvtav1")


def _mulligan_positive_int_env(name: str) -> int | None:
    raw = _os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


_MULLIGAN_ENCODER_QUEUE_MAXSIZE = _mulligan_positive_int_env("MULLIGAN_ENCODER_QUEUE_MAXSIZE")
_MULLIGAN_ENCODER_THREADS = _mulligan_positive_int_env("MULLIGAN_ENCODER_THREADS") or 4

# Idempotence marker. Stored on the PATCHED lerobot module rather than in this module's
# own namespace so it survives an ``importlib.reload`` of this module AND an import of it
# under a second name — in both cases lerobot itself is not re-imported, so the patches are
# already installed. See the module docstring for why re-application would crash.
_MULLIGAN_PATCH_MARKER = "_mulligan_real_lerobot_patches_applied"

try:
    from lerobot.datasets import dataset_writer as _dw_mod
    from lerobot.datasets import lerobot_dataset as _ld_mod
    from lerobot.datasets import video_utils as _vu_mod
except ImportError:
    _logging.getLogger(__name__).warning(
        "lerobot.datasets not available — Mulligan real-robot lerobot patches not applied. "
        "Dataset saving will fail if attempted."
    )
    _MULLIGAN_APPLY_PATCHES = False
else:
    _MULLIGAN_APPLY_PATCHES = not getattr(_ld_mod, _MULLIGAN_PATCH_MARKER, False)
    if not _MULLIGAN_APPLY_PATCHES:
        _logging.getLogger(__name__).debug(
            "Mulligan real-robot lerobot patches already applied in this interpreter; skipping."
        )

if _MULLIGAN_APPLY_PATCHES:
    # --------------------------------------------------------------------------- #
    # Patch 1: Default to streaming (real-time) AV1 video encoding for create + resume,
    #          and Patch 2: FF write-API shims. Both depend on the FF streaming API
    #          (VideoEncoderConfig + LeRobotDataset.resume). Guard them so a pre-FF
    #          lerobot still gets the Mulligan runtime patches below (it gets no Mulligan codec
    #          default at all — there is no encoder config to inject).
    # --------------------------------------------------------------------------- #
    # lerobot 0.5.x can stream frames straight to the video encoder during recording
    # (``streaming_encoding=True``) instead of buffering PNG frames and batch-encoding at
    # ``save_episode``: lower disk I/O + memory and faster saves. Default it ON for every
    # ``LeRobotDataset.create`` AND ``.resume`` (teleop / blind_dagger / blind_eval /
    # manifest_eval all funnel through one of these — create on first write,
    # resume to append to an existing dataset), and pin the per-camera encoder to the Mulligan
    # default codec (libsvtav1/AV1) so the streaming path encodes AV1 and honors the
    # ``MULLIGAN_VIDEO_CODEC`` rollback. Toggle streaming with ``MULLIGAN_STREAMING_ENCODING`` (set
    # to 0 to fall back to batch encoding).
    #
    # NOTE: streaming changes the on-robot recording/eval-save flow — validate it on the
    # next live session and set ``MULLIGAN_STREAMING_ENCODING=0`` if the encoder cannot keep up
    # at the camera frame rate.
    try:
        from lerobot.configs.video import VideoEncoderConfig as _VideoEncoderConfig
    except ImportError:
        _logging.getLogger(__name__).warning(
            "lerobot.configs.video.VideoEncoderConfig unavailable (pre-FF lerobot?); "
            "skipping streaming-encoding + write-API compatibility patches. "
            "Dataset saving falls back to the lerobot-native (batch) path."
        )
    else:
        _MULLIGAN_STREAMING_ENCODING = _os.environ.get(
            "MULLIGAN_STREAMING_ENCODING", "1"
        ).lower() not in (
            "0",
            "false",
            "no",
            "",
        )

        # lerobot renamed the per-camera encoder kwarg with the depth-maps feature
        # (3dd19d04, between v0.5.1-142 and -169): ``camera_encoder: VideoEncoderConfig``
        # became ``rgb_encoder: RGBEncoderConfig`` (a VideoEncoderConfig subclass).
        # Resolve the installed API ONCE from create()'s real signature — injecting the
        # wrong name is a TypeError on every dataset open, and silently skipping the
        # injection would drop the Mulligan codec/streaming defaults.
        import inspect as _inspect

        _create_params = _inspect.signature(_ld_mod.LeRobotDataset.create.__func__).parameters
        if "rgb_encoder" in _create_params:
            from lerobot.configs.video import RGBEncoderConfig as _EncoderCfg

            _ENCODER_KWARG = "rgb_encoder"
        elif "camera_encoder" in _create_params:
            _EncoderCfg = _VideoEncoderConfig
            _ENCODER_KWARG = "camera_encoder"
        else:
            raise RuntimeError(
                "LeRobotDataset.create accepts neither 'rgb_encoder' nor 'camera_encoder' — "
                "the encoder-config API moved again; update mulligan/real/policy/lerobot_patches.py "
                f"(params: {sorted(_create_params)})"
            )

        def _inject_streaming_kwargs(kwargs: dict) -> dict:
            """Default streaming_encoding ON and pin the Mulligan codec, unless the caller set them."""
            if _MULLIGAN_STREAMING_ENCODING and "streaming_encoding" not in kwargs:
                kwargs["streaming_encoding"] = True
            if (
                _MULLIGAN_ENCODER_QUEUE_MAXSIZE is not None
                and "encoder_queue_maxsize" not in kwargs
            ):
                kwargs["encoder_queue_maxsize"] = _MULLIGAN_ENCODER_QUEUE_MAXSIZE
            if _MULLIGAN_ENCODER_THREADS is not None and "encoder_threads" not in kwargs:
                kwargs["encoder_threads"] = _MULLIGAN_ENCODER_THREADS
            # Pin the per-camera encoder to the Mulligan codec when the caller did not specify
            # one, so BOTH the streaming and batch paths encode the Mulligan default
            # (libsvtav1) and a MULLIGAN_VIDEO_CODEC override applies to the streaming path
            # too (which keys off the encoder config, not encode_video_frames' vcodec).
            if kwargs.get(_ENCODER_KWARG) is None:
                kwargs[_ENCODER_KWARG] = _EncoderCfg(vcodec=_MULLIGAN_DEFAULT_VCODEC)
            return kwargs

        _original_create = _ld_mod.LeRobotDataset.create.__func__

        def _streaming_av1_create(cls, *args, **kwargs):
            return _original_create(cls, *args, **_inject_streaming_kwargs(kwargs))

        _ld_mod.LeRobotDataset.create = classmethod(_streaming_av1_create)

        _original_resume = _ld_mod.LeRobotDataset.resume.__func__

        def _streaming_av1_resume(cls, *args, **kwargs):
            return _original_resume(cls, *args, **_inject_streaming_kwargs(kwargs))

        _ld_mod.LeRobotDataset.resume = classmethod(_streaming_av1_resume)

        # ----------------------------------------------------------------------- #
        # Patch 2: start_image_writer / stop_image_writer shims.
        # ----------------------------------------------------------------------- #
        # The FF DatasetReader/DatasetWriter split moved these methods off
        # LeRobotDataset and onto the internal DatasetWriter. Mulligan's real-robot save
        # sites still call them on the dataset; add delegating shims so those sites
        # keep working without editing every call site. Only added when absent, so a
        # future lerobot that re-exposes them natively is not shadowed.
        if not hasattr(_ld_mod.LeRobotDataset, "start_image_writer"):

            def _ld_start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
                """Mulligan FF-compat shim: delegate to the active DatasetWriter.

                Fails loud on a read-only dataset (a bare ``LeRobotDataset(...)`` re-open
                has ``writer is None``) instead of silently no-op'ing and then crashing on
                the first ``add_frame`` — re-open with ``LeRobotDataset.resume(...)`` to
                append, or ``.create(...)`` to start a new recording.
                """
                writer = getattr(self, "writer", None)
                if writer is None:
                    raise RuntimeError(
                        "start_image_writer() called on a read-only LeRobotDataset (no "
                        "writer). FF lerobot makes the bare constructor read-only; re-open "
                        "with LeRobotDataset.resume(...) (append) or .create(...) (new)."
                    )
                writer.start_image_writer(num_processes, num_threads)

            def _ld_stop_image_writer(self) -> None:
                """Mulligan FF-compat shim: delegate stop to the writer; no-op when read-only."""
                writer = getattr(self, "writer", None)
                if writer is not None:
                    writer.stop_image_writer()

            _ld_mod.LeRobotDataset.start_image_writer = _ld_start_image_writer
            _ld_mod.LeRobotDataset.stop_image_writer = _ld_stop_image_writer

        # ----------------------------------------------------------------------- #
        # Patch 3: Mulligan lossless real-data video contract.
        # ----------------------------------------------------------------------- #
        # FF lerobot's streaming encoder is appropriate for lossy demos: when the
        # per-camera queue fills, feed_frame() logs and drops frames. For Mulligan datasets,
        # a dropped camera frame corrupts the LeRobot row<->video-frame contract and
        # later makes fast split/re-encode paths either fail or pad. Block the save
        # thread until the encoder accepts the frame instead. This runs in Mulligan's
        # background save thread after an episode ends, so backpressure affects save
        # latency rather than live robot control.
        def _mulligan_lossless_feed_frame(self, video_key: str, image) -> None:
            if not self._episode_active:
                raise RuntimeError("No active episode. Call start_episode() first.")
            if video_key not in self._frame_queues:
                raise KeyError(
                    f"Unknown video key {video_key!r}; active keys: {sorted(self._frame_queues)}"
                )

            frame = image.copy()
            last_wait_log = _time.monotonic()
            while True:
                thread = self._threads[video_key]
                if not thread.is_alive():
                    try:
                        status, msg = self._result_queues[video_key].get_nowait()
                    except _queue.Empty:
                        status, msg = None, None
                    if status == "error":
                        raise RuntimeError(f"Encoder thread for {video_key} crashed: {msg}")
                    raise RuntimeError(f"Encoder thread for {video_key} is not alive")

                try:
                    self._frame_queues[video_key].put(frame, timeout=1.0)
                    return
                except _queue.Full:
                    now = _time.monotonic()
                    if now - last_wait_log >= 10.0:
                        _logging.getLogger(__name__).warning(
                            "Waiting for streaming video encoder queue for %s; save will block "
                            "rather than dropping a frame.",
                            video_key,
                        )
                        last_wait_log = now

        _original_streaming_finish_episode = _vu_mod.StreamingVideoEncoder.finish_episode

        def _mulligan_lossless_finish_episode(self):
            dropped = {key: int(count) for key, count in self._dropped_frames.items() if count}
            results = _original_streaming_finish_episode(self)
            if dropped:
                raise RuntimeError(
                    "Streaming video encoder dropped frame(s) before the Mulligan lossless patch "
                    f"could block: {dropped}. Refusing to save a dataset with fewer video "
                    "frames than LeRobot rows."
                )
            return results

        _vu_mod.StreamingVideoEncoder.feed_frame = _mulligan_lossless_feed_frame
        _vu_mod.StreamingVideoEncoder.finish_episode = _mulligan_lossless_finish_episode

        _original_writer_save_episode = _dw_mod.DatasetWriter.save_episode
        _original_writer_save_episode_video = _dw_mod.DatasetWriter._save_episode_video

        def _mulligan_save_episode(self, episode_data=None, parallel_encoding: bool = True) -> None:
            episode_buffer = episode_data if episode_data is not None else self.episode_buffer
            expected_length = int(episode_buffer["size"])
            normalize_video_spans = int(getattr(self, "_batch_encoding_size", 1)) <= 1
            had_previous = hasattr(self, "_mulligan_expected_episode_length")
            previous = getattr(self, "_mulligan_expected_episode_length", None)
            if normalize_video_spans:
                self._mulligan_expected_episode_length = expected_length
            try:
                return _original_writer_save_episode(
                    self,
                    episode_data=episode_data,
                    parallel_encoding=parallel_encoding,
                )
            finally:
                if normalize_video_spans:
                    if had_previous:
                        self._mulligan_expected_episode_length = previous
                    else:
                        delattr(self, "_mulligan_expected_episode_length")

        def _mulligan_save_episode_video(
            self, video_key: str, episode_index: int, temp_path=None
        ) -> dict:
            metadata = _original_writer_save_episode_video(
                self,
                video_key=video_key,
                episode_index=episode_index,
                temp_path=temp_path,
            )
            expected_length = getattr(self, "_mulligan_expected_episode_length", None)
            if expected_length is None:
                return metadata

            from_key = f"videos/{video_key}/from_timestamp"
            to_key = f"videos/{video_key}/to_timestamp"
            fps = int(self._meta.fps)
            from_ts = float(metadata[from_key])
            old_to_ts = float(metadata[to_key])
            old_span = round(old_to_ts * fps) - round(from_ts * fps)
            if old_span != expected_length:
                _logging.getLogger(__name__).warning(
                    "Normalizing %s episode %s video metadata span from %s frame(s) "
                    "to saved data length %s.",
                    video_key,
                    episode_index,
                    old_span,
                    expected_length,
                )

            metadata[to_key] = from_ts + expected_length / fps
            new_span = round(float(metadata[to_key]) * fps) - round(from_ts * fps)
            if new_span != expected_length:
                raise RuntimeError(
                    f"Failed to normalize {video_key} episode {episode_index} video metadata: "
                    f"span={new_span}, expected_length={expected_length}, fps={fps}, "
                    f"from_ts={from_ts}, to_ts={metadata[to_key]}"
                )
            return metadata

        _dw_mod.DatasetWriter.save_episode = _mulligan_save_episode
        _dw_mod.DatasetWriter._save_episode_video = _mulligan_save_episode_video

    # --------------------------------------------------------------------------- #
    # Patch 4: atomic write_json (meta/info.json + stats are rewritten IN PLACE on
    # every save_episode via a bare truncate-then-write; a crash inside that window
    # leaves a truncated JSON that bricks dataset recovery — reconcile reads info
    # first). tmp + fsync + rename keeps either the old or the new content visible.
    # Applied to lerobot.utils.io_utils AND the from-import binding in
    # lerobot.datasets.io_utils (write_info/write_stats call the bound name).
    # --------------------------------------------------------------------------- #
    import json as _json

    from lerobot.datasets import io_utils as _ds_io_mod
    from lerobot.utils import io_utils as _io_mod

    def _mulligan_atomic_write_json(data: dict, fpath) -> None:
        from pathlib import Path as _Path

        fpath = _Path(fpath)
        fpath.parent.mkdir(exist_ok=True, parents=True)
        tmp_path = fpath.with_name(f".{fpath.name}.tmp-{_os.getpid()}")
        with open(tmp_path, "w") as f:
            _json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp_path, fpath)

    _io_mod.write_json = _mulligan_atomic_write_json
    _ds_io_mod.write_json = _mulligan_atomic_write_json

    # --------------------------------------------------------------------------- #
    # Patch 5: same-filesystem atomic finish for concatenate_video_files. Upstream
    # writes its temp mp4 in the system TMPDIR and shutil.move's it onto the LIVE
    # tail video chunk — cross-filesystem that degrades to copy2+unlink, a
    # non-atomic window proportional to the full file size (cap 200MB) on a file a
    # reader may already reference. Redirect the visible output to a dot-prefixed
    # temp NEXT TO the destination (upstream's internal cross-fs move now lands on
    # the invisible dotfile), then os.replace atomically onto the real target.
    # reencode_video has the same pattern but only runs in the offline
    # lerobot-edit-dataset script — out of the collection path, left alone.
    # --------------------------------------------------------------------------- #
    _original_concatenate_video_files = _vu_mod.concatenate_video_files

    def _mulligan_atomic_concatenate_video_files(
        input_video_paths, output_video_path, *args, **kwargs
    ):
        from pathlib import Path as _Path

        output_video_path = _Path(output_video_path)
        # Honor the overwrite=False skip against the REAL output (the redirected temp
        # never exists, so the original's own existence check would never fire).
        overwrite = args[0] if args else kwargs.get("overwrite", True)
        if not overwrite and output_video_path.exists():
            return _original_concatenate_video_files(
                input_video_paths, output_video_path, *args, **kwargs
            )
        staged = output_video_path.with_name(f".tmp_concat_{_os.getpid()}_{output_video_path.name}")
        # The tail chunk being appended to is usually also an INPUT; upstream reads
        # inputs fully before its final move, and our rename happens strictly after
        # the original returns, so redirecting only the OUTPUT is safe.
        try:
            result = _original_concatenate_video_files(input_video_paths, staged, *args, **kwargs)
            _os.replace(staged, output_video_path)
        finally:
            if staged.exists():
                staged.unlink()
        return result

    _vu_mod.concatenate_video_files = _mulligan_atomic_concatenate_video_files
    _dw_mod.concatenate_video_files = _mulligan_atomic_concatenate_video_files

    # Mulligan LeRobot runtime patches (diffusion-policy fixes + Mulligan extensions) apply on any
    # lerobot that exposes lerobot.datasets, independent of the streaming API above.
    from mulligan.utils.lerobot_patches import (
        apply_all_patches as _apply_mulligan_lerobot_patches,
    )

    _apply_mulligan_lerobot_patches()

    # Mark the patched lerobot module LAST, so a partial application (an exception part-way
    # through) is not recorded as complete and the failure stays loud on the next import.
    setattr(_ld_mod, _MULLIGAN_PATCH_MARKER, True)

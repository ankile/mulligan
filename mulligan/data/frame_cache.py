"""Decode-once shared-RAM frame cache for real-world video datasets.

Real LeRobot datasets store camera streams as AV1 mp4s; random-access decode
costs ~10 ms CPU per frame, and a 100k-step DP run re-decodes every frame
~100+ times (192 frame-fetches/step at B=64 x 3 cams against a ~160k-frame
corpus). This module decodes every SERVED frame exactly once at startup — a
sequential sweep per video file through the policy preprocess
(crop + antialias-resize + quantize, :func:`preprocess_uint8_batch`) — into
shared uint8 tensors at policy resolution. DataLoader workers then serve
frames by row index from RAM (fork inherits the shared storage; no pickling,
no per-worker copies) and ``data_loading`` collapses to memcpy time.

Numerics: cached bytes are ``quantize(preprocess(native))``, i.e. one
<=1/255 quantization before augmentation — the accepted uint8-native error
class. A post-build self-check decodes random rows through the stock path and
requires byte equality.

The cache attaches to ``FastDatasetReader`` (``reader._frame_cache``); the
reader's ``_query_videos`` consults it before touching any video file.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from mulligan.data.fast_lerobot_reader import FastDatasetReader

DEFAULT_DECODE_BATCH = 128
# Fraction of the memory budget the cache may claim. The rest is headroom for
# the model, dataloader prefetch, decoder buffers, and page cache.
DEFAULT_MEM_FRACTION = 0.5


def preprocess_uint8_batch(
    frames_uint8: torch.Tensor,
    *,
    crop_box: tuple[int, int, int, int] | None,
    crop_reference_hw: tuple[int, int] | None,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    """(B,3,H,W) uint8 native -> (B,3,th,tw) uint8 policy-resolution frames.

    float/crop/resize via the SAME preprocess_chw_tensor_for_policy the
    training path uses, quantized with the SAME shared formula the replay
    buffer / uint8-native tier use.
    """
    from mulligan.real.policy.image_preprocess import (
        preprocess_chw_tensor_for_policy,
        quantize_float01_to_uint8,
    )

    out = preprocess_chw_tensor_for_policy(
        frames_uint8,
        target_hw=target_hw,
        crop_box=crop_box,
        crop_reference_hw=crop_reference_hw,
    )
    return quantize_float01_to_uint8(out)


def _cgroup_memory_limit_bytes() -> int | None:
    """Effective memory limit for THIS process: the minimum finite limit along its
    cgroup ancestry (batch-scheduler jobs live in nested slices; the root often reads 'max')."""
    limits: list[int] = []
    try:
        cgroup_lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in cgroup_lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, cpath = parts
        if hierarchy == "0" and controllers == "":  # cgroup v2 (unified or hybrid mount)
            mounts = [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/unified")]
            fname = "memory.max"
        elif "memory" in controllers.split(","):  # cgroup v1 memory controller
            mounts = [Path("/sys/fs/cgroup/memory"), Path("/sys/fs/cgroup/cpu,memory")]
            fname = "memory.limit_in_bytes"
        else:
            continue
        for mount in mounts:
            p = mount / cpath.lstrip("/")
            while True:
                f = p / fname
                try:
                    raw = f.read_text().strip()
                except OSError:
                    raw = ""
                if raw and raw != "max":
                    v = int(raw)
                    if v < 1 << 60:  # v1 unlimited sentinel
                        limits.append(v)
                if p == mount:
                    break
                p = p.parent
    return min(limits) if limits else None


def _shm_free_bytes() -> int | None:
    """Free space in /dev/shm — share_memory_() allocates there under the default
    file_descriptor sharing strategy, independent of the RAM budget."""
    try:
        st = os.statvfs("/dev/shm")
    except OSError:
        return None
    return st.f_bavail * st.f_frsize


def _meminfo_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _job_memory_budget_bytes(mem_fraction: float) -> tuple[int | None, str]:
    """RAM-side cache budget: the effective cgroup limit (scaled), else host
    MemAvailable (scaled). The /dev/shm cap is applied separately by the caller
    (it must bind even under the MULLIGAN_REAL_FRAME_CACHE_MAX_GB override)."""
    cg = _cgroup_memory_limit_bytes()
    if cg is not None:
        return int(cg * mem_fraction), f"cgroup {cg / 1e9:.0f} GB x {mem_fraction}"
    avail = _meminfo_available_bytes()
    if avail is not None:
        return int(avail * mem_fraction), f"MemAvailable {avail / 1e9:.0f} GB x {mem_fraction}"
    return None, "no limit found"


def _require_fast_reader(inner) -> FastDatasetReader:
    reader = inner.reader
    if not isinstance(reader, FastDatasetReader):
        raise TypeError(
            f"frame cache requires FastDatasetReader (got {type(reader).__name__}); "
            "enable the fast reader before building the cache"
        )
    if inner.meta.depth_keys:
        raise ValueError(
            f"frame cache does not support depth video keys {inner.meta.depth_keys} "
            "(frames are cached with the RGB preprocess); run with --no-decoded-frame-cache"
        )
    return reader


def frame_cache_size_bytes(sub_datasets, target_hw: tuple[int, int]) -> int:
    th, tw = target_hw
    total = 0
    for sub in sub_datasets:
        _require_fast_reader(sub)
        n_rows = len(sub.reader._tensor_columns["timestamp"])
        total += n_rows * len(sub.meta.video_keys) * 3 * th * tw
    return total


def _served_episode_ranges(reader: FastDatasetReader) -> list[tuple[int, int, int]]:
    """Contiguous served-row ranges: (episode_index, row_from, row_to).

    Refuses an episode split across multiple runs: the attach-side dict keyed by
    episode index would silently keep only the last run and serve wrong frames.
    """
    ep_col = reader._tensor_columns["episode_index"]
    ranges: list[tuple[int, int, int]] = []
    n = len(ep_col)
    if n == 0:
        return ranges
    seen: set[int] = set()
    cur = int(ep_col[0])
    start = 0

    def _close(ep: int, lo: int, hi: int) -> None:
        if ep in seen:
            raise ValueError(
                f"episode {ep} appears in multiple non-contiguous served-row runs; the "
                "frame cache requires contiguous episodes (refusing a wrong-frame cache)"
            )
        seen.add(ep)
        ranges.append((ep, lo, hi))

    for i in range(1, n):
        e = int(ep_col[i])
        if e != cur:
            _close(cur, start, i)
            cur = e
            start = i
    _close(cur, start, n)
    return ranges


def build_frame_cache_for_dataset(
    inner,
    *,
    crop_feature_map: dict[str, tuple[int, int, int, int]],
    crop_reference_hw: tuple[int, int] | None,
    target_hw: tuple[int, int],
    threads: int = 16,
    self_check_rows: int = 8,
) -> dict[str, torch.Tensor]:
    """Decode every served frame of ``inner`` once into shared uint8 tensors.

    Returns {video_key: uint8 tensor (n_served_rows, 3, th, tw)}. Raises on any
    inconsistency (missing crop box, frame-index mismatch, failed self-check) —
    a silently wrong cache would corrupt every downstream experiment.
    """
    from torchcodec.decoders import VideoDecoder

    from lerobot.datasets.dataset_reader import decode_video_frames

    reader = _require_fast_reader(inner)
    meta = inner.meta
    for vid_key in meta.video_keys:
        if vid_key not in crop_feature_map:
            raise ValueError(
                f"frame cache requires a crop box for every video key; missing {vid_key!r} "
                f"(have {sorted(crop_feature_map)})"
            )

    ts_col = reader._tensor_columns["timestamp"]
    n_rows = len(ts_col)
    th, tw = target_hw
    caches = {
        vid_key: torch.empty((n_rows, 3, th, tw), dtype=torch.uint8).share_memory_()
        for vid_key in meta.video_keys
    }

    # Group served rows by (video_key, file): one sequential sweep per file.
    # jobs[(vid_key, path)] = list of (served_row, shifted_ts)
    jobs: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for ep_idx, row_from, row_to in _served_episode_ranges(reader):
        ep = meta.episodes[ep_idx]
        for vid_key in meta.video_keys:
            from_ts = float(ep[f"videos/{vid_key}/from_timestamp"])
            path = str(inner.root / meta.get_video_file_path(ep_idx, vid_key))
            pairs = jobs.setdefault((vid_key, path), [])
            for row in range(row_from, row_to):
                pairs.append((row, from_ts + float(ts_col[row])))

    def _sweep(vid_key: str, path: str, pairs: list[tuple[int, float]]) -> int:
        dec = VideoDecoder(path, device="cpu")
        avg_fps = dec.metadata.average_fps
        n_frames = dec.metadata.num_frames
        # Same index resolution as lerobot's torchcodec path (video_utils.py):
        # frame index = round(ts * average_fps).
        indexed = sorted((round(ts * avg_fps), row) for row, ts in pairs)
        for fidx, _row in indexed[:1] + indexed[-1:]:
            if not 0 <= fidx < n_frames:
                raise ValueError(
                    f"frame index {fidx} out of range [0, {n_frames}) for {path} — "
                    "timestamp/fps mapping is broken, refusing to build a wrong cache"
                )
        cache = caches[vid_key]
        box = crop_feature_map[vid_key]
        for start in range(0, len(indexed), DEFAULT_DECODE_BATCH):
            chunk = indexed[start : start + DEFAULT_DECODE_BATCH]
            batch = dec.get_frames_at(indices=[f for f, _ in chunk]).data  # (B,3,H,W) uint8
            views = preprocess_uint8_batch(
                batch, crop_box=box, crop_reference_hw=crop_reference_hw, target_hw=target_hw
            )
            cache[torch.tensor([r for _, r in chunk], dtype=torch.long)] = views
        return len(indexed)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = {
            pool.submit(_sweep, vid_key, path, pairs): (vid_key, path)
            for (vid_key, path), pairs in jobs.items()
        }
        for fut in futures:
            fut.result()  # re-raise loudly
    build_s = time.perf_counter() - t0

    # Self-check: random served rows must be byte-identical to the stock
    # decode -> preprocess -> quantize path.
    if self_check_rows > 0 and n_rows > 0:
        g = torch.Generator().manual_seed(0)
        check_rows = torch.randperm(n_rows, generator=g)[:self_check_rows].tolist()
        ranges = _served_episode_ranges(reader)
        for row in check_rows:
            ep_idx = next(e for e, f, t in ranges if f <= row < t)
            ep = meta.episodes[ep_idx]
            for vid_key in meta.video_keys:
                from_ts = float(ep[f"videos/{vid_key}/from_timestamp"])
                path = inner.root / meta.get_video_file_path(ep_idx, vid_key)
                frames = decode_video_frames(
                    path,
                    [from_ts + float(ts_col[row])],
                    reader._tolerance_s,
                    reader._video_backend,
                    return_uint8=True,
                )
                ref = preprocess_uint8_batch(
                    frames,
                    crop_box=crop_feature_map[vid_key],
                    crop_reference_hw=crop_reference_hw,
                    target_hw=target_hw,
                )[0]
                if not torch.equal(ref, caches[vid_key][row]):
                    raise ValueError(
                        f"frame-cache self-check failed at row {row} / {vid_key!r}: cached "
                        "bytes differ from the stock decode path — refusing to train on a "
                        "wrong cache"
                    )

    total_gb = sum(c.numel() for c in caches.values()) / 1e9
    print(
        f"  Frame cache built: {n_rows} rows x {len(caches)} cams "
        f"({total_gb:.1f} GB shared uint8) in {build_s:.0f}s"
    )
    return caches


def attach_frame_cache(inner, caches: dict[str, torch.Tensor]) -> None:
    """Point the dataset's FastDatasetReader at the built cache (view-attach pattern)."""
    reader = inner.reader
    if not isinstance(reader, FastDatasetReader):
        raise TypeError(f"frame cache requires FastDatasetReader, got {type(reader).__name__}")
    if set(caches) != set(reader._meta.video_keys):
        raise ValueError(
            f"cache keys {sorted(caches)} != dataset video keys {sorted(reader._meta.video_keys)}"
        )
    reader._frame_cache = caches
    reader._frame_cache_ranges = {
        ep_idx: (row_from, row_to) for ep_idx, row_from, row_to in _served_episode_ranges(reader)
    }


def build_and_attach_frame_caches(
    sub_datasets,
    *,
    crop_feature_map: dict[str, tuple[int, int, int, int]],
    crop_reference_hw: tuple[int, int] | None,
    target_hw: tuple[int, int],
    threads: int = 16,
    mem_fraction: float = DEFAULT_MEM_FRACTION,
    reserved_bytes: int = 0,
) -> bool:
    """Build + attach caches for every sub-dataset, gated on the job memory budget.

    ``reserved_bytes``: cache bytes already resident from a previous call in this
    process (e.g. the train cache when this call builds the eval cache) — charged
    against the RAM budget so the SUM stays under the limit.

    Returns True when caches are attached; False (with a loud WARN) when the
    corpus does not fit the budget — the decode path then runs unchanged, so
    this fallback is a pure perf knob (the experiment is identical either way).
    """
    inners = [getattr(sub, "_inner", sub) for sub in sub_datasets]
    need = frame_cache_size_bytes(inners, target_hw)
    # RAM budget: env override or effective cgroup/MemAvailable — minus caches
    # ALREADY resident from a previous call this process (reserved_bytes), so
    # train+eval cannot each pass a per-call gate while their sum blows the limit.
    env_gb = os.environ.get("MULLIGAN_REAL_FRAME_CACHE_MAX_GB", "")
    if env_gb:
        allowed, budget_src = float(env_gb) * 1e9, "MULLIGAN_REAL_FRAME_CACHE_MAX_GB"
    else:
        allowed, budget_src = _job_memory_budget_bytes(mem_fraction)
    if allowed is not None:
        allowed -= reserved_bytes
    # /dev/shm cap binds ALWAYS (also under the env override): share_memory_
    # allocates there. Re-sampled free space already reflects resident caches.
    shm = _shm_free_bytes()
    if shm is not None and (allowed is None or shm * 0.9 < allowed):
        allowed, budget_src = int(shm * 0.9), f"/dev/shm free {shm / 1e9:.0f} GB x 0.9"
    if allowed is not None and need > allowed:
        print(
            f"WARNING: --decoded-frame-cache needs {need / 1e9:.1f} GB but the budget is "
            f"{allowed / 1e9:.1f} GB ({budget_src}, {reserved_bytes / 1e9:.1f} GB already "
            "cached); falling back to per-sample video decode. Raise the job memory or "
            "shrink the datamix to enable the cache."
        )
        return False
    print(f"  Frame cache budget: {need / 1e9:.1f} GB needed, limit {budget_src}")
    # All-or-nothing: build every dataset's cache BEFORE attaching any. A partial
    # attach after a mid-build allocation failure (e.g. a co-resident job won a
    # /dev/shm race) would leave some sub-datasets serving raw native frames in
    # the native tier — broken geometry, not just slower.
    built: list[dict[str, torch.Tensor]] = []
    try:
        for inner in inners:
            built.append(
                build_frame_cache_for_dataset(
                    inner,
                    crop_feature_map=crop_feature_map,
                    crop_reference_hw=crop_reference_hw,
                    target_hw=target_hw,
                    threads=threads,
                )
            )
    except (RuntimeError, OSError) as exc:
        # Correctness failures raise ValueError and propagate; only
        # allocation/build-environment failures downgrade to the decode path.
        print(
            f"WARNING: --decoded-frame-cache allocation/build failed ({exc}); falling "
            "back to per-sample video decode (likely a /dev/shm race with a co-resident "
            "job — no cache attached, training proceeds on the decode path)."
        )
        del built
        return False
    for inner, caches in zip(inners, built, strict=True):
        attach_frame_cache(inner, caches)
        _serve_path_self_check(
            inner,
            crop_feature_map=crop_feature_map,
            crop_reference_hw=crop_reference_hw,
            target_hw=target_hw,
        )
    return True


def _serve_path_self_check(
    inner,
    *,
    crop_feature_map: dict[str, tuple[int, int, int, int]],
    crop_reference_hw: tuple[int, int] | None,
    target_hw: tuple[int, int],
    n_random: int = 6,
) -> None:
    """Exercise the ACTUAL serve path (reader._query_frame_cache row resolution)
    against the stock decode for random rows PLUS every episode's boundary rows
    of the first/last episode — the places an fps/rounding mismatch would hide."""
    from lerobot.datasets.dataset_reader import decode_video_frames

    reader = inner.reader
    ranges = _served_episode_ranges(reader)
    if not ranges:
        return
    ts_col = reader._tensor_columns["timestamp"]
    # Boundary rows of up to 8 evenly-spaced episodes (middle-of-file boundaries
    # in chunked multi-episode mp4s included), plus random rows.
    rows: set[int] = set()
    for ep_range in ranges[:: max(1, len(ranges) // 8)]:
        rows.update((ep_range[1], ep_range[2] - 1))
    rows.update((ranges[-1][1], ranges[-1][2] - 1))
    g = torch.Generator().manual_seed(1)
    n_rows = len(ts_col)
    rows.update(torch.randperm(n_rows, generator=g)[:n_random].tolist())
    prev_uint8 = reader._return_uint8
    reader._return_uint8 = True
    try:
        for row in sorted(rows):
            ep_idx = next(e for e, f, t in ranges if f <= row < t)
            ts = float(ts_col[row])
            served = reader._query_frame_cache(
                {vid_key: [ts] for vid_key in reader._meta.video_keys}, ep_idx
            )
            ep = reader._meta.episodes[ep_idx]
            for vid_key, got in served.items():
                from_ts = float(ep[f"videos/{vid_key}/from_timestamp"])
                frames = decode_video_frames(
                    inner.root / reader._meta.get_video_file_path(ep_idx, vid_key),
                    [from_ts + ts],
                    reader._tolerance_s,
                    reader._video_backend,
                    return_uint8=True,
                )
                ref = preprocess_uint8_batch(
                    frames,
                    crop_box=crop_feature_map[vid_key],
                    crop_reference_hw=crop_reference_hw,
                    target_hw=target_hw,
                )[0]
                if not torch.equal(ref, got):
                    raise ValueError(
                        f"frame-cache SERVE-path self-check failed at row {row} / "
                        f"{vid_key!r} (episode {ep_idx}): served bytes differ from the "
                        "stock decode — refusing to train on a wrong cache"
                    )
    finally:
        reader._return_uint8 = prev_uint8

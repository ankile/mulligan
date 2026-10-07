"""Fig. 3: per-task rollout image sequences (one row per real-world task).

Each row is one held-out evaluation episode of the final-round HiL-IDQL+Mulligan policy
from the side_1 camera: the median-length success of that arm on the task's final-round
held-out block. Episodes, crops, and frame picks are pinned in
``paper/data/real/task_sequences_provenance.json``; the videos are read from the
public ``mulligan/*`` release datasets at pinned revisions. Frames are not evenly
spaced: each is hand-picked at a rung of the task's stage ladder after reviewing
the full contact sheet, and the rung label is baked under each frame.

    python -m paper.fig_tasks                    # strips into FIGS_DIR
    python -m paper.fig_tasks --contact DIR      # numbered contact sheets (every 3rd frame)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from mulligan.plotting import paper
from mulligan.release.download import pinned_revision

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_SOURCES = ROOT / "paper/data/real/task_sequences_provenance.json"

# Contact sheets: workspace region of the 640x480 side_1 view.
CONTACT_CROP = (90, 0, 590, 330)

# Print geometry: the strip spans the full text width (5.5 in); 6 frames, no gaps.
TEXTWIDTH_IN = paper.DOC_TEXTWIDTH_IN["main"]
N_FRAMES = 6
GAP_IN = 0.0
LABEL_IN = 0.13  # rung label band under each frame
LABEL_PT = 6.5
DPI = 500  # output raster density (text baked at print size stays crisp)


def load_provenance() -> dict:
    """Pinned episodes, strip crops, enhancement, and frame picks per task."""
    return json.loads(SEQUENCE_SOURCES.read_text())


def episode_clip(task: dict, camera: str) -> tuple[Path, float, float, int]:
    """Download the one video file holding ``task``'s episode; return its span."""
    episodes = pd.read_parquet(
        hf_hub_download(
            task["repo"],
            "meta/episodes/chunk-000/file-000.parquet",
            repo_type="dataset",
            revision=pinned_revision(task["repo"]),
        )
    )
    ep = episodes.loc[episodes["episode_index"] == task["episode_index"]]
    if len(ep) != 1:
        raise RuntimeError(
            f"{task['repo']}: episode {task['episode_index']} matched {len(ep)} rows"
        )
    ep = ep.iloc[0]
    chunk = int(ep[f"videos/{camera}/chunk_index"])
    file_index = int(ep[f"videos/{camera}/file_index"])
    start, end = (
        float(ep[f"videos/{camera}/from_timestamp"]),
        float(ep[f"videos/{camera}/to_timestamp"]),
    )
    path = hf_hub_download(
        task["repo"],
        f"videos/{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4",
        repo_type="dataset",
        revision=pinned_revision(task["repo"]),
    )
    return Path(path), start, end, int(ep["length"])


def decode_episode(
    task: dict, camera: str, crop: tuple[int, int, int, int] | None
) -> list[Image.Image]:
    """Decoded frames ``[0, num_steps)`` of the episode, cropped to ``crop``."""
    path, start, end, length = episode_clip(task, camera)
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start - 1e-6:
                continue
            if t >= end - 1e-6:
                break
            img = frame.to_image()
            frames.append(img.crop(crop) if crop is not None else img)
    if len(frames) != length:
        raise RuntimeError(f"{path}: decoded {len(frames)} frames, episode length {length}")
    if length != task["length"]:
        raise RuntimeError(f"{task['repo']}: episode length {length}, pinned {task['length']}")
    if task["num_steps"] > length:
        raise RuntimeError(f"num_steps {task['num_steps']} > recorded length {length}")
    return frames[: task["num_steps"]]


def _font(size_px: int) -> ImageFont.FreeTypeFont:
    import matplotlib.font_manager as fm

    return ImageFont.truetype(fm.findfont("DejaVu Sans"), size_px)


def contact_sheet(
    frames: list[Image.Image], out: Path, *, fps: float, step: int = 3, cols: int = 8
) -> Path:
    picks = list(range(0, len(frames), step))
    if picks[-1] != len(frames) - 1:
        picks.append(len(frames) - 1)
    w = 250
    h = round(w * frames[0].size[1] / frames[0].size[0])
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(14)
    for i, idx in enumerate(picks):
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        sheet.paste(frames[idx].resize((w, h), Image.LANCZOS), (x, y))
        draw.text((x + 4, y + h + 1), f"{idx}  ({idx / fps:.1f}s)", fill="black", font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def enhance(img: Image.Image, factors: dict[str, float]) -> Image.Image:
    """Mild lift: the side camera renders the black breadboard flat and grey."""
    img = ImageEnhance.Color(img).enhance(factors["color"])
    img = ImageEnhance.Brightness(img).enhance(factors["brightness"])
    return ImageEnhance.Contrast(img).enhance(factors["contrast"])


def strip(frames: list[Image.Image], labels: list[str]) -> Image.Image:
    """Full-textwidth strip at DPI: N frames, white gaps, rung label under each."""
    n = len(frames)
    gap = round(GAP_IN * DPI)
    frame_w = round((TEXTWIDTH_IN * DPI - gap * (n - 1)) / n)
    src_w, src_h = frames[0].size
    frame_h = round(frame_w * src_h / src_w)
    label_h = round(LABEL_IN * DPI)
    out = Image.new("RGB", (frame_w * n + gap * (n - 1), frame_h + label_h), "white")
    draw = ImageDraw.Draw(out)
    font = _font(round(LABEL_PT / 72 * DPI))
    for i, (frame, label) in enumerate(zip(frames, labels)):
        x = i * (frame_w + gap)
        out.paste(frame.resize((frame_w, frame_h), Image.LANCZOS), (x, 0))
        tw = draw.textlength(label, font=font)
        draw.text(
            (x + (frame_w - tw) / 2, frame_h + label_h * 0.18), label, fill="black", font=font
        )
    return out


def _pick_indices(name: str, task: dict) -> list[int]:
    indices = [p["frame"] for p in task["picks"]]
    if len(indices) != N_FRAMES:
        raise ValueError(f"{name}: {len(indices)} picks, expected {N_FRAMES}")
    if indices != sorted(indices) or indices[-1] >= task["num_steps"]:
        raise ValueError(f"{name}: picks must be increasing and < {task['num_steps']}: {indices}")
    return indices


def build_task_sequences() -> list[Path]:
    """Render ``FIGS_DIR/task_sequence_{marker,nut,cable}.jpg``."""
    prov = load_provenance()
    outputs = []
    for name, task in prov["tasks"].items():
        indices = _pick_indices(name, task)
        frames = decode_episode(task, prov["camera"], tuple(prov["crops"][name]))
        out = paper.FIGS_DIR / f"task_sequence_{name}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        strip(
            [enhance(frames[i], prov["enhance"]) for i in indices],
            [p["label"] for p in task["picks"]],
        ).save(out, quality=90, optimize=True)
        print(f"wrote {out} ({out.stat().st_size // 1024} KB) frames {indices}")
        outputs.append(out)
    return outputs


def build_contact_sheets(out_dir: Path, camera: str | None = None) -> list[Path]:
    """Numbered contact sheets (every 3rd frame) for reviewing the picks."""
    prov = load_provenance()
    camera = camera or prov["camera"]
    tag = camera.rsplit(".", 1)[-1]
    outputs = []
    for name, task in prov["tasks"].items():
        frames = decode_episode(task, camera, CONTACT_CROP)
        out = contact_sheet(frames, out_dir / f"{name}_{tag}_contact.png", fps=prov["fps"])
        print(f"{name}: {len(frames)} frames -> {out}")
        outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contact", type=Path, default=None, metavar="DIR", help="only write contact sheets to DIR"
    )
    parser.add_argument("--camera", default=None, help="contact sheets only: camera key")
    args = parser.parse_args()
    if args.contact is not None:
        build_contact_sheets(args.contact, args.camera)
    else:
        build_task_sequences()


if __name__ == "__main__":
    main()

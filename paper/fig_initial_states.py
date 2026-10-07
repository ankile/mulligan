"""App. C: typical versus hardest initial states of Insert Marker and Thread Nut.

Starts come from the public round-0 teleoperation recordings
(``mulligan/real-{marker,square}-d2-c00-teleop-mixed``, 250 demonstrations each), read at
their pinned revisions. Selection rules, recomputed on every build and checked against
``paper/data/real/initial_states_provenance.json``:

* hardest: the start of the task's longest demonstration. On both tasks it lies at the
  near-minimum object x (closest to the robot base) with the object pointing toward
  the robot (yaw within 30 deg of 180).
* typical: among demonstrations within 2 steps of the task's median length, the one
  whose object yaw is farthest from 180 deg (pointing away from the robot).

Start frames use the side_2 camera, which shows the robot base at the right; the
operator's seat in its top-right corner is blurred. The Thread Nut strip shows the
non-prehensile reorientation in the hardest demonstration from the side_1 camera.

    python -m paper.figures --only initial_states
"""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image, ImageDraw, ImageFilter

from mulligan.plotting import paper
from mulligan.release.download import pinned_revision
from paper.fig_tasks import enhance

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "paper/data/real/initial_states_provenance.json"
DPI = 500
TEXTWIDTH_IN = paper.DOC_TEXTWIDTH_IN["main"]


def load_provenance() -> dict:
    """Pinned repos, selection results, crops, blur box and strip picks."""
    return json.loads(PROVENANCE.read_text())


def _download(repo: str, filename: str) -> str:
    return hf_hub_download(repo, filename, repo_type="dataset", revision=pinned_revision(repo))


def episodes(task: dict) -> pd.DataFrame:
    """Per-episode start pose (x, y, yaw of the object) and length of the recording."""
    files = sorted(
        f
        for f in list_repo_files(
            task["repo"], repo_type="dataset", revision=pinned_revision(task["repo"])
        )
        if f.startswith("data/") and f.endswith(".parquet")
    )
    data = pd.concat(pd.read_parquet(_download(task["repo"], f)) for f in files)
    o = task["obj"]
    ep = data.groupby("episode_index").agg(
        x=(f"{o}_x", "first"),
        y=(f"{o}_y", "first"),
        yaw=(f"{o}_yaw", "first"),
        length=("frame_index", "size"),
    )
    ep["yaw_deg"] = np.degrees(ep["yaw"]) % 360
    ep["from_robot_deg"] = np.abs((ep["yaw_deg"] - 180 + 180) % 360 - 180)
    if len(ep) != 250:
        raise RuntimeError(f"{task['repo']}: {len(ep)} episodes, expected 250")
    return ep


def select(ep: pd.DataFrame) -> dict[str, int]:
    hardest = int(ep["length"].idxmax())
    if ep.loc[hardest, "from_robot_deg"] >= 30:
        raise RuntimeError("the longest demonstration does not point at the robot")
    near_median = ep[np.abs(ep["length"] - ep["length"].median()) <= 2]
    typical = int(near_median["from_robot_deg"].idxmax())
    return {"typical": typical, "hardest": hardest}


def check_picks(name: str, task: dict, ep: pd.DataFrame) -> dict[str, int]:
    """The recomputed selection must equal the pinned one."""
    picks = select(ep)
    if float(ep["length"].median()) != task["median_length"]:
        raise RuntimeError(
            f"{name}: median length {ep['length'].median()}, pinned {task['median_length']}"
        )
    for kind, idx in picks.items():
        pinned = task["picks"][kind]
        row = ep.loc[idx]
        found = dict(
            episode_index=idx,
            x_m=round(float(row["x"]), 4),
            y_m=round(float(row["y"]), 4),
            yaw_deg=round(float(row["yaw_deg"]), 1),
            length_steps=int(row["length"]),
        )
        if any(found[k] != pinned[k] for k in found):
            raise RuntimeError(f"{name} {kind}: selected {found}, pinned {pinned}")
    return picks


def frame_at(repo: str, episode: int, camera: str, t: float) -> Image.Image:
    """The first decoded frame at or after ``t`` seconds into ``episode``."""
    meta = pd.read_parquet(_download(repo, "meta/episodes/chunk-000/file-000.parquet"))
    row = meta.set_index("episode_index").loc[episode]
    k = f"videos/{camera}"
    path = _download(
        repo,
        f"{k}/chunk-{int(row[k + '/chunk_index']):03d}/file-{int(row[k + '/file_index']):03d}.mp4",
    )
    target = float(row[k + "/from_timestamp"]) + t
    with av.open(path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if float(frame.pts * stream.time_base) >= target - 1e-6:
                return frame.to_image()
    raise RuntimeError(f"{path}: no frame at {target}")


def blur_operator(img: Image.Image, box: list[int]) -> Image.Image:
    """Blur the operator with a feathered mask so the edit does not draw the eye."""
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle(tuple(box), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(25))
    return Image.composite(img.filter(ImageFilter.GaussianBlur(18)), img, mask)


def save(img: Image.Image, width_in: float, out: Path) -> Path:
    w = round(width_in * DPI)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.resize((w, round(w * img.size[1] / img.size[0])), Image.LANCZOS).save(
        out, quality=90, optimize=True
    )
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


def build_initial_states() -> list[Path]:
    """Render ``FIGS_DIR/initial_state_{marker,nut}_{typical,hardest}.jpg`` and the Nut strip."""
    prov = load_provenance()
    outputs = []
    for name, task in prov["tasks"].items():
        picks = check_picks(name, task, episodes(task))
        for kind, idx in picks.items():
            frame = frame_at(task["repo"], idx, prov["start_camera"], 0.0)
            img = enhance(
                blur_operator(frame, prov["operator_box"]).crop(tuple(prov["start_crop"])),
                prov["enhance"],
            )
            outputs.append(
                save(img, TEXTWIDTH_IN / 2, paper.FIGS_DIR / f"initial_state_{name}_{kind}.jpg")
            )
    nut = prov["tasks"]["nut"]
    hardest = nut["picks"]["hardest"]["episode_index"]
    frames = [
        enhance(
            frame_at(nut["repo"], hardest, prov["strip_camera"], p["time_s"]).crop(
                tuple(prov["strip_crop"])
            ),
            prov["enhance"],
        )
        for p in prov["nut_reorientation"]
    ]
    w, h = frames[0].size
    strip = Image.new("RGB", (w * len(frames), h), "white")
    for i, f in enumerate(frames):
        strip.paste(f, (i * w, 0))
    outputs.append(
        save(strip, TEXTWIDTH_IN, paper.FIGS_DIR / "initial_state_nut_reorientation.jpg")
    )
    return outputs


if __name__ == "__main__":
    build_initial_states()

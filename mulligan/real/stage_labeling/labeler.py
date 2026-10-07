"""Task-agnostic Gemini video-labeling engine.

The reusable core of the labeler: retry-with-backoff, Files-API upload
caching, schema-validated generation, per-(episode, sample) checkpoint/resume,
and the self-consistency threadpool — all driven by a
:class:`~mulligan.real.stage_specs.tasks.StageLabelTaskSpec` and a
:class:`LabelerConfig`. The task content (system prompt, response schema, sensor
caps) comes from the spec; this module supplies only the mechanics.

``google.genai`` is imported lazily (inside the functions that need it) because
it is uninstallable in several environments; building items, prompts, schema, and
sensor caps never requires it.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import pandas as pd

from mulligan.real.stage_specs.schema import required_keys
from mulligan.real.stage_labeling.genai_schema import to_genai_schema
from mulligan.real.stage_specs.sensor_constraints import apply_sensor_constraints

if TYPE_CHECKING:
    from google.genai import types

    from mulligan.real.stage_specs.tasks import StageLabelTaskSpec

RETRIABLE_CODES = {429, 500, 502, 503, 504}
RETRIABLE_API_INITIAL_DELAY_S = 20.0
RETRIABLE_API_BACKOFF_FACTOR = 1.8
RETRIABLE_API_MAX_BASE_DELAY_S = 300.0
HARD_QUOTA_ERROR_MARKERS = (
    "prepayment credits are depleted",
    "billing account is disabled",
    "billing account has been closed",
)
TRANSIENT_RESPONSE_DELAY_S = 3.0
GEMINI_ROUTE_ENV = "MULLIGAN_GEMINI_ROUTE"
GEMINI_ROUTES = frozenset({"auto", "developer", "vertex"})

_FILE_UPLOAD_CACHE: dict[str, str] = {}
_FILE_UPLOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class LabelerConfig:
    """Per-run labeling knobs."""

    run_name: str
    output_dir: Path
    build_dir: Path = Path("/tmp/mulligan_stage_labeling_assets")
    # Focused-cascade few-shot media can come from a fixed reviewed calibration
    # dataset rather than the target dataset.  Keeping this path separate avoids
    # interpreting the same integer episode id in two different repositories.
    few_shot_build_dir: Path | None = None
    model: str = "gemini-3.5-flash"
    prompt_variant: str | None = None  # None -> spec.default_prompt_variant
    samples: int = 3
    temperature: float = 0.3
    # Gemini 3 thinks by default and the (hidden) thinking tokens draw from this
    # same budget; 8192 occasionally truncated to an empty MAX_TOKENS response.
    max_output_tokens: int = 32768
    thinking_level: str | None = None  # None=SDK default; "low"/"high" bounds Gemini-3 thinking
    # Tokens-per-frame budget for media. Gemini downsamples video frames hard by
    # default (small objects like the nut are lost); "low"/"medium"/"high" trade
    # context for per-frame detail. Our clips are short, so "high" is affordable.
    media_resolution: str | None = None  # None=SDK default; "low"|"medium"|"high"
    video_fps: float = 5.0  # the fps METADATA sent to the VLM, not the dataset fps
    input_mode: str = "dual"  # "dual" (separate videos + stills) | "combo" (hstack)
    video_transport: str = "inline"  # "inline" | "file" (Files API)
    send_final_crops: bool = False
    # The zoomed grasp-WINDOW montage (a short approach strip from the wrist view).
    # Separate from send_final_crops: it is the decisive evidence for the S0<->S1
    # line on the failed-grasp episodes, where the final-frame crops add nothing.
    send_grasp_crops: bool = False
    # In-context-learning exemplars: (episode_index, label, why) triples prepended
    # as labeled REFERENCE clips before the target, so the model can classify by
    # comparison rather than by abstract rule. Each exemplar's combo video is shown.
    few_shot: tuple[tuple[int, str, str], ...] = ()
    reference_image: Path | None = None
    workers: int = 4
    resume: bool = False
    episodes: tuple[int, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Generation + retry (lazy genai).
# --------------------------------------------------------------------------- #


def jittered_retry_delay_s(
    retry_index: int,
    *,
    initial_delay_s: float = RETRIABLE_API_INITIAL_DELAY_S,
    backoff_factor: float = RETRIABLE_API_BACKOFF_FACTOR,
    max_base_delay_s: float = RETRIABLE_API_MAX_BASE_DELAY_S,
    jitter_unit: float | None = None,
) -> float:
    """Return the shared one-based exponential delay with multiplicative jitter.

    The exponential base is capped before applying the ``0.7..1.3``
    jitter band. ``jitter_unit`` is injectable only to make policy tests exact;
    production callers use a fresh process-local random sample.
    """

    if isinstance(retry_index, bool) or not isinstance(retry_index, int) or retry_index < 1:
        raise ValueError("retry_index must be a positive integer")
    policy_values = (initial_delay_s, backoff_factor, max_base_delay_s)
    if any(not math.isfinite(value) or value < 0 for value in policy_values):
        raise ValueError("retry delay policy values must be finite and non-negative")
    if backoff_factor < 1:
        raise ValueError("backoff_factor must be >= 1")
    sampled_unit = random.random() if jitter_unit is None else jitter_unit
    if not math.isfinite(sampled_unit) or not 0.0 <= sampled_unit <= 1.0:
        raise ValueError("jitter_unit must be finite and within [0, 1]")
    base_delay_s = min(
        initial_delay_s * (backoff_factor ** (retry_index - 1)),
        max_base_delay_s,
    )
    return base_delay_s * (0.7 + 0.6 * sampled_unit)


def is_hard_quota_error(error: object) -> bool:
    """Return whether a provider error requires an account change, not a retry."""

    error_text = str(error).lower()
    return any(marker in error_text for marker in HARD_QUOTA_ERROR_MARKERS)


def make_client(*, route: Literal["auto", "developer", "vertex"] | None = None) -> Any:
    """Construct a Gemini client through an explicit or environment-selected route.

    ``developer`` uses the Gemini Developer API with ``GEMINI_API_KEY`` and
    ``v1beta``. It is explicit about both ``api_key`` and ``vertexai=False``, so
    a stale Vertex environment cannot silently redirect a paid API-key run.

    ``vertex`` uses Vertex AI + ADC with the project/location already selected by
    ``GOOGLE_CLOUD_PROJECT`` and ``GOOGLE_CLOUD_LOCATION``. ``auto`` follows the
    SDK's default behavior, deriving the route from
    ``GOOGLE_GENAI_USE_VERTEXAI``.

    When ``route`` is omitted, ``MULLIGAN_GEMINI_ROUTE`` selects the route and defaults
    to ``developer``. ``auto`` remains available only as an explicit compatibility
    route. Invalid or incomplete routes fail before any request. The Developer API
    stays on ``v1beta`` because its ``v1`` endpoint rejects structured-output fields
    used by this labeler.
    """
    from google import genai
    from google.genai.types import HttpOptions

    selected = (route or os.environ.get(GEMINI_ROUTE_ENV, "developer")).strip().lower()
    if selected not in GEMINI_ROUTES:
        raise ValueError(
            f"invalid {GEMINI_ROUTE_ENV}={selected!r}; expected one of {sorted(GEMINI_ROUTES)}"
        )
    if selected == "developer":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Gemini Developer API route requires GEMINI_API_KEY; put it in a private "
                "environment file, never in source or experiment provenance"
            )
        return genai.Client(
            vertexai=False,
            api_key=api_key,
            http_options=HttpOptions(api_version="v1beta"),
        )
    if selected == "vertex":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        if not project or not location:
            raise RuntimeError(
                "Vertex route requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION"
            )
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true"
    api_version = "v1" if use_vertex else "v1beta"
    return genai.Client(http_options=HttpOptions(api_version=api_version))


def client_route(client: Any) -> dict[str, Any]:
    """Return auditable routing metadata without exposing credentials."""
    api = client._api_client
    return {
        "vertexai": bool(client.vertexai),
        "project": api.project,
        "location": api.location,
        "api_key_used": bool(api.api_key),
        "api_version": api._http_options.api_version,
    }


# Global PROACTIVE throttle: keep at least ``MULLIGAN_GEMINI_MIN_INTERVAL_S`` seconds
# between the START of consecutive Gemini calls (across all threads), so a tight
# quota is never tripped in the first place rather than absorbed by 429 backoff.
# Off by default (interval 0); opt in via the env var for slow overnight trickles.
_THROTTLE_LOCK = threading.Lock()
_LAST_CALL_MONO = [0.0]


def _proactive_throttle() -> None:
    interval = float(os.environ.get("MULLIGAN_GEMINI_MIN_INTERVAL_S", "0") or 0)
    if interval <= 0:
        return
    with _THROTTLE_LOCK:
        wait = _LAST_CALL_MONO[0] + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_MONO[0] = time.monotonic()


def _response_usage(response: Any) -> dict[str, Any] | None:
    # ``usage_metadata`` is part of the SDK response contract. Access it
    # directly so an incompatible SDK object fails loudly instead of silently
    # losing cost-accounting provenance. The SDK may explicitly return None.
    usage = response.usage_metadata
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        "total_tokens": usage.total_token_count,
    }


def generate_with_retry(
    client: Any,
    model: str,
    contents: Any,
    config: Any,
    validate: Callable[[str], Any] | None = None,
    max_attempts: int = 12,
) -> tuple[str, dict[str, Any] | None]:
    """Call the model with backoff on retriable API errors and in-loop
    re-generation on a malformed/truncated (non-API) JSON response.

    Uses a long jittered backoff for sparse quota windows,
    a short backoff for transient truncations, and a loud raise on exhaustion.
    Honors the ``MULLIGAN_GEMINI_MIN_INTERVAL_S`` proactive throttle.
    """
    from google.genai import errors

    api_retry_index = 0
    for attempt in range(1, max_attempts + 1):
        try:
            _proactive_throttle()
            response = client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as exc:
            if is_hard_quota_error(exc):
                raise RuntimeError(
                    "Gemini API quota is unavailable because billing/prepayment is inactive; "
                    "not retrying a request that cannot succeed without an account change."
                ) from exc
            if exc.code not in RETRIABLE_CODES or attempt == max_attempts:
                raise RuntimeError(
                    f"Gemini API request failed (code {exc.code}) after {attempt} attempt(s). "
                    "If RESOURCE_EXHAUSTED persists, check AI Studio quota/billing; the run "
                    "checkpoint supports resume."
                ) from exc
            api_retry_index += 1
            sleep_s = jittered_retry_delay_s(api_retry_index)
            print(
                f"  retriable API error {exc.code}, attempt {attempt}/{max_attempts}, "
                f"sleeping {sleep_s:.0f}s [{str(exc)[:160]}]"
            )
            time.sleep(sleep_s)
            continue
        # ``response.text`` can raise for blocked or part-less candidates; retry those
        # like an empty response.
        text_error: Exception | None = None
        try:
            text = response.text
        except Exception as exc:  # noqa: BLE001 - SDK response properties may raise arbitrary errors.
            text = ""
            text_error = exc
        if not text:
            # Empty text is almost always a Gemini-3 MAX_TOKENS truncation (hidden
            # thinking ate the budget). Regenerate — temperature>0 varies the
            # thinking length, so a retry typically returns valid JSON.
            finish = None
            if getattr(response, "candidates", None):
                finish = getattr(response.candidates[0], "finish_reason", None)
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Empty response text from {model} after {attempt} attempt(s) "
                    f"(finish_reason={finish}); raise max_output_tokens. {response}"
                ) from text_error
            sleep_s = jittered_retry_delay_s(
                1,
                initial_delay_s=TRANSIENT_RESPONSE_DELAY_S,
                backoff_factor=1.0,
                max_base_delay_s=TRANSIENT_RESPONSE_DELAY_S,
            )
            detail = "" if text_error is None else f" [{type(text_error).__name__}: {text_error}]"
            print(
                f"  empty response (finish_reason={finish}), attempt "
                f"{attempt}/{max_attempts}, sleeping {sleep_s:.0f}s{detail}"
            )
            time.sleep(sleep_s)
            continue
        if validate is not None:
            try:
                validate(text)
            except ValueError as exc:  # JSONDecodeError (subclass) + missing-keys
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Response failed JSON validation after {attempt} attempt(s): {exc}"
                    ) from exc
                sleep_s = jittered_retry_delay_s(
                    1,
                    initial_delay_s=TRANSIENT_RESPONSE_DELAY_S,
                    backoff_factor=1.0,
                    max_base_delay_s=TRANSIENT_RESPONSE_DELAY_S,
                )
                print(
                    f"  malformed/truncated response, attempt {attempt}/{max_attempts}, "
                    f"re-generating in {sleep_s:.0f}s [{str(exc)[:120]}]"
                )
                time.sleep(sleep_s)
                continue
        return text, _response_usage(response)
    raise AssertionError("unreachable")


def inline_video_part(path: Path, fps: float) -> types.Part:
    from google.genai import types

    return types.Part(
        inline_data=types.Blob(data=Path(path).read_bytes(), mime_type="video/mp4"),
        video_metadata=types.VideoMetadata(fps=fps),
    )


def file_video_part(client: Any, path: Path, fps: float) -> types.Part:
    """Video part via the Files API (cached per path). Inline payloads hit an
    opaque 429 on some projects; uploads dodge it and persist ~48h server-side,
    so the N samples per episode reuse one upload."""
    from google.genai import types

    key = str(path)
    with _FILE_UPLOAD_LOCK:
        uri = _FILE_UPLOAD_CACHE.get(key)
    if uri is None:
        print(f"  uploading {Path(path).name} via Files API")
        f = client.files.upload(file=key)
        while f.name and f.state is not None and f.state.name == "PROCESSING":
            time.sleep(2)
            f = client.files.get(name=f.name)
        if f.state is None or f.state.name != "ACTIVE" or not f.uri:
            raise RuntimeError(f"Files API upload for {path} ended in state {f.state}")
        uri = f.uri
        with _FILE_UPLOAD_LOCK:
            _FILE_UPLOAD_CACHE[key] = uri
    return types.Part(
        file_data=types.FileData(file_uri=uri, mime_type="video/mp4"),
        video_metadata=types.VideoMetadata(fps=fps),
    )


# --------------------------------------------------------------------------- #
# Prompt + parsing.
# --------------------------------------------------------------------------- #

_REFERENCE_TEXT = (
    "REFERENCE IMAGE (not part of the episode): this is what a successful, fully "
    "completed end-state looks like from the same cameras. Compare the final pose "
    "in the episode against this reference."
)


def json_prompt(
    item: dict[str, Any],
    input_mode: str,
    side_label: str = "SIDE",
    wrist_label: str = "WRIST",
) -> str:
    context = {
        "episode_index": item["episode_index"],
        "video_duration_s": round(float(item["duration_s"]), 2),
        "gripper_sensor_trace": item["sensor_trace"],
    }
    if input_mode == "dual":
        context["gripper_position_per_second"] = item["gripper_series_1hz"]
        header = (
            f"Label this episode from the attached {side_label} video, {wrist_label} video, "
            "and the two FINAL-FRAME stills.\n"
            "All sensor fields below are proprioceptive robot data, not human labels.\n"
        )
    else:
        header = (
            f"Label this episode from the attached side-by-side video "
            f"(LEFT={side_label.lower()} camera, RIGHT={wrist_label.lower()} camera).\n"
            "The gripper_sensor_trace below is proprioceptive sensor data from the robot, not a human label.\n"
            "Remember: a jaw close does not prove a grasp; jaws never reopening proves there was no release.\n"
        )
    return (
        header
        + f"\nContext:\n{json.dumps(context, indent=2)}\n\n"
        + "Return one JSON object following the response schema."
    )


def parse_response(text: str, required: set[str]) -> dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    missing = required - set(parsed)
    if missing:
        raise ValueError(f"Response missing required keys: {sorted(missing)}")
    return parsed


# --------------------------------------------------------------------------- #
# Part assembly + per-sample call.
# --------------------------------------------------------------------------- #


def build_parts(
    spec: StageLabelTaskSpec, config: LabelerConfig, client: Any, item: dict[str, Any]
) -> list[types.Part]:
    """Assemble the request parts for one episode (dual or combo)."""
    from google.genai import types

    # Camera-role tokens (default "SIDE"/"WRIST"; routing overrides to SIDE-1/SIDE-2).
    # Upper token where the original text used caps; lowered token where it used
    # lowercase, so the default reproduces the original bytes exactly.
    side_label = spec.side_camera_label
    wrist_label = spec.wrist_camera_label

    # Fail loud on an unsupported combination: build_items does NOT thread the
    # per-task labels into the combo (hstack) asset, whose drawtext is baked as
    # SIDE/WRIST. Running combo mode for a task with non-default labels would
    # silently pair e.g. SIDE-1/SIDE-2 prompt text with SIDE/WRIST-burned video.
    if config.input_mode == "combo" and (side_label, wrist_label) != ("SIDE", "WRIST"):
        raise ValueError(
            f"{spec.name}: input_mode='combo' is not supported for a task with "
            f"non-default camera labels ({side_label!r}/{wrist_label!r}): the combo "
            "asset's drawtext is built with the default SIDE/WRIST tokens (per-task "
            "labels are not threaded into build_items' combo_video yet). Use "
            "input_mode='dual', or thread the labels through combo_video first."
        )

    def _video(path: str) -> types.Part:
        if config.video_transport == "file":
            return file_video_part(client, Path(path), config.video_fps)
        return inline_video_part(Path(path), config.video_fps)

    def _png(path: str) -> types.Part:
        return types.Part(
            inline_data=types.Blob(data=Path(path).read_bytes(), mime_type="image/png")
        )

    parts: list[types.Part] = []
    if config.reference_image is not None:
        parts.append(types.Part(text=_REFERENCE_TEXT))
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    data=config.reference_image.read_bytes(), mime_type="image/png"
                )
            )
        )

    if config.few_shot:
        from mulligan.real.stage_labeling.assets import task_build_dir

        assets_dir = task_build_dir(spec, config.build_dir) / "assets"
        parts.append(
            types.Part(
                text=(
                    "FIRST, study these LABELED REFERENCE EXAMPLES (each is a full side-by-side "
                    "combo clip of a different episode with its correct max_stage). Use them as the "
                    "calibration standard for the SAME judgement on the target episode that follows."
                )
            )
        )
        for ep, label, why in config.few_shot:
            parts.append(
                types.Part(text=f"REFERENCE EXAMPLE — correct label max_stage={label}. {why}")
            )
            parts.append(_video(str(assets_dir / f"episode_{ep:03d}_combo.mp4")))
        parts.append(
            types.Part(
                text="END OF REFERENCE EXAMPLES. Now label the TARGET episode below by the same standard:"
            )
        )

    if config.input_mode == "dual":
        parts += [
            types.Part(text=f"{side_label} camera video:"),
            _video(item["side_path"]),
            types.Part(text=f"{wrist_label} camera video (same episode, synchronized):"),
            _video(item["wrist_path"]),
            types.Part(text=f"FINAL frame, {side_label.lower()} camera:"),
            _png(item["side_final_frame"]),
            types.Part(text=f"FINAL frame, {wrist_label.lower()} camera:"),
            _png(item["wrist_final_frame"]),
        ]
        # Optional extra synchronized views (e.g. routing's wrist_left 3rd cam),
        # each a video + final frame with its role label. Absent for side+wrist tasks.
        for lbl, vpath, fpath in zip(
            item.get("extra_labels", []),
            item.get("extra_paths", []),
            item.get("extra_final_frames", []),
        ):
            parts += [
                types.Part(text=f"{lbl} camera video (same episode, synchronized):"),
                _video(vpath),
                types.Part(text=f"FINAL frame, {lbl.lower()} camera:"),
                _png(fpath),
            ]
        if config.send_final_crops:
            parts += [
                types.Part(
                    text=f"ZOOMED final frame, {side_label.lower()} camera (center crop, 3x):"
                ),
                _png(item["side_final_crop"]),
                types.Part(
                    text=f"ZOOMED final frame, {wrist_label.lower()} camera (center crop, 3x):"
                ),
                _png(item["wrist_final_crop"]),
            ]
            if item.get("grasp_early_crop"):
                parts += [
                    types.Part(
                        text=f"ZOOMED EARLY GRASP frame, {wrist_label.lower()} camera "
                        "(~0.5s after the jaws closed, crop, 3x):"
                    ),
                    _png(item["grasp_early_crop"]),
                ]
            if item.get("grasp_moment_crop"):
                parts += [
                    types.Part(
                        text=f"ZOOMED GRASP-MOMENT frame, {wrist_label.lower()} camera "
                        "(~1.5s after the jaws closed, crop, 3x):"
                    ),
                    _png(item["grasp_moment_crop"]),
                ]
            if item.get("release_moment_crop"):
                parts += [
                    types.Part(
                        text=f"ZOOMED RELEASE-MOMENT frame, {wrist_label.lower()} camera "
                        "(~0.3s before the jaws reopened, crop, 3x):"
                    ),
                    _png(item["release_moment_crop"]),
                ]
        if config.send_grasp_crops and item.get("grasp_window_crop"):
            parts += [
                types.Part(
                    text=f"ZOOMED GRASP-WINDOW strip, {wrist_label.lower()} camera (left->right in time, "
                    "centred on the grasp/approach moment; 3x crop). Use this to judge "
                    "whether the OPEN jaws straddled the object squarely at the approach "
                    "(the S0 vs S1 decision):"
                ),
                _png(item["grasp_window_crop"]),
            ]
        parts.append(types.Part(text=json_prompt(item, config.input_mode, side_label, wrist_label)))
    else:
        parts += [
            _video(item["combo_path"]),
            types.Part(text=json_prompt(item, config.input_mode, side_label, wrist_label)),
        ]
    return parts


def run_sample(
    spec: StageLabelTaskSpec,
    config: LabelerConfig,
    client: Any,
    item: dict[str, Any],
    sample_idx: int,
    system_prompt: str,
    schema: Any,
    required: set[str],
) -> dict[str, Any]:
    from google.genai import types

    parts = build_parts(spec, config, client, item)
    # Vertex AI requires an explicit role on Content ("user"|"model"); the
    # Developer API tolerated its absence.
    contents = types.Content(role="user", parts=parts)
    gen_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=config.temperature,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=config.max_output_tokens,
        thinking_config=(
            types.ThinkingConfig(thinking_level=config.thinking_level)
            if config.thinking_level
            else None
        ),
        media_resolution=(
            types.MediaResolution(f"MEDIA_RESOLUTION_{config.media_resolution.upper()}")
            if config.media_resolution
            else None
        ),
    )
    raw_text, usage = generate_with_retry(
        client, config.model, contents, gen_config, validate=lambda t: parse_response(t, required)
    )
    parsed = parse_response(raw_text, required)
    if int(parsed["episode_index"]) != int(item["episode_index"]):
        parsed["episode_index"] = int(item["episode_index"])
    parsed = apply_sensor_constraints(spec.sensor_rules, item, parsed)
    return {
        "episode_index": int(item["episode_index"]),
        "sample_idx": sample_idx,
        "model": config.model,
        "temperature": config.temperature,
        "video_fps": config.video_fps,
        "prompt_variant": config.prompt_variant or spec.default_prompt_variant,
        "input_mode": config.input_mode,
        "video_transport": config.video_transport,
        "raw_text": raw_text,
        "usage": usage,
        "parsed": parsed,
    }


# --------------------------------------------------------------------------- #
# Checkpoint + driver.
# --------------------------------------------------------------------------- #


def _checkpoint_path(output_dir: Path, run_name: str) -> Path:
    return output_dir / run_name / "raw_results_partial.json"


# Config keys that change what a sample MEANS; a resume that reuses samples produced under a
# different value silently corrupts the run (e.g. reusing samples after a prompt-variant change).
# `samples`/`temperature` are excluded: adding samples is a legitimate resume and temperature
# does not change the labeling contract.
_RESUME_SEMANTIC_KEYS = (
    "model",
    "prompt_variant",
    "video_fps",
    "media_resolution",
    "input_mode",
    "video_transport",
    "send_final_crops",
    "send_grasp_crops",
    "dataset_repo_id",
    "gemini_route",
)


def _guard_resume_config(
    spec: StageLabelTaskSpec,
    config: LabelerConfig,
    run_dir: Path,
    gemini_route: dict[str, Any],
    *,
    new_requests_pending: bool,
) -> None:
    """Fail loud if a resumed checkpoint was produced under a different labeling config.

    Resume matches cached samples by ``(episode_index, sample_idx)`` only, so without this
    guard a prompt/model/media/dataset change would silently reuse stale samples while
    provenance is rewritten to the NEW config. Only keys PRESENT in the cached provenance are
    compared, so pre-existing checkpoints (written before a key was persisted) don't spuriously
    trip; a missing provenance file downgrades to a loud warning rather than a crash."""
    provenance_path = run_dir / "provenance.json"
    if not provenance_path.exists():
        print(
            f"WARNING: resuming {config.run_name} but {provenance_path} is absent — cannot "
            f"verify the cached samples share this run's labeling config; proceeding on trust."
        )
        return
    prev = json.loads(provenance_path.read_text())
    if "gemini_route" not in prev and new_requests_pending:
        raise RuntimeError(
            f"resume route unknown for {config.run_name}: cached provenance predates the "
            "Gemini route audit and new requests are pending. Use a fresh run_name instead "
            "of mixing unaudited cached samples with the current route. A replay with zero "
            "new requests remains allowed because it cannot mix backends."
        )
    cur = {
        "model": config.model,
        "prompt_variant": config.prompt_variant or spec.default_prompt_variant,
        "video_fps": config.video_fps,
        "media_resolution": config.media_resolution,
        "input_mode": config.input_mode,
        "video_transport": config.video_transport,
        "send_final_crops": config.send_final_crops,
        "send_grasp_crops": config.send_grasp_crops,
        "dataset_repo_id": spec.dataset_repo_id,
        "gemini_route": gemini_route,
    }
    diffs = {k: (prev[k], cur[k]) for k in _RESUME_SEMANTIC_KEYS if k in prev and prev[k] != cur[k]}
    if diffs:
        raise RuntimeError(
            f"resume config mismatch for {config.run_name}: cached samples were produced under a "
            f"DIFFERENT labeling config; reusing them would silently corrupt this run. Differing "
            f"keys (cached -> current): {diffs}. Clear {run_dir} or use a fresh run_name to relabel."
        )


def _write_checkpoint(results: list[dict[str, Any]], output_dir: Path, run_name: str) -> None:
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _checkpoint_path(output_dir, run_name).write_text(json.dumps(results, indent=2) + "\n")


def write_outputs(
    spec: StageLabelTaskSpec,
    config: LabelerConfig,
    results: list[dict[str, Any]],
    *,
    gemini_route: dict[str, Any] | None = None,
) -> Path:
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw_results.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = [{**r["parsed"], "sample_idx": r["sample_idx"]} for r in results]
    pd.DataFrame(rows).to_csv(run_dir / "sample_labels.csv", index=False)
    provenance = {
        "task": spec.name,
        "run_name": config.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": config.model,
        "prompt_variant": config.prompt_variant or spec.default_prompt_variant,
        "samples_per_episode": config.samples,
        "temperature": config.temperature,
        "video_fps": config.video_fps,
        "media_resolution": config.media_resolution,
        "input_mode": config.input_mode,
        "video_transport": config.video_transport,
        "send_final_crops": config.send_final_crops,
        "send_grasp_crops": config.send_grasp_crops,
        "dataset_repo_id": spec.dataset_repo_id,
        "blind": True,
        "result_count": len(results),
        "episodes": sorted({r["episode_index"] for r in results}),
    }
    if gemini_route is not None:
        provenance["gemini_route"] = gemini_route
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return run_dir / "sample_labels.csv"


def run_labeler(
    spec: StageLabelTaskSpec,
    config: LabelerConfig,
    items: list[dict[str, Any]],
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Self-consistency threadpool over (item, sample) with checkpoint/resume.

    Returns the raw per-sample records and writes raw_results.json /
    sample_labels.csv / provenance.json under ``output_dir/run_name``.
    """
    variant = config.prompt_variant or spec.default_prompt_variant
    system_prompt = spec.system_prompt(variant)
    schema = to_genai_schema(spec)
    required = set(required_keys(spec))

    if client is None:
        client = make_client()
    gemini_route = client_route(client)

    results: list[dict[str, Any]] = []
    ckpt = _checkpoint_path(config.output_dir, config.run_name)
    if config.resume and ckpt.exists():
        results = json.loads(ckpt.read_text())
        print(f"resuming {config.run_name}: {len(results)} samples already complete")
    completed = {(r["episode_index"], r["sample_idx"]) for r in results}

    pending = [
        (item, s)
        for item in items
        for s in range(config.samples)
        if (item["episode_index"], s) not in completed
    ]
    if config.resume and ckpt.exists():
        _guard_resume_config(
            spec,
            config,
            ckpt.parent,
            gemini_route,
            new_requests_pending=bool(pending),
        )
    lock = threading.Lock()

    def _worker(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
        item, sample_idx = job
        return run_sample(spec, config, client, item, sample_idx, system_prompt, schema, required)

    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(_worker, job): job for job in pending}
        try:
            for future in as_completed(futures):
                result = future.result()  # raises loudly on worker failure
                with lock:
                    results.append(result)
                    _write_checkpoint(results, config.output_dir, config.run_name)
                p = result["parsed"]
                gist = f"S{p[spec.stage_field]} {p[spec.final_state_field]}"
                print(
                    f"episode {result['episode_index']:03d} sample {result['sample_idx']}: "
                    f"{gist} ({p['confidence']}) [{len(results)}/{len(pending) + len(completed)}]"
                )
        except BaseException:
            # Don't let the with-block drain queued jobs (burning quota) after the
            # checkpoint loop is already dead.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    out = write_outputs(spec, config, results, gemini_route=gemini_route)
    print(out)
    return results

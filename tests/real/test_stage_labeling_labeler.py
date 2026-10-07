"""Unit guards for mulligan.real.stage_labeling.labeler (the genai engine).

No network: a fake client + a google.genai stub exercise the retry/validate
control flow, JSON parsing, the dual/combo part assembly, and the
checkpoint/resume threadpool. google.genai is uninstallable here, so the stub is
mandatory; it is installed before importing the labeler.
"""

from __future__ import annotations

import sys
import types as _pytypes
from pathlib import Path

import pytest


def _install_genai_stub() -> None:
    if "google.genai" in sys.modules:
        return
    google = sys.modules.get("google") or _pytypes.ModuleType("google")
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
    for name in (
        "Schema", "Part", "Blob", "FileData", "VideoMetadata", "Content", "GenerateContentConfig", "HttpOptions",
    ):  # fmt: skip
        setattr(types_mod, name, _Stored)
    genai.Client = _Stored
    genai.errors = errors
    genai.types = types_mod
    google.genai = genai
    sys.modules.setdefault("google", google)
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.errors"] = errors
    sys.modules["google.genai.types"] = types_mod


_install_genai_stub()

import mulligan.real.stage_specs as sl  # noqa: E402
from mulligan.real.stage_labeling import labeler  # noqa: E402
from mulligan.real.stage_labeling.labeler import LabelerConfig  # noqa: E402

MARKER = "marker_d2"


class _Usage:
    prompt_token_count = 10
    candidates_token_count = 20
    thoughts_token_count = 5
    total_token_count = 35


class _Resp:
    def __init__(self, text, usage=None):
        self.text = text
        self.usage_metadata = usage


class _Models:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def generate_content(self, model, contents, config):
        action = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(action, Exception):
            raise action
        return action


class _Client:
    def __init__(self, script):
        self.models = _Models(script)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(labeler.time, "sleep", lambda *_a, **_k: None)


# --------------------------------------------------------------------------- #
# Explicit Gemini API routing.
# --------------------------------------------------------------------------- #


def _capture_genai_client(monkeypatch):
    from google import genai

    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return _pytypes.SimpleNamespace(**kwargs)

    monkeypatch.setattr(genai, "Client", build)
    return calls


def test_make_client_developer_is_explicit_and_uses_v1beta(monkeypatch):
    calls = _capture_genai_client(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "private-test-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

    client = labeler.make_client(route="developer")

    assert len(calls) == 1
    assert client.vertexai is False
    assert client.api_key == "private-test-key"
    assert client.http_options.api_version == "v1beta"
    assert "project" not in calls[0]


def test_make_client_defaults_to_developer_api(monkeypatch):
    calls = _capture_genai_client(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "private-test-key")
    monkeypatch.delenv("MULLIGAN_GEMINI_ROUTE", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")

    client = labeler.make_client()

    assert len(calls) == 1
    assert client.vertexai is False
    assert client.api_key == "private-test-key"
    assert client.http_options.api_version == "v1beta"


def test_make_client_developer_requires_key(monkeypatch):
    _capture_genai_client(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="requires GEMINI_API_KEY"):
        labeler.make_client(route="developer")


def test_make_client_vertex_is_explicit_and_uses_v1(monkeypatch):
    calls = _capture_genai_client(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    client = labeler.make_client(route="vertex")

    assert len(calls) == 1
    assert client.vertexai is True
    assert client.project == "test-project"
    assert client.location == "global"
    assert client.http_options.api_version == "v1"
    assert "api_key" not in calls[0]


def test_make_client_rejects_unknown_route(monkeypatch):
    _capture_genai_client(monkeypatch)
    with pytest.raises(ValueError, match="invalid MULLIGAN_GEMINI_ROUTE"):
        labeler.make_client(route="mystery")  # type: ignore[arg-type]


def test_configure_developer_api_sets_route_without_printing_key(monkeypatch, capsys):
    from mulligan.real.stage_labeling.cascade_pipeline.common import (
        configure_gemini_developer_api,
    )

    monkeypatch.setenv("GEMINI_API_KEY", "must-not-appear")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    configure_gemini_developer_api()

    assert labeler.os.environ["MULLIGAN_GEMINI_ROUTE"] == "developer"
    assert labeler.os.environ["GOOGLE_GENAI_USE_VERTEXAI"] == "false"
    output = capsys.readouterr().out
    assert "api_key=set" in output
    assert "must-not-appear" not in output


def test_configure_gemini_allows_explicit_vertex_fallback(monkeypatch, capsys):
    from mulligan.real.stage_labeling.cascade_pipeline.common import configure_gemini

    monkeypatch.setenv("MULLIGAN_GEMINI_ROUTE", "vertex")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "fallback-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    assert configure_gemini() == "vertex"
    assert labeler.os.environ["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
    assert "project=fallback-project" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# generate_with_retry.
# --------------------------------------------------------------------------- #


def test_jittered_retry_delay_has_one_based_backoff_cap_and_jitter_band():
    assert labeler.jittered_retry_delay_s(1, jitter_unit=0.0) == pytest.approx(14.0)
    assert labeler.jittered_retry_delay_s(2, jitter_unit=0.5) == pytest.approx(36.0)
    assert labeler.jittered_retry_delay_s(10, jitter_unit=1.0) == pytest.approx(390.0)
    assert labeler.jittered_retry_delay_s(
        4,
        initial_delay_s=labeler.TRANSIENT_RESPONSE_DELAY_S,
        backoff_factor=1.0,
        max_base_delay_s=labeler.TRANSIENT_RESPONSE_DELAY_S,
        jitter_unit=0.5,
    ) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="positive integer"):
        labeler.jittered_retry_delay_s(0, jitter_unit=0.5)


def test_generate_with_retry_uses_shared_delay_for_api_and_response_retries(monkeypatch):
    from google.genai import errors

    delay_calls: list[tuple[int, dict[str, float]]] = []
    sleeps: list[float] = []

    def delay(retry_index: int, **policy: float) -> float:
        delay_calls.append((retry_index, policy))
        return float(len(delay_calls))

    def validate(text: str) -> None:
        if text == "bad json":
            raise ValueError("malformed")

    monkeypatch.setattr(labeler, "jittered_retry_delay_s", delay)
    monkeypatch.setattr(labeler.time, "sleep", sleeps.append)
    client = _Client(
        [
            errors.APIError(503),
            _Resp(""),
            _Resp("bad json"),
            _Resp("{}"),
        ]
    )

    text, _usage = labeler.generate_with_retry(
        client,
        "m",
        None,
        None,
        validate=validate,
    )

    short_policy = {
        "initial_delay_s": labeler.TRANSIENT_RESPONSE_DELAY_S,
        "backoff_factor": 1.0,
        "max_base_delay_s": labeler.TRANSIENT_RESPONSE_DELAY_S,
    }
    assert text == "{}"
    assert delay_calls == [(1, {}), (1, short_policy), (1, short_policy)]
    assert sleeps == [1.0, 2.0, 3.0]


def test_generate_success_first_try():
    client = _Client([_Resp('{"a": 1}', _Usage())])
    text, usage = labeler.generate_with_retry(client, "m", None, None)
    assert text == '{"a": 1}'
    assert usage == {
        "prompt_tokens": 10,
        "output_tokens": 20,
        "thinking_tokens": 5,
        "total_tokens": 35,
    }


def test_generate_retries_retriable_then_succeeds():
    from google.genai import errors

    client = _Client([errors.APIError(503), errors.APIError(429), _Resp("{}")])
    text, _ = labeler.generate_with_retry(client, "m", None, None)
    assert text == "{}"
    assert client.models.calls == 3


def test_generate_raises_on_non_retriable():
    from google.genai import errors

    client = _Client([errors.APIError(400)])
    with pytest.raises(RuntimeError, match="code 400"):
        labeler.generate_with_retry(client, "m", None, None)


def test_generate_fails_immediately_when_prepayment_is_depleted():
    from google.genai import errors

    depleted = errors.APIError(
        429,
        {"error": {"message": "Your prepayment credits are depleted."}},
    )
    client = _Client([depleted])
    with pytest.raises(RuntimeError, match="not retrying"):
        labeler.generate_with_retry(client, "m", None, None)
    assert client.models.calls == 1


@pytest.mark.parametrize(
    "message",
    [
        "Your prepayment credits are depleted.",
        "Billing account is disabled",
        "The billing account has been closed",
    ],
)
def test_hard_quota_predicate_is_shared_and_case_insensitive(message: str):
    assert labeler.is_hard_quota_error(RuntimeError(message))
    assert not labeler.is_hard_quota_error(RuntimeError("quota window is temporarily exhausted"))


def test_generate_regenerates_on_bad_json_then_succeeds():
    client = _Client([_Resp("not json"), _Resp('{"x": 1}')])
    validate = lambda t: labeler.parse_response(t, set())  # noqa: E731
    text, _ = labeler.generate_with_retry(client, "m", None, None, validate=validate)
    assert text == '{"x": 1}'
    assert client.models.calls == 2


def test_generate_raises_after_max_attempts_on_persistent_bad_json():
    client = _Client([_Resp("still not json")])
    validate = lambda t: labeler.parse_response(t, set())  # noqa: E731
    with pytest.raises(RuntimeError, match="failed JSON validation"):
        labeler.generate_with_retry(client, "m", None, None, validate=validate, max_attempts=3)
    assert client.models.calls == 3


class _RaisingTextResp:
    usage_metadata = None
    candidates = [_pytypes.SimpleNamespace(finish_reason="SAFETY")]

    @property
    def text(self):
        raise ValueError("response has no text part")


def test_generate_retries_when_response_text_property_raises():
    client = _Client([_RaisingTextResp(), _Resp("{}")])
    text, _ = labeler.generate_with_retry(client, "m", None, None)
    assert text == "{}"
    assert client.models.calls == 2


def test_generate_raises_runtime_error_when_response_text_keeps_raising():
    client = _Client([_RaisingTextResp()])
    with pytest.raises(RuntimeError, match="Empty response") as info:
        labeler.generate_with_retry(client, "m", None, None, max_attempts=2)
    assert isinstance(info.value.__cause__, ValueError)
    assert client.models.calls == 2


def test_generate_raises_on_empty_text():
    client = _Client([_Resp("")])
    with pytest.raises(RuntimeError, match="Empty response"):
        labeler.generate_with_retry(client, "m", None, None)


# --------------------------------------------------------------------------- #
# parse_response + json_prompt.
# --------------------------------------------------------------------------- #


def test_parse_response_fail_loud():
    assert labeler.parse_response('{"a": 1, "b": 2}', {"a"}) == {"a": 1, "b": 2}
    with pytest.raises(ValueError, match="missing required keys"):
        labeler.parse_response('{"a": 1}', {"a", "b"})
    with pytest.raises(ValueError, match="Expected JSON object"):
        labeler.parse_response("[1, 2]", set())


def test_json_prompt_modes():
    item = {
        "episode_index": 7,
        "duration_s": 9.3,
        "sensor_trace": {"jaw_close_time_s": 3.0},
        "gripper_series_1hz": [0.0, 0.8],
    }
    dual = labeler.json_prompt(item, "dual")
    assert "SIDE video, WRIST video" in dual
    assert "gripper_position_per_second" in dual  # 1 Hz series included in dual
    combo = labeler.json_prompt(item, "combo")
    assert "side-by-side video" in combo
    assert "gripper_position_per_second" not in combo


# --------------------------------------------------------------------------- #
# build_parts (dual / combo / crops), reading real temp files.
# --------------------------------------------------------------------------- #


def _make_item(tmp_path: Path) -> dict:
    names = [
        "side.mp4", "wrist.mp4", "combo.mp4", "side_final.png", "wrist_final.png",
        "side_crop.png", "wrist_crop.png", "grasp.png", "early.png", "release.png",
    ]  # fmt: skip
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    return {
        "episode_index": 0,
        "duration_s": 8.0,
        "sensor_trace": {"jaw_close_time_s": 3.0, "jaw_reopen_time_s": 6.0},
        "gripper_series_1hz": [0.0, 0.8],
        "side_path": str(tmp_path / "side.mp4"),
        "wrist_path": str(tmp_path / "wrist.mp4"),
        "combo_path": str(tmp_path / "combo.mp4"),
        "side_final_frame": str(tmp_path / "side_final.png"),
        "wrist_final_frame": str(tmp_path / "wrist_final.png"),
        "side_final_crop": str(tmp_path / "side_crop.png"),
        "wrist_final_crop": str(tmp_path / "wrist_crop.png"),
        "grasp_moment_crop": str(tmp_path / "grasp.png"),
        "grasp_early_crop": str(tmp_path / "early.png"),
        "release_moment_crop": str(tmp_path / "release.png"),
    }


def _texts(parts):
    return [getattr(p, "text", None) for p in parts if getattr(p, "text", None) is not None]


def test_build_parts_combo(tmp_path):
    spec = sl.get_label_task_spec(MARKER)
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, input_mode="combo")
    parts = labeler.build_parts(spec, cfg, None, _make_item(tmp_path))
    assert len(parts) == 2  # combo video + prompt
    assert _texts(parts)[-1].startswith("Label this episode from the attached side-by-side")


def test_build_parts_dual_no_crops(tmp_path):
    spec = sl.get_label_task_spec(MARKER)
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, input_mode="dual")
    parts = labeler.build_parts(spec, cfg, None, _make_item(tmp_path))
    labels = _texts(parts)
    assert labels[:4] == [
        "SIDE camera video:",
        "WRIST camera video (same episode, synchronized):",
        "FINAL frame, side camera:",
        "FINAL frame, wrist camera:",
    ]
    assert len(parts) == 9  # 4 labels + 2 videos + 2 stills + prompt


def test_build_parts_dual_with_crops(tmp_path):
    spec = sl.get_label_task_spec(MARKER)
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, input_mode="dual", send_final_crops=True)
    parts = labeler.build_parts(spec, cfg, None, _make_item(tmp_path))
    labels = " | ".join(_texts(parts))
    assert "ZOOMED final frame, side" in labels
    assert "ZOOMED EARLY GRASP" in labels
    assert "ZOOMED GRASP-MOMENT" in labels
    assert "ZOOMED RELEASE-MOMENT" in labels


def test_build_parts_reference_image_leads(tmp_path):
    spec = sl.get_label_task_spec(MARKER)
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"r")
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, input_mode="combo", reference_image=ref)
    parts = labeler.build_parts(spec, cfg, None, _make_item(tmp_path))
    assert _texts(parts)[0].startswith("REFERENCE IMAGE")


# --------------------------------------------------------------------------- #
# Camera-role labels: default (marker/square) reproduce SIDE/WRIST byte-for-byte;
# routing_d2 emits its custom SIDE-1/SIDE-2 tokens (dual-camera spec: two side
# views, no wrist).
# --------------------------------------------------------------------------- #


# Exact byte-level pins for the ORIGINAL default wording (pre-parameterization);
# a default-label drift cannot hide behind a substring match.
_DEFAULT_DUAL_HEADER = (
    "Label this episode from the attached SIDE video, WRIST video, "
    "and the two FINAL-FRAME stills.\n"
    "All sensor fields below are proprioceptive robot data, not human labels.\n"
)
_DEFAULT_COMBO_HEADER = (
    "Label this episode from the attached side-by-side video "
    "(LEFT=side camera, RIGHT=wrist camera).\n"
    "The gripper_sensor_trace below is proprioceptive sensor data from the robot, "
    "not a human label.\n"
    "Remember: a jaw close does not prove a grasp; jaws never reopening proves "
    "there was no release.\n"
)


def _expected_dual_crop_labels(side: str, wrist: str) -> list[str]:
    """The full ordered text-part label list for dual + send_final_crops with every
    crop present in the item (as _make_item provides)."""
    return [
        f"{side} camera video:",
        f"{wrist} camera video (same episode, synchronized):",
        f"FINAL frame, {side.lower()} camera:",
        f"FINAL frame, {wrist.lower()} camera:",
        f"ZOOMED final frame, {side.lower()} camera (center crop, 3x):",
        f"ZOOMED final frame, {wrist.lower()} camera (center crop, 3x):",
        f"ZOOMED EARLY GRASP frame, {wrist.lower()} camera (~0.5s after the jaws closed, crop, 3x):",
        f"ZOOMED GRASP-MOMENT frame, {wrist.lower()} camera (~1.5s after the jaws closed, crop, 3x):",
        f"ZOOMED RELEASE-MOMENT frame, {wrist.lower()} camera (~0.3s before the jaws reopened, crop, 3x):",
    ]


def test_json_prompt_default_labels_byte_identical():
    item = {
        "episode_index": 7,
        "duration_s": 9.3,
        "sensor_trace": {"jaw_close_time_s": 3.0},
        "gripper_series_1hz": [0.0, 0.8],
    }
    # Defaults must reproduce the original SIDE/WRIST wording EXACTLY (prefix pin).
    assert labeler.json_prompt(item, "dual").startswith(_DEFAULT_DUAL_HEADER)
    assert labeler.json_prompt(item, "combo").startswith(_DEFAULT_COMBO_HEADER)
    # Routing-style labels: uppercase token in dual, lowercased token in combo.
    dual_r = labeler.json_prompt(item, "dual", "SIDE-1", "SIDE-2")
    assert dual_r.startswith(
        "Label this episode from the attached SIDE-1 video, SIDE-2 video, "
        "and the two FINAL-FRAME stills.\n"
    )
    combo_r = labeler.json_prompt(item, "combo", "SIDE-1", "SIDE-2")
    assert combo_r.startswith(
        "Label this episode from the attached side-by-side video "
        "(LEFT=side-1 camera, RIGHT=side-2 camera).\n"
    )


def test_build_parts_default_labels_match_marker_and_square(tmp_path):
    expected = _expected_dual_crop_labels("SIDE", "WRIST")
    for task in ("marker_d2", "square_d2"):
        spec = sl.get_label_task_spec(task)
        cfg = LabelerConfig(
            run_name="r", output_dir=tmp_path, input_mode="dual", send_final_crops=True
        )
        parts = labeler.build_parts(spec, cfg, None, _make_item(tmp_path))
        labels = _texts(parts)
        # Exact full label-list equality (every part label, not substrings) + the
        # exact default prompt header on the trailing json_prompt part.
        assert labels[:-1] == expected, task
        assert labels[-1].startswith(_DEFAULT_DUAL_HEADER), task


def test_build_parts_routing_uses_custom_side1_side2_labels(tmp_path):
    spec = sl.get_label_task_spec("routing_d2")
    assert (spec.side_camera_label, spec.wrist_camera_label) == ("SIDE-1", "SIDE-2")
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, input_mode="dual", send_final_crops=True)
    parts = labeler.build_parts(spec, cfg, None, _make_item(tmp_path))
    labels = _texts(parts)
    assert labels[:-1] == _expected_dual_crop_labels("SIDE-1", "SIDE-2")
    assert labels[-1].startswith(
        "Label this episode from the attached SIDE-1 video, SIDE-2 video, "
        "and the two FINAL-FRAME stills.\n"
    )
    # No stray "wrist"/"WRIST" role token leaks into routing's dual prompt parts.
    joined = " | ".join(labels)
    assert "wrist" not in joined.lower()


def test_build_parts_combo_guard_raises_for_non_default_labels(tmp_path):
    """Fail-loud guard: combo mode is unsupported for non-default camera labels
    (the combo asset's drawtext is baked SIDE/WRIST; per-task labels are not
    threaded into build_items' combo_video)."""
    routing = sl.get_label_task_spec("routing_d2")
    combo_cfg = LabelerConfig(run_name="r", output_dir=tmp_path, input_mode="combo")
    with pytest.raises(ValueError, match="combo.*not supported.*non-default camera labels"):
        labeler.build_parts(routing, combo_cfg, None, _make_item(tmp_path))
    # Default-label tasks keep working in combo mode ...
    marker = sl.get_label_task_spec(MARKER)
    parts = labeler.build_parts(marker, combo_cfg, None, _make_item(tmp_path))
    assert len(parts) == 2  # combo video + prompt
    # ... and routing keeps working in dual mode.
    dual_cfg = LabelerConfig(run_name="r", output_dir=tmp_path, input_mode="dual")
    parts = labeler.build_parts(routing, dual_cfg, None, _make_item(tmp_path))
    assert _texts(parts)[0] == "SIDE-1 camera video:"


# --------------------------------------------------------------------------- #
# run_labeler: checkpoint / resume / outputs (run_sample stubbed).
# --------------------------------------------------------------------------- #

_FAKE_ROUTE = {
    "vertexai": False,
    "project": None,
    "location": None,
    "api_key_used": True,
    "api_version": "v1beta",
}


def test_run_labeler_checkpoint_and_outputs(tmp_path, monkeypatch):
    spec = sl.get_label_task_spec(MARKER)

    def fake_run_sample(
        spec,
        config,
        client,
        item,
        sample_idx,
        system_prompt,
        schema,
        required,
    ):
        return {
            "episode_index": item["episode_index"],
            "sample_idx": sample_idx,
            "parsed": {
                spec.stage_field: 4,
                spec.final_state_field: "marker_in_gripper_at_holder",
                "confidence": "high",
            },
        }

    monkeypatch.setattr(labeler, "run_sample", fake_run_sample)
    monkeypatch.setattr(labeler, "client_route", lambda _client: _FAKE_ROUTE)
    items = [{"episode_index": 0}, {"episode_index": 1}]
    cfg = LabelerConfig(
        run_name="run1", output_dir=tmp_path, build_dir=tmp_path, samples=2, workers=2
    )

    results = labeler.run_labeler(spec, cfg, items, client=object())
    assert len(results) == 4  # 2 episodes x 2 samples
    run_dir = tmp_path / "run1"
    assert (run_dir / "raw_results.json").exists()
    assert (run_dir / "sample_labels.csv").exists()
    import json

    prov = json.loads((run_dir / "provenance.json").read_text())
    assert prov["task"] == MARKER and prov["result_count"] == 4
    assert prov["gemini_route"] == _FAKE_ROUTE

    # Resume: a checkpoint with 2 of 4 done -> only the remaining 2 run.
    (run_dir / "raw_results_partial.json").write_text(
        json.dumps(
            [
                {
                    "episode_index": 0,
                    "sample_idx": 0,
                    "parsed": {
                        spec.stage_field: 4,
                        spec.final_state_field: "x",
                        "confidence": "low",
                    },
                },
                {
                    "episode_index": 0,
                    "sample_idx": 1,
                    "parsed": {
                        spec.stage_field: 4,
                        spec.final_state_field: "x",
                        "confidence": "low",
                    },
                },
            ]
        )
    )
    calls = {"n": 0}
    orig = fake_run_sample

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(labeler, "run_sample", counting)
    cfg_resume = LabelerConfig(
        run_name="run1", output_dir=tmp_path, build_dir=tmp_path, samples=2, workers=2, resume=True
    )
    results2 = labeler.run_labeler(spec, cfg_resume, items, client=object())
    assert len(results2) == 4
    assert calls["n"] == 2  # only episode 1's two samples re-run


def _write_provenance(run_dir: Path, spec, cfg: LabelerConfig, **overrides) -> None:
    import json

    prov = {
        "model": cfg.model,
        "prompt_variant": cfg.prompt_variant or spec.default_prompt_variant,
        "video_fps": cfg.video_fps,
        "media_resolution": cfg.media_resolution,
        "input_mode": cfg.input_mode,
        "video_transport": cfg.video_transport,
        "send_final_crops": cfg.send_final_crops,
        "send_grasp_crops": cfg.send_grasp_crops,
        "dataset_repo_id": spec.dataset_repo_id,
        "gemini_route": _FAKE_ROUTE,
    }
    prov.update(overrides)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps(prov))


def test_resume_guard_passes_on_matching_config(tmp_path):
    spec = sl.get_label_task_spec(MARKER)
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, build_dir=tmp_path)
    _write_provenance(tmp_path, spec, cfg)
    labeler._guard_resume_config(
        spec, cfg, tmp_path, _FAKE_ROUTE, new_requests_pending=True
    )  # must not raise


def test_resume_guard_raises_on_config_change(tmp_path):
    # cached samples were produced under a different prompt -> reusing them corrupts the run
    spec = sl.get_label_task_spec(MARKER)
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, build_dir=tmp_path)
    _write_provenance(tmp_path, spec, cfg, prompt_variant="SOME_OTHER_VARIANT")
    with pytest.raises(RuntimeError, match="resume config mismatch"):
        labeler._guard_resume_config(spec, cfg, tmp_path, _FAKE_ROUTE, new_requests_pending=True)


def test_resume_guard_raises_on_route_change(tmp_path):
    spec = sl.get_label_task_spec(MARKER)
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, build_dir=tmp_path)
    _write_provenance(tmp_path, spec, cfg)
    vertex_route = {
        "vertexai": True,
        "project": "old-project",
        "location": "global",
        "api_key_used": False,
        "api_version": "v1",
    }
    with pytest.raises(RuntimeError, match="gemini_route"):
        labeler._guard_resume_config(spec, cfg, tmp_path, vertex_route, new_requests_pending=True)


def test_resume_guard_skips_keys_absent_in_old_provenance(tmp_path):
    # a checkpoint written before media_resolution was persisted must not spuriously trip
    import json

    spec = sl.get_label_task_spec(MARKER)
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, build_dir=tmp_path)
    _write_provenance(tmp_path, spec, cfg)
    prov = json.loads((tmp_path / "provenance.json").read_text())
    prov.pop("media_resolution")
    (tmp_path / "provenance.json").write_text(json.dumps(prov))
    labeler._guard_resume_config(
        spec, cfg, tmp_path, _FAKE_ROUTE, new_requests_pending=True
    )  # must not raise


def test_resume_guard_rejects_legacy_route_when_new_requests_are_pending(tmp_path):
    import json

    spec = sl.get_label_task_spec(MARKER)
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, build_dir=tmp_path)
    _write_provenance(tmp_path, spec, cfg)
    prov = json.loads((tmp_path / "provenance.json").read_text())
    prov.pop("gemini_route")
    (tmp_path / "provenance.json").write_text(json.dumps(prov))

    with pytest.raises(RuntimeError, match="resume route unknown"):
        labeler._guard_resume_config(spec, cfg, tmp_path, _FAKE_ROUTE, new_requests_pending=True)


def test_resume_guard_allows_legacy_route_for_zero_request_replay(tmp_path):
    import json

    spec = sl.get_label_task_spec(MARKER)
    cfg = LabelerConfig(run_name="r", output_dir=tmp_path, build_dir=tmp_path)
    _write_provenance(tmp_path, spec, cfg)
    prov = json.loads((tmp_path / "provenance.json").read_text())
    prov.pop("gemini_route")
    (tmp_path / "provenance.json").write_text(json.dumps(prov))

    labeler._guard_resume_config(spec, cfg, tmp_path, _FAKE_ROUTE, new_requests_pending=False)

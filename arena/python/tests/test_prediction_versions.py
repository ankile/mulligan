"""Offline client and shared hash contract tests. No deployment credentials used."""

import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from convex import ConvexInt64, json_to_convex

from policy_arena.client import PolicyArenaAPIError, PolicyArenaClient
from policy_arena.prediction_hashes import (
    CONTENT_PROTOCOL,
    canonical_bytes,
    canonical_digest,
    manifest_digest,
    prediction_digest,
    prediction_payload,
)


FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/stage-prediction-hashes.json"


def row(episode=0):
    return {
        "episode_index": episode,
        "label": {"max_stage": 2, "success": False},
        "episode_duration_s": 4.0,
        "evidence": {"artifact": "s3://private/run/labels.jsonl"},
        "canonical_response": {
            "schema": "trajectory-label/v1",
            "event_times": [0.1, 2],
        },
    }


def metadata():
    return {
        "run_key": "campaign-v1",
        "dataset_repo": "test/episodes",
        "task": "routing_d2",
        "taxonomy_version": "test-v1",
        "taxonomy_hash": "a" * 64,
        "pipeline": {"name": "pipeline", "version": "v1", "git_commit": "abc"},
        "source": "test",
        "provenance": {"campaign_sha256": "b" * 64},
    }


class Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps({"ok": True, "value": self.value}).encode()


class PredictionHashTest(unittest.TestCase):
    def test_shared_cross_language_vectors(self):
        fixtures = json.loads(FIXTURES.read_text())
        self.assertEqual(fixtures["protocol"], "arena-prediction-content/v1")
        for vector in fixtures["values"]:
            with self.subTest(vector["name"]):
                self.assertEqual(canonical_bytes(vector["value"]).hex(), vector["canonical_hex"])
                self.assertEqual(canonical_digest(vector["value"]), vector["sha256"])
        for vector in fixtures["predictions"]:
            with self.subTest(vector["name"]):
                payload = {
                    **vector["row"],
                    "episode_index": int(vector["row"]["episode_index"]),
                }
                self.assertEqual(prediction_digest(payload), vector["sha256"])
        for vector in fixtures["manifests"]:
            with self.subTest(vector["name"]):
                rows = [{**r, "episode_index": int(r["episode_index"])} for r in vector["rows"]]
                self.assertEqual(manifest_digest(rows), vector["sha256"])

    def test_order_numbers_unicode_and_optional_presence(self):
        self.assertEqual(
            canonical_digest({"b": 1, "a": -0.0}), canonical_digest({"a": 0, "b": 1.0})
        )
        self.assertNotEqual(canonical_digest(True), canonical_digest(1))
        self.assertEqual(canonical_bytes({"\ue000": 1, "\U00010000": 2})[:9], b"o2:s3:\xee\x80\x80")
        self.assertNotEqual(
            prediction_digest(row()), prediction_digest({**row(), "vote_summary": None})
        )
        self.assertEqual(manifest_digest([row(12), row(2)]), manifest_digest([row(2), row(12)]))

    def test_rejects_non_json_nonfinite_lossy_and_surrogates(self):
        for value in [
            float("nan"),
            float("inf"),
            -float("inf"),
            2**53,
            -(2**53),
            b"bytes",
            {1: "x"},
            (1, 2),
        ]:
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                canonical_bytes(value)
        with self.assertRaises(UnicodeEncodeError):
            canonical_bytes("\ud800")
        nested = None
        for _ in range(65):
            nested = [nested]
        with self.assertRaisesRegex(ValueError, "nesting depth"):
            canonical_bytes(nested)

    def test_strict_row_contract_and_duplicates(self):
        for bad in [
            {**row(), "episode_index": True},
            {**row(), "episode_index": 1.5},
            {**row(), "episode_index": -1},
            {**row(), "episode_index": 2**63},
            {**row(), "episode_duration_s": -1},
            {**row(), "confidence": None},
            {**row(), "episode_duration_s": 0},
            {**row(), "source_revision": "a" * 41},
            {**row(), "label": []},
            {**row(), "evidnce": {}},
            {**row(), "violation_codes": [3]},
            {**row(), "evidence": "x" * (128 * 1024)},
        ]:
            with self.subTest(bad=repr(bad)[:100]), self.assertRaises(ValueError):
                prediction_payload(bad)
        missing = row()
        del missing["evidence"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            prediction_payload(missing)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            manifest_digest([row(), row()])
        self.assertEqual(
            prediction_digest(row(2**63 - 1)),
            prediction_digest({**row(), "episode_index": ConvexInt64(2**63 - 1)}),
        )


class PredictionVersionClientTest(unittest.TestCase):
    def setUp(self):
        self.client = PolicyArenaClient("https://test.convex.cloud", api_key="test.secret")
        self.client.client = Mock()

    @patch("policy_arena.client.urlopen")
    def test_http_append_encodes_only_episode_as_int64(self, request):
        request.return_value = Response({"inserted": 1, "unchanged": 0})
        result = self.client.append_stage_predictions("run-id", [row(2**63 - 1)])
        self.assertEqual(result["inserted"], 1)
        sent = request.call_args.args[0]
        self.assertEqual(
            sent.full_url,
            "https://test.convex.site/api/v1/mutate/stagePredictions/appendBatch",
        )
        self.assertEqual(sent.headers["Authorization"], "Bearer test.secret")
        decoded = json_to_convex(json.loads(sent.data))
        self.assertEqual(decoded["rows"][0]["episode_index"].value, 2**63 - 1)
        self.assertIsInstance(decoded["rows"][0]["label"]["max_stage"], float)

    @patch("policy_arena.client.urlopen")
    def test_conflict_from_server_is_reported_and_never_retried(self, request):
        request.side_effect = HTTPError(
            "https://test.invalid",
            409,
            "Conflict",
            None,
            io.BytesIO(b'{"ok":false,"error":"immutable prediction conflict","code":"conflict"}'),
        )
        with self.assertRaisesRegex(PolicyArenaAPIError, "immutable prediction conflict") as raised:
            self.client.append_stage_predictions("run-id", [row()])
        self.assertEqual(raised.exception.status, 409)
        self.assertEqual(request.call_count, 1)

    @patch.object(PolicyArenaClient, "_mutation")
    def test_explicit_activation_requires_expected_selection(self, mutation):
        self.client.activate_stage_prediction_run("run-id", expected_active_run_id=None)
        mutation.assert_called_once_with(
            "stagePredictions:activate",
            {"run_id": "run-id", "expected_active_run_id": None},
        )

    @patch("policy_arena.client.urlopen")
    def test_exact_retry_preserves_wire_payload(self, request):
        request.side_effect = [
            Response({"inserted": 1, "unchanged": 0}),
            Response({"inserted": 0, "unchanged": 1}),
        ]
        self.assertEqual(self.client.append_stage_predictions("run", [row()])["inserted"], 1)
        self.assertEqual(self.client.append_stage_predictions("run", [row()])["unchanged"], 1)
        self.assertEqual(
            request.call_args_list[0].args[0].data,
            request.call_args_list[1].args[0].data,
        )

    def test_pagination_reads_all_pages_and_detects_stuck_cursor(self):
        self.client.client.query.side_effect = [
            {"page": [row(0)], "isDone": False, "continueCursor": "next"},
            {"page": [row(1)], "isDone": True, "continueCursor": "done"},
        ]
        self.assertEqual(len(self.client.fetch_stage_predictions("run")), 2)
        self.assertEqual(
            self.client.client.query.call_args.args[1]["paginationOpts"]["cursor"],
            "next",
        )
        self.client.client.query.side_effect = None
        self.client.client.query.return_value = {
            "page": [],
            "isDone": False,
            "continueCursor": "stuck",
        }
        with self.assertRaisesRegex(RuntimeError, "did not advance"):
            self.client.fetch_stage_predictions("run")

    @patch.object(PolicyArenaClient, "_mutation")
    def test_review_prediction_binding_is_exact_and_optional(self, mutation):
        args = {
            "task": "routing_d2",
            "dataset_repo": "test/episodes",
            "episode_index": 0,
            "taxonomy_version": "v1",
            "status": "draft",
        }
        self.client.save_stage_review(**args, prediction_id="pred", prediction_sha256="a" * 64)
        payload = mutation.call_args.args[1]
        self.assertEqual(payload["prediction_id"], "pred")
        self.assertEqual(payload["prediction_sha256"], "a" * 64)
        self.client.save_stage_review(**args)
        self.assertNotIn("prediction_id", mutation.call_args.args[1])
        self.client.save_stage_review(
            **args, copied_from_review_id="source-review", prefill_pushed_at=123
        )
        self.assertEqual(mutation.call_args.args[1]["copied_from_review_id"], "source-review")
        self.assertEqual(mutation.call_args.args[1]["prefill_pushed_at"], 123.0)
        for extra in [
            {"prediction_id": "pred"},
            {"prediction_sha256": "a" * 64},
        ]:
            with self.assertRaises(ValueError):
                self.client.save_stage_review(**args, **extra)

    def _upload_mocks(self, rows):
        meta = {
            **metadata(),
            "content_protocol": CONTENT_PROTOCOL,
            "expected_count": len(rows),
            "manifest_sha256": manifest_digest(rows),
        }
        stored = [{**r, "run_id": "run-id", "content_sha256": prediction_digest(r)} for r in rows]
        self.client.begin_stage_prediction_run = Mock(return_value="run-id")
        self.client.append_stage_predictions = Mock(return_value={"inserted": 1, "unchanged": 0})
        self.client.fetch_stage_predictions = Mock(return_value=stored)
        self.client.get_stage_prediction_run = Mock(
            side_effect=[
                {**meta, "status": "uploading"},
                {**meta, "status": "published"},
            ]
        )
        self.client.publish_stage_prediction_run = Mock(return_value="run-id")
        self.client.activate_stage_prediction_run = Mock()
        return stored

    def test_verified_upload_batches_and_never_activates(self):
        rows = [row(i) for i in range(51)]
        self._upload_mocks(rows)
        order = Mock()
        order.attach_mock(self.client.fetch_stage_predictions, "read")
        order.attach_mock(self.client.publish_stage_prediction_run, "publish")
        self.assertEqual(self.client.upload_stage_prediction_run(rows, **metadata()), "run-id")
        self.assertEqual(
            [len(c.args[1]) for c in self.client.append_stage_predictions.call_args_list],
            [50, 1],
        )
        self.assertEqual([c[0] for c in order.mock_calls], ["read", "publish"])
        self.client.activate_stage_prediction_run.assert_not_called()

    def test_invalid_later_row_prevents_any_upload(self):
        rows = [row(i) for i in range(51)]
        self._upload_mocks(rows)
        rows[-1]["confidence"] = None
        with self.assertRaises(ValueError):
            self.client.upload_stage_prediction_run(rows, **metadata())
        self.client.begin_stage_prediction_run.assert_not_called()

    def test_large_rows_pack_under_batch_byte_limit(self):
        rows = [{**row(i), "evidence": "x" * (120 * 1024)} for i in range(12)]
        self._upload_mocks(rows)
        self.client.upload_stage_prediction_run(rows, **metadata())
        self.assertEqual(
            [len(c.args[1]) for c in self.client.append_stage_predictions.call_args_list],
            [8, 4],
        )

    def test_numeric_limits_fail_before_network(self):
        self.client._mutation = Mock()
        for count in (0, 10001, 2.0, True):
            with self.assertRaises(ValueError):
                self.client.begin_stage_prediction_run(
                    **metadata(), expected_count=count, manifest_sha256="a" * 64
                )
        for size in (0, 51, True, 1.5):
            with self.assertRaises(ValueError):
                self.client.upload_stage_prediction_run([row()], **metadata(), chunk_size=size)
        for rows in (
            [],
            [row(i) for i in range(51)],
            [{**row(i), "evidence": "x" * (120 * 1024)} for i in range(9)],
        ):
            with self.assertRaises(ValueError):
                self.client.append_stage_predictions("run", rows)
        self.client._mutation.assert_not_called()

    def test_corrupt_missing_duplicate_and_wrong_run_readback_never_publish(self):
        for damage in (
            "missing",
            "content",
            "digest",
            "duplicate",
            "wrong-run",
            "manifest",
        ):
            with self.subTest(damage):
                rows = [row(0), row(1)]
                stored = self._upload_mocks(rows)
                if damage == "missing":
                    stored.pop()
                elif damage == "content":
                    stored[0]["label"] = {"max_stage": 9}
                elif damage == "digest":
                    stored[0]["content_sha256"] = "0" * 64
                elif damage == "duplicate":
                    stored[1] = copy.deepcopy(stored[0])
                elif damage == "wrong-run":
                    stored[0]["run_id"] = "another-run"
                elif damage == "manifest":
                    stored[0]["label"] = {"max_stage": 9}
                    stored[0]["content_sha256"] = prediction_digest(stored[0], stored=True)
                with self.assertRaises((RuntimeError, ValueError)):
                    self.client.upload_stage_prediction_run(rows, **metadata())
                self.client.publish_stage_prediction_run.assert_not_called()

    def test_changed_run_metadata_prevents_publish(self):
        self._upload_mocks([row()])
        self.client.get_stage_prediction_run.side_effect = None
        self.client.get_stage_prediction_run.return_value = {
            **metadata(),
            "content_protocol": CONTENT_PROTOCOL,
            "source": "changed",
            "expected_count": 1,
            "manifest_sha256": manifest_digest([row()]),
        }
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            self.client.upload_stage_prediction_run([row()], **metadata())
        self.client.publish_stage_prediction_run.assert_not_called()

    def test_unknown_content_protocol_prevents_publish(self):
        self._upload_mocks([row()])
        self.client.get_stage_prediction_run.side_effect = None
        self.client.get_stage_prediction_run.return_value = {"content_protocol": "future/v2"}
        with self.assertRaisesRegex(RuntimeError, "unsupported content protocol"):
            self.client.upload_stage_prediction_run([row()], **metadata())
        self.client.publish_stage_prediction_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

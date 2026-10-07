import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from convex import ConvexInt64, json_to_convex

from policy_arena.client import (
    PolicyArenaAPIError,
    PolicyArenaClient,
    _normalize_convex_json_numbers,
)
from policy_arena.types import PolicyInput, RoundInput, RoundResultInput


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class PolicyArenaClientTest(unittest.TestCase):
    def test_derives_http_action_url(self):
        client = PolicyArenaClient("https://example.convex.cloud", api_key="test.key")
        self.assertEqual(
            client._api_url,
            "https://example.convex.site/api/v1",
        )

    def test_write_requires_machine_key(self):
        client = PolicyArenaClient("https://example.convex.cloud", api_key=None)
        client._api_key = None
        with self.assertRaisesRegex(PolicyArenaAPIError, "machine API key"):
            client.register_dataset(unittest.mock.Mock(to_dict=lambda: {}))

    @patch("policy_arena.client.urlopen")
    def test_submission_sends_auth_and_idempotency_headers(self, mock_urlopen):
        mock_urlopen.return_value = _Response({"ok": True, "value": "session-id"})
        client = PolicyArenaClient(
            "https://example.convex.cloud",
            api_key="pa_test.secret",
        )
        result = client.submit_eval_session(
            dataset_repo="org/dataset",
            policies=[PolicyInput("Policy", "org/model", "task")],
            rounds=[
                RoundInput(
                    0,
                    [RoundResultInput("org/model", True, 7)],
                )
            ],
            idempotency_key="submission-123",
        )

        self.assertEqual(result, "session-id")
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.headers["Authorization"], "Bearer pa_test.secret")
        self.assertEqual(request.headers["Idempotency-key"], "submission-123")
        body = json_to_convex(json.loads(request.data))
        self.assertEqual(body["rounds"][0]["round_index"].value, 0)
        self.assertEqual(
            body["rounds"][0]["results"][0]["episode_index"].value,
            7,
        )

    @patch("policy_arena.client.urlopen")
    def test_http_error_is_reported(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            url="https://example.invalid",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(b'{"ok":false,"code":"insufficient_scope","error":"lacks curate scope"}'),
        )
        client = PolicyArenaClient(
            "https://example.convex.cloud",
            api_key="pa_test.secret",
        )
        with self.assertRaisesRegex(PolicyArenaAPIError, "lacks curate scope") as raised:
            client.set_task_status("task", "testing")
        self.assertEqual(raised.exception.status, 403)
        self.assertEqual(raised.exception.code, "insufficient_scope")
        self.assertIsNone(raised.exception.error_id)

    @patch("policy_arena.client.urlopen")
    def test_internal_error_carries_safe_correlation_id(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            url="https://example.invalid",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(
                b'{"ok":false,"code":"mutation_failed",'
                b'"error":"Machine mutation failed",'
                b'"error_id":"550e8400-e29b-41d4-a716-446655440000"}'
            ),
        )
        client = PolicyArenaClient(
            "https://example.convex.cloud",
            api_key="pa_test.secret",
        )
        with self.assertRaisesRegex(PolicyArenaAPIError, "error_id=550e8400") as raised:
            client.set_task_status("task", "testing")
        self.assertEqual(raised.exception.status, 500)
        self.assertEqual(raised.exception.code, "mutation_failed")
        self.assertEqual(
            raised.exception.error_id,
            "550e8400-e29b-41d4-a716-446655440000",
        )

    def test_convex_int_round_trip_fixture(self):
        self.assertEqual(ConvexInt64(3).value, 3)

    def test_normalizes_integral_float64_without_touching_int64_wrappers(self):
        value = {
            "requested_at": 1788312198587,
            "count": {"$integer": "AwAAAAAAAAA="},
            "ok": True,
        }
        normalized = _normalize_convex_json_numbers(value)
        self.assertEqual(normalized["requested_at"], 1788312198587.0)
        self.assertEqual(normalized["count"], value["count"])
        self.assertIs(normalized["ok"], True)

    @patch.object(PolicyArenaClient, "_mutation")
    def test_cancel_apply_job_uses_scoped_machine_route(self, mutation):
        mutation.return_value = "job-id"
        client = PolicyArenaClient(
            "https://example.convex.cloud",
            api_key="pa_test.secret",
        )

        self.assertEqual(client.cancel_apply_job("job-id"), "job-id")
        mutation.assert_called_once_with("applyJobs:cancel", {"id": "job-id"})

    def test_round_result_input_carries_subtask_marks(self):
        d = RoundResultInput("org/model", False, 7, num_frames=10, num_subtask_marks=1).to_dict()
        self.assertEqual(d["num_subtask_marks"].value, 1)
        self.assertNotIn("num_subtask_marks", RoundResultInput("org/model", True, 7).to_dict())


if __name__ == "__main__":
    unittest.main()

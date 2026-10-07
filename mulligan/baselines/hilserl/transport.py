"""agentlace transport between the actor and the learner.

Learner: ``TrainerServer`` binding REP ``port`` (requests) and PUB ``port+1``
(param broadcast). Actor: ``TrainerClient`` connecting to both; only the actor
initiates TCP (works through an ``ssh -L`` tunnel with ``--ip localhost``).

Custom request types (payloads are pickled + lz4 by agentlace):

    hello         {session_id, hilserl_sha, repo_sha, config_hash}
                  -> {ingested_ids, counters, params, param_version}
    push-episode  {session_id, episode_id, record, arrays}
                  -> {ingested: bool, duplicate: bool, counters}
    status        {} -> {counters, param_version}

Broadcast payload: {params, param_version}. The learner refuses a hello whose
``hilserl_sha`` content hash or pinned-config hash differs from its own. The gate is a
content hash, not the commit sha: an unrelated commit in the repo must not lock a
running actor out of its learner (see ``session.hilserl_sha``).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

REQUEST_TYPES = ["hello", "push-episode", "status"]


def trainer_config(port: int):
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(
        port_number=int(port), broadcast_port=int(port) + 1, request_types=list(REQUEST_TYPES)
    )


class LearnerServer:
    def __init__(self, port: int, request_callback: Callable[[str, dict], dict]):
        from agentlace.data.data_store import QueuedDataStore
        from agentlace.trainer import TrainerServer

        self.port = int(port)
        self.server = TrainerServer(
            trainer_config(port), request_callback=request_callback, log_level=logging.WARNING
        )
        # agentlace's client calls update() on its data stores at connect time; give
        # it one tiny store so that path is a no-op instead of an error log.
        self.server.register_data_store("noop", QueuedDataStore(1))
        self.server.start(threaded=True)

    def publish(self, payload: dict) -> None:
        self.server.publish_network(payload)

    def stop(self) -> None:
        self.server.stop()


class ActorClient:
    def __init__(self, ip: str, port: int, timeout_ms: int = 5000, wait_for_server: bool = True):
        from agentlace.data.data_store import QueuedDataStore
        from agentlace.trainer import TrainerClient

        self.client = TrainerClient(
            "actor",
            ip,
            trainer_config(port),
            data_stores={"noop": QueuedDataStore(1)},
            log_level=logging.WARNING,
            wait_for_server=wait_for_server,
            timeout_ms=timeout_ms,
        )

    def request(self, type_: str, payload: dict) -> Optional[dict]:
        """None on timeout (learner unreachable); the caller decides what to do."""
        return self.client.request(type_, payload)

    def on_params(self, callback: Callable[[dict], None]) -> None:
        self.client.recv_network_callback(callback)

    def stop(self) -> None:
        self.client.stop()
